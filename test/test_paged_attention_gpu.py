import json

import pytest
import torch

if torch.cuda.is_available():
    import flash_attn  # noqa: F401
else:
    pytest.skip("the paged path is CUDA only", allow_module_level=True)

from nanoserve.model.model import NanoForCausalLM  # noqa: E402
from nanoserve.model_runner import ModelRunner  # noqa: E402
from nanoserve.scheduler import Scheduler  # noqa: E402
from nanoserve.sequence import SamplingParams, SeqStatus, Sequence  # noqa: E402

pytestmark = pytest.mark.gpu

BLOCK_SIZE = 256

ATOL = 5e-3
RTOL = 0.0

TINY_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
}


@pytest.fixture(scope="session", params=["Qwen3ForCausalLM", "Qwen2ForCausalLM"])
def checkpoint(request, tmp_path_factory):
    from safetensors.torch import save_file

    path = tmp_path_factory.mktemp(request.param)
    (path / "config.json").write_text(
        json.dumps({"architectures": [request.param], **TINY_CONFIG})
    )
    torch.manual_seed(0)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        from nanoserve.model.config import ModelConfig
        model = NanoForCausalLM(ModelConfig.from_pretrained(path))
    finally:
        torch.set_default_dtype(prev)
    save_file(model.state_dict(), str(path / "model.safetensors"))
    return path


@pytest.fixture
def make_runner(checkpoint):
    def _make(num_blocks=8):
        return ModelRunner(
            str(checkpoint),
            block_size=BLOCK_SIZE,
            device="cuda",
            num_blocks=num_blocks,
            max_num_batched_tokens=2048,
        )
    return _make


def make_seq(seq_id: int, num_tokens: int, max_new_tokens: int = 256) -> Sequence:
    torch.manual_seed(seq_id)
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (num_tokens,)).tolist()
    return Sequence(
        seq_id,
        ids,
        SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens),
    )


def reference_logits(model, token_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(token_ids, device="cuda")
    with torch.inference_mode():
        out = model(ids, torch.arange(len(token_ids), device="cuda"))
    return out[-1].float()


def run_step(runner, scheduler):
    batch = scheduler.step()
    assert batch is not None, "scheduler ran out of work mid-test"
    expected = [reference_logits(runner.model, seq.token_ids) for seq in batch.seqs]

    logits = runner.forward(batch)
    assert logits.shape == (len(batch.seqs), TINY_CONFIG["vocab_size"])
    tokens = runner.sample(logits, batch)

    for seq, row, want, token in zip(batch.seqs, logits, expected, tokens):
        assert torch.allclose(row.float(), want, atol=ATOL, rtol=RTOL), (
            f"seq {seq.seq_id} at {seq.num_tokens} tokens: max diff "
            f"{(row.float() - want).abs().max().item():.4f}"
        )
        assert token == int(want.argmax()), \
            f"seq {seq.seq_id} sampled a different token"
        if batch.is_prefill:
            seq.on_prefilled()
        seq.on_token(token, now=0.0)
        if seq.is_stopped:
            scheduler.finish(seq, now=0.0)
    return batch


@pytest.mark.parametrize(
    "prompt_len",
    [1, BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, 3 * BLOCK_SIZE + 5],
    ids=lambda n: f"{n}tok",
)
def test_prefill_matches_reference_at_every_block_boundary(make_runner, prompt_len):
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, prompt_len))
    run_step(runner, scheduler)


def test_prefill_writes_kv_for_every_prompt_token(make_runner):
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    seq = make_seq(1, BLOCK_SIZE + 3)
    scheduler.add_request(seq)
    run_step(runner, scheduler)

    table = runner.block_manager.get_block_table(seq.seq_id)
    assert len(table) == 2
    k = runner.kv_cache.k
    assert k[:, table[0]].any()
    assert k[:, table[1], :3].any()
    assert not k[:, table[1], 3:].any()


@pytest.mark.parametrize(
    "prompt_len", [BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1], ids=lambda n: f"{n}tok"
)
def test_decode_tracks_the_reference_across_block_boundaries(make_runner, prompt_len):
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, prompt_len))
    for _ in range(4):
        run_step(runner, scheduler)


def test_batched_steps_match_running_each_sequence_alone(make_runner):
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    for seq_id, length in enumerate([BLOCK_SIZE + 1, 3, 2 * BLOCK_SIZE], start=1):
        scheduler.add_request(make_seq(seq_id, length))

    batch = run_step(runner, scheduler)
    assert batch.is_prefill and len(batch.seqs) == 3
    for _ in range(4):
        batch = run_step(runner, scheduler)
        assert not batch.is_prefill and len(batch.seqs) == 3


def test_a_late_arrival_joins_running_sequences(make_runner):
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, BLOCK_SIZE + 2))
    for _ in range(3):
        run_step(runner, scheduler)

    scheduler.add_request(make_seq(2, 5))
    assert run_step(runner, scheduler).is_prefill
    for _ in range(3):
        assert len(run_step(runner, scheduler).seqs) == 2


def test_a_preempted_sequence_resumes_where_it_left_off(make_runner):
    runner = make_runner(num_blocks=3)
    scheduler = Scheduler(block_manager=runner.block_manager)
    victim = make_seq(2, BLOCK_SIZE, max_new_tokens=6)
    hog = make_seq(1, BLOCK_SIZE, max_new_tokens=3)
    scheduler.add_request(hog)
    scheduler.add_request(victim)

    run_step(runner, scheduler)
    run_step(runner, scheduler)
    assert victim.status is SeqStatus.PREEMPTED
    assert victim.num_computed_tokens == 0
    assert victim.num_tokens == BLOCK_SIZE + 1
    assert scheduler.running == [hog]

    while victim.status is not SeqStatus.FINISHED:
        run_step(runner, scheduler)
    assert len(victim.output_token_ids) == 6
    assert runner.block_manager.num_free_blocks() == 3


def test_vram_profiling_sizes_the_cache_to_what_is_free(make_runner, checkpoint):
    runner = ModelRunner(
        str(checkpoint),
        block_size=BLOCK_SIZE,
        device="cuda",
        gpu_memory_utilization=0.5,
        max_num_batched_tokens=1024,
    )
    total = torch.cuda.get_device_properties(0).total_memory
    assert runner.block_manager.num_blocks > 0
    assert runner.kv_cache.nbytes() < 0.5 * total
    assert runner.kv_cache.num_blocks == runner.block_manager.num_blocks

    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, BLOCK_SIZE + 1))
    run_step(runner, scheduler)
