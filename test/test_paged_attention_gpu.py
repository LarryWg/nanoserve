"""The paged path must compute what the plain path computes.

The plain path runs one sequence with no cache, and it is already checked
against transformers token by token. So these tests never ask whether the
output reads well. They ask whether it holds the same numbers, step by
step, with the cache and the block tables in the way.

Every test drives the real scheduler and block manager, because the bugs
worth catching live between them: an off by one in cache_seqlens, a block
table read before its slot was reserved, an evicted sequence that comes
back in different blocks.

A tiny random model stands in for a checkpoint: 2 layers, 4 query heads
over 2 key heads, head_dim 32, vocab 256. It covers GQA, block edges and
batching in milliseconds, and needs no download.
"""
import json

import pytest
import torch

if torch.cuda.is_available():
    # A GPU box without the kernels is a broken environment, not a reason to
    # skip. Skipping there would let a wrong install pass as a green run.
    import flash_attn  # noqa: F401
else:
    pytest.skip("the paged path is CUDA only", allow_module_level=True)

from nanoserve.model.model import NanoForCausalLM  # noqa: E402
from nanoserve.model_runner import ModelRunner  # noqa: E402
from nanoserve.scheduler import Scheduler  # noqa: E402
from nanoserve.sequence import SamplingParams, SeqStatus, Sequence  # noqa: E402

pytestmark = pytest.mark.gpu

BLOCK_SIZE = 256   # the smallest page the kernel accepts

# Measured on this model (RTX 4090, fp16): the two paths differ by at most
# 9.8e-4 on logits of size 1.5, which is one fp16 step. That is the kernel
# adding numbers in a different order, not a different answer. The bound
# leaves 5x room, since a real bug moves logits by whole units. rtol stays
# 0, so small steady drift near zero cannot hide.
ATOL = 5e-3
RTOL = 0.0

TINY_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,     # GQA, 2 query heads per key head
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
    """A tiny random checkpoint on disk, one per architecture.

    Both matter here. Qwen3 norms q and k before RoPE, Qwen2 carries a bias
    on the projections. Writing a real checkpoint also means the runner is
    loaded the same way a server would load it.
    """
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
    # Any token ids will do, as long as they are the same every run.
    torch.manual_seed(seq_id)
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (num_tokens,)).tolist()
    return Sequence(
        seq_id,
        ids,
        SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens),
    )


def reference_logits(model, token_ids: list[int]) -> torch.Tensor:
    """The trusted path: whole context, no cache."""
    ids = torch.tensor(token_ids, device="cuda")
    with torch.inference_mode():
        out = model(ids, torch.arange(len(token_ids), device="cuda"))
    return out[-1].float()


def run_step(runner, scheduler):
    """One engine step, checked before it is taken.

    Whatever the batch holds, each sequence's logits must match redoing its
    whole context from scratch. For a decode that means the cache holds
    exactly the keys and values the plain path would have recomputed.
    """
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


# ---------- prefill ----------

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
    """The prompt's keys and values land in this sequence's own blocks."""
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    seq = make_seq(1, BLOCK_SIZE + 3)
    scheduler.add_request(seq)
    run_step(runner, scheduler)

    table = runner.block_manager.get_block_table(seq.seq_id)
    assert len(table) == 2
    k = runner.kv_cache.k
    assert k[:, table[0]].any()          # first block: full
    assert k[:, table[1], :3].any()      # second block: 3 tokens spilled in
    assert not k[:, table[1], 3:].any()  # and nothing beyond them


# ---------- decode ----------

@pytest.mark.parametrize(
    "prompt_len", [BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1], ids=lambda n: f"{n}tok"
)
def test_decode_tracks_the_reference_across_block_boundaries(make_runner, prompt_len):
    """Prefill, then decode past the edge of a block.

    Each prompt length drops the first new token at a different offset, so
    together they cover a block that fills exactly, one that fills while
    decoding, and one that was already full.
    """
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, prompt_len))
    # Four steps is enough. One prompt opens a new block on the first
    # decode, one on the second, one writes into a block prefill opened.
    for _ in range(4):
        run_step(runner, scheduler)


def test_batched_steps_match_running_each_sequence_alone(make_runner):
    """Sequences of different lengths must not leak into each other.

    Stopping that is the whole job of cu_seqlens during prefill, and of the
    block tables and cache_seqlens during decode.
    """
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
    """Continuous batching in miniature. A request that arrives while
    another is decoding shares the cache without disturbing it."""
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, BLOCK_SIZE + 2))
    for _ in range(3):
        run_step(runner, scheduler)

    scheduler.add_request(make_seq(2, 5))
    assert run_step(runner, scheduler).is_prefill      # prefill_first
    for _ in range(3):
        assert len(run_step(runner, scheduler).seqs) == 2


# ---------- preemption ----------

def test_a_preempted_sequence_resumes_where_it_left_off(make_runner):
    """Throwing work away is only free if redoing it is exact.

    Three blocks, two sequences that each fill one. The cache runs out on
    the first decode, so the youngest sequence is evicted and its blocks go
    back. It waits for the other to finish, then prefills its whole context
    again, prompt plus the token it had already made, into different blocks.

    Both sequences are checked at every step, so blocks filled in the wrong
    order, or a resumed sequence that lost a token, fail on the next logits.
    """
    runner = make_runner(num_blocks=3)
    scheduler = Scheduler(block_manager=runner.block_manager)
    victim = make_seq(2, BLOCK_SIZE, max_new_tokens=6)
    hog = make_seq(1, BLOCK_SIZE, max_new_tokens=3)
    scheduler.add_request(hog)      # admitted first, so the victim is younger
    scheduler.add_request(victim)

    run_step(runner, scheduler)                     # prefill both
    run_step(runner, scheduler)                     # decode, victim evicted
    assert victim.status is SeqStatus.PREEMPTED
    assert victim.num_computed_tokens == 0          # cache gone
    assert victim.num_tokens == BLOCK_SIZE + 1      # tokens kept
    assert scheduler.running == [hog]

    while victim.status is not SeqStatus.FINISHED:
        run_step(runner, scheduler)                 # hog ends, victim resumes
    assert len(victim.output_token_ids) == 6
    assert runner.block_manager.num_free_blocks() == 3   # nothing leaked


# ---------- cache sizing ----------

def test_vram_profiling_sizes_the_cache_to_what_is_free(make_runner, checkpoint):
    """num_blocks is measured, not guessed.

    Half the card at most, so the check is about the maths and not about
    how much memory this machine happens to have.
    """
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
    # The manager hands out ids for the cache that actually exists.
    assert runner.kv_cache.num_blocks == runner.block_manager.num_blocks

    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, BLOCK_SIZE + 1))
    run_step(runner, scheduler)
