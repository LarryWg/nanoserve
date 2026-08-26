import pytest
import torch

if torch.cuda.is_available():
    import flash_attn  # noqa: F401
else:
    pytest.skip("the paged path is CUDA only", allow_module_level=True)

from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from nanoserve.block_manager import BlockManager  # noqa: E402
from nanoserve.model_runner import ModelRunner  # noqa: E402
from nanoserve.scheduler import Scheduler  # noqa: E402
from nanoserve.sequence import SamplingParams, SeqStatus, Sequence  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

MODELS = [
    pytest.param("Qwen/Qwen3-0.6B", id="qwen3-qknorm"),
    pytest.param("Qwen/Qwen2.5-0.5B-Instruct", id="qwen2-bias-tied"),
]

BLOCK_SIZE = 256
NUM_BLOCKS = 8
NUM_GREEDY_TOKENS = 50

BF16_LOGIT_NOISE = 1.0
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a village at the foot of a mountain, there lived",
    "1, 2, 3,",
]

_RUNNERS: dict[str, ModelRunner] = {}


def get_runner(repo_id: str) -> ModelRunner:
    if repo_id not in _RUNNERS:
        _RUNNERS[repo_id] = ModelRunner(
            snapshot_download(repo_id),
            block_size=BLOCK_SIZE,
            device="cuda",
            num_blocks=NUM_BLOCKS,
        )
    runner = _RUNNERS[repo_id]
    runner.block_manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    return runner


def hf_greedy(repo_id: str, prompt_ids: list[int]) -> list[int]:
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, dtype=torch.bfloat16).to("cuda")
    model.eval()
    with torch.inference_mode():
        out = model.generate(
            torch.tensor([prompt_ids], device="cuda"),
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=NUM_GREEDY_TOKENS,
            min_new_tokens=NUM_GREEDY_TOKENS,
        )
    del model
    torch.cuda.empty_cache()
    return out[0, len(prompt_ids):].tolist()


def run_to_completion(runner: ModelRunner, seqs: list[Sequence]) -> None:
    scheduler = Scheduler(block_manager=runner.block_manager)
    for seq in seqs:
        scheduler.add_request(seq)
    while any(seq.status is not SeqStatus.FINISHED for seq in seqs):
        batch = scheduler.step()
        assert batch is not None, "scheduler idled with unfinished sequences"
        tokens = runner.sample(runner.forward(batch), batch)
        for seq, token in zip(batch.seqs, tokens):
            if batch.is_prefill:
                seq.on_prefilled()
            seq.on_token(token, now=0.0)
            if seq.is_stopped:
                scheduler.finish(seq, now=0.0)


def reference_logits(model, token_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor(token_ids, device="cuda")
    with torch.inference_mode():
        out = model(ids, torch.arange(len(token_ids), device="cuda"))
    return out[-1].float()


def make_seq(seq_id: int, prompt_ids: list[int]) -> Sequence:
    return Sequence(
        seq_id,
        prompt_ids,
        SamplingParams(temperature=0.0, max_new_tokens=NUM_GREEDY_TOKENS),
    )


@pytest.mark.parametrize("repo_id", MODELS)
def test_paged_greedy_decode_matches_hf(repo_id):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0].tolist()

    seq = make_seq(1, prompt_ids)
    run_to_completion(get_runner(repo_id), [seq])

    assert seq.output_token_ids == hf_greedy(repo_id, prompt_ids)


@pytest.mark.parametrize("repo_id", MODELS[:1])
def test_a_shared_batch_computes_what_each_sequence_computes_alone(repo_id):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompts = [tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in PROMPTS]
    runner = get_runner(repo_id)

    seqs = [make_seq(i, ids) for i, ids in enumerate(prompts)]
    scheduler = Scheduler(block_manager=runner.block_manager)
    for seq in seqs:
        scheduler.add_request(seq)

    decided = ties = 0
    while any(seq.status is not SeqStatus.FINISHED for seq in seqs):
        batch = scheduler.step()
        expected = [reference_logits(runner.model, s.token_ids) for s in batch.seqs]
        logits = runner.forward(batch)
        tokens = runner.sample(logits, batch)
        for seq, row, want, token in zip(batch.seqs, logits, expected, tokens):
            gap = (row.float() - want).abs().max().item()
            assert gap < BF16_LOGIT_NOISE, (
                f"seq {seq.seq_id} at {seq.num_tokens} tokens: logits are "
                f"{gap:.3f} off the plain path, far past bf16 noise"
            )
            top2 = want.topk(2).values
            if (top2[0] - top2[1]).item() > BF16_LOGIT_NOISE:
                assert token == int(want.argmax())
                decided += 1
            else:
                ties += 1
            if batch.is_prefill:
                seq.on_prefilled()
            seq.on_token(token, now=0.0)
            if seq.is_stopped:
                scheduler.finish(seq, now=0.0)

    assert decided > 3 * ties, f"{ties} near-ties out of {decided + ties} steps"


@pytest.mark.parametrize("repo_id", MODELS[:1])
def test_solo_runs_are_reproducible(repo_id):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(PROMPTS[1], return_tensors="pt").input_ids[0].tolist()
    runner = get_runner(repo_id)

    runs = []
    for seq_id in (1, 2):
        seq = make_seq(seq_id, prompt_ids)
        run_to_completion(runner, [seq])
        runs.append(seq.output_token_ids)
    assert runs[0] == runs[1]


@pytest.mark.parametrize("repo_id", MODELS[:1])
def test_the_cache_is_returned_after_every_sequence(repo_id):
    runner = get_runner(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0].tolist()
    run_to_completion(runner, [make_seq(1, prompt_ids), make_seq(2, prompt_ids)])

    assert runner.block_manager.num_free_blocks() == NUM_BLOCKS
