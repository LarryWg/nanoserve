"""The paged engine must generate what HuggingFace generates.

The plain forward pass is already checked against transformers. This is the
same check one layer up: same checkpoints, same greedy decode, but now
every token after the first comes out of the paged cache, written on an
earlier step and read back through a block table. A wrong slot, a stale
cache_seqlens, or a block edge handled off by one all show up as a token
that differs from HF.

The batching test does NOT ask for identical tokens, on purpose. Running
three prompts together moves the logits by up to 0.27 against running them
alone, because the kernel splits its work differently when the batch grows
and bf16 remembers the difference. On logits of size 16 to 25, where one
bf16 step is already 0.125, that is noise. It is still enough to flip a
token whose top two choices sit 0.125 apart, or 0.000 apart, and both
happen within 10 tokens here. No engine that batches gets identical tokens
across batch sizes in bf16, vLLM included.

So the test asks what is true: every sequence in a shared batch computes
what it computes alone, within that noise, and picks the same token
whenever the choice is not a coin flip.

Needs CUDA. Run with `pytest -m slow`.
"""
import pytest
import torch

pytest.importorskip("flash_attn", reason="the paged path is flash-attn only")
transformers = pytest.importorskip("transformers")

from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from nanoserve.block_manager import BlockManager  # noqa: E402
from nanoserve.model_runner import ModelRunner  # noqa: E402
from nanoserve.scheduler import Scheduler  # noqa: E402
from nanoserve.sequence import SamplingParams, SeqStatus, Sequence  # noqa: E402

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="paged attention is CUDA only"),
]

MODELS = [
    pytest.param("Qwen/Qwen3-0.6B", id="qwen3-qknorm"),
    pytest.param("Qwen/Qwen2.5-0.5B-Instruct", id="qwen2-bias-tied"),
]

BLOCK_SIZE = 256          # the smallest page the kernel accepts
NUM_BLOCKS = 8            # 2048 tokens of cache, fixed by hand so the test
                          # does not depend on free memory
NUM_GREEDY_TOKENS = 50

# How far apart two correct bf16 runs of this model may land on a logit.
# Measured over 300 steps (RTX 4090), paged against the plain path: mean
# 0.25, max 0.66, on logits of size 16 to 25 where one bf16 step is already
# 0.125. The number is the same alone (0.656) as in a batch of three
# (0.625), which is what says it is addition order and not batching going
# wrong. 1.0 leaves half again as much room. A wrong block table would move
# logits by whole units.
BF16_LOGIT_NOISE = 1.0
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a village at the foot of a mountain, there lived",
    "1, 2, 3,",
]

_RUNNERS: dict[str, ModelRunner] = {}


def get_runner(repo_id: str) -> ModelRunner:
    """One runner per checkpoint, a fresh block manager per test.

    Weights are slow to load and never change, so the runner is shared. Who
    owns which block is not shared. A test that fails halfway would leave
    blocks held and break the next test for no good reason.
    """
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
    """HF's own answer. repetition_penalty is pinned because instruct
    checkpoints ship a config with 1.1 in it, which HF applies even when
    sampling is off."""
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
    """The engine loop without the threads or the server.

    Schedule, forward, sample, advance, retire. A real engine loop replaces
    this helper later and these tests keep passing.
    """
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
    """The trusted path: whole context, no cache. This is the forward pass
    already checked against transformers token by token."""
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
    """50 tokens, identical ids, no tolerance, through the paged cache."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0].tolist()

    seq = make_seq(1, prompt_ids)
    run_to_completion(get_runner(repo_id), [seq])

    assert seq.output_token_ids == hf_greedy(repo_id, prompt_ids)


@pytest.mark.parametrize("repo_id", MODELS[:1])
def test_a_shared_batch_computes_what_each_sequence_computes_alone(repo_id):
    """Batching must be invisible apart from bf16 noise.

    Three prompts of different lengths share every step, so they sit at
    different offsets in their blocks and finish out of order. Each one is
    checked at each step against redoing its own context from scratch. A
    block table pointing into someone else's keys would show up here as a
    gap of whole units, not hundredths.
    """
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
                assert token == int(want.argmax())   # not a coin flip
                decided += 1
            else:
                ties += 1
            if batch.is_prefill:
                seq.on_prefilled()
            seq.on_token(token, now=0.0)
            if seq.is_stopped:
                scheduler.finish(seq, now=0.0)

    # Stop the test from excusing itself. Close calls are meant to be rare,
    # so the token check above has to be doing real work.
    assert decided > 3 * ties, f"{ties} near-ties out of {decided + ties} steps"


@pytest.mark.parametrize("repo_id", MODELS[:1])
def test_solo_runs_are_reproducible(repo_id):
    """Same request, same engine, same tokens. What is in the batch is the
    only thing bf16 noise may depend on."""
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
    """The slow death of a serving engine is a block leak. Nothing fails,
    throughput just fades until no request can get in."""
    runner = get_runner(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0].tolist()
    run_to_completion(runner, [make_seq(1, prompt_ids), make_seq(2, prompt_ids)])

    assert runner.block_manager.num_free_blocks() == NUM_BLOCKS
