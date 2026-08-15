"""M2: the paged engine must generate what HuggingFace generates.

M1 pinned the model's forward pass against transformers with no KV cache at
all. This is the same gate one layer up: the same checkpoints, the same
greedy decode, but now every token after the first is produced from KV that
was written to the paged cache on an earlier step and read back through a
block table. A wrong slot, a stale cache_seqlens, or a block boundary
handled off by one all show up here as a token that differs from HF's.

The concurrency test is the one that could not exist before M2, and it is
deliberately NOT "batched output equals solo output token for token".
Measured on Qwen3-0.6B: running three prompts together moves the logits by
up to 0.27 against running them alone, because flash-attn splits the KV
reduction differently when the batch is bigger and bf16 does not forget the
difference. At logit magnitudes of 16-25, where one ulp is already 0.125,
that is noise -- but it is enough to flip a token whose top-2 margin is
0.125, or 0.000, both of which occur within 10 tokens of these prompts. No
engine that batches gets exact cross-batch reproducibility in bf16, vLLM
included.

So the property tested is the true one: at every step, every sequence in a
shared batch computes what recomputing that sequence alone computes, within
that noise floor, and picks the same token whenever the choice is not a
coin flip.

CUDA and flash-attn only, and the checkpoints are the same ~2.5 GB the M1
suite downloads. Run with `pytest -m slow`.
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

BLOCK_SIZE = 256          # the kernel's minimum page (see kv_cache.py)
NUM_BLOCKS = 8            # 2048 tokens of cache, sized by hand so the test
                          # does not depend on how much VRAM happens to be free
NUM_GREEDY_TOKENS = 50

# How far apart two correct bf16 computations of this model's logits are
# allowed to be. Measured over 300 steps on Qwen3-0.6B (RTX 4090, flash-attn
# 2.8.3), paged against the reference recompute: mean 0.25, max 0.66, at
# logit magnitudes of 16-25 where one bf16 ulp is already 0.125. The number
# is the same whether the sequence runs alone (max 0.656) or in a batch of
# three (max 0.625), which is the evidence that it is accumulation order and
# not something batching does wrong. 1.0 leaves ~1.5x headroom; a wrong
# block table moves logits by whole units.
BF16_LOGIT_NOISE = 1.0
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a village at the foot of a mountain, there lived",
    "1, 2, 3,",
]

_RUNNERS: dict[str, ModelRunner] = {}


def get_runner(repo_id: str) -> ModelRunner:
    """One runner per checkpoint, but a fresh BlockManager per test.

    Weights are expensive to load and are read-only, so the runner is
    shared. Block ownership is not: a test that fails mid-run would
    otherwise leave allocations behind and make the next test fail for a
    reason that has nothing to do with what it is checking.
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
    """HF's own answer. repetition_penalty is pinned for the same reason as
    in the M1 suite: instruct checkpoints ship a generation_config with 1.1
    baked in, and HF applies it even under do_sample=False."""
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
    """The engine loop, minus the threading and the HTTP server.

    This is exactly the body sketched in engine.py: schedule, forward,
    sample, advance, retire. When Engine lands it replaces this helper and
    these tests keep passing unchanged.
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
    """The trusted path: whole context, no cache, plain SDPA. This is the
    forward pass M1 pinned against transformers token-for-token."""
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
    """50 tokens, identical ids, no tolerance -- through the paged cache."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0].tolist()

    seq = make_seq(1, prompt_ids)
    run_to_completion(get_runner(repo_id), [seq])

    assert seq.output_token_ids == hf_greedy(repo_id, prompt_ids)


@pytest.mark.parametrize("repo_id", MODELS[:1])
def test_a_shared_batch_computes_what_each_sequence_computes_alone(repo_id):
    """Continuous batching must be invisible apart from bf16 noise.

    Three prompts of different lengths share every step, so they sit at
    different offsets in their blocks and finish out of order. At each step
    each one is checked against recomputing its own context from scratch on
    the reference path -- the same comparison the tiny-model suite makes,
    now with a real checkpoint, bf16, and neighbours in the batch. A block
    table pointing into someone else's KV would show up as a logit gap of
    whole units here, not hundredths.
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
                f"{gap:.3f} off the reference, far past bf16 noise"
            )
            top2 = want.topk(2).values
            if (top2[0] - top2[1]).item() > BF16_LOGIT_NOISE:
                assert token == int(want.argmax())   # not a coin flip: must agree
                decided += 1
            else:
                ties += 1
            if batch.is_prefill:
                seq.on_prefilled()
            seq.on_token(token, now=0.0)
            if seq.is_stopped:
                scheduler.finish(seq, now=0.0)

    # Guard against the test quietly excusing itself: near-ties are the rare
    # case, and the token assertion above has to actually be doing work.
    assert decided > 3 * ties, f"{ties} near-ties out of {decided + ties} steps"


@pytest.mark.parametrize("repo_id", MODELS[:1])
def test_solo_runs_are_reproducible(repo_id):
    """Same request, same engine, same tokens: batch composition is the only
    thing bf16 noise is allowed to depend on."""
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
    """The classic slow death of a serving engine is a block leak: nothing
    fails, throughput just decays until nothing can be admitted."""
    runner = get_runner(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    prompt_ids = tokenizer(PROMPTS[0], return_tensors="pt").input_ids[0].tolist()
    run_to_completion(runner, [make_seq(1, prompt_ids), make_seq(2, prompt_ids)])

    assert runner.block_manager.num_free_blocks() == NUM_BLOCKS
