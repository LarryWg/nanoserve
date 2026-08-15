"""The paged attention path must compute what the reference path computes.

The reference is the SDPA, no-cache, one-sequence-at-a-time path in
model.py -- the one the HF equivalence tests already pinned token-for-token.
So this suite never asks "does the engine produce good text"; it asks "does
the engine produce the SAME numbers as the implementation we already trust",
step by step, with the KV cache and the block tables in the way.

Every test drives the real Scheduler and the real BlockManager, because the
bugs worth catching live in the seams between them: an off-by-one in
cache_seqlens, a block table read before the slot was reserved, a sequence
whose re-prefill after preemption lands in different blocks.

A tiny randomly-initialized model stands in for a checkpoint: 2 layers, 4
query heads over 2 KV heads, head_dim 32, vocab 256. It exercises every
branch of the paged path (GQA, block boundaries, batching) in milliseconds
and needs no download. test_paged_decode_hf.py runs the same machinery
against a real Qwen3 checkpoint and HF's own generate.
"""
import json

import pytest
import torch

pytest.importorskip("flash_attn", reason="the paged path is flash-attn only")

from nanoserve.model.model import NanoForCausalLM  # noqa: E402
from nanoserve.model_runner import ModelRunner  # noqa: E402
from nanoserve.scheduler import Scheduler  # noqa: E402
from nanoserve.sequence import SamplingParams, SeqStatus, Sequence  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="paged attention is CUDA only"),
]

BLOCK_SIZE = 256   # flash-attn's minimum page size (see kv_cache.py)

# Measured on this model (RTX 4090, fp16, flash-attn 2.8.3): max
# |paged - reference| is 9.8e-4 on both prefill and decode logits, at logit
# magnitudes around 1.5. That is flash-attn accumulating in a different
# order than SDPA, i.e. one fp16 ulp, not a disagreement about the math.
# The bound below leaves 5x headroom so a real bug -- which shifts logits by
# whole units, not ulps -- fails loudly. rtol is 0 on purpose: a relative
# tolerance on logits near zero would hide exactly the small, systematic
# drift a stale cache entry produces. The assertion that actually decides
# the test is argmax equality, checked alongside.
ATOL = 5e-3
RTOL = 0.0

TINY_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,     # GQA: 2 query heads per KV head
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
    """A tiny random checkpoint on disk, one per architecture branch.

    Both branches matter to the paged path: Qwen3 applies qk_norm before
    RoPE, Qwen2 carries a bias on q/k/v_proj. Going through a real
    safetensors dir (rather than handing the runner a live nn.Module) also
    keeps ModelRunner's only entry point the one production uses.
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
    # Token ids are arbitrary but deterministic, and greedy sampling keeps
    # the whole test deterministic from here on.
    torch.manual_seed(seq_id)
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (num_tokens,)).tolist()
    return Sequence(
        seq_id,
        ids,
        SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens),
    )


def reference_logits(model, token_ids: list[int]) -> torch.Tensor:
    """The trusted path: whole context, no cache, plain SDPA."""
    ids = torch.tensor(token_ids, device="cuda")
    with torch.inference_mode():
        out = model(ids, torch.arange(len(token_ids), device="cuda"))
    return out[-1].float()


def run_step(runner, scheduler):
    """One engine step, checked against the reference before it is taken.

    Whatever the batch is, each sequence's logits must match recomputing
    that sequence's whole context from scratch. For a prefill that is
    trivially the same computation; for a decode it means the cache holds
    exactly the KV the reference would have recomputed.
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
    """The cache must hold the prompt's KV, in the sequence's own blocks."""
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
    """The M2 gate at tiny scale: prefill, then decode past a block edge.

    Each prompt length puts the first generated token at a different offset
    in its block, so between them the three interesting cases (block fills
    exactly, block fills mid-decode, block was already full) are all hit.
    """
    runner = make_runner()
    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, prompt_len))
    # Four steps is enough: at prompt_len == BLOCK_SIZE the very first decode
    # opens a new block, at BLOCK_SIZE - 1 the second one does, and at
    # BLOCK_SIZE + 1 the tokens land in a block the prefill already opened.
    for _ in range(4):
        run_step(runner, scheduler)


def test_batched_steps_match_running_each_sequence_alone(make_runner):
    """Sequences of different lengths sharing a step must not leak into
    each other: that is what cu_seqlens (prefill) and per-row cache_seqlens
    with block tables (decode) exist to prevent."""
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
    """Continuous batching in miniature: a request admitted while another
    is mid-decode shares the cache without disturbing it."""
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
    """Recompute preemption is only free if the recompute is exact.

    Three blocks, two sequences that each fill one: the cache runs out on
    the first decode step, the youngest sequence is evicted, and its blocks
    go back to the free list. It waits until the other sequence finishes,
    then re-prefills its whole context -- prompt plus the token it had
    already generated -- into DIFFERENT physical blocks and carries on.

    Every step of both sequences is checked against the reference, so a
    re-prefill that landed in the wrong slots, or a resumed sequence that
    lost its generated token, fails on the very next logits.
    """
    runner = make_runner(num_blocks=3)
    scheduler = Scheduler(block_manager=runner.block_manager)
    victim = make_seq(2, BLOCK_SIZE, max_new_tokens=6)
    hog = make_seq(1, BLOCK_SIZE, max_new_tokens=3)
    scheduler.add_request(hog)      # admitted first, so the victim is younger
    scheduler.add_request(victim)

    run_step(runner, scheduler)                     # prefill both
    run_step(runner, scheduler)                     # decode: victim is evicted
    assert victim.status is SeqStatus.PREEMPTED
    assert victim.num_computed_tokens == 0          # KV gone
    assert victim.num_tokens == BLOCK_SIZE + 1      # token kept
    assert scheduler.running == [hog]

    while victim.status is not SeqStatus.FINISHED:
        run_step(runner, scheduler)                 # hog finishes, victim resumes
    assert len(victim.output_token_ids) == 6
    assert runner.block_manager.num_free_blocks() == 3   # nothing leaked


# ---------- cache sizing ----------

def test_vram_profiling_sizes_the_cache_to_what_is_free(make_runner, checkpoint):
    """num_blocks is measured, not guessed (see ModelRunner._profile_num_blocks).

    Half the card at most, so the assertion is about the arithmetic rather
    than about how much VRAM the test machine happens to have.
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
    # The cache the manager hands out ids for is the cache that exists.
    assert runner.kv_cache.num_blocks == runner.block_manager.num_blocks

    scheduler = Scheduler(block_manager=runner.block_manager)
    scheduler.add_request(make_seq(1, BLOCK_SIZE + 1))
    run_step(runner, scheduler)
