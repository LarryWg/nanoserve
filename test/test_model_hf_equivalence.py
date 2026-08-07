"""M1 gate, part 2: the full model must match transformers token-for-token.

Nothing downstream (KV cache, scheduler, server) proceeds until this passes
-- DESIGN.md §9: "M1 is the gate for everything."

Two checkpoints, chosen so every architectural branch in ModelConfig is
exercised against a real HF model:
  - Qwen3-0.6B:            qk_norm=True,  attention_bias=False, untied lm_head
  - Qwen2.5-0.5B-Instruct: qk_norm=False, attention_bias=True,  tied lm_head

The gate (DESIGN.md §5), per checkpoint:
  1. logits match HF tightly when both run in fp32 (atol=1e-3 -- proves the
     math is right), and in bf16-as-served the next-token argmax agrees at
     every prompt position. DESIGN's "atol=1e-2 bf16" turns out to be
     unachievable between ANY two correct implementations: raw bf16 logits
     at magnitude ~20-30 have a representable step of ~0.1-0.25, and op
     ordering differences accumulate to ~1-2 ulp over 28 layers (measured
     max diff 0.28 on Qwen3-0.6B). The honest bf16 invariant is argmax
     identity, which is what the decode gate below relies on.
  2. 50-token greedy decode is identical to HF generate(do_sample=False).
     min_new_tokens=50 suppresses EOS stopping on the HF side so both
     loops always run the full 50 steps. repetition_penalty=1.0 is passed
     explicitly because instruct checkpoints ship a generation_config.json
     with repetition_penalty 1.1 baked in, and HF applies it EVEN under
     do_sample=False -- without this, HF isn't doing pure greedy and the
     comparison fails at the first repeated token (this actually happened:
     Qwen2.5 diverged at token 3, a 1.6-logit gap, i.e. not bf16 noise).

Runs on CPU (no GPU on this machine yet); bf16 CPU matmul/SDPA are
supported in torch 2.13 on Apple Silicon. First run downloads ~2.5 GB
of checkpoints. Verified with torch 2.13.0, transformers 5.14.1.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from nanoserve.model.model import NanoForCausalLM  # noqa: E402

MODELS = [
    pytest.param("Qwen/Qwen3-0.6B", id="qwen3-qknorm"),
    pytest.param("Qwen/Qwen2.5-0.5B-Instruct", id="qwen2-bias-tied"),
]

PROMPT = "The capital of France is"
NUM_GREEDY_TOKENS = 50

# Checkpoints load once per session, not once per test.
_CACHE: dict[str, tuple] = {}


def _load_pair(repo_id: str):
    if repo_id not in _CACHE:
        # snapshot_download gives us a local dir for the strict safetensors
        # loader; HF loads its own copy through from_pretrained.
        path = snapshot_download(repo_id)
        hf = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.bfloat16)
        hf.eval()
        ours = NanoForCausalLM.from_pretrained(path)
        ours.eval()
        tok = AutoTokenizer.from_pretrained(repo_id)
        _CACHE[repo_id] = (hf, ours, tok)
    return _CACHE[repo_id]


@pytest.mark.slow
@pytest.mark.parametrize("repo_id", MODELS)
def test_logits_match_hf(repo_id):
    hf, ours, tok = _load_pair(repo_id)
    ids = tok(PROMPT, return_tensors="pt").input_ids[0]

    # (a) fp32 upcast: isolates math correctness from bf16 noise. Both
    # models hold the same bf16 weights; upcasting is exact, so any real
    # architectural misunderstanding (wrong norm order, wrong RoPE pairing,
    # transposed projection) shows up here orders of magnitude above 1e-3.
    with torch.inference_mode():
        hf32 = hf.float()(ids.unsqueeze(0)).logits[0]
        our32 = ours.float()(ids, torch.arange(len(ids)))
    assert our32.shape == hf32.shape
    assert torch.allclose(our32, hf32, atol=1e-3, rtol=1e-3)

    # (b) bf16 as-served: logits are allowed bf16 rounding noise (measured:
    # max abs diff 0.28 on Qwen3-0.6B, i.e. 1-2 ulp at logit magnitudes
    # ~20-30 -- atol=1e-2 in raw bf16 is unachievable for ANY two correct
    # implementations with different op order). What the decode gate
    # actually relies on is argmax agreement, so that is what we pin:
    # the next-token choice must be identical at EVERY prompt position.
    with torch.inference_mode():
        hf_logits = hf(ids.unsqueeze(0)).logits[0].float()
        our_logits = ours(ids, torch.arange(len(ids))).float()
    assert torch.equal(our_logits.argmax(-1), hf_logits.argmax(-1))
    # Sanity bound: stays within a few bf16 ulp, no systematic drift.
    assert (our_logits - hf_logits).abs().max().item() < 0.5


@pytest.mark.slow
@pytest.mark.parametrize("repo_id", MODELS)
def test_greedy_decode_matches_hf(repo_id):
    """The gate itself: 50 greedy tokens, identical ids, no tolerance."""
    hf, ours, tok = _load_pair(repo_id)
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()

    hf_out = hf.generate(
        torch.tensor([prompt_ids]),
        do_sample=False,
        repetition_penalty=1.0,             # see docstring: gen_config ships 1.1
        max_new_tokens=NUM_GREEDY_TOKENS,
        min_new_tokens=NUM_GREEDY_TOKENS,   # suppress early EOS stop
    )[0, len(prompt_ids):].tolist()

    # No KV cache yet: recompute the full context each step. O(T^2) and
    # proud of it -- correctness first, the cache lands in M2.
    our_ids = list(prompt_ids)
    with torch.inference_mode():
        for _ in range(NUM_GREEDY_TOKENS):
            ids = torch.tensor(our_ids)
            logits = ours(ids, torch.arange(len(our_ids)))
            our_ids.append(int(logits[-1].argmax()))

    assert our_ids[len(prompt_ids):] == hf_out
