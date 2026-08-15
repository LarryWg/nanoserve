"""The full model must match transformers token-for-token.

Everything downstream (KV cache, scheduler, server) is built on the
assumption that this model computes exactly what HF computes, so this
comparison is the gate for the rest of the engine.

Two checkpoints cover every architectural branch in ModelConfig:
  - Qwen3-0.6B:            qk_norm=True,  attention_bias=False, untied lm_head
  - Qwen2.5-0.5B-Instruct: qk_norm=False, attention_bias=True,  tied lm_head

The gate, per checkpoint:
  1. logits match in fp32 (atol=1e-3 proves the math is right); in bf16 we
     pin argmax identity instead of a raw-logit atol. See
     test_logits_match_hf for why bf16 logits cannot be compared tightly.
  2. 50-token greedy decode is identical to HF generate(do_sample=False).
     repetition_penalty=1.0 is passed explicitly because instruct
     checkpoints ship a generation_config.json with 1.1 baked in, and HF
     applies it even under do_sample=False.

Runs on CPU, and on CUDA too when a GPU is visible: both models move to the
device under test, so the comparison is always same-backend. First run
downloads ~2.5 GB of checkpoints, cached afterwards.
Verified with torch 2.13.0, transformers 5.14.1.
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

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

PROMPT = "The capital of France is"
NUM_GREEDY_TOKENS = 50

# Checkpoints load once per (model, device), not once per test.
_CACHE: dict[tuple[str, str], tuple] = {}


def _load_pair(repo_id: str, device: str):
    key = (repo_id, device)
    if key not in _CACHE:
        # snapshot_download gives us a local dir for the strict safetensors
        # loader; HF loads its own copy through from_pretrained.
        path = snapshot_download(repo_id)
        hf = AutoModelForCausalLM.from_pretrained(
            repo_id, dtype=torch.bfloat16
        ).to(device)
        hf.eval()
        ours = NanoForCausalLM.from_pretrained(path, device=device)
        ours.eval()
        tok = AutoTokenizer.from_pretrained(repo_id)
        _CACHE[key] = (hf, ours, tok)
    return _CACHE[key]


@pytest.mark.slow
@pytest.mark.parametrize("repo_id", MODELS)
@pytest.mark.parametrize("device", DEVICES)
def test_logits_match_hf(repo_id, device):
    hf, ours, tok = _load_pair(repo_id, device)
    ids = tok(PROMPT, return_tensors="pt").input_ids[0].to(device)

    # (a) fp32 upcast: isolates math correctness from bf16 noise. Both
    # models hold the same bf16 weights; upcasting is exact, so any real
    # architectural misunderstanding (wrong norm order, wrong RoPE pairing,
    # transposed projection) shows up here orders of magnitude above 1e-3.
    with torch.inference_mode():
        hf32 = hf.float()(ids.unsqueeze(0)).logits[0]
        our32 = ours.float()(ids, torch.arange(len(ids), device=device))
    assert our32.shape == hf32.shape
    assert torch.allclose(our32, hf32, atol=1e-3, rtol=1e-3)

    # (b) bf16 as-served: logits are allowed bf16 rounding noise (measured:
    # max abs diff 0.28 on Qwen3-0.6B, i.e. 1-2 ulp at logit magnitudes
    # ~20-30). An atol of 1e-2 on raw bf16 logits is unachievable for ANY
    # two correct implementations with different op order. What the decode
    # gate actually relies on is argmax agreement, so that is what we pin:
    # the next-token choice must be identical at EVERY prompt position.
    with torch.inference_mode():
        hf_logits = hf(ids.unsqueeze(0)).logits[0].float()
        our_logits = ours(ids, torch.arange(len(ids), device=device)).float()
    assert torch.equal(our_logits.argmax(-1), hf_logits.argmax(-1))
    # Sanity bound: stays within a few bf16 ulp, no systematic drift.
    assert (our_logits - hf_logits).abs().max().item() < 0.5


@pytest.mark.slow
@pytest.mark.parametrize("repo_id", MODELS)
@pytest.mark.parametrize("device", DEVICES)
def test_greedy_decode_matches_hf(repo_id, device):
    """The gate itself: 50 greedy tokens, identical ids, no tolerance."""
    hf, ours, tok = _load_pair(repo_id, device)
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()

    hf_out = hf.generate(
        torch.tensor([prompt_ids], device=device),
        do_sample=False,
        repetition_penalty=1.0,             # see docstring: gen_config ships 1.1
        max_new_tokens=NUM_GREEDY_TOKENS,
        min_new_tokens=NUM_GREEDY_TOKENS,   # suppress early EOS stop
    )[0, len(prompt_ids):].tolist()

    # No KV cache yet: recompute the full context each step. O(T^2) and
    # proud of it: correctness first, the cache lands later.
    our_ids = list(prompt_ids)
    with torch.inference_mode():
        for _ in range(NUM_GREEDY_TOKENS):
            ids = torch.tensor(our_ids, device=device)
            logits = ours(ids, torch.arange(len(our_ids), device=device))
            our_ids.append(int(logits[-1].argmax()))

    assert our_ids[len(prompt_ids):] == hf_out
