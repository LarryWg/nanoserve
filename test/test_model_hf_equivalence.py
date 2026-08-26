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

_CACHE: dict[tuple[str, str], tuple] = {}


def _load_pair(repo_id: str, device: str):
    key = (repo_id, device)
    if key not in _CACHE:
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

    with torch.inference_mode():
        hf32 = hf.float()(ids.unsqueeze(0)).logits[0]
        our32 = ours.float()(ids, torch.arange(len(ids), device=device))
    assert our32.shape == hf32.shape
    assert torch.allclose(our32, hf32, atol=1e-3, rtol=1e-3)

    with torch.inference_mode():
        hf_logits = hf(ids.unsqueeze(0)).logits[0].float()
        our_logits = ours(ids, torch.arange(len(ids), device=device)).float()
    assert torch.equal(our_logits.argmax(-1), hf_logits.argmax(-1))
    assert (our_logits - hf_logits).abs().max().item() < 0.5


@pytest.mark.slow
@pytest.mark.parametrize("repo_id", MODELS)
@pytest.mark.parametrize("device", DEVICES)
def test_greedy_decode_matches_hf(repo_id, device):
    hf, ours, tok = _load_pair(repo_id, device)
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()

    hf_out = hf.generate(
        torch.tensor([prompt_ids], device=device),
        do_sample=False,
        repetition_penalty=1.0,
        max_new_tokens=NUM_GREEDY_TOKENS,
        min_new_tokens=NUM_GREEDY_TOKENS,
    )[0, len(prompt_ids):].tolist()

    our_ids = list(prompt_ids)
    with torch.inference_mode():
        for _ in range(NUM_GREEDY_TOKENS):
            ids = torch.tensor(our_ids, device=device)
            logits = ours(ids, torch.arange(len(our_ids), device=device))
            our_ids.append(int(logits[-1].argmax()))

    assert our_ids[len(prompt_ids):] == hf_out
