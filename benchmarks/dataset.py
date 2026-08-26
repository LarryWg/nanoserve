"""ShareGPT prompts, the standard serving workload.

Real conversations, so prompt and output lengths have the long tail that
makes scheduling interesting. Uniform-length synthetic prompts would hide
exactly the behaviour these benchmarks exist to measure.

Every engine and every repeat gets the identical list, sampled from a fixed
seed, so a difference between two runs is the engine and not the data.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

# The usual filter. Very short entries are mostly junk, and very long ones
# would dominate a 200-prompt sample on their own.
MIN_PROMPT_TOKENS = 4
MAX_PROMPT_TOKENS = 1024
MIN_OUTPUT_TOKENS = 4
MAX_TOTAL_TOKENS = 2048


@dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int
    # Filled in by whoever has the tokenizer. The in-process drivers submit
    # token ids directly, so neither engine pays for tokenizing mid-run.
    prompt_ids: list | None = None


def download() -> str:
    """Fetch the dataset, or return the cached copy. About 650 MB."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")


def load(
    tokenizer,
    num_prompts: int,
    seed: int = 0,
    fixed_output_len: int | None = None,
    path: str | None = None,
) -> list[Request]:
    """Sample num_prompts requests, tokenized with the served model's own
    tokenizer so the lengths mean the same thing to every engine."""
    with open(path or download()) as f:
        raw = json.load(f)

    # First human turn is the prompt, first reply gives the target length.
    pairs = [
        (c["conversations"][0]["value"], c["conversations"][1]["value"])
        for c in raw
        if len(c.get("conversations", [])) >= 2
    ]

    # Sampling before tokenizing keeps this to seconds instead of minutes.
    # Oversample, because the filter below throws some away.
    rng = random.Random(seed)
    rng.shuffle(pairs)

    requests: list[Request] = []
    for prompt, completion in pairs:
        if len(requests) == num_prompts:
            break
        prompt_len = len(tokenizer.encode(prompt))
        output_len = fixed_output_len or len(tokenizer.encode(completion))
        if prompt_len < MIN_PROMPT_TOKENS or output_len < MIN_OUTPUT_TOKENS:
            continue
        if prompt_len > MAX_PROMPT_TOKENS or prompt_len + output_len > MAX_TOTAL_TOKENS:
            continue
        requests.append(Request(prompt, prompt_len, output_len))

    if len(requests) < num_prompts:
        raise RuntimeError(
            f"only {len(requests)} of {num_prompts} prompts survived the filter"
        )
    return requests
