"""HuggingFace generate with static batching: the thing to beat.

Static batching is what you write before you know about continuous
batching. A batch starts together and finishes together, so every sequence
in it waits for the longest one, and no new request can join until the
whole batch is done. That idle time is the gap continuous batching closes,
and this baseline exists to measure it rather than assert it.

Two choices here are deliberately generous to HF, so the comparison cannot
be accused of strawmanning:

- Requests are sorted by output length before batching, so each batch is as
  uniform as it can be and the padding waste is as small as it can be.
- Batch size is swept and the best result wins.
"""
from __future__ import annotations

import time


def run(requests, model_path: str, batch_sizes, dtype="bfloat16") -> dict:
    """Generate everything, returning the best batch size and its timing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"          # decoder-only models generate on the right
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, dtype)).to("cuda")
    model.eval()

    ordered = sorted(requests, key=lambda r: r.output_len)
    results = {}
    for batch_size in batch_sizes:
        elapsed = _run_once(model, tokenizer, ordered, batch_size)
        output_tokens = sum(r.output_len for r in ordered)
        results[batch_size] = {
            "duration_s": elapsed,
            "output_tok_per_s": output_tokens / elapsed,
        }
        print(f"  batch {batch_size:3d}: {output_tokens / elapsed:8.1f} tok/s")

    best = max(results, key=lambda b: results[b]["output_tok_per_s"])
    return {"best_batch_size": best, "by_batch_size": results, **results[best]}


def _run_once(model, tokenizer, ordered, batch_size: int) -> float:
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(0, len(ordered), batch_size):
        batch = ordered[i:i + batch_size]
        # The whole batch runs for the longest output in it. That waste is
        # the point of the measurement, so it is not corrected for.
        length = max(r.output_len for r in batch)
        inputs = tokenizer(
            [r.prompt for r in batch], return_tensors="pt", padding=True
        ).to("cuda")
        with torch.inference_mode():
            model.generate(
                **inputs,
                do_sample=False,
                repetition_penalty=1.0,   # instruct configs ship 1.1 and apply it
                min_new_tokens=length,    # no early exit, same as ignore_eos
                max_new_tokens=length,
                pad_token_id=tokenizer.pad_token_id,
            )
    torch.cuda.synchronize()
    return time.perf_counter() - start
