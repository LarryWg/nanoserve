"""Measuring the places nanoserve is known to be behind vLLM.

The README names these gaps. Naming them is cheap; this file is what turns
each one into a number, so the writeup argues from measurements instead of
from architecture diagrams.

Two of them:

- Block size. flash-attn will not take a page smaller than 256 tokens, and
  vLLM's default is 16. A sequence only ever fills its last block partway,
  so the bigger page wastes more. That waste is pure arithmetic over the
  real prompt lengths, and needs no GPU.
- Per-step overhead. With no CUDA graphs, every decode step pays python and
  kernel launch cost that does not shrink with the batch. Timing a decode
  step across batch sizes shows how much of a small batch is overhead.
"""
from __future__ import annotations

import argparse
import json
import time


def fragmentation(lengths, block_sizes=(16, 256)) -> dict:
    """Slots held versus tokens actually stored, per block size."""
    out = {}
    tokens = sum(lengths)
    for block_size in block_sizes:
        slots = sum(-(-n // block_size) * block_size for n in lengths)
        out[block_size] = {
            "tokens": tokens,
            "slots_held": slots,
            "wasted_fraction": (slots - tokens) / slots,
            # What the same VRAM is worth in sequences, relative to 16.
            "slots_per_token": slots / tokens,
        }
    return out


def decode_step_times(model_path: str, batch_sizes, prompt_len=128, steps=40) -> dict:
    """Time one decode step at each batch size.

    Flat time as the batch grows means the step is overhead-bound, which is
    exactly what CUDA graphs would fix.
    """
    import torch

    from nanoserve.engine import Engine
    from nanoserve.model_runner import ModelRunner
    from nanoserve.scheduler import Scheduler
    from nanoserve.sequence import SamplingParams

    runner = ModelRunner(model_path)
    results = {}
    for batch_size in batch_sizes:
        scheduler = Scheduler(block_manager=runner.block_manager)
        engine = Engine(scheduler, runner)
        for _ in range(batch_size):
            engine.submit(list(range(prompt_len)),
                          SamplingParams(temperature=0.0, max_new_tokens=steps + 5))
        engine.step()                       # the prefill
        for _ in range(5):                  # warm up the decode path
            engine.step()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            engine.step()
        torch.cuda.synchronize()
        per_step = (time.perf_counter() - start) / steps

        results[batch_size] = {
            "ms_per_step": per_step * 1000,
            "tok_per_s": batch_size / per_step,
        }
        print(f"  batch {batch_size:3d}: {per_step * 1000:6.2f} ms/step  "
              f"{batch_size / per_step:8.1f} tok/s")
        for seq in list(scheduler.running):
            scheduler.finish(seq, now=0.0)
    return results


def main():
    parser = argparse.ArgumentParser(description="Measure the known gaps.")
    parser.add_argument("--model-path")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--out", default="results/gaps.json")
    args = parser.parse_args()

    result = {}
    if args.model_path:
        from transformers import AutoTokenizer

        import dataset

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        requests = dataset.load(tokenizer, args.num_prompts, seed=args.seed,
                                path=args.dataset)
        lengths = [r.prompt_len + r.output_len for r in requests]
        result["fragmentation"] = fragmentation(lengths)
        result["decode_step_times"] = decode_step_times(
            args.model_path, [int(x) for x in args.batch_sizes.split(",")])

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
