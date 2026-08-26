"""Offline throughput: every prompt available at t=0, drain them all.

No arrival process and no latency targets, just how fast the engine can
chew through a fixed pile of work. This is the only place HF appears,
because HF has no continuous-batching server and putting it behind one
would measure the queue we wrote for it rather than HF itself.

Each engine runs under its own interpreter (vLLM has its own venv), so the
imports are all lazy and only the engine being measured is loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import metrics
import dataset

DEADLINE_SLACK_S = 600


def run_nanoserve(requests, tokenizer, args) -> dict:
    """Drive the engine directly. No HTTP, so nothing measures the socket."""
    from nanoserve.engine import Engine
    from nanoserve.model_runner import ModelRunner
    from nanoserve.scheduler import Scheduler
    from nanoserve.sequence import SamplingParams

    runner = ModelRunner(
        args.model_path,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    scheduler = Scheduler(
        block_manager=runner.block_manager,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    engine = Engine(scheduler, runner)

    # Tokenizing is not generation, so it happens before the clock starts.
    prompts = [tokenizer.encode(r.prompt) for r in requests]
    for prompt_ids, req in zip(prompts, requests):
        engine.submit(
            prompt_ids,
            SamplingParams(temperature=0.0, max_new_tokens=req.output_len),
        )

    start = time.perf_counter()
    done = 0
    deadline = start + DEADLINE_SLACK_S
    while done < len(requests) and time.perf_counter() < deadline:
        done += sum(out.finished for out in engine.step())
    elapsed = time.perf_counter() - start
    if done < len(requests):
        raise RuntimeError(f"only {done}/{len(requests)} finished before the deadline")
    return {"duration_s": elapsed, "config": engine.info()}


def run_vllm(requests, tokenizer, args) -> dict:
    from vllm import LLM
    from vllm import SamplingParams as VllmSamplingParams

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    params = [
        VllmSamplingParams(temperature=0.0, max_tokens=r.output_len, ignore_eos=True)
        for r in requests
    ]
    start = time.perf_counter()
    llm.generate([r.prompt for r in requests], params)
    elapsed = time.perf_counter() - start
    return {"duration_s": elapsed, "config": {"engine": "vllm"}}


def run_hf(requests, tokenizer, args) -> dict:
    from baselines import hf_static

    sizes = [int(x) for x in args.hf_batch_sizes.split(",")]
    return hf_static.run(requests, args.model_path, sizes)


ENGINES = {"nanoserve": run_nanoserve, "vllm": run_vllm, "hf": run_hf}


def resolve_model(model: str) -> str:
    """A local snapshot. nanoserve's runner needs a directory, and pinning
    all three engines to the same one rules out reading different weights."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def main():
    parser = argparse.ArgumentParser(description="Offline throughput.")
    parser.add_argument("engine", choices=sorted(ENGINES))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--output-len", type=int, help="fixed, instead of ShareGPT's own")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", help="path to the ShareGPT json")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--hf-batch-sizes", default="8,16,32,64")
    parser.add_argument("--out", default="results/offline.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    args.model_path = resolve_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    requests = dataset.load(
        tokenizer, args.num_prompts, seed=args.seed,
        fixed_output_len=args.output_len, path=args.dataset,
    )

    result = ENGINES[args.engine](requests, tokenizer, args)
    output_tokens = sum(r.output_len for r in requests)
    prompt_tokens = sum(r.prompt_len for r in requests)
    result.update(
        engine=args.engine,
        num_requests=len(requests),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        output_tok_per_s=output_tokens / result["duration_s"],
        total_tok_per_s=(prompt_tokens + output_tokens) / result["duration_s"],
        requests_per_s=len(requests) / result["duration_s"],
        provenance=metrics.provenance(args.model_path, args.seed),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.engine:10s} {result['output_tok_per_s']:8.1f} output tok/s   "
          f"{result['total_tok_per_s']:8.1f} total tok/s   -> {args.out}")


if __name__ == "__main__":
    main()
