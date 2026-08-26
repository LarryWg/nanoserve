"""Online serving, driven in process.

Requests still arrive over time on a Poisson schedule, but there is no
socket in the middle: the driver submits straight into the engine and steps
it, so what gets measured is the scheduler rather than a server stack.

The loop is single threaded on purpose. Each turn it admits every request
whose arrival time has passed, then runs one step and timestamps the tokens
that came back. Arrivals land within one step of their schedule, which at
tens of milliseconds is finer than anything being measured, and there is no
lock or queue to wonder about afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import bench
import dataset


def run(engine, requests, schedule, sampling_for, deadline_s=1800) -> list:
    """Submit on schedule, step until everything finishes."""
    records = {}
    pending = list(zip(requests, schedule))
    pending.reverse()                    # pop() takes the earliest
    start = time.perf_counter()
    deadline = start + deadline_s
    live = 0

    while (pending or live) and time.perf_counter() < deadline:
        now = time.perf_counter() - start
        while pending and pending[-1][1] <= now:
            req, offset = pending.pop()
            seq_id = engine.submit(req.prompt_ids, sampling_for(req))
            records[seq_id] = bench.RequestRecord(
                req.prompt_len, req.output_len, 0, time.perf_counter()
            )
            live += 1

        outputs = engine.step()
        stamp = time.perf_counter()
        for out in outputs:
            record = records[out.seq_id]
            record.chunk_times.append(stamp)
            record.num_chunks += 1
            if out.finished:
                live -= 1

        if not outputs and pending:
            # Nothing to run yet; wait for the next arrival rather than spin.
            time.sleep(max(0.0, pending[-1][1] - (time.perf_counter() - start)))

    if live or pending:
        raise RuntimeError(f"{live} running and {len(pending)} unsent at the deadline")
    return [records[k] for k in sorted(records)]


def main():
    parser = argparse.ArgumentParser(description="Online serving, in process.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--engine", default="nanoserve", choices=("nanoserve", "vllm"))
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--decode-first", dest="prefill_first",
                        action="store_false", default=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--out", default="results/online.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from offline import resolve_model

    args.model_path = resolve_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    requests = dataset.load(tokenizer, args.num_prompts, seed=args.seed,
                            fixed_output_len=args.output_len, path=args.dataset)
    for req in requests:
        req.prompt_ids = tokenizer.encode(req.prompt)

    driver = NANOSERVE if args.engine == "nanoserve" else VLLM
    records, config = driver(args, requests, tokenizer)

    wall = max(r.chunk_times[-1] for r in records) - min(r.send_time for r in records)
    result = bench.summarize(records, args.request_rate, wall, 0.0)
    result["engine"] = args.engine
    result["server_config"] = config
    result["provenance"] = bench.provenance(args.model_path, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    s = result["summary"]
    print(f"{args.engine:10s} {s['output_tok_per_s']:8.1f} tok/s  "
          f"offered {s['offered_rate']:g} attained {s['attained_rate']:.2f}  "
          f"TTFT p50 {s['ttft_p50'] * 1000:.0f}ms p99 {s['ttft_p99'] * 1000:.0f}ms  "
          f"ITL p99 {s['itl_p99'] * 1000:.0f}ms  -> {args.out}")


def NANOSERVE(args, requests, tokenizer):
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
        prefill_first=args.prefill_first,
    )
    engine = Engine(scheduler, runner)

    # ignore_eos, so every engine generates the same number of tokens.
    def sampling_for(req):
        return SamplingParams(temperature=0.0, max_new_tokens=req.output_len)

    if args.warmup:
        run(engine, requests[:args.warmup], [0.0] * args.warmup, sampling_for)
    schedule = bench.poisson_schedule(args.request_rate, len(requests), args.seed)
    return run(engine, requests, schedule, sampling_for), engine.info()


def VLLM(args, requests, tokenizer):
    """vLLM's own step loop, driven the same way as nanoserve's.

    add_request/step is the engine underneath its servers, so this compares
    scheduler against scheduler with no HTTP on either side.
    """
    from vllm import LLMEngine, EngineArgs
    from vllm import SamplingParams as VllmSamplingParams
    from vllm import TokensPrompt

    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    ))

    schedule = bench.poisson_schedule(args.request_rate, len(requests), args.seed)
    records = {}
    pending = list(zip(range(len(requests)), requests, schedule))
    pending.reverse()
    start = time.perf_counter()
    live, seen = 0, {}

    while pending or live:
        now = time.perf_counter() - start
        while pending and pending[-1][2] <= now:
            index, req, _ = pending.pop()
            engine.add_request(
                str(index),
                TokensPrompt(prompt_token_ids=req.prompt_ids),
                VllmSamplingParams(temperature=0.0, max_tokens=req.output_len,
                                   ignore_eos=True),
            )
            records[str(index)] = bench.RequestRecord(
                req.prompt_len, req.output_len, 0, time.perf_counter()
            )
            seen[str(index)] = 0
            live += 1

        outputs = engine.step()
        stamp = time.perf_counter()
        for out in outputs:
            record = records[out.request_id]
            # vLLM reports the whole output each step, so count what is new.
            produced = len(out.outputs[0].token_ids)
            for _ in range(produced - seen[out.request_id]):
                record.chunk_times.append(stamp)
                record.num_chunks += 1
            seen[out.request_id] = produced
            if out.finished:
                live -= 1

        if not outputs and pending:
            time.sleep(max(0.0, pending[-1][2] - (time.perf_counter() - start)))

    return [records[k] for k in sorted(records, key=int)], {"engine": "vllm"}


if __name__ == "__main__":
    main()
