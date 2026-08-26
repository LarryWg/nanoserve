from __future__ import annotations

import argparse
import json
import os
import time

import metrics
import dataset


def run(engine, requests, schedule, sampling_for, deadline_s=1800) -> list:
    records = {}
    pending = list(zip(requests, schedule))
    pending.reverse()
    start = time.perf_counter()
    deadline = start + deadline_s
    live = 0

    while (pending or live) and time.perf_counter() < deadline:
        now = time.perf_counter() - start
        while pending and pending[-1][1] <= now:
            req, offset = pending.pop()
            seq_id = engine.submit(req.prompt_ids, sampling_for(req))
            records[seq_id] = metrics.RequestRecord(
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
            time.sleep(max(0.0, pending[-1][1] - (time.perf_counter() - start)))

    if live or pending:
        raise RuntimeError(f"{live} running and {len(pending)} unsent at the deadline")
    return [records[k] for k in sorted(records)]


def main():
    parser = argparse.ArgumentParser(description="Online serving, in process.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--engine", default="nanoserve", choices=("nanoserve", "vllm"))
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--rates", help="sweep these rates in one process, e.g. 1,2,4")
    parser.add_argument("--runs", type=int, default=1)
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
    engine, config = driver(args)
    rates = [float(x) for x in args.rates.split(",")] if args.rates else [args.request_rate]
    provenance = metrics.provenance(args.model_path, args.seed)

    for rate in rates:
        for run in range(1, args.runs + 1):
            schedule = metrics.poisson_schedule(rate, len(requests), run)
            records = engine(requests, schedule)
            wall = (max(r.chunk_times[-1] for r in records)
                    - min(r.send_time for r in records))
            result = metrics.summarize(records, rate, wall)
            result["engine"] = args.engine
            result["server_config"] = config
            result["provenance"] = provenance
            path = args.out
            if args.rates or args.runs > 1:
                stem = args.out[:-5] if args.out.endswith(".json") else args.out
                path = f"{stem}-r{rate:g}-run{run}.json"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
            s = result["summary"]
            print(f"{args.engine:10s} rate {rate:5g} run {run}  "
                  f"{s['output_tok_per_s']:8.1f} tok/s  "
                  f"attained {s['attained_rate']:.2f}  "
                  f"TTFT p50 {s['ttft_p50'] * 1000:.0f}ms p99 {s['ttft_p99'] * 1000:.0f}ms  "
                  f"ITL p99 {s['itl_p99'] * 1000:.0f}ms  -> {path}", flush=True)


def NANOSERVE(args):
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

    def sampling_for(req):
        return SamplingParams(temperature=0.0, max_new_tokens=req.output_len)

    warmed = [False]

    def one_point(requests, schedule):
        if args.warmup and not warmed[0]:
          run(engine, requests[:args.warmup], [0.0] * args.warmup, sampling_for)
          warmed[0] = True
        return run(engine, requests, schedule, sampling_for)

    return one_point, engine.info()


def VLLM(args):
    from vllm import LLMEngine, EngineArgs
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt

    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    ))

    def one_point(requests, schedule):
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
                records[str(index)] = metrics.RequestRecord(
                    req.prompt_len, req.output_len, 0, time.perf_counter()
                )
                seen[str(index)] = 0
                live += 1

            outputs = engine.step()
            stamp = time.perf_counter()
            for out in outputs:
                record = records[out.request_id]
                produced = len(out.outputs[0].token_ids)
                for _ in range(produced - seen[out.request_id]):
                    record.chunk_times.append(stamp)
                    record.num_chunks += 1
                seen[out.request_id] = produced
                if out.finished:
                    live -= 1

            if not outputs and pending:
                time.sleep(max(0.0, pending[-1][2] - (time.perf_counter() - start)))

        return [records[k] for k in sorted(records, key=int)]

    return one_point, {"engine": "vllm"}


if __name__ == "__main__":
    main()
