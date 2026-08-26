"""Load-testing harness.

Methodology:
- Workload: ShareGPT prompts (realistic length distribution), Poisson
  arrivals at a swept request rate (e.g. 1..32 req/s).
- Metrics per run: throughput (output tok/s), TTFT p50/p99, ITL p50/p99.
- Baselines on the same GPU, model, and prompts: naive HF generate with
  static batching, and vLLM. Where vLLM wins, quantify the gap and name
  the cause (kernels, scheduler, sampling overhead).
- Report hardware, model, dtype, and exact commit. 3 runs per point,
  plot mean with min/max band; no single-run charts.

Headline chart: throughput vs request rate (nanoserve / vLLM / HF), plus
TTFT vs load. With multiple GPUs add tokens/s vs TP degree against an
ideal-scaling line, and a stacked bar of compute vs NCCL time per step.

Two numbers here are easy to get wrong and worth stating:

- Attained rate is reported next to offered rate. Past saturation the queue
  grows without bound and a fixed prompt count just takes longer, so a run
  whose attained rate falls short of what was asked for is saturated, not
  slow.
- Client CPU time is reported too. A python load generator can become the
  thing being measured, and /metrics on the server is the cross-check.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import resource
import subprocess
import time
from dataclasses import asdict, dataclass, field


@dataclass
class RequestRecord:
    prompt_len: int
    output_len: int          # asked for
    num_chunks: int          # delivered; one chunk per token
    send_time: float
    chunk_times: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def ttft(self) -> float:
        return self.chunk_times[0] - self.send_time

    @property
    def e2e(self) -> float:
        return self.chunk_times[-1] - self.send_time

    @property
    def itls(self) -> list[float]:
        return [b - a for a, b in zip(self.chunk_times, self.chunk_times[1:])]


def poisson_schedule(rate: float, count: int, seed: int = 0) -> list[float]:
    """Arrival offsets in seconds. An infinite rate means send everything
    at once, which is the burst case the offline benchmarks use."""
    if rate == float("inf"):
        return [0.0] * count
    rng = random.Random(seed)
    offsets, clock = [], 0.0
    for _ in range(count):
        offsets.append(clock)
        clock += rng.expovariate(rate)
    return offsets


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank, so the answer is always a value that happened."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


async def one_request(client, url: str, req, model: str, into: list, index: int) -> None:
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.output_len,
        "temperature": 0.0,
        "ignore_eos": True,      # fixed output length, so engines compare
        "stream": True,
    }
    record = RequestRecord(req.prompt_len, req.output_len, 0, time.perf_counter())
    try:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:].strip() == "[DONE]":
                    break
                record.chunk_times.append(time.perf_counter())
                record.num_chunks += 1
    except Exception as exc:                      # a failed request is data
        record.error = f"{type(exc).__name__}: {exc}"
    # Slotted by index, not appended: results otherwise come back in
    # completion order and stop lining up with the prompts that made them.
    into[index] = record


async def benchmark(
    url: str,
    requests: list,
    rate: float,
    model: str = "nanoserve",
    seed: int = 0,
    client=None,
) -> dict:
    """Fire the requests on a Poisson schedule and gather what came back."""
    import httpx

    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    schedule = poisson_schedule(rate, len(requests), seed)
    records: list[RequestRecord] = [None] * len(requests)
    tasks = []
    cpu_before = _cpu_seconds()
    start = time.perf_counter()
    try:
        for index, (req, offset) in enumerate(zip(requests, schedule)):
            behind = offset - (time.perf_counter() - start)
            if behind > 0:
                await asyncio.sleep(behind)
            tasks.append(asyncio.create_task(
                one_request(client, url, req, model, records, index)))
        await asyncio.gather(*tasks)
    finally:
        if owned:
            await client.aclose()
    wall = time.perf_counter() - start
    return summarize(records, rate, wall, _cpu_seconds() - cpu_before)


def summarize(records: list, offered_rate: float, wall: float, client_cpu: float) -> dict:
    records = [r for r in records if r is not None]
    ok = [r for r in records if r.error is None and r.chunk_times]
    ttfts = [r.ttft for r in ok]
    itls = [x for r in ok for x in r.itls]
    e2es = [r.e2e for r in ok]
    output_tokens = sum(r.num_chunks for r in ok)
    return {
        "summary": {
            "num_requests": len(records),
            "num_failed": len(records) - len(ok),
            "duration_s": wall,
            "offered_rate": offered_rate,
            # Short of the offered rate means the server saturated, and the
            # latencies below describe a queue rather than the engine.
            "attained_rate": len(ok) / wall if wall else 0.0,
            "output_tokens": output_tokens,
            "output_tok_per_s": output_tokens / wall if wall else 0.0,
            "ttft_p50": percentile(ttfts, 50), "ttft_p99": percentile(ttfts, 99),
            "itl_p50": percentile(itls, 50), "itl_p99": percentile(itls, 99),
            "e2e_p50": percentile(e2es, 50), "e2e_p99": percentile(e2es, 99),
            # Above roughly 0.8 the load generator is a suspect itself.
            "client_cpu_fraction": client_cpu / wall if wall else 0.0,
        },
        "requests": [asdict(r) for r in records],
    }


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def provenance(model: str, seed: int) -> dict:
    """Everything needed to tell whether two runs are comparable."""
    info = {"model": model, "seed": seed, "gpu": _shell(
        "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")}
    info["commit"] = _shell("git rev-parse --short HEAD")
    for package in ("torch", "flash_attn", "vllm", "transformers"):
        try:
            info[package] = __import__(package).__version__
        except Exception:
            info[package] = None
    return info


def _shell(command: str) -> str | None:
    try:
        out = subprocess.run(command.split(), capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


async def _main(args) -> None:
    import httpx

    from transformers import AutoTokenizer

    import dataset

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    requests = dataset.load(
        tokenizer, args.num_prompts, seed=args.seed,
        fixed_output_len=args.output_len, path=args.dataset,
    )

    base = args.url.rsplit("/v1/", 1)[0]
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        health = (await client.get(f"{base}/health")).json()
        if args.warmup:
            await benchmark(args.url, requests[:args.warmup], float("inf"),
                            args.model, args.seed, client)
        result = await benchmark(args.url, requests, args.request_rate,
                                 args.model, args.seed, client)
        try:
            result["server_metrics"] = (await client.get(f"{base}/metrics")).json()
        except Exception:
            result["server_metrics"] = None

    result["engine"] = args.engine
    result["provenance"] = provenance(args.model, args.seed)
    result["server_config"] = health.get("config", health)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    print(f"{s['output_tok_per_s']:8.1f} tok/s   "
          f"offered {s['offered_rate']} attained {s['attained_rate']:.2f} req/s   "
          f"TTFT p50 {s['ttft_p50'] * 1000:.0f}ms p99 {s['ttft_p99'] * 1000:.0f}ms   "
          f"ITL p99 {s['itl_p99'] * 1000:.0f}ms   -> {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="nanoserve")
    parser.add_argument("--engine", default="nanoserve", help="label for the plots")
    parser.add_argument("--tokenizer", help="defaults to --model")
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--output-len", type=int, help="fixed, instead of ShareGPT's own")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", help="path to the ShareGPT json")
    parser.add_argument("--out", default="results/run.json")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
