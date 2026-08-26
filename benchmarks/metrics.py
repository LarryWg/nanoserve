"""What a benchmark run records, and how it is summarised.

Metric definitions, pinned here so every driver means the same thing:

- TTFT: submitting a request to its first token.
- ITL: the gap between consecutive tokens of one request. One timestamp
  per token, so a step that produced nothing printable still counts.
- E2E: submitting a request to its last token.
- Attained rate, next to the offered rate. Past saturation the queue grows
  without bound and a fixed prompt count simply takes longer, so a run that
  fell short of what was asked for is describing a queue, not an engine.
"""
from __future__ import annotations

import random
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class RequestRecord:
    prompt_len: int
    output_len: int          # asked for
    num_chunks: int          # delivered; one per token
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


def summarize(records: list, offered_rate: float, wall: float) -> dict:
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
            "attained_rate": len(ok) / wall if wall else 0.0,
            "output_tokens": output_tokens,
            "output_tok_per_s": output_tokens / wall if wall else 0.0,
            "ttft_p50": percentile(ttfts, 50), "ttft_p99": percentile(ttfts, 99),
            "itl_p50": percentile(itls, 50), "itl_p99": percentile(itls, 99),
            "e2e_p50": percentile(e2es, 50), "e2e_p99": percentile(e2es, 99),
        },
        "requests": [asdict(r) for r in records],
    }


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
