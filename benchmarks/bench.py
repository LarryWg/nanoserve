"""Load-testing harness. THIS produces the chart that gets you stars.

Methodology (be rigorous; reviewers on HN/r/LocalLLaMA will check):
- Workload: ShareGPT prompts (realistic length distribution), Poisson
  arrivals at a swept request rate (e.g. 1..32 req/s).
- Metrics per run: throughput (output tok/s), TTFT p50/p99, ITL p50/p99.
- Baselines, same GPU, same model, same prompts:
    1) naive HF generate with static batching (the strawman; include it
       but don't ONLY beat the strawman)
    2) vLLM (the real bar; expect to lose, so quantify by how much and
       explain each gap: kernel quality? scheduler? sampling overhead?)
- Report hardware, model, dtype, and exact commit for reproducibility.
- 3 runs per point, plot mean with min/max band. No single-run charts.

The headline chart: throughput vs request rate, three lines
(nanoserve / vLLM / HF), plus a TTFT-vs-load chart. With multiple GPUs,
add tokens/s vs TP degree (1, 2, 4) with an ideal-scaling line, and a
stacked bar of compute vs NCCL time per step from torch profiler.

Honesty rule for the writeup: the section "where we lose to vLLM and why"
will earn more credibility (and stars) than any number you win on.
"""
import argparse
import asyncio
import time

# TODO: async client firing requests at your SSE endpoint with Poisson
# inter-arrival times; collect per-request timestamps; dump JSON;
# separate plot script (matplotlib) reads JSON -> charts.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--num-prompts", type=int, default=200)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
