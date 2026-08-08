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
"""
import argparse

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
