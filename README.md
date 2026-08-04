# nanoserve

A distributed LLM inference engine you can actually read. Continuous batching,
paged KV cache, and tensor parallelism in ~2k lines of Python + Triton —
built to understand how vLLM-class systems work, benchmarked honestly against
the real thing.

> **Status: work in progress.** Roadmap below.

## Why this exists

Production inference engines are 100k+ line codebases. nanoserve implements the
core ideas — iteration-level scheduling, PagedAttention-style KV management,
Megatron-style tensor parallelism — at a scale one person can hold in their head,
with a writeup explaining every design decision and every place it loses to vLLM.

## Headline results

<!-- THE chart goes here: throughput vs request rate, nanoserve/vLLM/HF -->
<!-- Second chart: TP scaling 1/2/4 GPUs vs ideal -->

## Architecture

Request -> FastAPI -> Scheduler (continuous batching, token budget, preemption)
-> BlockManager (paged KV cache) -> ModelRunner (flash-attn prefill, paged
decode, TP sharding) -> streamed tokens.

<!-- One clean architecture diagram. One. Not five. -->

## Running it

Python 3.10+. No GPU required — the model definition is verified against
`transformers` on CPU, which is the whole point of the M1 gate.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

**What runs today:** the test suite. `nanoserve/engine.py` (server) and
`benchmarks/bench.py` raise `NotImplementedError` — see the roadmap.

On a CUDA machine, add the attention kernels. flash-attn compiles against your
installed torch, so it needs `--no-build-isolation`:

```bash
pip install -e ".[gpu]" --no-build-isolation
```

## Roadmap

- [ ] Stage 1: single-GPU — continuous batching, paged KV cache, streaming API,
      benchmark vs HF generate and vLLM
- [ ] Stage 2: tensor parallelism over raw NCCL collectives (2–4 GPUs),
      scaling curves + comms/compute breakdown
- [ ] Stage 3 (pick one): chunked prefill, or prefix caching (radix tree),
      or disaggregated prefill/decode

## Design writeup

Every non-obvious decision (block size, prefill-vs-decode priority, preemption
policy, all-reduce placement) is documented in [DESIGN.md] with measurements.

## Benchmarks

Methodology in `benchmarks/` — ShareGPT prompts, Poisson arrivals, 3 runs per
point, exact hardware/commit reported. Including where and why we lose to vLLM.
