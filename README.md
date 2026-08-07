# nanoserve

A distributed LLM inference engine you can actually read. Continuous batching,
paged KV cache, and tensor parallelism in ~2k lines of Python + Triton,
built to understand how vLLM-class systems work, benchmarked honestly against
the real thing.

## Why this exists

Production inference engines are 100k+ line codebases. nanoserve implements the
core ideas (iteration-level scheduling, PagedAttention-style KV management,
Megatron-style tensor parallelism) at a scale one person can hold in their head,
with a writeup explaining every design decision and every place it loses to vLLM.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for package management:

```bash
uv sync
```

## Tests

```bash
uv run pytest              # everything
uv run pytest -m "not slow"  # fast tests only, no model downloads
```

See [test/README.md](test/README.md) for what each test checks.

## Architecture

Request -> FastAPI -> Scheduler (continuous batching, token budget, preemption)
-> BlockManager (paged KV cache) -> ModelRunner (flash-attn prefill, paged
decode, TP sharding) -> streamed tokens.

## Benchmarks

Methodology in `benchmarks/`: ShareGPT prompts, Poisson arrivals, 3 runs per
point, exact hardware/commit reported. Including where and why we lose to vLLM.
