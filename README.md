# nanoserve

**An LLM inference engine you can actually read.**

Continuous batching, paged KV cache, and tensor parallelism in ~2k lines of
Python, built to understand how vLLM-class systems work, and benchmarked
honestly against the real thing.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![uv](https://img.shields.io/badge/pkg-uv-purple)
![tests](https://img.shields.io/badge/tests-28%20passing-brightgreen)

## Why this exists

Production inference engines are 100k+ line codebases. nanoserve implements
the same core ideas at a scale one person can hold in their head:

| Idea | Where it lives |
| --- | --- |
| PagedAttention-style KV management | `nanoserve/block_manager.py` |
| Iteration-level scheduling (continuous batching) | `nanoserve/scheduler.py` |
| HF-exact model forward pass | `nanoserve/model/` |

Every non-obvious decision is explained in a comment right where it happens,
written for someone learning inference systems for the first time.

## Quickstart

Uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
git clone https://github.com/LarryWg/nanoserve.git
cd nanoserve
uv sync
uv run pytest            # fast tests, no downloads
uv run pytest -m slow    # adds the HF equivalence suite (downloads ~2.5 GB)
```

The slow suite is the heart of the project: it loads real Qwen checkpoints
and proves nanoserve's forward pass matches HuggingFace **token-for-token**
in a 50-step greedy decode.

## How a request flows

```
prompt -> FastAPI -> Scheduler (continuous batching, preemption)
                  -> BlockManager (paged KV cache)
                  -> ModelRunner (prefill + decode)
                  -> streamed tokens
```

## Current state

- [x] Model forward pass verified token-for-token against HuggingFace
      (Qwen3-0.6B, Qwen2.5-0.5B-Instruct, every architectural branch covered)
- [x] Paged KV block manager with allocation invariants under test
- [ ] Continuous batching scheduler
- [ ] KV-cached decode path
- [ ] Benchmarks vs HF generate and vLLM

## Benchmarks

Methodology in `benchmarks/`: ShareGPT prompts, Poisson arrivals, 3 runs per
point, exact hardware and commit reported. Including where and why we lose
to vLLM.
