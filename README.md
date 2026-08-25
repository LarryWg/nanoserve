# nanoserve

**An LLM inference engine you can actually read.**

Continuous batching, paged KV cache, and tensor parallelism in ~2k lines of
Python, built to understand how vLLM-class systems work, and benchmarked
honestly against the real thing.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![uv](https://img.shields.io/badge/pkg-uv-purple)
![tests](https://img.shields.io/badge/tests-146%20passing-brightgreen)

## Why this exists

Production inference engines are 100k+ line codebases. nanoserve implements
the same core ideas at a scale one person can hold in their head:

| Idea | Where it lives |
| --- | --- |
| PagedAttention-style KV management | `nanoserve/block_manager.py`, `nanoserve/kv_cache.py` |
| Iteration-level scheduling (continuous batching) | `nanoserve/scheduler.py` |
| flash-attn varlen prefill + paged decode | `nanoserve/model/attention.py`, `nanoserve/model_runner.py` |
| Step loop and OpenAI-compatible streaming | `nanoserve/engine.py`, `nanoserve/server.py` |
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
and proves nanoserve matches HuggingFace **token-for-token** in a 50-step
greedy decode -- first for the plain forward pass, then again with every
token after the first served out of the paged KV cache.

To serve a model (needs a GPU, since decode runs on the paged path):

```bash
uv run python -m nanoserve.server Qwen/Qwen3-0.6B
curl localhost:8000/v1/completions \
  -d '{"prompt": "The capital of France is", "max_tokens": 16, "stream": true}'
```

The attention kernels are CUDA only. On a linux GPU box `uv sync` installs
them along with everything else; without a GPU the tests that need them skip
themselves. `docs/gpu-setup.md` explains why torch and the kernel wheel are
pinned to each other.

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
- [x] Continuous batching scheduler: FCFS admission, token budget,
      recompute preemption
- [x] Paged KV cache decode: flash-attn varlen prefill, `flash_attn_with_kvcache`
      decode, KV cache sized from measured free VRAM. Verified token-for-token
      against HF through the cache, and per step against the no-cache path
      across block boundaries, batching, and preemption
- [x] Engine step loop: submit, abort, and per-request token streams, with
      the scheduler under a lock and the forward pass outside it
- [x] OpenAI-compatible streaming server: `/v1/completions` over SSE,
      incremental detokenization, client disconnect frees the KV blocks.
      `/health` reports the measured cache size and `/metrics` the
      server's own TTFT, to check a load generator against
- [ ] Benchmarks vs HF generate and vLLM

## Benchmarks

Methodology in `benchmarks/`: ShareGPT prompts, Poisson arrivals, 3 runs per
point, exact hardware and commit reported. Including where and why we lose
to vLLM.
