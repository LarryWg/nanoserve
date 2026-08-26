# nanoserve

**An LLM inference engine you can actually read.**

Continuous batching and a paged KV cache in ~900 lines of single-GPU
Python, built to understand how vLLM-class systems work, and benchmarked
honestly against the real thing.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![uv](https://img.shields.io/badge/pkg-uv-purple)
![tests](https://img.shields.io/badge/tests-132%20passing-brightgreen)

## Why this exists

Production inference engines are 100k+ line codebases. nanoserve implements
the same core ideas at a scale one person can hold in their head:

| Idea | Where it lives |
| --- | --- |
| PagedAttention-style KV management | `nanoserve/block_manager.py`, `nanoserve/kv_cache.py` |
| Iteration-level scheduling (continuous batching) | `nanoserve/scheduler.py` |
| flash-attn varlen prefill + paged decode | `nanoserve/model/attention.py`, `nanoserve/model_runner.py` |
| Step loop: schedule, forward, sample, retire | `nanoserve/engine.py` |
| HF-exact model forward pass | `nanoserve/model/` |

Scope is deliberately single-GPU. Everything here is a technique that a
production engine also has, at a size one person can read in an afternoon.

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

To run it against a model (needs a GPU, since decode runs on the paged
path):

```bash
uv run python benchmarks/online.py --engine nanoserve \
    --model-path Qwen/Qwen3-0.6B --request-rate 8 --num-prompts 200
```

There is no HTTP server. Requests are submitted into the engine and it is
stepped directly, which is what the benchmarks measure.

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

Implemented:

- Model forward pass verified token-for-token against HuggingFace
  (Qwen3-0.6B, Qwen2.5-0.5B-Instruct)
- Paged KV block manager with allocation invariants under test
- Continuous batching scheduler: FCFS admission, token budget,
  recompute preemption
- Paged KV cache decode via flash-attn, sized from measured free VRAM
- In-process benchmarks against HF generate and vLLM under Poisson arrivals

Not implemented:

- CUDA graphs (measured: 24 ms/step of fixed overhead to eliminate)
- Smaller KV pages (measured: 24% of the cache wasted at 256 tokens)
- Chunked prefill
- Prefix caching
- Multi-GPU
- HTTP server (benchmarked through the Python API instead)

## Benchmarks

RTX 4090 24GB, Qwen3-0.6B, bf16, 200 ShareGPT prompts, Poisson arrivals,
mean of 3 runs. vLLM 0.27.1 at its defaults. Both engines are driven
in process through their own step loops, so this compares schedulers
rather than server stacks. Method in `benchmarks/README.md`.

![Output throughput vs request rate](docs/benchmarks/throughput.png)

| offered rate | nanoserve | | | vLLM | | |
| --- | --- | --- | --- | --- | --- | --- |
| req/s | tok/s | attained | TTFT p99 | tok/s | attained | TTFT p99 |
| 1 | 210 | 0.94 | 29 ms | 223 | 1.00 | 31 ms |
| 2 | 394 | 1.76 | 29 ms | 444 | 1.99 | 25 ms |
| 4 | 697 | 3.12 | 28 ms | 877 | 3.92 | 21 ms |
| 8 | 1128 | 5.05 | 353 ms | 1706 | 7.63 | 21 ms |
| 16 | 1331 | 5.96 | 6512 ms | 3196 | 14.30 | 24 ms |
| 32 | 1423 | 6.37 | 10719 ms | 5369 | 24.02 | 28 ms |

Offline, every prompt available at t=0: nanoserve **1040** tok/s, vLLM
**6018**, HF static batching **375**. Continuous batching is worth 2.8x
over static batching; the rest of this section is the 5.8x we give back.

**Read the attained column first.** nanoserve saturates at 6.4 req/s.
Past that the queue grows without bound, so the rows below it describe a
queue rather than an engine -- that is what a 10 second p99 is. vLLM was
still keeping up at 24 req/s and had not saturated at the top of the
sweep.

### Where it goes

One number explains most of it. A decode step costs about the same
whether it is serving 1 sequence or 64:

| batch | 1 | 4 | 16 | 64 |
| --- | --- | --- | --- | --- |
| ms/step | 22.7 | 23.3 | 25.1 | 24.1 |
| tok/s | 44 | 171 | 637 | 2661 |

At 0.6B the actual GPU work at batch 64 is a couple of milliseconds, so
roughly 20 ms of every step is fixed cost that has nothing to do with the
model: python building the step's metadata, several small host to device
copies, and the syncs sampling forces. vLLM erases nearly all of it with
CUDA graphs.

That is most of the saturation gap. At its ceiling nanoserve moves 1423
tok/s, which at 24 ms a step is about 34 tokens per step. vLLM is not
running a bigger batch -- it is running a comparable one several times
more often, which is also why its time to first token never leaves the
20-30 ms band while ours climbs into seconds.

Two more, measured rather than asserted:

- **Inter-token latency follows directly.** nanoserve's ITL p99 is
  51-79 ms against vLLM's 3-11 ms, which is the same fixed step seen from
  the other end.
- **Pages are 16x too big.** flash-attn will not take a KV page smaller
  than 256 tokens; vLLM's default is 16. Over these prompt lengths that
  wastes 24% of the cache against 1.7%, so the same VRAM holds fewer
  sequences.

### The scheduler knob, settled

`scheduler.py` has always carried a `prefill_first` flag and a note to
benchmark both settings once there was something to benchmark with. Both,
same machine, 18 runs each:

| | peak tok/s | peak attained | ITL p99 |
| --- | --- | --- | --- |
| prefill first | 835 | 4.38 req/s | 84-91 ms |
| decode first | 302 | 1.58 req/s | 27-31 ms |

Decode first does what it promises on latency, cutting ITL p99 by roughly
two thirds. It costs 64% of throughput to do it. With prefill only running
when nothing is decodable, admission starves, the batch never grows, and
since a step costs the same regardless of batch size, throughput is batch
size. The default was already right.

(Measured on an earlier build that still had the HTTP server, so those
absolute numbers are lower than the table above; the comparison between
the two settings is what holds.)

### What is next, and why

In order, each justified by a number above rather than by copying a
feature list:

1. **CUDA graphs.** 24 ms flat from batch 1 to 64 on a model whose real
   work is around 2 ms. The largest gap and the least architectural.
2. **Smaller pages.** 24% of the cache wasted against vLLM's 1.7%. Either
   pad and slice inside the 256-token block, or find a kernel that takes
   smaller ones.
3. **Chunked prefill.** The prefill_first result above is the evidence:
   throughput and inter-token latency are currently traded against each
   other, and chunking is how an engine stops having to choose.

Prefix caching belongs on the list but not in the order: nothing in a
ShareGPT workload measures prefix sharing, so there is no number here to
rank it by.

Deliberately not doing: anything that changes which tokens come out.
Speculative decoding and quantization both do, and the HF token-for-token
gate is what every other claim in this README rests on.
