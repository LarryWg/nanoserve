# nanoserve

**A small LLM inference engine you can read end to end.**

nanoserve implements continuous batching and a paged KV cache in about 1,000
lines of single-GPU Python. It is built to explain the core ideas behind
production inference engines without hiding them inside a large codebase.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![uv](https://img.shields.io/badge/pkg-uv-purple)
![tests](https://img.shields.io/badge/tests-132%20passing-brightgreen)

## What is included

- Continuous batching with FCFS admission and recompute preemption
- Paged KV cache allocation and block accounting
- FlashAttention prefill and paged decode
- Batched temperature and top-p sampling
- Qwen model loading from safetensors
- Token-for-token tests against Hugging Face

The project is intentionally single GPU. It does not include an HTTP server,
multi-GPU inference, CUDA graphs, chunked prefill, or prefix caching.

## Setup

Inference requires Linux, an NVIDIA GPU, and Python 3.12. See
[`docs/gpu-setup.md`](docs/gpu-setup.md) for the CUDA requirements.

```bash
git clone https://github.com/LarryWg/nanoserve.git
cd nanoserve
uv sync --python 3.12
```

## Generate text

```bash
uv run python generate.py "Explain paged attention"
```

The default model is `Qwen/Qwen3-0.6B`. The first run downloads the model.

```bash
uv run python generate.py "Write a short poem" \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --max-new-tokens 100 \
    --temperature 0.7 \
    --top-p 0.9
```

## Run tests

```bash
uv run pytest
uv run pytest -m slow
```

The default suite runs without model downloads. The slow suite downloads the
Qwen checkpoints and checks 50-token greedy decodes against Hugging Face.

## Request flow

```text
prompt token ids -> Engine -> Scheduler -> BlockManager
                       |                     |
                       +-> ModelRunner <-----+
                               |
                         generated token
```

## Benchmarks

On an RTX 4090 with Qwen3-0.6B, nanoserve reached 1,423 output tokens per
second in the online sweep and 1,040 output tokens per second offline.

The complete methodology, vLLM comparison, charts, and saturation analysis
are in [`benchmarks/README.md`](benchmarks/README.md).
