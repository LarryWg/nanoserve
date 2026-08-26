# Tests

Everything here compares nanoserve against ground truth: either a property
we can check by hand (block accounting) or the real transformers
implementations (RMSNorm, RoPE, the full model).

## Setup

From the repo root, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv` and installs everything (torch, transformers,
safetensors, pytest) from the lockfile.

## Running

Fast tests (no downloads, a few seconds). This is the default:

```bash
uv run pytest
```

The model equivalence tests download ~2.5 GB of checkpoints, so they are
marked `slow` and excluded by default. Run them explicitly:

```bash
uv run pytest -m slow
```

## Troubleshooting

If every test file fails with `ERROR` during collection, check the error
message:

- `ModuleNotFoundError: No module named 'torch'` (or transformers): pytest
  is running under the wrong Python. Use `uv run pytest`, which always runs
  in the project environment. If you activated `.venv` manually instead,
  run `.venv/bin/python -m pytest test/`.
- `ModuleNotFoundError: No module named 'nanoserve'`: fixed by the
  `pytest.ini` at the repo root (it adds the repo root to the import
  path). If you see this anyway, make sure `pytest.ini` exists and you
  are on a recent commit.

## What each file checks

- `test_layers.py`: RMSNorm and RoPE produce exactly the same numbers as
  the transformers implementations, including on a flat batch of tokens at
  unrelated positions (the serving case).
- `test_block_manager.py`: paged KV cache block accounting. Allocation,
  the block boundary off-by-one, freeing, and the "one block, one owner"
  invariant.
- `test_config.py`: config.json parsing rejects what it cannot compute
  correctly (missing dtype, sliding window, rope scaling, unknown archs)
  instead of guessing.
- `test_weights.py`: the strict safetensors loader's failure paths, pinned
  with tiny synthetic checkpoints (duplicate keys, tied-head mismatch,
  missing/unexpected keys).
- `test_sequence.py`: the sequence lifecycle the scheduler will drive:
  prefill, decode, preemption, stop conditions.
- `test_scheduler.py`: continuous batching policy. FCFS admission, the
  token budget, prefill-first vs decode-first, LIFO preemption, and finish.
- `test_engine.py`: the step loop, with a fake runner standing in for the
  model so it runs anywhere. A request from submit to finished, batching,
  abort from both the waiting queue and mid-generation, and streaming off
  the background thread.
- `test_server.py`: the HTTP layer end to end with a fake model behind it.
  A real engine and scheduler run, so a request goes all the way through
  submit, schedule, step, detokenize, and stream, on a laptop.
- `test_bench.py`: the load-test harness, driven against the fake-model
  server over ASGI. Pins the invariant the benchmarks rest on -- one SSE
  chunk per token -- plus the Poisson schedule, the percentiles, failed
  requests being recorded rather than swallowed, and saturation showing up
  as a lower attained rate.
- `test_gaps.py`: the block fragmentation arithmetic behind the README's
  claim that a 256-token page wastes more KV cache than vLLM's 16.
- `test_model_hf_equivalence.py`: the gate for the whole engine. Our model
  must match HF token-for-token. Marked `slow` because it downloads real
  checkpoints (~2.5 GB on first run, cached afterwards) and runs a 50-token
  greedy decode (a few minutes on CPU). Covers Qwen3-0.6B and
  Qwen2.5-0.5B-Instruct so every architectural branch is exercised. When a
  CUDA GPU is visible, every test runs on both CPU and GPU.
