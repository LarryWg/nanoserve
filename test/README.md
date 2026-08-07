# Tests

Everything here compares nanoserve against ground truth: either a property
we can check by hand (block accounting) or the real transformers
implementations (RMSNorm, RoPE, the full model).

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers safetensors pytest
```

## Running

Fast tests (no downloads, a few seconds):

```bash
pytest test/ -m "not slow"
```

Everything, including the model equivalence tests:

```bash
pytest test/
```

## What each file checks

- `test_layers.py`: RMSNorm and RoPE produce exactly the same numbers as
  the transformers implementations, including on a flat batch of tokens at
  unrelated positions (the serving case).
- `test_block_manager.py`: paged KV cache block accounting. Allocation,
  the block boundary off-by-one, freeing, and the "one block, one owner"
  invariant.
- `test_model_hf_equivalence.py`: the gate for the whole engine. Our model
  must match HF token-for-token. Marked `slow` because it downloads real
  checkpoints (~2.5 GB on first run, cached afterwards) and runs a 50-token
  greedy decode on CPU (a few minutes). Covers Qwen3-0.6B and
  Qwen2.5-0.5B-Instruct so every architectural branch is exercised.
