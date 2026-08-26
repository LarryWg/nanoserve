# Benchmarks

Numbers are only worth as much as the method behind them. This file is the
method, written down so the results can be argued with.

## What is measured

Two different questions, measured two different ways.

**Online serving** (`bench.py`) — requests arrive over time on a Poisson
schedule, as they do in production. This is where latency means anything.

- TTFT: time from sending a request to its first token
- ITL: time between consecutive tokens, one chunk per token
- Output throughput: tokens generated per second of wall clock
- Attained rate next to offered rate

**Offline throughput** (`offline.py`) — every prompt available at t=0, drain
them all. No latency targets, just how fast the engine chews through a fixed
pile of work.

HF appears only in the offline comparison. It has no continuous-batching
server, and putting it behind one would measure the queue we wrote for it
rather than HF itself.

## Reading the numbers

**Attained rate is the first thing to look at.** Past saturation the queue
grows without bound and a fixed prompt count simply takes longer, so a point
whose attained rate falls short of what was offered is describing a queue,
not an engine. The latency numbers at that point are real but they are not
measuring what they look like they measure.

**One timestamp per token, or the run is rejected.** Both drivers stamp
tokens as the engine hands them back. A driver that reads output in bulk
instead would give every token of a request the same timestamp, collapsing
TTFT into end-to-end latency and ITL into zero; `metrics.py` refuses any run
that looks like that rather than reporting it.

## Fairness

Anyone can win a benchmark by picking the rules afterwards. The rules here:

- `ignore_eos` everywhere, so every engine generates the **same number of
  tokens** for the same prompt. nanoserve and vLLM both support it; HF gets
  `min_new_tokens == max_new_tokens`.
- Same model, same bf16 weights, same tokenizer, same prompts, same seed.
  `repetition_penalty=1.0` pinned, because instruct checkpoints ship 1.1 in
  their config and HF applies it even with sampling off.
- One warmup run per configuration, discarded.
- Three runs per point. Charts show the mean with a min/max band, never a
  single run.
- HF's static batching gets the generous version: requests sorted by output
  length so padding waste is as small as it can be, and the batch size swept
  with the best taken.
- Each engine's **actual** KV cache size is recorded, not the utilization
  knob. nanoserve measures free VRAM at startup and vLLM computes it
  differently, so the same 0.9 does not mean the same cache.
- vLLM runs in its own venv. It pins its own torch, and running the baseline
  must not be able to change the thing being measured.

## Running it

```bash
uv sync --python 3.12 --group bench
bash benchmarks/baselines/setup_vllm.sh

MODEL=Qwen/Qwen3-0.6B bash benchmarks/run_sweep.sh     # the whole matrix
uv run --group bench python benchmarks/plot.py         # charts + results.md
```

One point at a time, against an already-running server:

```bash
uv run python benchmarks/bench.py --request-rate 8 --num-prompts 200 \
    --engine nanoserve --model Qwen/Qwen3-0.6B --out results/one.json
```

The known-gap measurements, which need no server:

```bash
uv run python benchmarks/gaps.py --model-path Qwen/Qwen3-0.6B
```

## Workload

ShareGPT (`ShareGPT_V3_unfiltered_cleaned_split.json`), the standard serving
workload. Real conversations, so prompt and output lengths have the long tail
that makes scheduling interesting; uniform synthetic prompts would hide
exactly the behaviour these benchmarks exist to measure.

First human turn is the prompt, first reply gives the target output length.
Prompts under 4 or over 1024 tokens are dropped, as are outputs under 4 and
totals over 2048 — the usual filter, which removes the degenerate entries
that would otherwise dominate a 200-prompt sample.

## Results

Charts and the numbers as a table are in the repository README. The raw
per-run summaries, including provenance for every run, are in
`docs/benchmarks/summaries.json`.

The full run files carry a timestamp for every token of every request,
which is 43 MB for one sweep, so they are not committed. `plot.py` reads
either: point it at a `results/` directory of raw runs.

Recorded for the committed sweep: RTX 4090 24GB, driver 580.159.04,
torch 2.8.0+cu126, flash-attn 2.8.3.post1, transformers 5.14.1,
vLLM 0.27.1, Qwen3-0.6B in bf16. nanoserve's KV cache measured 180,480
tokens against vLLM's 183,600, so both ran on effectively the same cache
budget.
