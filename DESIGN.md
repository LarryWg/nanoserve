# DESIGN.md — tinyserve/shardserve

A readable, tensor-parallel LLM inference engine. Continuous batching, paged KV
cache, multi-GPU serving in ~2k lines of Python, benchmarked honestly against vLLM.

**Non-goals (say no to these):** multi-model serving, quantization, LoRA,
speculative decoding, CPU inference, Windows, kernels beyond one Triton decode
kernel, any model family beyond Llama/Qwen dense models.

---

## 1. System overview

```
HTTP client
   │  POST /v1/completions (SSE streaming)
   ▼
FastAPI server (async)          — plumbing
   │  submit(prompt_ids, sampling) → seq_id; per-seq asyncio.Queue for tokens
   ▼
Engine loop (dedicated thread)  — plumbing
   │  while True: batch = scheduler.step(); logits = runner.forward(batch);
   │  tokens = sample(logits); append tokens; finish/stream
   ▼
Scheduler                       — CORE
   │  continuous batching, admission, preemption, token budget
   ▼
BlockManager                    — CORE
   │  paged KV cache: free list, block tables, append_slot, free
   ▼
ModelRunner                     — CORE decisions, assisted implementation
      own Llama nn.Module, HF weight loading, flash-attn prefill,
      paged decode, (Stage 2) TP sharding + NCCL all-reduces
```

Single process per TP rank. Rank 0: HTTP + scheduler + sampling. Ranks 1..N-1:
broadcast-receive loop that mirrors forward passes.

## 2. Sequence lifecycle

States: WAITING → RUNNING → (PREEMPTED → WAITING) → FINISHED.

A `Sequence` owns: seq_id, prompt_token_ids, output_token_ids, sampling params,
block_table (list of physical block ids), timing fields (arrival, first_token,
finish) for TTFT/ITL metrics.

Preemption policy: **recompute** (free blocks, drop KV, re-enter waiting queue,
re-prefill later). No CPU swap in v1. Rationale: swap adds PCIe transfer
machinery for a rare event; recompute costs one extra prefill. Document the
tradeoff; measure preemption frequency under load and report it.

## 3. BlockManager (core file #1)

KV cache: per layer, tensor `[num_blocks, 2, num_kv_heads, block_size, head_dim]`,
dtype bf16. `block_size = 16` tokens (make configurable; benchmark 8/16/32 and
include the chart in the writeup).

`num_blocks` computed at startup: profile free VRAM after weight load + a dummy
forward at max batch to reserve activation headroom, then
`num_blocks = floor(free_bytes * 0.90 / bytes_per_block_all_layers)`.

API:
- `can_allocate(num_tokens) -> bool` — admission control before prefill
- `allocate(num_tokens) -> list[int]` — pop from free list; raise `OutOfBlocks`
- `append_slot(block_table) -> block_table` — decode step; allocate a new block
  only when the last block is full (`num_tokens % block_size == 0` boundary —
  this off-by-one is the classic bug; test prompts at exactly block_size and
  block_size±1 tokens)
- `free(block_table)` — return blocks to free list

Invariants (assert them): a physical block belongs to at most one sequence (v1,
no sharing); free list size + sum of allocated == num_blocks; block tables never
contain duplicates.

## 4. Scheduler (core file #2)

Iteration-level (continuous) batching. Each `step()` returns either a prefill
batch or a decode batch (no chunked prefill in v1; it is the designated Stage 3
extension with a TTFT-vs-ITL tradeoff study).

Config: `max_num_seqs=64`, `max_num_batched_tokens=8192`, policy flag
`prefill_priority: bool`.

step() logic:
1. If `prefill_priority` and waiting non-empty: pop FCFS while
   (sum of prompt lens ≤ token budget) and `can_allocate` and
   `len(running) < max_num_seqs`. Allocate blocks. Return prefill batch.
2. Else decode: for each running seq call `append_slot`; on `OutOfBlocks`,
   preempt the **most recently arrived** running sequence (LIFO victim — it has
   the least sunk compute; document why FIFO victimization thrashes long
   sequences), free its blocks, requeue it, retry. Return decode batch.
3. Neither → None (engine idles on a condition variable).

Benchmark both prefill-first and decode-first; the comparison chart
(TTFT vs ITL across request rates) is a centerpiece of the writeup.

## 5. ModelRunner

- Own `nn.Module` for Llama/Qwen dense: RMSNorm, RoPE, GQA attention, SwiGLU
  MLP. Load HF safetensors by explicit name mapping. **Milestone M1: logits
  match `transformers` (atol=1e-2 bf16) and 50-token greedy decode is
  identical.** Nothing else proceeds until M1 passes.
- Prefill: `flash_attn_varlen_func` with cu_seqlens; write K/V into paged cache
  via a scatter using the block table (a small Triton or index_copy op).
- Decode: start with `flash_attn_with_kvcache` (paged). Then write a Triton
  paged-decode kernel and benchmark against it — keep both behind a flag;
  report the gap honestly (losing to flash-attn with a clear roofline
  explanation is expected and fine).
- Sampling: temperature + top-p, vectorized enough not to show up in profiles;
  measure and state its cost rather than optimizing blind.

## 6. Tensor parallelism (Stage 2)

Raw `torch.distributed`, NCCL backend, one process per GPU via torchrun.

Sharding (Megatron-style, derive on paper first):
- Attention: qkv_proj column-parallel (shard heads; GQA: shard KV heads,
  tp_size must divide num_kv_heads — assert it), o_proj row-parallel →
  **all_reduce #1** after o_proj.
- MLP: gate/up column-parallel, down row-parallel → **all_reduce #2**.
- Exactly 2 all-reduces per layer. Embedding + lm_head replicated in v1
  (quantify the memory cost per rank in the writeup; vocab-parallel is a noted
  extension).
- Weights sharded at load time per rank; never materialize full tensors on
  every rank.

Control plane: rank 0 runs scheduler + sampling; each step it broadcasts a
small metadata struct (seq ids, is_prefill, token ids, block tables) via
`broadcast_object_list`; all ranks execute the same forward. Measure broadcast
overhead; if it shows up in profiles, switch to pre-pinned tensor broadcast and
write up the delta.

Required Stage 2 artifacts: tokens/s vs TP degree (1/2/4) with ideal-scaling
line; per-step time breakdown compute vs NCCL (torch profiler / Nsight
Systems screenshot).

## 7. Server & streaming

FastAPI, `/v1/completions`, OpenAI-compatible request subset, SSE streaming.
Engine loop in a thread (GPU kernels release the GIL; note the multiprocess
alternative in the writeup). Per-sequence `asyncio.Queue` bridged with
`loop.call_soon_threadsafe`. Graceful handling of client disconnect → abort
sequence, free blocks (test this; leaked blocks are the classic slow-death bug).

## 8. Benchmark methodology (non-negotiable)

- Workload: ShareGPT prompts, Poisson arrivals, request-rate sweep 1→32 req/s.
- Metrics: output tok/s, TTFT p50/p99, ITL p50/p99.
- Baselines on identical hardware/model/prompts: HF `generate` static batching,
  and vLLM (expect to lose; quantify and explain each gap).
- 3 runs per point, mean with min/max band. Report GPU, model, dtype, commit.
- Writeup must include "where and why we lose to vLLM."

## 9. Milestones

- **M1** — own model matches HF token-for-token (greedy, 50 tokens). Gate for everything.
- **M2** — paged KV cache + single-seq decode correct across block boundaries.
- **M3** — continuous batching + streaming server; first benchmark chart.
- **M4** — TP=2 matches TP=1 outputs exactly (greedy); then TP=4 scaling curves.
- **M5** — Triton paged-decode kernel + kernel benchmark section.
- **M6** — writeup + launch.

Correctness tests at every milestone: fixed-seed greedy equivalence vs HF;
prompts of length block_size−1/exact/+1; concurrent 32-seq output equality vs
sequential runs; disconnect-frees-blocks test.

---

## Appendix: instructions for AI coding agents

Role: implement plumbing, core and tests against this spec. Do not add features outside the non-goals fence. Prefer
boring code over clever code; target readability for an audience learning
inference systems. Every module ≤ ~300 lines. No new dependencies beyond:
torch, flash-attn, triton, transformers (weights/tokenizer only), safetensors,
fastapi, uvicorn, numpy, matplotlib. All performance claims must come from the
benchmark harness, never estimated. When a design decision is ambiguous, stop
and ask rather than choosing silently.
