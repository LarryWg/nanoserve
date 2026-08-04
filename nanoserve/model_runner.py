"""Model execution: paged attention + (Stage 2) tensor parallelism.

Stage 1 pragmatic path — don't write attention kernels from scratch:
- Load HF Llama/Qwen weights into YOUR OWN model definition (a plain
  nn.Module you write; ~300 lines). You need your own module to control
  the attention call and, later, to shard weights for TP. Copying
  transformers' modeling code defeats the purpose.
- Use flash-attn's `flash_attn_varlen_func` for prefill and a paged decode
  kernel for decode. Options for decode:
    a) flash-attn's kv-cache paged interface (flash_attn_with_kvcache),
    b) write a Triton paged-decode kernel yourself (STRONG writeup material:
       compare your kernel vs flash-attn, show the gap, explain it).
- KV cache tensors: one per layer,
  [num_blocks, 2, num_kv_heads, block_size, head_dim], dtype fp16/bf16.

Stage 2 — tensor parallelism (raw torch.distributed, NCCL backend):
- Attention: shard heads across ranks (column-parallel qkv_proj,
  row-parallel o_proj -> ONE all_reduce after o_proj).
- MLP: column-parallel gate/up_proj, row-parallel down_proj -> ONE
  all_reduce after down_proj.
- Exactly 2 all_reduces per layer. Derive why on paper before coding —
  this derivation IS the interview.
- Embedding/lm_head: vocab-parallel or replicate (replicate is fine at
  this scale; say so and quantify the memory cost).
- Every rank runs the same scheduler decisions: rank 0 schedules and
  broadcasts batch metadata (or run the scheduler deterministically
  everywhere — discuss the tradeoff in the writeup).

Profiling to capture AS YOU GO (screenshots for the writeup):
- Nsight Systems trace showing compute/NCCL overlap (or lack of it)
- decode step time vs batch size (find where you go memory-bound)
"""
import torch


class ModelRunner:
    def __init__(self, model_path: str, block_size: int, tp_rank: int = 0, tp_size: int = 1):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.block_size = block_size
        # TODO Stage 1: build your nn.Module, load HF safetensors into it,
        #   allocate KV cache after measuring free VRAM (-> num_blocks).
        # TODO Stage 2: shard weights at load time based on (tp_rank, tp_size);
        #   never materialize the full weight on every rank (measure load time
        #   and peak host RAM — nice writeup detail).
        raise NotImplementedError

    @torch.inference_mode()
    def forward(self, batch) -> torch.Tensor:
        """Run one step for a ScheduledBatch, return logits for the last
        position of each sequence. Flatten variable-length sequences +
        block tables into kernel inputs (cu_seqlens etc.)."""
        raise NotImplementedError

    def sample(self, logits: torch.Tensor, batch) -> list[int]:
        """Temperature + top-p sampling. Keep it simple; vectorize later
        only if profiling shows it matters (it usually doesn't — say so)."""
        raise NotImplementedError
