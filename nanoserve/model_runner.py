"""Model execution: paged attention now, tensor parallelism later.

Pragmatic path for attention; do not write kernels from scratch yet:
- The model definition lives in nanoserve/model/model.py and loads HF
  safetensors weights. Having our own nn.Module is what lets us control
  the attention call and later shard weights across GPUs.
- Prefill: flash-attn's flash_attn_varlen_func, which takes cu_seqlens and
  handles the flat variable-length batch directly.
- Decode: start with flash-attn's paged kv-cache interface
  (flash_attn_with_kvcache). A hand-written Triton paged-decode kernel
  comes later, benchmarked against flash-attn so the gap is measured, not
  guessed.
- KV cache tensors: one per layer, shaped
  [num_blocks, 2, num_kv_heads, block_size, head_dim], dtype fp16/bf16.

Tensor parallelism (raw torch.distributed, NCCL backend):
- Attention: shard heads across ranks. qkv_proj is column-parallel, o_proj
  is row-parallel, with one all_reduce after o_proj.
- MLP: gate/up_proj are column-parallel, down_proj is row-parallel, with
  one all_reduce after down_proj.
- Exactly 2 all_reduces per layer. Deriving why on paper before coding is
  the exercise.
- Embedding and lm_head: replicate on every rank (fine at this scale;
  quantify the memory cost when writing it up).
- Every rank must run the same scheduler decisions: rank 0 schedules and
  broadcasts the batch metadata each step.

Profiling to capture as you go:
- Nsight Systems trace showing compute/NCCL overlap (or lack of it)
- decode step time vs batch size (find where the step goes memory-bound)
"""
import torch


class ModelRunner:
    def __init__(self, model_path: str, block_size: int, tp_rank: int = 0, tp_size: int = 1):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.block_size = block_size
        # TODO: build the model, load HF safetensors into it, and allocate
        #   the KV cache after measuring free VRAM (that gives num_blocks).
        # TODO (multi-GPU): shard weights at load time based on
        #   (tp_rank, tp_size); never materialize the full weight on every
        #   rank. Measuring load time and peak host RAM is a nice detail.
        raise NotImplementedError

    @torch.inference_mode()
    def forward(self, batch) -> torch.Tensor:
        """Run one step for a ScheduledBatch, return logits for the last
        position of each sequence. Flatten variable-length sequences +
        block tables into kernel inputs (cu_seqlens etc.)."""
        raise NotImplementedError

    def sample(self, logits: torch.Tensor, batch) -> list[int]:
        """Temperature + top-p sampling. Keep it simple; vectorize later
        only if profiling shows it matters (it usually does not)."""
        raise NotImplementedError
