"""Paged KV cache block manager.

THIS FILE IS THE HEART OF THE PROJECT. Implement it yourself — every design
decision here is a standard interview question.

Design (mirrors vLLM's PagedAttention, simplified):
- GPU KV cache is pre-allocated as a big tensor of shape
  [num_blocks, 2, num_kv_heads, block_size, head_dim] per layer
  (2 = key and value). block_size is typically 16 tokens.
- A sequence's KV lives in non-contiguous physical blocks; its block_table
  maps logical block i -> physical block id.
- Attention kernels gather K/V through the block table (see model_runner.py).

Decisions YOU must make and be able to defend:
1. block_size: 16 is standard. Smaller -> less internal fragmentation,
   more gather overhead. Measure it. Put the tradeoff chart in your writeup.
2. num_blocks: computed from free VRAM after weights + activations.
   Formula: (free_bytes * utilization) / bytes_per_block. Start with
   utilization=0.9 and explain OOM headroom.
3. Preemption policy when blocks run out: recompute (drop KV, re-prefill
   later) vs swap to CPU. Recompute is simpler — start there, state why.
4. (Stage 3 option) Prefix caching: hash block contents, refcount shared
   blocks, copy-on-write on divergence.
"""


class OutOfBlocks(Exception):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        # TODO: free list of physical block ids. A simple list/deque is fine.
        # TODO: (prefix caching, later) refcounts + content-hash -> block map.
        raise NotImplementedError("implement me")

    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def blocks_needed(self, num_tokens: int) -> int:
        """ceil(num_tokens / block_size)"""
        raise NotImplementedError

    def can_allocate(self, num_tokens: int) -> bool:
        """Used by the scheduler for admission control before prefill."""
        raise NotImplementedError

    def allocate(self, num_tokens: int) -> list[int]:
        """Allocate blocks for a prefill. Returns physical block ids.
        Raise OutOfBlocks if insufficient — scheduler decides what to preempt."""
        raise NotImplementedError

    def append_slot(self, block_table: list[int]) -> list[int]:
        """Called each decode step. If the last block is full, allocate one
        more physical block and append it to the table. Return updated table."""
        raise NotImplementedError

    def free(self, block_table: list[int]) -> None:
        """Return blocks to the free list (on finish or preemption-by-recompute)."""
        raise NotImplementedError
