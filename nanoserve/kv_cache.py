"""The physical paged KV cache: the tensors BlockManager hands out ids for.

Two objects, one job each. BlockManager decides WHICH block a sequence owns
and never touches a tensor; KVCache owns the memory and never decides
anything. Everything they share is the integer block id.

Layout: two tensors, k and v, each

    [num_layers, num_blocks, block_size, num_kv_heads, head_dim]

DESIGN.md sketched one tensor per layer shaped [num_blocks, 2, ...]. This
differs for a concrete reason: flash-attn's paged kernels take k_cache and
v_cache as separate arguments, each [num_blocks, block_size, num_kv_heads,
head_dim]. With the layer axis first, `self.k[layer]` is exactly that, as a
view with no copy. Interleaving k and v on axis 1 instead would force a
stride the kernel rejects. Two big allocations also beat 2*num_layers small
ones for fragmentation.

A "slot" is one token's seat in the cache: slot = block_id * block_size +
offset_in_block. Flattening the block and block_size axes turns a scatter
into a single index_copy_, which is why prefill writes stay one line.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

# The engine's block_size IS the kernel's page size, so the kernel's
# constraint lands here. Upstream flash-attn 2.8.3 requires a multiple of
# 256 -- measured, not read off a doc page: both paged entry points
# (flash_attn_with_kvcache and flash_attn_varlen_func with a block_table)
# raise "Paged KV cache block size must be divisible by 256" at 16.
#
# That is coarser than the 16 vLLM uses, because vLLM ships a patched fork
# of the kernel. The cost is internal fragmentation: a sequence wastes on
# average half a block. At Qwen3-0.6B one block is ~29 MB across all
# layers, so 64 concurrent sequences waste ~1% of a 24 GB cache -- real,
# bounded, and worth measuring rather than assuming. Block size becomes a
# free parameter again at M5, when the Triton paged-decode kernel lands and
# 16/32/64 can actually be benchmarked.
BLOCK_SIZE_MULTIPLE = 256


class KVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        if block_size % BLOCK_SIZE_MULTIPLE != 0 or block_size <= 0:
            raise ValueError(
                f"block_size must be a positive multiple of "
                f"{BLOCK_SIZE_MULTIPLE} (flash-attn paged kernel "
                f"requirement), got {block_size}"
            )
        if num_blocks <= 0:
            raise ValueError(f"need at least one block, got {num_blocks}")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        # zeros, not empty: padding slots inside a partially filled block are
        # read by the kernel's masked lanes on some paths, and NaNs from
        # uninitialized memory poison the softmax even when masked.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (k_cache, v_cache) pair flash-attn wants for this layer."""
        return self.k[layer_idx], self.v[layer_idx]

    def store(
        self,
        layer_idx: int,
        k: torch.Tensor,           # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,
        slot_mapping: torch.Tensor,  # [num_tokens] int64, flat slot per token
    ) -> None:
        """Scatter freshly computed K/V into their slots (prefill path).

        Decode does not call this: flash_attn_with_kvcache appends the new
        token's K/V itself, in the same kernel that reads the cache.
        """
        flat = (self.num_slots, k.shape[1], k.shape[2])
        self.k[layer_idx].view(flat).index_copy_(0, slot_mapping, k)
        self.v[layer_idx].view(flat).index_copy_(0, slot_mapping, v)

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


def bytes_per_block(
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Cost of one block across every layer, K and V. This is the unit the
    free-VRAM budget is divided by to get num_blocks."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    return 2 * num_layers * block_size * num_kv_heads * head_dim * itemsize


def slot_mapping(
    block_table: list[int], positions: Iterable[int], block_size: int
) -> list[int]:
    """Token positions -> flat cache slots, through one sequence's blocks.

    Position p lives in the p // block_size'th block THIS sequence owns; the
    block table turns that logical index into a physical block id. Nothing
    here assumes the physical blocks are contiguous or ordered, which is the
    whole point of paging.
    """
    return [
        block_table[p // block_size] * block_size + p % block_size for p in positions
    ]


def pad_block_tables(tables: list[list[int]]) -> list[list[int]]:
    """Rectangularize per-sequence block tables for the decode kernel.

    flash-attn wants a [batch, max_blocks] int tensor, so shorter sequences
    get padded. Block 0 is the pad value and is harmless: the kernel reads
    at most cache_seqlens tokens per sequence, so it never looks at a slot
    past a sequence's own last block.
    """
    width = max(len(t) for t in tables)
    return [t + [0] * (width - len(t)) for t in tables]
