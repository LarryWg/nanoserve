"""The paged KV cache: the tensors that hold every sequence's keys and values.

BlockManager decides which block a sequence owns. This file owns the memory.
All they share is an integer block id.

Layout is two tensors, k and v, each shaped
[num_layers, num_blocks, block_size, num_kv_heads, head_dim].
Layer first means k[layer] is already the tensor the kernel wants, no copy.

A slot is one token's seat in the cache:
    slot = block_id * block_size + offset inside the block
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

# block_size is the page size the kernel sees, so the kernel's limit lands
# here: flash attn 2.8.3 refuses any page size that 256 does not divide, on
# both of its paged entry points. vLLM uses 16 because it patches the kernel.
# A big page costs wasted space, half a block per sequence on average.
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
                "block_size must be a positive multiple of "
                f"{BLOCK_SIZE_MULTIPLE} (flash attn paged kernel "
                f"requirement), got {block_size}"
            )
        if num_blocks <= 0:
            raise ValueError(f"need at least one block, got {num_blocks}")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        # zeros, not empty. Unused slots in a half full block can still be
        # read, and stray NaNs would poison the softmax.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (k_cache, v_cache) pair the kernel wants for this layer."""
        return self.k[layer_idx], self.v[layer_idx]

    def store(
        self,
        layer_idx: int,
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,
        slot_mapping: torch.Tensor,  # [num_tokens] int64, one slot per token
    ) -> None:
        """Write fresh keys and values into their slots. Prefill only.

        Decode skips this. Its kernel appends the new key and value itself.
        """
        # Folding blocks and offsets into one axis makes this a single copy.
        flat_shape = (self.num_slots, k.shape[1], k.shape[2])
        self.k[layer_idx].view(flat_shape).index_copy_(0, slot_mapping, k)
        self.v[layer_idx].view(flat_shape).index_copy_(0, slot_mapping, v)

    def nbytes(self) -> int:
        """Total bytes held, keys and values together."""
        return 2 * self.k.numel() * self.k.element_size()


def bytes_per_block(
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """What one block costs across every layer, keys and values.

    Free VRAM divided by this is how many blocks fit.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    return 2 * num_layers * block_size * num_kv_heads * head_dim * itemsize


def slot_mapping(
    block_table: list[int], positions: Iterable[int], block_size: int
) -> list[int]:
    """Token positions to flat cache slots, for one sequence.

    Position p sits in this sequence's block number p // block_size. The
    block table turns that into a physical block id. Nothing here assumes
    the blocks sit next to each other, which is the point of paging.
    """
    return [
        block_table[p // block_size] * block_size + p % block_size for p in positions
    ]


def pad_block_tables(tables: list[list[int]]) -> list[list[int]]:
    """Make the block tables rectangular for the decode kernel.

    The kernel wants one [batch, max_blocks] tensor, so short sequences get
    padded. Block 0 is a safe filler because the kernel reads only as many
    tokens as cache_seqlens claims, never the padding.
    """
    width = max(len(t) for t in tables)
    return [t + [0] * (width - len(t)) for t in tables]
