from __future__ import annotations

from collections.abc import Iterable

import torch

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
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def num_slots(self) -> int:
        return self.num_blocks * self.block_size

    def layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer_idx], self.v[layer_idx]

    def store(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        flat_shape = (self.num_slots, k.shape[1], k.shape[2])
        self.k[layer_idx].view(flat_shape).index_copy_(0, slot_mapping, k)
        self.v[layer_idx].view(flat_shape).index_copy_(0, slot_mapping, v)

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()


def bytes_per_block(
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    itemsize = torch.empty(0, dtype=dtype).element_size()
    return 2 * num_layers * block_size * num_kv_heads * head_dim * itemsize


def slot_mapping(
    block_table: list[int], positions: Iterable[int], block_size: int
) -> list[int]:
    return [
        block_table[p // block_size] * block_size + p % block_size for p in positions
    ]


def pad_block_tables(tables: list[list[int]]) -> list[list[int]]:
    width = max(len(t) for t in tables)
    return [t + [0] * (width - len(t)) for t in tables]
