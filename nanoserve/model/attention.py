from __future__ import annotations

from dataclasses import dataclass

import torch

from ..kv_cache import KVCache

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:  # pragma: no cover
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

_MISSING = (
    "flash attn is required for the paged attention path and is CUDA only. "
    "The plain path (attn_metadata=None) runs without it."
)


@dataclass(frozen=True)
class AttentionMetadata:
    is_prefill: bool
    kv_cache: KVCache

    slot_mapping: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int = 0

    block_tables: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None


def paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx: int,
    meta: AttentionMetadata,
) -> torch.Tensor:
    if flash_attn_varlen_func is None:
        raise ImportError(_MISSING)
    if meta.is_prefill:
        return _prefill(q, k, v, layer_idx, meta)
    return _decode(q, k, v, layer_idx, meta)


def _prefill(q, k, v, layer_idx, meta):
    meta.kv_cache.store(layer_idx, k, v, meta.slot_mapping)
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=meta.cu_seqlens,
        cu_seqlens_k=meta.cu_seqlens,
        max_seqlen_q=meta.max_seqlen,
        max_seqlen_k=meta.max_seqlen,
        causal=True,
    )


def _decode(q, k, v, layer_idx, meta):
    k_cache, v_cache = meta.kv_cache.layer(layer_idx)
    out = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        k.unsqueeze(1),
        v.unsqueeze(1),
        cache_seqlens=meta.cache_seqlens,
        block_table=meta.block_tables,
        causal=True,
    )
    return out.squeeze(1)
