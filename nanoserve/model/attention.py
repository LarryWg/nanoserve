"""Attention over the paged KV cache, using flash attn.

Attention is the only layer a serving engine has to change. Every other
layer works one token at a time and does not care that several sequences
share the step. Attention cares: it has to know where one sequence stops
and the next starts, and where the keys and values from earlier steps live.

Prefill uses flash_attn_varlen_func. Our activations are already flat, and
cu_seqlens (0, len_a, len_a + len_b, ...) marks the boundaries so each
sequence gets its own causal mask. Keys and values are also written to the
cache here, because the decode steps that follow read them from there.

Decode uses flash_attn_with_kvcache with a block table. One query token per
sequence attends over that sequence's cached context. The same call stores
the new key and value, so decode needs no separate write.

GQA goes straight to the kernel. It accepts more query heads than key heads
and broadcasts them itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..kv_cache import KVCache

try:  # the paged path needs flash attn, which is CUDA only
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
    """What attention needs to know about this step.

    Built once per forward and read unchanged by every layer.
    """

    is_prefill: bool
    kv_cache: KVCache

    # Prefill only.
    slot_mapping: torch.Tensor | None = None   # [num_tokens] int64
    cu_seqlens: torch.Tensor | None = None     # [num_seqs + 1] int32
    max_seqlen: int = 0

    # Decode only.
    block_tables: torch.Tensor | None = None   # [num_seqs, max_blocks] int32
    cache_seqlens: torch.Tensor | None = None  # [num_seqs] int32


def paged_attention(
    q: torch.Tensor,             # [num_tokens, num_heads, head_dim]
    k: torch.Tensor,             # [num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,
    layer_idx: int,
    meta: AttentionMetadata,
) -> torch.Tensor:               # [num_tokens, num_heads, head_dim]
    if flash_attn_varlen_func is None:
        raise ImportError(_MISSING)
    if meta.is_prefill:
        return _prefill(q, k, v, layer_idx, meta)
    return _decode(q, k, v, layer_idx, meta)


def _prefill(q, k, v, layer_idx, meta):
    # Store first, then attend. The result is the same either way, but this
    # order keeps one rule true everywhere: if a token is computed, its keys
    # and values are in the cache.
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
    # The kernel wants [batch, seqlen, heads, dim]. A decode step is one
    # query token per sequence, so seqlen is 1 and our flat batch of tokens
    # is the batch of sequences.
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
