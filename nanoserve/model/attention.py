"""The serving attention path: flash-attn over a paged KV cache.

Attention is the only place in the model that a serving engine changes.
Everything else (norms, RoPE, MLP, residuals) is per-token math and does not
care that several sequences share the step. Attention does: it has to know
where one sequence's tokens end and the next one's begin, and where the KV
of tokens computed on earlier steps lives.

Two kernels cover the two kinds of step:

- Prefill: flash_attn_varlen_func. Our activations are already flat
  [num_tokens, ...], which is exactly what varlen wants; cu_seqlens (the
  cumulative token counts, [0, len_a, len_a+len_b, ...]) tells the kernel
  the boundaries so causal masking is applied per sequence rather than
  across the whole flat batch. K/V are also scattered into the paged cache
  here, because the decode steps that follow will read them from there.

- Decode: flash_attn_with_kvcache with a block table. One query token per
  sequence attends over that sequence's whole cached context, gathered
  through its blocks. The same kernel call also writes the new token's K/V
  into the cache at index cache_seqlens, so decode never needs a separate
  scatter.

GQA is handed to the kernel as-is: flash-attn accepts num_heads_q being a
multiple of num_heads_k and broadcasts internally. The repeat_interleave
the reference SDPA path needs (see model.py) would be pure waste here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..kv_cache import KVCache

try:  # only the paged path needs flash-attn, and it is CUDA-only
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:  # pragma: no cover - exercised by the import error below
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

_MISSING = (
    "flash-attn is required for the paged attention path. It ships as "
    "CUDA-only prebuilt wheels; see docs/gpu-setup.md. The reference SDPA "
    "path (attn_metadata=None) runs without it."
)


@dataclass(frozen=True)
class AttentionMetadata:
    """Everything attention needs about the step that the layers themselves
    cannot know. Built once per forward by ModelRunner and passed down
    unchanged; every layer reads the same object.
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
    # Write first, then attend. The order does not matter for correctness
    # (varlen reads the tensors we pass, not the cache), but writing here
    # keeps "KV for computed tokens is in the cache" true on every exit path.
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
    # The kernel is written for [batch, seqlen, heads, dim]; a decode step is
    # exactly one query token per sequence, so seqlen is 1 and the flat batch
    # of tokens IS the batch of sequences.
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
