"""Primitives shared by every layer: RMSNorm and RoPE.

Tensor layout convention for the whole engine: activations are FLAT --
[num_tokens, ...] with no batch dimension. Continuous batching packs sequences
of different lengths into one step, so a rectangular [batch, seq] tensor would
have to be padded; flat + cu_seqlens is what flash-attn varlen wants anyway.
Per-token positions are therefore an explicit int tensor, not a range.
"""
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Llama/Qwen RMSNorm.

    The fp32 upcast is not optional: in bf16 the sum of squares over a few
    thousand elements loses enough precision to visibly move the logits.
    HF upcasts, normalizes, casts back, and only then applies the weight.
    We match that order exactly so the comparison is apples to apples.
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair dimension i with i + head_dim/2 (NOT with i+1).

    This is the single most dangerous convention in the model. The RoPE paper
    rotates adjacent pairs (x0,x1), (x2,x3), ...; HF's checkpoint conversion
    permutes q_proj/k_proj weights so that the halves layout (x0,x_{d/2}),
    (x1,x_{1+d/2}), ... is equivalent. Implement the paper's interleaving
    against HF weights and the model still emits fluent, grammatical, wrong
    text. It degrades gracefully instead of crashing, which is exactly why
    the tests compare against HF token-for-token instead of eyeballing a
    sample.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin table, gathered per token position.

    Built once at startup for max_position_embeddings. A serving engine sees
    arbitrary, non-contiguous positions in a single flat batch (seq A at
    position 3, seq B at position 900), so a gather from a table is the
    natural op. Computing freqs per step would be both slower and awkward.
    """

    def __init__(self, head_dim: int, theta: float, max_position: int):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)   # duplicated to match _rotate_half
        # fp32 cache, cast at use: the table is tiny and the angles matter.
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [num_tokens] int -> two [num_tokens, head_dim] tensors."""
        return self.cos_cache[positions], self.sin_cache[positions]


def apply_rope(
    q: torch.Tensor,      # [num_tokens, num_heads, head_dim]
    k: torch.Tensor,      # [num_tokens, num_kv_heads, head_dim]
    cos: torch.Tensor,    # [num_tokens, head_dim]
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1).to(q.dtype)   # broadcast over the head axis
    sin = sin.unsqueeze(1).to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
