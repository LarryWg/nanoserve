from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .attention import AttentionMetadata, paged_attention
from .config import ModelConfig
from .layers import RMSNorm, RotaryEmbedding, apply_rope


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)

        if attn_metadata is None:
            out = self._sdpa(q, k, v)
        else:
            out = paged_attention(q, k, v, self.layer_idx, attn_metadata)
        return self.o_proj(out.reshape(T, self.num_heads * self.head_dim))

    def _sdpa(self, q, k, v) -> torch.Tensor:
        q = q.transpose(0, 1)
        k = k.transpose(0, 1).repeat_interleave(self.num_kv_groups, dim=0)
        v = v.transpose(0, 1).repeat_interleave(self.num_kv_groups, dim=0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(0, 1)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        h, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward(self, x, cos, sin, attn_metadata=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_metadata)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class NanoModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )

    def forward(self, input_ids, positions, attn_metadata=None):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_metadata)
        return self.norm(x)


class NanoForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = NanoModel(config)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                     bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, positions, attn_metadata)
        if logits_indices is not None:
            hidden = hidden[logits_indices]
        weight = (self.model.embed_tokens.weight if self.lm_head is None
                  else self.lm_head.weight)
        return F.linear(hidden, weight)

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu") -> "NanoForCausalLM":
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"device={device!r} but CUDA is not available")
        from .weights import load_weights
        config = ModelConfig.from_pretrained(model_path)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        try:
            model = cls(config)
        finally:
            torch.set_default_dtype(prev)
        model = model.to(device)
        load_weights(model, model_path)
        return model
