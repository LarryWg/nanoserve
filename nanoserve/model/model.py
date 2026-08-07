"""The model: a dense Llama/Qwen decoder, written to be read.

Our own nn.Module, verified to match transformers token-for-token. It is
deliberately NOT a serving model yet -- there is no KV cache and no
batching, so generation recomputes the full context every step. That is
slow and completely fine at test scale (~100 tokens); the attention call
is the one place that changes when a paged KV cache and varlen kernels
arrive, and everything above and below that call stays as written here.

Two structural choices, made once and held everywhere:

1. FLAT layout. Activations are [num_tokens, ...] with no batch dim, and
   positions arrive as an explicit int tensor (see layers.py for why).
   Only one sequence is ever run here, so positions are arange(T) -- but
   the forward signature is already the one a batched engine will use.

2. HF-faithful names. Submodules are named exactly as transformers names
   them (model.layers.{i}.self_attn.q_proj.weight, ...). That makes weight
   loading a strict load_state_dict instead of a handwritten key mapping --
   a mapping we wrote ourselves could encode the same misunderstanding as
   the model and silently "pass". Strictness is the test's best friend.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, RotaryEmbedding, apply_rope


class Attention(nn.Module):
    """GQA self-attention over a flat token stream.

    The three architectural ifs in this codebase all live here, driven by
    ModelConfig rather than by arch-name string checks scattered around:
      - attention_bias (Qwen2): q/k/v projections carry a bias; Llama/Qwen3
        do not. o_proj never does, in every arch we support.
      - qk_norm (Qwen3): a per-head RMSNorm over head_dim applied to q and k
        BEFORE RoPE. Weight shape is [head_dim], so the norm normalizes each
        head independently -- not the whole [T, heads*dim] vector.
      - GQA (all supported archs): num_kv_heads < num_attention_heads. SDPA
        wants matching head counts, so k/v are repeated num_kv_groups times.
        The repeat is materialized, not a view trick: a fused kernel can
        replace this whole path later, and clarity beats cleverness until
        then.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
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
        x: torch.Tensor,          # [T, hidden]
        cos: torch.Tensor,        # [T, head_dim]
        sin: torch.Tensor,        # [T, head_dim]
    ) -> torch.Tensor:
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)

        # SDPA wants [heads, T, head_dim]. is_causal=True is exactly the
        # right mask for one sequence with positions 0..T-1; a flat batch
        # packing several sequences would need per-sequence causal masking
        # (e.g. a varlen kernel) instead.
        q = q.transpose(0, 1)
        k = k.transpose(0, 1).repeat_interleave(self.num_kv_groups, dim=0)
        v = v.transpose(0, 1).repeat_interleave(self.num_kv_groups, dim=0)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(0, 1).reshape(T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x)). No biases anywhere."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        h, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: RMSNorm -> attn -> +res -> RMSNorm -> MLP -> +res.

    HF names are kept verbatim (input_layernorm, self_attn,
    post_attention_layernorm, mlp) so checkpoints load strictly.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class NanoModel(nn.Module):
    """The backbone: embedding -> N decoder layers -> final norm.

    Named `model` inside NanoForCausalLM so checkpoint keys line up with
    HF's `model.embed_tokens.weight`, `model.layers.0...`, `model.norm.weight`.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class NanoForCausalLM(nn.Module):
    """Backbone + lm_head. This is the class the HF equivalence tests
    compare against transformers' LlamaForCausalLM / Qwen2ForCausalLM /
    Qwen3ForCausalLM.

    forward takes the flat-layout pair (input_ids, positions) and returns
    logits for EVERY position [T, vocab] -- the caller slices what it needs
    (a single sequence wants only the last position; a batching engine
    wants the last position per sequence).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = NanoModel(config)
        if config.tie_word_embeddings:
            # Qwen2.5-0.5B/1.5B and some Llama checkpoints omit lm_head and
            # read out through the embedding matrix. Sharing the Parameter
            # (not copying it) keeps memory honest and matches HF semantics.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                     bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,     # [T] int64
        positions: torch.Tensor,     # [T] int64
    ) -> torch.Tensor:
        hidden = self.model(input_ids, positions)
        weight = (self.model.embed_tokens.weight if self.lm_head is None
                  else self.lm_head.weight)
        return F.linear(hidden, weight)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "NanoForCausalLM":
        """Build from a local HF checkpoint dir and load its weights.

        Construction happens under set_default_dtype(config.dtype) so
        parameters are born bf16/fp16: load_state_dict copy_()s into the
        existing parameter dtype, and we want the model to hold the
        checkpoint's dtype exactly -- comparing our fp32-held weights
        against HF's bf16 model would pass the atol while testing nothing
        about bf16 serving numerics. The RoPE cos/sin caches are explicitly
        fp32 in layers.py and are unaffected by the default dtype, by design.
        """
        from .weights import load_weights  # here, not top: keeps safetensors
                                           # importable-failure local to loading
        config = ModelConfig.from_pretrained(model_path)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        try:
            model = cls(config)
        finally:
            torch.set_default_dtype(prev)
        load_weights(model, model_path)
        return model
