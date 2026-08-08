"""HF config.json -> the handful of numbers the model actually needs.

Reading the config ourselves (rather than holding a transformers config object)
keeps the non-goals fence visible: an architecture we have not verified against
HF is rejected at load time, not silently mis-executed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

# Dense decoder-only models we have checked token-for-token against HF.
# Anything else fails loudly (see DESIGN.md non-goals).
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
}

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool   # Qwen2 carries a bias on q/k/v_proj; Llama does not
    qk_norm: bool          # Qwen3 RMSNorms q and k per head before RoPE
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({self.num_key_value_heads})"
            )

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head (GQA repeat factor). 1 == MHA."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(cls, path: str | Path) -> ModelConfig:
        raw = json.loads((Path(path) / "config.json").read_text())

        arch = (raw.get("architectures") or ["<missing>"])[0]
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"unsupported architecture {arch!r}; nanoserve supports "
                f"{sorted(SUPPORTED_ARCHITECTURES)}"
            )

        num_heads = raw["num_attention_heads"]
        hidden_size = raw["hidden_size"]
        # Qwen3 sets head_dim explicitly and it is NOT hidden_size // num_heads
        # (Qwen3-0.6B: 1024 hidden, 16 heads, head_dim 128). Never derive it
        # when the config states it.
        head_dim = raw.get("head_dim") or hidden_size // num_heads

        # Llama-3.1/3.2 rescale inv_freq per frequency band ("llama3" rope_type).
        # Ignoring it would still produce fluent-looking text that is silently
        # wrong, so reject rather than approximate.
        if raw.get("rope_scaling"):
            raise ValueError(
                f"rope_scaling={raw['rope_scaling']} is not implemented yet; "
                "use a model with plain RoPE (e.g. Qwen3, Qwen2.5, Llama-2)"
            )

        # Qwen2 configs may enable sliding-window attention on some layers,
        # where HF attends only over the last sliding_window tokens. We do
        # not implement that, so reject instead of computing full attention.
        if raw.get("use_sliding_window"):
            raise ValueError(
                "sliding-window attention is not implemented; "
                "use a checkpoint with use_sliding_window=false"
            )

        # HF treats a missing dtype field as float32. We refuse to guess
        # instead: a wrong guess would silently cast every weight.
        dtype_name = raw.get("torch_dtype") or raw.get("dtype")
        if dtype_name is None:
            raise ValueError(
                "config.json has no torch_dtype field; refusing to guess"
            )
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r}")

        return cls(
            architecture=arch,
            hidden_size=hidden_size,
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=num_heads,
            num_key_value_heads=raw.get("num_key_value_heads", num_heads),
            head_dim=head_dim,
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 10000.0),
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", arch == "Qwen2ForCausalLM"),
            qk_norm=arch == "Qwen3ForCausalLM",
            dtype=_DTYPES[dtype_name],
        )
