"""ModelConfig.from_pretrained against synthetic config.json files.

The HF equivalence tests only exercise two real checkpoints, so the
reject-don't-guess rules live here instead: each test writes a tiny
config.json and checks the parse.
"""
import json

import pytest
import torch

from nanoserve.model.config import ModelConfig


def write_config(tmp_path, **overrides):
    """A minimal valid Llama-style config; overrides tweak one field each."""
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 100,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "torch_dtype": "bfloat16",
    }
    raw.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return ModelConfig.from_pretrained(tmp_path)


def test_minimal_config_parses(tmp_path):
    cfg = write_config(tmp_path)
    assert cfg.hidden_size == 64
    assert cfg.head_dim == 16                     # derived: hidden // heads
    assert cfg.num_kv_groups == 2
    assert cfg.dtype is torch.bfloat16
    assert cfg.attention_bias is False            # Llama default
    assert cfg.tie_word_embeddings is False


def test_missing_dtype_raises(tmp_path):
    # HF would silently assume fp32; guessing either way changes numerics.
    with pytest.raises(ValueError, match="torch_dtype"):
        write_config(tmp_path, torch_dtype=None)


def test_unknown_dtype_raises(tmp_path):
    with pytest.raises(ValueError, match="unsupported dtype"):
        write_config(tmp_path, torch_dtype="float8")


def test_sliding_window_rejected(tmp_path):
    # Qwen2 variants can set this; we compute full attention, so refuse.
    with pytest.raises(ValueError, match="sliding-window"):
        write_config(
            tmp_path,
            architectures=["Qwen2ForCausalLM"],
            use_sliding_window=True,
            sliding_window=4096,
        )


def test_sliding_window_field_present_but_disabled_is_fine(tmp_path):
    # Real Qwen2 checkpoints ship these fields even when disabled.
    cfg = write_config(
        tmp_path,
        architectures=["Qwen2ForCausalLM"],
        use_sliding_window=False,
        sliding_window=32768,
    )
    assert cfg.attention_bias is True             # Qwen2 default


def test_explicit_head_dim_is_honored(tmp_path):
    # Qwen3-0.6B: hidden 1024, 16 heads, but head_dim 128 (not 64).
    cfg = write_config(
        tmp_path,
        architectures=["Qwen3ForCausalLM"],
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    assert cfg.head_dim == 128
    assert cfg.qk_norm is True                    # Qwen3 marker


def test_unsupported_architecture_raises(tmp_path):
    with pytest.raises(ValueError, match="unsupported architecture"):
        write_config(tmp_path, architectures=["MistralForCausalLM"])


def test_rope_scaling_rejected(tmp_path):
    # Llama-3 style frequency rescaling would be silently wrong if ignored.
    with pytest.raises(ValueError, match="rope_scaling"):
        write_config(tmp_path, rope_scaling={"rope_type": "llama3"})


def test_heads_not_divisible_by_kv_heads_raises(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        write_config(tmp_path, num_attention_heads=5, num_key_value_heads=2)
