import pytest
import torch
from safetensors.torch import save_file

from nanoserve.model.config import ModelConfig
from nanoserve.model.model import NanoForCausalLM
from nanoserve.model.weights import load_weights


def tiny_config(tie_word_embeddings=False):
    return ModelConfig(
        architecture="LlamaForCausalLM",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=False,
        qk_norm=False,
        dtype=torch.float32,
    )


def save_checkpoint(path, state, shards=1):
    if shards == 1:
        save_file(state, str(path / "model.safetensors"))
        return
    keys = sorted(state)
    half = len(keys) // 2
    save_file({k: state[k] for k in keys[:half]}, str(path / "model-00001.safetensors"))
    save_file({k: state[k] for k in keys[half:]}, str(path / "model-00002.safetensors"))


def test_happy_path_loads_exact_values(tmp_path):
    torch.manual_seed(0)
    donor = NanoForCausalLM(tiny_config())
    save_checkpoint(tmp_path, donor.state_dict())

    torch.manual_seed(1)
    model = NanoForCausalLM(tiny_config())
    load_weights(model, tmp_path)

    for key, value in donor.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)


def test_multi_shard_checkpoint_loads(tmp_path):
    donor = NanoForCausalLM(tiny_config())
    save_checkpoint(tmp_path, donor.state_dict(), shards=2)
    model = NanoForCausalLM(tiny_config())
    load_weights(model, tmp_path)
    assert torch.equal(
        model.state_dict()["model.embed_tokens.weight"],
        donor.state_dict()["model.embed_tokens.weight"],
    )


def test_duplicate_key_across_shards_raises(tmp_path):
    state = NanoForCausalLM(tiny_config()).state_dict()
    dup = "model.norm.weight"
    save_file(state, str(tmp_path / "a.safetensors"))
    save_file({dup: state[dup]}, str(tmp_path / "b.safetensors"))
    with pytest.raises(ValueError, match="duplicate key"):
        load_weights(NanoForCausalLM(tiny_config()), tmp_path)


def test_missing_key_fails_strict_load(tmp_path):
    state = NanoForCausalLM(tiny_config()).state_dict()
    del state["model.norm.weight"]
    save_checkpoint(tmp_path, state)
    with pytest.raises(RuntimeError):
        load_weights(NanoForCausalLM(tiny_config()), tmp_path)


def test_unexpected_key_fails_strict_load(tmp_path):
    state = NanoForCausalLM(tiny_config()).state_dict()
    state["bogus.weight"] = torch.zeros(4)
    save_checkpoint(tmp_path, state)
    with pytest.raises(RuntimeError):
        load_weights(NanoForCausalLM(tiny_config()), tmp_path)


def test_tied_checkpoint_with_equal_lm_head_is_dropped(tmp_path):
    state = NanoForCausalLM(tiny_config(tie_word_embeddings=True)).state_dict()
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    save_checkpoint(tmp_path, state)
    model = NanoForCausalLM(tiny_config(tie_word_embeddings=True))
    load_weights(model, tmp_path)


def test_tied_checkpoint_with_different_lm_head_raises(tmp_path):
    state = NanoForCausalLM(tiny_config(tie_word_embeddings=True)).state_dict()
    state["lm_head.weight"] = torch.randn_like(state["model.embed_tokens.weight"])
    save_checkpoint(tmp_path, state)
    with pytest.raises(ValueError, match="refusing to guess"):
        load_weights(NanoForCausalLM(tiny_config(tie_word_embeddings=True)), tmp_path)
