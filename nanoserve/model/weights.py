"""Safetensors -> NanoForCausalLM weight loading.

Because the module tree uses HF's exact parameter names (see model.py),
loading is a strict load_state_dict over the concatenation of every shard
-- no handwritten key mapping. The strictness is the point: a checkpoint
key we don't consume, or a parameter the checkpoint doesn't provide, raises
instead of leaving a silently-initialized (i.e. random) tensor in the model.
The HF equivalence tests compare token-for-token; a silent partial load is
how those tests lie.

Tied embeddings: checkpoints with tie_word_embeddings=true typically omit
lm_head.weight entirely. Our module has no lm_head in that case (the head
reads through embed_tokens), so nothing special happens at load time --
the missing key is expected, not an error.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open


def load_weights(model: torch.nn.Module, model_path: str | Path) -> None:
    model_path = Path(model_path)
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files under {model_path}")

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                # Duplicate keys across shards would be a packaging bug in
                # the checkpoint; overwriting silently could pick a stale copy.
                if key in state:
                    raise ValueError(f"duplicate key {key!r} across shards")
                state[key] = f.get_tensor(key)

    config = model.config
    # load_state_dict copy_()s into the existing parameters, which were
    # created in config.dtype (see NanoForCausalLM.from_pretrained), so the
    # cast is a no-op safety net rather than the mechanism.
    state = {k: v.to(config.dtype) for k, v in state.items()}

    if config.tie_word_embeddings and "lm_head.weight" in state:
        # The module has no lm_head to receive it; strict load would fail on
        # the unexpected key. The tie makes it redundant, so verify it is
        # actually the embedding matrix before dropping it.
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError(
                "config says tie_word_embeddings but lm_head.weight differs "
                "from embed_tokens.weight; refusing to guess"
            )
        del state["lm_head.weight"]

    model.load_state_dict(state, strict=True)
