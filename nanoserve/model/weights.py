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
                if key in state:
                    raise ValueError(f"duplicate key {key!r} across shards")
                state[key] = f.get_tensor(key)

    config = model.config
    state = {k: v.to(config.dtype) for k, v in state.items()}

    if config.tie_word_embeddings and "lm_head.weight" in state:
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError(
                "config says tie_word_embeddings but lm_head.weight differs "
                "from embed_tokens.weight; refusing to guess"
            )
        del state["lm_head.weight"]

    model.load_state_dict(state, strict=True)
