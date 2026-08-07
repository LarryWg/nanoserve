"""RMSNorm and RoPE must match transformers exactly.

We compare against the real transformers implementations rather than a
hand-written reference, because the thing being tested IS whether our reading
of HF's conventions is correct. A reference we wrote ourselves would encode
the same misunderstanding on both sides and pass.

Layout bridge: nanoserve is flat [num_tokens, heads, head_dim]; HF is
[batch, heads, seq, head_dim]. All comparisons use batch=1.
"""
import pytest
import torch

from nanoserve.model.layers import RMSNorm, RotaryEmbedding, apply_rope

transformers = pytest.importorskip("transformers")
from transformers.models.llama.modeling_llama import (  # noqa: E402
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 128, 8, 2, 16
EPS, THETA, MAX_POS = 1e-6, 10000.0, 512


def _flat_to_hf(x: torch.Tensor) -> torch.Tensor:
    """[T, H, D] -> [1, H, T, D]"""
    return x.transpose(0, 1).unsqueeze(0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_hf(dtype):
    torch.manual_seed(0)
    ours = RMSNorm(HIDDEN, EPS)
    theirs = LlamaRMSNorm(HIDDEN, EPS)
    w = torch.randn(HIDDEN)
    with torch.no_grad():
        ours.weight.copy_(w)
        theirs.weight.copy_(w)

    x = torch.randn(7, HIDDEN, dtype=dtype)
    assert torch.equal(ours(x), theirs(x))


def test_rmsnorm_upcasts_to_fp32():
    """A bf16-native implementation drifts here; ours must not."""
    torch.manual_seed(0)
    ours = RMSNorm(HIDDEN, EPS)
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)

    exact = ours(x.float()).to(torch.bfloat16)
    assert torch.allclose(ours(x).float(), exact.float(), atol=1e-2)


def _hf_rotary(seq_len):
    cfg = transformers.LlamaConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=THETA,
        max_position_embeddings=MAX_POS,
    )
    rope = LlamaRotaryEmbedding(config=cfg)
    positions = torch.arange(seq_len).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, HIDDEN)
    cos, sin = rope(dummy, positions)
    return cos, sin


def test_rope_table_matches_hf():
    seq_len = 32
    hf_cos, hf_sin = _hf_rotary(seq_len)
    ours = RotaryEmbedding(HEAD_DIM, THETA, MAX_POS)
    cos, sin = ours(torch.arange(seq_len))

    assert torch.allclose(cos, hf_cos[0], atol=1e-6)
    assert torch.allclose(sin, hf_sin[0], atol=1e-6)


def test_apply_rope_matches_hf():
    torch.manual_seed(0)
    seq_len = 32
    q = torch.randn(seq_len, HEADS, HEAD_DIM)
    k = torch.randn(seq_len, KV_HEADS, HEAD_DIM)

    rope = RotaryEmbedding(HEAD_DIM, THETA, MAX_POS)
    cos, sin = rope(torch.arange(seq_len))
    q_out, k_out = apply_rope(q, k, cos, sin)

    hf_cos, hf_sin = _hf_rotary(seq_len)
    hf_q, hf_k = apply_rotary_pos_emb(
        _flat_to_hf(q), _flat_to_hf(k), hf_cos, hf_sin, unsqueeze_dim=1
    )

    assert torch.allclose(_flat_to_hf(q_out), hf_q, atol=1e-6)
    assert torch.allclose(_flat_to_hf(k_out), hf_k, atol=1e-6)


def test_apply_rope_handles_noncontiguous_positions():
    """The serving case: one flat batch holding tokens at unrelated positions.

    Sequence A decoding at position 500 and sequence B prefilling at 0..3 share
    a step. Rotating token i by position i would be correct for a rectangular
    batch and wrong for every decode step here.
    """
    torch.manual_seed(0)
    positions = torch.tensor([500, 0, 1, 2, 3])
    q = torch.randn(len(positions), HEADS, HEAD_DIM)
    k = torch.randn(len(positions), KV_HEADS, HEAD_DIM)

    rope = RotaryEmbedding(HEAD_DIM, THETA, MAX_POS)
    cos, sin = rope(positions)
    q_out, _ = apply_rope(q, k, cos, sin)

    # Each row must equal that row rotated alone at its own position.
    for i, pos in enumerate(positions.tolist()):
        c, s = rope(torch.tensor([pos]))
        single, _ = apply_rope(q[i : i + 1], k[i : i + 1], c, s)
        assert torch.allclose(q_out[i], single[0], atol=1e-6)


def test_rope_is_halves_not_interleaved():
    """Pin the convention that silently produces fluent-but-wrong output.

    At position 0 rotation is identity, so use position 1 and check that
    dimension j mixes with j + head_dim/2 rather than with its neighbour.
    """
    rope = RotaryEmbedding(HEAD_DIM, THETA, MAX_POS)
    cos, sin = rope(torch.tensor([1]))
    half = HEAD_DIM // 2

    q = torch.zeros(1, 1, HEAD_DIM)
    q[0, 0, 0] = 1.0                     # excite dim 0 only
    out, _ = apply_rope(q, q.clone(), cos, sin)

    assert out[0, 0, 0] == pytest.approx(cos[0, 0].item())
    assert out[0, 0, half] == pytest.approx(sin[0, half].item())   # partner dim
    assert out[0, 0, 1] == pytest.approx(0.0)                      # neighbour untouched
