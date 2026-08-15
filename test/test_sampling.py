"""Sampling tests: the token-picking math, on CPU.

sample_tokens is deliberately a plain function over tensors rather than a
ModelRunner method, so the distribution logic can be tested without a GPU,
a checkpoint, or a KV cache. What is checked here is the two properties a
serving engine actually depends on: temperature 0 is exactly greedy, and
top_p never lets a token outside the nucleus be drawn.
"""
import torch

from nanoserve.model_runner import _top_p_filter, sample_tokens


def probs(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def assert_rows(out: torch.Tensor, *expected) -> None:
    # allclose, not ==: 0.7 is not exactly representable in fp32, and the
    # filter passes probabilities through arithmetic before returning them.
    assert torch.allclose(out, probs(*expected), atol=1e-6)


# ---------- top-p filtering ----------

def test_top_p_keeps_the_token_that_crosses_the_threshold():
    # Masses 0.5, 0.3, 0.2. With p=0.7 the nucleus is {0.5, 0.3}: the second
    # token crosses the line and is kept, not dropped.
    out = _top_p_filter(probs([0.5, 0.3, 0.2]), torch.tensor([0.7]))
    assert_rows(out, [0.5, 0.3, 0.0])


def test_top_p_keeps_the_single_dominant_token():
    # The classic off-by-one: comparing the cumulative mass INCLUDING each
    # token would zero this whole row, since 0.9 >= 0.5 already.
    out = _top_p_filter(probs([0.9, 0.05, 0.05]), torch.tensor([0.5]))
    assert_rows(out, [0.9, 0.0, 0.0])


def test_top_p_of_one_keeps_everything():
    row = probs([0.5, 0.3, 0.2])
    assert torch.equal(_top_p_filter(row, torch.tensor([1.0])), row)


def test_top_p_is_per_row():
    out = _top_p_filter(probs([0.5, 0.3, 0.2], [0.5, 0.3, 0.2]),
                        torch.tensor([0.4, 1.0]))
    assert_rows(out, [0.5, 0.0, 0.0], [0.5, 0.3, 0.2])


def test_top_p_ignores_position_and_follows_probability():
    # The nucleus is defined by rank, not by token id: the mass is at the
    # end of this row.
    out = _top_p_filter(probs([0.1, 0.2, 0.7]), torch.tensor([0.6]))
    assert_rows(out, [0.0, 0.0, 0.7])


# ---------- sampling ----------

def test_temperature_zero_is_exactly_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]])
    ids = sample_tokens(logits, torch.zeros(2), torch.ones(2))
    assert ids.tolist() == [1, 0]


def test_a_one_token_nucleus_is_greedy_too():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    ids = sample_tokens(logits, torch.tensor([1.0]), torch.tensor([1e-6]))
    assert ids.tolist() == [1]


def test_greedy_and_sampling_rows_share_one_batch():
    # The greedy row must not be perturbed by riding through the softmax
    # with the sampled one.
    logits = torch.tensor([[1.0, 5.0, 2.0], [0.0, 0.0, 8.0]])
    ids = sample_tokens(logits, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1e-6]))
    assert ids.tolist() == [1, 2]


def test_sampling_stays_inside_the_nucleus():
    torch.manual_seed(0)
    logits = torch.tensor([[3.0, 2.0, -5.0, -6.0]])
    # top_p=0.9 covers the two plausible tokens; the tail must never appear.
    draws = {
        int(sample_tokens(logits, torch.tensor([1.0]), torch.tensor([0.9]))[0])
        for _ in range(200)
    }
    assert draws == {0, 1}


def test_temperature_widens_the_distribution():
    torch.manual_seed(0)
    logits = torch.tensor([[3.0, 2.0]])
    cold = [int(sample_tokens(logits, torch.tensor([0.1]), torch.ones(1))[0])
            for _ in range(200)]
    hot = [int(sample_tokens(logits, torch.tensor([5.0]), torch.ones(1))[0])
           for _ in range(200)]
    assert sum(cold) < sum(hot)   # hotter picks the weaker token more often
