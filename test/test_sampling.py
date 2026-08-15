"""Sampling tests: how a token gets picked, on CPU.

sample_tokens is a plain function over tensors rather than a method, so it
can be tested without a GPU, a checkpoint, or a cache. Two properties
matter: a temperature of 0 is exactly greedy, and top_p never lets a token
outside the nucleus be drawn.
"""
import torch

from nanoserve.model_runner import _top_p_filter, sample_tokens


def probs(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def assert_rows(out: torch.Tensor, *expected) -> None:
    # allclose, not equals. 0.7 has no exact fp32 form, and the filter does
    # arithmetic on the way out.
    assert torch.allclose(out, probs(*expected), atol=1e-6)


# ---------- top p filtering ----------

def test_top_p_keeps_the_token_that_crosses_the_threshold():
    # Masses 0.5, 0.3, 0.2. At p=0.7 the nucleus is 0.5 and 0.3. The second
    # token crosses the line and is kept.
    out = _top_p_filter(probs([0.5, 0.3, 0.2]), torch.tensor([0.7]))
    assert_rows(out, [0.5, 0.3, 0.0])


def test_top_p_keeps_the_single_dominant_token():
    # The classic off by one. Counting each token's own mass would zero the
    # whole row here, since 0.9 already passes 0.5.
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
    # The nucleus follows probability, not token id. Here the mass sits at
    # the end of the row.
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
    # Riding through the softmax next to a sampled row must not disturb the
    # greedy one.
    logits = torch.tensor([[1.0, 5.0, 2.0], [0.0, 0.0, 8.0]])
    ids = sample_tokens(logits, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1e-6]))
    assert ids.tolist() == [1, 2]


def test_sampling_stays_inside_the_nucleus():
    torch.manual_seed(0)
    logits = torch.tensor([[3.0, 2.0, -5.0, -6.0]])
    # top_p of 0.9 covers the two likely tokens. The tail must never show up.
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
