"""Sequence lifecycle pins.

The scheduler (not written yet) drives sequences through these transitions,
so they are pinned here against the state machine the docstring promises:
WAITING -> RUNNING -> (PREEMPTED -> WAITING) -> FINISHED.
"""
import pytest

from nanoserve.sequence import SamplingParams, SeqStatus, Sequence


def make_seq(**kwargs):
    defaults = dict(seq_id=1, prompt_token_ids=[10, 11, 12],
                    sampling=SamplingParams())
    return Sequence(**(defaults | kwargs))


def test_empty_prompt_rejected():
    with pytest.raises(ValueError, match="prompt_token_ids"):
        make_seq(prompt_token_ids=[])


def test_prefill_then_decode_transitions():
    seq = make_seq()
    assert seq.status is SeqStatus.WAITING
    assert seq.is_prefill

    seq.on_prefilled()
    assert seq.status is SeqStatus.RUNNING
    assert not seq.is_prefill
    assert seq.num_computed_tokens == 3

    seq.on_token(token_id=42, now=1.0)
    assert seq.output_token_ids == [42]
    assert seq.num_computed_tokens == 4
    assert seq.last_token == 42


def test_first_token_time_stamped_once():
    seq = make_seq()
    seq.on_prefilled()
    seq.on_token(token_id=1, now=5.0)
    seq.on_token(token_id=2, now=9.0)
    assert seq.first_token_time == 5.0


def test_preemption_keeps_tokens_but_forgets_kv():
    """Recompute preemption throws KV away but keeps the generated tokens,
    so a preempted sequence prefills again (prompt + outputs) on readmission."""
    seq = make_seq()
    seq.on_prefilled()
    seq.on_token(token_id=42, now=1.0)

    seq.on_preempted()
    assert seq.status is SeqStatus.PREEMPTED
    assert seq.num_computed_tokens == 0
    assert seq.is_prefill                       # must recompute on readmission
    assert seq.token_ids == [10, 11, 12, 42]    # tokens survive


def test_is_stopped_on_max_new_tokens():
    seq = make_seq(sampling=SamplingParams(max_new_tokens=2))
    seq.on_prefilled()
    seq.on_token(1, 0.0)
    assert not seq.is_stopped
    seq.on_token(2, 0.0)
    assert seq.is_stopped


def test_is_stopped_on_stop_token():
    seq = make_seq(sampling=SamplingParams(stop_token_ids=(99,)))
    seq.on_prefilled()
    seq.on_token(1, 0.0)
    assert not seq.is_stopped
    seq.on_token(99, 0.0)
    assert seq.is_stopped


def test_on_finished():
    seq = make_seq()
    seq.on_prefilled()
    seq.on_finished(now=3.0)
    assert seq.status is SeqStatus.FINISHED
    assert seq.finish_time == 3.0
