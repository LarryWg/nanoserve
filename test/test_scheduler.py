import pytest

from nanoserve.block_manager import BlockManager
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import SamplingParams, SeqStatus, Sequence

BS = 4


def make_scheduler(num_blocks=8, block_size=BS, **kwargs):
    defaults = dict(max_num_seqs=64, max_num_batched_tokens=1024)
    return Scheduler(
        block_manager=BlockManager(num_blocks=num_blocks, block_size=block_size),
        **(defaults | kwargs),
    )


def make_seq(seq_id, prompt_len=BS):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling=SamplingParams(),
    )


def run_step(sched, batch):
    for seq in batch.seqs:
        if batch.is_prefill:
            seq.on_prefilled()
        seq.on_token(token_id=0, now=0.0)


def test_first_step_is_a_prefill_batch():
    sched = make_scheduler()
    sched.add_request(make_seq(1))
    sched.add_request(make_seq(2))
    batch = sched.step()
    assert batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [1, 2]
    assert len(sched.waiting) == 0


def test_token_budget_limits_prefill_admission():
    sched = make_scheduler(max_num_batched_tokens=BS)
    sched.add_request(make_seq(1))
    sched.add_request(make_seq(2))
    batch = sched.step()
    assert [s.seq_id for s in batch.seqs] == [1]
    assert len(sched.waiting) == 1


def test_max_num_seqs_limits_prefill_admission():
    sched = make_scheduler(max_num_seqs=1)
    sched.add_request(make_seq(1))
    sched.add_request(make_seq(2))
    assert len(sched.step().seqs) == 1


def test_block_shortage_limits_prefill_admission():
    sched = make_scheduler(num_blocks=1)
    sched.add_request(make_seq(1))
    sched.add_request(make_seq(2))
    assert [s.seq_id for s in sched.step().seqs] == [1]
    assert len(sched.waiting) == 1


def test_head_of_line_blocking_is_strict():
    sched = make_scheduler(max_num_batched_tokens=6)
    sched.add_request(make_seq(1, prompt_len=5))
    sched.add_request(make_seq(2, prompt_len=1))
    assert [s.seq_id for s in sched.step().seqs] == [1, 2]

    sched = make_scheduler(max_num_batched_tokens=5)
    sched.add_request(make_seq(1, prompt_len=5))
    sched.add_request(make_seq(2, prompt_len=1))
    assert [s.seq_id for s in sched.step().seqs] == [1]


def test_oversized_prompt_rejected_at_admission():
    sched = make_scheduler(max_num_batched_tokens=8)
    with pytest.raises(ValueError, match="budget"):
        sched.add_request(make_seq(1, prompt_len=9))


def test_prompt_larger_than_whole_cache_rejected():
    sched = make_scheduler(num_blocks=2)
    with pytest.raises(ValueError, match="KV blocks"):
        sched.add_request(make_seq(1, prompt_len=2 * BS + 1))


def test_decode_follows_prefill():
    sched = make_scheduler()
    sched.add_request(make_seq(1))
    run_step(sched, sched.step())
    batch = sched.step()
    assert not batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [1]


def test_prefill_first_runs_new_requests_before_decodes():
    sched = make_scheduler(prefill_first=True)
    sched.add_request(make_seq(1))
    run_step(sched, sched.step())
    sched.add_request(make_seq(2))
    batch = sched.step()
    assert batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [2]


def test_decode_first_keeps_running_stream_steady():
    sched = make_scheduler(prefill_first=False)
    sched.add_request(make_seq(1))
    run_step(sched, sched.step())
    sched.add_request(make_seq(2))
    batch = sched.step()
    assert not batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [1]


def test_decode_first_still_prefills_when_nothing_runs():
    sched = make_scheduler(prefill_first=False)
    sched.add_request(make_seq(1))
    assert sched.step().is_prefill


def test_step_returns_none_when_idle():
    assert make_scheduler().step() is None


def test_prefill_waits_when_blocks_short_but_decode_continues():
    sched = make_scheduler(num_blocks=3)
    sched.add_request(make_seq(1, prompt_len=BS))
    sched.add_request(make_seq(2, prompt_len=2 * BS + 1))
    run_step(sched, sched.step())
    batch = sched.step()
    assert not batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [1]
    assert sched.waiting[0].seq_id == 2


def test_decode_preemption_evicts_youngest_and_requeues():
    sched = make_scheduler(num_blocks=3)
    sched.add_request(make_seq(1, prompt_len=BS + 1))
    sched.add_request(make_seq(2, prompt_len=BS))
    run_step(sched, sched.step())
    assert sched.block_manager.num_free_blocks() == 0

    batch = sched.step()
    assert not batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [1]

    victim = sched.waiting[0]
    assert victim.seq_id == 2
    assert victim.status is SeqStatus.PREEMPTED
    assert victim.num_computed_tokens == 0
    assert victim.num_tokens == BS + 1


def test_preempted_sequence_reprefills_its_whole_context():
    sched = make_scheduler(num_blocks=3)
    sched.add_request(make_seq(1, prompt_len=BS + 1))
    sched.add_request(make_seq(2, prompt_len=BS))
    run_step(sched, sched.step())
    sched.step()
    sched.finish(sched.running[0], now=1.0)

    batch = sched.step()
    assert batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [2]
    assert sched.block_manager.num_free_blocks() == 1


def test_preemption_goes_to_front_of_waiting_queue():
    sched = make_scheduler(num_blocks=3)
    sched.add_request(make_seq(1, prompt_len=BS + 1))
    sched.add_request(make_seq(2, prompt_len=BS))
    run_step(sched, sched.step())
    sched.add_request(make_seq(3))
    sched.step()
    assert [s.seq_id for s in sched.waiting] == [2, 3]


def test_finish_frees_blocks_for_new_requests():
    sched = make_scheduler(num_blocks=2)
    sched.add_request(make_seq(1, prompt_len=BS))
    run_step(sched, sched.step())
    sched.add_request(make_seq(2, prompt_len=2 * BS))
    assert not sched.step().is_prefill

    sched.finish(sched.running[0], now=2.0)
    batch = sched.step()
    assert batch.is_prefill
    assert [s.seq_id for s in batch.seqs] == [2]


def test_finish_marks_sequence():
    sched = make_scheduler()
    sched.add_request(make_seq(1))
    run_step(sched, sched.step())
    seq = sched.running[0]
    sched.finish(seq, now=7.0)
    assert seq.status is SeqStatus.FINISHED
    assert seq.finish_time == 7.0
    assert sched.running == []
    assert sched.block_manager.num_free_blocks() == 8


def test_abort_removes_a_waiting_sequence_without_touching_blocks():
    sched = make_scheduler(max_num_seqs=1)
    sched.add_request(make_seq(1))
    sched.add_request(make_seq(2))
    run_step(sched, sched.step())
    free_before = sched.block_manager.num_free_blocks()

    sched.abort(2, now=5.0)

    assert list(sched.waiting) == []
    assert [s.seq_id for s in sched.running] == [1]
    assert sched.block_manager.num_free_blocks() == free_before


def test_abort_frees_a_running_sequence():
    sched = make_scheduler()
    sched.add_request(make_seq(1))
    run_step(sched, sched.step())
    seq = sched.running[0]

    sched.abort(1, now=5.0)

    assert sched.running == []
    assert seq.status is SeqStatus.FINISHED
    assert seq.finish_time == 5.0
    assert sched.block_manager.num_free_blocks() == 8


def test_abort_of_a_preempted_sequence_does_not_double_free():
    sched = make_scheduler(num_blocks=2)
    sched.add_request(make_seq(1, prompt_len=BS))
    sched.add_request(make_seq(2, prompt_len=BS))
    run_step(sched, sched.step())
    sched.step()
    victim = sched.waiting[0]
    assert victim.status is SeqStatus.PREEMPTED

    sched.abort(victim.seq_id, now=3.0)

    assert list(sched.waiting) == []
    assert victim.status is SeqStatus.FINISHED
    assert victim.finish_time == 3.0


def test_abort_of_an_unknown_sequence_is_ignored():
    sched = make_scheduler()
    sched.add_request(make_seq(1))
    run_step(sched, sched.step())

    sched.abort(999, now=1.0)

    assert [s.seq_id for s in sched.running] == [1]
    assert sched.block_manager.num_free_blocks() == 7
