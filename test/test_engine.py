import asyncio

import pytest

from nanoserve.block_manager import BlockManager
from nanoserve.engine import Engine, Output
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import SamplingParams, SeqStatus

BLOCK_SIZE = 4
NUM_BLOCKS = 8


class FakeRunner:
    def __init__(self, token: int = 7):
        self.token = token
        self.batches = []

    def forward(self, batch):
        self.batches.append(batch)
        return None

    def sample(self, logits, batch):
        return [self.token] * len(batch.seqs)


def make_engine(num_blocks: int = NUM_BLOCKS, **scheduler_kwargs) -> Engine:
    manager = BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    scheduler = Scheduler(block_manager=manager, **scheduler_kwargs)
    return Engine(scheduler, FakeRunner())


def drain(engine: Engine, max_steps: int = 50) -> list:
    outputs = []
    for _ in range(max_steps):
        step = engine.step()
        if not step:
            return outputs
        outputs.extend(step)
    raise AssertionError("engine never went idle")


def test_step_runs_a_request_to_completion():
    engine = make_engine()
    seq_id = engine.submit([1, 2], SamplingParams(temperature=0.0, max_new_tokens=3))

    outputs = drain(engine)

    assert [out.token_id for out in outputs] == [7, 7, 7]
    assert [out.seq_id for out in outputs] == [seq_id] * 3
    assert [out.finished for out in outputs] == [False, False, True]
    assert [b.is_prefill for b in engine.runner.batches] == [True, False, False]


def test_finished_request_leaves_nothing_behind():
    engine = make_engine()
    engine.submit([1, 2, 3], SamplingParams(max_new_tokens=4))

    drain(engine)

    assert engine.scheduler.running == []
    assert not engine.scheduler.waiting
    assert engine.scheduler.block_manager.num_free_blocks() == NUM_BLOCKS


def test_two_requests_share_the_decode_batch():
    engine = make_engine()
    engine.submit([1, 2], SamplingParams(max_new_tokens=3))
    engine.submit([3, 4], SamplingParams(max_new_tokens=3))

    outputs = drain(engine)

    assert len(outputs) == 6
    decode_sizes = [len(b.seqs) for b in engine.runner.batches if not b.is_prefill]
    assert decode_sizes[0] == 2


def test_submit_rejects_a_prompt_that_can_never_fit():
    engine = make_engine(max_num_batched_tokens=4)
    with pytest.raises(ValueError, match="step budget"):
        engine.submit([1, 2, 3, 4, 5], SamplingParams())


def test_abort_frees_a_running_request():
    engine = make_engine()
    seq_id = engine.submit([1, 2], SamplingParams(max_new_tokens=50))
    engine.step()
    assert engine.scheduler.block_manager.num_free_blocks() < NUM_BLOCKS

    engine.abort(seq_id)

    assert engine.step() == []
    assert engine.scheduler.block_manager.num_free_blocks() == NUM_BLOCKS
    assert engine.scheduler.running == []


def test_abort_drops_a_waiting_request():
    engine = make_engine(max_num_seqs=1)
    first = engine.submit([1, 2], SamplingParams(max_new_tokens=2))
    second = engine.submit([3, 4], SamplingParams(max_new_tokens=2))
    engine.step()

    engine.abort(second)
    outputs = drain(engine)

    assert {out.seq_id for out in outputs} == {first}
    assert engine.scheduler.block_manager.num_free_blocks() == NUM_BLOCKS


def test_abort_of_an_unknown_request_is_ignored():
    engine = make_engine()
    engine.abort(999)
    assert engine.step() == []


def test_streams_tokens_from_the_background_thread():
    async def main():
        engine = make_engine()
        engine.start(asyncio.get_running_loop())
        try:
            seq_id = engine.submit([1, 2], SamplingParams(max_new_tokens=3))
            return [out async for out in engine.outputs(seq_id)]
        finally:
            engine.stop()

    outputs = asyncio.run(asyncio.wait_for(main(), timeout=10))

    assert [out.token_id for out in outputs] == [7, 7, 7]
    assert outputs[-1].finished


def test_a_stop_token_ends_the_request_before_the_length_cap():
    engine = make_engine()
    engine.submit([1, 2], SamplingParams(max_new_tokens=50, stop_token_ids=(7,)))

    outputs = drain(engine)

    assert [out.finished for out in outputs] == [True]
    assert engine.scheduler.running == []
    assert engine.scheduler.block_manager.num_free_blocks() == NUM_BLOCKS


def test_a_rejected_request_leaves_no_stream_behind():
    engine = make_engine(max_num_batched_tokens=4)
    with pytest.raises(ValueError):
        engine.submit([1, 2, 3, 4, 5], SamplingParams())
    assert engine._streams == {}


def test_publishing_to_an_aborted_stream_is_dropped():
    engine = make_engine()
    seq_id = engine.submit([1, 2], SamplingParams())
    engine.abort(seq_id)
    assert seq_id not in engine._streams

    engine._publish(Output(seq_id=seq_id, token_id=7, finished=False))


class LockWatchingRunner(FakeRunner):
    engine: Engine

    def __init__(self):
        super().__init__()
        self.locked_during_forward = []

    def forward(self, batch):
        self.locked_during_forward.append(self.engine._lock.locked())
        return super().forward(batch)


def test_forward_runs_without_the_scheduler_lock_held():
    engine = make_engine()
    engine.runner = LockWatchingRunner()
    engine.runner.engine = engine
    engine.submit([1, 2], SamplingParams(max_new_tokens=2))

    drain(engine)

    assert engine.runner.locked_during_forward == [False, False]


def test_preemption_mid_step_keeps_outputs_aligned_with_the_batch():
    engine = make_engine(num_blocks=2)
    first = engine.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=2))
    second = engine.submit([5, 6, 7, 8], SamplingParams(max_new_tokens=2))

    prefill = engine.step()
    assert [out.seq_id for out in prefill] == [first, second]

    decode = engine.step()
    victim = engine.scheduler.waiting[0]
    assert [out.seq_id for out in decode] == [first]
    assert victim.seq_id == second
    assert victim.status is SeqStatus.PREEMPTED
    assert len(victim.output_token_ids) == 1
    assert victim.num_computed_tokens == 0

    rest = drain(engine)
    assert [out.seq_id for out in rest].count(second) == 1
    assert victim.status is SeqStatus.FINISHED
    assert engine.scheduler.block_manager.num_free_blocks() == 2


def test_the_loop_stamps_the_timestamps_the_benchmarks_read():
    engine = make_engine()
    engine.submit([1, 2], SamplingParams(max_new_tokens=3))
    seq = engine.scheduler.waiting[0]
    assert seq.first_token_time is None

    drain(engine)

    assert seq.first_token_time is not None
    assert seq.arrival_time <= seq.first_token_time <= seq.finish_time


def test_time_to_first_token_survives_preemption():
    engine = make_engine(num_blocks=2)
    engine.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=2))
    engine.submit([5, 6, 7, 8], SamplingParams(max_new_tokens=2))
    engine.step()
    victim = engine.scheduler.running[-1]
    stamped_at = victim.first_token_time
    assert stamped_at is not None

    engine.step()
    assert victim.status is SeqStatus.PREEMPTED
    drain(engine)

    assert victim.status is SeqStatus.FINISHED
    assert victim.first_token_time == stamped_at


def test_finished_requests_are_recorded_for_the_benchmarks():
    engine = make_engine()
    engine.submit([1, 2], SamplingParams(max_new_tokens=3))

    drain(engine)

    (record,) = engine.metrics()
    assert record.num_prompt_tokens == 2
    assert record.num_output_tokens == 3
    assert 0 <= record.ttft <= record.e2e


def test_an_aborted_request_is_left_out_of_the_metrics():
    engine = make_engine()
    seq_id = engine.submit([1, 2], SamplingParams(max_new_tokens=50))
    engine.step()
    engine.abort(seq_id)

    drain(engine)

    assert engine.metrics() == []


def test_metrics_history_is_bounded(monkeypatch):
    monkeypatch.setattr("nanoserve.engine.METRICS_HISTORY", 2)
    engine = make_engine()
    for _ in range(4):
        engine.submit([1, 2], SamplingParams(max_new_tokens=1))
        drain(engine)

    assert [record.seq_id for record in engine.metrics()] == [2, 3]


def test_a_preempted_request_reports_its_true_first_token_time():
    engine = make_engine(num_blocks=2)
    engine.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=2))
    engine.submit([5, 6, 7, 8], SamplingParams(max_new_tokens=2))
    engine.step()
    victim = engine.scheduler.running[-1]
    engine.step()
    assert victim.status is SeqStatus.PREEMPTED

    drain(engine)

    record = next(r for r in engine.metrics() if r.seq_id == victim.seq_id)
    assert record.num_output_tokens == 2
    assert record.ttft == victim.first_token_time - victim.arrival_time
    assert record.ttft <= record.e2e


class RunnerWithConfig(FakeRunner):
    class config:
        dtype = "torch.bfloat16"


def test_info_reports_the_dtype_only_when_the_runner_has_a_config():
    engine = make_engine()
    assert "dtype" not in engine.info()

    engine.runner = RunnerWithConfig()
    assert engine.info()["dtype"] == "torch.bfloat16"
