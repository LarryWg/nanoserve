"""The engine loop, without a GPU.

The loop's job is bookkeeping: drive the scheduler, advance sequences,
retire them, and hand the tokens out. None of that needs real logits, so a
fake runner stands in for the model and these tests run anywhere.
"""
import asyncio

import pytest

from nanoserve.block_manager import BlockManager
from nanoserve.engine import Engine
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import SamplingParams

BLOCK_SIZE = 4
NUM_BLOCKS = 8


class FakeRunner:
    """Returns a fixed token per sequence and remembers every batch it saw."""

    def __init__(self, token: int = 7):
        self.token = token
        self.batches = []

    def forward(self, batch):
        self.batches.append(batch)
        return None

    def sample(self, logits, batch):
        return [self.token] * len(batch.seqs)


def make_engine(**scheduler_kwargs) -> Engine:
    manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)
    scheduler = Scheduler(block_manager=manager, **scheduler_kwargs)
    return Engine(scheduler, FakeRunner())


def drain(engine: Engine, max_steps: int = 50) -> list:
    """Step until the engine goes idle."""
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
    # First step prefills the prompt, every later one decodes one token.
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
    engine.step()                       # prefill: the sequence now holds blocks
    assert engine.scheduler.block_manager.num_free_blocks() < NUM_BLOCKS

    engine.abort(seq_id)

    assert engine.step() == []          # the abort lands, then nothing is left
    assert engine.scheduler.block_manager.num_free_blocks() == NUM_BLOCKS
    assert engine.scheduler.running == []


def test_abort_drops_a_waiting_request():
    engine = make_engine(max_num_seqs=1)
    first = engine.submit([1, 2], SamplingParams(max_new_tokens=2))
    second = engine.submit([3, 4], SamplingParams(max_new_tokens=2))
    engine.step()                       # admits the first, the second waits

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
