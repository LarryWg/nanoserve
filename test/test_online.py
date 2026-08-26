"""The in-process serving driver, against a fake model.

This replaced an HTTP client, so the things the socket used to give for
free -- one timestamp per token, arrivals landing on schedule, every
request accounted for -- are now this file's job to pin.
"""
import time

import dataset
import online
from nanoserve.block_manager import BlockManager
from nanoserve.engine import Engine
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import SamplingParams
from conftest import FakeRunner


def make_engine(**kwargs):
    manager = BlockManager(num_blocks=64, block_size=8)
    scheduler = Scheduler(block_manager=manager, **kwargs)
    return Engine(scheduler, FakeRunner([ord("x")]))


def make_requests(*lengths):
    return [dataset.Request("hi there", 8, n, prompt_ids=[1, 2, 3]) for n in lengths]


def sampling_for(req):
    return SamplingParams(temperature=0.0, max_new_tokens=req.output_len)


def test_every_token_gets_its_own_timestamp():
    """The invariant the whole benchmark rests on, restated without a
    socket: a request that asked for n tokens produces n timestamps."""
    engine = make_engine()
    requests = make_requests(3, 7, 5)

    records = online.run(engine, requests, [0.0, 0.0, 0.0], sampling_for)

    assert [r.num_chunks for r in records] == [3, 7, 5]
    assert all(len(r.chunk_times) == r.num_chunks for r in records)


def test_records_come_back_in_submission_order():
    engine = make_engine()
    records = online.run(engine, make_requests(2, 9, 4), [0.0] * 3, sampling_for)
    assert [r.output_len for r in records] == [2, 9, 4]


def test_timestamps_are_ordered_and_after_the_send():
    engine = make_engine()
    (record,) = online.run(engine, make_requests(5), [0.0], sampling_for)

    assert record.chunk_times == sorted(record.chunk_times)
    assert record.chunk_times[0] >= record.send_time
    assert record.ttft <= record.e2e


def test_arrivals_are_held_until_their_scheduled_time():
    """A request scheduled late must not be submitted early, or the
    arrival process being measured is not the one that was asked for."""
    engine = make_engine()
    start = time.perf_counter()

    records = online.run(engine, make_requests(2, 2), [0.0, 0.35], sampling_for)

    gap = records[1].send_time - records[0].send_time
    assert gap >= 0.3, gap
    assert time.perf_counter() - start >= 0.35


def test_the_engine_finishes_everything_it_was_given():
    """No request may be silently dropped: a lost one would shorten the
    run and flatter the throughput number."""
    engine = make_engine()
    requests = make_requests(*([3] * 12))

    records = online.run(engine, requests, [0.0] * 12, sampling_for)

    assert len(records) == 12
    assert sum(r.num_chunks for r in records) == 36
    assert engine.scheduler.block_manager.num_free_blocks() == 64


def test_decode_first_is_reachable_from_the_driver():
    engine = make_engine(prefill_first=False)
    records = online.run(engine, make_requests(4), [0.0], sampling_for)
    assert records[0].num_chunks == 4
    assert engine.info()["prefill_first"] is False
