"""The load-test harness, checked against a fake model.

A benchmark that is wrong is worse than no benchmark, because the number
still looks like a number. The server takes an injected engine, so the
whole client path can be exercised on a laptop before any GPU time is
spent on it.
"""
import asyncio

import httpx
import pytest

import bench
import dataset
from nanoserve.block_manager import BlockManager
from nanoserve.engine import Engine
from nanoserve.scheduler import Scheduler
from nanoserve.server import create_app
from test_server import EOS, FakeRunner, FakeTokenizer


def run_against_fake_server(requests, rate=float("inf"), tokens=None):
    """Drive the real client against the real server over ASGI."""

    async def main():
        manager = BlockManager(num_blocks=64, block_size=8)
        scheduler = Scheduler(block_manager=manager)
        engine = Engine(scheduler, FakeRunner(tokens or [ord("x")]))
        engine.start(asyncio.get_running_loop())
        app = create_app(engine, FakeTokenizer())
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://fake", timeout=httpx.Timeout(None)
            ) as client:
                return await bench.benchmark(
                    "http://fake/v1/completions", requests, rate, client=client
                )
        finally:
            engine.stop()

    return asyncio.run(asyncio.wait_for(main(), timeout=30))


def make_requests(*lengths):
    return [dataset.Request("hi there", 8, n) for n in lengths]


def test_poisson_schedule_has_the_rate_it_was_asked_for():
    offsets = bench.poisson_schedule(rate=4.0, count=20_000, seed=1)
    mean_gap = offsets[-1] / (len(offsets) - 1)
    assert 0.24 < mean_gap < 0.26          # 1/4 second between arrivals
    assert offsets == sorted(offsets)


def test_an_infinite_rate_sends_everything_at_once():
    assert bench.poisson_schedule(float("inf"), 5) == [0.0] * 5


def test_percentile_returns_a_value_that_happened():
    values = [float(i) for i in range(1, 101)]
    assert bench.percentile(values, 50) == 50.0
    assert bench.percentile(values, 99) == 100.0
    assert bench.percentile([], 50) != bench.percentile([], 50)   # nan


def test_chunk_count_equals_the_tokens_that_were_asked_for():
    """The invariant the whole benchmark rests on. If a token can arrive
    without its own chunk, every ITL and throughput number is wrong."""
    result = run_against_fake_server(make_requests(3, 7, 5))

    assert result["summary"]["num_failed"] == 0
    assert [r["num_chunks"] for r in result["requests"]] == [3, 7, 5]
    assert result["summary"]["output_tokens"] == 15


def test_ignore_eos_holds_the_output_length_fixed():
    """The fairness control that makes two engines comparable at all. The
    runner emits EOS on its second token; the request must run to its full
    length anyway, or engines would be measured on different work."""
    result = run_against_fake_server(
        make_requests(6), tokens=[ord("a"), EOS, ord("b")]
    )

    assert result["requests"][0]["num_chunks"] == 6


def test_timings_come_back_ordered_and_positive():
    result = run_against_fake_server(make_requests(5))
    record = result["requests"][0]

    assert record["chunk_times"] == sorted(record["chunk_times"])
    assert record["chunk_times"][0] > record["send_time"]
    summary = result["summary"]
    assert 0 < summary["ttft_p50"] <= summary["e2e_p50"]
    assert summary["itl_p99"] >= 0


def test_a_failed_request_is_recorded_not_swallowed():
    """A run with errors in it must not look like a clean fast run."""
    requests = make_requests(4)

    async def main():
        async with httpx.AsyncClient(timeout=httpx.Timeout(0.01)) as client:
            # Nothing is listening on this port.
            return await bench.benchmark(
                "http://127.0.0.1:9/v1/completions", requests, float("inf"), client=client
            )

    result = asyncio.run(asyncio.wait_for(main(), timeout=30))

    assert result["summary"]["num_failed"] == 1
    assert result["summary"]["output_tokens"] == 0
    assert result["requests"][0]["error"]


def test_summary_reports_attained_rate_next_to_offered():
    """Saturation has to be visible. A run that could not keep up shows a
    lower attained rate, rather than only a worse latency."""
    result = run_against_fake_server(make_requests(2, 2), rate=1000.0)
    summary = result["summary"]

    assert summary["offered_rate"] == 1000.0
    assert summary["attained_rate"] < summary["offered_rate"]
    assert 0.0 <= summary["client_cpu_fraction"]
