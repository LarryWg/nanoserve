"""Metric definitions, pinned.

These are shared by every driver, so an error here is an error in every
number the project reports.
"""
import metrics


def test_poisson_schedule_has_the_rate_it_was_asked_for():
    offsets = metrics.poisson_schedule(rate=4.0, count=20_000, seed=1)
    mean_gap = offsets[-1] / (len(offsets) - 1)
    assert 0.24 < mean_gap < 0.26          # 1/4 second between arrivals
    assert offsets == sorted(offsets)


def test_an_infinite_rate_sends_everything_at_once():
    assert metrics.poisson_schedule(float("inf"), 5) == [0.0] * 5


def test_percentile_returns_a_value_that_happened():
    values = [float(i) for i in range(1, 101)]
    assert metrics.percentile(values, 50) == 50.0
    assert metrics.percentile(values, 99) == 100.0
    assert metrics.percentile([], 50) != metrics.percentile([], 50)   # nan


def make_record(send, chunks, output_len=None):
    return metrics.RequestRecord(
        prompt_len=8, output_len=output_len or len(chunks),
        num_chunks=len(chunks), send_time=send, chunk_times=list(chunks),
    )


def test_ttft_itl_and_e2e_come_off_the_timestamps():
    record = make_record(0.0, [0.5, 0.7, 1.0])
    assert record.ttft == 0.5
    assert record.e2e == 1.0
    assert [round(x, 2) for x in record.itls] == [0.2, 0.3]


def test_attained_rate_shows_a_run_that_could_not_keep_up():
    """Saturation has to be visible in the summary, or the latency numbers
    below it look like slowness rather than a queue."""
    records = [make_record(0.0, [1.0, 2.0]) for _ in range(4)]
    summary = metrics.summarize(records, offered_rate=100.0, wall=2.0)["summary"]

    assert summary["attained_rate"] == 2.0        # 4 requests in 2 seconds
    assert summary["offered_rate"] == 100.0
    assert summary["output_tok_per_s"] == 4.0


def test_a_failed_request_is_counted_not_silently_dropped():
    good = make_record(0.0, [0.5])
    bad = metrics.RequestRecord(8, 4, 0, 0.0, [], error="boom")
    summary = metrics.summarize([good, bad], offered_rate=1.0, wall=1.0)["summary"]

    assert summary["num_requests"] == 2
    assert summary["num_failed"] == 1
    assert summary["output_tokens"] == 1


def test_bulk_timestamped_tokens_are_rejected():
    """The failure this exists to catch: a driver that reads output only
    when a request finishes stamps every token at the same instant. TTFT
    then collapses into end-to-end latency and ITL goes to zero, which
    looks like a fast engine rather than a broken measurement."""
    bulk = make_record(0.0, [9.0] * 10)
    try:
        metrics.check_timestamps_are_incremental([bulk])
    except RuntimeError as exc:
        assert "timestamped together" in str(exc)
    else:
        raise AssertionError("bulk timestamps were not caught")


def test_normal_timestamps_pass_the_check():
    fine = make_record(0.0, [0.1 * i for i in range(1, 11)])
    metrics.check_timestamps_are_incremental([fine])


def test_short_requests_are_not_flagged():
    """A two-token request can legitimately land in one step."""
    metrics.check_timestamps_are_incremental([make_record(0.0, [1.0, 1.0])])
