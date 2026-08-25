"""The benchmark workload, without the 650 MB download.

load() takes both the file path and the tokenizer, so a handful of fake
conversations pins the sampling and the length filter. What matters here is
that a run is reproducible and that the prompts which reach the engine are
the ones the filter claims to keep.
"""
import json

import pytest

from benchmarks import dataset


class CharTokenizer:
    """One token per character, so a prompt's length is just len(text)."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


def write_dataset(tmp_path, pairs) -> str:
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(
        [{"conversations": [{"value": p}, {"value": c}]} for p, c in pairs]
    ))
    return str(path)


def load(path, num_prompts=1, **kwargs):
    return dataset.load(CharTokenizer(), num_prompts=num_prompts, path=path, **kwargs)


def test_the_same_seed_gives_the_same_prompts(tmp_path):
    """The whole point of the fixed seed: a difference between two runs has
    to be the engine, never the data."""
    path = write_dataset(tmp_path, [(f"prompt {'x' * i}", "reply " * 2) for i in range(40)])

    first = load(path, num_prompts=5, seed=1)
    again = load(path, num_prompts=5, seed=1)
    other = load(path, num_prompts=5, seed=2)

    assert [r.prompt for r in first] == [r.prompt for r in again]
    assert [r.prompt for r in first] != [r.prompt for r in other]


def test_load_returns_exactly_what_was_asked_for(tmp_path):
    path = write_dataset(tmp_path, [(f"prompt {'x' * i}", "a reply") for i in range(40)])

    requests = load(path, num_prompts=7)

    assert len(requests) == 7
    assert all(r.prompt_len == len(r.prompt) for r in requests)


def test_prompts_outside_the_length_window_are_dropped(tmp_path):
    """Each bound gets its own entry, so a broken comparison shows up as
    the wrong prompt surviving rather than a count being off by one."""
    keeper = "a good prompt"
    path = write_dataset(tmp_path, [
        ("srt", "a fine reply"),                        # prompt under the floor
        ("a fine prompt", "no"),                        # output under the floor
        ("x" * (dataset.MAX_PROMPT_TOKENS + 1), "a fine reply"),   # prompt too long
        ("x" * 1000, "y" * (dataset.MAX_TOTAL_TOKENS - 999)),      # total too long
        (keeper, "a fine reply"),
    ])

    (request,) = load(path, num_prompts=1)

    assert request.prompt == keeper


def test_fixed_output_len_overrides_the_measured_reply(tmp_path):
    """Load tests want every request to generate the same number of tokens,
    so throughput is not confounded by output length."""
    path = write_dataset(tmp_path, [("hello world", "a short reply")])

    (request,) = load(path, fixed_output_len=32)

    assert request.output_len == 32
    assert request.prompt_len == len("hello world")


def test_malformed_conversations_are_skipped(tmp_path):
    """Real ShareGPT has entries with a single turn and entries with no
    conversation at all. Neither may take a slot in the sample."""
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps([
        {"conversations": [{"value": "only one turn"}]},
        {},
        {"conversations": [{"value": "a real prompt"}, {"value": "a real reply"}]},
    ]))

    (request,) = load(str(path))

    assert request.prompt == "a real prompt"


def test_too_few_surviving_prompts_is_an_error(tmp_path):
    """Silently returning a shorter list would mean a benchmark run at a
    smaller request count than the one it reports."""
    path = write_dataset(tmp_path, [("hello world", "a reply")])

    with pytest.raises(RuntimeError, match="survived the filter"):
        load(path, num_prompts=5)
