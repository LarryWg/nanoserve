"""The HTTP layer, with a fake model behind it.

A real Engine and Scheduler run here; only the GPU and the tokenizer are
faked. So these tests cover the whole path a request takes -- submit,
schedule, step, detokenize, stream -- on a laptop.
"""
import json

import pytest
from fastapi.testclient import TestClient

from nanoserve.block_manager import BlockManager
from nanoserve.engine import Engine
from nanoserve.scheduler import Scheduler
from nanoserve.server import Detokenizer, create_app

EOS = 0


class FakeTokenizer:
    """One token per character, so tokens and text are easy to line up."""

    eos_token_id = EOS

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t != EOS)


class FakeRunner:
    """Replays a fixed script of tokens, the same one for every sequence."""

    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.step = 0

    def forward(self, batch):
        return None

    def sample(self, logits, batch):
        token = self.tokens[min(self.step, len(self.tokens) - 1)]
        self.step += 1
        return [token] * len(batch.seqs)


def make_client(tokens=None, **scheduler_kwargs) -> TestClient:
    manager = BlockManager(num_blocks=16, block_size=8)
    scheduler = Scheduler(block_manager=manager, **scheduler_kwargs)
    engine = Engine(scheduler, FakeRunner(tokens or [ord("x")]))
    return TestClient(create_app(engine, FakeTokenizer()))


def sse_chunks(text: str) -> list[dict]:
    """The JSON bodies of an SSE stream, minus the [DONE] terminator."""
    lines = [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_health():
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_completion_returns_the_generated_text():
    with make_client() as client:
        body = client.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 4}
        ).json()

    assert body["choices"][0]["text"] == "xxxx"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {
        "prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6
    }


def test_eos_stops_the_completion():
    with make_client(tokens=[ord("a"), ord("b"), EOS, ord("c")]) as client:
        body = client.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 20}
        ).json()

    assert body["choices"][0]["text"] == "ab"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_ignore_eos_keeps_generating():
    with make_client(tokens=[ord("a"), EOS, ord("c")]) as client:
        body = client.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 3, "ignore_eos": True},
        ).json()

    assert body["choices"][0]["finish_reason"] == "length"
    assert len(body["choices"][0]["text"]) == 2   # the eos itself decodes to ""


def test_stream_sends_one_chunk_per_token():
    with make_client() as client:
        with client.stream(
            "POST", "/v1/completions",
            json={"prompt": "hi", "max_tokens": 3, "stream": True},
        ) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            chunks = sse_chunks(response.read().decode())

    assert [c["choices"][0]["text"] for c in chunks] == ["x", "x", "x"]
    assert [c["choices"][0]["finish_reason"] for c in chunks] == [None, None, "length"]
    assert {c["id"] for c in chunks} == {chunks[0]["id"]}


def test_oversized_prompt_is_a_client_error():
    with make_client(max_num_batched_tokens=4) as client:
        response = client.post("/v1/completions", json={"prompt": "far too long"})

    assert response.status_code == 400
    assert "step budget" in response.json()["detail"]


def test_empty_prompt_is_a_client_error():
    with make_client() as client:
        assert client.post("/v1/completions", json={"prompt": ""}).status_code == 400


def test_concurrent_requests_are_batched_together():
    with make_client() as client:
        first = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
        second = client.post("/v1/completions", json={"prompt": "yo", "max_tokens": 2})

    assert first.json()["id"] != second.json()["id"]
    assert second.json()["choices"][0]["text"] == "xx"


class SplitTokenizer:
    """Splits one character across two tokens, the way UTF-8 does."""

    eos_token_id = None

    def encode(self, text):
        return [1]

    def decode(self, token_ids):
        return "é" if token_ids == [2, 3] else "�" * len(token_ids)


def test_detokenizer_holds_back_half_a_character():
    detok = Detokenizer(SplitTokenizer())
    assert detok.add(2) == ""      # first half, nothing sendable yet
    assert detok.add(3) == "é"
    assert detok.text == "é"
