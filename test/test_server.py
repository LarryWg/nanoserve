import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from nanoserve.block_manager import BlockManager
from nanoserve.engine import Engine
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import SamplingParams
from nanoserve.server import Detokenizer, _Completion, create_app

EOS = 0


class FakeTokenizer:
    eos_token_id = EOS

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t != EOS)


class FakeRunner:
    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.step = 0

    def forward(self, batch):
        return None

    def sample(self, logits, batch):
        token = self.tokens[min(self.step, len(self.tokens) - 1)]
        self.step += 1
        return [token] * len(batch.seqs)


NUM_BLOCKS = 16


def make_client_and_engine(tokens=None, runner=None, **scheduler_kwargs):
    manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=8)
    scheduler = Scheduler(block_manager=manager, **scheduler_kwargs)
    engine = Engine(scheduler, runner or FakeRunner(tokens or [ord("x")]))
    return TestClient(create_app(engine, FakeTokenizer())), engine


def make_client(tokens=None, **scheduler_kwargs) -> TestClient:
    return make_client_and_engine(tokens, **scheduler_kwargs)[0]


def sse_chunks(text: str) -> list[dict]:
    lines = [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_health_reports_the_config_it_is_running():
    with make_client() as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["config"]["block_size"] == 8
    assert body["config"]["num_blocks"] == NUM_BLOCKS
    assert body["config"]["kv_cache_tokens"] == NUM_BLOCKS * 8
    assert body["config"]["max_num_seqs"] == 64


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
    assert len(body["choices"][0]["text"]) == 2


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
    eos_token_id = None

    def encode(self, text):
        return [1]

    def decode(self, token_ids):
        return "é" if token_ids == [2, 3] else "�" * len(token_ids)


def test_detokenizer_holds_back_half_a_character():
    detok = Detokenizer(SplitTokenizer())
    assert detok.add(2) == ""
    assert detok.add(3) == "é"
    assert detok.text == "é"


def test_models_lists_the_one_model_being_served():
    with make_client() as client:
        body = client.get("/v1/models").json()

    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["nanoserve"]


def test_a_finished_request_gives_its_blocks_back():
    client, engine = make_client_and_engine()
    with client:
        client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 12})

    assert engine.scheduler.running == []
    assert not engine.scheduler.waiting
    assert engine.scheduler.block_manager.num_free_blocks() == NUM_BLOCKS


def test_hanging_up_mid_stream_frees_the_kv_blocks():
    manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=8)
    scheduler = Scheduler(block_manager=manager)
    runner = FakeRunner([ord("x")])
    engine = Engine(scheduler, runner)
    tokenizer = FakeTokenizer()

    async def hang_up_after_one_chunk():
        engine.start(asyncio.get_running_loop())
        seq_id = engine.submit(
            tokenizer.encode("hi"), SamplingParams(max_new_tokens=200)
        )
        stream = _Completion(engine, tokenizer, seq_id, 2, "nanoserve", ()).stream()
        chunk = await stream.__anext__()
        await stream.aclose()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and scheduler.running:
            await asyncio.sleep(0.01)
        return chunk

    try:
        chunk = asyncio.run(asyncio.wait_for(hang_up_after_one_chunk(), timeout=15))
    finally:
        engine.stop()

    assert chunk.startswith("data: ")
    assert runner.step < 200, "the request ran to completion; nothing was aborted"
    assert scheduler.running == []
    assert manager.num_free_blocks() == NUM_BLOCKS


class PairingRunner(FakeRunner):
    def __init__(self, tokens, scheduler):
        super().__init__(tokens)
        self.scheduler = scheduler
        self.batch_sizes = []
        self.waited = False

    def forward(self, batch):
        self.batch_sizes.append(len(batch.seqs))
        if not self.waited and len(batch.seqs) < 2:
            self.waited = True
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not self.scheduler.waiting:
                time.sleep(0.005)
        return None


def test_two_in_flight_requests_share_a_decode_batch():
    manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=8)
    scheduler = Scheduler(block_manager=manager)
    runner = PairingRunner([ord("x")], scheduler)
    client = TestClient(create_app(Engine(scheduler, runner), FakeTokenizer()))

    bodies = {}

    def post(name, prompt):
        bodies[name] = client.post(
            "/v1/completions", json={"prompt": prompt, "max_tokens": 6}
        ).json()

    with client:
        threads = [
            threading.Thread(target=post, args=(name, prompt))
            for name, prompt in (("first", "hi"), ("second", "yo"))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

    assert not any(t.is_alive() for t in threads)
    assert bodies["first"]["choices"][0]["text"] == "xxxxxx"
    assert bodies["second"]["choices"][0]["text"] == "xxxxxx"
    assert max(runner.batch_sizes) == 2, runner.batch_sizes


class PartialCharTokenizer:
    eos_token_id = None

    def encode(self, text):
        return [1] * len(text)

    def decode(self, token_ids):
        return "é" if token_ids == [2, 3] else "\ufffd" * len(token_ids)


def test_every_token_gets_a_chunk_even_a_half_finished_character():
    manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=8)
    scheduler = Scheduler(block_manager=manager)
    engine = Engine(scheduler, FakeRunner([2, 3]))
    client = TestClient(create_app(engine, PartialCharTokenizer()))

    with client:
        with client.stream(
            "POST", "/v1/completions",
            json={"prompt": "hi", "max_tokens": 2, "stream": True},
        ) as response:
            chunks = sse_chunks(response.read().decode())

    assert len(chunks) == 2
    assert [c["choices"][0]["text"] for c in chunks] == ["", "é"]


def test_metrics_report_what_the_server_measured():
    client, engine = make_client_and_engine()
    with client:
        client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 5})
        body = client.get("/metrics").json()

    assert body["finished"] == 1
    record = body["requests"][0]
    assert record["num_prompt_tokens"] == 2
    assert record["num_output_tokens"] == 5
    assert 0 <= record["ttft"] <= record["e2e"]


def test_the_server_can_serve_decode_first():
    from nanoserve.server import build_app  # noqa: F401

    manager = BlockManager(num_blocks=NUM_BLOCKS, block_size=8)
    scheduler = Scheduler(block_manager=manager, prefill_first=False)
    engine = Engine(scheduler, FakeRunner([ord("x")]))
    with TestClient(create_app(engine, FakeTokenizer())) as client:
        body = client.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 3}
        ).json()

    assert body["choices"][0]["text"] == "xxx"
    assert client.app is not None
    assert engine.info()["prefill_first"] is False
