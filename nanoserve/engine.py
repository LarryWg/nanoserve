from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass

from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence

IDLE_POLL_SECONDS = 0.1

METRICS_HISTORY = 10_000


@dataclass
class Output:
    seq_id: int
    token_id: int
    finished: bool


@dataclass
class RequestMetrics:
    seq_id: int
    num_prompt_tokens: int
    num_output_tokens: int
    ttft: float
    e2e: float

    @classmethod
    def from_sequence(cls, seq: Sequence) -> RequestMetrics:
        return cls(
            seq_id=seq.seq_id,
            num_prompt_tokens=len(seq.prompt_token_ids),
            num_output_tokens=len(seq.output_token_ids),
            ttft=seq.first_token_time - seq.arrival_time,
            e2e=seq.finish_time - seq.arrival_time,
        )


class Engine:
    def __init__(self, scheduler: Scheduler, runner: ModelRunner):
        self.scheduler = scheduler
        self.runner = runner
        self._lock = threading.Lock()
        self._seq_ids = itertools.count()
        self._streams: dict[int, asyncio.Queue] = {}
        self._aborted: set[int] = set()
        self._metrics: deque[RequestMetrics] = deque(maxlen=METRICS_HISTORY)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def submit(self, prompt_ids: list[int], sampling: SamplingParams) -> int:
        seq = Sequence(next(self._seq_ids), list(prompt_ids), sampling)
        with self._lock:
            self.scheduler.add_request(seq)
            self._streams[seq.seq_id] = asyncio.Queue()
        self._wake.set()
        return seq.seq_id

    async def outputs(self, seq_id: int):
        stream = self._streams[seq_id]
        try:
            while True:
                out = await stream.get()
                yield out
                if out.finished:
                    return
        finally:
            self._streams.pop(seq_id, None)

    def abort(self, seq_id: int) -> None:
        with self._lock:
            self._aborted.add(seq_id)
            self._streams.pop(seq_id, None)
        self._wake.set()

    def step(self) -> list[Output]:
        with self._lock:
            self._apply_aborts()
            batch = self.scheduler.step()
        if batch is None:
            return []

        tokens = self.runner.sample(self.runner.forward(batch), batch)
        now = time.monotonic()
        outputs = []
        with self._lock:
            for seq, token in zip(batch.seqs, tokens):
                if batch.is_prefill:
                    seq.on_prefilled()
                seq.on_token(token, now)
                stopped = seq.is_stopped
                if stopped:
                    self.scheduler.finish(seq, now)
                    self._metrics.append(RequestMetrics.from_sequence(seq))
                outputs.append(Output(seq.seq_id, token, stopped))
        return outputs

    def info(self) -> dict:
        manager = self.scheduler.block_manager
        info = {
            "block_size": manager.block_size,
            "num_blocks": manager.num_blocks,
            "kv_cache_tokens": manager.num_blocks * manager.block_size,
            "max_num_seqs": self.scheduler.max_num_seqs,
            "max_num_batched_tokens": self.scheduler.max_num_batched_tokens,
            "prefill_first": self.scheduler.prefill_first,
        }
        config = getattr(self.runner, "config", None)
        if config is not None:
            info["dtype"] = str(config.dtype)
        return info

    def metrics(self) -> list[RequestMetrics]:
        return list(self._metrics)

    def start(self, event_loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._event_loop = event_loop
        self._thread = threading.Thread(
            target=self._loop, name="nanoserve-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            outputs = self.step()
            for out in outputs:
                self._publish(out)
            if not outputs:
                self._wake.wait(IDLE_POLL_SECONDS)

    def _publish(self, out: Output) -> None:
        stream = self._streams.get(out.seq_id)
        if stream is None:
            return
        if self._event_loop is None:
            stream.put_nowait(out)
        else:
            self._event_loop.call_soon_threadsafe(stream.put_nowait, out)

    def _apply_aborts(self) -> None:
        if not self._aborted:
            return
        now = time.monotonic()
        for seq_id in self._aborted:
            self.scheduler.abort(seq_id, now)
        self._aborted.clear()
