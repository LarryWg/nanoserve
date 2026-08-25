"""Engine: the step loop that drives the model.

The scheduler decides who runs, the runner runs them, and the sequences
hold the results. This is the thing that turns those three into a server:
it repeats one step forever and hands the tokens out as they appear.

One step, in the order everything else assumes:

    batch = scheduler.step()      # reserves this step's KV slots
    logits = runner.forward(batch)
    tokens = runner.sample(logits, batch)
    seq.on_prefilled() / seq.on_token(tok)
    scheduler.finish(seq) for whoever stopped

Threading: the loop runs in its own thread and the HTTP server stays on the
event loop. GPU kernels release the GIL, so a thread is good enough here; a
separate process is the alternative if profiling ever says otherwise. Every
handoff between the two threads goes through exactly two places, `_lock`
around the scheduler and `_publish` on the way out.

Tensor parallelism: see model_runner.py for the sharding plan. Rank 0 owns
the HTTP server, the scheduler, and this loop; it broadcasts the batch
metadata each step and the other ranks run a receive loop.
"""
from __future__ import annotations

import asyncio
import itertools
import threading
import time
from dataclasses import dataclass

from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence
from .model_runner import ModelRunner

# How long an idle loop sleeps before looking again. Only reached when there
# is no work at all, and submit() wakes it early, so it costs nothing.
IDLE_POLL_SECONDS = 0.1


@dataclass
class Output:
    """One generated token on its way back to whoever asked for it."""
    seq_id: int
    token_id: int
    finished: bool


class Engine:
    def __init__(self, scheduler: Scheduler, runner: ModelRunner):
        self.scheduler = scheduler
        self.runner = runner
        self._lock = threading.Lock()
        self._seq_ids = itertools.count()
        self._streams: dict[int, asyncio.Queue] = {}
        self._aborted: set[int] = set()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def submit(self, prompt_ids: list[int], sampling: SamplingParams) -> int:
        """Queue a request and return its id. Safe to call from any thread.

        A prompt that can never fit raises here rather than waiting forever,
        so the caller gets the error on its own request.
        """
        seq = Sequence(next(self._seq_ids), list(prompt_ids), sampling)
        with self._lock:
            self.scheduler.add_request(seq)
            self._streams[seq.seq_id] = asyncio.Queue()
        self._wake.set()
        return seq.seq_id

    async def outputs(self, seq_id: int):
        """Yield a request's tokens as the loop produces them."""
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
        """Drop a request whose client went away.

        The work happens at the top of the next step, not here, because this
        runs on the server thread and the loop may be mid-forward with this
        sequence in its batch.
        """
        with self._lock:
            self._aborted.add(seq_id)
            self._streams.pop(seq_id, None)
        self._wake.set()

    def step(self) -> list[Output]:
        """Run one iteration. Returns what was generated, empty when idle.

        The lock is held around the scheduler and the sequences, never
        around the forward pass, so a request can arrive while the GPU is
        busy.
        """
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
                outputs.append(Output(seq.seq_id, token, stopped))
        return outputs

    def start(self, event_loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Run the loop in a background thread.

        Pass the server's event loop so finished tokens can hop threads onto
        it. Without one, outputs are still published and tests can read them
        straight off the queue.
        """
        self._event_loop = event_loop
        self._thread = threading.Thread(target=self._loop, name="nanoserve-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Cleared before the step, so a request arriving during it still
            # leaves the event set and the wait below returns immediately.
            self._wake.clear()
            outputs = self.step()
            for out in outputs:
                self._publish(out)
            if not outputs:
                self._wake.wait(IDLE_POLL_SECONDS)

    def _publish(self, out: Output) -> None:
        stream = self._streams.get(out.seq_id)
        if stream is None:
            return                       # aborted; nobody is reading
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
