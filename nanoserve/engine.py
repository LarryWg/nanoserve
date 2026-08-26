"""Engine: the step loop that drives the model.

The scheduler decides who runs, the runner runs them, and the sequences
hold the results. This is the thing that turns those three into something
you can hand a pile of requests to.

One step, in the order everything else assumes:

    batch = scheduler.step()      # reserves this step's KV slots
    logits = runner.forward(batch)
    tokens = runner.sample(logits, batch)
    seq.on_prefilled() / seq.on_token(tok)
    scheduler.finish(seq) for whoever stopped

Single threaded, on purpose. Requests are submitted between steps by
whoever is driving, so there is no lock, no queue, and no ordering to
reason about. An HTTP server would need those back; the benchmarks drive
the engine directly instead, which measures this loop rather than a server
stack.

Tensor parallelism: see model_runner.py for the sharding plan. Rank 0
schedules and broadcasts the batch metadata each step, and the other ranks
run a receive loop.
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass

from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence

# How many finished requests to remember. Bounded so a long run does not
# grow a list forever.
METRICS_HISTORY = 10_000


@dataclass
class Output:
    """One generated token on its way back to whoever asked for it."""
    seq_id: int
    token_id: int
    finished: bool


@dataclass
class RequestMetrics:
    """What one finished request cost, measured inside the engine."""
    seq_id: int
    num_prompt_tokens: int
    num_output_tokens: int
    ttft: float          # arrival to first token
    e2e: float           # arrival to last token

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
        self._seq_ids = itertools.count()
        self._metrics: deque[RequestMetrics] = deque(maxlen=METRICS_HISTORY)

    def submit(self, prompt_ids: list[int], sampling: SamplingParams) -> int:
        """Queue a request and return its id.

        A prompt that can never fit raises here rather than waiting
        forever, so the caller gets the error on its own request.
        """
        seq = Sequence(next(self._seq_ids), list(prompt_ids), sampling)
        self.scheduler.add_request(seq)
        return seq.seq_id

    def abort(self, seq_id: int) -> None:
        """Drop a request, wherever it is sitting."""
        self.scheduler.abort(seq_id, time.monotonic())

    def step(self) -> list[Output]:
        """Run one iteration. Returns what was generated, empty when idle."""
        batch = self.scheduler.step()
        if batch is None:
            return []

        tokens = self.runner.sample(self.runner.forward(batch), batch)
        now = time.monotonic()
        outputs = []
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
        """The knobs this engine is actually running with.

        The benchmarks record this instead of trusting the flags they think
        they passed, and num_blocks is measured from free VRAM at startup
        rather than set, so it cannot be known any other way.
        """
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
        """Timings for requests that finished on their own.

        Aborted requests are left out on purpose: one dropped early would
        otherwise look like a very fast one.
        """
        return list(self._metrics)
