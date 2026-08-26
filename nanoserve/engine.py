from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass

from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence

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
        self._seq_ids = itertools.count()
        self._metrics: deque[RequestMetrics] = deque(maxlen=METRICS_HISTORY)

    def submit(self, prompt_ids: list[int], sampling: SamplingParams) -> int:
        seq = Sequence(next(self._seq_ids), list(prompt_ids), sampling)
        self.scheduler.add_request(seq)
        return seq.seq_id

    def abort(self, seq_id: int) -> None:
        self.scheduler.abort(seq_id, time.monotonic())

    def step(self) -> list[Output]:
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
