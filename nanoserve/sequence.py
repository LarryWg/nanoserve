"""Request/sequence lifecycle for the serving engine.

A Sequence is one generation stream. The scheduler moves sequences through:
WAITING -> RUNNING -> FINISHED, with preemption sending RUNNING -> PREEMPTED
(the sequence rejoins the waiting queue and prefills again on readmission).

The Sequence owns token ids and metrics. It does NOT own its block table --
BlockManager does (keyed by seq_id), so there is exactly one source of truth
for physical block ownership.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
import time


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()   # evicted under memory pressure; must re-prefill
    FINISHED = auto()


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    stop_token_ids: tuple = ()


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    sampling: SamplingParams
    status: SeqStatus = SeqStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)

    # How many of this sequence's tokens currently have KV in the paged cache.
    # This, not len(output_token_ids), is what distinguishes prefill from
    # decode, because recompute preemption throws KV away while keeping the
    # tokens. A preempted sequence has outputs but zero computed tokens, so it
    # re-prefills prompt+outputs on readmission.
    num_computed_tokens: int = 0

    # Metrics timestamps; TTFT and ITL are derived from these.
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None
    finish_time: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")

    @property
    def token_ids(self) -> list[int]:
        """Full context. This is what a (re-)prefill must compute KV for."""
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens == 0

    @property
    def last_token(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    def on_prefilled(self) -> None:
        self.num_computed_tokens = self.num_tokens
        self.status = SeqStatus.RUNNING

    def on_token(self, token_id: int, now: float) -> None:
        if self.first_token_time is None:
            self.first_token_time = now
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1

    def on_preempted(self) -> None:
        self.num_computed_tokens = 0
        # Stays PREEMPTED while sitting in the waiting queue, so metrics can
        # count evictions. The next on_prefilled() makes it RUNNING again.
        self.status = SeqStatus.PREEMPTED

    def on_finished(self, now: float) -> None:
        self.finish_time = now
        self.status = SeqStatus.FINISHED

    @property
    def is_stopped(self) -> bool:
        """Stop condition, checked after each generated token."""
        if len(self.output_token_ids) >= self.sampling.max_new_tokens:
            return True
        return bool(self.output_token_ids) and self.output_token_ids[-1] in self.sampling.stop_token_ids
