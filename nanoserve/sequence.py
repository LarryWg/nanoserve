import time
from dataclasses import dataclass, field
from enum import Enum, auto


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
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

    num_computed_tokens: int = 0

    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None
    finish_time: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens == 0

    @property
    def last_token(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    @property
    def is_stopped(self) -> bool:
        if len(self.output_token_ids) >= self.sampling.max_new_tokens:
            return True
        if not self.output_token_ids:
            return False
        return self.output_token_ids[-1] in self.sampling.stop_token_ids

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
        self.status = SeqStatus.PREEMPTED

    def on_finished(self, now: float) -> None:
        self.finish_time = now
        self.status = SeqStatus.FINISHED
