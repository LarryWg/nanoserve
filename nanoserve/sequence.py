"""Request/sequence lifecycle for the serving engine.

A Sequence is one generation stream. The scheduler moves sequences through:
WAITING -> RUNNING -> (PREEMPTED -> WAITING) -> FINISHED
"""
from dataclasses import dataclass, field
from enum import Enum, auto
import time


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()   # evicted under memory pressure; must re-prefill or restore
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

    # Paged KV cache bookkeeping: logical block -> physical block mapping
    # lives in BlockManager; the sequence just knows its block table.
    block_table: list[int] = field(default_factory=list)

    # Metrics (report these in your writeup: TTFT, ITL)
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        """True if this sequence hasn't computed its prompt KV yet."""
        return len(self.output_token_ids) == 0

    def last_token(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]
