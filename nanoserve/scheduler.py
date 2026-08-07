"""Continuous batching scheduler.

Each engine step, the scheduler decides which sequences the model runs next.
That decision is the heart of a serving engine, so this file is meant to be
written and understood by hand.

The ideas:

1. Iteration-level scheduling (the point of continuous batching): sequences
   join and leave the batch every step, not only when a whole request
   finishes. This is why continuous batching beats static batching at high
   request rates.

2. Prefill vs decode: a step is either a prefill batch or a decode batch,
   never a mix. (Mixing them is called chunked prefill, a later extension:
   it trades faster first tokens for new requests against small pauses in
   the token stream of running ones.)

3. Token budget: cap the total tokens per step (max_num_batched_tokens) so
   one huge prompt cannot blow up activation memory or starve the decodes.

Policy decisions to make here:
- Admission: take waiting requests first-come-first-served.
- Prefill priority: run waiting prefills before decodes ("prefill-first",
  lower time to first token) or after ("decode-first", steadier output
  stream)? Keep it a config flag and benchmark both.
- Preemption: when append_slot raises OutOfBlocks mid-decode, which running
  sequence gets evicted? Evict the most recently arrived one (LIFO): it has
  the least sunk compute, and evicting the oldest would thrash long
  sequences.
"""
from dataclasses import dataclass, field
from collections import deque

from .sequence import Sequence
from .block_manager import BlockManager


@dataclass
class ScheduledBatch:
    """What the model runner executes this step."""
    seqs: list[Sequence]
    is_prefill: bool
    # For prefill: all prompt tokens. For decode: one token per sequence.
    # Model runner flattens these + block tables into kernel inputs.


@dataclass
class Scheduler:
    block_manager: BlockManager
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    waiting: deque = field(default_factory=deque)   # Sequence, FCFS
    running: list = field(default_factory=list)     # Sequence

    def add_request(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def step(self) -> ScheduledBatch | None:
        """Assemble the next batch. Rough shape:

        1. If waiting is non-empty and policy says prefill-first:
           pop sequences while (token budget ok AND can_allocate AND
           len(running) < max_num_seqs); allocate their blocks; return
           a prefill batch.
        2. Else decode: for each running seq, append_slot(). On OutOfBlocks,
           preempt a victim (free its blocks, push back to waiting) and retry.
           Return a decode batch of all running seqs.
        3. Return None if nothing to do.
        """
        raise NotImplementedError("implement me")

    def finish(self, seq: Sequence) -> None:
        """Free blocks, remove from running, mark finished."""
        raise NotImplementedError
