"""Continuous batching scheduler.

SECOND CORE FILE — implement the policy yourself.

Each engine step, the scheduler assembles one batch. Key concepts:

1. Iteration-level scheduling (the whole point of continuous batching):
   sequences join/leave the batch every step, not every request-completion.
   This is why throughput beats static batching at high QPS.

2. Prefill vs decode: a step is either a prefill batch or a decode batch
   (simple version), or mixed via chunked prefill (advanced — great Stage 3
   writeup material: chunked prefill trades TTFT of new requests against
   ITL spikes for running ones. Measure both and show the curve.)

3. Token budget: cap total tokens per step (e.g. max_num_batched_tokens=8192)
   so one huge prompt doesn't blow activation memory or stall decodes.

Policy decisions YOU own (each is an interview question):
- Admission: FCFS from the waiting queue? What starvation risks exist?
- Prefill priority: schedule waiting prefills before decodes ("prefill-first",
  better TTFT) or after ("decode-first", better ITL)? Make it a config flag
  and BENCHMARK BOTH — that comparison chart is writeup gold.
- Preemption: when append_slot raises OutOfBlocks mid-decode, which victim?
  vLLM evicts the most-recently-arrived (LIFO) — why? (Hint: fairness +
  recompute cost of long-running seqs.)
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
        raise NotImplementedError("implement me — this is your interview answer")

    def finish(self, seq: Sequence) -> None:
        """Free blocks, remove from running, mark finished."""
        raise NotImplementedError
