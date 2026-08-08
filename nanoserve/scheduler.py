"""Continuous batching scheduler.

Each engine step, the scheduler decides which sequences the model runs next.
That decision is the heart of a serving engine.

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

The policies, decided:
- Admission is first-come-first-served, and strict: if the sequence at the
  head of the waiting queue does not fit, later ones do not skip ahead.
  Skipping would be fairer to short prompts but lets big prompts starve.
- prefill_first chooses which batch runs when both are possible. True means
  new requests get their first token sooner; False means running requests
  get a steadier token stream. Benchmark both when the server exists.
- Preemption evicts the most recently admitted running sequence (LIFO): it
  has the least sunk compute, and evicting the oldest would thrash long
  sequences. Eviction is recompute-style: the blocks are freed, the tokens
  are kept, and the sequence prefills again from scratch on readmission.
"""
from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager, OutOfBlocks
from .sequence import Sequence, SeqStatus


@dataclass
class ScheduledBatch:
    """What the model runner executes this step.

    For prefill: every token of each admitted sequence (prompt plus any
    already-generated tokens, for sequences re-prefilling after preemption).
    For decode: exactly one new token per running sequence.
    """
    seqs: list[Sequence]
    is_prefill: bool


@dataclass
class Scheduler:
    block_manager: BlockManager
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    prefill_first: bool = True
    waiting: deque[Sequence] = field(default_factory=deque)
    running: list[Sequence] = field(default_factory=list)

    def add_request(self, seq: Sequence) -> None:
        """Queue a new request, rejecting up front what can never fit.

        Without these checks a too-big prompt would wait forever: every
        step would find it unschedulable and no error would ever surface.
        """
        if seq.num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"prompt has {seq.num_tokens} tokens, more than the step "
                f"budget ({self.max_num_batched_tokens})"
            )
        if self.block_manager.blocks_needed(seq.num_tokens) > self.block_manager.num_blocks:
            raise ValueError(
                f"prompt needs more KV blocks than the cache has "
                f"({self.block_manager.num_blocks})"
            )
        self.waiting.append(seq)

    def step(self) -> ScheduledBatch | None:
        """Assemble the next batch, or return None when there is no work.

        Prefill is attempted when the policy allows it; if no waiting
        sequence fits right now, the running sequences still get their
        decode step rather than idling.
        """
        if self.waiting and (self.prefill_first or not self.running):
            batch = self._try_prefill()
            if batch is not None:
                return batch
        if self.running:
            return self._decode()
        return None

    def _try_prefill(self) -> ScheduledBatch | None:
        """Admit waiting sequences until a limit says stop.

        Three independent limits, checked in order for the head of the
        queue: batch size (max_num_seqs), step token budget, and free KV
        blocks. The first "no" stops admission entirely (strict FCFS), so
        the loop is short and the policy is easy to reason about.
        """
        admitted: list[Sequence] = []
        token_budget = self.max_num_batched_tokens
        while self.waiting:
            seq = self.waiting[0]
            if len(self.running) + len(admitted) >= self.max_num_seqs:
                break
            if seq.num_tokens > token_budget:
                break
            if not self.block_manager.can_allocate(seq.num_tokens):
                break
            self.waiting.popleft()
            # A preempted sequence re-prefills its whole context: its KV was
            # freed, so num_tokens (prompt + outputs) is what needs blocks.
            self.block_manager.allocate(seq.seq_id, seq.num_tokens)
            token_budget -= seq.num_tokens
            admitted.append(seq)
        if not admitted:
            return None
        self.running.extend(admitted)
        return ScheduledBatch(seqs=admitted, is_prefill=True)

    def _decode(self) -> ScheduledBatch:
        """Reserve one new KV slot per running sequence.

        Decode batches always contain every running sequence; that is what
        makes decode latency predictable. The only failure is running out of
        blocks mid-step, handled by evicting the youngest sequence until the
        rest fit.
        """
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            try:
                self.block_manager.append_slot(seq.seq_id)
                i += 1
            except OutOfBlocks:
                victim = self.running[-1]
                self._preempt(victim)
                # The freed blocks guarantee the retry succeeds. If the
                # victim was seq itself, it has left running and index i
                # already points at the next sequence, so do not increment.
        return ScheduledBatch(seqs=list(self.running), is_prefill=False)

    def _preempt(self, victim: Sequence) -> None:
        """Evict a running sequence: free its KV, keep its tokens.

        The victim goes to the FRONT of the waiting queue. It already sank
        compute into its prompt and holds generated tokens, so letting newer
        requests jump it would waste both.
        """
        self.running.remove(victim)
        self.block_manager.free(victim.seq_id)
        victim.on_preempted()
        self.waiting.appendleft(victim)

    def finish(self, seq: Sequence, now: float) -> None:
        """Retire a completed sequence: free its KV and stamp the time."""
        self.running.remove(seq)
        self.block_manager.free(seq.seq_id)
        seq.on_finished(now)
