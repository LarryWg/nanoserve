from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager, OutOfBlocks
from .sequence import Sequence, SeqStatus


@dataclass
class ScheduledBatch:
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
        if seq.num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"prompt has {seq.num_tokens} tokens, more than the step "
                f"budget ({self.max_num_batched_tokens})"
            )
        cache_blocks = self.block_manager.num_blocks
        if self.block_manager.blocks_needed(seq.num_tokens) > cache_blocks:
            raise ValueError(
                f"prompt needs more KV blocks than the cache has ({cache_blocks})"
            )
        self.waiting.append(seq)

    def step(self) -> ScheduledBatch | None:
        if self.waiting and (self.prefill_first or not self.running):
            batch = self._try_prefill()
            if batch is not None:
                return batch
        if self.running:
            return self._decode()
        return None

    def _try_prefill(self) -> ScheduledBatch | None:
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
            self.block_manager.allocate(seq.seq_id, seq.num_tokens)
            token_budget -= seq.num_tokens
            admitted.append(seq)
        if not admitted:
            return None
        self.running.extend(admitted)
        return ScheduledBatch(seqs=admitted, is_prefill=True)

    def _decode(self) -> ScheduledBatch:
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            try:
                self.block_manager.append_slot(seq.seq_id)
                i += 1
            except OutOfBlocks:
                victim = self.running[-1]
                self._preempt(victim)
        return ScheduledBatch(seqs=list(self.running), is_prefill=False)

    def _preempt(self, victim: Sequence) -> None:
        self.running.remove(victim)
        self.block_manager.free(victim.seq_id)
        victim.on_preempted()
        self.waiting.appendleft(victim)

    def abort(self, seq_id: int, now: float) -> None:
        for seq in self.waiting:
            if seq.seq_id == seq_id:
                self.waiting.remove(seq)
                seq.on_finished(now)
                return
        for seq in self.running:
            if seq.seq_id == seq_id:
                self.finish(seq, now)
                return

    def finish(self, seq: Sequence, now: float) -> None:
        self.running.remove(seq)
        self.block_manager.free(seq.seq_id)
        seq.on_finished(now)
