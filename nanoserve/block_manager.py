"""Paged KV cache block manager.

Design decisions (see DESIGN.md and design review):
- Free list is a plain stack: physical block order is irrelevant because paged
  attention gathers through the block table; contiguity buys nothing.
- The manager OWNS all block tables (seq_id -> table). Sequences get read-only
  copies. Every exit path (finish, preempt, disconnect) is exactly free(seq_id).
- Allocation invariant: a new block is allocated in append_slot when the
  existing token count N satisfies N % block_size == 0.
- v1 invariant: one block, one owner. No refcounts. All returns to the free
  list go through _release_block() so prefix-caching refcounts later land in
  exactly one place.
"""
from __future__ import annotations


class OutOfBlocks(Exception):
    """Raised when an allocation cannot be satisfied. Caller (scheduler)
    decides whether to preempt."""


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))   # stack; pop() from end
        self._tables: dict[int, list[int]] = {}           # seq_id -> physical ids
        self._num_tokens: dict[int, int] = {}             # seq_id -> token count

    # ---------- queries ----------

    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        return -(-num_tokens // self.block_size)          # ceil division

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= len(self._free)

    def get_block_table(self, seq_id: int) -> list[int]:
        return list(self._tables[seq_id])                 # copy: read-only view

    def num_tokens(self, seq_id: int) -> int:
        return self._num_tokens[seq_id]

    # ---------- allocation ----------

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """Allocate blocks for a prefill of num_tokens. Atomic: on failure,
        no state changes. Raises OutOfBlocks if insufficient."""
        if seq_id in self._tables:
            raise ValueError(f"seq {seq_id} already has an allocation")
        needed = self.blocks_needed(num_tokens)
        if needed > len(self._free):                      # check BEFORE popping
            raise OutOfBlocks(f"need {needed}, have {len(self._free)}")
        self._tables[seq_id] = [self._free.pop() for _ in range(needed)]
        self._num_tokens[seq_id] = num_tokens
        self._check_invariants()

    def append_slot(self, seq_id: int) -> None:
        """Account for one new decode token. Allocates a new physical block
        iff the current token count is an exact multiple of block_size
        (i.e. every allocated slot is full). Atomic on failure."""
        n = self._num_tokens[seq_id]                      # KeyError if unknown
        if n % self.block_size == 0:
            if not self._free:
                raise OutOfBlocks("no free block for decode step")
            self._tables[seq_id].append(self._free.pop())
        self._num_tokens[seq_id] = n + 1                  # only after success
        self._check_invariants()

    # ---------- release ----------

    def free(self, seq_id: int) -> None:
        """The single cleanup path: finish, preemption, and client disconnect
        all end here. Raises KeyError on unknown/double free (v1 treats a
        double free as a bug, never a no-op)."""
        table = self._tables.pop(seq_id)                  # KeyError if absent
        del self._num_tokens[seq_id]
        for block_id in table:
            self._release_block(block_id)
        self._check_invariants()

    def _release_block(self, block_id: int) -> None:
        """Sole return path to the free list. Prefix caching (Stage 3) turns
        this into decref-and-return-if-zero; nothing else changes."""
        self._free.append(block_id)

    # ---------- invariants ----------

    def _check_invariants(self) -> None:
        allocated = sum(len(t) for t in self._tables.values())
        assert allocated + len(self._free) == self.num_blocks, "block leak/dup"
        seen: set[int] = set(self._free)
        for table in self._tables.values():
            for b in table:
                assert b not in seen, f"block {b} owned twice"
                seen.add(b)
