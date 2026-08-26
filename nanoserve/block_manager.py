"""Paged KV cache block manager: who owns which physical block.

This is the bookkeeping half of paging. KVCache holds the memory, this file
decides which sequence gets which block id, and the two share nothing but
integers.

The manager owns every block table, keyed by seq_id, and hands out read-only
copies. One source of truth for ownership means finish, preemption and client
disconnect are all the same call: free(seq_id).

One block has exactly one owner. There are no refcounts, which is most of why
this file is short; prefix caching is where they arrive, and every return to
the free list already goes through _release_block() so they land in one place.

The free list is a plain stack. Paged attention gathers through the block
table, so physical order buys nothing.
"""
from __future__ import annotations


class OutOfBlocks(Exception):
    """Raised when an allocation cannot be satisfied. The scheduler decides
    whether to preempt."""


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: list[int] = list(range(num_blocks))
        self._block_tables: dict[int, list[int]] = {}
        self._token_counts: dict[int, int] = {}
        self._num_allocated = 0

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= len(self._free_blocks)

    def get_block_table(self, seq_id: int) -> list[int]:
        """A copy, so callers can read but never mutate ownership state."""
        return list(self._block_tables[seq_id])

    def num_tokens(self, seq_id: int) -> int:
        return self._token_counts[seq_id]

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """Allocate blocks for a prefill of num_tokens. Atomic: on failure,
        no state changes. Raises OutOfBlocks if insufficient."""
        if seq_id in self._block_tables:
            raise ValueError(f"seq {seq_id} already has an allocation")
        needed = self.blocks_needed(num_tokens)
        if needed > len(self._free_blocks):
            raise OutOfBlocks(f"need {needed}, have {len(self._free_blocks)}")
        self._block_tables[seq_id] = [self._free_blocks.pop() for _ in range(needed)]
        self._token_counts[seq_id] = num_tokens
        self._num_allocated += needed
        self._check_accounting()

    def append_slot(self, seq_id: int) -> None:
        """Account for one new decode token. Allocates a new physical block
        iff the current token count is an exact multiple of block_size
        (i.e. every allocated slot is full). Atomic on failure."""
        token_count = self._token_counts[seq_id]
        if token_count % self.block_size == 0:
            if not self._free_blocks:
                raise OutOfBlocks("no free block for decode step")
            self._block_tables[seq_id].append(self._free_blocks.pop())
            self._num_allocated += 1
        self._token_counts[seq_id] = token_count + 1
        self._check_accounting()

    def free(self, seq_id: int) -> None:
        """The single cleanup path: finish, preemption, and client disconnect
        all end here. Raises KeyError on unknown/double free (v1 treats a
        double free as a bug, never a no-op)."""
        table = self._block_tables.pop(seq_id)
        del self._token_counts[seq_id]
        self._num_allocated -= len(table)
        for block_id in table:
            self._release_block(block_id)
        self._check_accounting()

    def _release_block(self, block_id: int) -> None:
        """Sole return path to the free list. When prefix caching arrives,
        this becomes decref-and-return-if-zero; nothing else changes."""
        self._free_blocks.append(block_id)

    def _check_accounting(self) -> None:
        """Every block is either owned or free, exactly once.

        Runs on every allocation, so it counts rather than scans. A decode
        step calls append_slot once per running sequence, and a scan over
        thousands of blocks there would cost more than the forward pass.
        check_no_shared_blocks() is the scanning version, for tests.
        """
        assert (
            self._num_allocated + len(self._free_blocks) == self.num_blocks
        ), "block leak/dup"

    def check_no_shared_blocks(self) -> None:
        """Assert no block id appears in two places at once.

        O(num_blocks), so it is not on the allocation path. Tests call it
        after the sequences of operations where aliasing could creep in.
        """
        seen: set[int] = set(self._free_blocks)
        assert len(seen) == len(self._free_blocks), "duplicate in the free list"
        for seq_id, table in self._block_tables.items():
            for block_id in table:
                assert block_id not in seen, f"block {block_id} owned twice"
                seen.add(block_id)
