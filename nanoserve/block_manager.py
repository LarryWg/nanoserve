from __future__ import annotations


class OutOfBlocks(Exception):
    pass


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
        return list(self._block_tables[seq_id])

    def num_tokens(self, seq_id: int) -> int:
        return self._token_counts[seq_id]

    def allocate(self, seq_id: int, num_tokens: int) -> None:
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
        token_count = self._token_counts[seq_id]
        if token_count % self.block_size == 0:
            if not self._free_blocks:
                raise OutOfBlocks("no free block for decode step")
            self._block_tables[seq_id].append(self._free_blocks.pop())
            self._num_allocated += 1
        self._token_counts[seq_id] = token_count + 1
        self._check_accounting()

    def free(self, seq_id: int) -> None:
        table = self._block_tables.pop(seq_id)
        del self._token_counts[seq_id]
        self._num_allocated -= len(table)
        for block_id in table:
            self._release_block(block_id)
        self._check_accounting()

    def _release_block(self, block_id: int) -> None:
        self._free_blocks.append(block_id)

    def _check_accounting(self) -> None:
        assert (
            self._num_allocated + len(self._free_blocks) == self.num_blocks
        ), "block leak/dup"

    def check_no_shared_blocks(self) -> None:
        seen: set[int] = set(self._free_blocks)
        assert len(seen) == len(self._free_blocks), "duplicate in the free list"
        for seq_id, table in self._block_tables.items():
            for block_id in table:
                assert block_id not in seen, f"block {block_id} owned twice"
                seen.add(block_id)
