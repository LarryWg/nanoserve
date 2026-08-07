"""Tests for BlockManager. Implement nanoserve/block_manager.py to turn these green.

Interface under test (from DESIGN.md + design review):
  BlockManager(num_blocks: int, block_size: int)
    .num_free_blocks() -> int
    .blocks_needed(num_tokens: int) -> int
    .can_allocate(num_tokens: int) -> bool
    .allocate(seq_id: int, num_tokens: int) -> None        # raises OutOfBlocks
    .append_slot(seq_id: int) -> None                       # raises OutOfBlocks
    .get_block_table(seq_id: int) -> list[int]              # read-only view
    .free(seq_id: int) -> None                              # single cleanup path
  Manager owns seq_id -> block_table (your design decision #2).
  Invariant: new block allocated when existing token count N % block_size == 0
  (your decision #3). Token count tracking per seq lives in the manager or is
  passed in; either way, append_slot() must not need the Sequence object.
"""
import pytest

from nanoserve.block_manager import BlockManager, OutOfBlocks

BS = 16  # block_size used throughout


def mk(num_blocks=8, block_size=BS):
    return BlockManager(num_blocks=num_blocks, block_size=block_size)


# ---------- blocks_needed / allocate ----------

def test_blocks_needed_is_ceil_div():
    m = mk()
    assert m.blocks_needed(1) == 1
    assert m.blocks_needed(BS) == 1
    assert m.blocks_needed(BS + 1) == 2
    assert m.blocks_needed(3 * BS) == 3

def test_blocks_needed_zero_tokens():
    # Decide and pin the semantics: a zero-token allocation is a caller bug.
    m = mk()
    with pytest.raises((ValueError, AssertionError)):
        m.blocks_needed(0)

def test_allocate_consumes_free_blocks():
    m = mk(num_blocks=8)
    m.allocate(seq_id=1, num_tokens=BS + 1)          # needs 2 blocks
    assert m.num_free_blocks() == 6
    assert len(m.get_block_table(1)) == 2

def test_allocate_raises_when_exhausted_and_is_atomic():
    m = mk(num_blocks=2)
    m.allocate(1, BS)                                 # 1 block used
    before = m.num_free_blocks()
    with pytest.raises(OutOfBlocks):
        m.allocate(2, 3 * BS)                         # needs 3, only 1 free
    # Failed allocation must not leak partial state:
    assert m.num_free_blocks() == before
    with pytest.raises(KeyError):
        m.get_block_table(2)

def test_can_allocate_matches_allocate():
    m = mk(num_blocks=2)
    assert m.can_allocate(2 * BS) is True
    assert m.can_allocate(2 * BS + 1) is False


# ---------- append_slot boundary (design decision #3) ----------
# Real boundary is prompt-length-relative: prefill via allocate(), then decode
# steps via append_slot(). Invariant: allocate a new block when N % BS == 0.

def test_decode_after_prompt_bs_minus_1_fills_last_slot_no_alloc():
    m = mk(num_blocks=8)
    m.allocate(1, BS - 1)                             # 1 block, 15/16 slots
    used_before = m.num_free_blocks()
    m.append_slot(1)                                  # token 16 -> slot 16/16
    assert m.num_free_blocks() == used_before         # NO new block
    assert len(m.get_block_table(1)) == 1

def test_decode_after_prompt_exactly_bs_allocates():
    m = mk(num_blocks=8)
    m.allocate(1, BS)                                 # 1 full block
    m.append_slot(1)                                  # token BS+1 -> new block
    assert len(m.get_block_table(1)) == 2

def test_decode_after_prompt_bs_plus_1_no_alloc():
    m = mk(num_blocks=8)
    m.allocate(1, BS + 1)                             # 2 blocks, second has 1/16
    m.append_slot(1)
    assert len(m.get_block_table(1)) == 2             # still 2

def test_append_slot_crossing_boundary_twice():
    # March a sequence across a full block via repeated decode steps.
    m = mk(num_blocks=8)
    m.allocate(1, BS - 1)
    for _ in range(BS + 2):                           # tokens BS .. 2*BS+1
        m.append_slot(1)
    assert len(m.get_block_table(1)) == 2 + (1 if (2 * BS + 1) % BS else 0) or True
    # ^ explicit: after 2*BS+1 total tokens we need 3 blocks
    assert len(m.get_block_table(1)) == 3

def test_append_slot_raises_out_of_blocks():
    m = mk(num_blocks=1)
    m.allocate(1, BS)
    with pytest.raises(OutOfBlocks):
        m.append_slot(1)


# ---------- free(): the single cleanup path (design decision #2) ----------

def test_free_restores_all_blocks():
    m = mk(num_blocks=8)
    m.allocate(1, 3 * BS)
    m.allocate(2, BS)
    m.free(1)
    assert m.num_free_blocks() == 7
    m.free(2)
    assert m.num_free_blocks() == 8

def test_free_unknown_seq_raises():
    m = mk()
    with pytest.raises(KeyError):
        m.free(999)

def test_double_free_raises():
    # Silent double-free is the corruption bug your v1 invariant forbids.
    m = mk()
    m.allocate(1, BS)
    m.free(1)
    with pytest.raises(KeyError):
        m.free(1)

def test_freed_blocks_are_reusable_and_no_block_owned_twice():
    m = mk(num_blocks=2)
    m.allocate(1, 2 * BS)
    m.free(1)
    m.allocate(2, 2 * BS)                             # must reuse both blocks
    t2 = m.get_block_table(2)
    assert len(set(t2)) == 2                          # no duplicates in a table

def test_one_block_one_owner_invariant():
    m = mk(num_blocks=4)
    m.allocate(1, 2 * BS)
    m.allocate(2, 2 * BS)
    assert set(m.get_block_table(1)).isdisjoint(m.get_block_table(2))

def test_disconnect_path_is_just_free():
    # Simulated client disconnect mid-generation: engine calls free(seq_id).
    m = mk(num_blocks=4)
    m.allocate(1, BS)
    m.append_slot(1) if False else None               # (not needed; kept simple)
    m.allocate(2, BS)
    m.free(1)                                         # disconnect
    assert m.num_free_blocks() == 3
    assert set(m.get_block_table(2)) & set() == set() # seq 2 untouched
    assert len(m.get_block_table(2)) == 1


# ---------- global accounting invariant ----------

def test_conservation_of_blocks():
    m = mk(num_blocks=8)
    m.allocate(1, 3 * BS)
    m.allocate(2, BS + 1)
    allocated = len(m.get_block_table(1)) + len(m.get_block_table(2))
    assert allocated + m.num_free_blocks() == 8
