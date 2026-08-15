"""KV cache tests: the address maths, on CPU.

Nothing here needs a GPU. The cache is tensors plus the map from a token
position to a slot in them, and that map is where a paged engine quietly
breaks. A wrong slot writes real keys and values into someone else's
context, and the model keeps producing fluent text. So the map is tested
on its own, away from any kernel.
"""
import pytest
import torch

from nanoserve.kv_cache import (
    KVCache,
    bytes_per_block,
    pad_block_tables,
    slot_mapping,
)

BS = 256  # block size, which is also the kernel's page size


def make_cache(num_layers=2, num_blocks=4, num_kv_heads=2, head_dim=8):
    return KVCache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=BS,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )


# ---------- slot arithmetic ----------

def test_slot_mapping_follows_the_block_table():
    # Blocks come off the free list in any order, so the map must never
    # assume 7 follows 3.
    table = [3, 7]
    slots = slot_mapping(table, range(BS + 2), BS)
    assert slots[0] == 3 * BS
    assert slots[BS - 1] == 3 * BS + BS - 1
    assert slots[BS] == 7 * BS          # first token of the second block
    assert slots[BS + 1] == 7 * BS + 1


def test_slot_mapping_is_dense_within_a_block():
    assert slot_mapping([1], range(BS), BS) == list(range(BS, 2 * BS))


def test_pad_block_tables_rectangularizes():
    assert pad_block_tables([[5], [2, 9], [1, 4, 8]]) == [
        [5, 0, 0],
        [2, 9, 0],
        [1, 4, 8],
    ]


def test_pad_block_tables_leaves_equal_length_tables_alone():
    assert pad_block_tables([[5, 6], [2, 9]]) == [[5, 6], [2, 9]]


def test_bytes_per_block_counts_k_and_v_in_every_layer():
    # keys and values, 4 layers, BS tokens, 2 heads, 8 dims, 2 bytes
    assert bytes_per_block(4, BS, 2, 8, torch.bfloat16) == 2 * 4 * BS * 2 * 8 * 2


# ---------- the cache tensors ----------

def test_block_size_must_suit_the_paged_kernel():
    with pytest.raises(ValueError, match="multiple of 256"):
        KVCache(1, 4, 16, 2, 8, torch.float32, "cpu")


def test_cache_starts_zeroed():
    # Junk memory would reach the kernel as NaNs in the empty slots of a
    # half full block.
    cache = make_cache()
    assert not cache.k.any()
    assert not cache.v.any()


def test_store_writes_exactly_the_mapped_slots():
    cache = make_cache()
    k = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    v = -k
    slots = torch.tensor(slot_mapping([2], range(3), BS))

    cache.store(layer_idx=1, k=k, v=v, slot_mapping=slots)

    k_cache, v_cache = cache.layer(1)
    assert torch.equal(k_cache[2, :3], k)
    assert torch.equal(v_cache[2, :3], v)
    # The rest of the block, other blocks, and other layers stay empty.
    assert not k_cache[2, 3:].any()
    assert not k_cache[[0, 1, 3]].any()
    assert not cache.layer(0)[0].any()


def test_store_handles_a_block_boundary():
    cache = make_cache()
    tokens = BS + 1
    k = torch.randn(tokens, 2, 8)
    slots = torch.tensor(slot_mapping([0, 3], range(tokens), BS))

    cache.store(0, k, k, slots)

    k_cache, _ = cache.layer(0)
    assert torch.equal(k_cache[0], k[:BS])       # first block filled
    assert torch.equal(k_cache[3, 0], k[BS])     # the extra token goes here
    assert not k_cache[3, 1:].any()


def test_num_slots_is_every_token_seat():
    assert make_cache(num_blocks=4).num_slots == 4 * BS
