import gaps


def test_a_sequence_that_exactly_fills_its_blocks_wastes_nothing():
    result = gaps.fragmentation([256, 512], block_sizes=(256,))
    assert result[256]["wasted_fraction"] == 0.0


def test_one_token_over_a_block_boundary_holds_a_whole_extra_block():
    result = gaps.fragmentation([257], block_sizes=(256,))
    assert result[256]["slots_held"] == 512
    assert result[256]["wasted_fraction"] == 255 / 512


def test_the_big_page_wastes_much_more_than_the_small_one():
    lengths = [37, 128, 260, 513, 900, 1500, 64, 300]
    result = gaps.fragmentation(lengths, block_sizes=(16, 256))

    assert result[16]["wasted_fraction"] < 0.05
    assert result[256]["wasted_fraction"] > 0.25
    assert result[16]["tokens"] == result[256]["tokens"] == sum(lengths)
