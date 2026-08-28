import pytest

from quant_research.walkforward import walk_forward_splits


def test_rolling_walk_forward_preserves_order() -> None:
    splits = walk_forward_splits(20, train_size=8, test_size=4)

    assert [(s.train_start, s.train_end, s.test_start, s.test_end) for s in splits] == [
        (0, 8, 8, 12),
        (4, 12, 12, 16),
        (8, 16, 16, 20),
    ]


def test_anchored_walk_forward_expands_training_set() -> None:
    splits = walk_forward_splits(18, train_size=6, test_size=3, anchored=True)

    assert splits[0].train_slice == slice(0, 6)
    assert splits[-1].train_slice == slice(0, 15)
    assert all(split.train_end == split.test_start for split in splits)


def test_invalid_window_sizes_are_rejected() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(20, train_size=0, test_size=5)
    with pytest.raises(ValueError):
        walk_forward_splits(20, train_size=5, test_size=5, step_size=0)
