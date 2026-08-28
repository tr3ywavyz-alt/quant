"""Chronological train/test split helpers for time-series research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)


def walk_forward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    anchored: bool = False,
) -> list[WalkForwardSplit]:
    """Generate non-leaking chronological train/test windows.

    ``train_end`` and ``test_end`` are exclusive indexes. With ``anchored``
    enabled, every training window begins at index zero while its end expands.
    Otherwise the training window rolls forward with each step.
    """

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")

    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    splits: list[WalkForwardSplit] = []
    train_start = 0
    train_end = train_size

    while train_end + test_size <= n_samples:
        splits.append(
            WalkForwardSplit(
                train_start=0 if anchored else train_start,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + test_size,
            )
        )

        train_end += step
        if not anchored:
            train_start += step

    return splits
