"""Performance metrics with explicit edge-case semantics."""

from __future__ import annotations

from collections.abc import Iterable
import math


def profit_factor(pnl: Iterable[float]) -> float:
    """Return gross profits divided by absolute gross losses.

    Empty input returns 0.0. If profitable observations exist with no losses,
    the result is positive infinity.
    """

    values = [float(value) for value in pnl]
    gross_profit = sum(value for value in values if value > 0.0)
    gross_loss = -sum(value for value in values if value < 0.0)

    if gross_loss == 0.0:
        return math.inf if gross_profit > 0.0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(equity_curve: Iterable[float]) -> float:
    """Return maximum peak-to-trough drawdown in equity units.

    The function expects an equity curve, not returns. A non-decreasing or
    empty curve has zero drawdown.
    """

    values = [float(value) for value in equity_curve]
    if not values:
        return 0.0

    peak = values[0]
    worst = 0.0
    for equity in values:
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def expectancy(pnl: Iterable[float]) -> float:
    """Return arithmetic mean P&L per observation."""

    values = [float(value) for value in pnl]
    return sum(values) / len(values) if values else 0.0


def win_rate(pnl: Iterable[float]) -> float:
    """Return fraction of strictly profitable observations."""

    values = [float(value) for value in pnl]
    return sum(value > 0.0 for value in values) / len(values) if values else 0.0
