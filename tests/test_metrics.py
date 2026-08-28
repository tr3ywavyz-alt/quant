import math

from quant_research.metrics import expectancy, max_drawdown, profit_factor, win_rate


def test_profit_factor_uses_gross_wins_and_losses() -> None:
    assert profit_factor([100, -50, 25, -25]) == 125 / 75


def test_profit_factor_handles_no_losses_and_empty_input() -> None:
    assert math.isinf(profit_factor([10, 20]))
    assert profit_factor([]) == 0.0


def test_max_drawdown_tracks_running_peak() -> None:
    equity = [1000, 1100, 1075, 900, 950, 1200, 1150]
    assert max_drawdown(equity) == 200.0


def test_expectancy_and_win_rate() -> None:
    pnl = [10, -5, 0, 15]
    assert expectancy(pnl) == 5.0
    assert win_rate(pnl) == 0.5
