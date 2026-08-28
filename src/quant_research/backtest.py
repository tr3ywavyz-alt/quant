"""Minimal trade-level backtest accounting.

This module deliberately avoids strategy-specific signal generation. It turns
already-generated trade outcomes into an auditable equity curve and summary.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import expectancy, max_drawdown, profit_factor, win_rate


@dataclass(frozen=True, slots=True)
class Trade:
    """One completed trade expressed in account currency."""

    gross_pnl: float
    fees: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        if self.fees < 0:
            raise ValueError("fees must be non-negative")
        if self.slippage < 0:
            raise ValueError("slippage must be non-negative")

    @property
    def net_pnl(self) -> float:
        return float(self.gross_pnl) - float(self.fees) - float(self.slippage)


@dataclass(frozen=True, slots=True)
class BacktestReport:
    trades: int
    starting_equity: float
    ending_equity: float
    net_pnl: float
    profit_factor: float
    max_drawdown: float
    expectancy: float
    win_rate: float


def evaluate_trades(
    trades: list[Trade] | tuple[Trade, ...],
    *,
    starting_equity: float,
) -> BacktestReport:
    """Evaluate completed trades after fees and slippage.

    Parameters
    ----------
    trades:
        Chronologically ordered completed trades.
    starting_equity:
        Positive account equity before the first trade.
    """

    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")

    pnl = [trade.net_pnl for trade in trades]
    equity = float(starting_equity)
    equity_curve = [equity]

    for result in pnl:
        equity += result
        equity_curve.append(equity)

    return BacktestReport(
        trades=len(pnl),
        starting_equity=float(starting_equity),
        ending_equity=equity,
        net_pnl=sum(pnl),
        profit_factor=profit_factor(pnl),
        max_drawdown=max_drawdown(equity_curve),
        expectancy=expectancy(pnl),
        win_rate=win_rate(pnl),
    )
