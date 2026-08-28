"""Small, auditable primitives for quantitative strategy research."""

from .backtest import BacktestReport, Trade, evaluate_trades
from .metrics import max_drawdown, profit_factor
from .walkforward import WalkForwardSplit, walk_forward_splits

__all__ = [
    "BacktestReport",
    "Trade",
    "WalkForwardSplit",
    "evaluate_trades",
    "max_drawdown",
    "profit_factor",
    "walk_forward_splits",
]
