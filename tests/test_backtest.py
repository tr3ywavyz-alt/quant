import pytest

from quant_research.backtest import Trade, evaluate_trades


def test_trade_costs_reduce_net_pnl() -> None:
    trade = Trade(gross_pnl=100, fees=2.5, slippage=5)
    assert trade.net_pnl == 92.5


def test_evaluate_trades_builds_risk_aware_report() -> None:
    trades = [
        Trade(gross_pnl=100, fees=5),
        Trade(gross_pnl=-40, fees=5),
        Trade(gross_pnl=75, fees=5),
    ]
    report = evaluate_trades(trades, starting_equity=1000)

    assert report.trades == 3
    assert report.net_pnl == 115
    assert report.ending_equity == 1115
    assert report.max_drawdown == 45
    assert report.profit_factor == 165 / 45
    assert report.expectancy == pytest.approx(115 / 3)
    assert report.win_rate == pytest.approx(2 / 3)


def test_invalid_costs_and_equity_are_rejected() -> None:
    with pytest.raises(ValueError):
        Trade(gross_pnl=10, fees=-1)
    with pytest.raises(ValueError):
        evaluate_trades([], starting_equity=0)
