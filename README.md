# Quant Research Toolkit

A compact, test-driven Python toolkit for evaluating systematic trading ideas without hiding risk behind headline returns.

The project focuses on the parts of research that are easy to get wrong: drawdown measurement, profit-factor accounting, walk-forward splits, transaction-cost-aware P&L, and reproducible validation.

> **Research only.** This repository is an engineering portfolio project, not financial advice and not a claim of live trading performance.

## What this demonstrates

- **Clean research primitives** — deterministic metrics and backtest accounting with explicit assumptions.
- **Risk-first evaluation** — max drawdown, profit factor, expectancy, win rate, and exposure-aware summaries.
- **Walk-forward validation** — chronological train/test windows that avoid random-shuffle leakage.
- **Execution realism** — configurable fees and slippage are deducted at the trade level.
- **Software quality** — typed Python, unit tests, linting, and GitHub Actions CI.

## Repository layout

```text
src/quant_research/
  backtest.py      # transaction-cost-aware trade accounting
  metrics.py       # robust performance statistics
  walkforward.py   # chronological validation splits
tests/             # unit tests for edge cases and invariants
docs/architecture.md
.github/workflows/ci.yml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Example:

```python
from quant_research.backtest import Trade, evaluate_trades

trades = [
    Trade(gross_pnl=125.0, fees=2.5, slippage=4.0),
    Trade(gross_pnl=-60.0, fees=2.5, slippage=3.0),
    Trade(gross_pnl=90.0, fees=2.5, slippage=4.0),
]

report = evaluate_trades(trades, starting_equity=10_000)
print(report)
```

## Design philosophy

A strategy is not interesting because one metric is large. It is interesting when its behavior remains understandable after costs, adverse sequences, parameter changes, and time-separated validation.

This toolkit therefore keeps the core intentionally small and inspectable. The goal is to make assumptions obvious enough that another engineer can challenge them.

## Engineering notes

- Core calculations are dependency-free.
- Inputs are validated and edge cases are covered by tests.
- Drawdown is measured from the running equity peak.
- Profit factor is reported as `inf` only when there are profits and no losses; an empty result reports `0.0`.
- Walk-forward windows preserve time ordering and support anchored or rolling training sets.

## Roadmap

- bootstrap confidence intervals for strategy metrics
- purged / embargoed cross-validation helpers
- position-sizing and portfolio-level risk aggregation
- benchmark comparison utilities
- reproducible experiment manifests

## License

See [LICENSE](LICENSE).
