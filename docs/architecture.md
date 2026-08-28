# Architecture

The toolkit is intentionally split into small layers so research assumptions stay visible.

## 1. Signal generation is outside the core

The package does not decide when to buy or sell. Signal generation is strategy-specific and tends to create the most accidental coupling. The public core begins after a hypothetical strategy has produced completed trade outcomes.

This separation makes it possible to test accounting and validation logic independently from a particular alpha hypothesis.

## 2. Trade accounting

`backtest.py` models one completed trade with three explicit values:

- gross P&L
- fees
- slippage

Net P&L is always calculated as:

```text
net_pnl = gross_pnl - fees - slippage
```

Costs cannot be negative. The evaluator builds an equity curve in chronological order and computes all summary statistics from net, not gross, outcomes.

## 3. Metrics

`metrics.py` contains deterministic, side-effect-free calculations.

Important invariants:

- drawdown is measured from the running high-water mark
- flat trades are neither wins nor losses for profit-factor totals
- win rate counts strictly positive observations
- empty input has defined behavior rather than raising accidental division errors

Keeping these functions small makes them easy to verify with hand-calculated test cases.

## 4. Walk-forward validation

`walkforward.py` creates chronological integer index ranges instead of shuffling observations.

Two modes are supported:

- **rolling:** fixed-length training data moves through time
- **anchored:** training data begins at index zero and expands through time

In both cases, the test window begins exactly where the training window ends, preventing overlap between training and evaluation data.

## 5. Testing strategy

The tests emphasize invariants and edge cases rather than snapshotting implementation details. Examples include:

- a known peak-to-trough drawdown
- all-winning and empty profit-factor inputs
- transaction costs changing both equity and risk metrics
- exact rolling / anchored window boundaries
- invalid inputs failing loudly

## 6. CI

GitHub Actions runs linting and tests on Python 3.11, 3.12, and 3.13. The matrix is deliberately small enough to remain fast while demonstrating compatibility across supported interpreter versions.

## Next architectural steps

A production research platform would add experiment manifests, immutable dataset fingerprints, purged cross-validation, bootstrap uncertainty estimates, portfolio aggregation, and richer execution models. Those belong above this compact core rather than being mixed into the primitives themselves.
