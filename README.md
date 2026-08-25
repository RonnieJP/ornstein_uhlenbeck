# Ornstein-Uhlenbeck Process for Pairs Trading

## Project aims

This project investigates the Ornstein-Uhlenbeck (OU) process as a model for mean-reverting spreads in pairs trading. The project combines the mathematical theory of the OU process with numerical simulation, parameter estimation, and backtesting.

The project progresses from the underlying stochastic process to its application in pairs trading:

1. Derive and investigate the properties of the OU process.
2. Validate the theoretical transition distribution numerically.
3. Estimate OU parameters from observed mean-reverting data.
4. Construct and evaluate a threshold-based trading strategy on a synthetic spread.
5. Apply the same methodology to a spread constructed from real asset-price data.

The synthetic backtest provides a controlled setting in which the true parameters are known, while the real-data backtest shows how the methodology can be applied when the underlying parameters must be estimated from observations.

## Package Structure

The Python package `ornstein_uhlenbeck` is split into four modules.

- `ou.py` provides the main machinery required for simulation, transition distributions, and stationary properties.
- `estimation.py` provides a numerical method for estimating the OU parameters from an observed mean-reverting time series.
- `strategy.py` provides trading strategy machinery: functions for threshold construction, positions, PnL calculation, and trade-level statistics.
- `utils.py` provides helper tools used for plotting throughout the project.

The package is separated from the research notes so that the mathematical and computational components can be reused independently of the research notes.

## Research Notes

The research notes document the investigation of the model and its application to pairs trading, progressing from theory through numerical validation to empirical backtesting.

💡 Right click and open in new tab for best viewing.

### 1. OU Theory

👉 [Ornstein-Uhlenbeck theory notebook](https://ronniejp.github.io/ornstein_uhlenbeck/notebooks/ou_theory.html)

Introduces the OU stochastic differential equation and derives its main properties, including the conditional distribution, expectation, variance, and stationary distribution.

### 2. Numerical Validation

👉 [Numerical Validation of OU transition](https://ronniejp.github.io/ornstein_uhlenbeck/notebooks/ou_validation.html)

Checks the theoretical transition distribution against simulated OU paths, providing numerical evidence that the implementation agrees with the mathematical model.

### 3. Parameter Estimation

👉 [Estimation of Model Parameters](https://ronniejp.github.io/ornstein_uhlenbeck/notebooks/ou_estimation.html)

Develops and evaluates the method used to estimate the OU parameters from an observed time series.

### 4. Pairs-Trading Backtest

👉 [Synthetic and Real Spread Backtest](https://ronniejp.github.io/ornstein_uhlenbeck/notebooks/ou_strategy_backtest.html)

Applies the estimated OU model to a threshold-based mean-reversion strategy.

## Project Structure

The repository is organised so that the reusable Python implementation is separated from the Quarto research notes.

```text
ornstein_uhlenbeck/
├── notebooks/
│   ├── ou_estimation.qmd
│   ├── ou_strategy_backtest.qmd
│   ├── ou_theory.qmd
│   └── ou_validation.qmd
├── src/
│   └── ornstein_uhlenbeck/
│       ├── __init__.py
│       ├── estimation.py
│       ├── ou.py
│       ├── strategy.py
│       └── utils.py
└── README.md
```

### Scope

The purpose of the project is to demonstrate the mathematical and computational application of the OU process to pairs trading. It is not intended to constitute a production trading system.

The backtests use simplified assumptions and do not attempt to model all practical considerations such as transaction costs, slippage, market impact, position sizing, or portfolio constraints.

The results should therefore be interpreted as an investigation of the modelling framework and strategy mechanics rather than evidence of a deployable trading strategy.
