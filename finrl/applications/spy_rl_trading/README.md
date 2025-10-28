# SPY RL Trading System

A comprehensive reinforcement learning trading system for SPY (S&P 500 ETF) built with FinRL framework and PPO agent.

## Quick Start

**Train a PPO agent in 5 minutes**:

```bash
# Install dependencies
poetry install
poetry shell

# Download SPY data and train agent
python -m finrl.applications.spy_rl_trading.pipeline train

# Backtest on 2025 hold-out data
python -m finrl.applications.spy_rl_trading.pipeline backtest
```

See [quickstart.md](../../specs/1-spy-rl-trading/quickstart.md) for detailed tutorial.

## Features

- **Data Processing**: SPY daily OHLCV data (2020-2025) with 10 technical indicators + VIX
- **Trading Environment**: Gymnasium-compliant discrete action space (BUY/HOLD/SELL)
- **PPO Agent**: Stable-Baselines3 implementation with TensorBoard monitoring
- **Backtesting**: Risk-adjusted performance metrics (Sharpe, max drawdown, win rate)
- **Hyperparameter Tuning**: Grid search over PPO configurations

## Architecture

Integrates with FinRL three-layer stack:

```
┌─────────────────────────────────────────────────┐
│  Applications Layer (this module)              │
│  - SPY trading pipeline                         │
│  - Performance metrics                          │
│  - Hyperparameter tuning                        │
├─────────────────────────────────────────────────┤
│  Agents Layer (FinRL SB3 integration)          │
│  - PPO agent wrapper                            │
│  - TensorBoard callbacks                        │
│  - Model save/load utilities                    │
├─────────────────────────────────────────────────┤
│  Meta Layer (FinRL core)                       │
│  - Yahoo Finance data processor                 │
│  - Trading environment base class               │
│  - Technical indicator computation              │
└─────────────────────────────────────────────────┘
```

## User Stories

### US1: Training and Validation (P1 - MVP)
Train PPO agent on 2020-2024 SPY data, observe converging rewards, validate profitability.

### US2: Backtesting on Hold-Out Data (P2)
Evaluate trained agent on 2025 test data, compute Sharpe ratio and compare to buy-and-hold baseline.

### US3: Hyperparameter Optimization (P3)
Grid search over PPO hyperparameters, identify optimal configuration for maximum Sharpe ratio.

## Documentation

- [Quickstart Guide](../../specs/1-spy-rl-trading/quickstart.md): Step-by-step tutorial
- [API Reference](../../specs/1-spy-rl-trading/contracts/API.md): Module specifications
- [Data Model](../../specs/1-spy-rl-trading/data-model.md): Entity definitions
- [Research](../../specs/1-spy-rl-trading/research.md): Technical decisions

## Success Criteria

- SC-001: Load SPY data for 2020-2025 without errors
- SC-002: Agent achieves >5% cumulative return on training data
- SC-003: Agent delivers ≥0% return on 2025 hold-out data
- SC-004: Sharpe ratio ≥ 0.5 and exceeds buy-and-hold baseline
- SC-010: Complete runnable quickstart tutorial provided

## License

Same as FinRL framework (MIT License)
