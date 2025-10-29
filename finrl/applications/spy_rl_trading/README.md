# SPY RL Trading System

A comprehensive reinforcement learning trading system for SPY (S&P 500 ETF) built with FinRL framework and PPO agent.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 116+](https://img.shields.io/badge/tests-116%2B-green.svg)]()

## 🚀 Quick Start

**Train and backtest a PPO agent**:

```bash
# Install dependencies
poetry install && poetry shell

# Run training example (Phase 3)
python -m finrl.applications.spy_rl_trading.example_training

# Run backtesting example (Phase 4)
python -m finrl.applications.spy_rl_trading.example_backtesting

# Run hyperparameter tuning (Phase 5)
python -m finrl.applications.spy_rl_trading.example_hyperparam_tuning
```

See [GUIDE.md](GUIDE.md) for comprehensive tutorial.

## ✨ Features

- ✅ **Complete Training Pipeline** - Train PPO agents on historical SPY data (2020-2025)
- ✅ **Comprehensive Backtesting** - Evaluate performance with lookahead-bias prevention
- ✅ **Hyperparameter Tuning** - Grid search with automated analysis and visualization
- ✅ **Advanced Reporting** - Equity curves, drawdown charts, HTML reports
- ✅ **Baseline Comparison** - Compare agent vs. buy-and-hold strategy
- ✅ **Statistical Analysis** - Multiple runs with aggregate statistics
- ✅ **116+ Unit Tests** - Comprehensive test coverage
- ✅ **FinRL Compliant** - Follows FinRL three-layer architecture
- ✅ **Production Ready** - Logging, validation, error handling

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

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [GUIDE.md](GUIDE.md) | **Comprehensive user guide** with installation, configuration, and usage |
| [API.md](API.md) | **Complete API reference** for all modules and functions |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | **Development progress** and phase completion status |
| [Quickstart Guide](../../specs/1-spy-rl-trading/quickstart.md) | Step-by-step tutorial |
| [Technical Research](../../specs/1-spy-rl-trading/research.md) | Design decisions and hyperparameter choices |

## 🧪 Testing

```bash
# Run all SPY RL tests
pytest unit_tests/applications/spy_rl_trading/ -v

# Run with coverage
pytest unit_tests/applications/spy_rl_trading/ --cov --cov-report=html

# Run specific test module
pytest unit_tests/applications/spy_rl_trading/test_backtest.py -v
```

**Test Coverage**: 116+ unit and integration tests

## 📊 Performance Metrics

The system computes comprehensive metrics:

- **Returns**: Total return, annual return, alpha (excess return)
- **Risk-Adjusted**: Sharpe ratio (risk-adjusted return)
- **Risk**: Maximum drawdown, volatility
- **Trading**: Win rate, action distribution
- **Comparison**: Agent vs. buy-and-hold baseline

## 🎯 Success Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| **SC-001** | Load SPY data (2020-2025) without errors | ✅ |
| **SC-002** | Agent achieves >5% return on training data | ✅ |
| **SC-003** | Agent delivers ≥0% return on test data | ✅ |
| **SC-004** | Sharpe ratio ≥0.5, beats buy-and-hold | ✅ |
| **SC-006** | Model reproducibility with seeds | ✅ |
| **SC-010** | Complete runnable examples | ✅ |

## 🗂️ Project Structure

```
finrl/applications/spy_rl_trading/
├── README.md                       # This file
├── GUIDE.md                        # Comprehensive user guide
├── API.md                          # API documentation
├── IMPLEMENTATION_STATUS.md        # Development progress
│
├── data_processor.py               # SPY data processing
├── environment.py                  # Trading environment
├── agent.py                        # PPO agent wrapper
├── pipeline.py                     # Training orchestration
├── backtest.py                     # Backtesting engine
├── report.py                       # Visualization & reporting
├── metrics.py                      # Performance metrics
├── hyperparam_sweep.py             # Hyperparameter tuning
├── hyperparam_analysis.py          # Analysis tools
│
├── example_training.py             # Training example
├── example_backtesting.py          # Backtesting example
├── example_hyperparam_tuning.py    # Tuning example
│
└── config.py                       # Local configuration

unit_tests/applications/spy_rl_trading/
├── test_data_processor.py          # Data tests (10 cases)
├── test_environment.py             # Environment tests (13 cases)
├── test_backtest.py                # Backtest tests (93 cases)
├── test_backtest_pipeline.py       # Integration tests (23 cases)
└── test_hyperparam_sweep.py        # Hyperparameter tests
```

## 🤝 Contributing

This module follows FinRL Constitution principles:

1. **Three-Layer Architecture** - Maintains separation of concerns
2. **Test-First Development** - ≥80% test coverage required
3. **Gymnasium Compliance** - Standard RL environment interface
4. **Documentation** - Docstrings for all public functions

## 📄 License

MIT License - Same as FinRL framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/AI4Finance-Foundation/FinRL/issues)
- **Documentation**: [FinRL Docs](https://finrl.readthedocs.io/)
- **Community**: [FinRL Slack](https://join.slack.com/t/ai4financeworkspace/shared_invite/)

## 🎉 Acknowledgments

Built with FinRL, Stable-Baselines3, Gymnasium, and love for quantitative finance.

---

**Version**: 1.0.0
**Last Updated**: 2025-10-29
**Status**: Production Ready
**Phases Complete**: 1-6 (Full Implementation)
