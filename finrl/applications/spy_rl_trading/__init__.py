"""SPY RL Trading System - Reinforcement Learning for S&P 500 ETF Trading.

This module implements a complete reinforcement learning trading system for SPY
using PPO (Proximal Policy Optimization) algorithm via Stable-Baselines3.

Components:
    - data_processor: SPY-specific data processing (Yahoo Finance integration)
    - environment: Discrete action trading environment (BUY/HOLD/SELL)
    - agent: PPO agent wrapper with TensorBoard integration
    - pipeline: End-to-end training orchestration
    - backtest: Comprehensive backtesting engine with performance metrics
    - metrics: Performance analytics (Sharpe ratio, max drawdown, win rate)
    - report: Visualization and reporting tools

Architecture:
    Extends FinRL three-layer architecture:
    - Meta Layer: Reuses data processors and environment base classes
    - Agents Layer: Wraps Stable-Baselines3 PPO implementation
    - Applications Layer: SPY trading-specific logic (this module)

Example (Training):
    >>> from finrl.applications.spy_rl_trading import pipeline
    >>> from finrl.config import SPY_CONFIG, PPO_PARAMS
    >>>
    >>> # Train agent
    >>> trained_model, metrics = pipeline.train_agent(
    ...     config=SPY_CONFIG,
    ...     symbol="SPY",
    ...     total_timesteps=100_000
    ... )

Example (Backtesting):
    >>> from finrl.applications.spy_rl_trading.backtest import backtest_agent
    >>> from finrl.applications.spy_rl_trading.report import BacktestReporter
    >>>
    >>> # Run backtest
    >>> result = backtest_agent(
    ...     model_path="trained_models/spy_ppo",
    ...     test_env=test_env,
    ...     price_history=spy_prices,
    ...     output_dir="results/backtest"
    ... )
    >>>
    >>> # Generate report
    >>> reporter = BacktestReporter(result)
    >>> reporter.generate_html_report("reports/backtest.html")
"""

__version__ = "1.0.0"
__author__ = "FinRL Contributors"
__all__ = [
    "data_processor",
    "environment",
    "agent",
    "pipeline",
    "metrics",
    "backtest",
    "report",
    "hyperparam_sweep",
    "hyperparam_analysis",
]
