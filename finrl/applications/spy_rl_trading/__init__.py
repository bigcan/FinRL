"""SPY RL Trading System - Reinforcement Learning for S&P 500 ETF Trading.

This module implements a complete reinforcement learning trading system for SPY
using PPO (Proximal Policy Optimization) algorithm via Stable-Baselines3.

Components:
    - data_processor: SPY-specific data processing (Yahoo Finance integration)
    - environment: Discrete action trading environment (BUY/HOLD/SELL)
    - agent: PPO agent wrapper with TensorBoard integration
    - pipeline: End-to-end training and backtesting orchestration
    - metrics: Performance analytics (Sharpe ratio, max drawdown, win rate)

Architecture:
    Extends FinRL three-layer architecture:
    - Meta Layer: Reuses data processors and environment base classes
    - Agents Layer: Wraps Stable-Baselines3 PPO implementation
    - Applications Layer: SPY trading-specific logic (this module)

Example:
    >>> from finrl.applications.spy_rl_trading import pipeline
    >>> from finrl.config import SPY_CONFIG, PPO_PARAMS
    >>>
    >>> # Train agent
    >>> trained_model, metrics = pipeline.train_agent(
    ...     config=SPY_CONFIG,
    ...     symbol="SPY",
    ...     total_timesteps=100_000
    ... )
    >>>
    >>> # Backtest on hold-out data
    >>> results = pipeline.backtest_agent(trained_model, test_data)
    >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
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
]
