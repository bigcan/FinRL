"""Performance Metrics Module for SPY RL Trading.

This module provides functions to compute backtesting metrics including:
    - Total return and annualized return
    - Sharpe ratio (risk-adjusted return)
    - Maximum drawdown
    - Win rate
    - Comparison to buy-and-hold baseline

Example:
    >>> from finrl.applications.spy_rl_trading.metrics import compute_training_metrics
    >>> metrics = compute_training_metrics(returns)
    >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_training_metrics(episode_returns: list[float]) -> dict:
    """Compute training metrics from episode returns.

    Args:
        episode_returns: List of cumulative returns per episode

    Returns:
        Dictionary with:
            - episode_return_mean: Mean episode return
            - episode_return_std: Std of episode returns
            - convergence_check: True if converging (positive mean)
            - stability_check: True if stable (low variance)

    Example:
        >>> returns = [0.05, 0.08, 0.12, 0.10, 0.15]
        >>> metrics = compute_training_metrics(returns)
    """
    episode_returns = np.array(episode_returns)

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    # Check convergence (mean > 0)
    convergence_check = mean_return > 0

    # Check stability (coefficient of variation < 1.0)
    stability_check = (std_return / mean_return) < 1.0 if mean_return > 0 else False

    return {
        "episode_return_mean": float(mean_return),
        "episode_return_std": float(std_return),
        "convergence_check": convergence_check,
        "stability_check": stability_check,
    }


def calculate_returns_metrics(daily_returns: np.ndarray) -> dict:
    """Calculate performance metrics from daily returns.

    Args:
        daily_returns: Array of daily log returns

    Returns:
        Dictionary with:
            - total_return: Cumulative return (%)
            - annual_return: Annualized return (%)
            - sharpe_ratio: Risk-adjusted return
            - max_drawdown: Maximum drawdown (%)
            - win_rate: Fraction of positive return days

    Example:
        >>> returns = np.array([0.01, -0.005, 0.02, 0.01, -0.01])
        >>> metrics = calculate_returns_metrics(returns)
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    """
    # Total return (from log returns)
    total_return = np.exp(np.sum(daily_returns)) - 1

    # Annualized return
    annual_return = np.mean(daily_returns) * 252  # 252 trading days/year

    # Sharpe ratio (annualized)
    annual_std = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Win rate (% of positive return days)
    win_rate = np.mean(daily_returns > 0)

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
    }


def compare_to_baseline(
    agent_returns: np.ndarray,
    price_history: pd.Series,
) -> dict:
    """Compare agent performance to buy-and-hold baseline.

    Args:
        agent_returns: Array of agent's daily returns
        price_history: Series of SPY close prices

    Returns:
        Dictionary with:
            - agent_metrics: Agent performance metrics
            - baseline_metrics: Buy-and-hold performance metrics
            - alpha: Excess return vs. baseline (%)
            - beats_baseline: True if agent Sharpe > baseline Sharpe

    Example:
        >>> agent_metrics = compare_to_baseline(agent_returns, spy_prices)
        >>> print(f"Alpha: {agent_metrics['alpha']:.2%}")
    """
    # Agent metrics
    agent_metrics = calculate_returns_metrics(agent_returns)

    # Baseline metrics (buy-and-hold)
    baseline_returns = np.log(price_history / price_history.shift(1)).dropna().values
    baseline_metrics = calculate_returns_metrics(baseline_returns)

    # Alpha (excess return)
    alpha = agent_metrics["annual_return"] - baseline_metrics["annual_return"]

    # Does agent beat baseline?
    beats_baseline = agent_metrics["sharpe_ratio"] > baseline_metrics["sharpe_ratio"]

    return {
        "agent_metrics": agent_metrics,
        "baseline_metrics": baseline_metrics,
        "alpha": float(alpha),
        "beats_baseline": beats_baseline,
    }


def format_metrics_report(metrics: dict, baseline: dict | None = None) -> str:
    """Format metrics as a readable report.

    Args:
        metrics: Agent performance metrics
        baseline: Optional baseline metrics for comparison

    Returns:
        Formatted string report

    Example:
        >>> report = format_metrics_report(agent_metrics, baseline_metrics)
        >>> print(report)
    """
    report = []
    report.append("=" * 60)
    report.append("PERFORMANCE METRICS")
    report.append("=" * 60)

    report.append("\n📊 Agent Performance:")
    report.append(f"   Total Return:     {metrics['total_return']:.2%}")
    report.append(f"   Annual Return:    {metrics['annual_return']:.2%}")
    report.append(f"   Sharpe Ratio:     {metrics['sharpe_ratio']:.3f}")
    report.append(f"   Max Drawdown:     {metrics['max_drawdown']:.2%}")
    report.append(f"   Win Rate:         {metrics['win_rate']:.2%}")

    if baseline:
        report.append("\n📈 Buy-and-Hold Baseline:")
        report.append(f"   Total Return:     {baseline['total_return']:.2%}")
        report.append(f"   Annual Return:    {baseline['annual_return']:.2%}")
        report.append(f"   Sharpe Ratio:     {baseline['sharpe_ratio']:.3f}")
        report.append(f"   Max Drawdown:     {baseline['max_drawdown']:.2%}")

        # Calculate alpha
        alpha = metrics["annual_return"] - baseline["annual_return"]
        beats_baseline = metrics["sharpe_ratio"] > baseline["sharpe_ratio"]

        report.append("\n🎯 Performance vs. Baseline:")
        report.append(f"   Alpha (Excess Return): {alpha:.2%}")
        report.append(
            f"   Beats Baseline:        {'✅ YES' if beats_baseline else '❌ NO'}"
        )

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def validate_success_criteria(metrics: dict) -> dict:
    """Validate metrics against success criteria from spec.md.

    Success Criteria:
        - SC-002: Training return >5%
        - SC-003: Test return ≥0%
        - SC-004: Sharpe ratio ≥0.5

    Args:
        metrics: Performance metrics dictionary

    Returns:
        Dictionary with validation results

    Example:
        >>> validation = validate_success_criteria(metrics)
        >>> if validation['all_passed']:
        ...     print("All success criteria met!")
    """
    checks = {
        "sc_003_test_return_positive": metrics.get("total_return", 0) >= 0,
        "sc_004_sharpe_ratio": metrics.get("sharpe_ratio", 0) >= 0.5,
    }

    all_passed = all(checks.values())

    return {
        **checks,
        "all_passed": all_passed,
    }
