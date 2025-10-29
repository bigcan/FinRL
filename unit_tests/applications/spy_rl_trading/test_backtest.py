"""Unit tests for SPY RL Trading Backtest Engine.

Tests verify:
    - Backtest execution without retraining
    - Metrics computation (Sharpe, drawdown, win rate)
    - Baseline comparison
    - Deterministic behavior with fixed seed
    - Edge case handling

Run with:
    pytest unit_tests/applications/spy_rl_trading/test_backtest.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.applications.spy_rl_trading.metrics import (
    calculate_returns_metrics,
    compare_to_baseline,
)


class TestBacktestMetrics:
    """Test suite for backtest metrics computation."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation with known returns."""
        # Known returns: mean=0.001, std=0.02, annualized Sharpe ≈ 0.79
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.005] * 50)
        metrics = calculate_returns_metrics(daily_returns)

        assert "sharpe_ratio" in metrics
        assert isinstance(metrics["sharpe_ratio"], float)
        assert -5 < metrics["sharpe_ratio"] < 5  # Reasonable range

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Scenario: gain 10%, then lose 20%, then recover 5%
        daily_returns = np.array([0.1, -0.2, 0.05])
        metrics = calculate_returns_metrics(daily_returns)

        assert "max_drawdown" in metrics
        assert metrics["max_drawdown"] < 0  # Drawdown should be negative
        assert metrics["max_drawdown"] >= -1  # Cannot exceed -100%

    def test_win_rate_calculation(self):
        """Test win rate calculation (% positive days)."""
        # 60% win rate: 3 positive, 2 negative
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.005])
        metrics = calculate_returns_metrics(daily_returns)

        assert "win_rate" in metrics
        assert 0 <= metrics["win_rate"] <= 1
        assert metrics["win_rate"] == 0.6  # 3/5 positive days

    def test_total_return_calculation(self):
        """Test cumulative return calculation from log returns."""
        # Log returns: [0.01, 0.01, 0.01] → total ≈ 3.04%
        daily_returns = np.array([0.01, 0.01, 0.01])
        metrics = calculate_returns_metrics(daily_returns)

        assert "total_return" in metrics
        expected_return = np.exp(0.03) - 1  # Sum of log returns
        assert np.isclose(metrics["total_return"], expected_return, atol=1e-6)

    def test_annual_return_calculation(self):
        """Test annualized return calculation."""
        # Mean daily return 0.001 → annual ≈ 0.252 (25.2%)
        daily_returns = np.array([0.001] * 252)
        metrics = calculate_returns_metrics(daily_returns)

        assert "annual_return" in metrics
        expected_annual = 0.001 * 252
        assert np.isclose(metrics["annual_return"], expected_annual, atol=1e-6)

    def test_baseline_comparison(self):
        """Test agent vs. buy-and-hold baseline comparison."""
        # Create synthetic agent returns and price history
        agent_returns = np.array([0.01, 0.02, 0.005, -0.01, 0.015])
        price_history = pd.Series([100, 101, 103, 102, 101, 102])

        comparison = compare_to_baseline(agent_returns, price_history)

        assert "agent_metrics" in comparison
        assert "baseline_metrics" in comparison
        assert "alpha" in comparison
        assert "beats_baseline" in comparison
        assert isinstance(comparison["beats_baseline"], bool)

    def test_alpha_calculation(self):
        """Test alpha (excess return) calculation."""
        # Agent: 10% annual return, Baseline: 5% annual return → alpha = 5%
        agent_returns = np.array([0.01] * 10)  # 10% total
        price_history = pd.Series([100] + [100 * (1.005 ** i) for i in range(1, 11)])

        comparison = compare_to_baseline(agent_returns, price_history)
        alpha = comparison["alpha"]

        assert isinstance(alpha, float)
        # Agent should outperform baseline
        assert alpha > 0

    def test_edge_case_zero_returns(self):
        """Test metrics with zero returns (no trading)."""
        daily_returns = np.zeros(100)
        metrics = calculate_returns_metrics(daily_returns)

        assert metrics["total_return"] == 0
        assert metrics["sharpe_ratio"] == 0  # Zero std → zero Sharpe
        assert metrics["win_rate"] == 0  # No positive days

    def test_edge_case_all_positive_returns(self):
        """Test metrics with 100% win rate."""
        daily_returns = np.array([0.01] * 100)
        metrics = calculate_returns_metrics(daily_returns)

        assert metrics["win_rate"] == 1.0
        assert metrics["max_drawdown"] == 0  # No drawdown
        assert metrics["sharpe_ratio"] > 0

    def test_edge_case_all_negative_returns(self):
        """Test metrics with 0% win rate."""
        daily_returns = np.array([-0.01] * 100)
        metrics = calculate_returns_metrics(daily_returns)

        assert metrics["win_rate"] == 0.0
        assert metrics["max_drawdown"] < 0
        assert metrics["sharpe_ratio"] < 0


class TestBacktestDeterminism:
    """Test suite for deterministic backtest behavior."""

    def test_deterministic_metrics_same_seed(self):
        """Test that metrics are deterministic with same input."""
        daily_returns = np.random.RandomState(42).randn(100) * 0.01

        # Compute metrics twice
        metrics1 = calculate_returns_metrics(daily_returns)
        metrics2 = calculate_returns_metrics(daily_returns)

        # Should be identical
        assert metrics1 == metrics2

    def test_baseline_comparison_deterministic(self):
        """Test baseline comparison is deterministic."""
        np.random.seed(42)
        agent_returns = np.random.randn(100) * 0.01
        price_history = pd.Series(100 * np.cumprod(1 + np.random.randn(101) * 0.01))

        # Run twice
        comparison1 = compare_to_baseline(agent_returns, price_history)
        comparison2 = compare_to_baseline(agent_returns, price_history)

        assert comparison1["alpha"] == comparison2["alpha"]
        assert comparison1["beats_baseline"] == comparison2["beats_baseline"]


class TestBacktestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_single_day_returns(self):
        """Test metrics with single trading day."""
        daily_returns = np.array([0.01])
        metrics = calculate_returns_metrics(daily_returns)

        # Should handle gracefully
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics

    def test_empty_returns_raises_error(self):
        """Test that empty returns raise appropriate error."""
        with pytest.raises((ValueError, IndexError)):
            calculate_returns_metrics(np.array([]))

    def test_nan_returns_handling(self):
        """Test behavior with NaN returns."""
        daily_returns = np.array([0.01, np.nan, 0.02])

        # Should either handle gracefully or raise informative error
        try:
            metrics = calculate_returns_metrics(daily_returns)
            # If handled, should not have NaN in metrics
            assert not np.isnan(metrics["total_return"])
        except (ValueError, RuntimeWarning):
            # Or raise informative error
            pass

    def test_extreme_returns(self):
        """Test metrics with extreme returns."""
        # Extreme returns: +50%, -50%, +100%
        daily_returns = np.array([0.5, -0.5, 1.0])
        metrics = calculate_returns_metrics(daily_returns)

        # Should compute without errors
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert np.isfinite(metrics["total_return"])


class TestBacktestValidation:
    """Test suite for validation checks."""

    def test_metrics_dict_structure(self):
        """Test that metrics dict has expected structure."""
        daily_returns = np.array([0.01, -0.005, 0.02])
        metrics = calculate_returns_metrics(daily_returns)

        # Required keys
        required_keys = [
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
        ]
        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_comparison_dict_structure(self):
        """Test that comparison dict has expected structure."""
        agent_returns = np.array([0.01, 0.02, 0.005])
        price_history = pd.Series([100, 101, 103, 104])

        comparison = compare_to_baseline(agent_returns, price_history)

        # Required keys
        required_keys = [
            "agent_metrics",
            "baseline_metrics",
            "alpha",
            "beats_baseline",
        ]
        for key in required_keys:
            assert key in comparison

        # Nested structure
        assert isinstance(comparison["agent_metrics"], dict)
        assert isinstance(comparison["baseline_metrics"], dict)
        assert isinstance(comparison["alpha"], float)
        assert isinstance(comparison["beats_baseline"], bool)

    def test_metrics_value_ranges(self):
        """Test that metric values are in reasonable ranges."""
        daily_returns = np.random.RandomState(42).randn(252) * 0.01
        metrics = calculate_returns_metrics(daily_returns)

        # Sharpe ratio typically -3 to 3 for daily returns
        assert -10 < metrics["sharpe_ratio"] < 10

        # Win rate between 0 and 1
        assert 0 <= metrics["win_rate"] <= 1

        # Max drawdown between -1 and 0
        assert -1 <= metrics["max_drawdown"] <= 0

        # Annual return reasonable (not 1000%)
        assert -5 < metrics["annual_return"] < 5
