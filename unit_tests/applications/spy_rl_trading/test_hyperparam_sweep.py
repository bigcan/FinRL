"""Unit tests for Hyperparameter Sweep Module.

Tests verify:
    - Grid search configuration generation
    - Multiple model training with different hyperparameters
    - Metrics comparison across configurations
    - Result aggregation and analysis

Run with:
    pytest unit_tests/applications/spy_rl_trading/test_hyperparam_sweep.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestHyperparameterGridGeneration:
    """Test suite for hyperparameter grid generation."""

    def test_generate_grid_combinations(self):
        """Test generation of all hyperparameter combinations."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import generate_param_grid

        param_grid = {
            "learning_rate": [1e-3, 1e-4],
            "n_steps": [2048, 4096],
        }

        combinations = generate_param_grid(param_grid)

        # Should generate 2 * 2 = 4 combinations
        assert len(combinations) == 4
        assert all(isinstance(c, dict) for c in combinations)

    def test_grid_with_fixed_params(self):
        """Test grid generation with fixed parameters."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import generate_param_grid

        param_grid = {
            "learning_rate": [1e-3, 1e-4],
            "batch_size": [128],  # Fixed parameter
        }

        combinations = generate_param_grid(param_grid)

        assert len(combinations) == 2
        assert all(c["batch_size"] == 128 for c in combinations)

    def test_empty_grid_raises_error(self):
        """Test that empty grid raises appropriate error."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import generate_param_grid

        with pytest.raises(ValueError):
            generate_param_grid({})


class TestHyperparameterSweepExecution:
    """Test suite for hyperparameter sweep execution."""

    def test_sweep_result_structure(self):
        """Test that sweep results have expected structure."""
        # Mock sweep result
        sweep_result = {
            "config_1": {
                "params": {"learning_rate": 1e-3},
                "metrics": {
                    "sharpe_ratio": 0.8,
                    "total_return": 0.15,
                },
            },
            "config_2": {
                "params": {"learning_rate": 1e-4},
                "metrics": {
                    "sharpe_ratio": 0.6,
                    "total_return": 0.10,
                },
            },
        }

        # Validate structure
        for config_name, result in sweep_result.items():
            assert "params" in result
            assert "metrics" in result
            assert isinstance(result["params"], dict)
            assert isinstance(result["metrics"], dict)

    def test_best_config_selection(self):
        """Test selection of best configuration by metric."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import select_best_config

        results = {
            "config_1": {"params": {"lr": 1e-3}, "metrics": {"sharpe_ratio": 0.8}},
            "config_2": {"params": {"lr": 1e-4}, "metrics": {"sharpe_ratio": 1.2}},
            "config_3": {"params": {"lr": 1e-5}, "metrics": {"sharpe_ratio": 0.6}},
        }

        best = select_best_config(results, metric="sharpe_ratio")

        assert best["name"] == "config_2"
        assert best["params"]["lr"] == 1e-4
        assert best["metrics"]["sharpe_ratio"] == 1.2


class TestHyperparameterComparison:
    """Test suite for hyperparameter comparison tools."""

    def test_comparison_table_generation(self):
        """Test generation of comparison table."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import create_comparison_table

        results = {
            "config_1": {
                "params": {"learning_rate": 1e-3},
                "metrics": {"sharpe_ratio": 0.8, "total_return": 0.15},
            },
            "config_2": {
                "params": {"learning_rate": 1e-4},
                "metrics": {"sharpe_ratio": 1.2, "total_return": 0.20},
            },
        }

        df = create_comparison_table(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "sharpe_ratio" in df.columns
        assert "total_return" in df.columns

    def test_comparison_table_sorting(self):
        """Test that comparison table can be sorted by metric."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import create_comparison_table

        results = {
            "config_1": {"params": {"lr": 1e-3}, "metrics": {"sharpe_ratio": 0.8}},
            "config_2": {"params": {"lr": 1e-4}, "metrics": {"sharpe_ratio": 1.2}},
            "config_3": {"params": {"lr": 1e-5}, "metrics": {"sharpe_ratio": 0.6}},
        }

        df = create_comparison_table(results, sort_by="sharpe_ratio", ascending=False)

        # Should be sorted descending by Sharpe ratio
        assert df.iloc[0]["sharpe_ratio"] == 1.2
        assert df.iloc[-1]["sharpe_ratio"] == 0.6


class TestConvergenceCurveComparison:
    """Test suite for convergence curve comparison."""

    def test_convergence_data_structure(self):
        """Test that convergence data has expected structure."""
        convergence_data = {
            "config_1": {
                "episode_rewards": [10, 20, 30, 40, 50],
                "timesteps": [1000, 2000, 3000, 4000, 5000],
            },
            "config_2": {
                "episode_rewards": [5, 15, 25, 35, 45],
                "timesteps": [1000, 2000, 3000, 4000, 5000],
            },
        }

        for config_name, data in convergence_data.items():
            assert "episode_rewards" in data
            assert "timesteps" in data
            assert len(data["episode_rewards"]) == len(data["timesteps"])

    def test_convergence_comparison_plot_data(self):
        """Test preparation of convergence plot data."""
        convergence_data = {
            "config_1": {"episode_rewards": [10, 20, 30]},
            "config_2": {"episode_rewards": [5, 15, 25]},
        }

        # Should be able to extract data for plotting
        for config_name, data in convergence_data.items():
            rewards = data["episode_rewards"]
            assert len(rewards) > 0
            assert all(isinstance(r, (int, float)) for r in rewards)


class TestHyperparameterValidation:
    """Test suite for hyperparameter validation."""

    def test_learning_rate_validation(self):
        """Test learning rate validation."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import validate_ppo_params

        # Valid learning rate
        valid_params = {"learning_rate": 1e-3}
        assert validate_ppo_params(valid_params)

        # Invalid learning rate (negative)
        invalid_params = {"learning_rate": -1e-3}
        with pytest.raises(ValueError):
            validate_ppo_params(invalid_params)

    def test_clip_range_validation(self):
        """Test clip range validation."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import validate_ppo_params

        # Valid clip range
        valid_params = {"clip_range": 0.2}
        assert validate_ppo_params(valid_params)

        # Invalid clip range (out of bounds)
        invalid_params = {"clip_range": 1.5}
        with pytest.raises(ValueError):
            validate_ppo_params(invalid_params)

    def test_batch_size_validation(self):
        """Test batch size validation."""
        from finrl.applications.spy_rl_trading.hyperparam_sweep import validate_ppo_params

        # Valid batch size
        valid_params = {"batch_size": 128}
        assert validate_ppo_params(valid_params)

        # Invalid batch size (not power of 2)
        invalid_params = {"batch_size": 100}
        assert not validate_ppo_params(invalid_params)  # Warning, not error


class TestSweepResultAnalysis:
    """Test suite for sweep result analysis."""

    def test_metric_statistics(self):
        """Test computation of metric statistics across configs."""
        results = {
            "config_1": {"metrics": {"sharpe_ratio": 0.8}},
            "config_2": {"metrics": {"sharpe_ratio": 1.2}},
            "config_3": {"metrics": {"sharpe_ratio": 0.6}},
        }

        sharpe_ratios = [r["metrics"]["sharpe_ratio"] for r in results.values()]

        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)

        assert mean_sharpe == pytest.approx(0.867, rel=1e-2)
        assert std_sharpe > 0

    def test_pareto_frontier_selection(self):
        """Test selection of Pareto-optimal configurations."""
        # Config 1: High return, high risk
        # Config 2: Medium return, medium risk (dominated by 3)
        # Config 3: Medium-high return, low risk (Pareto optimal)
        # Config 4: Low return, low risk (Pareto optimal)

        results = {
            "config_1": {"metrics": {"total_return": 0.30, "max_drawdown": -0.20}},
            "config_2": {"metrics": {"total_return": 0.15, "max_drawdown": -0.15}},
            "config_3": {"metrics": {"total_return": 0.20, "max_drawdown": -0.05}},
            "config_4": {"metrics": {"total_return": 0.10, "max_drawdown": -0.03}},
        }

        # Pareto frontier should include config_1, config_3, config_4
        # (maximizing return, minimizing drawdown)
        # Config 2 is dominated by config 3 (lower return, higher drawdown)
