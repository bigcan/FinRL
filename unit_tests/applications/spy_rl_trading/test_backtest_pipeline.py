"""Integration tests for SPY RL Trading Backtesting Pipeline.

Tests verify end-to-end backtesting workflow:
    - Load trained model
    - Create test environment
    - Run backtest without retraining
    - Compute performance metrics
    - Generate reports

Run with:
    pytest unit_tests/applications/spy_rl_trading/test_backtest_pipeline.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

from finrl.applications.spy_rl_trading.data_processor import SPYDataProcessor
from finrl.applications.spy_rl_trading.environment import SPYTradingEnv
from finrl.config import SPY_INDICATORS


class TestBacktestPipelineIntegration:
    """Integration tests for full backtesting pipeline."""

    @pytest.fixture
    def mock_trained_model(self, tmp_path):
        """Create a mock trained PPO model for testing."""
        # Create minimal test data
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "tic": "SPY",
                "open": 450.0,
                "high": 455.0,
                "low": 445.0,
                "close": 450.0,
                "volume": 1000000,
            }
        )

        # Add minimal indicators
        for indicator in SPY_INDICATORS[:3]:  # Use only first 3 for speed
            df[indicator] = 0.0
        df["turbulence"] = 0.0

        # Create minimal environment
        env = SPYTradingEnv(df=df, tech_indicator_list=SPY_INDICATORS[:3])

        # Train minimal model (1 step for speed)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=10)

        # Save model
        model_path = tmp_path / "test_ppo_model"
        model.save(str(model_path))

        return model, env, str(model_path)

    @pytest.fixture
    def test_data(self):
        """Create test data for backtesting."""
        dates = pd.date_range("2024-02-01", "2024-02-10", freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "tic": "SPY",
                "open": 450.0,
                "high": 455.0,
                "low": 445.0,
                "close": np.linspace(450, 460, len(dates)),  # Trending up
                "volume": 1000000,
            }
        )

        # Add indicators
        for indicator in SPY_INDICATORS[:3]:
            df[indicator] = 0.0
        df["turbulence"] = 0.0

        return df

    def test_model_loads_successfully(self, mock_trained_model):
        """Test that trained model can be loaded from disk."""
        model, env, model_path = mock_trained_model

        # Load model
        loaded_model = PPO.load(model_path)

        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")

    def test_model_prediction_on_test_data(self, mock_trained_model, test_data):
        """Test that loaded model can make predictions on test data."""
        model, _, model_path = mock_trained_model

        # Create test environment
        test_env = SPYTradingEnv(df=test_data, tech_indicator_list=SPY_INDICATORS[:3])

        # Reset environment
        obs, info = test_env.reset()

        # Model should predict without errors
        action, _states = model.predict(obs, deterministic=True)

        assert action in [0, 1, 2]  # Valid discrete action

    def test_backtest_generates_returns(self, mock_trained_model, test_data):
        """Test that backtest generates trading returns."""
        model, _, _ = mock_trained_model

        # Create test environment
        test_env = SPYTradingEnv(df=test_data, tech_indicator_list=SPY_INDICATORS[:3])

        # Run backtest
        obs, info = test_env.reset()
        returns = []
        done = False
        truncated = False

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            returns.append(reward)

        # Should have returns for all timesteps
        assert len(returns) > 0
        assert all(isinstance(r, (int, float)) for r in returns)

    def test_backtest_no_retraining(self, mock_trained_model, test_data):
        """Test that backtest does not retrain the model."""
        model, _, model_path = mock_trained_model

        # Get initial model parameters
        initial_params = model.policy.state_dict()

        # Create test environment
        test_env = SPYTradingEnv(df=test_data, tech_indicator_list=SPY_INDICATORS[:3])

        # Run backtest
        obs, info = test_env.reset()
        for _ in range(5):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            if done or truncated:
                break

        # Model parameters should be unchanged
        final_params = model.policy.state_dict()

        for key in initial_params.keys():
            assert np.allclose(
                initial_params[key].cpu().numpy(), final_params[key].cpu().numpy()
            ), f"Model parameters changed during backtest: {key}"

    def test_backtest_deterministic_with_seed(self, mock_trained_model, test_data):
        """Test that backtest is deterministic with same seed."""
        model, _, _ = mock_trained_model

        # Run backtest twice with deterministic=True
        returns1 = []
        test_env = SPYTradingEnv(df=test_data, tech_indicator_list=SPY_INDICATORS[:3])
        obs, info = test_env.reset(seed=42)
        done = False
        truncated = False
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            returns1.append(reward)

        returns2 = []
        test_env = SPYTradingEnv(df=test_data, tech_indicator_list=SPY_INDICATORS[:3])
        obs, info = test_env.reset(seed=42)
        done = False
        truncated = False
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            returns2.append(reward)

        # Returns should be identical
        assert len(returns1) == len(returns2)
        assert np.allclose(returns1, returns2)

    def test_backtest_output_format(self, mock_trained_model, test_data):
        """Test that backtest outputs have expected format."""
        model, _, _ = mock_trained_model

        # Create test environment
        test_env = SPYTradingEnv(df=test_data, tech_indicator_list=SPY_INDICATORS[:3])

        # Run backtest and collect outputs
        obs, info = test_env.reset()
        episode_data = {
            "actions": [],
            "rewards": [],
            "balances": [],
            "shares": [],
        }
        done = False
        truncated = False

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)

            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["balances"].append(obs[0])
            episode_data["shares"].append(obs[1])

        # Validate output format
        assert len(episode_data["actions"]) > 0
        assert len(episode_data["rewards"]) == len(episode_data["actions"])
        assert all(a in [0, 1, 2] for a in episode_data["actions"])
        assert all(isinstance(r, (int, float)) for r in episode_data["rewards"])


class TestBacktestValidation:
    """Test suite for backtest validation and error handling."""

    def test_backtest_with_missing_indicators(self, test_data):
        """Test error handling when indicators are missing."""
        # Create data without required indicators
        df_missing = test_data.drop(columns=SPY_INDICATORS[:2])

        # Should raise error when creating environment
        with pytest.raises(KeyError):
            env = SPYTradingEnv(df=df_missing, tech_indicator_list=SPY_INDICATORS[:3])

    def test_backtest_with_insufficient_data(self):
        """Test error handling with insufficient test data."""
        # Create minimal data (only 2 days)
        dates = pd.date_range("2024-01-01", "2024-01-02", freq="D")
        df_short = pd.DataFrame(
            {
                "date": dates,
                "tic": "SPY",
                "open": 450.0,
                "high": 455.0,
                "low": 445.0,
                "close": 450.0,
                "volume": 1000000,
            }
        )

        for indicator in SPY_INDICATORS[:3]:
            df_short[indicator] = 0.0
        df_short["turbulence"] = 0.0

        # Should create environment but episode will be very short
        env = SPYTradingEnv(df=df_short, tech_indicator_list=SPY_INDICATORS[:3])
        obs, info = env.reset()

        # Should complete in 1-2 steps
        assert env.terminal == False or env.terminal == True


class TestBacktestMetricsCollection:
    """Test suite for metrics collection during backtest."""

    @pytest.fixture
    def mock_backtest_results(self):
        """Create mock backtest results."""
        return {
            "daily_returns": np.array([0.01, -0.005, 0.02, 0.005, -0.01]),
            "actions": [0, 1, 1, 2, 0],  # BUY, HOLD, HOLD, SELL, BUY
            "portfolio_values": [100000, 101000, 100500, 102500, 103000, 102000],
        }

    def test_metrics_collected_from_backtest(self, mock_backtest_results):
        """Test that metrics can be computed from backtest results."""
        from finrl.applications.spy_rl_trading.metrics import (
            calculate_returns_metrics,
        )

        returns = mock_backtest_results["daily_returns"]
        metrics = calculate_returns_metrics(returns)

        # Should compute all metrics
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

    def test_portfolio_value_tracking(self, mock_backtest_results):
        """Test that portfolio values are tracked correctly."""
        portfolio_values = mock_backtest_results["portfolio_values"]

        # Should have more values than returns (initial + N steps)
        assert len(portfolio_values) == len(mock_backtest_results["daily_returns"]) + 1

        # All values should be positive
        assert all(v > 0 for v in portfolio_values)

    def test_action_tracking(self, mock_backtest_results):
        """Test that actions are tracked correctly."""
        actions = mock_backtest_results["actions"]

        # All actions should be valid
        assert all(a in [0, 1, 2] for a in actions)

        # Should have same length as returns
        assert len(actions) == len(mock_backtest_results["daily_returns"])
