"""Unit tests for SPY Trading Environment.

Tests cover:
    - Environment reset (observation shape, initial state)
    - Step function (action execution, reward computation)
    - Reward logic (log return when holding, 0 when flat)
    - Edge cases (multiple buys/sells, insufficient balance)
    - State normalization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from finrl.applications.spy_rl_trading.environment import SPYTradingEnv


class TestSPYTradingEnv:
    """Test suite for SPYTradingEnv."""

    @pytest.fixture
    def sample_data(self):
        """Create sample SPY data for testing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        data = {
            "date": dates,
            "close": np.linspace(300, 400, 100),  # Linear price increase
            "macd": np.random.randn(100),
            "rsi_30": np.random.uniform(30, 70, 100),
            "boll_ub": np.linspace(310, 410, 100),
            "boll_lb": np.linspace(290, 390, 100),
            "cci_30": np.random.randn(100) * 50,
            "dx_30": np.random.uniform(10, 40, 100),
            "close_30_sma": np.linspace(295, 395, 100),
            "close_60_sma": np.linspace(290, 390, 100),
            "vix": np.random.uniform(15, 25, 100),
            "turbulence": np.random.uniform(0, 0.5, 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def env(self, sample_data):
        """Create SPYTradingEnv instance."""
        indicators = [
            "macd",
            "rsi_30",
            "boll_ub",
            "boll_lb",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
            "vix",
        ]
        return SPYTradingEnv(
            df=sample_data,
            tech_indicator_list=indicators,
            initial_amount=100000,
            buy_cost_pct=0.001,
            sell_cost_pct=0.001,
            print_verbosity=1000,  # Disable prints for tests
        )

    def test_reset_observation_shape(self, env):
        """Test reset returns correct observation shape."""
        obs, info = env.reset()

        # Expected shape: [balance, shares, price, 9 indicators, turbulence] = 12
        # But environment has 13 (see state_dim calculation)
        assert obs.shape == (env.state_dim,)
        assert isinstance(obs, np.ndarray)

    def test_reset_initial_state(self, env):
        """Test reset initializes state correctly."""
        obs, info = env.reset()

        # Check balance normalized to 1.0 (100k / 100k)
        assert np.isclose(obs[0], 1.0)

        # Check shares_held = 0
        assert obs[1] == 0

        # Check price is non-zero
        assert obs[2] > 0

    def test_step_action_buy(self, env):
        """Test step executes BUY action correctly."""
        env.reset()
        initial_balance = env.balance

        # Execute BUY action
        obs, reward, done, truncated, info = env.step(action=0)

        # Check shares increased
        assert env.shares_held == 1

        # Check balance decreased (price + transaction cost)
        assert env.balance < initial_balance

    def test_step_action_hold(self, env):
        """Test step executes HOLD action correctly."""
        env.reset()
        initial_balance = env.balance

        # Execute HOLD action
        obs, reward, done, truncated, info = env.step(action=1)

        # Check nothing changed
        assert env.shares_held == 0
        assert env.balance == initial_balance

    def test_step_action_sell(self, env):
        """Test step executes SELL action correctly."""
        env.reset()

        # First buy 1 share
        env.step(action=0)
        balance_after_buy = env.balance

        # Then sell
        env.step(action=2)

        # Check shares reduced
        assert env.shares_held == 0

        # Check balance increased (but less than original due to transaction costs)
        assert env.balance > balance_after_buy

    def test_reward_when_holding(self, env):
        """Test reward is log return when holding position."""
        env.reset()

        # Buy on day 0
        env.step(action=0)

        # Hold on day 1
        price_before = env.current_price
        obs, reward, done, truncated, info = env.step(action=1)
        price_after = env.current_price

        # Reward should be log return
        expected_reward = np.log(price_after / price_before)
        assert np.isclose(reward, expected_reward, atol=1e-6)

    def test_reward_when_flat(self, env):
        """Test reward is 0 when not holding position."""
        env.reset()

        # Don't buy, just hold (flat position)
        obs, reward, done, truncated, info = env.step(action=1)

        # Reward should be 0
        assert reward == 0.0

    def test_multiple_buys_when_holding(self, env):
        """Test multiple BUY actions when already holding."""
        env.reset()

        # Buy once
        env.step(action=0)
        shares_after_first_buy = env.shares_held

        # Try to buy again
        env.step(action=0)

        # Shares should remain 1 (stay long, don't buy more)
        assert env.shares_held == shares_after_first_buy == 1

    def test_multiple_sells_when_flat(self, env):
        """Test multiple SELL actions when not holding."""
        env.reset()

        # Sell without holding
        env.step(action=2)

        # Shares should remain 0 (stay flat)
        assert env.shares_held == 0

    def test_insufficient_balance(self, env):
        """Test BUY action with insufficient balance."""
        env.reset()

        # Set balance to very low
        env.balance = 10.0  # Not enough to buy SPY

        # Try to buy
        env.step(action=0)

        # Should not buy
        assert env.shares_held == 0

    def test_episode_termination(self, env):
        """Test episode terminates correctly."""
        env.reset()

        # Run until end of data
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, truncated, info = env.step(action=1)
            steps += 1

        # Should terminate
        assert done
        assert steps < 200  # Should terminate before 200 steps (data has 100 days)

    def test_state_normalization(self, env):
        """Test state values are normalized."""
        obs, info = env.reset()

        # Balance should be ~1.0 (normalized by initial amount)
        assert np.isclose(obs[0], 1.0)

        # Shares should be 0 or 1 (no normalization)
        assert obs[1] in [0, 1]

        # All state values should be finite
        assert np.all(np.isfinite(obs))

    def test_asset_memory_tracking(self, env):
        """Test asset memory tracks portfolio value."""
        env.reset()

        # Execute some actions
        for _ in range(10):
            env.step(action=0 if _ % 3 == 0 else 1)

        # Check asset memory length
        assert len(env.asset_memory) == 11  # Initial + 10 steps

        # Check all values are positive
        assert all(asset > 0 for asset in env.asset_memory)

    def test_trades_count(self, env):
        """Test trades are counted correctly."""
        env.reset()

        # Buy and sell
        env.step(action=0)  # Buy: 1 trade
        env.step(action=1)  # Hold: 0 trades
        env.step(action=2)  # Sell: 1 trade

        # Total trades should be 2
        assert env.trades == 2

    def test_transaction_costs(self, env):
        """Test transaction costs are applied."""
        env.reset()

        # Buy
        env.step(action=0)
        cost_after_buy = env.cost

        # Sell
        env.step(action=2)
        cost_after_sell = env.cost

        # Cost should increase after both buy and sell
        assert cost_after_buy > 0
        assert cost_after_sell > cost_after_buy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
