"""SPY Trading Environment - Discrete Action Trading for SPY.

This module implements a Gymnasium-compliant trading environment specifically
designed for SPY (S&P 500 ETF) with discrete actions (BUY/HOLD/SELL).

Key Features:
    - Discrete action space: {0: BUY, 1: HOLD, 2: SELL}
    - Log-return reward (only when holding position)
    - Single-unit position sizing (position ∈ {0, 1})
    - Normalized state space for stable PPO training

Example:
    >>> from finrl.applications.spy_rl_trading.environment import SPYTradingEnv
    >>> from finrl.applications.spy_rl_trading.data_processor import SPYDataProcessor
    >>>
    >>> processor = SPYDataProcessor()
    >>> df = processor.download_data("2020-01-01", "2024-12-31")
    >>> df = processor.clean_data(df)
    >>> df = processor.add_technical_indicator(df, indicators)
    >>> df = processor.add_vix(df)
    >>>
    >>> env = SPYTradingEnv(df=df, tech_indicator_list=indicators)
    >>> obs, info = env.reset()
    >>> obs, reward, done, truncated, info = env.step(action=0)  # BUY
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SPYTradingEnv(gym.Env):
    """Discrete action trading environment for SPY.

    State Space (13 dimensions):
        [0]: balance (normalized by initial_amount)
        [1]: shares_held (0 or 1)
        [2]: price (current close price, normalized)
        [3-11]: technical indicators (normalized)
        [12]: turbulence index (normalized)

    Action Space (Discrete):
        0: BUY - Purchase 1 share if not holding
        1: HOLD - Maintain current position
        2: SELL - Sell 1 share if holding

    Reward:
        - Log return: log(price_t / price_{t-1}) if holding
        - 0 if flat (not holding)

    Termination:
        - Episode ends when reaching last day in dataset
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list: list[str],
        initial_amount: int = 100000,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        hmax: int = 1,  # Max 1 share (discrete)
        reward_scaling: float = 1.0,
        print_verbosity: int = 10,
    ):
        """Initialize SPY trading environment.

        Args:
            df: DataFrame with OHLCV + indicators + turbulence
            tech_indicator_list: List of technical indicator column names
            initial_amount: Starting cash balance (default: $100,000)
            buy_cost_pct: Transaction cost for buying (default: 0.1%)
            sell_cost_pct: Transaction cost for selling (default: 0.1%)
            hmax: Maximum shares per action (default: 1 for discrete)
            reward_scaling: Scaling factor for rewards (default: 1.0)
            print_verbosity: Print frequency for episode stats (default: 10)
        """
        self.df = df
        self.tech_indicator_list = tech_indicator_list
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.hmax = hmax
        self.reward_scaling = reward_scaling
        self.print_verbosity = print_verbosity

        # Action space: Discrete(3) = {0: BUY, 1: HOLD, 2: SELL}
        self.action_space = spaces.Discrete(3)

        # Observation space: [balance, shares, price, indicators..., turbulence]
        self.state_dim = 2 + 1 + len(tech_indicator_list) + 1  # 13 for SPY
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Episode tracking
        self.day = 0
        self.terminal = False
        self.episode = 0

        # State variables
        self.balance = self.initial_amount
        self.shares_held = 0
        self.current_price = 0

        # Memory for tracking
        self.asset_memory = []  # Total portfolio value over time
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = []

        # Cost tracking
        self.cost = 0
        self.trades = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial state observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset to day 0
        self.day = 0
        self.terminal = False
        self.balance = self.initial_amount
        self.shares_held = 0

        # Get initial price
        self.current_price = self.df.loc[self.day, "close"]

        # Reset tracking
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.df.loc[self.day, "date"]]
        self.cost = 0
        self.trades = 0

        # Get initial state
        state = self._get_state()

        return state, {}

    def step(self, action: int):
        """Execute one timestep of the environment.

        Args:
            action: Action to take (0: BUY, 1: HOLD, 2: SELL)

        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended
            truncated: Whether episode was truncated (always False)
            info: Additional information
        """
        self.terminal = self.day >= len(self.df) - 2

        if self.terminal:
            # Episode ended
            return (
                self._get_state(),
                0.0,
                True,
                False,
                {
                    "total_asset": self.balance
                    + self.shares_held * self.current_price,
                    "trades": self.trades,
                    "cost": self.cost,
                },
            )

        # Execute action
        self._execute_action(action)

        # Move to next day
        self.day += 1
        prev_price = self.current_price
        self.current_price = self.df.loc[self.day, "close"]

        # Calculate reward (log return if holding, else 0)
        reward = self._calculate_reward(prev_price, self.current_price)

        # Get next state
        next_state = self._get_state()

        # Track portfolio value
        total_asset = self.balance + self.shares_held * self.current_price
        self.asset_memory.append(total_asset)
        self.rewards_memory.append(reward)
        self.actions_memory.append(action)
        self.date_memory.append(self.df.loc[self.day, "date"])

        # Print progress
        if self.day % self.print_verbosity == 0:
            print(
                f"Day {self.day}: Balance=${self.balance:.2f}, "
                f"Shares={self.shares_held}, Price=${self.current_price:.2f}, "
                f"Total=${total_asset:.2f}, Reward={reward:.6f}"
            )

        return next_state, reward, False, False, {}

    def _execute_action(self, action: int):
        """Execute trading action.

        Args:
            action: 0 (BUY), 1 (HOLD), or 2 (SELL)
        """
        if action == 0:  # BUY
            if self.shares_held == 0:
                # Buy 1 share if we have enough balance
                cost_per_share = self.current_price * (1 + self.buy_cost_pct)
                if self.balance >= cost_per_share:
                    self.balance -= cost_per_share
                    self.shares_held = 1
                    self.cost += self.current_price * self.buy_cost_pct
                    self.trades += 1
            # If already holding, do nothing (stay long)

        elif action == 2:  # SELL
            if self.shares_held > 0:
                # Sell 1 share
                proceeds = self.current_price * (1 - self.sell_cost_pct)
                self.balance += proceeds
                self.shares_held = 0
                self.cost += self.current_price * self.sell_cost_pct
                self.trades += 1
            # If not holding, do nothing (stay flat)

        # action == 1 (HOLD): Do nothing

    def _calculate_reward(self, prev_price: float, current_price: float) -> float:
        """Calculate reward for current step.

        Reward = log(price_t / price_{t-1}) if holding, else 0.

        Args:
            prev_price: Price at t-1
            current_price: Price at t

        Returns:
            Reward (scaled log return)
        """
        if self.shares_held > 0:
            # Holding: reward is log return
            log_return = np.log(current_price / prev_price)
            return log_return * self.reward_scaling
        else:
            # Not holding: no reward
            return 0.0

    def _get_state(self) -> np.ndarray:
        """Get current state observation.

        State = [balance_norm, shares, price_norm, indicators_norm, turbulence_norm]

        Returns:
            State array (shape: (13,) for SPY)
        """
        # Normalize balance by initial amount
        balance_norm = self.balance / self.initial_amount

        # Shares (0 or 1, no normalization needed)
        shares = self.shares_held

        # Current price (use log scale for normalization)
        price = self.df.loc[self.day, "close"]
        price_norm = price / 100.0  # Rough normalization (SPY ~$300-400)

        # Technical indicators (already computed in df)
        indicators = []
        for indicator in self.tech_indicator_list:
            value = self.df.loc[self.day, indicator]
            # Normalize each indicator (simple scaling)
            if "rsi" in indicator or "dx" in indicator:
                # RSI and DX are [0, 100]
                indicators.append(value / 100.0)
            elif "cci" in indicator:
                # CCI can be large, clip to [-200, 200]
                indicators.append(np.clip(value, -200, 200) / 200.0)
            else:
                # Other indicators: use price-relative normalization
                indicators.append(value / price if price > 0 else 0.0)

        # Turbulence index (if exists)
        if "turbulence" in self.df.columns:
            turbulence = self.df.loc[self.day, "turbulence"]
            turbulence_norm = np.clip(turbulence, 0, 2)  # Cap at 2σ
        else:
            turbulence_norm = 0.0

        # Construct state vector
        state = np.array(
            [balance_norm, shares, price_norm] + indicators + [turbulence_norm],
            dtype=np.float32,
        )

        return state

    def render(self, mode="human"):
        """Render environment (print current state)."""
        print(f"Day: {self.day}, Date: {self.df.loc[self.day, 'date']}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares: {self.shares_held}")
        print(f"Price: ${self.current_price:.2f}")
        print(
            f"Total Asset: ${self.balance + self.shares_held * self.current_price:.2f}"
        )
        print(f"Trades: {self.trades}, Cost: ${self.cost:.2f}")

    def get_sb_env(self):
        """Get Stable-Baselines3 compatible environment.

        Returns:
            DummyVecEnv wrapper for single environment
        """
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: self])
        return env
