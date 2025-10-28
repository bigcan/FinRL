"""Configuration for SPY RL Trading System.

This module contains SPY-specific configuration separate from the main
FinRL config.py to maintain modularity and allow easy customization.

The configuration is organized into:
    - Date ranges for training and testing
    - Trading environment parameters
    - PPO hyperparameters
    - Technical indicators
    - Paths for models and logs
"""

from __future__ import annotations

# Date ranges for training and testing
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-12-31"

# SPY ticker
SPY_SYMBOL = "SPY"

# Trading environment configuration
SPY_ENV_CONFIG = {
    "initial_amount": 100000,  # Starting capital: $100,000
    "buy_cost_pct": 0.001,  # Transaction cost: 0.1%
    "sell_cost_pct": 0.001,
    "hmax": 1,  # Maximum shares per action (discrete: 0 or 1)
    "reward_scaling": 1.0,  # Log return scaling factor
    "state_space": 13,  # [balance, shares, price, 10 indicators, turbulence]
    "action_space": 3,  # {0: BUY, 1: HOLD, 2: SELL}
}

# PPO hyperparameters (from research.md)
PPO_PARAMS = {
    "n_steps": 2048,  # Rollout buffer size (~8 trading years)
    "batch_size": 64,  # Batch size for policy updates
    "n_epochs": 10,  # Epochs per update
    "learning_rate": 3e-4,  # Policy learning rate
    "clip_range": 0.2,  # PPO clip ratio (epsilon)
    "clip_range_vf": 0.1,  # Value function clip range
    "ent_coef": 0.01,  # Entropy coefficient (exploration bonus)
    "vf_coef": 0.5,  # Value function loss weight
    "max_grad_norm": 0.5,  # Gradient clipping
    "gae_lambda": 0.95,  # GAE (Generalized Advantage Estimation) lambda
    "use_sde": False,  # Stochastic Policy (disabled for simplicity)
    "sde_sample_freq": -1,  # Not used
    "target_kl": 0.01,  # Target KL divergence (early stopping)
    "policy_kwargs": {
        "net_arch": [256, 256],  # Hidden layer sizes
        "activation_fn": "relu",  # ReLU activation (SB3 uses torch.nn.ReLU)
    },
}

# Technical indicators (aligned with FinRL standards)
INDICATORS = [
    "macd",  # Moving Average Convergence Divergence
    "boll_ub",  # Bollinger Bands upper bound
    "boll_lb",  # Bollinger Bands lower bound
    "rsi_30",  # 30-period Relative Strength Index
    "cci_30",  # 30-period Commodity Channel Index
    "dx_30",  # 30-period Directional Movement Index
    "close_30_sma",  # 30-period Simple Moving Average
    "close_60_sma",  # 60-period Simple Moving Average
    "vix",  # Volatility Index (market fear gauge)
]

# Paths for models and logs
TRAINED_MODEL_DIR = "./trained_models/spy_rl_trading"
TENSORBOARD_LOG_DIR = "./tensorboard_logs/spy_ppo"
RESULTS_DIR = "./results/spy_rl_trading"
DATA_SAVE_DIR = "./datasets/spy_rl_trading"
