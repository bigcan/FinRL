"""Example configuration script for SPY RL Trading System.

This script demonstrates how to customize SPY trading configuration
by overriding defaults from finrl/config.py.

Usage:
    python -m finrl.applications.spy_rl_trading.config_example
"""

from __future__ import annotations

from finrl.config import (
    SPY_CONFIG,
    SPY_INDICATORS,
    SPY_PPO_PARAMS,
    SPY_SYMBOL,
    SPY_TEST_END,
    SPY_TEST_START,
    SPY_TRAIN_END,
    SPY_TRAIN_START,
)

# ============================================================================
# Example 1: Customize Date Ranges
# ============================================================================

# Override training period (use 3 years instead of 5)
CUSTOM_TRAIN_START = "2021-01-01"
CUSTOM_TRAIN_END = "2023-12-31"
CUSTOM_TEST_START = "2024-01-01"
CUSTOM_TEST_END = "2024-12-31"

print("Custom Date Configuration:")
print(f"  Train: {CUSTOM_TRAIN_START} to {CUSTOM_TRAIN_END}")
print(f"  Test:  {CUSTOM_TEST_START} to {CUSTOM_TEST_END}")

# ============================================================================
# Example 2: Customize Trading Environment
# ============================================================================

# Modify initial capital and transaction costs
CUSTOM_ENV_CONFIG = SPY_CONFIG.copy()
CUSTOM_ENV_CONFIG.update(
    {
        "initial_amount": 50000,  # Start with $50k instead of $100k
        "buy_cost_pct": 0.0005,  # Lower transaction cost (0.05%)
        "sell_cost_pct": 0.0005,
    }
)

print("\nCustom Environment Configuration:")
print(f"  Initial Capital: ${CUSTOM_ENV_CONFIG['initial_amount']:,}")
print(f"  Transaction Cost: {CUSTOM_ENV_CONFIG['buy_cost_pct']:.4%}")

# ============================================================================
# Example 3: Customize PPO Hyperparameters
# ============================================================================

# Experiment with different learning rate
CUSTOM_PPO_PARAMS = SPY_PPO_PARAMS.copy()
CUSTOM_PPO_PARAMS.update(
    {
        "learning_rate": 1e-3,  # Higher learning rate for faster convergence
        "n_steps": 1024,  # Smaller rollout buffer
        "ent_coef": 0.05,  # Higher entropy for more exploration
    }
)

print("\nCustom PPO Hyperparameters:")
print(f"  Learning Rate: {CUSTOM_PPO_PARAMS['learning_rate']}")
print(f"  Rollout Steps: {CUSTOM_PPO_PARAMS['n_steps']}")
print(f"  Entropy Coeff: {CUSTOM_PPO_PARAMS['ent_coef']}")

# ============================================================================
# Example 4: Customize Technical Indicators
# ============================================================================

# Use subset of indicators for faster training
CUSTOM_INDICATORS = [
    "macd",
    "rsi_30",
    "boll_ub",
    "boll_lb",
    "vix",
]

print("\nCustom Technical Indicators:")
print(f"  Indicators: {CUSTOM_INDICATORS}")
print(f"  Count: {len(CUSTOM_INDICATORS)} (vs {len(SPY_INDICATORS)} default)")

# ============================================================================
# Example 5: Hyperparameter Grid for Tuning
# ============================================================================

# Define grid of hyperparameters to test
HYPERPARAMETER_GRID = {
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "clip_range": [0.1, 0.2, 0.3],
    "ent_coef": [0.01, 0.05, 0.1],
}

print("\nHyperparameter Grid for Tuning:")
for param, values in HYPERPARAMETER_GRID.items():
    print(f"  {param}: {values}")

total_combinations = 1
for values in HYPERPARAMETER_GRID.values():
    total_combinations *= len(values)
print(f"  Total Combinations: {total_combinations}")

# ============================================================================
# Example 6: Using Custom Configuration in Training
# ============================================================================

def create_custom_training_config():
    """Create custom configuration for training."""
    return {
        "symbol": SPY_SYMBOL,
        "train_start": CUSTOM_TRAIN_START,
        "train_end": CUSTOM_TRAIN_END,
        "test_start": CUSTOM_TEST_START,
        "test_end": CUSTOM_TEST_END,
        "env_config": CUSTOM_ENV_CONFIG,
        "ppo_params": CUSTOM_PPO_PARAMS,
        "indicators": CUSTOM_INDICATORS,
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPY RL Trading - Custom Configuration Example")
    print("=" * 60)

    config = create_custom_training_config()

    print("\nFull Custom Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print(
        "To use this configuration, import it in your training script:\n"
        "  from finrl.applications.spy_rl_trading.config_example import CUSTOM_PPO_PARAMS"
    )
    print("=" * 60)
