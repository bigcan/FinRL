"""Training Pipeline for SPY RL Trading System.

This module orchestrates the full train-test workflow:
    1. Download and prepare SPY data
    2. Create trading environment
    3. Train PPO agent
    4. Save trained model
    5. Backtest on hold-out data
    6. Generate performance report

Example:
    >>> from finrl.applications.spy_rl_trading.pipeline import train_agent
    >>> from finrl.config import SPY_PPO_PARAMS, SPY_INDICATORS
    >>>
    >>> trained_model, metrics = train_agent(
    ...     symbol="SPY",
    ...     train_start="2020-01-01",
    ...     train_end="2024-12-31",
    ...     indicators=SPY_INDICATORS,
    ...     ppo_params=SPY_PPO_PARAMS,
    ...     total_timesteps=100_000
    ... )
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from finrl.applications.spy_rl_trading.agent import PPOAgent
from finrl.applications.spy_rl_trading.data_processor import SPYDataProcessor
from finrl.applications.spy_rl_trading.environment import SPYTradingEnv
from finrl.applications.spy_rl_trading.metrics import (
    calculate_returns_metrics,
    compare_to_baseline,
    compute_training_metrics,
    format_metrics_report,
)
from finrl.config import SPY_INDICATORS, SPY_PPO_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_agent(
    symbol: str = "SPY",
    train_start: str = "2020-01-01",
    train_end: str = "2024-12-31",
    indicators: list[str] | None = None,
    ppo_params: dict | None = None,
    total_timesteps: int = 100_000,
    tensorboard_log: str = "./tensorboard_logs/spy_ppo",
    model_save_path: str = "./trained_models/spy_ppo_discrete",
    initial_amount: int = 100_000,
) -> tuple[Any, dict]:
    """Train PPO agent on SPY data.

    Full pipeline:
        1. Download SPY OHLCV data
        2. Clean and add technical indicators
        3. Create trading environment
        4. Initialize and train PPO agent
        5. Save trained model
        6. Return model and training metrics

    Args:
        symbol: Ticker symbol (default: "SPY")
        train_start: Training start date
        train_end: Training end date
        indicators: List of technical indicators
        ppo_params: PPO hyperparameters
        total_timesteps: Total training timesteps
        tensorboard_log: TensorBoard log directory
        model_save_path: Path to save trained model
        initial_amount: Initial trading capital

    Returns:
        Tuple of (trained_model, training_metrics)

    Example:
        >>> model, metrics = train_agent(
        ...     train_start="2020-01-01",
        ...     train_end="2024-12-31",
        ...     total_timesteps=100_000
        ... )
        >>> print(f"Training complete! Mean return: {metrics['episode_return_mean']:.2%}")
    """
    logger.info("=" * 60)
    logger.info("SPY RL TRADING - TRAINING PIPELINE")
    logger.info("=" * 60)

    # Set defaults
    indicators = indicators or SPY_INDICATORS
    ppo_params = ppo_params or SPY_PPO_PARAMS

    # Step 1: Download data
    logger.info(f"\n📥 Step 1/5: Downloading {symbol} data...")
    logger.info(f"   Date range: {train_start} to {train_end}")

    processor = SPYDataProcessor()
    df = processor.download_data(
        start_date=train_start,
        end_date=train_end,
        ticker_list=[symbol],
    )
    logger.info(f"   ✓ Downloaded {len(df)} days of data")

    # Step 2: Clean and prepare data
    logger.info("\n🧹 Step 2/5: Cleaning data and adding indicators...")

    df = processor.clean_data(df)
    logger.info(f"   ✓ Data cleaned ({len(df)} days remain)")

    df = processor.add_technical_indicator(df, indicators)
    logger.info(f"   ✓ Added {len(indicators)} technical indicators")

    df = processor.add_vix(df)
    logger.info("   ✓ Added VIX volatility index")

    df = processor.calculate_turbulence(df)
    logger.info("   ✓ Calculated turbulence index")

    # Step 3: Create environment
    logger.info("\n🏗️  Step 3/5: Creating trading environment...")

    env = SPYTradingEnv(
        df=df,
        tech_indicator_list=indicators,
        initial_amount=initial_amount,
        print_verbosity=1000,  # Less verbose during training
    )
    logger.info(f"   ✓ Environment created (state_dim={env.state_dim})")

    # Step 4: Train agent
    logger.info("\n🤖 Step 4/5: Training PPO agent...")
    logger.info(f"   Total timesteps: {total_timesteps}")
    logger.info(f"   Learning rate: {ppo_params.get('learning_rate', 'N/A')}")
    logger.info(f"   TensorBoard logs: {tensorboard_log}")

    agent = PPOAgent(env=env, config=ppo_params)
    agent.create_model(tensorboard_log=tensorboard_log)

    trained_model = agent.train(
        total_timesteps=total_timesteps,
        tb_log_name="spy_ppo_training",
    )

    logger.info("   ✓ Training complete!")

    # Step 5: Save model
    logger.info("\n💾 Step 5/5: Saving trained model...")

    agent.save(model_save_path)
    logger.info(f"   ✓ Model saved to {model_save_path}.zip")

    # Compute training metrics
    if hasattr(env, "rewards_memory") and len(env.rewards_memory) > 0:
        training_metrics = compute_training_metrics(env.rewards_memory)
        logger.info("\n📊 Training Metrics:")
        logger.info(
            f"   Mean Episode Return: {training_metrics['episode_return_mean']:.2%}"
        )
        logger.info(
            f"   Return Std: {training_metrics['episode_return_std']:.2%}"
        )
        logger.info(f"   Converged: {'✅ Yes' if training_metrics['convergence_check'] else '❌ No'}")
    else:
        training_metrics = {}

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 60)

    return trained_model, training_metrics


def backtest_agent(
    model: Any,
    test_start: str = "2025-01-01",
    test_end: str = "2025-12-31",
    symbol: str = "SPY",
    indicators: list[str] | None = None,
    initial_amount: int = 100_000,
) -> dict:
    """Backtest trained agent on hold-out data.

    Args:
        model: Trained PPO model (or path to model)
        test_start: Test period start date
        test_end: Test period end date
        symbol: Ticker symbol
        indicators: Technical indicators
        initial_amount: Initial capital

    Returns:
        Dictionary with backtest results and metrics

    Example:
        >>> results = backtest_agent(
        ...     model=trained_model,
        ...     test_start="2025-01-01",
        ...     test_end="2025-12-31"
        ... )
        >>> print(f"Test Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    """
    logger.info("=" * 60)
    logger.info("SPY RL TRADING - BACKTESTING PIPELINE")
    logger.info("=" * 60)

    indicators = indicators or SPY_INDICATORS

    # Step 1: Download test data
    logger.info(f"\n📥 Step 1/3: Downloading test data...")
    logger.info(f"   Date range: {test_start} to {test_end}")

    processor = SPYDataProcessor()
    df = processor.download_data(
        start_date=test_start,
        end_date=test_end,
        ticker_list=[symbol],
    )
    logger.info(f"   ✓ Downloaded {len(df)} days")

    # Step 2: Prepare data
    logger.info("\n🧹 Step 2/3: Preparing test data...")

    df = processor.clean_data(df)
    df = processor.add_technical_indicator(df, indicators)
    df = processor.add_vix(df)
    df = processor.calculate_turbulence(df)
    logger.info("   ✓ Data prepared")

    # Step 3: Run backtest
    logger.info("\n🧪 Step 3/3: Running backtest...")

    env = SPYTradingEnv(
        df=df,
        tech_indicator_list=indicators,
        initial_amount=initial_amount,
        print_verbosity=1000,
    )

    # Load model if path provided
    if isinstance(model, str):
        agent = PPOAgent(env=env)
        agent.load(model)
        model = agent.model

    # Run episode
    obs, info = env.reset()
    done = False
    daily_returns = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        daily_returns.append(reward)

    # Calculate metrics
    daily_returns = np.array(daily_returns)
    metrics = calculate_returns_metrics(daily_returns)

    # Compare to baseline
    baseline_comparison = compare_to_baseline(
        agent_returns=daily_returns,
        price_history=df["close"],
    )

    # Generate report
    report = format_metrics_report(
        metrics=metrics,
        baseline=baseline_comparison["baseline_metrics"],
    )

    logger.info("\n" + report)

    logger.info("\n" + "=" * 60)
    logger.info("✅ BACKTESTING COMPLETE!")
    logger.info("=" * 60)

    return {
        **metrics,
        "baseline_comparison": baseline_comparison,
        "daily_returns": daily_returns,
    }


if __name__ == "__main__":
    # Example usage: Train and backtest
    from finrl.config import SPY_TRAIN_END, SPY_TRAIN_START, SPY_TEST_END, SPY_TEST_START

    print("\n🚀 Starting SPY RL Trading Pipeline...\n")

    # Train agent
    model, train_metrics = train_agent(
        train_start=SPY_TRAIN_START,
        train_end=SPY_TRAIN_END,
        total_timesteps=10_000,  # Small for demo
    )

    # Backtest agent
    test_metrics = backtest_agent(
        model=model,
        test_start=SPY_TEST_START,
        test_end=SPY_TEST_END,
    )

    print("\n✅ Pipeline complete!")
