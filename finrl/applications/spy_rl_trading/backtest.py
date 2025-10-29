"""Backtesting Engine for SPY RL Trading.

This module provides comprehensive backtesting capabilities for trained PPO agents,
including performance metrics computation, baseline comparison, and result analysis.

Key Features:
    - Run trained agents on out-of-sample test data
    - Compute Sharpe ratio, max drawdown, win rate
    - Compare to buy-and-hold baseline
    - Ensure no lookahead bias (test data never seen during training)
    - Generate detailed backtest reports

Example:
    >>> from finrl.applications.spy_rl_trading.backtest import Backtester
    >>> from stable_baselines3 import PPO
    >>>
    >>> model = PPO.load("trained_models/spy_ppo")
    >>> backtester = Backtester(model=model, test_env=test_env)
    >>> results = backtester.run()
    >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from finrl.applications.spy_rl_trading.metrics import (
    calculate_returns_metrics,
    compare_to_baseline,
    format_metrics_report,
    validate_success_criteria,
)

# Configure logger
logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results with comprehensive metrics.

    Attributes:
        daily_returns: Array of daily log returns
        actions: List of actions taken (0=BUY, 1=HOLD, 2=SELL)
        portfolio_values: Time series of portfolio NAV
        metrics: Dictionary of performance metrics
        baseline_comparison: Comparison to buy-and-hold baseline
        success_criteria: Validation against spec.md success criteria
    """

    def __init__(
        self,
        daily_returns: np.ndarray,
        actions: list[int],
        portfolio_values: list[float],
        metrics: dict,
        baseline_comparison: dict | None = None,
        success_criteria: dict | None = None,
    ):
        """Initialize backtest result container.

        Args:
            daily_returns: Array of daily log returns
            actions: List of discrete actions taken
            portfolio_values: Time series of portfolio NAV
            metrics: Performance metrics dictionary
            baseline_comparison: Optional baseline comparison results
            success_criteria: Optional success criteria validation
        """
        self.daily_returns = daily_returns
        self.actions = actions
        self.portfolio_values = portfolio_values
        self.metrics = metrics
        self.baseline_comparison = baseline_comparison
        self.success_criteria = success_criteria

    def to_dict(self) -> dict:
        """Convert result to dictionary format.

        Returns:
            Dictionary with all backtest results
        """
        return {
            "daily_returns": self.daily_returns,
            "actions": self.actions,
            "portfolio_values": self.portfolio_values,
            "metrics": self.metrics,
            "baseline_comparison": self.baseline_comparison,
            "success_criteria": self.success_criteria,
        }

    def get_report(self) -> str:
        """Generate formatted text report.

        Returns:
            Formatted string report
        """
        baseline = (
            self.baseline_comparison["baseline_metrics"]
            if self.baseline_comparison
            else None
        )
        return format_metrics_report(self.metrics, baseline)


class Backtester:
    """Backtesting engine for trained PPO agents.

    Runs trained agents on out-of-sample test data without retraining,
    computes performance metrics, and compares to baselines.

    Ensures no lookahead bias - test data is never seen during training.
    """

    def __init__(
        self,
        model: PPO,
        test_env: Any,
        price_history: pd.Series | None = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        """Initialize backtester.

        Args:
            model: Trained PPO model (loaded from disk)
            test_env: Test trading environment (SPYTradingEnv)
            price_history: Optional SPY price history for baseline comparison
            deterministic: Use deterministic policy (default: True)
            verbose: Logging verbosity (0=silent, 1=info, 2=debug)
        """
        self.model = model
        self.test_env = test_env
        self.price_history = price_history
        self.deterministic = deterministic
        self.verbose = verbose

        # Validate model is not in training mode
        self.model.policy.eval()

        logger.info("Backtester initialized")
        logger.info(f"Deterministic policy: {deterministic}")

    def run(
        self,
        seed: int | None = None,
        compute_baseline: bool = True,
        validate_criteria: bool = True,
    ) -> BacktestResult:
        """Run backtest on test environment.

        Args:
            seed: Optional random seed for reproducibility
            compute_baseline: Compute buy-and-hold baseline comparison
            validate_criteria: Validate against success criteria

        Returns:
            BacktestResult object with comprehensive metrics

        Example:
            >>> results = backtester.run(seed=42, compute_baseline=True)
            >>> print(results.get_report())
        """
        logger.info("Starting backtest...")

        # Reset environment
        obs, info = self.test_env.reset(seed=seed)

        # Storage for episode data
        daily_returns = []
        actions = []
        portfolio_values = [self.test_env.initial_amount]
        shares_held = []
        prices = []

        done = False
        truncated = False
        step_count = 0

        # Run episode
        while not (done or truncated):
            # Predict action (no retraining!)
            action, _states = self.model.predict(obs, deterministic=self.deterministic)
            action = int(action)  # Convert to Python int

            # Execute action
            obs, reward, done, truncated, info = self.test_env.step(action)

            # Record data
            actions.append(action)
            daily_returns.append(reward)
            portfolio_values.append(self.test_env.asset_memory[-1])
            shares_held.append(obs[1])
            prices.append(obs[2])

            step_count += 1

            if self.verbose >= 2 and step_count % 50 == 0:
                logger.debug(
                    f"Step {step_count}: Action={action}, Reward={reward:.6f}, "
                    f"Portfolio=${portfolio_values[-1]:.2f}"
                )

        logger.info(f"Backtest complete: {step_count} steps")

        # Convert to numpy arrays
        daily_returns = np.array(daily_returns)
        actions = np.array(actions)
        portfolio_values_array = np.array(portfolio_values)

        # Compute metrics
        logger.info("Computing performance metrics...")
        metrics = calculate_returns_metrics(daily_returns)

        # Baseline comparison
        baseline_comparison = None
        if compute_baseline and self.price_history is not None:
            logger.info("Computing baseline comparison...")
            baseline_comparison = compare_to_baseline(daily_returns, self.price_history)
            metrics.update(
                {
                    "alpha": baseline_comparison["alpha"],
                    "beats_baseline": baseline_comparison["beats_baseline"],
                }
            )

        # Success criteria validation
        success_criteria = None
        if validate_criteria:
            logger.info("Validating success criteria...")
            success_criteria = validate_success_criteria(metrics)

            if success_criteria["all_passed"]:
                logger.info("✅ All success criteria passed!")
            else:
                logger.warning("⚠️  Some success criteria not met")
                for key, value in success_criteria.items():
                    if key != "all_passed" and not value:
                        logger.warning(f"   ❌ {key}: Failed")

        # Create result object
        result = BacktestResult(
            daily_returns=daily_returns,
            actions=actions.tolist(),
            portfolio_values=portfolio_values,
            metrics=metrics,
            baseline_comparison=baseline_comparison,
            success_criteria=success_criteria,
        )

        if self.verbose >= 1:
            logger.info("\n" + result.get_report())

        return result

    def run_multiple(
        self,
        n_runs: int = 10,
        seeds: list[int] | None = None,
        compute_baseline: bool = True,
    ) -> list[BacktestResult]:
        """Run multiple backtest trials for statistical analysis.

        Args:
            n_runs: Number of backtest runs
            seeds: Optional list of random seeds (length must match n_runs)
            compute_baseline: Compute baseline for each run

        Returns:
            List of BacktestResult objects

        Example:
            >>> results = backtester.run_multiple(n_runs=10)
            >>> sharpe_ratios = [r.metrics['sharpe_ratio'] for r in results]
            >>> print(f"Mean Sharpe: {np.mean(sharpe_ratios):.3f}")
        """
        if seeds is None:
            seeds = list(range(n_runs))
        elif len(seeds) != n_runs:
            raise ValueError(f"seeds length ({len(seeds)}) must match n_runs ({n_runs})")

        logger.info(f"Running {n_runs} backtest trials...")
        results = []

        for i, seed in enumerate(seeds):
            logger.info(f"Trial {i+1}/{n_runs} (seed={seed})")
            result = self.run(seed=seed, compute_baseline=compute_baseline, validate_criteria=False)
            results.append(result)

        # Aggregate statistics
        sharpe_ratios = [r.metrics["sharpe_ratio"] for r in results]
        total_returns = [r.metrics["total_return"] for r in results]

        logger.info("\n" + "=" * 60)
        logger.info("AGGREGATE STATISTICS (Multiple Runs)")
        logger.info("=" * 60)
        logger.info(f"Sharpe Ratio:  {np.mean(sharpe_ratios):.3f} ± {np.std(sharpe_ratios):.3f}")
        logger.info(f"Total Return:  {np.mean(total_returns):.2%} ± {np.std(total_returns):.2%}")
        logger.info("=" * 60)

        return results

    def save_results(self, result: BacktestResult, output_dir: str | Path) -> None:
        """Save backtest results to disk.

        Args:
            result: BacktestResult object to save
            output_dir: Directory to save results (created if not exists)

        Saves:
            - results.csv: Daily returns and portfolio values
            - metrics.json: Performance metrics
            - report.txt: Formatted text report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save time series data
        df = pd.DataFrame(
            {
                "daily_return": result.daily_returns,
                "action": result.actions,
                "portfolio_value": result.portfolio_values[1:],  # Skip initial value
            }
        )
        df.to_csv(output_dir / "results.csv", index=False)
        logger.info(f"Saved results to {output_dir / 'results.csv'}")

        # Save metrics
        import json

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(result.to_dict()["metrics"], f, indent=2)
        logger.info(f"Saved metrics to {output_dir / 'metrics.json'}")

        # Save report
        with open(output_dir / "report.txt", "w") as f:
            f.write(result.get_report())
        logger.info(f"Saved report to {output_dir / 'report.txt'}")


def backtest_agent(
    model_path: str,
    test_env: Any,
    price_history: pd.Series | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
) -> BacktestResult:
    """Convenience function to backtest a saved agent.

    Args:
        model_path: Path to saved PPO model
        test_env: Test trading environment
        price_history: Optional price history for baseline
        seed: Optional random seed
        output_dir: Optional directory to save results

    Returns:
        BacktestResult object

    Example:
        >>> from finrl.applications.spy_rl_trading.backtest import backtest_agent
        >>> result = backtest_agent(
        ...     model_path="trained_models/spy_ppo",
        ...     test_env=test_env,
        ...     price_history=spy_prices,
        ...     output_dir="results/backtest_2025"
        ... )
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Create backtester
    backtester = Backtester(
        model=model, test_env=test_env, price_history=price_history, verbose=1
    )

    # Run backtest
    result = backtester.run(seed=seed, compute_baseline=price_history is not None)

    # Save results if output directory specified
    if output_dir:
        backtester.save_results(result, output_dir)

    return result
