"""Hyperparameter Sweep Module for SPY RL Trading.

This module provides comprehensive hyperparameter tuning capabilities including:
    - Grid search over PPO hyperparameters
    - Convergence curve comparison across configurations
    - Best configuration selection based on test metrics
    - Visualization and reporting tools

Example:
    >>> from finrl.applications.spy_rl_trading.hyperparam_sweep import HyperparameterSweep
    >>>
    >>> param_grid = {
    ...     "learning_rate": [1e-3, 1e-4, 1e-5],
    ...     "clip_range": [0.1, 0.2, 0.3],
    ... }
    >>>
    >>> sweep = HyperparameterSweep(
    ...     train_env=train_env,
    ...     test_env=test_env,
    ...     param_grid=param_grid
    ... )
    >>>
    >>> results = sweep.run(total_timesteps=50000)
    >>> best_config = sweep.get_best_config(metric="sharpe_ratio")
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from finrl.applications.spy_rl_trading.agent import PPOAgent
from finrl.applications.spy_rl_trading.backtest import Backtester

logger = logging.getLogger(__name__)


def generate_param_grid(param_grid: dict[str, list]) -> list[dict]:
    """Generate all combinations from hyperparameter grid.

    Args:
        param_grid: Dictionary with parameter names and value lists

    Returns:
        List of parameter dictionaries (all combinations)

    Raises:
        ValueError: If param_grid is empty

    Example:
        >>> grid = {"learning_rate": [1e-3, 1e-4], "batch_size": [64, 128]}
        >>> combinations = generate_param_grid(grid)
        >>> len(combinations)
        4
    """
    if not param_grid:
        raise ValueError("param_grid cannot be empty")

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return combinations


def validate_ppo_params(params: dict) -> bool:
    """Validate PPO hyperparameters.

    Args:
        params: Dictionary of PPO parameters

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If parameters are out of valid ranges
    """
    if "learning_rate" in params:
        lr = params["learning_rate"]
        if lr <= 0:
            raise ValueError(f"learning_rate must be positive, got {lr}")

    if "clip_range" in params:
        clip = params["clip_range"]
        if not 0 < clip < 1:
            raise ValueError(f"clip_range must be in (0, 1), got {clip}")

    if "batch_size" in params:
        bs = params["batch_size"]
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {bs}")
        # Warn if not power of 2 (but don't error)
        if bs & (bs - 1) != 0:
            logger.warning(f"batch_size {bs} is not a power of 2 (recommended for efficiency)")
            return True

    return True


def select_best_config(
    results: dict[str, dict],
    metric: str = "sharpe_ratio",
) -> dict:
    """Select best configuration based on a metric.

    Args:
        results: Dictionary of config results
        metric: Metric to optimize (default: sharpe_ratio)

    Returns:
        Dictionary with best configuration info

    Example:
        >>> best = select_best_config(results, metric="sharpe_ratio")
        >>> print(f"Best config: {best['name']}")
    """
    best_name = None
    best_value = float("-inf")

    for config_name, result in results.items():
        value = result["metrics"].get(metric, float("-inf"))
        if value > best_value:
            best_value = value
            best_name = config_name

    return {
        "name": best_name,
        "params": results[best_name]["params"],
        "metrics": results[best_name]["metrics"],
    }


def create_comparison_table(
    results: dict[str, dict],
    sort_by: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Create comparison table from sweep results.

    Args:
        results: Dictionary of config results
        sort_by: Optional metric to sort by
        ascending: Sort order (default: False = descending)

    Returns:
        DataFrame with comparison metrics

    Example:
        >>> df = create_comparison_table(results, sort_by="sharpe_ratio")
        >>> print(df)
    """
    rows = []
    for config_name, result in results.items():
        row = {"config": config_name}
        row.update(result["params"])
        row.update(result["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    return df


class HyperparameterSweep:
    """Hyperparameter sweep orchestrator for PPO agents.

    Performs grid search over PPO hyperparameters, trains multiple agents,
    and compares performance on test data.
    """

    def __init__(
        self,
        train_env: Any,
        test_env: Any,
        param_grid: dict[str, list],
        price_history: pd.Series | None = None,
        base_config: dict | None = None,
        verbose: int = 1,
    ):
        """Initialize hyperparameter sweep.

        Args:
            train_env: Training environment
            test_env: Test environment for evaluation
            param_grid: Grid of hyperparameters to search
            price_history: Optional price history for baseline comparison
            base_config: Base configuration (merged with grid params)
            verbose: Logging verbosity
        """
        self.train_env = train_env
        self.test_env = test_env
        self.param_grid = param_grid
        self.price_history = price_history
        self.base_config = base_config or {}
        self.verbose = verbose

        # Generate parameter combinations
        self.param_combinations = generate_param_grid(param_grid)
        logger.info(f"Generated {len(self.param_combinations)} parameter combinations")

        # Storage for results
        self.results = {}
        self.convergence_data = {}

    def run(
        self,
        total_timesteps: int = 50000,
        tensorboard_log: str = "./tensorboard_logs/hyperparam_sweep",
        save_models: bool = True,
        model_dir: str = "trained_models/hyperparam_sweep",
    ) -> dict[str, dict]:
        """Run hyperparameter sweep.

        Args:
            total_timesteps: Training timesteps per configuration
            tensorboard_log: Directory for TensorBoard logs
            save_models: Save trained models to disk
            model_dir: Directory for saved models

        Returns:
            Dictionary of results for each configuration

        Example:
            >>> results = sweep.run(total_timesteps=100000)
            >>> best = sweep.get_best_config()
        """
        logger.info("=" * 80)
        logger.info("HYPERPARAMETER SWEEP - STARTING")
        logger.info("=" * 80)
        logger.info(f"Configurations to test: {len(self.param_combinations)}")
        logger.info(f"Training timesteps per config: {total_timesteps:,}")

        for idx, params in enumerate(self.param_combinations, 1):
            config_name = f"config_{idx}"
            logger.info(f"\n[{idx}/{len(self.param_combinations)}] Training {config_name}")
            logger.info(f"Parameters: {params}")

            # Merge with base config
            full_config = {**self.base_config, **params}

            # Validate parameters
            try:
                validate_ppo_params(full_config)
            except ValueError as e:
                logger.error(f"Invalid parameters for {config_name}: {e}")
                continue

            # Train agent
            try:
                agent = PPOAgent(env=self.train_env, config=full_config)
                model = agent.create_model(
                    tensorboard_log=f"{tensorboard_log}/{config_name}",
                    verbose=self.verbose,
                )
                model = agent.train(
                    total_timesteps=total_timesteps,
                    tb_log_name=config_name,
                )

                # Save model if requested
                if save_models:
                    model_path = Path(model_dir) / config_name
                    model_path.mkdir(parents=True, exist_ok=True)
                    agent.save(str(model_path / "model"))
                    logger.info(f"Model saved to {model_path}")

                # Backtest on test data
                logger.info("Running backtest on test data...")
                backtester = Backtester(
                    model=model,
                    test_env=self.test_env,
                    price_history=self.price_history,
                    verbose=0,
                )
                backtest_result = backtester.run(
                    seed=42,
                    compute_baseline=self.price_history is not None,
                    validate_criteria=False,
                )

                # Store results
                self.results[config_name] = {
                    "params": params,
                    "metrics": backtest_result.metrics,
                    "backtest_result": backtest_result,
                }

                # Log metrics
                logger.info(f"Sharpe Ratio: {backtest_result.metrics['sharpe_ratio']:.3f}")
                logger.info(f"Total Return: {backtest_result.metrics['total_return']:.2%}")

            except Exception as e:
                logger.error(f"Error training {config_name}: {e}")
                continue

        logger.info("\n" + "=" * 80)
        logger.info("HYPERPARAMETER SWEEP - COMPLETE")
        logger.info("=" * 80)

        return self.results

    def get_best_config(self, metric: str = "sharpe_ratio") -> dict:
        """Get best configuration based on metric.

        Args:
            metric: Metric to optimize

        Returns:
            Dictionary with best configuration

        Example:
            >>> best = sweep.get_best_config(metric="sharpe_ratio")
            >>> print(f"Best Sharpe: {best['metrics']['sharpe_ratio']:.3f}")
        """
        if not self.results:
            raise ValueError("No results available. Run sweep first.")

        return select_best_config(self.results, metric=metric)

    def get_comparison_table(
        self,
        sort_by: str = "sharpe_ratio",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Get comparison table of all configurations.

        Args:
            sort_by: Metric to sort by
            ascending: Sort order

        Returns:
            DataFrame with comparison metrics
        """
        if not self.results:
            raise ValueError("No results available. Run sweep first.")

        return create_comparison_table(self.results, sort_by=sort_by, ascending=ascending)

    def plot_comparison(
        self,
        metric: str = "sharpe_ratio",
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """Plot comparison of configurations.

        Args:
            metric: Metric to plot
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No results available. Run sweep first.")

        fig, ax = plt.subplots(figsize=figsize)

        config_names = list(self.results.keys())
        metric_values = [self.results[c]["metrics"][metric] for c in config_names]

        bars = ax.bar(range(len(config_names)), metric_values, alpha=0.7, edgecolor="black")

        # Highlight best configuration
        best_idx = np.argmax(metric_values)
        bars[best_idx].set_color("green")

        ax.set_xlabel("Configuration", fontsize=12)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Hyperparameter Sweep - {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(config_names)))
        ax.set_xticklabels(config_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_pareto_frontier(
        self,
        metric_x: str = "total_return",
        metric_y: str = "sharpe_ratio",
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """Plot Pareto frontier of configurations.

        Args:
            metric_x: Metric for x-axis
            metric_y: Metric for y-axis
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No results available. Run sweep first.")

        fig, ax = plt.subplots(figsize=figsize)

        x_values = []
        y_values = []
        labels = []

        for config_name, result in self.results.items():
            x_values.append(result["metrics"][metric_x])
            y_values.append(result["metrics"][metric_y])
            labels.append(config_name)

        # Scatter plot
        scatter = ax.scatter(x_values, y_values, s=100, alpha=0.6, edgecolors="black")

        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (x_values[i], y_values[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # Highlight best by each metric
        best_x_idx = np.argmax(x_values)
        best_y_idx = np.argmax(y_values)
        ax.scatter(
            [x_values[best_x_idx]],
            [y_values[best_x_idx]],
            s=200,
            c="red",
            marker="*",
            edgecolors="black",
            label=f"Best {metric_x}",
        )
        ax.scatter(
            [x_values[best_y_idx]],
            [y_values[best_y_idx]],
            s=200,
            c="green",
            marker="^",
            edgecolors="black",
            label=f"Best {metric_y}",
        )

        ax.set_xlabel(metric_x.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(metric_y.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            f"Hyperparameter Configurations - {metric_x} vs {metric_y}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved Pareto plot to {save_path}")

        if show:
            plt.show()

        return fig
