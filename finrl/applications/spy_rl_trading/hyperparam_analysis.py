"""Hyperparameter Analysis Module for SPY RL Trading.

This module provides analysis and visualization tools for hyperparameter sweep results:
    - Convergence curve comparison
    - Sensitivity analysis
    - Correlation analysis between parameters and metrics
    - Statistical significance testing

Example:
    >>> from finrl.applications.spy_rl_trading.hyperparam_analysis import HyperparamAnalyzer
    >>>
    >>> analyzer = HyperparamAnalyzer(sweep_results)
    >>> analyzer.plot_convergence_curves(save_path="convergence.png")
    >>> analyzer.analyze_parameter_sensitivity("learning_rate", "sharpe_ratio")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class HyperparamAnalyzer:
    """Analyzer for hyperparameter sweep results.

    Provides comprehensive analysis and visualization tools.
    """

    def __init__(self, results: dict[str, dict]):
        """Initialize analyzer with sweep results.

        Args:
            results: Dictionary of sweep results from HyperparameterSweep
        """
        self.results = results
        self.df = self._results_to_dataframe()

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results dictionary to DataFrame for analysis.

        Returns:
            DataFrame with parameters and metrics
        """
        rows = []
        for config_name, result in self.results.items():
            row = {"config": config_name}
            row.update(result["params"])
            row.update(result["metrics"])
            rows.append(row)

        return pd.DataFrame(rows)

    def analyze_parameter_sensitivity(
        self,
        param_name: str,
        metric_name: str,
        plot: bool = True,
        save_path: str | Path | None = None,
    ) -> dict:
        """Analyze sensitivity of metric to parameter variations.

        Args:
            param_name: Parameter to analyze
            metric_name: Metric to analyze
            plot: Generate visualization
            save_path: Optional path to save figure

        Returns:
            Dictionary with sensitivity analysis results
        """
        if param_name not in self.df.columns:
            raise ValueError(f"Parameter {param_name} not found in results")

        if metric_name not in self.df.columns:
            raise ValueError(f"Metric {metric_name} not found in results")

        # Group by parameter value
        grouped = self.df.groupby(param_name)[metric_name].agg(["mean", "std", "count"])

        # Calculate correlation
        correlation = self.df[param_name].corr(self.df[metric_name])

        results = {
            "param_name": param_name,
            "metric_name": metric_name,
            "correlation": correlation,
            "grouped_stats": grouped,
        }

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Scatter plot
            ax1.scatter(self.df[param_name], self.df[metric_name], alpha=0.6, s=100)
            ax1.set_xlabel(param_name.replace("_", " ").title(), fontsize=12)
            ax1.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
            ax1.set_title(
                f"Sensitivity: {metric_name} vs {param_name}\nCorrelation: {correlation:.3f}",
                fontsize=13,
                fontweight="bold",
            )
            ax1.grid(True, alpha=0.3)

            # Box plot
            param_values = sorted(self.df[param_name].unique())
            box_data = [
                self.df[self.df[param_name] == val][metric_name].values
                for val in param_values
            ]
            ax2.boxplot(box_data, labels=[f"{val:.0e}" for val in param_values])
            ax2.set_xlabel(param_name.replace("_", " ").title(), fontsize=12)
            ax2.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
            ax2.set_title(f"Distribution by {param_name}", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved sensitivity plot to {save_path}")

            plt.close()

        return results

    def plot_correlation_matrix(
        self,
        params: list[str] | None = None,
        metrics: list[str] | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """Plot correlation matrix between parameters and metrics.

        Args:
            params: List of parameters to include (None = all)
            metrics: List of metrics to include (None = all)
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Determine columns to include
        if params is None:
            param_cols = [c for c in self.df.columns if c not in ["config"]]
        else:
            param_cols = params

        if metrics is not None:
            param_cols = [c for c in param_cols if c in params or c in metrics]

        # Select numerical columns only
        numerical_df = self.df[param_cols].select_dtypes(include=[np.number])

        # Compute correlation matrix
        corr_matrix = numerical_df.corr()

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Correlation Matrix - Parameters vs Metrics", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved correlation matrix to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_metric_distributions(
        self,
        metrics: list[str] | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (14, 4),
    ) -> plt.Figure:
        """Plot distribution of metrics across configurations.

        Args:
            metrics: List of metrics to plot (None = common metrics)
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ["sharpe_ratio", "total_return", "max_drawdown"]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric not in self.df.columns:
                logger.warning(f"Metric {metric} not found, skipping")
                continue

            values = self.df[metric].values

            # Histogram
            ax.hist(values, bins=15, alpha=0.7, color="steelblue", edgecolor="black")

            # Add vertical line for mean
            mean_val = np.mean(values)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.3f}")

            ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"Distribution: {metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved metric distributions to {save_path}")

        if show:
            plt.show()

        return fig

    def generate_summary_report(self) -> str:
        """Generate text summary report of sweep results.

        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 80)
        report.append("HYPERPARAMETER SWEEP - SUMMARY REPORT")
        report.append("=" * 80)

        # Configuration count
        report.append(f"\nTotal Configurations Tested: {len(self.results)}")

        # Best by each metric
        metrics = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
        report.append("\n📊 Best Configurations by Metric:")
        report.append("-" * 80)

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            if metric == "max_drawdown":
                # For drawdown, minimize (less negative is better)
                best_idx = self.df[metric].idxmax()
            else:
                # For other metrics, maximize
                best_idx = self.df[metric].idxmax()

            best_row = self.df.loc[best_idx]
            report.append(f"\n{metric.replace('_', ' ').title()}:")
            report.append(f"  Config: {best_row['config']}")
            report.append(f"  Value: {best_row[metric]:.4f}")

            # Show parameters
            param_cols = [c for c in self.df.columns if c not in ["config"] + metrics]
            for param in param_cols:
                if param in best_row:
                    report.append(f"  {param}: {best_row[param]}")

        # Statistical summary
        report.append("\n\n📈 Statistical Summary:")
        report.append("-" * 80)
        for metric in metrics:
            if metric not in self.df.columns:
                continue

            mean_val = self.df[metric].mean()
            std_val = self.df[metric].std()
            min_val = self.df[metric].min()
            max_val = self.df[metric].max()

            report.append(f"\n{metric.replace('_', ' ').title()}:")
            report.append(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
            report.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def find_robust_configs(
        self,
        metric: str = "sharpe_ratio",
        top_n: int = 3,
    ) -> pd.DataFrame:
        """Find most robust configurations (high performance, low variance).

        Args:
            metric: Metric to evaluate
            top_n: Number of top configurations to return

        Returns:
            DataFrame with top configurations
        """
        # For now, just return top N by metric
        # In future, could consider variance across multiple runs
        sorted_df = self.df.sort_values(by=metric, ascending=False)
        return sorted_df.head(top_n)
