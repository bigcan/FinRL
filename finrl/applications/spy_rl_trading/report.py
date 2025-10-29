"""Enhanced Reporting Module for SPY RL Trading Backtests.

This module provides comprehensive reporting capabilities including:
    - Formatted text reports
    - Performance visualization (equity curves, drawdown charts)
    - HTML report generation
    - Comparison tables for multiple backtests
    - Action distribution analysis

Example:
    >>> from finrl.applications.spy_rl_trading.report import BacktestReporter
    >>> reporter = BacktestReporter(backtest_result)
    >>> reporter.generate_html_report("reports/backtest_2025.html")
    >>> reporter.plot_equity_curve(save_path="reports/equity_curve.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl.applications.spy_rl_trading.backtest import BacktestResult

logger = logging.getLogger(__name__)


class BacktestReporter:
    """Enhanced reporter for backtest results with visualization.

    Provides comprehensive reporting including plots, tables, and HTML output.
    """

    def __init__(self, result: BacktestResult):
        """Initialize reporter with backtest result.

        Args:
            result: BacktestResult object from backtester
        """
        self.result = result

    def generate_text_report(self, include_actions: bool = True) -> str:
        """Generate comprehensive text report.

        Args:
            include_actions: Include action distribution analysis

        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 80)
        report.append("SPY RL TRADING - BACKTEST REPORT")
        report.append("=" * 80)

        # Performance metrics
        report.append("\n📊 PERFORMANCE METRICS")
        report.append("-" * 80)
        metrics = self.result.metrics
        report.append(f"Total Return:        {metrics['total_return']:>10.2%}")
        report.append(f"Annual Return:       {metrics['annual_return']:>10.2%}")
        report.append(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.3f}")
        report.append(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        report.append(f"Win Rate:            {metrics['win_rate']:>10.2%}")

        # Baseline comparison
        if self.result.baseline_comparison:
            report.append("\n📈 BASELINE COMPARISON (Buy-and-Hold SPY)")
            report.append("-" * 80)
            baseline = self.result.baseline_comparison["baseline_metrics"]
            report.append(f"Baseline Total Return:   {baseline['total_return']:>10.2%}")
            report.append(f"Baseline Sharpe Ratio:   {baseline['sharpe_ratio']:>10.3f}")
            report.append(f"Baseline Max Drawdown:   {baseline['max_drawdown']:>10.2%}")

            alpha = self.result.baseline_comparison["alpha"]
            beats = self.result.baseline_comparison["beats_baseline"]
            report.append(f"\nAlpha (Excess Return):   {alpha:>10.2%}")
            report.append(f"Beats Baseline:          {'>10' if beats else ' ' * 10}{'✅ YES' if beats else '❌ NO'}")

        # Action distribution
        if include_actions:
            report.append("\n🎯 TRADING ACTIVITY")
            report.append("-" * 80)
            actions = np.array(self.result.actions)
            action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}

            for action_id, action_name in action_names.items():
                count = np.sum(actions == action_id)
                pct = count / len(actions) * 100
                report.append(f"{action_name:>6} Actions: {count:>6} ({pct:>5.1f}%)")

            total_trades = np.sum(actions != 1)  # All non-HOLD actions
            report.append(f"\nTotal Trades: {total_trades}")

        # Success criteria
        if self.result.success_criteria:
            report.append("\n✅ SUCCESS CRITERIA VALIDATION")
            report.append("-" * 80)
            criteria = self.result.success_criteria

            status_icon = "✅" if criteria["all_passed"] else "⚠️ "
            report.append(f"Overall Status: {status_icon}")

            for key, value in criteria.items():
                if key != "all_passed":
                    icon = "✅" if value else "❌"
                    report.append(f"  {icon} {key}: {'PASS' if value else 'FAIL'}")

        # Portfolio statistics
        report.append("\n💰 PORTFOLIO STATISTICS")
        report.append("-" * 80)
        portfolio_values = np.array(self.result.portfolio_values)
        report.append(f"Initial Value:  ${portfolio_values[0]:>15,.2f}")
        report.append(f"Final Value:    ${portfolio_values[-1]:>15,.2f}")
        report.append(f"Peak Value:     ${np.max(portfolio_values):>15,.2f}")
        report.append(f"Min Value:      ${np.min(portfolio_values):>15,.2f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def plot_equity_curve(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """Plot portfolio value over time (equity curve).

        Args:
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        portfolio_values = self.result.portfolio_values
        ax.plot(portfolio_values, linewidth=2, label="Agent Portfolio")

        # Add initial value reference line
        ax.axhline(
            y=portfolio_values[0],
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Initial Value",
        )

        ax.set_xlabel("Trading Day", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.set_title("Equity Curve - SPY RL Trading Agent", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved equity curve to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_drawdown(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """Plot drawdown chart over time.

        Args:
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate drawdown
        portfolio_values = np.array(self.result.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max

        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color="red")
        ax.plot(drawdown, linewidth=1.5, color="darkred")

        # Highlight max drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd_value = drawdown[max_dd_idx]
        ax.scatter(
            [max_dd_idx],
            [max_dd_value],
            color="red",
            s=100,
            zorder=5,
            label=f"Max Drawdown: {max_dd_value:.2%}",
        )

        ax.set_xlabel("Trading Day", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.set_title("Drawdown Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved drawdown chart to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_returns_distribution(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (10, 6),
        bins: int = 50,
    ) -> plt.Figure:
        """Plot distribution of daily returns.

        Args:
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size (width, height)
            bins: Number of histogram bins

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        daily_returns = self.result.daily_returns

        # Histogram
        ax.hist(daily_returns, bins=bins, alpha=0.7, color="steelblue", edgecolor="black")

        # Add vertical line for mean
        mean_return = np.mean(daily_returns)
        ax.axvline(
            mean_return, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_return:.4f}"
        )

        # Add vertical line for zero
        ax.axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        ax.set_xlabel("Daily Log Return", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Daily Returns", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved returns distribution to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_action_distribution(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
        figsize: tuple = (8, 6),
    ) -> plt.Figure:
        """Plot distribution of trading actions.

        Args:
            save_path: Optional path to save figure
            show: Display figure interactively
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        actions = np.array(self.result.actions)
        action_names = ["BUY", "HOLD", "SELL"]
        action_counts = [np.sum(actions == i) for i in range(3)]

        colors = ["green", "gray", "red"]
        bars = ax.bar(action_names, action_counts, color=colors, alpha=0.7, edgecolor="black")

        # Add count labels on bars
        for bar, count in zip(bars, action_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{count}\n({count/len(actions)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Action Distribution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved action distribution to {save_path}")

        if show:
            plt.show()

        return fig

    def generate_html_report(
        self,
        output_path: str | Path,
        include_plots: bool = True,
    ) -> None:
        """Generate comprehensive HTML report with embedded plots.

        Args:
            output_path: Path to save HTML file
            include_plots: Include visualization plots
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate plots if requested
        plot_dir = output_path.parent / "plots"
        if include_plots:
            plot_dir.mkdir(exist_ok=True)
            self.plot_equity_curve(save_path=plot_dir / "equity_curve.png")
            self.plot_drawdown(save_path=plot_dir / "drawdown.png")
            self.plot_returns_distribution(save_path=plot_dir / "returns_dist.png")
            self.plot_action_distribution(save_path=plot_dir / "action_dist.png")

        # Generate HTML
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>SPY RL Trading - Backtest Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }")
        html.append("h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }")
        html.append("h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; }")
        html.append(".metrics { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
        html.append(".metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #ecf0f1; }")
        html.append(".metric-label { font-weight: bold; color: #555; }")
        html.append(".metric-value { color: #2c3e50; }")
        html.append(".plot { text-align: center; margin: 20px 0; }")
        html.append(".plot img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }")
        html.append(".success { color: #27ae60; font-weight: bold; }")
        html.append(".warning { color: #e74c3c; font-weight: bold; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")

        html.append("<h1>SPY RL Trading - Backtest Report</h1>")

        # Metrics section
        html.append("<h2>Performance Metrics</h2>")
        html.append("<div class='metrics'>")
        metrics = self.result.metrics
        for key, value in metrics.items():
            if isinstance(value, float):
                if "return" in key or "rate" in key:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)

            html.append(f"<div class='metric-row'>")
            html.append(f"<span class='metric-label'>{key.replace('_', ' ').title()}:</span>")
            html.append(f"<span class='metric-value'>{formatted_value}</span>")
            html.append(f"</div>")
        html.append("</div>")

        # Success criteria
        if self.result.success_criteria:
            html.append("<h2>Success Criteria</h2>")
            html.append("<div class='metrics'>")
            criteria = self.result.success_criteria
            status_class = "success" if criteria["all_passed"] else "warning"
            status_text = "✅ PASSED" if criteria["all_passed"] else "⚠️  FAILED"
            html.append(f"<div class='metric-row'>")
            html.append(f"<span class='metric-label'>Overall Status:</span>")
            html.append(f"<span class='metric-value {status_class}'>{status_text}</span>")
            html.append(f"</div>")
            html.append("</div>")

        # Plots
        if include_plots:
            html.append("<h2>Visualizations</h2>")

            html.append("<div class='plot'>")
            html.append("<h3>Equity Curve</h3>")
            html.append("<img src='plots/equity_curve.png' alt='Equity Curve'>")
            html.append("</div>")

            html.append("<div class='plot'>")
            html.append("<h3>Drawdown</h3>")
            html.append("<img src='plots/drawdown.png' alt='Drawdown'>")
            html.append("</div>")

            html.append("<div class='plot'>")
            html.append("<h3>Returns Distribution</h3>")
            html.append("<img src='plots/returns_dist.png' alt='Returns Distribution'>")
            html.append("</div>")

            html.append("<div class='plot'>")
            html.append("<h3>Action Distribution</h3>")
            html.append("<img src='plots/action_dist.png' alt='Action Distribution'>")
            html.append("</div>")

        html.append("</body>")
        html.append("</html>")

        # Write HTML file
        with open(output_path, "w") as f:
            f.write("\n".join(html))

        logger.info(f"Generated HTML report: {output_path}")


def compare_backtests(
    results: list[BacktestResult],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compare multiple backtest results in a summary table.

    Args:
        results: List of BacktestResult objects
        labels: Optional labels for each backtest (default: "Run 1", "Run 2", ...)

    Returns:
        DataFrame with comparison metrics

    Example:
        >>> results = [result1, result2, result3]
        >>> comparison = compare_backtests(results, labels=["PPO", "A2C", "TD3"])
        >>> print(comparison)
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(results))]

    if len(labels) != len(results):
        raise ValueError("labels length must match results length")

    # Extract metrics
    comparison_data = []
    for label, result in zip(labels, results):
        metrics = result.metrics.copy()
        metrics["label"] = label
        comparison_data.append(metrics)

    df = pd.DataFrame(comparison_data)
    df = df.set_index("label")

    return df
