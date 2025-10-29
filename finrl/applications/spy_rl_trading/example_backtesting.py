"""Example Backtesting Script for SPY RL Trading System.

This script demonstrates the complete backtesting workflow:
    1. Load trained PPO model
    2. Prepare test data (2025 hold-out period)
    3. Run comprehensive backtest
    4. Generate performance reports and visualizations
    5. Compare to buy-and-hold baseline

Usage:
    python -m finrl.applications.spy_rl_trading.example_backtesting

Requirements:
    - Trained model from Phase 3 (example_training.py)
    - Internet connection for downloading SPY test data
"""

from __future__ import annotations

import logging
from pathlib import Path

from stable_baselines3 import PPO

from finrl.applications.spy_rl_trading.backtest import Backtester, backtest_agent
from finrl.applications.spy_rl_trading.data_processor import SPYDataProcessor
from finrl.applications.spy_rl_trading.environment import SPYTradingEnv
from finrl.applications.spy_rl_trading.report import BacktestReporter
from finrl.config import SPY_INDICATORS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run complete backtesting example."""

    logger.info("=" * 80)
    logger.info("SPY RL TRADING - BACKTESTING EXAMPLE")
    logger.info("=" * 80)

    # =============================
    # Step 1: Configuration
    # =============================
    logger.info("\n📋 Step 1: Configuration")

    # Test period (hold-out data - not seen during training)
    # Note: 2025 data may not be available yet. For testing, use recent historical period
    # like 2024-01-01 to 2024-12-31 as hold-out if you trained on 2020-2023
    TEST_START = "2024-01-01"  # Adjust based on your training period
    TEST_END = "2024-12-31"

    # Model path (from Phase 3 training)
    MODEL_PATH = "trained_models/spy_ppo_example/spy_ppo_model"

    # Output directory for results
    OUTPUT_DIR = "results/backtest_example"

    logger.info(f"Test Period: {TEST_START} to {TEST_END}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")

    # =============================
    # Step 2: Load Test Data
    # =============================
    logger.info("\n📊 Step 2: Loading Test Data")

    processor = SPYDataProcessor()

    # Download test data
    logger.info(f"Downloading SPY data for {TEST_START} to {TEST_END}...")
    df_test = processor.download_data(start_date=TEST_START, end_date=TEST_END)

    # Clean data
    logger.info("Cleaning data...")
    df_test = processor.clean_data(df_test)

    # Add technical indicators
    logger.info(f"Computing {len(SPY_INDICATORS)} technical indicators...")
    df_test = processor.add_technical_indicator(df_test, tech_indicator_list=SPY_INDICATORS)

    # Add turbulence index
    logger.info("Computing turbulence index...")
    df_test = processor.add_vix(df_test)

    logger.info(f"✅ Test data prepared: {len(df_test)} trading days")

    # Save price history for baseline comparison
    price_history = df_test.set_index("date")["close"]

    # =============================
    # Step 3: Create Test Environment
    # =============================
    logger.info("\n🏗️  Step 3: Creating Test Environment")

    test_env = SPYTradingEnv(
        df=df_test,
        tech_indicator_list=SPY_INDICATORS,
        initial_amount=100000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1.0,
        print_verbosity=10,
    )

    logger.info("✅ Test environment created")

    # =============================
    # Step 4: Load Trained Model
    # =============================
    logger.info("\n🤖 Step 4: Loading Trained Model")

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        logger.error(f"❌ Model not found at {MODEL_PATH}")
        logger.error("Please run example_training.py first to train a model.")
        return

    logger.info(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully")

    # =============================
    # Step 5: Run Backtest
    # =============================
    logger.info("\n🚀 Step 5: Running Backtest")

    # Create backtester
    backtester = Backtester(
        model=model,
        test_env=test_env,
        price_history=price_history,
        deterministic=True,  # Use deterministic policy
        verbose=1,
    )

    # Run backtest with seed for reproducibility
    logger.info("Running backtest with seed=42 for reproducibility...")
    result = backtester.run(seed=42, compute_baseline=True, validate_criteria=True)

    logger.info("\n✅ Backtest complete!")

    # =============================
    # Step 6: Display Results
    # =============================
    logger.info("\n📈 Step 6: Backtest Results")

    # Print text report
    print("\n" + result.get_report())

    # =============================
    # Step 7: Generate Visualizations
    # =============================
    logger.info("\n🎨 Step 7: Generating Visualizations")

    reporter = BacktestReporter(result)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    logger.info("Generating equity curve...")
    reporter.plot_equity_curve(save_path=output_dir / "equity_curve.png")

    logger.info("Generating drawdown chart...")
    reporter.plot_drawdown(save_path=output_dir / "drawdown.png")

    logger.info("Generating returns distribution...")
    reporter.plot_returns_distribution(save_path=output_dir / "returns_distribution.png")

    logger.info("Generating action distribution...")
    reporter.plot_action_distribution(save_path=output_dir / "action_distribution.png")

    logger.info(f"✅ Visualizations saved to {output_dir}")

    # =============================
    # Step 8: Generate HTML Report
    # =============================
    logger.info("\n📄 Step 8: Generating HTML Report")

    html_path = output_dir / "backtest_report.html"
    reporter.generate_html_report(html_path, include_plots=True)

    logger.info(f"✅ HTML report generated: {html_path}")

    # =============================
    # Step 9: Save Detailed Results
    # =============================
    logger.info("\n💾 Step 9: Saving Detailed Results")

    backtester.save_results(result, output_dir)

    logger.info(f"✅ Detailed results saved to {output_dir}")

    # =============================
    # Step 10: Multiple Runs (Statistical Analysis)
    # =============================
    logger.info("\n🔬 Step 10: Running Multiple Backtests for Statistical Analysis")

    logger.info("Running 5 backtest trials with different seeds...")
    multi_results = backtester.run_multiple(n_runs=5, seeds=[42, 123, 456, 789, 1011])

    # Compute aggregate statistics
    import numpy as np

    sharpe_ratios = [r.metrics["sharpe_ratio"] for r in multi_results]
    total_returns = [r.metrics["total_return"] for r in multi_results]

    logger.info("\n📊 Aggregate Statistics (5 Runs):")
    logger.info(f"   Sharpe Ratio:  {np.mean(sharpe_ratios):.3f} ± {np.std(sharpe_ratios):.3f}")
    logger.info(f"   Total Return:  {np.mean(total_returns):.2%} ± {np.std(total_returns):.2%}")

    # =============================
    # Summary
    # =============================
    logger.info("\n" + "=" * 80)
    logger.info("✅ BACKTESTING EXAMPLE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"View HTML report: file://{html_path.resolve()}")
    logger.info("\nNext steps:")
    logger.info("  1. Review the HTML report for detailed analysis")
    logger.info("  2. Examine the equity curve and drawdown charts")
    logger.info("  3. Compare metrics to success criteria in spec.md")
    logger.info("  4. If metrics are unsatisfactory, tune hyperparameters (Phase 5)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
