"""Example Hyperparameter Tuning Script for SPY RL Trading.

This script demonstrates comprehensive hyperparameter tuning workflow:
    1. Define hyperparameter grid
    2. Run grid search with multiple PPO configurations
    3. Compare convergence curves
    4. Analyze parameter sensitivity
    5. Select optimal configuration
    6. Generate comprehensive reports

Usage:
    python -m finrl.applications.spy_rl_trading.example_hyperparam_tuning

Note: This is a compute-intensive operation. For quick testing, reduce
      total_timesteps or param_grid size.
"""

from __future__ import annotations

import logging
from pathlib import Path

from finrl.applications.spy_rl_trading.data_processor import SPYDataProcessor
from finrl.applications.spy_rl_trading.environment import SPYTradingEnv
from finrl.applications.spy_rl_trading.hyperparam_analysis import HyperparamAnalyzer
from finrl.applications.spy_rl_trading.hyperparam_sweep import HyperparameterSweep
from finrl.config import SPY_INDICATORS, SPY_PPO_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run complete hyperparameter tuning example."""

    logger.info("=" * 80)
    logger.info("SPY RL TRADING - HYPERPARAMETER TUNING EXAMPLE")
    logger.info("=" * 80)

    # =============================
    # Step 1: Configuration
    # =============================
    logger.info("\n📋 Step 1: Configuration")

    # Training period
    TRAIN_START = "2020-01-01"
    TRAIN_END = "2023-12-31"

    # Test period
    TEST_START = "2024-01-01"
    TEST_END = "2024-12-31"

    # Output directory
    OUTPUT_DIR = "results/hyperparam_tuning"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logger.info(f"Training Period: {TRAIN_START} to {TRAIN_END}")
    logger.info(f"Test Period: {TEST_START} to {TEST_END}")

    # =============================
    # Step 2: Define Hyperparameter Grid
    # =============================
    logger.info("\n🔬 Step 2: Define Hyperparameter Grid")

    # Note: Reduced grid for faster execution. Expand for production.
    param_grid = {
        "learning_rate": [3e-4, 1e-4, 3e-5],  # Learning rate sweep
        "clip_range": [0.1, 0.2, 0.3],  # PPO clip range
        "ent_coef": [0.0, 0.01],  # Entropy coefficient
    }

    # Base configuration (non-varying parameters)
    base_config = {
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }

    n_configs = 1
    for param_values in param_grid.values():
        n_configs *= len(param_values)

    logger.info(f"Hyperparameter Grid: {param_grid}")
    logger.info(f"Base Configuration: {base_config}")
    logger.info(f"Total Configurations: {n_configs}")

    # =============================
    # Step 3: Load Training Data
    # =============================
    logger.info("\n📊 Step 3: Loading Training Data")

    processor = SPYDataProcessor()

    logger.info(f"Downloading SPY data for {TRAIN_START} to {TRAIN_END}...")
    df_train = processor.download_data(start_date=TRAIN_START, end_date=TRAIN_END)
    df_train = processor.clean_data(df_train)
    df_train = processor.add_technical_indicator(df_train, tech_indicator_list=SPY_INDICATORS)
    df_train = processor.add_vix(df_train)

    logger.info(f"✅ Training data: {len(df_train)} days")

    # =============================
    # Step 4: Load Test Data
    # =============================
    logger.info("\n📊 Step 4: Loading Test Data")

    logger.info(f"Downloading SPY data for {TEST_START} to {TEST_END}...")
    df_test = processor.download_data(start_date=TEST_START, end_date=TEST_END)
    df_test = processor.clean_data(df_test)
    df_test = processor.add_technical_indicator(df_test, tech_indicator_list=SPY_INDICATORS)
    df_test = processor.add_vix(df_test)

    price_history = df_test.set_index("date")["close"]

    logger.info(f"✅ Test data: {len(df_test)} days")

    # =============================
    # Step 5: Create Environments
    # =============================
    logger.info("\n🏗️  Step 5: Creating Environments")

    train_env = SPYTradingEnv(
        df=df_train,
        tech_indicator_list=SPY_INDICATORS,
        initial_amount=100000,
        print_verbosity=0,  # Reduce verbosity for sweep
    )

    test_env = SPYTradingEnv(
        df=df_test,
        tech_indicator_list=SPY_INDICATORS,
        initial_amount=100000,
        print_verbosity=0,
    )

    logger.info("✅ Environments created")

    # =============================
    # Step 6: Run Hyperparameter Sweep
    # =============================
    logger.info("\n🚀 Step 6: Running Hyperparameter Sweep")

    sweep = HyperparameterSweep(
        train_env=train_env,
        test_env=test_env,
        param_grid=param_grid,
        price_history=price_history,
        base_config=base_config,
        verbose=0,
    )

    # Note: Reduced timesteps for demo. Use 100K+ for production.
    results = sweep.run(
        total_timesteps=20000,  # Increase to 100000+ for real tuning
        tensorboard_log=f"{OUTPUT_DIR}/tensorboard",
        save_models=True,
        model_dir=f"{OUTPUT_DIR}/models",
    )

    logger.info(f"\n✅ Sweep complete: {len(results)} configurations tested")

    # =============================
    # Step 7: Identify Best Configuration
    # =============================
    logger.info("\n🎯 Step 7: Identifying Best Configuration")

    best_config = sweep.get_best_config(metric="sharpe_ratio")

    logger.info(f"\n🏆 Best Configuration (by Sharpe Ratio):")
    logger.info(f"   Config: {best_config['name']}")
    logger.info(f"   Sharpe Ratio: {best_config['metrics']['sharpe_ratio']:.3f}")
    logger.info(f"   Total Return: {best_config['metrics']['total_return']:.2%}")
    logger.info(f"   Parameters:")
    for param, value in best_config['params'].items():
        logger.info(f"     {param}: {value}")

    # =============================
    # Step 8: Generate Comparison Table
    # =============================
    logger.info("\n📊 Step 8: Generating Comparison Table")

    comparison_df = sweep.get_comparison_table(sort_by="sharpe_ratio", ascending=False)
    comparison_csv = Path(OUTPUT_DIR) / "comparison_table.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    logger.info(f"\nTop 3 Configurations:")
    print(comparison_df.head(3).to_string())
    logger.info(f"\n✅ Full comparison table saved to {comparison_csv}")

    # =============================
    # Step 9: Visualizations
    # =============================
    logger.info("\n🎨 Step 9: Generating Visualizations")

    # Comparison plot
    logger.info("Generating comparison plot...")
    sweep.plot_comparison(
        metric="sharpe_ratio",
        save_path=Path(OUTPUT_DIR) / "comparison_sharpe.png",
    )

    # Pareto frontier
    logger.info("Generating Pareto frontier...")
    sweep.plot_pareto_frontier(
        metric_x="total_return",
        metric_y="sharpe_ratio",
        save_path=Path(OUTPUT_DIR) / "pareto_frontier.png",
    )

    logger.info(f"✅ Visualizations saved to {OUTPUT_DIR}")

    # =============================
    # Step 10: Advanced Analysis
    # =============================
    logger.info("\n🔬 Step 10: Advanced Analysis")

    analyzer = HyperparamAnalyzer(results)

    # Sensitivity analysis
    logger.info("Analyzing parameter sensitivity...")
    for param in param_grid.keys():
        analyzer.analyze_parameter_sensitivity(
            param_name=param,
            metric_name="sharpe_ratio",
            plot=True,
            save_path=Path(OUTPUT_DIR) / f"sensitivity_{param}.png",
        )

    # Correlation matrix
    logger.info("Generating correlation matrix...")
    analyzer.plot_correlation_matrix(
        save_path=Path(OUTPUT_DIR) / "correlation_matrix.png",
    )

    # Metric distributions
    logger.info("Plotting metric distributions...")
    analyzer.plot_metric_distributions(
        save_path=Path(OUTPUT_DIR) / "metric_distributions.png",
    )

    # Summary report
    logger.info("Generating summary report...")
    summary_report = analyzer.generate_summary_report()
    report_path = Path(OUTPUT_DIR) / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write(summary_report)

    print("\n" + summary_report)

    # =============================
    # Step 11: Robust Configuration Selection
    # =============================
    logger.info("\n💪 Step 11: Robust Configuration Selection")

    robust_configs = analyzer.find_robust_configs(metric="sharpe_ratio", top_n=3)
    logger.info("\nTop 3 Robust Configurations:")
    print(robust_configs[["config", "sharpe_ratio", "total_return", "max_drawdown"]].to_string())

    # =============================
    # Summary
    # =============================
    logger.info("\n" + "=" * 80)
    logger.info("✅ HYPERPARAMETER TUNING EXAMPLE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("\nKey Files:")
    logger.info(f"  - Comparison table: {comparison_csv}")
    logger.info(f"  - Summary report: {report_path}")
    logger.info(f"  - Visualizations: {OUTPUT_DIR}/*.png")
    logger.info(f"  - Trained models: {OUTPUT_DIR}/models/")
    logger.info("\nRecommendation:")
    logger.info(f"  Use config '{best_config['name']}' with:")
    for param, value in best_config['params'].items():
        logger.info(f"    {param}: {value}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
