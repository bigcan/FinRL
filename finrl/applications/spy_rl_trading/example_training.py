"""Example Training Script for SPY RL Trading System.

This script demonstrates the complete workflow for training a PPO agent
on SPY data and backtesting on hold-out data.

Usage:
    python -m finrl.applications.spy_rl_trading.example_training

Or import and run:
    >>> from finrl.applications.spy_rl_trading.example_training import run_example
    >>> run_example()
"""

from __future__ import annotations

from finrl.applications.spy_rl_trading.pipeline import backtest_agent, train_agent
from finrl.config import (
    SPY_INDICATORS,
    SPY_PPO_PARAMS,
    SPY_TEST_END,
    SPY_TEST_START,
    SPY_TRAIN_END,
    SPY_TRAIN_START,
)


def run_example(
    train_start: str = SPY_TRAIN_START,
    train_end: str = SPY_TRAIN_END,
    test_start: str = SPY_TEST_START,
    test_end: str = SPY_TEST_END,
    total_timesteps: int = 100_000,
):
    """Run complete example workflow.

    Steps:
        1. Train PPO agent on 2020-2024 SPY data
        2. Save trained model
        3. Backtest on 2025 hold-out data
        4. Display performance metrics

    Args:
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date
        test_end: Test end date
        total_timesteps: Total training timesteps (default: 100K)

    Returns:
        Dictionary with training and test results
    """
    print("\n" + "=" * 80)
    print("SPY RL TRADING SYSTEM - EXAMPLE TRAINING WORKFLOW")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Training a PPO agent on SPY historical data (2020-2024)")
    print("  2. Backtesting the trained agent on hold-out data (2025)")
    print("  3. Comparing performance to buy-and-hold baseline")
    print("\n" + "=" * 80 + "\n")

    # ========================================================================
    # STEP 1: Train Agent
    # ========================================================================

    print("\n🚀 STEP 1: TRAINING PPO AGENT\n")

    trained_model, train_metrics = train_agent(
        symbol="SPY",
        train_start=train_start,
        train_end=train_end,
        indicators=SPY_INDICATORS,
        ppo_params=SPY_PPO_PARAMS,
        total_timesteps=total_timesteps,
        tensorboard_log="./tensorboard_logs/spy_ppo_example",
        model_save_path="./trained_models/spy_ppo_example",
        initial_amount=100_000,
    )

    print("\n✅ Training complete!")
    print(f"   Model saved to: ./trained_models/spy_ppo_example.zip")
    print(f"   TensorBoard logs: ./tensorboard_logs/spy_ppo_example")
    print(
        f"\n💡 To monitor training: tensorboard --logdir ./tensorboard_logs/spy_ppo_example"
    )

    # ========================================================================
    # STEP 2: Backtest Agent
    # ========================================================================

    print("\n\n🧪 STEP 2: BACKTESTING ON HOLD-OUT DATA\n")

    test_results = backtest_agent(
        model=trained_model,
        test_start=test_start,
        test_end=test_end,
        symbol="SPY",
        indicators=SPY_INDICATORS,
        initial_amount=100_000,
    )

    # ========================================================================
    # STEP 3: Display Results
    # ========================================================================

    print("\n\n📊 STEP 3: FINAL RESULTS\n")

    print("=" * 80)
    print("TRAINING METRICS")
    print("=" * 80)
    if train_metrics:
        print(
            f"  Mean Episode Return: {train_metrics.get('episode_return_mean', 0):.2%}"
        )
        print(
            f"  Return Std Dev:      {train_metrics.get('episode_return_std', 0):.2%}"
        )
        print(
            f"  Converged:           {'✅ Yes' if train_metrics.get('convergence_check', False) else '❌ No'}"
        )
        print(
            f"  Stable:              {'✅ Yes' if train_metrics.get('stability_check', False) else '❌ No'}"
        )
    else:
        print("  Training metrics not available")

    print("\n" + "=" * 80)
    print("BACKTEST METRICS (2025 Hold-Out Data)")
    print("=" * 80)
    print(f"  Total Return:     {test_results['total_return']:.2%}")
    print(f"  Annual Return:    {test_results['annual_return']:.2%}")
    print(f"  Sharpe Ratio:     {test_results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:     {test_results['max_drawdown']:.2%}")
    print(f"  Win Rate:         {test_results['win_rate']:.2%}")

    baseline = test_results["baseline_comparison"]["baseline_metrics"]
    print("\n" + "=" * 80)
    print("BUY-AND-HOLD BASELINE")
    print("=" * 80)
    print(f"  Total Return:     {baseline['total_return']:.2%}")
    print(f"  Sharpe Ratio:     {baseline['sharpe_ratio']:.3f}")

    alpha = test_results["baseline_comparison"]["alpha"]
    beats = test_results["baseline_comparison"]["beats_baseline"]
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"  Alpha (Excess Return): {alpha:.2%}")
    print(f"  Beats Baseline:        {'✅ YES' if beats else '❌ NO'}")

    # ========================================================================
    # STEP 4: Success Criteria Validation
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA VALIDATION")
    print("=" * 80)

    sc_003 = test_results["total_return"] >= 0
    sc_004 = test_results["sharpe_ratio"] >= 0.5

    print(f"  SC-003 (Test Return ≥0%):     {'✅ PASS' if sc_003 else '❌ FAIL'}")
    print(f"  SC-004 (Sharpe Ratio ≥0.5):   {'✅ PASS' if sc_004 else '❌ FAIL'}")

    all_passed = sc_003 and sc_004
    print(f"\n  Overall:                      {'✅ ALL PASS' if all_passed else '⚠️ PARTIAL PASS'}")

    # ========================================================================
    # Conclusion
    # ========================================================================

    print("\n" + "=" * 80)
    print("✅ EXAMPLE WORKFLOW COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Experiment with hyperparameters (finrl/applications/spy_rl_trading/config_example.py)")
    print("  2. Run longer training (increase total_timesteps)")
    print("  3. Try different date ranges or technical indicators")
    print(
        "  4. Monitor training with TensorBoard: tensorboard --logdir ./tensorboard_logs/spy_ppo_example"
    )
    print("\n" + "=" * 80 + "\n")

    return {
        "training": train_metrics,
        "testing": test_results,
        "success_criteria": {"sc_003": sc_003, "sc_004": sc_004, "all_passed": all_passed},
    }


if __name__ == "__main__":
    # Run example with default parameters
    results = run_example(
        total_timesteps=10_000  # Small for quick demo (use 100K for full training)
    )
