# SPY PPO Training Runs

Summary of quick-validation (10K steps) versus extended (250K steps) experiments.

| Metric | 10K Steps | 250K Steps |
| --- | --- | --- |
| **Training Mean Episode Return** | -0.02% | 0.03% |
| **Training Return Std** | 1.79% | 1.58% |
| **Training Convergence Flag** | No | Yes |
| **Test Total Return (2025)** | 18.55% | 13.08% |
| **Test Annual Return** | 20.81% | 15.03% |
| **Test Sharpe Ratio** | 1.014 | 0.782 |
| **Test Max Drawdown** | -19.21% | -16.62% |
| **Test Win Rate** | 57.28% | 37.86% |
| **Alpha vs Buy-and-Hold** | -0.06% | -5.84% |

Notes:
- 10K-step statistics come from the preliminary smoke test provided earlier.
- 250K-step statistics are from `trained_models/spy_ppo_discrete_250k_metrics.json` and the associated backtest log produced on 2025-10-30.
- Both runs used SPY daily data (2020-01-01 → 2024-12-31) for training and 2025-01-01 → 2025-10-30 out-of-sample evaluation with the default PPO configuration.
