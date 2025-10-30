# SPY PPO Training Runs

Summary of quick-validation (10K steps) versus extended (250K steps) experiments.

| Metric | 10K Steps | 250K Steps | 250K Steps (ent=0.03, target_kl=0.03) |
| --- | --- | --- | --- |
| **Training Mean Episode Return** | -0.02% | 0.03% | 0.02% |
| **Training Return Std** | 1.79% | 1.58% | 1.36% |
| **Training Convergence Flag** | No | Yes | Yes |
| **Test Total Return (2025)** | 18.55% | 13.08% | 18.55% |
| **Test Annual Return** | 20.81% | 15.03% | 20.81% |
| **Test Sharpe Ratio** | 1.014 | 0.782 | 1.014 |
| **Test Max Drawdown** | -19.21% | -16.62% | -19.21% |
| **Test Win Rate** | 57.28% | 37.86% | 57.28% |
| **Alpha vs Buy-and-Hold** | -0.06% | -5.84% | -0.06% |

Notes:
- 10K-step statistics come from the preliminary smoke test provided earlier.
- 250K-step statistics are from `trained_models/spy_ppo_discrete_250k_metrics.json` and the associated backtest log produced on 2025-10-30.
- 250K-step (ent=0.03, target_kl=0.03) statistics are from `trained_models/spy_ppo_discrete_250k_ent003_kl003_metrics.json` captured on 2025-10-30 after updating PPO hyperparameters.
- Both runs used SPY daily data (2020-01-01 → 2024-12-31) for training and 2025-01-01 → 2025-10-30 out-of-sample evaluation with the default PPO configuration.
