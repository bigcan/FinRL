# Development Log

This file tracks noteworthy experiments, fixes, and decisions shared between Codex and Claude while working on FinRL.

## How to Update
- Append the newest entry at the top of the log to keep recent context visible.
- Include date, agent (Codex or Claude), and a concise summary of actions, results, and next steps.
- Link to relevant artifacts (commit SHAs, notebooks, trained models, tensorboard runs) instead of inlining large outputs.
- Avoid rewriting or deleting past entries; add a short follow-up note if something is superseded.

## Template
```
### YYYY-MM-DD – Agent Name
- Context: short description of what triggered the work.
- Actions: bullet list of commands/code touched.
- Results: metrics, observations, or links to saved artifacts.
- Next Steps: optional reminders or open questions.
```

### 2025-10-30 – Codex
- Context: Investigated PPO hyperparameter tweak (ent_coef/target_kl) to recover 250K-step generalization.
- Actions: Raised `ent_coef` and `target_kl` to 0.03 in `finrl/config.py` and `finrl/applications/spy_rl_trading/config.py`; reran 250K-step training/backtest saving artifacts to `trained_models/spy_ppo_discrete_250k_ent003_kl003*.{zip,json}`.
- Results: Training variance dropped (1.36%), out-of-sample metrics aligned with baseline 10K run (Sharpe 1.01, total return 18.6%). Logged comparison in `analysis/spy_ppo_runs.md`.
- Next Steps: Monitor TensorBoard run `spy_ppo_training_3` for KL/entropy trends; consider further sweep if stability check remains false.

<!-- Log entries start here; add newest above this line. -->
