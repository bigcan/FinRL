# Feature Specification: SPY RL Trading System

**Feature Branch**: `1-spy-rl-trading`
**Created**: 2025-10-27
**Status**: Draft
**Input**: User description: "build a comprehensive RL trading system using finRL framework with the following setups: market:stock, symbol: spy, timeframe: daily, 2020-2025, reward: log-return, action space: discrete (buy, hold, sell)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Quantitative Researcher Training and Validating SPY Trading Strategies (Priority: P1)

A quantitative researcher wants to develop and validate a reinforcement learning trading strategy for SPY
(S&P 500 ETF) using historical daily price data. The researcher needs to train an agent on 2020-2025
historical data to learn optimal trading decisions based on price movements and technical indicators.
The system should provide clear feedback on agent performance through reward signals based on log returns.

**Why this priority**: This is the core value proposition—enabling RL-based strategy development with
historical backtesting. Without this, the system has no demonstrable capability.

**Independent Test**: Can be fully tested by: (1) Loading SPY daily data for 2020-2025, (2) Training an
agent using a discrete action space (buy/hold/sell), (3) Observing converging reward signal (log returns),
(4) Validating learned policy captures market trends.

**Acceptance Scenarios**:

1. **Given** historical SPY daily OHLCV data for 2020-2025/, **When** agent trains for N episodes,
   **Then** agent achieves cumulative log-return >0 (profitable on training set)

2. **Given** a trained agent, **When** system evaluates actions at each trading day,
   **Then** each action (buy/hold/sell) receives a log-return reward signal

3. **Given** multiple training runs, **When** agent initializations vary (random seed changes),
   **Then** learned policies converge to similar profitability metrics (±10% variance acceptable)

---

### User Story 2 - Backtesting Trained Agent on Out-of-Sample Data (Priority: P2)

A trader wants to backtest the trained RL agent on held-out test data to evaluate real-world performance
before deploying to paper trading. The system should replay the agent's learned policy on new market
conditions and generate detailed performance reports showing cumulative returns, drawdowns, and Sharpe ratio.

**Why this priority**: Backtesting is critical for validating generalization. Without it, researchers
cannot distinguish between overfitting and true strategy quality.

**Independent Test**: Can be fully tested by: (1) Splitting data (train/test sets), (2) Loading test set
separately, (3) Running trained agent on test set, (4) Computing performance metrics (return, max drawdown,
Sharpe ratio), (5) Comparing to buy-and-hold baseline.

**Acceptance Scenarios**:

1. **Given** a trained agent and hold-out test period (e.g., 2024-2025),
   **When** agent executes on test data using learned policy,
   **Then** agent generates trading signals without access to test data during training

2. **Given** agent performance metrics from test set,
   **When** comparing RL strategy to buy-and-hold baseline,
   **Then** risk-adjusted returns (Sharpe ratio) are computed for both strategies

3. **Given** multiple test runs with identical random seed,
   **When** running backtest multiple times,
   **Then** results are deterministic (identical trading signals and returns)

---

### User Story 3 - PPO Hyperparameter Tuning and Optimization (Priority: P3)

A researcher wants to experiment with different PPO hyperparameter configurations (learning rate, clip ratio,
entropy coefficient) to find the optimal agent for SPY trading. The system should support tuning key PPO
parameters while maintaining consistent environment behavior and tracking convergence metrics.

**Why this priority**: This enables advanced users to optimize strategy performance through hyperparameter
tuning. It's important but not essential for initial system capability; PPO with default settings achieves
baseline profitability.

**Independent Test**: Can be fully tested by: (1) Training PPO agents with different hyperparameters,
(2) Comparing reward curves and Sharpe ratios across runs, (3) Identifying optimal configuration,
(4) Validating reproducibility with fixed random seed.

**Acceptance Scenarios**:

1. **Given** multiple PPO hyperparameter configurations (e.g., learning rate: 1e-3, 1e-4, 1e-5; clip ratio: 0.2, 0.3),
   **When** training separate PPO agents with each config on same data,
   **Then** system allows side-by-side comparison of reward curves and test Sharpe ratios

2. **Given** a preferred PPO agent (highest Sharpe ratio on test data),
   **When** user saves and loads agent from disk,
   **Then** loaded agent produces identical trading signals on same data

---

### Edge Cases

- What happens when SPY experiences a circuit breaker (trading halt)? System should handle missing data gracefully.
- How does system handle market gaps (weekends, holidays)? Daily data should skip non-trading days automatically.
- What if historical data contains data quality issues (missing OHLCV, extreme outliers)? System must validate data before training.
- What happens if agent learns to always hold (policy collapse)? System should detect this and warn user.
- How does system handle SPY corporate actions (splits, dividend dates)? Adjusted close prices handle this automatically.

## Production Hardening Goals (Phase 2)

### Goal 7: Dependency Management & Reproducibility
- **G7**: Achieve deterministic builds across all environments through Poetry-based dependency management, lockfile enforcement, and comprehensive version pinning. Eliminate setuptools artifacts and pre-release dependency pins to ensure production stability.

### Goal 8: CI/CD Pipeline & Quality Automation
- **G8**: Implement comprehensive continuous integration with automated testing across Python 3.10-3.12, coverage reporting (≥80% threshold), security vulnerability scanning (pip-audit, CodeQL, Bandit), and automated release workflows with SBOM generation.

### Goal 9: Test Hardening & Determinism
- **G9**: Separate unit tests from integration tests using pytest markers, eliminate brittle test assertions (exact row counts), implement deterministic test fixtures with data snapshots, and add property-based testing with Hypothesis for robust edge case coverage.

### Goal 10: Runtime Observability & Configuration
- **G10**: Implement structured logging with structlog (JSON format), centralized configuration management via pydantic-settings with environment variable support, reproducibility hooks for global seed management, and comprehensive debug/info/warning log coverage.

### Goal 11: Docker Containerization
- **G11**: Create multi-stage Docker builds with non-root user execution, docker-compose orchestration for training/backtesting/paper-trading workflows, multi-architecture support (amd64, arm64), and optimized layer caching for faster builds.

### Goal 12: Finance-Specific Production Controls
- **G12**: Implement realistic transaction cost models (commission, spread, market impact, SEC fees), slippage modeling based on volume, NYSE trading calendar integration via pandas_market_calendars, risk limits (max drawdown, daily loss, position size), broker adapter abstraction for multiple brokers, and trading mode separation (backtest/paper/live).

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST download SPY daily OHLCV data from a reliable source (Yahoo Finance default) for any date range (minimum: 1 year, tested: 2020-2025)

- **FR-002**: System MUST clean historical data by removing NaN values, handling splits/dividends via adjusted close prices, and validating data continuity (no gaps except non-trading days)

- **FR-003**: System MUST compute technical indicators (SMA, RSI, MACD, Bollinger Bands) on cleaned price data and expose them as environment features

- **FR-004**: System MUST provide a Gymnasium-compliant trading environment that accepts discrete actions (0=buy, 1=hold, 2=sell) and returns state, reward, and done signals

- **FR-005**: System MUST compute rewards as log returns (log(P_t / P_{t-1})) at each daily timestep, scaled by position (long/flat) to reflect realized profit/loss

- **FR-006**: System MUST implement PPO (Proximal Policy Optimization) agent via Stable-Baselines3 library, supporting configurable hyperparameters (learning rate, clip ratio, entropy coefficient, batch size, n_steps)

- **FR-007**: System MUST save trained agent models to disk and restore them without degradation (deterministic loading with same random seed)

- **FR-008**: System MUST support train/test data splitting at a specified date (e.g., train: 2020-2024, test: 2025) and prevent data leakage

- **FR-009**: System MUST compute backtest performance metrics (cumulative return, max drawdown, Sharpe ratio, win rate) for trained agent and buy-and-hold baseline

- **FR-010**: System MUST log all training metrics to TensorBoard and provide callback hooks for monitoring reward convergence during training

- **FR-011**: System MUST allow hyperparameter configuration via `config.py` without requiring code changes (learning rate, episode length, network architecture)

- **FR-012**: System MUST expose environment state as [current_balance, shares_held, SPY_price, technical_indicators, turbulence_index] at each timestep

### Key Entities

- **Market Data**: Daily OHLCV candles for SPY symbol, spanning 2020-2025, with technical indicators (SMA, RSI, MACD) computed from close prices

- **Trading Environment**: Gymnasium-compliant simulator that executes discrete actions in a market, tracks portfolio balance and shares held, computes log-return rewards

- **DRL Agent (PPO)**: Neural network policy trained via Proximal Policy Optimization that maps observed state (price, indicators) to discrete action distribution (buy/hold/sell probabilities), learns to maximize cumulative log returns through policy gradient updates with clipped objective

- **Trading Decision**: Tuple of (action, position, log_return) where action ∈ {buy, hold, sell}, position reflects current holdings (1=long, 0=flat), reward is realized daily log return

- **Backtest Result**: Record of agent performance on test data including cumulative return %, max drawdown %, Sharpe ratio, win rate (% days with positive return)

## Trading Mechanics & Reward Model

- **Execution timeline**: At trading day `t`, the agent observes features derived from prices up to the close of `t`. The chosen action is executed at the close of `t`, and the reward uses the log return between `close_t` and `close_{t+1}` that materializes overnight. No future bars (price or indicators) are accessible prior to reward computation.
- **Action semantics**: Discrete actions are {`buy`, `hold`, `sell`}. `buy` transitions the portfolio from flat to long one unit; repeated `buy` while already long is treated as a no-op. `sell` closes an existing long to flat; short positions are out of scope and rejected with a warning in debug mode. `hold` maintains the prior position. Invalid transitions never mutate state.
- **Position sizing & inventory**: Positions represent a single unit of SPY (1 share). All cash that is not used to hold the unit remains in the cash component; leverage and partial fills are out of scope. Initial state starts with 1 unit of purchasable cash and zero holdings.
- **Reward function**: `log_ret_t = ln(close_{t+1} / close_t)`. Per-step reward is `reward_t = position_t * log_ret_t - cost_t` where `position_t ∈ {0, 1}`. Transaction cost per edge defaults to 2 bps (1 bps buy + 1 bps sell) and is configurable. Rewards are zero when flat.
- **Transaction costs**: Costs are applied only when the action triggers a position change. Cost parameters are supplied via environment kwargs and default to `buy_cost_bps=1`, `sell_cost_bps=1`. The buy-and-hold baseline uses identical costs to ensure fair comparisons.
- **Data preprocessing**: Indicators are computed on adjusted close prices with rolling windows that only reference historical data (right-aligned, `closed='left'`). Indicator joins preserve the trading-day index; calendar gaps (weekends/holidays) remain in the index with forward-filled indicators while price stays NaN and the step is skipped.
- **State normalization**: Continuous observations (prices, indicators) are z-scored using statistics fit on the training window and reused for validation/test to prevent leakage. Normalization parameters are serialized alongside the trained model.

## Backtest Integrity Requirements

- **Lookahead guard**: Unit tests assert that the environment never queries `t+1` data prior to calling `step(...)`, and that rewards use only post-action information.
- **Cash & PnL accounting**: Portfolio net asset value (NAV) updates as `cash + position * close_t`. NAV snapshots, realized PnL, and cost accrual are written per step for plotting and regression tests.
- **Baseline alignment**: The buy-and-hold benchmark uses the same adjusted data, transaction costs, and rebalance cadence as the agent pipeline.
- **Failure handling**: Missing price bars result in skipping the trading step and logging a warning; agents continue with the next available day.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 (Gate)**: Data Loading — system loads SPY daily OHLCV data for 2020-2025 without errors, with >99% data completeness (missing ≤5 trading days per calendar year).

- **SC-002 (Gate)**: Training Convergence — after training on 2020-2024 data for 100K-500K timesteps, agent achieves cumulative log return >5% on the training window; runs that miss the mark must be diagnosed before release.

- **SC-003 (Gate)**: Test Generalization — trained agent on 2020-2024 data delivers ≥0% cumulative return on the held-out 2025 window.

- **SC-004 (Target)**: Risk-Adjusted Performance — agent Sharpe ratio on test data is ≥0.5 (daily returns annualized with √252) and exceeds the buy-and-hold Sharpe. Failing the target triggers a follow-up investigation but is not a blocker for the initial release.

- **SC-005 (Target)**: PPO Hyperparameter Robustness — three distinct hyperparameter configs (varying learning rate and clip range) converge to profitable test returns with Sharpe ratios within ±20% of the cohort mean.

- **SC-006 (Gate)**: Model Reproducibility — loading a saved policy with identical random seeds reproduces actions and NAV series exactly (tolerance 1e-9).

- **SC-007 (Target)**: Training Stability — TensorBoard reward curves (100-episode moving average) fluctuate within ±20% once convergence starts; sharp collapses require triage.

- **SC-008 (Target)**: Backtest Execution — full train → backtest pipeline completes within 30 minutes on an 8-core CPU and a single mid-range GPU (e.g., RTX 3060). A trimmed “smoke” config must finish under 5 minutes for CI.

- **SC-009 (Signal)**: Hyperparameter Sensitivity — scaling PPO learning rate by 10× shifts Sharpe ratio by ≤20% while remaining profitable; deviations inform future tuning tasks.

- **SC-010 (Gate)**: Documentation — a runnable example notebook and README section walk through data prep, training, and backtesting using the CLI commands listed below.

---

### Production Hardening Success Criteria (Phase 2)

- **SC-H001 (Gate)**: Dependency Management — all dependencies pinned in `poetry.lock` with semantic version ranges, zero high/critical security vulnerabilities detected by pip-audit/CodeQL, and reproducible builds across Linux/macOS/Windows environments verified by deterministic `poetry.lock` hash.

- **SC-H002 (Gate)**: CI/CD Pipeline — GitHub Actions workflow runs successfully across Python 3.10, 3.11, 3.12 matrix with unit tests achieving ≥80% coverage, integration tests passing with deterministic fixtures, security scans (Bandit, CodeQL, pip-audit) reporting zero critical issues, and automated SBOM generation on release tags.

- **SC-H003 (Target)**: Test Quality — unit tests separated from integration tests via pytest markers (`@pytest.mark.unit`, `@pytest.mark.integration`), brittle assertions eliminated (tolerance-based comparisons), property-based tests added for core functions (Hypothesis), and deterministic test fixtures implemented using frozen data snapshots.

- **SC-H004 (Gate)**: Runtime Observability — structured logging implemented with structlog (JSON format), centralized configuration via pydantic-settings with environment variable support (SPY_* prefix), reproducibility hooks ensure global seed management, and logs capture all critical operations (data loading, training, backtesting, trading decisions).

- **SC-H005 (Target)**: Finance Controls — transaction cost model realistic (<1% cumulative impact on backtest returns), slippage model accounts for volume, NYSE trading calendar prevents weekend/holiday execution, risk limits prevent catastrophic losses (max drawdown ≤20%, daily loss ≤5%), and paper trading functional with Alpaca sandbox environment.

- **SC-H006 (Signal)**: Live Trading Readiness — broker adapter abstraction implemented with base interface supporting order placement/cancellation/status queries, dry-run mode operational (logs trades without execution), monitoring active (optional Prometheus metrics endpoint), and separation of backtest/paper/live modes enforced via configuration flags.

## Metric Definitions

- **Cumulative return**: `(NAV_T / NAV_0) - 1`, computed from the backtest NAV series.
- **Log return**: `ln(NAV_t / NAV_{t-1})`; rewards assume NAV stays positive.
- **Sharpe ratio**: `(μ_d - r_f) / σ_d * √252` where `μ_d` and `σ_d` are mean and standard deviation of daily log returns, and `r_f` defaults to 0 for both agent and baseline.
- **Max drawdown**: `max_t (1 - NAV_t / peak_t)` with `peak_t` as the running NAV maximum.
- **Win rate**: Percentage of trading days with positive log return while the agent holds a position.
- **Transaction cost attribution**: `cost_bps * price` expressed as NAV delta; stored separately to reconcile PnL.

## Assumptions

- SPY data is available from Yahoo Finance or Alpaca API without authentication issues for 2020-2025 period
- Daily timeframe provides sufficient signal for learning (hourly/minute data not required for initial implementation)
- Log-return-based rewards are sufficient; more complex reward shaping (Sharpe ratio optimization) deferred to future iterations
- Discrete action space (buy/hold/sell) is adequate; continuous position sizing (e.g., portfolio %) deferred
- Buy and sell actions execute at daily close price without slippage modeling (realistic slippage can be added later)
- Historical SPY data reflects real market conditions without structural breaks; 2020-2025 period chosen to include COVID crisis and post-pandemic recovery
- PPO algorithm (Proximal Policy Optimization) is suitable for discrete action trading tasks; default hyperparameters from Stable-Baselines3 provide reasonable starting point
- PPO requires ≥1 year (252+ trading days) of training data to learn meaningful patterns; 5 years (2020-2024) provides ample training signal
- Technical indicators (SMA, RSI, MACD) standardized in `config.py` `INDICATORS` list are sufficient for agent observation and will be z-scored using statistics learned on the training split only.
- Yahoo Finance adjusted close prices (split/dividend adjusted) serve as the canonical data source; gaps larger than one trading day trigger data quality warnings and manual inspection.

## Constraints

- Training must complete within computational limits of standard GPU/CPU hardware (no distributed training required for initial MVP)
- No real-time market data or live trading required for v1.0 (backtesting and paper trading deferred)
- System must not use future data (no lookahead bias in any backtest or evaluation)
- Agent must operate on daily frequency without intraday state (daily close prices only)

## Out of Scope (Explicitly Excluded / Deferred)

- Alternative DRL algorithms (A2C, TD3, DDPG, etc.) [deferred to v2.0; PPO is proven for discrete trading]
- Real-time paper trading with Alpaca (deferred to v2.0)
- Risk management features (stop-loss, position limits) [will be environment variants in future releases]
- Multi-asset portfolio optimization (SPY only for v1.0; multi-asset strategies deferred)
- Factor/sentiment analysis (pure price-based technical indicators for v1.0)
- Portfolio rebalancing (single-asset strategy)

## Reproducibility & Artifacts

- **Seed management**: CLI accepts a `--seed` argument that drives `random`, `numpy`, `torch`, and SB3 seeds. GPU runs set `torch.backends.cudnn.deterministic = True` and disable benchmark mode when deterministic behavior is requested.
- **Model storage**: Trained policies, scaler parameters, and run metadata are written to `trained_models/<run_name>/` (default `trained_models/ppo_spy_daily`).
- **Data cache**: Raw CSV/Parquet downloads live under `datasets/`; processed arrays are versioned by timestamp to avoid collisions across experiments.
- **Logging**: TensorBoard logs go to `tensorboard_log/<run_name>/`. Backtests write NAV series, metrics CSV, and plots under `results/<run_name>/`.
- **Dependency pinning**: `pyproject.toml` pins versions for Stable-Baselines3, Gymnasium, pandas, and numpy. Spec changes that require upgrades must be documented with migration notes.

## Testing Strategy

- `unit_tests/environments/test_spy_env.py`: Validates discrete action transitions, rejects shorting, checks reward math with transaction costs, and ensures no lookahead on a synthetic price series.
- `unit_tests/environments/test_indicator_windows.py`: Confirms rolling indicator calculations only use historical data and keep index alignment with the price frame.
- `unit_tests/agents/test_ppo_reproducibility.py`: Trains a tiny PPO agent for a few epochs on synthetic data, saves, reloads, and asserts identical action sequences given the same seed.
- `unit_tests/backtest/test_metrics.py`: Regression tests for Sharpe ratio, max drawdown, win rate, and cumulative return computations.
- CI smoke run: `poetry run pytest -k spy_env` plus a short training script (`--timesteps 1024`) to ensure end-to-end wiring stays under 5 minutes.

## CLI & Path Mapping

- **Direct training**: `poetry run python - <<'PY'` invoking `finrl/train.py:1` with `ticker_list=['SPY']`, `drl_lib='stable_baselines3'`, `model_name='ppo'`, and `cwd='trained_models/ppo_spy_daily'` trains the agent using `finrl/meta/env_stock_trading/env_stocktrading_np.py:1`.
- **Direct backtest**: `poetry run python - <<'PY'` calling `finrl/test.py:1` loads the saved policy and evaluates on 2025 data via `finrl/meta/env_stock_trading/env_stocktrading.py:1`.
- **Launcher wrapper**: `poetry run python finrl/main.py --mode=train|test` seeds directories declared in `finrl/config.py:1`. For SPY runs, either introduce `finrl/applications/spy_cli.py` to forward CLI flags or temporarily override the ticker list before execution.
- **Artifacts**: Generated assets respect the directories from `finrl/config.py` (`datasets/`, `trained_models/`, `tensorboard_log/`, `results/`). Success criteria reference these locations for verification.
- **Documentation**: `examples/spy_ppo_daily.ipynb` (to be created) mirrors the CLI commands and includes equity curve, drawdown, and Sharpe visualizations.
