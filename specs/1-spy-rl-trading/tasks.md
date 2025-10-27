# Tasks: SPY RL Trading System

**Input**: Design documents from `/specs/1-spy-rl-trading/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/API.md ✅
**Feature Branch**: `1-spy-rl-trading`
**Tests**: Included (per FinRL Constitution Principle V: Test-First & Observability)
**Organization**: Tasks grouped by user story (P1, P2, P3) for independent implementation and testing

---

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: Parallelizable (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (US1, US2, US3) for user story phase tasks
- **File paths**: Absolute from repo root; `finrl/` prefix for FinRL integration

---

## Implementation Strategy

**MVP Scope**: User Story 1 (P1) - Training and validating SPY trading strategy
- Phase 1-2 setup + Phase 3 (US1) = ~2-3 weeks development
- Phase 4 (US2) adds backtesting = +1 week
- Phase 5 (US3) adds hyperparameter tuning = +1 week

**Dependencies**: US1 → US2 (needs trained model), US2 → US3 (hyperparameter comparison)

**Parallel Execution**:
- Phase 1: Sequential (project setup)
- Phase 2: Parallel data processor & environment setup
- Phase 3 (US1): Parallel model & training infrastructure
- Phase 4 (US2): Sequential (depends on US1 output)
- Phase 5 (US3): Sequential (depends on US1 & US2)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and FinRL module structure
**Duration**: 1-2 days
**Blocker**: Must complete before Phases 2-5

- [ ] T001 Create project structure: `finrl/applications/spy_rl_trading/` with `__init__.py`, `config.py`, `README.md`
- [ ] T002 Create test structure: `unit_tests/applications/spy_rl_trading/` with `__init__.py`
- [ ] T003 [P] Create documentation structure: `specs/1-spy-rl-trading/contracts/` (already created)
- [ ] T004 Copy FinRL template config: add `SPY_CONFIG` and `PPO_PARAMS` to `finrl/config.py`

**Checkpoint**: Project structure in place; dependencies installed; ready for Phase 2

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data and environment infrastructure used by all user stories
**Duration**: 3-5 days
**Blocker**: US1, US2, US3 cannot proceed until complete
**⚠️ CRITICAL**: No user story work can begin until Phase 2 complete

### Data Processing Foundation

- [ ] T005 [P] Implement `SPYDataProcessor` in `finrl/applications/spy_rl_trading/data_processor.py`:
  - Inherit from FinRL's YahooFinance processor
  - Implement: `download_data(start, end)`, `clean_data()`, `add_technical_indicators()`, `df_to_array()`
  - Handle SPY-specific requirements: adjusted close, NaN validation, outlier flagging
  - Reference: `finrl/meta/data_processors/processor_yahoofinance.py` as template

- [ ] T006 Unit test data processor in `unit_tests/applications/spy_rl_trading/test_data_processor.py`:
  - Test download_data: verify OHLCV columns, date range, non-null values
  - Test clean_data: verify NaN removal, gap detection (99% threshold), outlier flagging
  - Test add_technical_indicators: verify 10 indicators computed, column count = 17
  - Test df_to_array: verify 3 arrays returned (price, tech, turbulence), correct shapes

- [ ] T007 [P] Implement `SPYTradingEnvironment` in `finrl/applications/spy_rl_trading/environment.py`:
  - Inherit from `gymnasium.Env`
  - Implement required methods: `__init__`, `reset`, `step`, `_get_state`, `_calculate_reward`
  - Action space: `Discrete(3)` = {0: BUY, 1: HOLD, 2: SELL}
  - Observation space: `Box(shape=(13,), dtype=float32)` = [balance, shares, price, 10 indicators, turbulence]
  - Reward function: log return if holding, 0 if flat
  - Reference: `finrl/meta/env_stock_trading/env_stocktrading.py` as template

- [ ] T008 Unit test environment in `unit_tests/applications/spy_rl_trading/test_environment.py`:
  - Test reset: verify observation shape (13,), balance = initial, shares = 0
  - Test step: verify action execution (buy/hold/sell), reward computation, state transitions
  - Test reward logic: log return when holding, 0 when flat
  - Test edge cases: multiple buys when holding (stay long), multiple sells when flat (stay flat)
  - Test normalization: all observations in valid range

### Configuration Setup

- [ ] T009 Configure `finrl/config.py` SPY-specific settings:
  - Add `SPY_SYMBOL = "SPY"`
  - Add date ranges: `SPY_TRAIN_START = "2020-01-01"`, `SPY_TRAIN_END = "2024-12-31"`, `SPY_TEST_START = "2025-01-01"`, `SPY_TEST_END = "2025-12-31"`
  - Add `SPY_INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix']`
  - Add `PPO_PARAMS` dict with hyperparameters per research.md §1

- [ ] T010 Create example config script: `finrl/applications/spy_rl_trading/config_example.py` showing how to override defaults

**Checkpoint**: Foundation complete - SPY data processor, trading environment, and configuration ready. All unit tests passing. Ready for US1 implementation.

---

## Phase 3: User Story 1 - Training and Validating SPY Trading Strategies (Priority: P1) 🎯 MVP

**Goal**: Quantitative researcher can train a PPO agent on 2020-2024 SPY data, observe converging rewards, and validate profitability

**Independent Test**:
1. Load SPY daily data for 2020-2024 (~1260 days)
2. Create trading environment with SPY processor output
3. Train PPO agent for 100K timesteps
4. Verify cumulative training return >5% (profitable on training set)
5. Verify reward curves smooth with TensorBoard logging

**Duration**: 4-6 days
**Dependencies**: Phase 2 complete

### Tests for User Story 1 (Test-First Implementation)

- [ ] T011 [P] [US1] Unit test PPO agent wrapper in `unit_tests/applications/spy_rl_trading/test_agent.py`:
  - Test agent initialization with PPO_PARAMS
  - Test training loop: verify total_timesteps executed
  - Test model save/load: verify deterministic behavior with fixed seed
  - Test convergence: verify reward curve increases over time

- [ ] T012 [P] [US1] Integration test full training pipeline in `unit_tests/applications/spy_rl_trading/test_pipeline.py`:
  - Test end-to-end: data download → clean → indicators → environment → training → model save
  - Verify training completes without errors
  - Verify final model saves successfully
  - Verify TensorBoard logs generated

### Implementation for User Story 1

- [ ] T013 [P] [US1] Implement `PPOAgent` wrapper in `finrl/applications/spy_rl_trading/agent.py`:
  - Wrapper around Stable-Baselines3 PPO
  - Methods: `__init__(env, config)`, `train(total_timesteps, tb_log_name)`, `save(path)`, `load(path)`
  - Integrate TensorBoard callback for monitoring
  - Use PPO_PARAMS from config.py for hyperparameters
  - Reference: `finrl/agents/stablebaselines3/models.py` as template

- [ ] T014 [P] [US1] Implement training pipeline in `finrl/applications/spy_rl_trading/pipeline.py`:
  - Function: `train_agent(config, symbol="SPY")` orchestrates full pipeline
  - Steps: download_data → clean_data → add_indicators → df_to_array → create_env → train_agent → save_model
  - Return: trained model object, training metrics dict
  - Error handling: graceful failures for missing data, training interruption

- [ ] T015 [US1] Create TensorBoard monitoring script in `finrl/applications/spy_rl_trading/monitor.py`:
  - Function to parse TensorBoard logs and plot reward curves
  - Display: episode return, policy loss, value loss, KL divergence
  - Utility: help users identify convergence issues

- [ ] T016 [US1] Implement metrics module in `finrl/applications/spy_rl_trading/metrics.py`:
  - Function: `compute_training_metrics(agent, env)` → dict with episode_return, steps, training_time
  - Includes: convergence check (reward increasing), stability check (low variance)

- [ ] T017 [US1] Add logging and validation in pipeline:
  - Add logger to pipeline.py, agent.py, environment.py
  - Log: data shape, training start/stop, convergence status
  - Validation: assert training return >0 before model save

- [ ] T018 [US1] Create example notebook: `finrl/applications/spy_rl_trading/Example_US1_Training.ipynb`
  - Step-by-step: load config, download data, train agent, visualize reward curves
  - Expected output: trained model saved, TensorBoard logs
  - Audience: quantitative researcher with basic Python knowledge

**Checkpoint**: User Story 1 complete - Researcher can train SPY agent, observe convergence, save trained model. MVP ready for demonstration.

---

## Phase 4: User Story 2 - Backtesting Trained Agent on Out-of-Sample Data (Priority: P2)

**Goal**: Trader can load trained agent, evaluate on 2025 test data, compute performance metrics (Sharpe, drawdown), compare to buy-and-hold baseline

**Independent Test**:
1. Load trained model from Phase 3
2. Create environment with 2025 test data
3. Run agent through test episodes without retraining
4. Compute Sharpe ratio, max drawdown, win rate
5. Compare metrics to buy-and-hold baseline

**Duration**: 3-4 days
**Dependencies**: Phase 2 complete, US1 complete (needs trained model from Phase 3)

### Tests for User Story 2

- [ ] T019 [P] [US2] Unit test backtest engine in `unit_tests/applications/spy_rl_trading/test_backtest.py`:
  - Test metrics computation: verify Sharpe ratio, max drawdown, win rate calculation
  - Test baseline computation: buy-and-hold return and Sharpe
  - Test determinism: identical results with same seed
  - Test edge cases: 100% win rate, negative returns

- [ ] T020 [P] [US2] Integration test backtesting pipeline in `unit_tests/applications/spy_rl_trading/test_backtest_pipeline.py`:
  - Test end-to-end: load model → create test env → run backtest → compute metrics
  - Verify output format: BacktestResult dict with all required fields

### Implementation for User Story 2

- [ ] T021 [P] [US2] Implement backtest engine in `finrl/applications/spy_rl_trading/backtest.py`:
  - Class: `Backtester` with method `run(model, test_env) → BacktestResult`
  - Evaluates trained agent on test data (no retraining)
  - Returns: dict with total_return, sharpe_ratio, max_drawdown, win_rate, baseline metrics
  - Ensure no lookahead bias: test data never seen during training
  - Reference: `finrl/plot.py` for metrics computation patterns

- [ ] T022 [US2] Implement metrics comparison in `finrl/applications/spy_rl_trading/metrics.py` (extend T016):
  - Function: `compare_to_baseline(agent_returns, spy_price_history)` → comparison dict
  - Compute: agent Sharpe vs. baseline Sharpe, agent return vs. baseline return, alpha (excess return)

- [ ] T023 [US2] Create reporting module in `finrl/applications/spy_rl_trading/report.py`:
  - Function: `generate_backtest_report(backtest_result)` → formatted string report
  - Output: Total return %, Sharpe ratio, max drawdown %, win rate, vs. baseline comparison

- [ ] T024 [US2] Add logging to backtest in `finrl/applications/spy_rl_trading/backtest.py`:
  - Log: backtest start/stop, daily trading decisions, final metrics
  - Validation: assert agent generates non-trivial trading signals (not 100% hold)

- [ ] T025 [US2] Create example notebook: `finrl/applications/spy_rl_trading/Example_US2_Backtesting.ipynb`
  - Load trained model from US1
  - Run backtest on 2025 test data
  - Display performance metrics and comparison charts
  - Audience: trader evaluating strategy

**Checkpoint**: User Story 2 complete - Trader can backtest trained agent, evaluate performance vs. baseline, generate reports.

---

## Phase 5: User Story 3 - PPO Hyperparameter Tuning and Optimization (Priority: P3)

**Goal**: Advanced researcher can train multiple PPO agents with different hyperparameters, compare convergence curves, identify optimal configuration

**Independent Test**:
1. Define 3+ hyperparameter configurations (learning rates, clip ratios)
2. Train separate agents for each config on same 2020-2024 data
3. Plot convergence curves side-by-side
4. Compute test Sharpe ratio for each trained model
5. Identify config with highest Sharpe ratio on 2025 test data

**Duration**: 3-5 days
**Dependencies**: Phase 2 complete, US1 complete (core training), US2 complete (metrics comparison)

### Tests for User Story 3

- [ ] T026 [P] [US3] Unit test hyperparameter sweep in `unit_tests/applications/spy_rl_trading/test_hyperparam_sweep.py`:
  - Test config generation: verify all hyperparameter combinations created
  - Test batch training: verify multiple models trained independently
  - Test comparison: verify metrics compared correctly across runs

### Implementation for User Story 3

- [ ] T027 [P] [US3] Implement hyperparameter sweep module in `finrl/applications/spy_rl_trading/hyperparam_sweep.py`:
  - Function: `grid_search(config_grid, train_env, test_env)` → results dict
  - Trains multiple agents with different PPO_PARAMS
  - Returns: convergence curves, test metrics for each config
  - Parameters: learning_rate, clip_ratio, entropy_coef, batch_size variations

- [ ] T028 [US3] Implement hyperparameter comparison/visualization in `finrl/applications/spy_rl_trading/hyperparam_analysis.py`:
  - Function: `plot_convergence_curves(results)` → matplotlib figure
  - Display: reward curves for each hyperparameter config, annotated with final Sharpe
  - Function: `find_best_config(results)` → best hyperparameter dict

- [ ] T029 [US3] Add configuration validation in `finrl/config.py`:
  - Validate PPO_PARAMS ranges (learning rate >0, clip ratio ∈ (0,1), etc.)
  - Warn if hyperparameters are outside recommended ranges

- [ ] T030 [US3] Create example notebook: `finrl/applications/spy_rl_trading/Example_US3_Hyperparameter_Tuning.ipynb`
  - Define grid of hyperparameter configs
  - Run grid search on 2020-2024 training data
  - Evaluate each trained model on 2025 test data
  - Plot convergence curves and Sharpe comparison
  - Recommend best configuration
  - Audience: advanced researcher optimizing strategy

**Checkpoint**: User Story 3 complete - Researcher can tune PPO hyperparameters, compare convergence, identify optimal configuration.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Production readiness, documentation, testing completeness
**Duration**: 3-5 days
**Blocker**: Done after all user stories

### Documentation & Examples

- [ ] T031 [P] Complete README: `finrl/applications/spy_rl_trading/README.md`
  - Overview of SPY RL trading system
  - Quick start: "Run Example_US1_Training.ipynb to train agent in 5 minutes"
  - Feature overview: training, backtesting, hyperparameter tuning
  - Architecture: integration with FinRL three-layer stack

- [ ] T032 [P] Create API documentation: `finrl/applications/spy_rl_trading/API.md`
  - Document all public functions and classes
  - Parameter descriptions, return types, examples
  - Reference contracts/API.md for detailed signatures

- [ ] T033 Create comprehensive guide: `finrl/applications/spy_rl_trading/GUIDE.md`
  - Installation and setup instructions
  - Data format requirements and validation
  - Training configuration and hyperparameter tuning
  - Backtesting methodology and interpretation
  - Troubleshooting section (common errors, fixes)

### Testing Completeness

- [ ] T034 [P] Run test coverage: `pytest unit_tests/applications/spy_rl_trading/ --cov`
  - Target: ≥80% coverage for data_processor.py, environment.py, agent.py
  - Generate coverage report

- [ ] T035 [P] Integration test full pipeline: `python finrl/applications/spy_rl_trading/integration_test.py`
  - Load config → download SPY data → train agent → backtest → generate report
  - Verify no errors, valid output

- [ ] T036 Create performance benchmark: `finrl/applications/spy_rl_trading/benchmark.py`
  - Measure: training time (target <30 min for 100K steps), backtest time (<5 min), inference latency (<10ms)
  - Compare to targets from plan.md

### Code Quality

- [ ] T037 [P] Run pre-commit checks: `black`, `isort`, `flake8` on all SPY module files
  - Ensure FinRL code style compliance

- [ ] T038 Add docstrings to all public functions and classes per FinRL standards

- [ ] T039 Final review: code review by FinRL contributor checklist
  - Verify Constitution compliance (Three-Layer, Data Processor, Gymnasium, Algorithms, Tests)
  - Verify no lookahead bias in backtesting
  - Verify TensorBoard logging enabled

**Checkpoint**: Production-ready SPY RL trading system with complete documentation, comprehensive testing, and performance validation.

---

## Summary: Task Organization & Parallelization

### Phase Sequence

```
Phase 1: Setup (T001-T004)
    ↓
Phase 2: Foundation (T005-T010) [BLOCKER: must complete before user stories]
    ├→ Phase 3: US1 Training (T011-T018) [P1: MVP]
    │   ├→ Phase 4: US2 Backtesting (T019-T025) [P2: depends on US1]
    │   │   └→ Phase 5: US3 Tuning (T026-T030) [P3: depends on US1+US2]
    │   │       └→ Phase 6: Polish (T031-T039) [Final: after all stories]
```

### Parallelizable Sections

**Phase 2**: T005 (data processor) and T007 (environment) can run in parallel
- **T005** (data processor) + **T007** (environment): No dependencies; different modules
- **T006** (processor test) depends on T005 completion
- **T008** (environment test) depends on T007 completion

**Phase 3 (US1)**: T011, T012, T013 can start after T006+T008 pass
- **T013** (PPO agent) and **T014** (pipeline) can run in parallel
- **T013** depends on Phase 2 complete
- **T014** depends on T013 implementation
- **T015-T018** sequential within US1 phase

**Phase 4 (US2)**: Depends on Phase 2 + US1 model available
- **T019** (backtest test) and **T021** (backtest engine) can run in parallel
- **T022-T024** sequential implementation

**Phase 5 (US3)**: Depends on Phase 2 + US1 + US2 complete
- **T026** (test) and **T027** (sweep) can start together

**Phase 6 (Polish)**: Can start after US3, runs in parallel
- **T031-T033** (docs) parallel
- **T034-T039** (testing) parallel

### Timeline Estimates

| Phase | Tasks | Duration | Critical |
|-------|-------|----------|----------|
| Phase 1 | T001-T004 | 1-2 days | Blocker |
| Phase 2 | T005-T010 | 3-5 days | Blocker |
| Phase 3 (US1) | T011-T018 | 4-6 days | ✅ MVP |
| Phase 4 (US2) | T019-T025 | 3-4 days | Depends on US1 |
| Phase 5 (US3) | T026-T030 | 3-5 days | Depends on US1+US2 |
| Phase 6 (Polish) | T031-T039 | 3-5 days | Final |
| **Total** | **39 tasks** | **17-27 days** | MVP: 8-13 days |

### MVP Scope (2-3 weeks)

**Minimum Viable Product** = Phase 1 + Phase 2 + Phase 3 (US1)
- Researcher can train PPO agent on SPY 2020-2024 data
- Observe converging reward curves via TensorBoard
- Save trained model to disk
- Validates FinRL integration (Three-Layer, Gymnasium, Tests)

**Additional** = Phase 4 (US2): Backtesting capability (+1 week)

**Complete** = Phase 5 (US3) + Phase 6: Full system with tuning & polish (+2 weeks)

---

## Task Checklist Format Validation

**Format Standard**: `- [ ] [TaskID] [P?] [Story?] Description with file path`

✅ All 39 tasks follow format:
- Checkbox: `- [ ]` prefix
- Task ID: T001, T002, ..., T039
- Parallelizable marker: `[P]` where applicable
- Story label: `[US1]`, `[US2]`, `[US3]` for user story tasks only
- Description: Specific action + file path(s)

Examples from task list:
- ✅ `- [ ] T001 Create project structure: finrl/applications/spy_rl_trading/`
- ✅ `- [ ] T005 [P] Implement SPYDataProcessor in finrl/applications/spy_rl_trading/data_processor.py`
- ✅ `- [ ] T012 [P] [US1] Unit test PPO agent wrapper in unit_tests/applications/spy_rl_trading/test_agent.py`
- ✅ `- [ ] T019 [P] [US2] Unit test backtest engine in unit_tests/applications/spy_rl_trading/test_backtest.py`

---

**Phase 1 Status**: ✅ Tasks generated. Ready for development.

**Next**: Use task list to begin Phase 1 setup. Commit tasks.md to feature branch.
