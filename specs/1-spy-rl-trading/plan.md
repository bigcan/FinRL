# Implementation Plan: SPY RL Trading System

**Branch**: `1-spy-rl-trading` | **Date**: 2025-10-27 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/1-spy-rl-trading/spec.md`

## Summary

Build a comprehensive reinforcement learning trading system for SPY (S&P 500 ETF) using FinRL framework and PPO agent. System ingests daily OHLCV data (2020-2025), trains a PPO agent via Stable-Baselines3 to maximize log returns using discrete actions (buy/hold/sell), and validates performance through backtesting on hold-out test data. Architecture integrates market data processors, Gymnasium trading environment, PPO agent, and performance analytics across FinRL's three-layer stack (Meta/Agents/Applications).

## Technical Context

**Language/Version**: Python 3.10+ (FinRL requirement)

**Primary Dependencies**:
- FinRL framework (meta layer: data processors, environments; agents layer: SB3 integration)
- Stable-Baselines3 (PPO implementation)
- Gymnasium (trading environment base class)
- Pandas, NumPy (data processing)
- YahooFinance / Alpaca API (market data)
- TensorBoard (training monitoring)
- Scikit-learn (performance metrics)

**Storage**: CSV/Parquet files (local disk) for historical OHLCV data and trained models; TensorBoard logs for training artifacts

**Testing**: pytest + pytest-cov for unit/integration tests; custom backtesting validation suite

**Target Platform**: Linux/macOS/Windows (development); cloud GPU optional for faster training

**Project Type**: Single project (Python package extending FinRL applications layer)

**Performance Goals**:
- Training convergence: <30 min for 100K timesteps on CPU (4-core)
- Backtest execution: <5 min for 1-year test period
- Model inference: <10ms per trading day decision

**Constraints**:
- Daily data only (no intraday features)
- No real-time trading (backtesting/paper trading deferred to v2.0)
- No lookahead bias (strict train/test split at specified date)
- Discrete action space (buy/hold/sell only)

**Scale/Scope**:
- Single asset (SPY)
- Single time period (2020-2025)
- ~1300 trading days training + ~250 trading days testing
- ~5-10K lines of code (environment, data processor, training pipeline, metrics)

## Constitution Check

**GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.**

### FinRL Constitution Compliance (v1.0.0)

| Principle | Requirement | Compliance | Notes |
|-----------|-------------|-----------|-------|
| **I. Three-Layer Architecture** | Meta (data/env), Agents (PPO), Applications (trading strategy) must be decoupled | ✅ PASS | Plan integrates FinRL's three layers: uses Meta data processors, Agents SB3-PPO, Applications trading script |
| **II. Data Processor Abstraction** | Consistent interface (download_data, clean_data, add_technical_indicator, add_vix, df_to_array) | ✅ PASS | Leverages existing YahooFinance processor in finrl/meta/data_processors/; adheres to standard interface |
| **III. Gymnasium Compliance** | Environment inherits from gymnasium.Env; implements __init__, reset, step, _get_state, _calculate_reward | ✅ PASS | Uses existing StockTradingEnv or creates minimal custom variant; implements all required methods |
| **IV. DRL Algorithm Abstraction** | Unified training interface (get_model, train_model); hyperparameters in config.py | ✅ PASS | Leverages FinRL's SB3 agent wrapper; stores PPO_PARAMS in finrl/config.py following convention |
| **V. Test-First & Observability** | Unit tests for processors/env/agent; TensorBoard logging mandatory | ✅ PASS | Plan includes unit tests for SPY processor variant, environment reward logic, and TensorBoard integration via SB3 callbacks |

**Gate Status**: ✅ **PASS** - Plan complies with all five FinRL principles. No waivers needed.

## Project Structure

### Documentation (this feature)

```text
specs/1-spy-rl-trading/
├── spec.md                  # Feature specification
├── plan.md                  # This file
├── research.md              # Phase 0 (TBD)
├── data-model.md            # Phase 1 (TBD)
├── quickstart.md            # Phase 1 (TBD)
├── contracts/               # Phase 1 (TBD)
└── checklists/
    └── requirements.md      # Quality checklist
```

### Source Code (repository root, extending FinRL)

```text
# NEW: SPY-specific trading application
finrl/applications/spy_rl_trading/
├── __init__.py
├── config.py                # SPY config (dates, tickers, paths, PPO_PARAMS)
├── data_processor.py        # SPY YahooFinance processor (wraps existing processor)
├── environment.py           # SPY trading environment (extends StockTradingEnv)
├── agent.py                 # PPO agent trainer (wraps SB3 DRLAgent)
├── pipeline.py              # Train-test-trade orchestrator
├── metrics.py               # Backtest performance analytics
├── backtest.py              # Backtesting engine
└── example_notebook.ipynb   # End-to-end tutorial

# MODIFIED: FinRL core config (add SPY_PARAMS section)
finrl/config.py              # Add SPO_PARAMS, SPY_DATA_DATES, etc.

# TESTS: Unit and integration tests
unit_tests/applications/spy_rl_trading/
├── test_data_processor.py   # SPY data ingestion validation
├── test_environment.py      # SPY trading env reward logic
├── test_agent.py            # PPO training convergence
└── test_pipeline.py         # End-to-end train-test workflow
```

**Structure Decision**: Single project structure extending FinRL's applications layer. SPY trading system is implemented as a standalone application module (finrl/applications/spy_rl_trading/) that reuses Meta layer (data processors, environments) and Agents layer (SB3 PPO integration) via FinRL's standard interfaces. This adheres to FinRL Constitution Principle I (Three-Layer Architecture).

## Complexity Tracking

> **Gate Waivers**: None required. Plan complies with FinRL Constitution (v1.0.0) without deviations.

| Item | Status | Notes |
|------|--------|-------|
| Architecture Alignment | ✅ | Three-layer FinRL design preserved |
| Dependency Footprint | ✅ | All dependencies already in FinRL (Stable-Baselines3, Gymnasium, Pandas) |
| Test Coverage | ✅ | Unit tests for processor, environment, agent; integration test for full pipeline |
| Documentation | ✅ | Spec complete; plan outlines research→design→tasks phases |

## Phase 0: Research & Unknowns

### Clarifications Resolved

1. **PPO Hyperparameter Defaults**
   - Task: Identify recommended PPO hyperparameters for discrete trading
   - Output: PPO_PARAMS in finrl/config.py (learning_rate, clip_ratio, batch_size, n_steps, etc.)

2. **Environment Reward Scaling**
   - Task: Validate log-return reward scaling for training stability
   - Output: Reward computation formula documented in environment.py docstring

3. **Data Quality Standards**
   - Task: Define acceptance criteria for SPY data (missing values, outliers, gaps)
   - Output: Data validation rules in data_processor.py.clean_data()

### Research Tasks (Dispatch)

```
Task 1: "Research optimal PPO hyperparameters for discrete action trading (FinRL context)"
Task 2: "Research log-return reward scaling for policy gradient stability"
Task 3: "Research data quality validation patterns for financial time series"
Task 4: "Research TensorBoard callback integration in Stable-Baselines3 training loops"
Task 5: "Research Sharpe ratio computation for backtesting evaluation"
```

**Phase 0 Deliverable**: `research.md` containing decision summaries and rationale for all unknowns.

## Phase 1: Design & Data Model

### Data Model (research.md → data-model.md)

Entities extracted from spec:

- **MarketData**: OHLCV candles with technical indicators
- **TradingEnvironment**: State [balance, shares, price, indicators, turbulence] → Actions {buy, hold, sell} → Rewards (log return)
- **PPOAgent**: Policy network trained via Stable-Baselines3
- **BacktestResult**: Performance metrics (return %, Sharpe, max drawdown, win rate)

### API Contracts (Phase 1)

Contracts for key components:

- `DataProcessor.download_data()` → DataFrame (OHLCV + indicators)
- `TradingEnvironment.reset()` → state dict
- `TradingEnvironment.step(action)` → (state, reward, done, info)
- `PPOAgent.train()` → trained model + logs
- `Backtester.run(model, test_data)` → BacktestResult

**Output**: `contracts/data_processor_api.md`, `contracts/environment_api.md`, `contracts/agent_api.md`

### Agent Context Update (Phase 1)

- Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType claude`
- Update Claude Code context with SPY RL system: PPO agent, discrete action space, log-return rewards
- Preserve existing FinRL guidance; add new SPY-specific patterns

### Quickstart (Phase 1)

Generate `quickstart.md`: Step-by-step tutorial for researcher to train SPY agent and validate on test data.

## Phase 1 Completion Summary

### Artifacts Generated

✅ **research.md** (Phase 0): All 7 research questions resolved with evidence-based decisions
- PPO hyperparameters: lr=3e-4, clip=0.2, net=[256,256]
- Reward function: Log returns scaled by position
- Data quality: 99% completeness, 5σ outlier detection
- TensorBoard integration: SB3 native logging
- Backtest metrics: Sharpe, max drawdown, win rate
- Data splitting: Chronological train/test split (2020-2024/2025)
- Technical indicators: FinRL standard 10 indicators + VIX

✅ **data-model.md** (Phase 1): 6 core entities defined with validation rules
- MarketData: OHLCV + 10 technical indicators
- EnvironmentState: 13-dimensional observation space
- Action: Discrete {BUY, HOLD, SELL}
- Reward: Log-return scaled by position
- Episode: Training episode metrics
- BacktestResult: Performance summary with Sharpe ratio

✅ **contracts/API.md** (Phase 1): Complete API specifications for 4 modules
- DataProcessor: download_data, clean_data, add_indicators, df_to_array
- TradingEnvironment: reset, step, _get_state, _calculate_reward
- PPOAgent: train, save, load
- Backtester: run (agent evaluation)

✅ **quickstart.md** (Phase 1): Step-by-step tutorial for SPY RL trading
- Environment setup with Poetry
- Data preparation script
- Training script with TensorBoard monitoring
- Backtest script with performance metrics
- Hyperparameter tuning guide
- Troubleshooting and next steps

✅ **CLAUDE.md** (Agent Context Update): Updated with SPY RL system context
- Added Python 3.10+ language requirement
- Added CSV/Parquet storage for OHLCV data and models
- Added TensorBoard logging for training artifacts

### Constitution Check Re-Evaluation

**Post-Design Gate Status**: ✅ **PASS**

All five FinRL Constitution principles remain compliant after Phase 1 design:

| Principle | Compliance | Notes |
|-----------|-----------|-------|
| **I. Three-Layer Architecture** | ✅ PASS | Plan extends Applications layer while reusing Meta (data processors, environments) and Agents (SB3 PPO) layers |
| **II. Data Processor Abstraction** | ✅ PASS | Uses existing YahooFinance processor; follows standard interface |
| **III. Gymnasium Compliance** | ✅ PASS | Extends StockTradingEnv; implements all required methods (reset, step, _get_state, _calculate_reward) |
| **IV. DRL Algorithm Abstraction** | ✅ PASS | Leverages SB3 PPO via FinRL agent wrapper; hyperparameters in config.py |
| **V. Test-First & Observability** | ✅ PASS | Unit tests planned for all components; TensorBoard integration mandatory |

**No waivers required.** Design fully complies with FinRL Constitution v1.0.0.

## Phase 2: Tasks (Deferred to /speckit.tasks)

Phase 2 generates actionable task list in `tasks.md`:
- Data ingestion & validation tasks
- Environment implementation tasks
- Agent training pipeline tasks
- Backtesting & metrics tasks
- Documentation & testing tasks

**/speckit.tasks command generates tasks.md** (not /speckit.plan)

## Next Steps

1. ✅ **Phase 0 Complete**: research.md generated with all clarifications resolved
2. ✅ **Phase 1 Complete**: data-model.md, contracts/API.md, quickstart.md generated
3. ✅ **Agent Context Updated**: CLAUDE.md updated with SPY RL system context
4. **Phase 2 Ready**: Execute `/speckit.tasks` to generate actionable `tasks.md`
5. **Implementation**: Follow task list; execute development in dependency order

---

**Status**: ✅ **Phase 0-1 COMPLETE**. Ready for `/speckit.tasks` to generate Phase 2 implementation tasks.
