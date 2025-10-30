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

**Status**: ✅ **Phase 0-6 COMPLETE**. Production-ready implementation delivered.

---

## PHASE 2: PRODUCTION HARDENING (NEW)

**Status**: 🆕 Planning
**Duration**: 6 weeks
**Timeline**: November-December 2025
**Dependencies**: Phases 1-6 (COMPLETE ✅)

### Overview

Transform the production-ready SPY RL Trading System into an enterprise-grade platform with:
- Deterministic builds and reproducible environments
- Automated CI/CD with comprehensive quality gates
- Production observability and configuration management
- Containerized deployment infrastructure
- Finance-specific controls (realistic costs, risk limits, live trading)

### Goals

**G7: Dependency Management**
- Achieve deterministic, reproducible builds across environments
- Pin all dependencies with appropriate version ranges
- Resolve library conflicts and pre-release pins
- Eliminate setuptools artifacts, standardize on Poetry

**G8: CI/CD Pipeline**
- Automated testing across Python 3.10, 3.11, 3.12
- Comprehensive coverage reporting (≥80% threshold)
- Security scanning (pip-audit, CodeQL, Bandit)
- Automated releases with SBOM generation
- Deterministic test fixtures for offline testing

**G9: Observability & Operations**
- Structured JSON logging for production environments
- Centralized configuration with environment variable support
- Reproducible results with global seed management
- Remove all hardcoded paths and constants
- Production-grade error handling and monitoring

**G10: Containerization**
- Multi-stage Docker images for training and deployment
- Jupyter development environment for research
- docker-compose orchestration for multi-service setup
- Multi-architecture builds (amd64, arm64) via GitHub Actions

**G11: Finance-Specific Controls**
- Realistic transaction costs (commission, spread, market impact, SEC fees)
- Volume-based slippage model
- Market calendar integration for trading day validation
- Position and risk limits (drawdown, daily loss, position size)

**G12: Live Trading Infrastructure**
- Broker adapter pattern with Alpaca implementation
- Trading mode switching (backtest, paper, live)
- Dry-run and simulation capabilities
- Paper trading validation and safety checks
- Monitoring and alerting (optional)

---

## Phase 7: Dependency Management (Week 1)

**Goal**: Deterministic, reproducible builds with locked dependencies

**Duration**: 1 week (20 hours)
**Tasks**: T040-T045 (6 tasks)

**Key Activities**:
1. Remove setuptools artifacts (setup.py, requirements.txt)
2. Clean and consolidate pyproject.toml
3. Pin Python version to ^3.10,<3.13
4. Resolve dependency conflicts:
   - Remove alpaca-trade-api, migrate to alpaca-py
   - Pin stable-baselines3 to stable 2.x
   - Pin ray[rllib] to 2.9.x
5. Generate poetry.lock with appropriate version ranges
6. Verify reproducible builds across Python 3.10/3.11/3.12

**Success Criteria**: SC-H001
- All dependencies pinned in poetry.lock
- Zero high/critical security vulnerabilities
- Builds reproducible across Linux, macOS, Windows

**Deliverables**:
- Clean pyproject.toml (Poetry only)
- poetry.lock with pinned versions
- Updated installation documentation
- Dependency management policy documented

**Risks**:
- Medium: Library conflicts may require code changes in agent wrappers
- Mitigation: Extensive CI testing on multiple Python versions

---

## Phase 8: CI/CD Pipeline (Week 1-2)

**Goal**: Automated testing, quality gates, security scanning, releases

**Duration**: 1-2 weeks (44 hours)
**Tasks**: T046-T052 (7 tasks)

**Key Activities**:
1. GitHub Actions workflow with Python version matrix
2. Coverage reporting with codecov.io (≥80% threshold)
3. Deterministic test fixtures (spy_data_2020_2023.parquet)
4. Security scanning:
   - pip-audit for dependency vulnerabilities
   - CodeQL for static analysis
   - Bandit for security linting
5. License checking and SBOM generation
6. Automated release workflow (tag → build → publish)
7. Separate integration test workflow

**Success Criteria**: SC-H002
- Tests passing on Python 3.10, 3.11, 3.12
- Coverage maintained ≥80%
- Automated releases functional
- Security scans clean

**Deliverables**:
- .github/workflows/ci.yml (test matrix)
- .github/workflows/release.yml (automated releases)
- .github/workflows/docker.yml (container builds)
- .coveragerc (coverage configuration)
- unit_tests/fixtures/ (test data snapshots)
- scripts/generate_test_fixtures.py

**Risks**:
- Low: Test fixtures may become stale
- Mitigation: Quarterly regeneration schedule

---

## Phase 9: Test Hardening (Week 2-3)

**Goal**: Robust, isolated test suite with proper separation

**Duration**: 1-2 weeks (34 hours)
**Tasks**: T053-T058 (6 tasks)

**Key Activities**:
1. Audit and fix brittle tests (remove exact row counts)
2. Reorganize test structure:
   - unit_tests/.../unit/ (fast, offline)
   - unit_tests/.../integration/ (slow, end-to-end)
3. Configure pytest markers (unit, integration, network)
4. Property-based testing with Hypothesis (optional)
5. Fix flaky tests (add seeds, remove race conditions)
6. Update GUIDE.md test documentation

**Success Criteria**: SC-H002 (maintained)
- All tests reliable and fast
- Unit/integration properly separated
- No flaky tests in CI

**Deliverables**:
- Reorganized test directory structure
- pytest.ini with markers
- Fixed brittle assertions
- Property-based tests (optional)
- Updated test documentation

**Risks**:
- Low: Test reorganization may break imports
- Mitigation: Incremental migration with validation

---

## Phase 10: Runtime & Operations (Week 3-4)

**Goal**: Production observability with structured logging and configuration

**Duration**: 1-2 weeks (40 hours)
**Tasks**: T059-T064 (6 tasks)

**Key Activities**:
1. Structured logging with structlog:
   - JSON format for production
   - Console format for development
   - Training, backtesting, system metrics
2. Centralized configuration with pydantic-settings:
   - SPYTradingSettings class
   - Environment variable support (SPY_*)
   - .env file support
3. Remove all hardcoded paths (use settings)
4. Reproducibility hooks:
   - set_global_seed() function
   - Deterministic torch configuration
5. Update GUIDE.md for operations

**Success Criteria**: SC-H003
- Structured JSON logging operational
- All settings configurable via environment
- Seeds enable reproducibility

**Deliverables**:
- logging_config.py (structlog configuration)
- settings.py (pydantic-settings)
- reproducibility.py (seed management)
- .env.example (environment template)
- Updated GUIDE.md (operations section)

**Risks**:
- Low: Configuration migration may miss edge cases
- Mitigation: Comprehensive code review

---

## Phase 11: Docker & Containerization (Week 4)

**Goal**: Containerized deployment with multi-stage builds

**Duration**: 1 week (21 hours)
**Tasks**: T065-T069 (5 tasks)

**Key Activities**:
1. Multi-stage Dockerfile (builder → runtime)
2. Jupyter development image
3. docker-compose.yml for orchestration:
   - Services: training, backtesting, jupyter
   - Volumes: data, models, results
4. Multi-architecture builds (amd64, arm64)
5. Update GUIDE.md for Docker usage

**Success Criteria**: SC-H004
- Docker images build successfully
- Multi-arch support (amd64, arm64)
- docker-compose working

**Deliverables**:
- docker/Dockerfile (production runtime)
- docker/Dockerfile.dev (Jupyter development)
- docker-compose.yml (orchestration)
- .dockerignore
- .github/workflows/docker.yml
- Updated GUIDE.md (Docker section)

**Risks**:
- Low: Docker build failures on ARM64
- Mitigation: Multi-stage builds with error handling

---

## Phase 12: Finance-Specific Hardening (Week 5-6)

**Goal**: Production trading infrastructure with realistic costs and risk limits

**Duration**: 2 weeks (83 hours)
**Tasks**: T070-T079 (10 tasks)

**Key Activities**:
1. Transaction cost model:
   - Commission, spread, market impact, SEC fees
   - Default SPY parameters (1bp spread, 2bp impact)
2. Slippage model with volume impact
3. Market calendar integration (NYSE trading days)
4. Position and risk limits:
   - Max drawdown (20% default)
   - Daily loss limit (5% default)
   - Position size limits
5. Broker adapter pattern:
   - Abstract BrokerAdapter base class
   - AlpacaBroker implementation
   - Retry logic with exponential backoff
6. Trading modes (backtest, paper, live)
7. Dry-run and simulation capabilities
8. Monitoring and alerting (optional)
9. Paper trading validation
10. Finance documentation updates

**Success Criteria**: SC-H005, SC-H006
- Transaction costs realistic (<1% impact)
- Risk limits prevent >20% drawdown
- Paper trading functional
- Broker adapter implemented
- Monitoring operational (optional)

**Deliverables**:
- transaction_costs.py (cost model)
- risk_limits.py (position/risk controls)
- brokers/base.py (adapter interface)
- brokers/alpaca.py (Alpaca implementation)
- trading_engine.py (mode switching)
- monitoring/ (Prometheus/Grafana, optional)
- Updated GUIDE.md (finance section)

**Risks**:
- High: Broker API integration complexity
- Mitigation: Adapter pattern with comprehensive testing

---

## Updated Timeline

### Completed Work (Phases 1-6)
**October 2025**: Implementation complete
- 37/39 tasks (95%)
- 2 tasks deferred (covered by integration tests)
- Status: ✅ Production-ready

### New Work (Phases 7-12)
**November-December 2025**: Production hardening
- Week 1: Phase 7 (Dependency Management)
- Week 1-2: Phase 8 (CI/CD Pipeline)
- Week 2-3: Phase 9 (Test Hardening)
- Week 3-4: Phase 10 (Runtime & Operations)
- Week 4: Phase 11 (Docker & Containerization)
- Week 5-6: Phase 12 (Finance-Specific Hardening)

**Total**: 40 tasks, 242 hours, 6 weeks

---

## Success Criteria (Phase 2)

### SC-H001: Dependency Management
- All dependencies pinned in poetry.lock
- Zero high/critical security vulnerabilities
- Build reproducible across environments

### SC-H002: CI/CD
- Tests passing on Python 3.10, 3.11, 3.12
- Coverage maintained ≥80%
- Automated releases functional

### SC-H003: Observability
- Structured JSON logging operational
- All settings configurable via environment
- Seeds enable reproducibility

### SC-H004: Containerization
- Docker images build successfully
- Multi-arch support (amd64, arm64)
- docker-compose working

### SC-H005: Finance Controls
- Transaction costs realistic (<1% impact)
- Risk limits prevent >20% drawdown
- Paper trading functional

### SC-H006: Live Trading
- Broker adapter implemented
- Dry-run mode operational
- Monitoring active (optional)

---

## Risk Management

### High Priority Risks

1. **Dependency conflicts break training**
   - Impact: High
   - Likelihood: Medium
   - Mitigation: Extensive CI matrix testing on Python 3.10/3.11/3.12

2. **Broker API changes**
   - Impact: High
   - Likelihood: Medium
   - Mitigation: Adapter pattern with versioning, comprehensive error handling

3. **Live trading losses**
   - Impact: Critical
   - Likelihood: Low
   - Mitigation: Extensive paper trading validation, risk limits, safety checks

### Medium Priority Risks

1. **Test fixtures become stale**
   - Impact: Medium
   - Likelihood: High
   - Mitigation: Quarterly regeneration schedule, automated validation

2. **Docker build failures**
   - Impact: Low
   - Likelihood: Medium
   - Mitigation: Multi-stage builds with error handling, multi-arch testing

---

**Status**: ✅ **Phases 1-6 COMPLETE**. 🆕 **Phases 7-12 PLANNED**. Ready to begin hardening implementation.
