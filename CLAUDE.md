# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinRL is a financial reinforcement learning framework implementing a three-layer architecture:
- **Meta Layer** (`finrl/meta/`): Market environments, data processors, and preprocessors
- **Agents Layer** (`finrl/agents/`): DRL algorithm implementations (ElegantRL, Stable-Baselines3, RLlib)
- **Applications Layer** (`finrl/applications/`): Domain-specific trading strategies

The core workflow follows a **train-test-trade** pipeline orchestrated through `train.py`, `test.py`, and `trade.py`.

## Environment Setup

### Installation
```bash
# Using Poetry (recommended)
poetry install
poetry shell

# Or using pip (editable install)
pip install -e .
```

### Pre-commit Hooks
```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

### API Configuration
- Copy `.env.example` to `.env` if using API-based data sources
- Set credentials in `finrl/config_private.py` for Alpaca trading:
  - `ALPACA_API_KEY`
  - `ALPACA_API_SECRET`

## Development Commands

### Running the Main Pipeline
```bash
# Train a model
python finrl/main.py --mode=train
python -m finrl --mode=train

# Test a trained model
python finrl/main.py --mode=test

# Live/paper trading
python finrl/main.py --mode=trade
```

### Running Individual Modules
```bash
# Direct training with custom parameters
python finrl/train.py

# Testing with specific model
python finrl/test.py

# Paper trading
python finrl/trade.py
```

### Testing
```bash
# Run all tests
poetry run pytest unit_tests

# Run specific test module
poetry run pytest unit_tests/environments/test_env_cashpenalty.py

# Run with pattern matching
poetry run pytest -k "downloader"
```

### Code Formatting
```bash
# Format all code (via pre-commit hooks)
poetry run pre-commit run --all-files

# Individual tools
poetry run black finrl/
poetry run isort finrl/
poetry run flake8 finrl/
```

## Architecture Deep Dive

### Three-Layer System

**1. Meta Layer** (`finrl/meta/`)
- **Data Processors** (`data_processors/`): Source-specific data fetchers and cleaners
  - `processor_yahoofinance.py`: Yahoo Finance integration (default)
  - `processor_alpaca.py`: Alpaca API integration
  - `processor_wrds.py`: WRDS TAQ data
  - `processor_ccxt.py`: Cryptocurrency exchanges
  - Each processor implements: `download_data()`, `clean_data()`, `add_technical_indicator()`, `add_vix()`, `df_to_array()`

- **Environments** (`env_*/`): Gymnasium-compatible trading environments
  - `env_stock_trading/`: Stock trading environments with variants for cash penalties, stop-loss, numpy-optimized versions
  - `env_cryptocurrency_trading/`: Crypto-specific environments
  - `env_portfolio_allocation/`: Portfolio rebalancing environments
  - `env_portfolio_optimization/`: Modern portfolio theory implementations

- **Preprocessors** (`preprocessor/`): Feature engineering and data transformation
  - Technical indicators via `stockstats` library
  - Turbulence index calculation
  - VIX integration

**2. Agents Layer** (`finrl/agents/`)
- **ElegantRL** (`elegantrl/`): High-performance RL algorithms
- **Stable-Baselines3** (`stablebaselines3/`): Standard RL library integration
  - Includes hyperparameter optimization (`hyperparams_opt.py`)
  - Ray Tune integration (`tune_sb3.py`)
- **RLlib** (`rllib/`): Distributed RL via Ray
- **Portfolio Optimization** (`portfolio_optimization/`): Mean-variance optimization and deep portfolio methods

**3. Applications Layer** (`finrl/applications/`)
- **Stock Trading**: Ensemble methods, fundamental analysis, rolling window strategies
- **Cryptocurrency Trading**: Crypto-specific strategies
- **High Frequency Trading**: Sub-minute trading applications
- **Portfolio Allocation**: Asset allocation strategies

### Data Flow

1. **Download**: `DataProcessor` → source-specific processor → raw OHLCV data
2. **Clean**: Missing value handling, timezone alignment
3. **Feature Engineering**: Technical indicators, VIX, turbulence
4. **Array Conversion**: `df_to_array()` → `(price_array, tech_array, turbulence_array)`
5. **Environment Setup**: Arrays → `env_config` → Gymnasium environment
6. **Agent Training**: Environment → DRL agent → trained model
7. **Backtesting**: Trained model → test environment → performance metrics
8. **Trading**: Trained model → live/paper trading environment → orders

### Configuration System

- **`finrl/config.py`**: Global settings
  - Date ranges: `TRAIN_START_DATE`, `TRAIN_END_DATE`, `TEST_START_DATE`, etc.
  - Model hyperparameters: `A2C_PARAMS`, `PPO_PARAMS`, `DDPG_PARAMS`, `TD3_PARAMS`, `SAC_PARAMS`, `ERL_PARAMS`
  - Technical indicators: `INDICATORS` list
  - Directory paths: `DATA_SAVE_DIR`, `TRAINED_MODEL_DIR`, `TENSORBOARD_LOG_DIR`, `RESULTS_DIR`

- **`finrl/config_tickers.py`**: Predefined ticker lists
  - `DOW_30_TICKER`, `NAS_100_TICKER`, `SP_500_TICKER`, etc.

- **`finrl/config_private.py`**: API credentials (gitignored)

## Data Sources

The framework supports multiple data sources via processor abstraction:

| Source | Coverage | Frequency | Processor |
|--------|----------|-----------|-----------|
| Yahoo Finance | US/Global stocks | 1min-1day | `processor_yahoofinance.py` |
| Alpaca | US stocks/ETFs | 1min-1day | `processor_alpaca.py` |
| WRDS | US securities | 1ms-1day | `processor_wrds.py` |
| CCXT | Cryptocurrency | 1s-1day | `processor_ccxt.py` |
| EOD Historical | Global stocks | 1min-1day | `processor_eodhd.py` |
| JoinQuant | CN securities | 1min-1day | `processor_joinquant.py` |
| Sinopac | Taiwan securities | 1min-1day | `processor_sinopac.py` |

Each processor implements a consistent interface for seamless switching.

## DRL Library Integration

### ElegantRL (Default)
```python
from finrl.agents.elegantrl.models import DRLAgent
agent = DRLAgent(env=env, price_array=price_array,
                 tech_array=tech_array, turbulence_array=turbulence_array)
model = agent.get_model("ppo", model_kwargs=ERL_PARAMS)
trained_model = agent.train_model(model=model, cwd="./output", total_timesteps=1e5)
```

### Stable-Baselines3
```python
from finrl.agents.stablebaselines3.models import DRLAgent
agent = DRLAgent(env=env_instance)
model = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
trained_model = agent.train_model(model=model, total_timesteps=1e6)
```

### RLlib
```python
from finrl.agents.rllib.models import DRLAgent
agent = DRLAgent(env=env, price_array=price_array,
                 tech_array=tech_array, turbulence_array=turbulence_array)
model, config = agent.get_model("ppo")
trained_model = agent.train_model(model=model, model_name="ppo",
                                  model_config=config, total_episodes=100)
```

## Environment Variants

- **`env_stocktrading.py`**: Base stock trading environment
- **`env_stocktrading_np.py`**: NumPy-optimized version for faster training
- **`env_stocktrading_cashpenalty.py`**: Penalizes holding excessive cash
- **`env_stocktrading_stoploss.py`**: Implements stop-loss mechanisms
- **`env_stock_papertrading.py`**: Real-time paper trading with Alpaca integration

Choose based on training speed requirements and desired trading constraints.

## Jupyter Notebooks

Example workflows are in `examples/`:
- **Stock_NeurIPS2018_*.ipynb**: Complete train-test-backtest pipeline (NeurIPS 2018 paper)
- **FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb**: Ensemble strategy (ICAIF 2020 paper)
- **FinRL_PaperTrading_Demo.ipynb**: Live paper trading demonstration
- **FinRL_PortfolioOptimizationEnv_Demo.ipynb**: Portfolio optimization examples

## Key Conventions

### Model Training
- Models are saved to `cwd` parameter (e.g., `"./test_ppo"`)
- TensorBoard logs go to `TENSORBOARD_LOG_DIR`
- Use `break_step` (ElegantRL) or `total_timesteps` (SB3) for training duration

### Technical Indicators
Default indicators in `config.py`:
- `macd`: Moving Average Convergence Divergence
- `boll_ub`, `boll_lb`: Bollinger Bands (upper/lower)
- `rsi_30`: 30-period Relative Strength Index
- `cci_30`: 30-period Commodity Channel Index
- `dx_30`: 30-period Directional Movement Index
- `close_30_sma`, `close_60_sma`: Simple Moving Averages

Add new indicators via `stockstats` library syntax.

### Environment State Space
- **State**: `[balance, shares_held..., price..., tech_indicators..., turbulence]`
- **Action**: Continuous values for position sizing (typically [-1, 1] per stock)
- **Reward**: Portfolio value change (can be customized per environment)

## Code Style

- **PEP 8** with 127-character line limit
- **Black** for auto-formatting
- **isort** for import ordering with `from __future__ import annotations`
- **flake8** ignores: `F401` (unused imports), `W503` (line break before binary operator), `E203` (whitespace before colon)
- Use lowercase_with_underscores for functions/variables, UpperCamelCase for classes

## Common Development Patterns

### Adding a New Data Source
1. Create `finrl/meta/data_processors/processor_newsource.py`
2. Implement required methods: `download_data()`, `clean_data()`, `add_technical_indicator()`, `add_vix()`, `df_to_array()`
3. Register in `finrl/meta/data_processor.py` `__init__` method
4. Add unit tests in `unit_tests/downloaders/`

### Creating a Custom Environment
1. Subclass `gymnasium.Env` or extend existing environment in `finrl/meta/env_stock_trading/`
2. Implement: `__init__()`, `reset()`, `step()`, `_get_state()`, `_calculate_reward()`
3. Define `action_space` and `observation_space` (Gymnasium spaces)
4. Add unit test in `unit_tests/environments/`

### Training with Custom Hyperparameters
```python
custom_params = {
    "n_steps": 2048,
    "learning_rate": 0.0003,
    "batch_size": 128,
    "ent_coef": 0.01,
}

train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=DOW_30_TICKER,
    data_source="yahoofinance",
    time_interval="1D",
    technical_indicator_list=INDICATORS,
    drl_lib="stable_baselines3",
    env=StockTradingEnv,
    model_name="ppo",
    agent_params=custom_params,
    total_timesteps=1e6,
)
```

## Specification Management with Speckit

### Overview

This project uses **speckit** (a structured specification framework) to maintain comprehensive feature specifications in the `specs/` directory. All major features and structural changes MUST be documented using speckit before implementation.

**Philosophy**: Specifications are the single source of truth for feature design, requirements, and implementation plans.

### Speckit Structure

Each feature specification lives in `specs/{feature-number}-{feature-name}/`:

```
specs/
└── 1-spy-rl-trading/              # Feature specification directory
    ├── spec.md                     # Core specification (user stories, requirements, success criteria)
    ├── plan.md                     # Implementation plan (phases, timeline, goals)
    ├── tasks.md                    # Task breakdown (T001, T002, etc.)
    ├── data-model.md               # Entity definitions and relationships
    ├── research.md                 # Technical research and decisions
    ├── quickstart.md               # Getting started guide
    ├── hardening.md                # Production hardening details (if applicable)
    ├── deployment.md               # Deployment strategies (if applicable)
    ├── observability.md            # Logging/monitoring (if applicable)
    └── contracts/                  # Interface contracts
        ├── API.md                  # API specifications
        ├── BROKER_ADAPTER.md       # Broker interface
        ├── TRANSACTION_COST.md     # Cost modeling
        └── RISK_LIMITS.md          # Risk management
```

### Core Specification Files

**spec.md** (Required):
- User stories with acceptance criteria
- Functional and non-functional requirements
- Success criteria (measurable outcomes)
- Assumptions, constraints, and out-of-scope items
- Testing strategy

**plan.md** (Required):
- Implementation phases with durations
- Goals and deliverables
- Dependencies and risks
- Timeline and milestones

**tasks.md** (Required):
- Granular task breakdown (T001, T002, etc.)
- Task dependencies and estimates
- Acceptance criteria per task
- Progress tracking (✅ complete, ⏳ in progress, ❌ blocked)

**contracts/** (Optional):
- API contracts and interface definitions
- Data model contracts
- Integration contracts for external systems

### When to Update Specifications

**ALWAYS update specs when**:
1. **Adding new features**: Create complete spec before coding
2. **Structural changes**: Update `spec.md` and `plan.md` with architectural changes
3. **New phases**: Add phases to `plan.md` and tasks to `tasks.md`
4. **Production hardening**: Add `hardening.md`, `deployment.md`, `observability.md`
5. **API changes**: Update `contracts/` with new interfaces
6. **Success criteria changes**: Update measurable outcomes in `spec.md`

**Example workflow** (SPY RL Trading System hardening):
```bash
# 1. User requests production hardening
# 2. Analyze requirements and create comprehensive hardening plan
# 3. Update existing spec files (DO NOT create new spec directory)
#    - tasks.md: Add Phase 7-12 with 40 new tasks (T040-T079)
#    - plan.md: Add Phase 2 section with goals, timelines
#    - spec.md: Add hardening goals (G7-G12) and success criteria (SC-H001 to SC-H006)
# 4. Create supporting docs (hardening.md, deployment.md, observability.md)
# 5. Create contracts (BROKER_ADAPTER.md, TRANSACTION_COST.md, RISK_LIMITS.md)
# 6. Commit all spec changes together
# 7. Implement following spec as single source of truth
```

### Specification Update Workflow

**Step 1: Identify Scope**
```bash
# Determine if changes fit existing spec or need new feature
# - Extension: Update existing spec (e.g., 1-spy-rl-trading)
# - New feature: Create new spec directory (e.g., 2-multi-asset-trading)
```

**Step 2: Update Core Files**
```bash
# Update in order:
# 1. spec.md - Add goals, requirements, success criteria
# 2. plan.md - Add phases, timeline, dependencies
# 3. tasks.md - Add granular tasks with estimates
```

**Step 3: Add Supporting Documentation**
```bash
# Create as needed:
# - hardening.md - Production readiness details
# - deployment.md - Deployment strategies
# - observability.md - Logging/monitoring
# - contracts/*.md - Interface definitions
```

**Step 4: Commit Specifications**
```bash
# Commit spec updates BEFORE implementation
git add specs/{feature-name}/
git commit -m "docs: add {feature-name} specifications"
git push
```

**Step 5: Implement Following Spec**
```bash
# Use spec as single source of truth
# - Refer to tasks.md for task order
# - Validate against success criteria in spec.md
# - Update task status as you progress
```

### Best Practices

**DO**:
- ✅ Update specs BEFORE coding
- ✅ Keep specs and code in sync
- ✅ Use existing spec directory for extensions
- ✅ Document all architectural decisions
- ✅ Define measurable success criteria
- ✅ Track task progress in tasks.md

**DON'T**:
- ❌ Skip spec updates for "small changes"
- ❌ Create new spec for feature extensions
- ❌ Implement without reviewing spec
- ❌ Leave specs outdated
- ❌ Mix multiple features in one spec

### Example: SPY RL Trading System

**Phase 1-6** (Initial Implementation):
- Created `specs/1-spy-rl-trading/` with complete specifications
- Implemented 37/39 tasks following spec
- Achieved production-ready MVP

**Phase 7-12** (Production Hardening):
- UPDATED existing `1-spy-rl-trading` spec (not new directory)
- Added 40 new tasks (T040-T079) to `tasks.md`
- Added Phase 2 section to `plan.md`
- Added hardening goals to `spec.md`
- Created `hardening.md`, `deployment.md`, `observability.md`
- Created contract files for broker, costs, risk limits
- Ready for Phase 7-12 implementation

### Specification-Driven Development Benefits

1. **Clarity**: Everyone understands what's being built
2. **Traceability**: Link code to requirements
3. **Quality**: Success criteria defined upfront
4. **Documentation**: Specs serve as comprehensive docs
5. **Planning**: Accurate estimates and timelines
6. **Review**: Easy to review before implementation
7. **Maintenance**: Single source of truth for features

### Reference

For detailed speckit usage, see:
- Example: `specs/1-spy-rl-trading/` (complete spec structure)
- Tasks format: `specs/1-spy-rl-trading/tasks.md`
- Plan format: `specs/1-spy-rl-trading/plan.md`

## Performance Considerations

- Use `env_stocktrading_np.py` for NumPy-optimized training (2-3x faster)
- For large portfolios (>30 stocks), increase `net_dimension` in `ERL_PARAMS`
- GPU acceleration available for ElegantRL and Stable-Baselines3
- Ray RLlib enables distributed training across multiple CPUs/GPUs

## Troubleshooting

### Common Issues
- **ValueError: array dimensions must match**: Ensure training date range includes all ticker data
- **KeyError: 'eval_times'**: Add `eval_times` to `ERL_PARAMS` dictionary
- **Import errors**: Run `poetry install` or verify `requirements.txt` dependencies
- **API connection failures**: Check credentials in `config_private.py` and network access

### Debugging
- Use `--help` flag with main scripts for CLI options
- Check TensorBoard logs: `tensorboard --logdir tensorboard_log`
- Enable verbose logging in environment `__init__` methods
- Validate data shapes after `df_to_array()` conversion

## Active Technologies
- Python 3.10+ (FinRL requirement)
 (1-spy-rl-trading)
- CSV/Parquet files (local disk) for historical OHLCV data and trained models; TensorBoard logs for training artifacts
 (1-spy-rl-trading)

## Recent Changes
- 1-spy-rl-trading: Added Python 3.10+ (FinRL requirement)
