<!--
  SYNC IMPACT REPORT
  Version Change: 0.0.0 (template) → 1.0.0 (initial ratification)
  Modified Principles: (none, initial adoption)
  Added Sections: All five core principles, Governance
  Removed Sections: (none)
  Templates Updated: ✅ plan-template.md, spec-template.md, tasks-template.md (references aligned)
  Follow-up TODOs: None - constitution fully specified
-->

# FinRL Constitution

## Core Principles

### I. Three-Layer Architecture
FinRL must maintain separation of concerns across three distinct layers: Meta (environments
& data), Agents (DRL algorithms), and Applications (domain-specific strategies). Each layer
operates independently with well-defined interfaces. No cross-layer dependencies that bypass
abstraction boundaries. The train-test-trade pipeline must orchestrate these layers without
leaking implementation details.

### II. Data Processor Abstraction
All market data sources MUST implement a consistent processor interface: `download_data()`,
`clean_data()`, `add_technical_indicator()`, `add_vix()`, `df_to_array()`. Switching data
sources (Yahoo Finance, Alpaca, WRDS, CCXT, etc.) requires no changes to upstream agents
or applications. Each processor is independently testable with unit tests verifying data
integrity and technical indicator correctness.

### III. Gymnasium Compliance
All trading environments MUST inherit from `gymnasium.Env` and implement required methods:
`__init__()`, `reset()`, `step()`, `_get_state()`, `_calculate_reward()`. Action and
observation spaces MUST be properly defined using Gymnasium space types. Environment variants
(StockTradingEnv, StockTradingEnvNP, StockTradingEnvCashPenalty, etc.) are allowable only
when they modify environment-specific constraints—not core trading logic.

### IV. DRL Algorithm Abstraction
Agent implementations MUST expose a consistent training interface regardless of underlying
DRL library (ElegantRL, Stable-Baselines3, RLlib). Key methods: `get_model()`, `train_model()`.
Hyperparameters stored in `config.py` dictionaries (ERL_PARAMS, PPO_PARAMS, etc.) are
versioned and documented. Algorithm-specific tuning is isolated within agent modules; changes
to hyperparameters require explicit justification in commit messages.

### V. Test-First & Observability
Unit tests MUST exist for all processors (data integrity), environments (reward logic),
and agent training workflows. Tests verify both contract (inputs/outputs) and behavior
(convergence, stability). TensorBoard logging is MANDATORY for training runs. Logs enable
debugging of training dynamics, reward curve progression, and hyperparameter impact. No
production deployment without validated test coverage (≥80% for critical paths) and logged
training evidence.

## Development Workflow

### Code Quality Standards
- **Formatting**: Black (127-char line limit), isort (PEP 8), flake8 (F401, W503, E203 ignored)
- **Pre-commit Hooks**: All hooks run automatically before commits; violations block the commit
- **Type Hints**: Strongly encouraged for new code; existing code migration not mandatory
- **Documentation**: Docstrings for public functions; README for new data sources/environments

### Versioning & Breaking Changes
FinRL follows Semantic Versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes to processor interface, environment API, or agent contract
- **MINOR**: New data source, new environment variant, new algorithm integration
- **PATCH**: Bug fixes, hyperparameter tuning, documentation updates, performance improvements

Breaking changes MUST include migration guides documenting how users upgrade their code.

### Configuration Management
- **`config.py`**: Global settings (dates, tickers, hyperparameters, paths)
- **`config_tickers.py`**: Predefined ticker lists (DOW_30, NAS_100, SP_500, etc.)
- **`config_private.py`** (gitignored): API credentials (Alpaca keys, etc.)
- **Environment Variables**: Supported via `.env` for API-based data sources

No hardcoded paths, credentials, or configuration values allowed in source code.

### Testing Strategy
- **Unit Tests**: Processor data output validation, environment reward calculation, agent model instantiation
- **Integration Tests**: Train-test pipeline end-to-end (small dataset, few episodes for speed)
- **Performance Benchmarks**: NumPy-optimized vs. standard environment variants; training speed comparisons
- Test files located in `unit_tests/` and must run via `poetry run pytest`

## Governance

All development MUST comply with this constitution. The constitution supersedes ad-hoc
practices and style preferences. Amendments require:

1. **Documentation**: Proposed change submitted with rationale and impact analysis
2. **Community Review**: Discussion in issues/PRs; 72-hour comment period minimum
3. **Approval**: Consensus from core maintainers (≥2 approvals)
4. **Migration Plan**: If breaking change, provide explicit upgrade path for users
5. **Template Updates**: Any constitutional changes propagate to `.specify/templates/`

Code reviews MUST verify compliance with all five core principles. If a PR violates a
principle, request changes with specific guidance. Complexity in hyperparameter tuning or
algorithm selection MUST be justified—avoid speculative optimizations (YAGNI principle).

**Compliance Verification**: The `/speckit.plan` command enforces "Constitution Check"
validation before approving implementation plans. The `/speckit.tasks` command ensures
task definitions align with test-first and observability principles.

**Version**: 1.0.0 | **Ratified**: 2025-10-27 | **Last Amended**: 2025-10-27
