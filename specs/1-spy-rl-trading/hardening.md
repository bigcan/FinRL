# Production Hardening - Technical Decisions

**Feature**: SPY RL Trading System - Production Hardening
**Phase**: Phase 2 (Phases 7-12)
**Status**: Planning
**Last Updated**: 2025-10-30

---

## Overview

This document details the technical decisions, rationale, and implementation approach for production-hardening the SPY RL Trading System. It covers dependency management, CI/CD, testing, observability, containerization, and finance-specific controls.

---

## Phase 7: Dependency Management & Reproducibility

### Objective
Achieve deterministic, reproducible builds across all environments by standardizing on Poetry and eliminating dependency conflicts.

### Current State Issues

**Issue A: Multiple Build Systems**
- **Problem**: Both `setup.py` (setuptools) and `pyproject.toml` (Poetry) exist, causing confusion and build inconsistencies
- **Impact**: Developers unsure which tool to use, inconsistent environments across team members
- **Evidence**: `setup.py` references `setuptools.find_packages()` while `pyproject.toml` has partial Poetry configuration

**Issue B: Pre-release Dependencies**
- **Problem**: Pinned to pre-release versions (e.g., `stable-baselines3>=2.0.0a9`)
- **Impact**: Unstable API, potential breaking changes, poor ecosystem compatibility
- **Evidence**: Pre-release pins in `pyproject.toml` dependencies section

**Issue C: Conflicting Libraries**
- **Problem**: `alpaca-trade-api` (archived) conflicts with actively maintained `alpaca-py`
- **Impact**: No security updates, deprecated features, maintenance burden
- **Evidence**: Both libraries present in dependency list, causing namespace conflicts

**Issue D: Ray Version Incompatibility**
- **Problem**: Ray 2.10+ drops Python 3.9 support, conflicts with current Python version requirements
- **Impact**: Cannot upgrade to latest Ray without breaking existing environments
- **Evidence**: `ray[rllib]>=2.10.0` pin incompatible with Python 3.9 users

### Technical Decisions

**Decision 1: Standardize on Poetry**
- **Rationale**: Poetry provides superior dependency resolution, lockfile determinism, and virtual environment management compared to setuptools
- **Implementation**: Remove `setup.py`, migrate all dependencies to `pyproject.toml`, enforce `poetry.lock` in version control
- **Trade-offs**: Requires all developers to install Poetry, but gains reproducibility and conflict resolution

**Decision 2: Pin Stable Releases**
- **Rationale**: Pre-release versions are unstable and unsuitable for production use
- **Implementation**: Pin `stable-baselines3>=2.0.0,<3.0.0` (stable 2.x), verify API compatibility
- **Trade-offs**: May miss cutting-edge features, but gains stability and predictability

**Decision 3: Migrate to alpaca-py**
- **Rationale**: `alpaca-trade-api` is archived (no updates since 2023), `alpaca-py` is actively maintained
- **Implementation**: Remove `alpaca-trade-api`, install `alpaca-py>=0.8.2`, update import paths in `finrl/meta/data_processors/processor_alpaca.py`
- **Trade-offs**: Requires code changes in data processor, but eliminates security risk

**Decision 4: Pin Ray 2.9.x**
- **Rationale**: Ray 2.9.x is latest stable version supporting Python 3.9-3.12
- **Implementation**: Pin `ray[rllib]>=2.9.0,<2.10.0`, update RLlib agent code if needed
- **Trade-offs**: Cannot use Ray 2.10+ features, but maintains Python 3.9 compatibility

### Version Pinning Strategy

**Semantic Versioning Approach**:
```toml
# Core dependencies (stable releases only)
stable-baselines3 = "^2.0.0"  # Allow 2.x patches, block 3.0
gymnasium = "^0.29.0"          # Stable API
numpy = "^1.24.0,<2.0.0"       # Block numpy 2.x (breaking changes)
pandas = "^2.0.0"              # Allow 2.x patches
torch = "^2.0.0"               # Allow 2.x patches

# FinRL-specific
yfinance = "^0.2.28"           # Yahoo Finance API
alpaca-py = "^0.8.2"           # Alpaca trading (actively maintained)
stockstats = "^0.6.2"          # Technical indicators

# Ray ecosystem (Python 3.9-3.12 compatible)
ray = { version = "^2.9.0,<2.10.0", extras = ["rllib"] }

# Development tools
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
black = "^23.7.0"
isort = "^5.12.0"
flake8 = "^6.0.0"
```

**Lockfile Enforcement**:
- Commit `poetry.lock` to version control
- CI/CD validates lockfile hash matches
- Use `poetry install --no-root` in CI to ensure reproducibility
- Periodic dependency audits with `poetry show --outdated`

### Verification Tests

**T042: Poetry Migration**
```bash
# Verify Poetry configuration is valid
poetry check

# Verify lockfile is deterministic
poetry lock --check

# Verify all dependencies resolve
poetry install --dry-run
```

**T043: Dependency Conflicts**
```bash
# Verify alpaca-py imports work
python -c "from alpaca.data import StockHistoricalDataClient; print('OK')"

# Verify stable-baselines3 2.x
python -c "import stable_baselines3; assert stable_baselines3.__version__.startswith('2.'), 'Must be 2.x'"

# Verify Ray 2.9.x compatibility
python -c "import ray; assert ray.__version__.startswith('2.9.'), 'Must be 2.9.x'"
```

---

## Phase 8: CI/CD Pipeline & Quality Automation

### Objective
Implement comprehensive continuous integration with automated testing, coverage reporting, security scanning, and release automation.

### GitHub Actions Workflow Architecture

**Workflow 1: Test Matrix (`test.yml`)**
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install Poetry
        run: pipx install poetry
      - name: Install dependencies
        run: poetry install --with dev
      - name: Run unit tests
        run: poetry run pytest unit_tests -m unit --cov=finrl --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

**Workflow 2: Integration Tests (`integration.yml`)**
- Requires network access for Yahoo Finance data
- Uses pytest marker `@pytest.mark.integration`
- Runs on schedule (nightly) to avoid rate limits
- Caches downloaded data for faster runs

**Workflow 3: Security Scanning (`security.yml`)**
```yaml
name: Security
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: pip-audit (dependency vulnerabilities)
        run: |
          pip install pip-audit
          pip-audit --require-hashes --desc
      - name: Bandit (Python security linter)
        run: |
          pip install bandit
          bandit -r finrl/ -f json -o bandit-report.json
      - name: CodeQL (static analysis)
        uses: github/codeql-action/analyze@v3
```

**Workflow 4: Release Automation (`release.yml`)**
- Triggered on tag push (e.g., `v1.1.0`)
- Generates SBOM using CycloneDX
- Creates GitHub release with changelog
- Optional: Publishes to PyPI (if configured)

### Coverage Requirements

**Target Coverage**: ≥80% (unit tests), ≥70% (integration tests)

**Coverage Configuration** (`.coveragerc`):
```ini
[run]
source = finrl
omit =
    */tests/*
    */test_*.py
    */__pycache__/*
    */venv/*

[report]
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

**Coverage Enforcement**:
- Fail CI if coverage drops below threshold
- Generate HTML reports for local debugging
- Upload to codecov.io for historical tracking

### Security Scanning Tools

**pip-audit**:
- Scans `poetry.lock` for known CVEs
- Blocks merges if high/critical vulnerabilities found
- Weekly scheduled runs to catch new CVEs

**Bandit**:
- Static security analysis for Python code
- Checks for hardcoded secrets, SQL injection, insecure random, etc.
- Configured via `.bandit` file

**CodeQL**:
- Deep semantic code analysis
- Detects security vulnerabilities and code smells
- Language: Python, queries: security-and-quality

### Deterministic Test Fixtures

**Problem**: Network-dependent tests are slow and flaky
**Solution**: Pre-downloaded data snapshots

**Implementation**:
```python
# unit_tests/fixtures/spy_data_2020_2024.parquet (frozen snapshot)
# Downloaded once, version-controlled, used for all unit tests

@pytest.fixture
def spy_training_data():
    """Load pre-downloaded SPY data for 2020-2024."""
    data_path = Path(__file__).parent / "fixtures" / "spy_data_2020_2024.parquet"
    return pd.read_parquet(data_path)

# Integration tests still use live data to verify API connectivity
@pytest.mark.integration
def test_yahoo_finance_download():
    """Verify Yahoo Finance API connectivity (integration test)."""
    processor = YahooFinanceProcessor()
    data = processor.download_data(
        ticker_list=["SPY"],
        start_date="2020-01-01",
        end_date="2020-01-31"
    )
    assert len(data) > 0, "Failed to download data"
```

---

## Phase 9: Test Hardening & Determinism

### Objective
Separate unit tests from integration tests, eliminate brittle assertions, and implement property-based testing for robust edge case coverage.

### Test Organization Strategy

**Pytest Markers**:
```python
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests (fast, no network)",
    "integration: Integration tests (network required)",
    "network: Tests requiring internet access",
    "slow: Slow-running tests (>5 seconds)",
]
```

**Usage**:
```bash
# Run only unit tests (CI default)
pytest -m unit

# Run only integration tests (nightly CI)
pytest -m integration

# Run all except network tests (offline development)
pytest -m "not network"

# Run slow tests on weekends
pytest -m slow
```

### Brittle Test Patterns to Eliminate

**Brittle Pattern 1: Exact Row Counts**
```python
# ❌ BAD: Exact row count (brittle if data changes)
def test_download_spy_data():
    data = processor.download_data(["SPY"], "2020-01-01", "2020-12-31")
    assert len(data) == 252, "Expected 252 trading days"

# ✅ GOOD: Tolerance-based assertion
def test_download_spy_data():
    data = processor.download_data(["SPY"], "2020-01-01", "2020-12-31")
    assert 250 <= len(data) <= 255, f"Expected ~252 trading days, got {len(data)}"
```

**Brittle Pattern 2: Exact Floating-Point Comparisons**
```python
# ❌ BAD: Exact float comparison
assert sharpe_ratio == 0.75

# ✅ GOOD: Tolerance-based comparison
assert abs(sharpe_ratio - 0.75) < 0.01, f"Sharpe {sharpe_ratio} not within tolerance"
# Or use pytest.approx
assert sharpe_ratio == pytest.approx(0.75, rel=0.01)
```

**Brittle Pattern 3: Hardcoded Dates**
```python
# ❌ BAD: Hardcoded future dates
data = processor.download_data(["SPY"], "2024-01-01", "2024-12-31")

# ✅ GOOD: Relative dates or frozen snapshots
from datetime import datetime, timedelta
end_date = datetime.now() - timedelta(days=30)  # 30 days ago
start_date = end_date - timedelta(days=365)
data = processor.download_data(["SPY"], start_date, end_date)
```

### Property-Based Testing with Hypothesis

**Use Case**: Test invariants that should hold for all inputs

**Example 1: Sharpe Ratio Properties**
```python
from hypothesis import given, strategies as st

@given(
    returns=st.lists(st.floats(min_value=-0.1, max_value=0.1), min_size=252, max_size=252)
)
def test_sharpe_ratio_properties(returns):
    """Sharpe ratio should be finite and bounded."""
    sharpe = calculate_sharpe_ratio(returns)

    # Property 1: Sharpe ratio is always finite
    assert np.isfinite(sharpe), "Sharpe ratio should be finite"

    # Property 2: Zero returns → zero Sharpe
    if all(r == 0 for r in returns):
        assert sharpe == 0.0, "Zero returns should yield zero Sharpe"

    # Property 3: Positive mean returns → positive Sharpe (with low volatility)
    if np.mean(returns) > 0 and np.std(returns) < 0.01:
        assert sharpe > 0, "Positive returns with low vol should yield positive Sharpe"
```

**Example 2: Environment Step Invariants**
```python
@given(
    action=st.integers(min_value=0, max_value=2),  # buy=0, hold=1, sell=2
    balance=st.floats(min_value=0, max_value=100000),
    shares=st.integers(min_value=0, max_value=100)
)
def test_environment_step_invariants(action, balance, shares):
    """Environment step should maintain invariants."""
    env = create_test_environment(initial_balance=balance, initial_shares=shares)

    state, reward, done, truncated, info = env.step(action)

    # Invariant 1: Balance + (shares * price) should not exceed initial NAV
    nav = info["balance"] + info["shares"] * info["price"]
    assert nav >= 0, "NAV should never be negative"

    # Invariant 2: Shares should never be negative (no shorting)
    assert info["shares"] >= 0, "Shares should never be negative"

    # Invariant 3: Balance should never be negative (no leverage)
    assert info["balance"] >= 0, "Balance should never be negative"
```

### Deterministic Test Fixtures

**Implementation**:
```python
# unit_tests/fixtures/__init__.py
import pandas as pd
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

def load_spy_snapshot(period: str) -> pd.DataFrame:
    """Load pre-downloaded SPY data snapshot."""
    snapshot_file = FIXTURES_DIR / f"spy_{period}.parquet"
    if not snapshot_file.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_file}")
    return pd.read_parquet(snapshot_file)

# Available snapshots:
# - spy_2020_2024.parquet: Training data (1258 rows)
# - spy_2025_sample.parquet: Test data (sample 50 rows)
# - spy_indicators.parquet: Pre-computed technical indicators
```

**Generating Snapshots**:
```bash
# Run once to generate snapshots (requires network)
poetry run python unit_tests/fixtures/generate_snapshots.py

# Commit snapshots to version control
git add unit_tests/fixtures/*.parquet
git commit -m "Add deterministic test fixtures"
```

---

## Phase 10: Runtime Observability & Configuration

### Objective
Implement structured logging, centralized configuration, and reproducibility hooks to enable production-grade observability.

### Structured Logging with structlog

**Why structlog?**
- JSON-formatted logs for machine readability
- Contextual logging with structured data
- Performance-optimized for high-throughput systems
- Integration with cloud logging services (CloudWatch, Stackdriver)

**Configuration** (`finrl/applications/spy_rl_trading/logging_config.py`):
```python
import structlog
import logging
import sys

def configure_logging(level: str = "INFO", json_output: bool = True):
    """Configure structured logging for SPY RL Trading System."""

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        # Production: JSON output
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Development: Human-readable console output
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
```

**Usage**:
```python
import structlog

logger = structlog.get_logger(__name__)

# Log with structured context
logger.info("data_loaded",
            ticker="SPY",
            rows=1258,
            start_date="2020-01-01",
            end_date="2024-12-31",
            missing_days=3)

# Output (JSON):
# {
#   "event": "data_loaded",
#   "ticker": "SPY",
#   "rows": 1258,
#   "start_date": "2020-01-01",
#   "end_date": "2024-12-31",
#   "missing_days": 3,
#   "level": "info",
#   "timestamp": "2025-10-30T12:34:56.789012Z",
#   "logger": "finrl.applications.spy_rl_trading.data_processor"
# }
```

### Centralized Configuration with pydantic-settings

**Why pydantic-settings?**
- Type-safe configuration with validation
- Environment variable support with prefix
- Auto-documentation via Pydantic schemas
- Immutable settings (prevents runtime modification)

**Configuration Schema** (`finrl/applications/spy_rl_trading/settings.py`):
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class SPYSettings(BaseSettings):
    """SPY RL Trading System settings."""

    model_config = SettingsConfigDict(
        env_prefix="SPY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Data settings
    ticker: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    data_source: Literal["yahoofinance", "alpaca"] = "yahoofinance"

    # Training settings
    total_timesteps: int = 500_000
    learning_rate: float = 0.0003
    batch_size: int = 128
    n_steps: int = 2048

    # Backtesting settings
    initial_capital: float = 100_000.0
    transaction_cost_bps: int = 2
    risk_free_rate: float = 0.0

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # Reproducibility
    random_seed: int = 42

# Singleton instance
settings = SPYSettings()
```

**Usage**:
```python
from finrl.applications.spy_rl_trading.settings import settings

# Access settings
print(f"Training for {settings.total_timesteps} timesteps")

# Override via environment variables
# SPY_TOTAL_TIMESTEPS=1000000 python train.py
```

### Reproducibility Hooks

**Global Seed Management**:
```python
import random
import numpy as np
import torch
from finrl.applications.spy_rl_trading.settings import settings

def set_global_seed(seed: int = None):
    """Set global random seed for reproducibility."""
    seed = seed or settings.random_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("global_seed_set", seed=seed)
```

**Usage**:
```python
# At start of training script
set_global_seed()

# Or override
set_global_seed(seed=12345)
```

---

## Phase 11: Docker Containerization

### Objective
Create multi-stage Docker builds with non-root user execution, docker-compose orchestration, and multi-architecture support.

### Multi-Stage Dockerfile

**Why Multi-Stage?**
- Smaller final image (only runtime dependencies)
- Faster builds (layer caching)
- Secure builds (no build tools in production image)

**Dockerfile**:
```dockerfile
# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev dependencies)
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main --no-interaction --no-ansi

# Stage 2: Runtime stage
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash finrl

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application code
COPY finrl/ /app/finrl/
COPY unit_tests/ /app/unit_tests/

# Set ownership
RUN chown -R finrl:finrl /app

# Switch to non-root user
USER finrl

# Set PATH to include virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "-m", "finrl.applications.spy_rl_trading.example_training"]
```

**Build**:
```bash
docker build -t finrl-spy:latest .

# Multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 -t finrl-spy:latest .
```

### Docker Compose Orchestration

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  # Training service
  train:
    build:
      context: .
      dockerfile: Dockerfile
    image: finrl-spy:latest
    container_name: spy-training
    command: ["python", "-m", "finrl.applications.spy_rl_trading.example_training"]
    volumes:
      - ./trained_models:/app/trained_models
      - ./tensorboard_logs:/app/tensorboard_logs
      - ./datasets:/app/datasets
    environment:
      - SPY_LOG_LEVEL=INFO
      - SPY_RANDOM_SEED=42
    networks:
      - finrl-network

  # Backtesting service
  backtest:
    image: finrl-spy:latest
    container_name: spy-backtesting
    command: ["python", "-m", "finrl.applications.spy_rl_trading.example_backtesting"]
    volumes:
      - ./trained_models:/app/trained_models:ro
      - ./results:/app/results
      - ./datasets:/app/datasets:ro
    environment:
      - SPY_LOG_LEVEL=INFO
    networks:
      - finrl-network
    depends_on:
      - train

  # TensorBoard service
  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: spy-tensorboard
    command: ["tensorboard", "--logdir=/logs", "--host=0.0.0.0"]
    ports:
      - "6006:6006"
    volumes:
      - ./tensorboard_logs:/logs:ro
    networks:
      - finrl-network

networks:
  finrl-network:
    driver: bridge

volumes:
  trained_models:
  tensorboard_logs:
  results:
  datasets:
```

**Usage**:
```bash
# Train model
docker-compose up train

# Run backtest
docker-compose up backtest

# Start TensorBoard (access at http://localhost:6006)
docker-compose up tensorboard

# Full pipeline
docker-compose up train backtest tensorboard
```

---

## Phase 12: Finance-Specific Production Controls

### Objective
Implement realistic transaction costs, slippage modeling, risk limits, broker adapters, and trading mode separation.

### Transaction Cost Model

**Components**:
1. **Commission**: Fixed cost per trade (e.g., $0 for Robinhood, $0.0035/share for Interactive Brokers)
2. **Spread**: Bid-ask spread (typically 0.01-0.05% for SPY)
3. **Market Impact**: Price movement caused by order (negligible for SPY)
4. **SEC Fees**: $0.0000278 per dollar sold (US equities only)

**Implementation** (`finrl/applications/spy_rl_trading/transaction_costs.py`):
```python
from dataclasses import dataclass

@dataclass
class TransactionCost:
    """Transaction cost breakdown."""
    commission: float
    spread: float
    market_impact: float
    sec_fee: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.market_impact + self.sec_fee

def calculate_transaction_cost(
    price: float,
    shares: int,
    side: str,  # "buy" or "sell"
    commission_per_share: float = 0.0,
    spread_bps: float = 2.0,
    market_impact_bps: float = 0.0,
    sec_fee_per_dollar: float = 0.0000278,
) -> TransactionCost:
    """Calculate transaction cost for a trade."""

    notional = price * shares

    # Commission
    commission = commission_per_share * shares

    # Spread (half-spread on each side)
    spread = notional * (spread_bps / 10000) / 2

    # Market impact (only for large orders, negligible for SPY)
    market_impact = notional * (market_impact_bps / 10000)

    # SEC fee (only on sells)
    sec_fee = notional * sec_fee_per_dollar if side == "sell" else 0.0

    return TransactionCost(
        commission=commission,
        spread=spread,
        market_impact=market_impact,
        sec_fee=sec_fee,
    )
```

### Slippage Model

**Slippage**: Difference between expected price and actual execution price due to market movement and order size.

**Model**:
```python
def calculate_slippage(
    price: float,
    shares: int,
    average_volume: float,
    side: str,  # "buy" or "sell"
) -> float:
    """Calculate slippage based on order size vs. average volume."""

    # Participation rate (order size / average volume)
    participation_rate = shares / average_volume

    # Slippage scales with square root of participation rate
    # Typical SPY: 0.01% slippage for 0.1% participation
    base_slippage_bps = 1.0  # 0.01%
    slippage_bps = base_slippage_bps * np.sqrt(participation_rate / 0.001)

    # Slippage is negative for buys (pay more), positive for sells (receive less)
    sign = 1 if side == "buy" else -1
    slippage = price * (slippage_bps / 10000) * sign

    return slippage
```

### NYSE Trading Calendar

**Integration** (`pandas_market_calendars`):
```python
import pandas_market_calendars as mcal

# Get NYSE calendar
nyse = mcal.get_calendar("NYSE")

# Get trading days in range
trading_days = nyse.schedule(start_date="2020-01-01", end_date="2024-12-31")

# Check if date is trading day
def is_trading_day(date: str) -> bool:
    """Check if date is a NYSE trading day."""
    schedule = nyse.schedule(start_date=date, end_date=date)
    return len(schedule) > 0

# Filter dataframe to trading days only
df_trading = df[df.index.isin(trading_days.index)]
```

### Risk Limits

**Risk Limit Types**:
1. **Max Drawdown**: Stop trading if cumulative drawdown exceeds threshold (e.g., 20%)
2. **Daily Loss**: Stop trading if single-day loss exceeds threshold (e.g., 5%)
3. **Position Size**: Limit position size as percentage of NAV (e.g., 100% for SPY single-asset)
4. **Leverage**: Prevent leverage (cash + position value ≤ NAV)

**Implementation** (`finrl/applications/spy_rl_trading/risk_limits.py`):
```python
from dataclasses import dataclass

@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_drawdown_pct: float = 0.20  # 20%
    max_daily_loss_pct: float = 0.05  # 5%
    max_position_size_pct: float = 1.00  # 100%
    allow_leverage: bool = False

class RiskManager:
    """Risk limit enforcement."""

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.peak_nav = 0.0
        self.previous_nav = 0.0

    def check_limits(self, current_nav: float, position_value: float) -> dict:
        """Check if risk limits are breached."""

        # Update peak NAV
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav

        # Calculate drawdown
        drawdown = (self.peak_nav - current_nav) / self.peak_nav if self.peak_nav > 0 else 0.0

        # Calculate daily loss
        daily_loss = (self.previous_nav - current_nav) / self.previous_nav if self.previous_nav > 0 else 0.0

        # Update previous NAV
        self.previous_nav = current_nav

        # Check limits
        breaches = {}
        if drawdown > self.limits.max_drawdown_pct:
            breaches["max_drawdown"] = f"Drawdown {drawdown:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}"

        if daily_loss > self.limits.max_daily_loss_pct:
            breaches["daily_loss"] = f"Daily loss {daily_loss:.2%} exceeds limit {self.limits.max_daily_loss_pct:.2%}"

        if not self.limits.allow_leverage and position_value > current_nav:
            breaches["leverage"] = f"Position value {position_value} exceeds NAV {current_nav}"

        return breaches
```

### Broker Adapter Pattern

**Purpose**: Abstract broker-specific APIs to enable multi-broker support

**Interface** (see `contracts/BROKER_ADAPTER.md`):
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Order:
    """Order representation."""
    symbol: str
    quantity: int
    side: str  # "buy" or "sell"
    order_type: str  # "market" or "limit"
    limit_price: Optional[float] = None

class BrokerAdapter(ABC):
    """Broker adapter interface."""

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place order and return order ID."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Get order status."""
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account balance and positions."""
        pass
```

**Implementations**:
- `AlpacaBrokerAdapter`: Alpaca API integration (paper/live)
- `BacktestBrokerAdapter`: Simulated execution for backtesting
- `DryRunBrokerAdapter`: Logs orders without execution

### Trading Mode Separation

**Modes**:
1. **Backtest**: Historical simulation, no real orders
2. **Paper**: Simulated trading with live data, no real money
3. **Live**: Real trading with real money

**Configuration**:
```python
from enum import Enum

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

# Environment variable
# SPY_TRADING_MODE=paper python trade.py
trading_mode = TradingMode(settings.trading_mode)

# Adapter selection
if trading_mode == TradingMode.BACKTEST:
    adapter = BacktestBrokerAdapter()
elif trading_mode == TradingMode.PAPER:
    adapter = AlpacaBrokerAdapter(base_url=ALPACA_PAPER_URL)
elif trading_mode == TradingMode.LIVE:
    adapter = AlpacaBrokerAdapter(base_url=ALPACA_LIVE_URL)
```

---

## Risk Management & Validation

### Pre-Production Checklist

**Phase 7: Dependency Management**
- [ ] All dependencies pinned in `poetry.lock`
- [ ] Zero high/critical vulnerabilities (pip-audit)
- [ ] Build reproducible across Linux/macOS/Windows

**Phase 8: CI/CD Pipeline**
- [ ] GitHub Actions workflows passing
- [ ] Unit test coverage ≥80%
- [ ] Integration tests passing with deterministic fixtures
- [ ] Security scans (Bandit, CodeQL) clean

**Phase 9: Test Quality**
- [ ] Unit tests separated from integration tests
- [ ] Brittle assertions eliminated
- [ ] Property-based tests added
- [ ] Deterministic fixtures in place

**Phase 10: Observability**
- [ ] Structured logging implemented
- [ ] Centralized configuration via pydantic-settings
- [ ] Reproducibility hooks (global seed) working

**Phase 11: Docker**
- [ ] Multi-stage Dockerfile building successfully
- [ ] Non-root user execution verified
- [ ] docker-compose orchestration functional

**Phase 12: Finance Controls**
- [ ] Transaction cost model realistic
- [ ] Slippage model implemented
- [ ] NYSE calendar integration working
- [ ] Risk limits preventing catastrophic losses
- [ ] Broker adapter abstraction complete
- [ ] Trading mode separation enforced

### Acceptance Criteria Validation

All success criteria (SC-H001 through SC-H006) must be validated with evidence before marking phases complete. See `spec.md` for detailed success criteria definitions.

---

## Appendix

### Technology Stack

- **Build System**: Poetry 1.7.1+
- **Python**: 3.10, 3.11, 3.12
- **DRL**: Stable-Baselines3 2.x, Ray 2.9.x
- **Logging**: structlog, python-logging
- **Config**: pydantic-settings
- **Testing**: pytest, pytest-cov, Hypothesis
- **CI/CD**: GitHub Actions, codecov.io
- **Security**: pip-audit, Bandit, CodeQL
- **Containers**: Docker, docker-compose
- **Finance**: pandas_market_calendars, alpaca-py

### References

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [structlog Documentation](https://www.structlog.org/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [pandas_market_calendars](https://pandas-market-calendars.readthedocs.io/)

---

**Document Revision History**:
- 2025-10-30: Initial version covering Phases 7-12 technical decisions
