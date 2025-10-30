# Observability - Logging, Metrics, & Monitoring

**Feature**: SPY RL Trading System - Observability Framework
**Phase**: Phase 10 (Runtime & Operations)
**Status**: Planning
**Last Updated**: 2025-10-30

---

## Overview

This document details the observability framework for the SPY RL Trading System, covering structured logging, configuration management, reproducibility hooks, and monitoring strategies for production environments.

---

## Structured Logging with structlog

### Why Structured Logging?

**Benefits**:
- **Machine-Readable**: JSON format enables automated log analysis
- **Contextual**: Attach structured metadata to every log event
- **Performance**: Faster than string concatenation for high-throughput systems
- **Cloud-Native**: Native integration with CloudWatch, Stackdriver, Elasticsearch
- **Debugging**: Rich context accelerates troubleshooting

**Comparison**:
```python
# ❌ Traditional logging (unstructured)
logging.info(f"Data loaded for {ticker}: {len(df)} rows from {start} to {end}")

# ✅ Structured logging (contextual)
logger.info("data_loaded",
            ticker=ticker,
            rows=len(df),
            start_date=start,
            end_date=end,
            missing_days=missing_count)

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

---

### Configuration

**logging_config.py** (`finrl/applications/spy_rl_trading/logging_config.py`):

```python
"""Structured logging configuration for SPY RL Trading System."""

import structlog
import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(
    level: LogLevel = "INFO",
    json_output: bool = True,
    show_caller: bool = True,
) -> None:
    """
    Configure structured logging for SPY RL Trading System.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON format. If False, human-readable console format.
        show_caller: If True, include caller module and line number in logs
    """

    # Timestamper with ISO 8601 format
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    # Shared processors for both JSON and console output
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add caller information if requested
    if show_caller:
        shared_processors.append(structlog.processors.CallsiteParameterAdder())

    if json_output:
        # Production: JSON output for machine parsing
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
        # Development: Human-readable console output with colors
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True)
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

    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Example usage
if __name__ == "__main__":
    # Development mode (human-readable)
    configure_logging(level="DEBUG", json_output=False)

    # Production mode (JSON)
    # configure_logging(level="INFO", json_output=True)

    logger = structlog.get_logger(__name__)
    logger.info("test_message", key1="value1", key2=42)
```

---

### Usage Patterns

**1. Basic Logging**:
```python
import structlog

logger = structlog.get_logger(__name__)

# Info log
logger.info("data_processor_initialized", ticker="SPY", indicators=9)

# Warning log
logger.warning("missing_data_detected", rows_missing=5, pct_missing=0.4)

# Error log
try:
    result = risky_operation()
except Exception as e:
    logger.error("operation_failed", error=str(e), exc_info=True)
```

**2. Contextual Logging with Binding**:
```python
# Bind context that persists across log calls
logger = structlog.get_logger(__name__).bind(
    ticker="SPY",
    start_date="2020-01-01",
    end_date="2024-12-31",
)

# All subsequent logs include ticker, start_date, end_date
logger.info("downloading_data")
logger.info("data_downloaded", rows=1258)
logger.info("indicators_computed", indicators=["SMA", "RSI", "MACD"])
```

**3. Performance Logging**:
```python
import time
import structlog

logger = structlog.get_logger(__name__)

def timed_operation(operation_name: str):
    """Decorator for timing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    "operation_completed",
                    operation=operation_name,
                    duration_seconds=duration,
                    success=True,
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "operation_failed",
                    operation=operation_name,
                    duration_seconds=duration,
                    error=str(e),
                    success=False,
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator

@timed_operation("data_download")
def download_data(ticker: str, start_date: str, end_date: str):
    # Download logic
    pass
```

---

### Log Levels & Use Cases

**DEBUG**:
- Detailed information for debugging
- Function entry/exit points
- Variable values at checkpoints
- Only enabled during development

```python
logger.debug("environment_step",
             step=t,
             action=action,
             state_shape=state.shape,
             reward=reward,
             done=done)
```

**INFO**:
- General operational information
- Successful completions
- Milestone events
- Always enabled in production

```python
logger.info("training_started",
            total_timesteps=500_000,
            learning_rate=0.0003,
            batch_size=128)

logger.info("training_completed",
            final_episode_reward=125.3,
            episodes=1000,
            duration_minutes=45.2)
```

**WARNING**:
- Recoverable issues
- Degraded performance
- Data quality concerns
- Deprecated features

```python
logger.warning("data_quality_issue",
               missing_rows=3,
               pct_missing=0.24,
               action="forward_filled")

logger.warning("slow_convergence",
               episodes_without_improvement=50,
               current_sharpe=0.35,
               target_sharpe=0.50)
```

**ERROR**:
- Errors requiring attention
- Failed operations
- Exceptions caught and handled
- Not fatal to system

```python
logger.error("api_request_failed",
             endpoint="https://data.alpaca.markets",
             status_code=503,
             retry_count=3,
             exc_info=True)
```

**CRITICAL**:
- System-level failures
- Unrecoverable errors
- Immediate human intervention required

```python
logger.critical("model_load_failed",
                model_path="/app/trained_models/ppo.zip",
                error="File not found",
                exc_info=True)
```

---

## Centralized Configuration Management

### Why pydantic-settings?

**Benefits**:
- **Type Safety**: Pydantic validates types at runtime
- **Environment Variables**: Automatic parsing from env vars
- **Immutability**: Settings are frozen after initialization
- **Documentation**: Auto-generated schema documentation
- **Defaults**: Clear default values with overrides

---

### Configuration Schema

**settings.py** (`finrl/applications/spy_rl_trading/settings.py`):

```python
"""Centralized configuration for SPY RL Trading System."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, List
from pathlib import Path


class SPYSettings(BaseSettings):
    """
    SPY RL Trading System configuration.

    All settings can be overridden via environment variables with SPY_ prefix.
    Example: SPY_TOTAL_TIMESTEPS=1000000 python train.py
    """

    model_config = SettingsConfigDict(
        env_prefix="SPY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,  # Immutable after initialization
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Data Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ticker: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    data_source: Literal["yahoofinance", "alpaca"] = "yahoofinance"
    indicators: List[str] = [
        "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
        "dx_30", "close_30_sma", "close_60_sma", "vix"
    ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Training Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    total_timesteps: int = 500_000
    learning_rate: float = 0.0003
    batch_size: int = 128
    n_steps: int = 2048
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Backtesting Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    initial_capital: float = 100_000.0
    transaction_cost_bps: int = 2  # 2 basis points (0.02%)
    risk_free_rate: float = 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Risk Management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    max_drawdown_pct: float = 0.20  # 20%
    max_daily_loss_pct: float = 0.05  # 5%
    max_position_size_pct: float = 1.00  # 100% (single asset)
    allow_leverage: bool = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Trading Mode
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    trading_mode: Literal["backtest", "paper", "live"] = "backtest"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Logging Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True
    log_show_caller: bool = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Reproducibility
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    random_seed: int = 42

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Paths
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    data_dir: Path = Path("./datasets")
    models_dir: Path = Path("./trained_models")
    tensorboard_dir: Path = Path("./tensorboard_logs")
    results_dir: Path = Path("./results")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # API Credentials (Paper Trading)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"


# Singleton instance
settings = SPYSettings()


# Example usage
if __name__ == "__main__":
    print(settings.model_dump_json(indent=2))
```

---

### Usage Examples

**1. Basic Access**:
```python
from finrl.applications.spy_rl_trading.settings import settings

# Access settings
print(f"Training for {settings.total_timesteps} timesteps")
print(f"Learning rate: {settings.learning_rate}")

# Use in code
data_processor = SPYDataProcessor(
    ticker=settings.ticker,
    start_date=settings.start_date,
    end_date=settings.end_date,
)
```

**2. Environment Variable Overrides**:
```bash
# Override via environment variables (SPY_ prefix)
export SPY_TOTAL_TIMESTEPS=1000000
export SPY_LEARNING_RATE=0.001
export SPY_LOG_LEVEL=DEBUG

python train.py
```

**3. .env File Configuration**:
```bash
# .env file
SPY_TICKER=SPY
SPY_START_DATE=2020-01-01
SPY_END_DATE=2024-12-31
SPY_TOTAL_TIMESTEPS=500000
SPY_LEARNING_RATE=0.0003
SPY_LOG_LEVEL=INFO
SPY_LOG_JSON=true
SPY_RANDOM_SEED=42

# Alpaca credentials
SPY_ALPACA_API_KEY=your_paper_key
SPY_ALPACA_API_SECRET=your_paper_secret
SPY_ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**4. Testing with Custom Settings**:
```python
import pytest
from finrl.applications.spy_rl_trading.settings import SPYSettings

@pytest.fixture
def test_settings():
    """Override settings for testing."""
    return SPYSettings(
        total_timesteps=1024,  # Quick smoke test
        learning_rate=0.001,
        log_level="DEBUG",
        log_json=False,
        random_seed=12345,
    )

def test_training(test_settings):
    """Test training with custom settings."""
    # Use test_settings instead of global settings
    pass
```

---

## Reproducibility Hooks

### Global Seed Management

**Purpose**: Ensure reproducible results across runs for debugging and validation

**Implementation** (`finrl/applications/spy_rl_trading/reproducibility.py`):

```python
"""Reproducibility hooks for deterministic training and backtesting."""

import random
import numpy as np
import torch
import structlog
from finrl.applications.spy_rl_trading.settings import settings

logger = structlog.get_logger(__name__)


def set_global_seed(seed: int = None) -> None:
    """
    Set global random seed for reproducibility across all libraries.

    Args:
        seed: Random seed. If None, uses settings.random_seed.
    """
    seed = seed or settings.random_seed

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Disable cuDNN auto-tuner for determinism (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("global_seed_set",
                seed=seed,
                cuda_available=torch.cuda.is_available(),
                deterministic_mode=True)


def get_seed_info() -> dict:
    """
    Get current seed configuration for logging.

    Returns:
        Dictionary with seed information.
    """
    return {
        "random_seed": settings.random_seed,
        "numpy_random_state": np.random.get_state()[1][0],
        "torch_initial_seed": torch.initial_seed(),
        "cuda_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
    }


# Example usage
if __name__ == "__main__":
    from finrl.applications.spy_rl_trading.logging_config import configure_logging

    configure_logging(level="INFO", json_output=False)

    # Set global seed
    set_global_seed()

    # Log seed info
    seed_info = get_seed_info()
    logger.info("seed_configuration", **seed_info)

    # Verify reproducibility
    print("Random sample (Python):", random.randint(0, 100))
    print("Random sample (NumPy):", np.random.randint(0, 100))
    print("Random sample (PyTorch):", torch.randint(0, 100, (1,)).item())
```

**Usage in Training Script**:
```python
from finrl.applications.spy_rl_trading.logging_config import configure_logging
from finrl.applications.spy_rl_trading.reproducibility import set_global_seed
from finrl.applications.spy_rl_trading.settings import settings

def main():
    # Initialize logging
    configure_logging(
        level=settings.log_level,
        json_output=settings.log_json,
    )

    # Set global seed for reproducibility
    set_global_seed()

    # Train model
    train_agent(...)

if __name__ == "__main__":
    main()
```

---

## Production Monitoring

### Key Metrics to Track

**Training Metrics**:
- Episode reward (mean, std, min, max)
- Training loss (policy loss, value loss, entropy)
- Episode length
- Learning rate (if adaptive)
- Gradient norm
- FPS (frames per second)

**Backtesting Metrics**:
- Cumulative return
- Sharpe ratio
- Max drawdown
- Win rate
- Average trade duration
- Total trades

**System Metrics**:
- CPU usage
- Memory usage
- GPU utilization (if applicable)
- Disk I/O
- Network I/O (for API calls)

**Business Metrics**:
- Portfolio NAV
- Daily PnL
- Realized PnL vs. unrealized PnL
- Transaction costs
- Slippage costs

---

### TensorBoard Integration

**Built-in Integration**:
```python
from stable_baselines3 import PPO

# PPO automatically logs to TensorBoard
model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log=str(settings.tensorboard_dir),
    verbose=1,
)

model.learn(total_timesteps=settings.total_timesteps)
```

**View Logs**:
```bash
tensorboard --logdir ./tensorboard_logs
# Open http://localhost:6006
```

**Custom Metrics**:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=settings.tensorboard_dir / "custom")

# Log scalars
writer.add_scalar("metrics/sharpe_ratio", sharpe_ratio, epoch)
writer.add_scalar("metrics/max_drawdown", max_drawdown, epoch)

# Log histograms
writer.add_histogram("returns/distribution", returns, epoch)

# Log images
writer.add_image("equity_curve", equity_curve_img, epoch)

writer.close()
```

---

### Prometheus Integration (Optional)

**Why Prometheus?**
- Time-series metrics database
- Pull-based architecture
- Grafana integration for dashboards
- Alerting support

**Implementation** (`finrl/applications/spy_rl_trading/metrics.py`):

```python
"""Prometheus metrics for production monitoring."""

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import structlog

logger = structlog.get_logger(__name__)

# Training metrics
training_episodes_total = Counter(
    "spy_training_episodes_total",
    "Total number of training episodes",
)

training_episode_reward = Gauge(
    "spy_training_episode_reward",
    "Current episode reward",
)

training_duration_seconds = Histogram(
    "spy_training_duration_seconds",
    "Training duration in seconds",
    buckets=[60, 120, 300, 600, 1800, 3600],
)

# Backtesting metrics
backtest_sharpe_ratio = Gauge(
    "spy_backtest_sharpe_ratio",
    "Backtest Sharpe ratio",
)

backtest_max_drawdown = Gauge(
    "spy_backtest_max_drawdown",
    "Backtest maximum drawdown",
)

# Trading metrics
trading_orders_total = Counter(
    "spy_trading_orders_total",
    "Total number of trading orders",
    ["side"],  # buy/sell
)

trading_portfolio_nav = Gauge(
    "spy_trading_portfolio_nav",
    "Current portfolio NAV",
)


def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
    logger.info("prometheus_metrics_server_started", port=port)
```

**Usage**:
```python
from finrl.applications.spy_rl_trading.metrics import (
    training_episodes_total,
    training_episode_reward,
    start_metrics_server,
)

# Start metrics server (once at app startup)
start_metrics_server(port=9090)

# Update metrics during training
training_episodes_total.inc()
training_episode_reward.set(episode_reward)
```

**Prometheus Configuration** (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'spy_rl_trading'
    static_configs:
      - targets: ['localhost:9090']
```

---

## Alert Configuration

### Alerting Strategies

**1. Training Failures**:
```python
# Log critical errors for alerting
if episode_reward < -100:
    logger.critical("training_divergence",
                    episode=episode,
                    reward=episode_reward,
                    alert=True)
```

**2. Data Quality Issues**:
```python
if missing_pct > 5.0:
    logger.error("data_quality_alert",
                 missing_pct=missing_pct,
                 rows_missing=rows_missing,
                 alert=True)
```

**3. Risk Limit Breaches**:
```python
if current_drawdown > max_drawdown_pct:
    logger.critical("risk_limit_breach",
                    current_drawdown=current_drawdown,
                    max_allowed=max_drawdown_pct,
                    alert=True)
```

**4. System Resource Exhaustion**:
```python
import psutil

memory_usage = psutil.virtual_memory().percent
if memory_usage > 90:
    logger.warning("high_memory_usage",
                   memory_pct=memory_usage,
                   alert=True)
```

---

## Appendix

### Quick Reference

**Logging**:
```python
import structlog
logger = structlog.get_logger(__name__)
logger.info("event_name", key1="value1", key2=42)
```

**Configuration**:
```python
from finrl.applications.spy_rl_trading.settings import settings
print(settings.total_timesteps)
```

**Reproducibility**:
```python
from finrl.applications.spy_rl_trading.reproducibility import set_global_seed
set_global_seed()  # Uses settings.random_seed
```

**Environment Variables**:
```bash
export SPY_LOG_LEVEL=DEBUG
export SPY_TOTAL_TIMESTEPS=1000000
python train.py
```

---

**Document Revision History**:
- 2025-10-30: Initial version covering structured logging, configuration management, and monitoring
