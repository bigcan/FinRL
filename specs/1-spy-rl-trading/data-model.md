# Data Model: SPY RL Trading System (Phase 1)

**Feature**: SPY RL Trading System | **Branch**: `1-spy-rl-trading` | **Date**: 2025-10-27

## Overview

This document defines the core data entities for the SPY RL trading system. Each entity represents a key concept in the trading pipeline: market data, environment state, agent actions, and performance results.

---

## Entity: MarketData

**Purpose**: Daily OHLCV candles for SPY with computed technical indicators.

**Fields**:

| Field | Type | Description | Constraints | Source |
|-------|------|-------------|-------------|--------|
| `date` | datetime | Trading date | Non-null, sorted ascending | Yahoo Finance |
| `open` | float | Opening price | >0 | Yahoo Finance |
| `high` | float | Highest price | ≥ open, high | Yahoo Finance |
| `low` | float | Lowest price | ≤ open, low | Yahoo Finance |
| `close` | float | Closing price (adjusted for splits/dividends) | >0 | Yahoo Finance |
| `volume` | int | Trading volume | ≥0 | Yahoo Finance |
| `macd` | float | MACD indicator | -∞ to +∞ | Computed (stockstats) |
| `boll_ub` | float | Bollinger Bands upper bound | ≥ close | Computed (stockstats) |
| `boll_lb` | float | Bollinger Bands lower bound | ≤ close | Computed (stockstats) |
| `rsi_30` | float | 30-period RSI | [0, 100] | Computed (stockstats) |
| `cci_30` | float | 30-period CCI | -∞ to +∞ | Computed (stockstats) |
| `dx_30` | float | 30-period DX | [0, 100] | Computed (stockstats) |
| `close_30_sma` | float | 30-period simple moving average | >0 | Computed (stockstats) |
| `close_60_sma` | float | 60-period simple moving average | >0 | Computed (stockstats) |
| `vix` | float | Volatility index (market fear) | [0, 100] typically | Yahoo Finance |
| `daily_return` | float | Log return: log(close_t / close_{t-1}) | -∞ to +∞ | Computed |

**Relationships**:
- One MarketData row per trading day
- Ordered by date (strict chronological order, no gaps >1 trading day)

**Validation**:
- OHLC consistency: low ≤ close ≤ high, low ≤ open ≤ high
- Price continuity: no future prices (no data leakage)
- Completeness: ≥99% trading days per year (≤5 missing days/year)
- Technical indicators: all non-null after warmup period (60 days for SMA)

**Usage**:
- Training environment receives batches of MarketData rows to compute state
- Backtest engine iterates through MarketData chronologically
- Metrics computed from daily_return field

---

## Entity: EnvironmentState

**Purpose**: Observation space at each trading timestep t.

**Fields**:

| Field | Type | Description | Range | Normalized? |
|-------|------|-------------|-------|-------------|
| `balance` | float | Cash balance in account | [0, ∞) | Yes (divide by initial balance) |
| `shares_held` | int | Number of SPY shares in position | {0, 1} (discrete, 1 unit) | No |
| `price_t` | float | Current SPY close price at time t | >0 | Yes (log scale or zscore) |
| `macd_t` | float | MACD at time t | -∞ to +∞ | Yes (zscore) |
| `boll_ub_t` | float | Bollinger upper at time t | >0 | Yes (percentage of price) |
| `boll_lb_t` | float | Bollinger lower at time t | >0 | Yes (percentage of price) |
| `rsi_30_t` | float | RSI at time t | [0, 100] | Yes (divide by 100) |
| `cci_30_t` | float | CCI at time t | -∞ to +∞ | Yes (zscore) |
| `dx_30_t` | float | DX at time t | [0, 100] | Yes (divide by 100) |
| `sma_30_t` | float | 30-SMA at time t | >0 | Yes (price - sma30) / sma30 |
| `sma_60_t` | float | 60-SMA at time t | >0 | Yes (price - sma60) / sma60 |
| `vix_t` | float | VIX at time t | [0, 100] typically | Yes (divide by 100) |
| `turbulence_index` | float | Market turbulence (std of past 20 returns) | [0, ∞) | Yes (cap at 2σ) |

**Representation**:
- Python: NumPy array or gym.spaces.Box (shape: (13,), dtype: float32)
- Gymnasium Space: `Box(low=-∞, high=∞, shape=(13,), dtype=np.float32)`

**Normalization**:
- All features normalized to approximate [0, 1] range for stable policy learning
- Normalization computed per training run (fit on training data only, apply to test data)

**Relationships**:
- One EnvironmentState per trading day
- Deterministic function of MarketData at time t + portfolio state (balance, shares_held)

**Validation**:
- balance ≥0 (cannot go negative)
- shares_held ∈ {0, 1} (single unit position)
- All technical indicators present (no NaN after warmup)

**Usage**:
- Input to PPO policy network (neural network input layer)
- Returned by environment.reset() and environment.step()

---

## Entity: Action

**Purpose**: Trading decision at each timestep.

**Fields**:

| Field | Type | Description | Valid Values | Impact |
|-------|------|-------------|--------------|--------|
| `action` | int | Discrete trading action | {0, 1, 2} | Determines reward and next position |

**Action Space**:
- Gymnasium Space: `Discrete(3)`
- Mapping:
  - **0**: BUY → If not holding, buy 1 SPY share; if holding, stay long
  - **1**: HOLD → Maintain current position (do nothing)
  - **2**: SELL → If holding, sell 1 SPY share; if flat, stay flat

**Execution**:
- Action executed at close price at time t
- No slippage modeling (v1.0 assumes perfect execution)
- No transaction costs (v1.0 assumes zero fees)

**Relationships**:
- One action per EnvironmentState
- Action sampled from PPO policy π(action | state)

**Validation**:
- Action must be in {0, 1, 2}
- No constraints on action frequency (agent can buy/sell consecutive days)

**Usage**:
- Output of PPO policy network (output layer: action logits)
- Input to environment.step() to compute reward and next state

---

## Entity: Reward

**Purpose**: Training signal for PPO policy at each timestep.

**Fields**:

| Field | Type | Description | Computation |
|-------|------|-------------|-------------|
| `log_return_t` | float | Daily log return: log(close_t / close_{t-1}) | np.log(price_t / price_{t-1}) |
| `reward` | float | Scaled reward if position=1, else 0 | log_return_t if shares_held==1, else 0.0 |

**Computation Logic**:

```python
def calculate_reward(self, action, day):
    """Reward = log return only when holding; 0 when flat."""
    current_price = self.data['close'][day]
    prev_price = self.data['close'][day - 1]
    log_return = np.log(current_price / prev_price)

    if self.shares_held == 1:  # Currently holding
        return log_return
    else:  # Currently flat
        return 0.0
```

**Relationships**:
- One reward per timestep
- Deterministic function of (action, price_t, price_{t-1}, shares_held)

**Scaling**:
- Rewards in range [-0.10, 0.10] typically (daily returns rarely >10%)
- No additional scaling (log returns naturally bounded)

**Validation**:
- Reward must be finite (no NaN, inf)
- Episode cumulative reward should converge to >5% over training (SC-002)

**Usage**:
- Returned by environment.step() as second value
- Accumulated over episode for policy gradient calculation
- Monitored in TensorBoard for convergence tracking

---

## Entity: Episode

**Purpose**: Complete training episode (one trading year or fixed timestep window).

**Fields**:

| Field | Type | Description | Value |
|-------|------|-------------|-------|
| `episode_id` | int | Unique episode identifier | Incremented per reset |
| `start_date` | datetime | Episode start date | From training data |
| `end_date` | datetime | Episode end date (or terminal) | After n_steps or MarketData exhausted |
| `episode_length` | int | Number of timesteps in episode | Typical: 252 (1 year) |
| `cumulative_reward` | float | Sum of rewards: Σ reward_t | Should be >5% for profitable policies |
| `final_balance` | float | Portfolio value at episode end | Starting balance × (1 + cumulative_return) |
| `max_drawdown` | float | Maximum cumulative loss during episode | Min(cumulative_return) |

**Relationships**:
- Multiple episodes per training run
- Non-overlapping episodes (no data reuse)

**Validation**:
- episode_length ≤ 252 (training episodes limited to ~1 year)
- cumulative_reward convergence: mean(cumulative_reward over last 100 episodes) > 5%

**Usage**:
- Training metric: monitor cumulative reward per episode
- Convergence criterion: stop training if reward plateau detected

---

## Entity: BacktestResult

**Purpose**: Summary of agent performance on hold-out test data.

**Fields**:

| Field | Type | Description | Computation |
|-------|------|-------------|-------------|
| `test_period` | str | Date range | "2025-01-01 to 2025-12-31" |
| `total_trades` | int | Number of buy/sell actions | Count of action != 1 |
| `buy_count` | int | Number of buy actions | Count of action == 0 |
| `sell_count` | int | Number of sell actions | Count of action == 2 |
| `win_trades` | int | Trades with positive return | Count of daily_return > 0 while holding |
| `loss_trades` | int | Trades with negative return | Count of daily_return < 0 while holding |
| `total_return` | float | Cumulative return % | (final_balance / initial_balance) - 1 |
| `annual_return` | float | Annualized return % | total_return (since period is ~1 year) |
| `sharpe_ratio` | float | Risk-adjusted return | (annual_return / annual_volatility) where volatility = std(daily_returns) × √252 |
| `max_drawdown` | float | Maximum cumulative loss % | Min(cumulative_return) / Max(cumulative_return) |
| `win_rate` | float | % days with positive return | Count(daily_return > 0) / total_days |
| `baseline_return` | float | Buy-and-hold baseline return % | (price_final / price_initial) - 1 |
| `baseline_sharpe` | float | Buy-and-hold baseline Sharpe | (annual_return_baseline / annual_vol_baseline) |

**Relationships**:
- One BacktestResult per trained model evaluation
- Always computed on hold-out test data (never training data)

**Validation**:
- total_return > 0 (profitable, SC-003)
- sharpe_ratio ≥ 0.5 (risk-adjusted performance, SC-004)
- total_return vs baseline_return: compare RL to passive buy-and-hold

**Usage**:
- Summary metric: report final performance
- Comparison: RL agent vs. baseline
- Validation: ensure agent generalizes to test data

---

## Data Flow Diagram

```
┌─────────────────┐
│  Yahoo Finance  │
│   OHLCV Data    │
└────────┬────────┘
         │
         v
┌─────────────────────────────────┐
│  DataProcessor.download_data()  │
│  - Load SPY daily 2020-2025     │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  DataProcessor.clean_data()     │
│  - Remove NaN, validate gaps    │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│  DataProcessor.add_indicators() │
│  - Compute technical indicators │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────────────────┐
│  DataFrame: MarketData (13 columns)        │
│  1260 rows (2020-2024 training)            │
│  + 252 rows (2025 testing)                 │
└────────┬────────────────────────────────────┘
         │
    ┌────┴────┐
    │          │
    v          v
┌──────────────────┐   ┌─────────────────┐
│ TradingEnv       │   │ TradingEnv      │
│ (train data)     │   │ (test data)     │
└────────┬─────────┘   └────────┬────────┘
         │                      │
         v                      v
┌──────────────────────────────────────┐
│ Environment.reset()                  │
│ → EnvironmentState (shape: (13,))    │
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│ PPO Policy Network                   │
│ Input: EnvironmentState (13,)        │
│ Output: Action logits (3,)           │
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│ Environment.step(action)             │
│ → (state_next, reward, done, info)   │
│ where reward = Reward                │
└────────┬─────────────────────────────┘
         │
         v
┌──────────────────────────────────────┐
│ Episode (accumulated rewards)        │
│ → cumulative_reward, final_balance   │
└────────┬─────────────────────────────┘
         │
    ┌────┴──────────────────────┐
    │ (training loop repeat)    │
    │                           │
    v                           v
(train mode)              (test/eval mode)
    │                           │
    └───────────────────────────┘
                │
                v
        ┌──────────────────────────────────┐
        │ BacktestResult                   │
        │ - total_return, sharpe_ratio     │
        │ - max_drawdown, win_rate         │
        │ - vs. baseline metrics           │
        └──────────────────────────────────┘
```

---

## Summary: Key Data Entities

| Entity | Purpose | Example Values |
|--------|---------|-----------------|
| **MarketData** | Daily OHLCV + indicators | 1512 rows (SPY 2020-2025) |
| **EnvironmentState** | Agent observation | shape (13,): [0.95, 1, 380.5, -0.02, ...] |
| **Action** | Trading decision | 0 (buy), 1 (hold), 2 (sell) |
| **Reward** | Training signal | 0.005 (if holding and log return = 0.5%) |
| **Episode** | Training batch | 252 timesteps, cumulative return = 8% |
| **BacktestResult** | Final performance | Sharpe = 0.75, total return = 12% |

---

**Phase 1 Status**: ✅ Data model defined. Ready for contract definitions and quickstart.
