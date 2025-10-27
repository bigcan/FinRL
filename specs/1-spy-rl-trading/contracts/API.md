# API Contracts: SPY RL Trading System

**Feature**: SPY RL Trading System | **Branch**: `1-spy-rl-trading` | **Date**: 2025-10-27

## Module: DataProcessor

**Purpose**: Download, clean, and featurize SPY market data.

### Method: `download_data(start_date, end_date) → DataFrame`

**Signature**:
```python
def download_data(
    self,
    start_date: str,          # "YYYY-MM-DD"
    end_date: str,            # "YYYY-MM-DD"
    symbol: str = "SPY"       # Ticker symbol
) -> pd.DataFrame
```

**Returns**:
- DataFrame with columns: date, open, high, low, close, volume, adjusted_close
- Shape: (N, 7) where N = number of trading days in range
- Index: DatetimeIndex (non-business days excluded)
- Data types: date (datetime64), OHLCV (float64), volume (int64)

**Errors**:
- `ValueError`: If start_date > end_date or date format invalid
- `ConnectionError`: If Yahoo Finance API unreachable
- `KeyError`: If ticker symbol not found

**Example**:
```python
processor = DataProcessor()
df = processor.download_data("2020-01-01", "2024-12-31")
# df.shape = (1260, 7)  # ~5 years × 252 trading days/year
```

---

### Method: `clean_data(df) → DataFrame`

**Signature**:
```python
def clean_data(self, df: pd.DataFrame) -> pd.DataFrame
```

**Input**:
- Raw OHLCV DataFrame from download_data()

**Returns**:
- Cleaned DataFrame with:
  - NaN rows removed
  - OHLC consistency validated (low ≤ close ≤ high, etc.)
  - Missing trading days flagged if >1% missing

**Side Effects**:
- Logs warnings if outliers detected (>5σ daily return)
- Prints data quality report

**Errors**:
- `ValueError`: If >1% trading days missing (fails data completeness check)

**Example**:
```python
df_raw = processor.download_data("2020-01-01", "2024-12-31")
df_clean = processor.clean_data(df_raw)
# Rows with NaN removed; data quality validated
```

---

### Method: `add_technical_indicators(df) → DataFrame`

**Signature**:
```python
def add_technical_indicators(
    self,
    df: pd.DataFrame,
    indicators: List[str] = None  # Default: config.INDICATORS
) -> pd.DataFrame
```

**Input**:
- Cleaned OHLCV DataFrame

**Returns**:
- DataFrame with added columns: macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma, vix, daily_return
- Shape: (N, 17) with all numeric columns float64

**Computation**:
- Technical indicators computed using stockstats library
- VIX downloaded from Yahoo Finance (market volatility index)
- daily_return = log(close_t / close_{t-1})
- Warmup period: first 60 rows may have NaN (SMA requires history)

**Example**:
```python
df_indicators = processor.add_technical_indicators(df_clean)
# df_indicators.shape = (1260, 17)
# Columns: date, open, high, low, close, volume, adjusted_close, + 10 indicators
```

---

### Method: `df_to_array(df) → Tuple[np.ndarray, np.ndarray, np.ndarray]`

**Signature**:
```python
def df_to_array(
    self,
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Returns**:
- Tuple of 3 arrays:
  1. **price_array**: shape (N, 5) - [open, high, low, close, volume] (float32)
  2. **tech_array**: shape (N, 10) - [macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma, close_60_sma, vix, daily_return] (float32)
  3. **turbulence_array**: shape (N,) - Market turbulence index (float32)

**Computation**:
- price_array: direct conversion from DataFrame columns
- tech_array: normalized technical indicators (zscore normalization)
- turbulence_array: rolling standard deviation of returns (20-day window)

**Example**:
```python
price_array, tech_array, turbulence_array = processor.df_to_array(df_indicators)
# price_array.shape = (1260, 5)
# tech_array.shape = (1260, 10)
# turbulence_array.shape = (1260,)
```

---

## Module: TradingEnvironment

**Purpose**: Gymnasium-compliant trading simulator for SPY.

### Method: `__init__(price_array, tech_array, turbulence_array, config) → None`

**Signature**:
```python
def __init__(
    self,
    price_array: np.ndarray,       # Shape (N, 5)
    tech_array: np.ndarray,        # Shape (N, 10)
    turbulence_array: np.ndarray,  # Shape (N,)
    config: dict = None            # {initial_balance, max_steps, ...}
) -> None
```

**Initializes**:
- Internal state: balance, shares_held, day counter
- Action space: Discrete(3) = {0: BUY, 1: HOLD, 2: SELL}
- Observation space: Box(shape=(13,), dtype=float32) = [balance, shares_held, price, 10 indicators, turbulence]

**Example**:
```python
env = TradingEnvironment(price_array, tech_array, turbulence_array)
# Action space: Discrete(3)
# Observation space: Box(shape=(13,), low=-inf, high=inf, dtype=float32)
```

---

### Method: `reset() → np.ndarray`

**Signature**:
```python
def reset(self, seed: int = None) -> np.ndarray
```

**Returns**:
- Initial observation: shape (13,) float32
- Elements: [balance_normalized, shares_held, price, tech_1, ..., tech_10, turbulence]

**Side Effects**:
- Resets internal state: balance = initial_balance, shares_held = 0, day = 0
- Sets random seed if provided

**Example**:
```python
obs, info = env.reset()
# obs.shape = (13,)
# obs[0] ≈ 1.0 (normalized initial balance)
# obs[1] = 0.0 (no shares at start)
```

---

### Method: `step(action) → Tuple[np.ndarray, float, bool, dict]`

**Signature**:
```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]
```

**Input**:
- action ∈ {0, 1, 2}
  - 0: BUY (if not holding, buy 1 share; else stay long)
  - 1: HOLD (maintain position)
  - 2: SELL (if holding, sell; else stay flat)

**Returns**:
- **observation**: shape (13,) float32 - next state
- **reward**: float - log return if holding, else 0
- **terminated**: bool - True if end of episode (day >= N-1)
- **info**: dict - {"day": int, "balance": float, "price": float, "shares": int, "action": int}

**Side Effects**:
- Updates internal state: balance, shares_held, day counter
- Executes trade at current day's close price (no slippage)

**Errors**:
- `ValueError`: If action not in {0, 1, 2}

**Example**:
```python
obs, reward, terminated, info = env.step(0)  # BUY action
# reward = 0.005 (if holding and daily return was +0.5%)
# obs.shape = (13,)
# info = {"day": 1, "balance": 99950, "price": 380.5, "shares": 1, "action": 0}
```

---

### Method: `_get_state() → np.ndarray`

**Signature**:
```python
def _get_state(self) -> np.ndarray
```

**Returns**:
- Observation: shape (13,) float32
- Concatenation: [balance_norm, shares_held, price_norm, tech_array_normalized, turbulence_norm]

**Normalization**:
- balance_norm = balance / initial_balance
- price_norm = zscore(price) based on training data statistics
- tech_array: already normalized in df_to_array()
- turbulence_norm = min(turbulence / 2.0, 1.0) (capped at market extreme)

---

### Method: `_calculate_reward() → float`

**Signature**:
```python
def _calculate_reward(self) -> float
```

**Returns**:
- reward: float
  - If shares_held == 1: log_return for the day
  - Else: 0.0

**Computation**:
```python
log_return = np.log(price_t / price_{t-1})
reward = log_return if self.shares_held == 1 else 0.0
```

---

## Module: PPOAgent

**Purpose**: Wrapper around Stable-Baselines3 PPO for SPY trading.

### Method: `__init__(env, config) → None`

**Signature**:
```python
def __init__(
    self,
    env: TradingEnvironment,
    config: dict = None  # PPO_PARAMS from config.py
) -> None
```

**Initializes**:
- PPO policy via Stable-Baselines3
- TensorBoard logging directory
- Callbacks for early stopping

**Example**:
```python
from stable_baselines3 import PPO
agent = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    clip_range=0.2,
    # ... other params from PPO_PARAMS
    tensorboard_log="./tensorboard_logs"
)
```

---

### Method: `train(total_timesteps, callback=None) → None`

**Signature**:
```python
def train(
    self,
    total_timesteps: int,      # e.g., 100_000
    callback: BaseCallback = None
) -> None
```

**Side Effects**:
- Trains PPO policy on environment for total_timesteps
- Logs metrics to TensorBoard (episode return, loss, etc.)
- May stop early if callback triggered (e.g., StopTrainingOnMaxSteps)

**Example**:
```python
agent.learn(
    total_timesteps=100_000,
    tb_log_name="spy_ppo_run_1"
)
# View logs: tensorboard --logdir ./tensorboard_logs
```

---

### Method: `save(path) → None`

**Signature**:
```python
def save(self, path: str) -> None
```

**Side Effects**:
- Saves trained policy to disk (pickle format)
- Filename: e.g., "spy_ppo_model.zip"

**Example**:
```python
agent.save("./models/spy_ppo_trained")
```

---

### Method: `load(path) → PPO`

**Signature**:
```python
@staticmethod
def load(path: str, env: TradingEnvironment = None) -> PPO
```

**Returns**:
- Loaded PPO policy ready for inference

**Example**:
```python
trained_agent = PPO.load("./models/spy_ppo_trained", env=test_env)
```

---

## Module: Backtester

**Purpose**: Evaluate trained agent on hold-out test data.

### Method: `run(agent, env) → BacktestResult`

**Signature**:
```python
def run(
    self,
    agent: PPO,
    env: TradingEnvironment
) -> dict  # BacktestResult fields
```

**Returns**:
- Dictionary with keys:
  - `total_return`: float (e.g., 0.12 for 12%)
  - `sharpe_ratio`: float (e.g., 0.75)
  - `max_drawdown`: float (e.g., -0.08 for -8%)
  - `win_rate`: float (e.g., 0.55 for 55%)
  - `total_trades`: int
  - `baseline_return`: float (buy-and-hold)
  - `baseline_sharpe`: float

**Example**:
```python
backtester = Backtester()
results = backtester.run(trained_agent, test_env)
# results = {
#     "total_return": 0.12,
#     "sharpe_ratio": 0.75,
#     "max_drawdown": -0.08,
#     "win_rate": 0.55,
#     ...
# }
```

---

## Summary: API Contracts

| Module | Key Methods | Input | Output |
|--------|------------|-------|--------|
| **DataProcessor** | download_data, clean_data, add_indicators, df_to_array | Date range, symbol | DataFrames → Arrays |
| **TradingEnvironment** | reset, step, _get_state, _calculate_reward | action (int) | observation, reward, done |
| **PPOAgent** | train, save, load | total_timesteps | trained model |
| **Backtester** | run | agent, env | BacktestResult dict |

---

**Phase 1 Status**: ✅ API contracts defined. Ready for quickstart and task generation.
