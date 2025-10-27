# Research: SPY RL Trading System (Phase 0)

**Feature**: SPY RL Trading System | **Branch**: `1-spy-rl-trading` | **Date**: 2025-10-27

## Overview

This document consolidates research findings for key technical decisions in the SPY RL trading system. Each section addresses a clarification from the feature specification and provides evidence-based recommendations.

---

## 1. PPO Hyperparameter Selection for Discrete Trading

### Decision

Implement PPO via Stable-Baselines3 with the following **default hyperparameters** for SPY discrete trading:

```python
PPO_PARAMS = {
    "n_steps": 2048,              # Rollout buffer size (one episode = ~252 days)
    "batch_size": 64,             # Batch size for policy updates
    "n_epochs": 10,               # Epochs per update
    "learning_rate": 3e-4,        # Policy learning rate
    "clip_range": 0.2,            # PPO clip ratio (epsilon)
    "clip_range_vf": 0.1,         # Value function clip range
    "ent_coef": 0.01,             # Entropy coefficient (exploration bonus)
    "vf_coef": 0.5,               # Value function loss weight
    "max_grad_norm": 0.5,         # Gradient clipping
    "gae_lambda": 0.95,           # GAE (Generalized Advantage Estimation) lambda
    "use_sde": False,             # Stochastic Policy (disabled for simplicity)
    "sde_sample_freq": -1,        # Not used
    "target_kl": 0.01,            # Target KL divergence (early stopping)
    "policy": "MlpPolicy",        # Multi-layer perceptron policy (feed-forward)
    "policy_kwargs": {
        "net_arch": [256, 256],   # Hidden layer sizes
        "activation_fn": "relu",  # ReLU activation
    }
}
```

### Rationale

1. **n_steps = 2048**: One trading year ≈ 252 days; 2048 steps ≈ 8 trading years. Suitable for on-policy learning from 5-year historical data.

2. **batch_size = 64**: Small batches provide stable policy updates without excessive variance. Standard for financial data.

3. **learning_rate = 3e-4**: Conservative learning rate; financial data is noisy, aggressive updates risk divergence.

4. **clip_range = 0.2**: Standard PPO value; prevents excessive policy changes that destabilize trading signals.

5. **ent_coef = 0.01**: Low entropy bonus encourages exploitation over exploration; trading rewards are well-defined (exploitation preferred).

6. **net_arch = [256, 256]**: Moderate network capacity; SPY daily data has limited state space (price + ~10 indicators). Larger networks risk overfitting.

7. **target_kl = 0.01**: Early stopping prevents policy divergence; recommended for financial domains.

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| DQN (value-based) | PPO more stable for policy gradient on continuous state space |
| A2C (actor-critic) | PPO superior variance reduction via GAE; PPO is standard in RL trading |
| TD3 (continuous control) | Discrete action space; PPO specifically designed for discrete actions |
| Large network [512, 512] | Financial data sparse; risk of overfitting to historical noise |
| High learning rate (1e-3) | Noisy financial rewards; conservative rate prevents divergence |

### Validation Plan

- Train PPO with default params; verify convergence to >5% training return (SC-002)
- Test hyperparameter sensitivity (SC-009): vary learning_rate 10× → measure Sharpe ratio variance
- Run 5 seeds; measure ±10% policy convergence variance (US1 scenario 3)

---

## 2. Environment Reward Function (Log Return)

### Decision

Implement daily log-return reward scaled by position:

```python
def calculate_reward(self):
    """
    Compute daily log-return reward.

    Returns log(close_t / close_{t-1}) if holding (position=1),
    else 0 if flat (position=0).
    """
    current_price = self.data['close'][self.day]
    prev_price = self.data['close'][self.day - 1]
    log_return = np.log(current_price / prev_price)

    # Scale by position: reward only if agent holds
    if self.position == 1:  # Long
        reward = log_return
    else:  # Flat
        reward = 0.0

    return reward
```

### Rationale

1. **Log return vs. simple return**: Log returns are additive (more interpretable for RL); avoid biasing toward small prices.

2. **Position scaling**: Agent receives reward only when holding, incentivizing buy signals before upturns and sell signals before downturns.

3. **Daily frequency**: Sufficient resolution for trend capture; intraday complexity deferred.

4. **No transaction costs**: v1.0 assumes perfect execution; realism can be added in environment variants.

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| Simple return (% change) | Log returns more stable numerically; standard in quant finance |
| Sharpe ratio optimization | Deferred to v2.0; requires expensive rolling variance calculation |
| Reward shaping (e.g., -1 for hold) | Pure log return simpler; penalizing hold may bias toward overtrading |
| Scalp returns (intraday) | Daily aggregation sufficient; intraday complexity deferred |

### Validation Plan

- Verify reward computation in environment unit tests (test_environment.py)
- Visualize reward curves during training; confirm agent receives positive cumulative rewards
- Backtest: compare agent Sharpe ratio to buy-and-hold baseline (SC-004)

---

## 3. Data Quality & Validation Standards

### Decision

Apply following **data quality checks** in SPY processor:

1. **Completeness**: Minimum 252 trading days per year; alert if >5 missing days/year.
2. **Outliers**: Flag daily returns >5σ; review before training.
3. **Data continuity**: Enforce strict chronological order; no future prices.
4. **Adjusted close**: Use adjusted close to handle splits/dividends automatically.
5. **NaN handling**: Remove or forward-fill missing OHLCV values (except volume=0 valid).

```python
def clean_data(self):
    """Clean SPY OHLCV data per FinRL standard processor interface."""
    # Step 1: Remove NaN rows
    self.df = self.df.dropna(subset=['close', 'open', 'high', 'low', 'volume'])

    # Step 2: Check continuity
    dates = pd.to_datetime(self.df['date'])
    bdays = pd.bdate_range(dates.min(), dates.max())
    if len(dates) < 0.99 * len(bdays):
        raise ValueError(f"Data missing >1% trading days: {len(dates)}/{len(bdays)}")

    # Step 3: Flag outliers
    self.df['daily_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
    outliers = np.abs(self.df['daily_return']) > 5 * self.df['daily_return'].std()
    if outliers.sum() > 0:
        print(f"WARNING: {outliers.sum()} outlier(s) detected; review before training")

    return self.df
```

### Rationale

1. **99% completeness**: SPY is liquid; >99% coverage is realistic and ensures sufficient training data.
2. **5σ threshold**: Extreme moves (circuit breakers, data errors) warrant review; doesn't invalidate data.
3. **Adjusted close**: Automatic dividend/split handling matches market reality.

### Validation Plan

- Unit test: verify clean_data() rejects corrupted datasets
- Unit test: verify clean_data() flags outliers (e.g., 2020 March COVID drop)
- Integration test: confirm full 2020-2025 SPY dataset passes validation

---

## 4. TensorBoard Integration for Training Monitoring

### Decision

Use **Stable-Baselines3 built-in TensorBoard support** to log:

```python
# In agent trainer
agent = PPO(
    policy="MlpPolicy",
    env=trading_env,
    tensorboard_log="./tensorboard_logs/spy_ppo",
    **ppo_params
)

agent.learn(
    total_timesteps=100_000,  # Adjust for 2020-2024 data
    tb_log_name="spy_ppo_run_1",
    callback=StopTrainingOnMaxSteps(max_steps=100_000)
)

# View training: tensorboard --logdir ./tensorboard_logs/spy_ppo
```

**Logged metrics**:
- Episode return (cumulative log return per episode)
- Episode length (days per episode)
- Policy loss and value loss
- Explained variance
- KL divergence (policy divergence from old policy)

### Rationale

1. **SB3 native support**: Built-in logging; no additional dependencies.
2. **Real-time visualization**: TensorBoard enables monitoring during long training runs.
3. **Debugging**: Loss curves help diagnose training issues (divergence, plateau).
4. **Constitutional requirement**: (FR-010, V. Test-First & Observability) mandates TensorBoard logging.

### Validation Plan

- Configure TensorBoard logging in training script
- Verify logs generated during test run
- Validate smooth convergence (SC-007): <±20% moving average oscillation

---

## 5. Backtesting Metrics: Sharpe Ratio & Max Drawdown

### Decision

Compute backtest metrics per standard financial analysis:

```python
def calculate_metrics(returns):
    """Compute backtest performance metrics."""
    cumulative_return = np.exp(np.sum(returns)) - 1  # From log returns

    # Sharpe ratio (annualized)
    annual_std = np.std(returns) * np.sqrt(252)  # 252 trading days/year
    annual_return = np.mean(returns) * 252
    sharpe = annual_return / annual_std if annual_std > 0 else 0

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_dd = np.min(drawdown)

    # Win rate (% days with positive return)
    win_rate = np.mean(returns > 0)

    return {
        "total_return": cumulative_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate
    }
```

### Rationale

1. **Sharpe ratio**: Standard risk-adjusted metric; enables comparison to baselines.
2. **Max drawdown**: Measures downside risk; critical for risk-averse investors.
3. **Win rate**: Simple intuitive metric; % of profitable days.
4. **Annualization**: Financial convention; allows comparison across different time periods.

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| Sortino ratio (downside volatility) | Sharpe simpler; Sortino deferred to v2.0 |
| Calmar ratio (return/max dd) | Sharpe + max dd separately more informative |
| Custom reward metric | Sharpe/drawdown align with financial standards |

### Validation Plan

- Unit test: verify metrics computation on synthetic data
- Backtest: compare agent Sharpe to buy-and-hold baseline (SP-004)
- Confirm agent Sharpe ≥0.5 on 2025 test data (SC-004)

---

## 6. Data Splitting Strategy (No Lookahead Bias)

### Decision

Implement **strict chronological split** at specified date:

```python
# Config
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-12-31"

# Data loading
train_data = download_data(TRAIN_START_DATE, TRAIN_END_DATE)  # ~1260 days
test_data = download_data(TEST_START_DATE, TEST_END_DATE)     # ~252 days

# Agent trained on train_data only
train_env = TradingEnv(train_data)
agent.learn(env=train_env, ...)

# Agent evaluated on test_data (no retraining)
test_env = TradingEnv(test_data)
backtest_results = agent.evaluate(env=test_env, ...)
```

### Rationale

1. **Chronological**: Respects temporal ordering; mimics real trading (no future data).
2. **No data leakage**: Technical indicators computed per-date; no forward-looking stats.
3. **Generalization test**: 2025 data unseen during training; true out-of-sample validation.

### Validation Plan

- Unit test: verify train/test date ranges are non-overlapping
- Integration test: confirm agent performance degrades on test data (vs. training) if overfit
- Validate test data never used during training (assertion in code)

---

## 7. Technical Indicators for Environment State

### Decision

Use **FinRL standard indicators** from `config.py`:

```python
INDICATORS = [
    'macd',           # Moving Average Convergence Divergence
    'boll_ub',        # Bollinger Bands upper
    'boll_lb',        # Bollinger Bands lower
    'rsi_30',         # 30-period RSI
    'cci_30',         # 30-period Commodity Channel Index
    'dx_30',          # 30-period Directional Movement Index
    'close_30_sma',   # 30-period Simple Moving Average
    'close_60_sma',   # 60-period Simple Moving Average
    'vix'             # Volatility Index (market fear gauge)
]
```

### Rationale

1. **FinRL convention**: Standard indicators used across FinRL ecosystem; enables code reuse.
2. **Complementary signals**: Mix of trend (SMA), volatility (Bollinger), momentum (RSI, MACD), and market regime (VIX).
3. **Lightweight**: ~10 features; reduces state space; computational efficiency.
4. **Stability**: Indicators mature, well-tested in trading systems.

### Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| Deep learning features (CNN on prices) | Overhead; simpler indicators sufficient for daily trading |
| Sentiment analysis | External data dependency; deferred to v2.0 |
| Additional indicators | Avoid overfitting; 10 features adequate for agent |

### Validation Plan

- Verify indicators computed correctly (unit test in data processor)
- Visualize indicator distributions for SPY 2020-2025 (sanity check)
- Confirm state space dimensionality: [balance, shares, price, 10 indicators, turbulence] = 13 dimensions

---

## Summary: Research Findings → Implementation Plan

| Research Item | Decision | Deliverable |
|---------------|----------|-------------|
| 1. PPO Hyperparameters | Default params: lr=3e-4, clip=0.2, net=[256,256] | finrl/config.py: PPO_PARAMS |
| 2. Reward Function | Daily log return scaled by position | finrl/applications/spy_rl_trading/environment.py |
| 3. Data Quality | 99% completeness, 5σ outliers flagged | finrl/applications/spy_rl_trading/data_processor.py |
| 4. TensorBoard Integration | SB3 native logging → ./tensorboard_logs | finrl/applications/spy_rl_trading/agent.py |
| 5. Backtest Metrics | Sharpe, max drawdown, win rate | finrl/applications/spy_rl_trading/metrics.py |
| 6. Data Splitting | Chronological split: train 2020-2024, test 2025 | finrl/config.py: Date ranges |
| 7. Technical Indicators | FinRL standard 10 indicators + VIX | finrl/config.py: INDICATORS |

---

**Phase 0 Status**: ✅ Complete. All research questions answered; ready for Phase 1 design (data-model.md, contracts/, quickstart.md).
