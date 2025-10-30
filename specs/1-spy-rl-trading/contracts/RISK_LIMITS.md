# Risk Limits Interface Contract

**Feature**: SPY RL Trading System - Risk Management
**Phase**: Phase 12 (Finance-Specific Hardening)
**Status**: Planning
**Last Updated**: 2025-10-30

---

## Overview

This document defines the risk management interface for the SPY RL Trading System. Risk limits prevent catastrophic losses and enforce trading guardrails in backtesting, paper trading, and live trading environments.

---

## Risk Limit Types

### 1. Maximum Drawdown Limit

**Definition**: Maximum allowable decline from peak portfolio value

**Purpose**: Prevent catastrophic losses, trigger emergency stop-trading

**Typical Values**:
- **Conservative**: 10-15%
- **Moderate**: 15-20%
- **Aggressive**: 20-30%

**Formula**:
```
drawdown = (peak_nav - current_nav) / peak_nav
breach = drawdown > max_drawdown_pct
```

**Action on Breach**: Stop all trading, liquidate positions, notify user

---

### 2. Daily Loss Limit

**Definition**: Maximum allowable loss in a single trading day

**Purpose**: Prevent runaway losses, limit exposure to extreme volatility

**Typical Values**:
- **Conservative**: 2-3%
- **Moderate**: 3-5%
- **Aggressive**: 5-10%

**Formula**:
```
daily_loss = (previous_day_nav - current_nav) / previous_day_nav
breach = daily_loss > max_daily_loss_pct
```

**Action on Breach**: Stop trading for remainder of day, resume next trading day

---

### 3. Position Size Limit

**Definition**: Maximum position size as percentage of portfolio NAV

**Purpose**: Prevent over-concentration, enforce diversification

**Typical Values**:
- **Single Asset (SPY)**: 100% (fully invested)
- **Multiple Assets**: 10-30% per asset
- **High Risk Assets**: 5-10% per asset

**Formula**:
```
position_size_pct = (position_value / portfolio_nav) * 100
breach = position_size_pct > max_position_size_pct
```

**Action on Breach**: Reject order, log warning

---

### 4. Leverage Limit

**Definition**: Maximum leverage allowed (margin borrowing)

**Purpose**: Prevent margin calls, reduce risk of forced liquidation

**Typical Values**:
- **No Leverage**: 1.0x (cash account)
- **Conservative**: 1.5x
- **Moderate**: 2.0x
- **Aggressive**: 3.0x (pattern day trader)

**Formula**:
```
leverage = total_position_value / equity
breach = leverage > max_leverage
```

**Action on Breach**: Reject order, reduce positions to meet leverage limit

---

### 5. Order Size Limit

**Definition**: Maximum order size per trade

**Purpose**: Prevent fat-finger errors, limit single-trade impact

**Typical Values**:
- **Retail**: 100-1000 shares
- **Small Account**: 1-10 shares
- **Large Account**: 1000-10000 shares

**Formula**:
```
breach = order_quantity > max_order_size
```

**Action on Breach**: Reject order, log warning

---

### 6. Trading Frequency Limit

**Definition**: Maximum number of trades per day/week

**Purpose**: Prevent pattern day trader violations, reduce transaction costs

**Typical Values**:
- **Pattern Day Trader Threshold**: 4 day trades per 5 trading days
- **Conservative**: 1-2 trades per day
- **Active**: 5-10 trades per day

**Formula**:
```
breach = trades_today > max_trades_per_day
```

**Action on Breach**: Reject order until next trading day

---

## Interface Definition

### RiskLimits (Configuration)

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class RiskLimits:
    """
    Risk limit configuration.

    All limits are optional (None = no limit).
    """

    # Drawdown limits
    max_drawdown_pct: Decimal | None = Decimal("0.20")  # 20%
    max_daily_loss_pct: Decimal | None = Decimal("0.05")  # 5%

    # Position limits
    max_position_size_pct: Decimal | None = Decimal("1.00")  # 100% (single asset)
    max_leverage: Decimal | None = Decimal("1.0")  # No leverage (cash account)

    # Order limits
    max_order_size: int | None = None  # No limit
    max_trades_per_day: int | None = None  # No limit
    max_trades_per_week: int | None = None  # No limit

    # Pattern day trader limits
    pdt_protection: bool = True  # Prevent PDT violations
    pdt_threshold: int = 4  # 4 day trades per 5 trading days

    def __post_init__(self):
        """Validate risk limits."""
        if self.max_drawdown_pct is not None and self.max_drawdown_pct <= 0:
            raise ValueError("max_drawdown_pct must be positive")

        if self.max_daily_loss_pct is not None and self.max_daily_loss_pct <= 0:
            raise ValueError("max_daily_loss_pct must be positive")

        if self.max_leverage is not None and self.max_leverage < Decimal("1.0"):
            raise ValueError("max_leverage must be >= 1.0")
```

---

### RiskViolation (Exception)

```python
from enum import Enum
from dataclasses import dataclass

class RiskViolationType(Enum):
    """Risk violation types."""
    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS = "daily_loss"
    POSITION_SIZE = "position_size"
    LEVERAGE = "leverage"
    ORDER_SIZE = "order_size"
    TRADING_FREQUENCY = "trading_frequency"
    PATTERN_DAY_TRADER = "pattern_day_trader"


@dataclass
class RiskViolation:
    """
    Risk limit violation details.

    Attributes:
        violation_type: Type of violation
        current_value: Current value that triggered violation
        limit_value: Configured limit value
        message: Human-readable violation message
        severity: Violation severity ("warning", "error", "critical")
    """
    violation_type: RiskViolationType
    current_value: Decimal
    limit_value: Decimal
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"{self.violation_type.value}: {self.message} (current={self.current_value}, limit={self.limit_value})"


class RiskLimitViolationError(Exception):
    """Exception raised when risk limit violated."""

    def __init__(self, violation: RiskViolation):
        self.violation = violation
        super().__init__(str(violation))
```

---

### RiskManager (Core Implementation)

```python
from abc import ABC, abstractmethod
from typing import List
from decimal import Decimal
from datetime import datetime, timedelta

class RiskManager(ABC):
    """
    Abstract risk manager interface.

    All risk managers must implement this interface to ensure consistent
    risk enforcement across backtesting and live trading.
    """

    @abstractmethod
    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        current_nav: Decimal,
        current_positions: dict,
    ) -> List[RiskViolation]:
        """
        Check if order violates any risk limits.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            quantity: Number of shares
            price: Order price
            current_nav: Current portfolio NAV
            current_positions: Current positions {symbol: quantity}

        Returns:
            List of risk violations (empty if no violations)
        """
        pass

    @abstractmethod
    def check_portfolio(
        self,
        current_nav: Decimal,
        positions: dict,
    ) -> List[RiskViolation]:
        """
        Check if current portfolio state violates any risk limits.

        Args:
            current_nav: Current portfolio NAV
            positions: Current positions {symbol: quantity}

        Returns:
            List of risk violations (empty if no violations)
        """
        pass

    @abstractmethod
    def update_state(
        self,
        current_nav: Decimal,
        timestamp: datetime = None,
    ) -> None:
        """
        Update risk manager state (peak NAV, daily start NAV, trade count).

        Args:
            current_nav: Current portfolio NAV
            timestamp: Current timestamp
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset risk manager state (peak NAV, daily counters)."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """
        Get current risk manager state for logging/debugging.

        Returns:
            Dictionary with risk manager state
        """
        pass
```

---

## Concrete Implementation

### StandardRiskManager

```python
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict
import structlog

logger = structlog.get_logger(__name__)


class StandardRiskManager(RiskManager):
    """
    Standard risk manager implementation.

    Enforces all risk limits defined in RiskLimits configuration.
    """

    def __init__(self, limits: RiskLimits):
        """
        Initialize risk manager.

        Args:
            limits: Risk limit configuration
        """
        self.limits = limits

        # State tracking
        self.peak_nav: Decimal = Decimal(0)
        self.daily_start_nav: Decimal = Decimal(0)
        self.last_update_date: datetime | None = None
        self.trades_today: int = 0
        self.trades_this_week: List[datetime] = []
        self.day_trades: List[datetime] = []  # Last 5 business days

    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        current_nav: Decimal,
        current_positions: Dict[str, int],
    ) -> List[RiskViolation]:
        """Check order against risk limits."""
        violations = []

        # 1. Check order size limit
        if self.limits.max_order_size is not None:
            if quantity > self.limits.max_order_size:
                violations.append(
                    RiskViolation(
                        violation_type=RiskViolationType.ORDER_SIZE,
                        current_value=Decimal(quantity),
                        limit_value=Decimal(self.limits.max_order_size),
                        message=f"Order size {quantity} exceeds limit {self.limits.max_order_size}",
                        severity="error",
                    )
                )

        # 2. Check position size limit (after order execution)
        if self.limits.max_position_size_pct is not None:
            # Calculate new position after order
            current_quantity = current_positions.get(symbol, 0)
            if side == "buy":
                new_quantity = current_quantity + quantity
            else:
                new_quantity = current_quantity - quantity

            position_value = abs(new_quantity) * price
            position_size_pct = (position_value / current_nav) if current_nav > 0 else Decimal(0)

            if position_size_pct > self.limits.max_position_size_pct:
                violations.append(
                    RiskViolation(
                        violation_type=RiskViolationType.POSITION_SIZE,
                        current_value=position_size_pct,
                        limit_value=self.limits.max_position_size_pct,
                        message=f"Position size {position_size_pct:.2%} exceeds limit {self.limits.max_position_size_pct:.2%}",
                        severity="error",
                    )
                )

        # 3. Check leverage limit
        if self.limits.max_leverage is not None:
            # Calculate total position value after order
            total_position_value = sum(
                abs(qty) * price for qty in current_positions.values()
            )

            # Add new order
            if side == "buy":
                total_position_value += quantity * price

            leverage = (total_position_value / current_nav) if current_nav > 0 else Decimal(0)

            if leverage > self.limits.max_leverage:
                violations.append(
                    RiskViolation(
                        violation_type=RiskViolationType.LEVERAGE,
                        current_value=leverage,
                        limit_value=self.limits.max_leverage,
                        message=f"Leverage {leverage:.2f}x exceeds limit {self.limits.max_leverage:.2f}x",
                        severity="error",
                    )
                )

        # 4. Check trading frequency limit
        if self.limits.max_trades_per_day is not None:
            if self.trades_today >= self.limits.max_trades_per_day:
                violations.append(
                    RiskViolation(
                        violation_type=RiskViolationType.TRADING_FREQUENCY,
                        current_value=Decimal(self.trades_today),
                        limit_value=Decimal(self.limits.max_trades_per_day),
                        message=f"Daily trade limit {self.limits.max_trades_per_day} reached",
                        severity="warning",
                    )
                )

        # 5. Check pattern day trader limit (if enabled)
        if self.limits.pdt_protection:
            # Check if this would be a day trade
            # (Simplified: assume day trade if holding position from earlier today)
            recent_day_trades = len([
                dt for dt in self.day_trades
                if dt > datetime.now() - timedelta(days=5)
            ])

            if recent_day_trades >= self.limits.pdt_threshold:
                violations.append(
                    RiskViolation(
                        violation_type=RiskViolationType.PATTERN_DAY_TRADER,
                        current_value=Decimal(recent_day_trades),
                        limit_value=Decimal(self.limits.pdt_threshold),
                        message=f"Pattern day trader threshold {self.limits.pdt_threshold} would be exceeded",
                        severity="critical",
                    )
                )

        return violations

    def check_portfolio(
        self,
        current_nav: Decimal,
        positions: Dict[str, int],
    ) -> List[RiskViolation]:
        """Check portfolio state against risk limits."""
        violations = []

        # 1. Check maximum drawdown
        if self.limits.max_drawdown_pct is not None:
            # Update peak NAV
            if current_nav > self.peak_nav:
                self.peak_nav = current_nav

            # Calculate drawdown
            if self.peak_nav > 0:
                drawdown = (self.peak_nav - current_nav) / self.peak_nav

                if drawdown > self.limits.max_drawdown_pct:
                    violations.append(
                        RiskViolation(
                            violation_type=RiskViolationType.MAX_DRAWDOWN,
                            current_value=drawdown,
                            limit_value=self.limits.max_drawdown_pct,
                            message=f"Drawdown {drawdown:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}",
                            severity="critical",
                        )
                    )

        # 2. Check daily loss limit
        if self.limits.max_daily_loss_pct is not None:
            if self.daily_start_nav > 0:
                daily_loss = (self.daily_start_nav - current_nav) / self.daily_start_nav

                if daily_loss > self.limits.max_daily_loss_pct:
                    violations.append(
                        RiskViolation(
                            violation_type=RiskViolationType.DAILY_LOSS,
                            current_value=daily_loss,
                            limit_value=self.limits.max_daily_loss_pct,
                            message=f"Daily loss {daily_loss:.2%} exceeds limit {self.limits.max_daily_loss_pct:.2%}",
                            severity="critical",
                        )
                    )

        return violations

    def update_state(
        self,
        current_nav: Decimal,
        timestamp: datetime = None,
    ) -> None:
        """Update risk manager state."""
        timestamp = timestamp or datetime.now()

        # Update peak NAV
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav
            logger.debug("new_peak_nav", peak_nav=float(self.peak_nav))

        # Reset daily counters if new trading day
        if self.last_update_date is None or timestamp.date() != self.last_update_date.date():
            self.daily_start_nav = current_nav
            self.trades_today = 0
            logger.info("new_trading_day",
                        date=timestamp.date(),
                        daily_start_nav=float(self.daily_start_nav))

        self.last_update_date = timestamp

    def reset(self) -> None:
        """Reset risk manager state."""
        self.peak_nav = Decimal(0)
        self.daily_start_nav = Decimal(0)
        self.last_update_date = None
        self.trades_today = 0
        self.trades_this_week = []
        self.day_trades = []
        logger.info("risk_manager_reset")

    def get_state(self) -> dict:
        """Get current risk manager state."""
        return {
            "peak_nav": float(self.peak_nav),
            "daily_start_nav": float(self.daily_start_nav),
            "trades_today": self.trades_today,
            "trades_this_week": len(self.trades_this_week),
            "day_trades_last_5_days": len([
                dt for dt in self.day_trades
                if dt > datetime.now() - timedelta(days=5)
            ]),
        }
```

---

## Integration with Trading System

### Environment Integration

```python
from finrl.applications.spy_rl_trading.risk_limits import StandardRiskManager, RiskLimits

class SPYTradingEnvironment:
    """Trading environment with risk management."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        risk_limits: RiskLimits = None,
    ):
        self.data = data
        self.cash = initial_capital
        self.shares = 0

        # Initialize risk manager
        self.risk_manager = StandardRiskManager(
            limits=risk_limits or RiskLimits()
        )
        self.risk_manager.update_state(Decimal(initial_capital))

    def step(self, action: int):
        """Execute action with risk limit checks."""
        current_price = Decimal(str(self.data.iloc[self.current_step]["close"]))
        current_nav = Decimal(str(self.cash + self.shares * float(current_price)))

        # Update risk manager state
        self.risk_manager.update_state(current_nav)

        # Check portfolio risk limits
        portfolio_violations = self.risk_manager.check_portfolio(
            current_nav=current_nav,
            positions={"SPY": self.shares},
        )

        if any(v.severity == "critical" for v in portfolio_violations):
            # Critical violation: halt trading
            logger.critical("critical_risk_violation",
                            violations=[str(v) for v in portfolio_violations])
            return self._get_state(), 0.0, True, False, {"violations": portfolio_violations}

        # Execute action if no critical violations
        if action == 0:  # Buy
            # Check order risk limits
            order_violations = self.risk_manager.check_order(
                symbol="SPY",
                side="buy",
                quantity=1,
                price=current_price,
                current_nav=current_nav,
                current_positions={"SPY": self.shares},
            )

            if order_violations:
                logger.warning("order_rejected_risk_violation",
                               violations=[str(v) for v in order_violations])
                # Reject order
            else:
                # Execute order
                if self.cash >= float(current_price):
                    self.cash -= float(current_price)
                    self.shares += 1
                    self.risk_manager.trades_today += 1

        elif action == 2:  # Sell
            if self.shares > 0:
                # Check order risk limits
                order_violations = self.risk_manager.check_order(
                    symbol="SPY",
                    side="sell",
                    quantity=1,
                    price=current_price,
                    current_nav=current_nav,
                    current_positions={"SPY": self.shares},
                )

                if order_violations:
                    logger.warning("order_rejected_risk_violation",
                                   violations=[str(v) for v in order_violations])
                else:
                    # Execute order
                    self.cash += float(current_price)
                    self.shares -= 1
                    self.risk_manager.trades_today += 1

        # Calculate reward
        # ...
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from decimal import Decimal

def test_max_drawdown_violation():
    """Test maximum drawdown violation."""
    limits = RiskLimits(max_drawdown_pct=Decimal("0.20"))
    manager = StandardRiskManager(limits)

    # Set peak NAV
    manager.update_state(Decimal("100000"))

    # Check at 25% drawdown (violation)
    violations = manager.check_portfolio(
        current_nav=Decimal("75000"),
        positions={"SPY": 100},
    )

    assert len(violations) == 1
    assert violations[0].violation_type == RiskViolationType.MAX_DRAWDOWN
    assert violations[0].severity == "critical"


def test_daily_loss_violation():
    """Test daily loss violation."""
    limits = RiskLimits(max_daily_loss_pct=Decimal("0.05"))
    manager = StandardRiskManager(limits)

    # Set daily start NAV
    manager.update_state(Decimal("100000"))

    # Check at 6% loss (violation)
    violations = manager.check_portfolio(
        current_nav=Decimal("94000"),
        positions={"SPY": 100},
    )

    assert len(violations) == 1
    assert violations[0].violation_type == RiskViolationType.DAILY_LOSS
    assert violations[0].severity == "critical"


def test_position_size_violation():
    """Test position size violation."""
    limits = RiskLimits(max_position_size_pct=Decimal("0.50"))  # 50% max
    manager = StandardRiskManager(limits)

    # Check order that would result in 60% position (violation)
    violations = manager.check_order(
        symbol="SPY",
        side="buy",
        quantity=100,
        price=Decimal("600.0"),  # $60K position
        current_nav=Decimal("100000"),
        current_positions={"SPY": 0},
    )

    assert len(violations) == 1
    assert violations[0].violation_type == RiskViolationType.POSITION_SIZE
```

---

## Appendix

### Risk Limit Recommendations

| Account Type | Max Drawdown | Daily Loss | Position Size | Leverage |
|-------------|--------------|------------|---------------|----------|
| **Conservative** | 10% | 2% | 20% | 1.0x |
| **Moderate** | 15% | 3% | 50% | 1.0x |
| **Aggressive** | 20% | 5% | 100% | 1.5x |
| **High Risk** | 30% | 10% | 100% | 2.0x |

### References

- [FINRA Pattern Day Trader Rules](https://www.finra.org/investors/learn-to-invest/advanced-investing/day-trading-margin-requirements-know-rules)
- [Risk Management Best Practices (CFA Institute)](https://www.cfainstitute.org/en/research/foundation/2011/risk-management)
- [Kelly Criterion for Position Sizing](https://en.wikipedia.org/wiki/Kelly_criterion)

---

**Document Revision History**:
- 2025-10-30: Initial version defining risk limits interface contract
