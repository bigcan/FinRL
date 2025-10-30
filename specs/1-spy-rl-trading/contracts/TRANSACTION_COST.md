# Transaction Cost Model Interface Contract

**Feature**: SPY RL Trading System - Transaction Cost Modeling
**Phase**: Phase 12 (Finance-Specific Hardening)
**Status**: Planning
**Last Updated**: 2025-10-30

---

## Overview

This document defines the transaction cost model interface for realistic backtesting and live trading simulation. Accurate transaction cost modeling is critical for evaluating strategy profitability and preventing overfitting to frictionless market assumptions.

---

## Transaction Cost Components

### 1. Commission

**Definition**: Fixed fee charged by broker per trade or per share

**Typical Values**:
- **Zero-Commission Brokers** (Robinhood, Webull): $0
- **Traditional Brokers** (TD Ameritrade): $0 (since 2019)
- **Interactive Brokers**: $0.0035/share (min $0.35, max 1% of trade value)
- **Institutional**: Negotiated (typically $0.001-$0.005/share)

**Formula**:
```
commission = commission_per_share * shares + commission_flat_fee
```

---

### 2. Bid-Ask Spread

**Definition**: Difference between best bid and best ask price

**Typical Values**:
- **SPY**: 0.01-0.02% (0.1-0.2 bps) during market hours
- **Less liquid stocks**: 0.1-0.5% (10-50 bps)
- **Illiquid stocks**: 1-5% (100-500 bps)

**Impact**:
- **Buy**: Pay ask price (higher)
- **Sell**: Receive bid price (lower)
- **Round-trip cost**: Full spread

**Formula**:
```
spread_cost = (price * spread_bps / 10000) / 2  # Half-spread per trade
```

---

### 3. Market Impact

**Definition**: Price movement caused by order itself

**Factors**:
- **Order Size**: Larger orders → higher impact
- **Liquidity**: Less liquid → higher impact
- **Urgency**: Aggressive orders → higher impact

**Typical Values**:
- **SPY** (high liquidity): Negligible for retail orders (<$1M)
- **Mid-cap stocks**: 0.1-0.5% for $100K orders
- **Small-cap stocks**: 0.5-2% for $100K orders

**Model** (Square-root law):
```
market_impact = price * alpha * sqrt(order_size / average_daily_volume)

where:
  alpha = market impact coefficient (typically 0.1-0.5 for stocks)
  order_size = number of shares
  average_daily_volume = average daily trading volume
```

---

### 4. SEC Fees

**Definition**: Regulatory fee charged by SEC on sales of US equities

**Current Rate**: $27.80 per million dollars of sales (0.00278%)

**Formula**:
```
sec_fee = (price * shares * 27.80 / 1_000_000) if side == "sell" else 0
```

**Effective Date**: Updated periodically by SEC (check latest rate)

---

### 5. Exchange Fees

**Definition**: Fees charged by exchanges (NYSE, NASDAQ)

**Typical Values**:
- **NYSE**: $0.0025/share for tape A/B stocks (capped at $75)
- **NASDAQ**: $0.003/share for tape C stocks (capped at $0.75% of trade value)
- **ECN Rebates**: -$0.001 to +$0.003/share (maker/taker fees)

**Note**: Often absorbed by broker, not passed to retail customers

---

### 6. Slippage

**Definition**: Difference between expected price and actual execution price

**Causes**:
- Market movement between decision and execution
- Order size exceeding best bid/ask size
- Volatility

**Typical Values**:
- **Market orders**: 0.01-0.1% (1-10 bps)
- **Limit orders**: Minimal (but risk of no fill)
- **High volatility**: 0.1-0.5% (10-50 bps)

**Model** (Volume-based):
```
slippage = price * slippage_bps * sqrt(order_size / average_volume) / 10000

where:
  slippage_bps = base slippage in basis points
  order_size = number of shares
  average_volume = recent average trading volume
```

---

## Interface Definition

### TransactionCostModel (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

@dataclass
class TransactionCostBreakdown:
    """
    Detailed breakdown of transaction costs.

    Attributes:
        commission: Broker commission
        spread: Bid-ask spread cost
        market_impact: Price impact from order size
        sec_fee: SEC regulatory fee (sales only)
        exchange_fee: Exchange fee (if applicable)
        slippage: Execution slippage
        total: Sum of all costs
    """
    commission: Decimal
    spread: Decimal
    market_impact: Decimal
    sec_fee: Decimal
    exchange_fee: Decimal
    slippage: Decimal

    @property
    def total(self) -> Decimal:
        """Calculate total transaction cost."""
        return (
            self.commission
            + self.spread
            + self.market_impact
            + self.sec_fee
            + self.exchange_fee
            + self.slippage
        )


class TransactionCostModel(ABC):
    """
    Abstract transaction cost model interface.

    All cost models must implement this interface to ensure consistent
    cost calculation across backtesting and live trading.
    """

    @abstractmethod
    def calculate_cost(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: int,
        price: Decimal,
        timestamp: str = None,
    ) -> TransactionCostBreakdown:
        """
        Calculate transaction cost breakdown for a trade.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            quantity: Number of shares
            price: Execution price
            timestamp: Execution timestamp (optional, for historical data)

        Returns:
            TransactionCostBreakdown with detailed cost components
        """
        pass

    @abstractmethod
    def get_total_cost(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: int,
        price: Decimal,
    ) -> Decimal:
        """
        Calculate total transaction cost (convenience method).

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Number of shares
            price: Execution price

        Returns:
            Total cost as Decimal
        """
        pass

    @abstractmethod
    def get_effective_price(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: int,
        market_price: Decimal,
    ) -> Decimal:
        """
        Calculate effective execution price including costs.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Number of shares
            market_price: Current market price

        Returns:
            Effective price (market_price + costs for buy, market_price - costs for sell)
        """
        pass
```

---

## Concrete Implementations

### 1. SimpleCostModel

**Purpose**: Fixed basis point cost (e.g., 2 bps per trade)

**Usage**: Quick backtesting, simplified cost assumptions

```python
from decimal import Decimal

class SimpleCostModel(TransactionCostModel):
    """
    Simple fixed-cost model.

    Applies a fixed basis point cost to all trades.
    """

    def __init__(self, cost_bps: int = 2):
        """
        Initialize simple cost model.

        Args:
            cost_bps: Cost in basis points (default: 2 = 0.02%)
        """
        self.cost_bps = cost_bps

    def calculate_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        timestamp: str = None,
    ) -> TransactionCostBreakdown:
        """Calculate cost as fixed percentage."""
        notional = price * quantity
        total_cost = notional * Decimal(self.cost_bps) / Decimal(10000)

        return TransactionCostBreakdown(
            commission=Decimal(0),
            spread=total_cost,  # Model as spread cost
            market_impact=Decimal(0),
            sec_fee=Decimal(0),
            exchange_fee=Decimal(0),
            slippage=Decimal(0),
        )

    def get_total_cost(self, symbol: str, side: str, quantity: int, price: Decimal) -> Decimal:
        """Get total cost."""
        return self.calculate_cost(symbol, side, quantity, price).total

    def get_effective_price(
        self, symbol: str, side: str, quantity: int, market_price: Decimal
    ) -> Decimal:
        """Calculate effective price."""
        cost_per_share = market_price * Decimal(self.cost_bps) / Decimal(10000)
        if side == "buy":
            return market_price + cost_per_share
        else:
            return market_price - cost_per_share
```

**Example**:
```python
model = SimpleCostModel(cost_bps=2)
cost = model.calculate_cost(symbol="SPY", side="buy", quantity=100, price=Decimal("450.0"))
# cost.total = $0.09 (2 bps of $45,000)
```

---

### 2. RealisticCostModel

**Purpose**: Multi-component cost model (commission + spread + SEC fee + slippage)

**Usage**: Accurate backtesting, production simulations

```python
from decimal import Decimal
import pandas as pd

class RealisticCostModel(TransactionCostModel):
    """
    Realistic cost model with multiple components.

    Includes commission, spread, SEC fee, and slippage.
    """

    def __init__(
        self,
        commission_per_share: Decimal = Decimal("0.0"),
        commission_flat: Decimal = Decimal("0.0"),
        spread_bps: Decimal = Decimal("2.0"),
        sec_fee_per_dollar: Decimal = Decimal("0.0000278"),
        slippage_bps: Decimal = Decimal("1.0"),
        volume_data: pd.DataFrame = None,
    ):
        """
        Initialize realistic cost model.

        Args:
            commission_per_share: Commission per share (default: $0 for zero-commission brokers)
            commission_flat: Flat commission per trade (default: $0)
            spread_bps: Bid-ask spread in basis points (default: 2 = 0.02%)
            sec_fee_per_dollar: SEC fee rate per dollar (default: 0.00278%)
            slippage_bps: Base slippage in basis points (default: 1 = 0.01%)
            volume_data: DataFrame with volume data for slippage calculation
        """
        self.commission_per_share = commission_per_share
        self.commission_flat = commission_flat
        self.spread_bps = spread_bps
        self.sec_fee_per_dollar = sec_fee_per_dollar
        self.slippage_bps = slippage_bps
        self.volume_data = volume_data

    def calculate_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        timestamp: str = None,
    ) -> TransactionCostBreakdown:
        """Calculate cost with multiple components."""
        notional = price * quantity

        # 1. Commission
        commission = self.commission_per_share * quantity + self.commission_flat

        # 2. Spread (half-spread per trade)
        spread = notional * self.spread_bps / Decimal(20000)  # Half-spread

        # 3. Market impact (negligible for SPY retail orders)
        market_impact = Decimal(0)

        # 4. SEC fee (only on sells)
        sec_fee = notional * self.sec_fee_per_dollar if side == "sell" else Decimal(0)

        # 5. Exchange fee (assume absorbed by broker)
        exchange_fee = Decimal(0)

        # 6. Slippage (volume-based if data available)
        if self.volume_data is not None and timestamp is not None:
            # Volume-based slippage model
            try:
                volume = self.volume_data.loc[timestamp, "volume"]
                participation_rate = Decimal(quantity) / Decimal(volume)
                slippage = notional * self.slippage_bps * participation_rate.sqrt() / Decimal(10000)
            except (KeyError, AttributeError):
                # Fallback to fixed slippage
                slippage = notional * self.slippage_bps / Decimal(10000)
        else:
            # Fixed slippage
            slippage = notional * self.slippage_bps / Decimal(10000)

        return TransactionCostBreakdown(
            commission=commission,
            spread=spread,
            market_impact=market_impact,
            sec_fee=sec_fee,
            exchange_fee=exchange_fee,
            slippage=slippage,
        )

    def get_total_cost(self, symbol: str, side: str, quantity: int, price: Decimal) -> Decimal:
        """Get total cost."""
        return self.calculate_cost(symbol, side, quantity, price).total

    def get_effective_price(
        self, symbol: str, side: str, quantity: int, market_price: Decimal
    ) -> Decimal:
        """Calculate effective price."""
        cost = self.calculate_cost(symbol, side, quantity, market_price)
        cost_per_share = cost.total / quantity

        if side == "buy":
            return market_price + cost_per_share
        else:
            return market_price - cost_per_share
```

**Example**:
```python
model = RealisticCostModel(
    commission_per_share=Decimal("0.0"),
    spread_bps=Decimal("2.0"),
    sec_fee_per_dollar=Decimal("0.0000278"),
    slippage_bps=Decimal("1.0"),
)

cost = model.calculate_cost(symbol="SPY", side="sell", quantity=100, price=Decimal("450.0"))
# cost.commission = $0.00
# cost.spread = $0.045 (2 bps half-spread)
# cost.sec_fee = $1.25 (0.00278% of $45,000)
# cost.slippage = $0.045 (1 bps)
# cost.total = $1.34
```

---

### 3. InteractiveBrokersCostModel

**Purpose**: Interactive Brokers specific cost structure

**Usage**: Accurate modeling for IB customers

```python
class InteractiveBrokersCostModel(TransactionCostModel):
    """
    Interactive Brokers cost model.

    Tiered pricing structure:
    - $0.0035/share (min $0.35, max 1% of trade value)
    """

    def calculate_cost(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        timestamp: str = None,
    ) -> TransactionCostBreakdown:
        """Calculate IB-specific costs."""
        notional = price * quantity

        # IB commission
        commission_per_share = Decimal("0.0035")
        commission_min = Decimal("0.35")
        commission_max = notional * Decimal("0.01")  # 1% of trade value

        commission = max(
            commission_min,
            min(commission_per_share * quantity, commission_max),
        )

        # Spread + SEC fee + slippage (same as realistic model)
        spread = notional * Decimal("2.0") / Decimal(20000)
        sec_fee = notional * Decimal("0.0000278") if side == "sell" else Decimal(0)
        slippage = notional * Decimal("1.0") / Decimal(10000)

        return TransactionCostBreakdown(
            commission=commission,
            spread=spread,
            market_impact=Decimal(0),
            sec_fee=sec_fee,
            exchange_fee=Decimal(0),
            slippage=slippage,
        )

    def get_total_cost(self, symbol: str, side: str, quantity: int, price: Decimal) -> Decimal:
        return self.calculate_cost(symbol, side, quantity, price).total

    def get_effective_price(
        self, symbol: str, side: str, quantity: int, market_price: Decimal
    ) -> Decimal:
        cost = self.calculate_cost(symbol, side, quantity, market_price)
        cost_per_share = cost.total / quantity
        return market_price + cost_per_share if side == "buy" else market_price - cost_per_share
```

---

## Integration with Backtesting

### Environment Integration

```python
from finrl.applications.spy_rl_trading.transaction_costs import RealisticCostModel

class SPYTradingEnvironment:
    """Trading environment with transaction costs."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_model: TransactionCostModel = None,
    ):
        self.data = data
        self.cash = initial_capital
        self.shares = 0
        self.cost_model = cost_model or SimpleCostModel(cost_bps=2)

    def step(self, action: int):
        """Execute action with transaction costs."""
        current_price = self.data.iloc[self.current_step]["close"]

        if action == 0:  # Buy
            # Calculate effective buy price
            effective_price = self.cost_model.get_effective_price(
                symbol="SPY",
                side="buy",
                quantity=1,
                market_price=current_price,
            )

            if self.cash >= effective_price:
                self.cash -= effective_price
                self.shares += 1

        elif action == 2:  # Sell
            if self.shares > 0:
                # Calculate effective sell price
                effective_price = self.cost_model.get_effective_price(
                    symbol="SPY",
                    side="sell",
                    quantity=1,
                    market_price=current_price,
                )

                self.cash += effective_price
                self.shares -= 1

        # Calculate reward (log return)
        # ...
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from decimal import Decimal

def test_simple_cost_model():
    """Test simple cost model."""
    model = SimpleCostModel(cost_bps=2)
    cost = model.calculate_cost("SPY", "buy", 100, Decimal("450.0"))
    assert cost.total == Decimal("0.09")  # 2 bps of $45,000

def test_realistic_cost_model_buy():
    """Test realistic cost model buy."""
    model = RealisticCostModel(spread_bps=Decimal("2.0"), slippage_bps=Decimal("1.0"))
    cost = model.calculate_cost("SPY", "buy", 100, Decimal("450.0"))
    # Spread: $0.045, SEC fee: $0, slippage: $0.045
    assert cost.total == pytest.approx(Decimal("0.09"), rel=0.01)

def test_realistic_cost_model_sell():
    """Test realistic cost model sell."""
    model = RealisticCostModel(
        spread_bps=Decimal("2.0"),
        sec_fee_per_dollar=Decimal("0.0000278"),
        slippage_bps=Decimal("1.0"),
    )
    cost = model.calculate_cost("SPY", "sell", 100, Decimal("450.0"))
    # Spread: $0.045, SEC fee: $1.25, slippage: $0.045
    assert cost.total == pytest.approx(Decimal("1.34"), rel=0.01)
```

---

## Appendix

### Cost Model Comparison

| Model | Commission | Spread | SEC Fee | Slippage | Market Impact | Use Case |
|-------|-----------|--------|---------|----------|---------------|----------|
| SimpleCostModel | ❌ | ✅ (fixed) | ❌ | ❌ | ❌ | Quick backtesting |
| RealisticCostModel | ✅ | ✅ | ✅ | ✅ | ❌ | Accurate backtesting |
| InteractiveBrokersCostModel | ✅ (IB-specific) | ✅ | ✅ | ✅ | ❌ | IB customers |

### References

- [Interactive Brokers Commission Schedule](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php)
- [SEC Fee Schedule](https://www.sec.gov/fast-answers/answersfee-assess-htm.html)
- [Market Microstructure (Hasbrouck, 2007)](https://doi.org/10.1093/0195301080.001.0001)

---

**Document Revision History**:
- 2025-10-30: Initial version defining transaction cost model interface
