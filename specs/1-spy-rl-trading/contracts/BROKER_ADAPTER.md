# Broker Adapter Interface Contract

**Feature**: SPY RL Trading System - Broker Abstraction
**Phase**: Phase 12 (Finance-Specific Hardening)
**Status**: Planning
**Last Updated**: 2025-10-30

---

## Overview

This document defines the broker adapter interface contract for the SPY RL Trading System. The broker adapter abstraction enables multi-broker support (Alpaca, Interactive Brokers, etc.) and separates trading modes (backtest, paper, live) through a unified interface.

---

## Design Principles

**1. Abstraction**: Broker-specific implementation details hidden behind common interface
**2. Extensibility**: Easy to add new broker implementations
**3. Mode Separation**: Clear distinction between backtest/paper/live trading
**4. Type Safety**: Strong typing with dataclasses and type hints
**5. Error Handling**: Explicit error handling with typed exceptions

---

## Core Entities

### Order

```python
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime
from decimal import Decimal

@dataclass
class Order:
    """
    Order representation for broker execution.

    Attributes:
        symbol: Trading symbol (e.g., "SPY")
        quantity: Number of shares (positive for buy/sell)
        side: Order side ("buy" or "sell")
        order_type: Order type ("market", "limit", "stop", "stop_limit")
        limit_price: Limit price for limit orders (required if order_type="limit")
        stop_price: Stop price for stop orders (required if order_type="stop")
        time_in_force: Time in force ("day", "gtc", "ioc", "fok")
        client_order_id: Client-specified order ID (optional)
    """
    symbol: str
    quantity: int
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    client_order_id: Optional[str] = None

    def __post_init__(self):
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price required for limit orders")

        if self.order_type in ["stop", "stop_limit"] and self.stop_price is None:
            raise ValueError("stop_price required for stop orders")
```

### OrderStatus

```python
from enum import Enum

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"  # Order submitted but not yet accepted
    ACCEPTED = "accepted"  # Order accepted by broker
    FILLED = "filled"  # Order fully filled
    PARTIALLY_FILLED = "partially_filled"  # Order partially filled
    CANCELED = "canceled"  # Order canceled
    REJECTED = "rejected"  # Order rejected by broker
    EXPIRED = "expired"  # Order expired (time_in_force)
```

### OrderExecution

```python
@dataclass
class OrderExecution:
    """
    Order execution details.

    Attributes:
        order_id: Broker-assigned order ID
        client_order_id: Client-specified order ID
        symbol: Trading symbol
        side: Order side
        quantity: Ordered quantity
        filled_quantity: Filled quantity
        status: Current order status
        filled_avg_price: Average fill price (None if not filled)
        submitted_at: Order submission timestamp
        filled_at: Order fill timestamp (None if not filled)
        commission: Commission charged (None if not filled)
        error_message: Error message if rejected
    """
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    status: OrderStatus
    filled_avg_price: Optional[Decimal] = None
    submitted_at: datetime = None
    filled_at: Optional[datetime] = None
    commission: Optional[Decimal] = None
    error_message: Optional[str] = None
```

### Position

```python
@dataclass
class Position:
    """
    Current position representation.

    Attributes:
        symbol: Trading symbol
        quantity: Number of shares held (positive=long, negative=short)
        avg_entry_price: Average entry price
        market_value: Current market value
        unrealized_pnl: Unrealized profit/loss
        cost_basis: Total cost basis
    """
    symbol: str
    quantity: int
    avg_entry_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    cost_basis: Decimal
```

### Account

```python
@dataclass
class Account:
    """
    Account information.

    Attributes:
        account_id: Broker account ID
        buying_power: Available buying power
        cash: Available cash
        portfolio_value: Total portfolio value (cash + positions)
        equity: Account equity
        initial_margin: Initial margin requirement
        maintenance_margin: Maintenance margin requirement
        pattern_day_trader: Whether account is flagged as PDT
    """
    account_id: str
    buying_power: Decimal
    cash: Decimal
    portfolio_value: Decimal
    equity: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    pattern_day_trader: bool
```

---

## Interface Definition

### BrokerAdapter (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BrokerAdapter(ABC):
    """
    Abstract broker adapter interface.

    All broker implementations must implement this interface to ensure
    consistent behavior across backtesting, paper trading, and live trading.
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to broker.

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If credentials invalid
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from broker and cleanup resources.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connection is active.

        Returns:
            True if connected, False otherwise
        """
        pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Order Management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """
        Place order with broker.

        Args:
            order: Order to place

        Returns:
            order_id: Broker-assigned order ID

        Raises:
            OrderRejectedError: If order rejected by broker
            InsufficientFundsError: If insufficient buying power
            BrokerAPIError: If broker API error occurs
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.

        Args:
            order_id: Broker-assigned order ID

        Returns:
            True if cancellation successful, False otherwise

        Raises:
            OrderNotFoundError: If order_id invalid
            BrokerAPIError: If broker API error occurs
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderExecution:
        """
        Get order execution status.

        Args:
            order_id: Broker-assigned order ID

        Returns:
            OrderExecution with current status

        Raises:
            OrderNotFoundError: If order_id invalid
            BrokerAPIError: If broker API error occurs
        """
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderExecution]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders

        Raises:
            BrokerAPIError: If broker API error occurs
        """
        pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Account & Position Management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    def get_account(self) -> Account:
        """
        Get account information.

        Returns:
            Account with current balances and equity

        Raises:
            BrokerAPIError: If broker API error occurs
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position if held, None if no position

        Raises:
            BrokerAPIError: If broker API error occurs
        """
        pass

    @abstractmethod
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping symbol to Position

        Raises:
            BrokerAPIError: If broker API error occurs
        """
        pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Market Data
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @abstractmethod
    def get_current_price(self, symbol: str) -> Decimal:
        """
        Get current market price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price

        Raises:
            SymbolNotFoundError: If symbol invalid
            BrokerAPIError: If broker API error occurs
        """
        pass

    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market open, False otherwise
        """
        pass
```

---

## Concrete Implementations

### 1. BacktestBrokerAdapter

**Purpose**: Simulated execution for backtesting

**Features**:
- Instant fills at market price (no slippage)
- Configurable transaction costs
- Historical price lookup
- No network I/O (fast)

**Usage**:
```python
from finrl.applications.spy_rl_trading.broker import BacktestBrokerAdapter

adapter = BacktestBrokerAdapter(
    initial_cash=100_000.0,
    price_data=historical_prices,  # DataFrame with OHLCV
    transaction_cost_bps=2,
)

# Place order
order = Order(symbol="SPY", quantity=10, side="buy")
order_id = adapter.place_order(order)

# Check status
status = adapter.get_order_status(order_id)
assert status.status == OrderStatus.FILLED
```

---

### 2. AlpacaBrokerAdapter

**Purpose**: Alpaca API integration (paper/live)

**Features**:
- Real-time order submission
- WebSocket for order updates
- Market data integration
- Paper trading support

**Usage**:
```python
from finrl.applications.spy_rl_trading.broker import AlpacaBrokerAdapter

adapter = AlpacaBrokerAdapter(
    api_key="your_paper_key",
    api_secret="your_paper_secret",
    base_url="https://paper-api.alpaca.markets",  # Paper trading
)

adapter.connect()

# Place market order
order = Order(symbol="SPY", quantity=10, side="buy", order_type="market")
order_id = adapter.place_order(order)

# Wait for fill
import time
while True:
    status = adapter.get_order_status(order_id)
    if status.status == OrderStatus.FILLED:
        break
    time.sleep(1)

adapter.disconnect()
```

---

### 3. DryRunBrokerAdapter

**Purpose**: Logs orders without execution (testing)

**Features**:
- No actual order placement
- Structured logging of all operations
- Useful for validation and debugging

**Usage**:
```python
from finrl.applications.spy_rl_trading.broker import DryRunBrokerAdapter

adapter = DryRunBrokerAdapter()

# Place order (logged but not executed)
order = Order(symbol="SPY", quantity=10, side="buy")
order_id = adapter.place_order(order)
# Logs: {"event": "order_placed_dry_run", "symbol": "SPY", "quantity": 10, ...}
```

---

## Custom Exceptions

```python
class BrokerError(Exception):
    """Base exception for broker errors."""
    pass

class ConnectionError(BrokerError):
    """Connection to broker failed."""
    pass

class AuthenticationError(BrokerError):
    """Authentication failed (invalid credentials)."""
    pass

class OrderRejectedError(BrokerError):
    """Order rejected by broker."""
    pass

class InsufficientFundsError(BrokerError):
    """Insufficient buying power for order."""
    pass

class OrderNotFoundError(BrokerError):
    """Order ID not found."""
    pass

class SymbolNotFoundError(BrokerError):
    """Trading symbol not found."""
    pass

class BrokerAPIError(BrokerError):
    """Broker API error occurred."""
    pass
```

---

## Usage Example

### Training Pipeline with Broker Adapter

```python
from finrl.applications.spy_rl_trading.broker import BacktestBrokerAdapter
from finrl.applications.spy_rl_trading.environment import SPYTradingEnvironment

# Load historical data
data = load_spy_data("2020-01-01", "2024-12-31")

# Create backtest adapter
adapter = BacktestBrokerAdapter(
    initial_cash=100_000.0,
    price_data=data,
    transaction_cost_bps=2,
)

# Create environment with broker adapter
env = SPYTradingEnvironment(
    data=data,
    broker_adapter=adapter,
)

# Train agent
agent = PPOAgent(env)
agent.train(total_timesteps=500_000)
```

### Paper Trading with Broker Adapter

```python
from finrl.applications.spy_rl_trading.broker import AlpacaBrokerAdapter
from finrl.applications.spy_rl_trading.settings import settings

# Create Alpaca adapter (paper trading)
adapter = AlpacaBrokerAdapter(
    api_key=settings.alpaca_api_key,
    api_secret=settings.alpaca_api_secret,
    base_url="https://paper-api.alpaca.markets",
)

adapter.connect()

# Load trained model
model = load_model("trained_models/ppo.zip")

# Execute trading loop
while adapter.is_market_open():
    # Get current state
    account = adapter.get_account()
    position = adapter.get_position("SPY")

    # Get agent action
    action = model.predict(state)

    # Place order based on action
    if action == 0:  # Buy
        order = Order(symbol="SPY", quantity=1, side="buy")
        adapter.place_order(order)
    elif action == 2:  # Sell
        if position and position.quantity > 0:
            order = Order(symbol="SPY", quantity=1, side="sell")
            adapter.place_order(order)

    time.sleep(60)  # Wait 1 minute

adapter.disconnect()
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from finrl.applications.spy_rl_trading.broker import BacktestBrokerAdapter

def test_place_order():
    """Test order placement."""
    adapter = BacktestBrokerAdapter(initial_cash=100_000.0)
    order = Order(symbol="SPY", quantity=10, side="buy")
    order_id = adapter.place_order(order)
    assert order_id is not None

def test_insufficient_funds():
    """Test insufficient funds error."""
    adapter = BacktestBrokerAdapter(initial_cash=100.0)
    order = Order(symbol="SPY", quantity=1000, side="buy")  # Too large
    with pytest.raises(InsufficientFundsError):
        adapter.place_order(order)

def test_order_cancellation():
    """Test order cancellation."""
    adapter = BacktestBrokerAdapter(initial_cash=100_000.0)
    order = Order(symbol="SPY", quantity=10, side="buy", order_type="limit", limit_price=Decimal("400.0"))
    order_id = adapter.place_order(order)
    success = adapter.cancel_order(order_id)
    assert success
```

---

## Appendix

### Broker Comparison

| Feature | BacktestBrokerAdapter | AlpacaBrokerAdapter | DryRunBrokerAdapter |
|---------|----------------------|---------------------|---------------------|
| **Execution** | Simulated | Real | Logged only |
| **Network I/O** | No | Yes | No |
| **Cost** | Free | Free (paper) | Free |
| **Speed** | Fast | Slow | Fast |
| **Use Case** | Backtesting | Paper/Live | Testing |

### Future Enhancements

- **Interactive Brokers Adapter**: Support for IBKR API
- **Binance Adapter**: Cryptocurrency trading support
- **Order Batching**: Submit multiple orders in single API call
- **WebSocket Streaming**: Real-time order updates
- **Order Routing**: Smart order routing across brokers

---

**Document Revision History**:
- 2025-10-30: Initial version defining broker adapter interface contract
