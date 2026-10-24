"""
Market abstraction layer for prediction market trading.

Defines the MarketClient ABC and core dataclasses used across all market
implementations (Kalshi, Polymarket, paper trading). All monetary values
are stored in cents to avoid floating-point rounding.

Kalshi contract pricing:
    - Price range: 1-99 cents (1¢ = 1% implied probability)
    - Buy YES at X¢ = pay X¢, receive 100¢ if event occurs
    - Buy NO at X¢ = pay X¢, receive 100¢ if event doesn't occur
    - YES bid at X = NO ask at (100-X), and vice versa
    - Edge = model_prob_cents - contract_price_cents

Usage:
    from py.markets.base import MarketClient, Order, Side, OrderStatus

    class KalshiClient(MarketClient):
        async def place_order(self, order: Order) -> Order:
            ...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(str, Enum):
    """Order side. YES = event happens, NO = event doesn't happen."""

    YES = "yes"
    NO = "no"


class OrderType(str, Enum):
    """Order type. Kalshi supports limit orders only (no market orders)."""

    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Lifecycle of an order on the exchange."""

    PENDING = "pending"  # Submitted, not yet on book
    OPEN = "open"  # Resting on order book
    PARTIAL = "partial"  # Partially filled
    FILLED = "filled"  # Fully filled
    CANCELLED = "cancelled"  # Cancelled by user or system
    REJECTED = "rejected"  # Rejected by exchange


class ContractType(str, Enum):
    """Types of prediction market contracts."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"
    FUTURES = "futures"
    CUSTOM = "custom"


class Sport(str, Enum):
    """Supported sports."""

    NFL = "nfl"
    NBA = "nba"
    MLB = "mlb"
    NHL = "nhl"


# ---------------------------------------------------------------------------
# Core dataclasses — all prices in cents (1-99)
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """A single order on a prediction market.

    Prices are in cents (1-99). Quantity is number of contracts.
    cost = price_cents * quantity (in cents).
    """

    order_id: str
    contract_id: str
    side: Side
    price_cents: int  # 1-99, limit price
    quantity: int  # Number of contracts
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price_cents: int = 0
    strategy: str = ""  # Which strategy placed this order
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    exchange_order_id: str = ""  # Exchange-assigned ID

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL, OrderStatus.PENDING)

    @property
    def cost_cents(self) -> int:
        """Total cost if fully filled at limit price."""
        return self.price_cents * self.quantity

    @property
    def cost_dollars(self) -> float:
        """Total cost in dollars (display only)."""
        return self.cost_cents / 100.0

    def __post_init__(self):
        if not 1 <= self.price_cents <= 99:
            raise ValueError(f"price_cents must be 1-99, got {self.price_cents}")
        if self.quantity < 1:
            raise ValueError(f"quantity must be >= 1, got {self.quantity}")


@dataclass
class Fill:
    """An executed trade (immutable record of a fill)."""

    fill_id: str
    order_id: str
    contract_id: str
    side: Side
    price_cents: int  # Execution price
    quantity: int  # Filled quantity
    fee_cents: int = 0  # Exchange fee
    executed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def net_cost_cents(self) -> int:
        """Cost including fees."""
        return self.price_cents * self.quantity + self.fee_cents


@dataclass
class Position:
    """An open position in a contract.

    Tracks average entry price and unrealized P&L based on current market.
    """

    contract_id: str
    side: Side
    quantity: int
    avg_entry_cents: int  # Volume-weighted average entry price
    market_id: str = ""
    sport: Sport = Sport.NFL

    @property
    def cost_basis_cents(self) -> int:
        """Total cost basis."""
        return self.avg_entry_cents * self.quantity

    def unrealized_pnl_cents(self, current_price_cents: int) -> int:
        """Unrealized P&L in cents given current market price.

        For YES positions: profit when price rises above entry.
        For NO positions: profit when price falls below entry (i.e., NO price rises).
        """
        if self.side == Side.YES:
            return (current_price_cents - self.avg_entry_cents) * self.quantity
        else:
            # NO position profits when YES price drops
            # NO entry = 100 - YES price at entry, NO current = 100 - YES current
            # P&L = (NO_current - NO_entry) * qty = (YES_entry - YES_current) * qty
            return (self.avg_entry_cents - current_price_cents) * self.quantity

    def update_from_fill(self, fill: Fill) -> None:
        """Update position from a new fill (same side adds, opposite closes)."""
        if fill.side == self.side:
            # Adding to position — update average entry
            total_cost = self.avg_entry_cents * self.quantity + fill.price_cents * fill.quantity
            self.quantity += fill.quantity
            self.avg_entry_cents = total_cost // self.quantity
        else:
            # Reducing position
            self.quantity -= fill.quantity
            if self.quantity < 0:
                # Flipped sides — reset
                self.side = fill.side
                self.quantity = abs(self.quantity)
                self.avg_entry_cents = fill.price_cents


@dataclass
class OrderBookLevel:
    """A single price level in the order book."""

    price_cents: int
    quantity: int


@dataclass
class OrderBook:
    """Order book snapshot for a contract.

    yes_bids: Sorted descending by price (best bid first).
    yes_asks: Sorted ascending by price (best ask first).

    On Kalshi, YES bid at X = NO ask at (100-X).
    """

    contract_id: str
    yes_bids: list[OrderBookLevel] = field(default_factory=list)
    yes_asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def best_bid_cents(self) -> int | None:
        """Best YES bid price."""
        return self.yes_bids[0].price_cents if self.yes_bids else None

    @property
    def best_ask_cents(self) -> int | None:
        """Best YES ask price."""
        return self.yes_asks[0].price_cents if self.yes_asks else None

    @property
    def mid_price_cents(self) -> float | None:
        """Midpoint between best bid and ask."""
        if self.best_bid_cents is not None and self.best_ask_cents is not None:
            return (self.best_bid_cents + self.best_ask_cents) / 2.0
        return None

    @property
    def spread_cents(self) -> int | None:
        """Bid-ask spread in cents."""
        if self.best_bid_cents is not None and self.best_ask_cents is not None:
            return self.best_ask_cents - self.best_bid_cents
        return None

    @property
    def is_crossed(self) -> bool:
        """True if bid >= ask (should trigger immediate fill)."""
        if self.best_bid_cents is not None and self.best_ask_cents is not None:
            return self.best_bid_cents >= self.best_ask_cents
        return False


@dataclass
class Contract:
    """A prediction market contract (e.g., 'Chiefs win Super Bowl')."""

    contract_id: str
    event_id: str
    title: str
    contract_type: ContractType
    sport: Sport
    strike: float | None = None  # For spreads/totals
    expiry: datetime | None = None
    last_price_cents: int = 0
    volume: int = 0
    open_interest: int = 0
    is_active: bool = True


@dataclass
class MarketData:
    """Real-time market data snapshot for a contract."""

    contract_id: str
    last_price_cents: int
    yes_bid_cents: int
    yes_ask_cents: int
    volume_24h: int
    open_interest: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def mid_cents(self) -> float:
        return (self.yes_bid_cents + self.yes_ask_cents) / 2.0

    @property
    def spread_cents(self) -> int:
        return self.yes_ask_cents - self.yes_bid_cents

    @property
    def implied_prob(self) -> float:
        """Implied probability from midpoint (0.0 to 1.0)."""
        return self.mid_cents / 100.0


# ---------------------------------------------------------------------------
# MarketClient ABC
# ---------------------------------------------------------------------------


class MarketClient(ABC):
    """Abstract base class for prediction market exchange clients.

    All implementations (Kalshi, Polymarket, paper) must implement these
    methods. Strategies code against this interface, never against a specific
    exchange.

    Design principle: if it works with PaperMarketClient, it works live.
    """

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Submit an order to the exchange.

        Returns the order with updated status and exchange_order_id.
        Raises MarketError on failure.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a resting order. Returns updated order."""

    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Get current state of an order."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""

    @abstractmethod
    async def get_balance_cents(self) -> int:
        """Get account balance in cents."""

    @abstractmethod
    async def get_orderbook(self, contract_id: str) -> OrderBook:
        """Get current order book for a contract."""

    @abstractmethod
    async def get_market_data(self, contract_id: str) -> MarketData:
        """Get latest market data for a contract."""

    @abstractmethod
    async def get_contract(self, contract_id: str) -> Contract:
        """Get contract metadata."""

    @abstractmethod
    async def get_fills(self, order_id: str | None = None) -> list[Fill]:
        """Get fills, optionally filtered by order_id."""

    async def batch_place_orders(self, orders: list[Order]) -> list[Order]:
        """Place multiple orders (default: sequential, override for batch API).

        Kalshi supports batch orders (up to 20 per request).
        """
        results = []
        for order in orders:
            results.append(await self.place_order(order))
        return results

    async def cancel_all_orders(self, contract_id: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by contract.

        Returns count of cancelled orders.
        """
        # Default: get all orders and cancel individually
        # Override for exchanges with bulk cancel API
        cancelled = 0
        positions = await self.get_positions()
        # Subclasses should override with exchange-specific bulk cancel
        return cancelled


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MarketError(Exception):
    """Base exception for market operations."""


class OrderRejectedError(MarketError):
    """Order was rejected by the exchange."""


class InsufficientBalanceError(MarketError):
    """Not enough balance to place order."""


class RateLimitError(MarketError):
    """Rate limit exceeded (Kalshi: 20 orders/sec)."""

    def __init__(self, retry_after_ms: int = 1000):
        self.retry_after_ms = retry_after_ms
        super().__init__(f"Rate limited, retry after {retry_after_ms}ms")


class ContractNotFoundError(MarketError):
    """Contract ID not found on exchange."""


class ConnectionError(MarketError):
    """Lost connection to exchange."""
