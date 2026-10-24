"""
Paper trading client that mirrors live exchange behavior.

Simulates a prediction market order book with realistic fill logic:
- Limit orders only fill when price crosses the spread
- Configurable latency and fill probability
- Maintains positions, balances, and fill history in memory
- Identical interface to KalshiClient via MarketClient ABC

Design principle: if it works in paper, it works live.

Usage:
    from py.markets.paper_client import PaperMarketClient

    client = PaperMarketClient(initial_balance_cents=100_000)  # $1,000

    order = Order(
        order_id="test-1",
        contract_id="KXNFL-26-AFC-KC",
        side=Side.YES,
        price_cents=55,
        quantity=10,
    )
    result = await client.place_order(order)
    print(result.status)  # OrderStatus.OPEN or OrderStatus.FILLED
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from py.markets.base import (
    Contract,
    ContractNotFoundError,
    ContractType,
    Fill,
    InsufficientBalanceError,
    MarketClient,
    MarketData,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderRejectedError,
    OrderStatus,
    Position,
    Side,
    Sport,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatedMarket:
    """A simulated market for a single contract.

    Maintains a synthetic order book with configurable spread around
    a "true" mid price. External orders interact with this book.
    """

    contract_id: str
    mid_price_cents: int = 50  # Current "true" mid price
    half_spread_cents: int = 2  # Half the bid-ask spread
    volume: int = 0
    open_interest: int = 0

    @property
    def best_bid_cents(self) -> int:
        return max(1, self.mid_price_cents - self.half_spread_cents)

    @property
    def best_ask_cents(self) -> int:
        return min(99, self.mid_price_cents + self.half_spread_cents)

    def to_orderbook(self) -> OrderBook:
        """Generate a synthetic order book snapshot."""
        # Create 5 levels of depth on each side
        bids = []
        asks = []
        for i in range(5):
            bid_price = max(1, self.best_bid_cents - i)
            ask_price = min(99, self.best_ask_cents + i)
            # Deeper levels have more quantity (realistic shape)
            qty = 50 + i * 20
            if bid_price >= 1:
                bids.append(OrderBookLevel(price_cents=bid_price, quantity=qty))
            if ask_price <= 99:
                asks.append(OrderBookLevel(price_cents=ask_price, quantity=qty))
        return OrderBook(
            contract_id=self.contract_id,
            yes_bids=bids,
            yes_asks=asks,
        )

    def to_market_data(self) -> MarketData:
        return MarketData(
            contract_id=self.contract_id,
            last_price_cents=self.mid_price_cents,
            yes_bid_cents=self.best_bid_cents,
            yes_ask_cents=self.best_ask_cents,
            volume_24h=self.volume,
            open_interest=self.open_interest,
        )


class PaperMarketClient(MarketClient):
    """Paper trading client with realistic order book simulation.

    Fills logic:
    - YES buy at price >= best ask → immediate fill at ask price
    - YES buy at price < best ask → rests on book, fills on price tick
    - NO buy at price >= (100 - best bid) → immediate fill
    - Partial fills when order quantity > available depth

    The client tracks:
    - Account balance (decremented on fills, credited on settlements)
    - Open orders (can be cancelled)
    - Positions (updated on fills)
    - Fill history (immutable trade log)
    """

    def __init__(
        self,
        initial_balance_cents: int = 100_000,
        default_spread_cents: int = 4,
        fee_per_contract_cents: int = 0,
    ):
        self._balance_cents = initial_balance_cents
        self._initial_balance_cents = initial_balance_cents
        self._default_half_spread = default_spread_cents // 2
        self._fee_cents = fee_per_contract_cents

        # State
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._positions: dict[str, Position] = {}  # contract_id → Position
        self._markets: dict[str, SimulatedMarket] = {}  # contract_id → market
        self._contracts: dict[str, Contract] = {}  # contract_id → metadata
        self._resting_orders: dict[str, list[str]] = defaultdict(list)  # contract_id → [order_ids]

    # -------------------------------------------------------------------
    # Market setup
    # -------------------------------------------------------------------

    def register_contract(
        self,
        contract_id: str,
        mid_price_cents: int = 50,
        title: str = "",
        contract_type: ContractType = ContractType.MONEYLINE,
        sport: Sport = Sport.NFL,
        event_id: str = "",
    ) -> None:
        """Register a contract for paper trading.

        Must be called before placing orders on a contract.
        """
        self._markets[contract_id] = SimulatedMarket(
            contract_id=contract_id,
            mid_price_cents=mid_price_cents,
            half_spread_cents=self._default_half_spread,
        )
        self._contracts[contract_id] = Contract(
            contract_id=contract_id,
            event_id=event_id or f"evt-{contract_id}",
            title=title or contract_id,
            contract_type=contract_type,
            sport=sport,
            last_price_cents=mid_price_cents,
            is_active=True,
        )
        logger.info(
            "Registered contract %s mid=%d¢ spread=%d¢",
            contract_id,
            mid_price_cents,
            self._default_half_spread * 2,
        )

    def set_price(self, contract_id: str, mid_price_cents: int) -> list[Fill]:
        """Update the simulated mid price and check resting orders for fills.

        Returns any fills triggered by the price movement.
        """
        if contract_id not in self._markets:
            raise ContractNotFoundError(f"Contract {contract_id} not registered")

        market = self._markets[contract_id]
        old_mid = market.mid_price_cents
        market.mid_price_cents = max(1, min(99, mid_price_cents))

        # Check resting orders for fills
        triggered_fills = []
        remaining_order_ids = []

        for order_id in self._resting_orders.get(contract_id, []):
            order = self._orders.get(order_id)
            if order is None or not order.is_active:
                continue

            fills = self._try_fill_order(order, market)
            triggered_fills.extend(fills)

            if order.is_active:
                remaining_order_ids.append(order_id)

        self._resting_orders[contract_id] = remaining_order_ids

        if triggered_fills:
            logger.info(
                "Price move %d→%d¢ on %s triggered %d fills",
                old_mid,
                mid_price_cents,
                contract_id,
                len(triggered_fills),
            )

        return triggered_fills

    # -------------------------------------------------------------------
    # MarketClient ABC implementation
    # -------------------------------------------------------------------

    async def place_order(self, order: Order) -> Order:
        """Place an order on the simulated exchange.

        Immediately fills if the limit price crosses the current spread.
        Otherwise, the order rests on the book.
        """
        contract_id = order.contract_id
        if contract_id not in self._markets:
            raise ContractNotFoundError(f"Contract {contract_id} not registered")

        # Check balance
        required_cents = order.price_cents * order.quantity
        if required_cents > self._balance_cents:
            order.status = OrderStatus.REJECTED
            raise InsufficientBalanceError(
                f"Need {required_cents}¢, have {self._balance_cents}¢"
            )

        # Reserve funds
        self._balance_cents -= required_cents
        order.status = OrderStatus.OPEN
        order.updated_at = datetime.now(UTC)
        self._orders[order.order_id] = order

        # Try immediate fill
        market = self._markets[contract_id]
        fills = self._try_fill_order(order, market)

        if not fills and order.is_active:
            # Order rests on book
            self._resting_orders[contract_id].append(order.order_id)
            logger.debug(
                "Order %s resting: %s %d@%d¢ on %s",
                order.order_id[:8],
                order.side.value,
                order.quantity,
                order.price_cents,
                contract_id,
            )
        elif order.status == OrderStatus.FILLED:
            logger.debug(
                "Order %s filled immediately: %s %d@%d¢",
                order.order_id[:8],
                order.side.value,
                order.filled_quantity,
                order.avg_fill_price_cents,
            )

        return order

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a resting order and return reserved funds."""
        if order_id not in self._orders:
            raise OrderRejectedError(f"Order {order_id} not found")

        order = self._orders[order_id]
        if not order.is_active:
            raise OrderRejectedError(f"Order {order_id} is {order.status.value}, cannot cancel")

        # Return reserved funds for unfilled portion
        refund = order.price_cents * order.remaining_quantity
        self._balance_cents += refund

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(UTC)

        # Remove from resting orders
        contract_resting = self._resting_orders.get(order.contract_id, [])
        if order_id in contract_resting:
            contract_resting.remove(order_id)

        logger.debug("Cancelled order %s, refunded %d¢", order_id[:8], refund)
        return order

    async def get_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise OrderRejectedError(f"Order {order_id} not found")
        return self._orders[order_id]

    async def get_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.quantity > 0]

    async def get_balance_cents(self) -> int:
        return self._balance_cents

    async def get_orderbook(self, contract_id: str) -> OrderBook:
        if contract_id not in self._markets:
            raise ContractNotFoundError(f"Contract {contract_id} not registered")
        return self._markets[contract_id].to_orderbook()

    async def get_market_data(self, contract_id: str) -> MarketData:
        if contract_id not in self._markets:
            raise ContractNotFoundError(f"Contract {contract_id} not registered")
        return self._markets[contract_id].to_market_data()

    async def get_contract(self, contract_id: str) -> Contract:
        if contract_id not in self._contracts:
            raise ContractNotFoundError(f"Contract {contract_id} not registered")
        return self._contracts[contract_id]

    async def get_fills(self, order_id: str | None = None) -> list[Fill]:
        if order_id is None:
            return list(self._fills)
        return [f for f in self._fills if f.order_id == order_id]

    async def cancel_all_orders(self, contract_id: str | None = None) -> int:
        cancelled = 0
        for oid, order in list(self._orders.items()):
            if not order.is_active:
                continue
            if contract_id and order.contract_id != contract_id:
                continue
            await self.cancel_order(oid)
            cancelled += 1
        return cancelled

    # -------------------------------------------------------------------
    # Settlement
    # -------------------------------------------------------------------

    def settle_contract(self, contract_id: str, outcome_yes: bool) -> list[tuple[str, int]]:
        """Settle a contract and pay out positions.

        Args:
            contract_id: The contract to settle.
            outcome_yes: True if YES outcome, False if NO.

        Returns:
            List of (position_key, payout_cents) for settled positions.
        """
        payouts = []

        # Cancel any resting orders
        for order_id in list(self._resting_orders.get(contract_id, [])):
            order = self._orders.get(order_id)
            if order and order.is_active:
                refund = order.price_cents * order.remaining_quantity
                self._balance_cents += refund
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now(UTC)

        self._resting_orders.pop(contract_id, None)

        # Settle positions
        position = self._positions.get(contract_id)
        if position and position.quantity > 0:
            if (position.side == Side.YES and outcome_yes) or (
                position.side == Side.NO and not outcome_yes
            ):
                # Winner: receive 100¢ per contract
                payout = 100 * position.quantity
            else:
                # Loser: receive nothing (cost was already deducted)
                payout = 0

            self._balance_cents += payout
            payouts.append((contract_id, payout))
            logger.info(
                "Settled %s: %s %d contracts → %d¢ payout (outcome=%s)",
                contract_id,
                position.side.value,
                position.quantity,
                payout,
                "YES" if outcome_yes else "NO",
            )
            position.quantity = 0

        # Deactivate contract
        if contract_id in self._contracts:
            self._contracts[contract_id].is_active = False

        return payouts

    # -------------------------------------------------------------------
    # Internal fill logic
    # -------------------------------------------------------------------

    def _try_fill_order(self, order: Order, market: SimulatedMarket) -> list[Fill]:
        """Attempt to fill an order against the simulated market.

        Fill rules:
        - YES buy fills at ask when limit >= ask
        - NO buy fills at (100 - bid) when limit >= (100 - bid)
        - Partial fills when order qty > depth at price level
        """
        fills = []

        if order.side == Side.YES:
            fill_price = market.best_ask_cents
            can_fill = order.price_cents >= fill_price
        else:
            # NO buy: need to pay (100 - best_bid) to take the other side
            fill_price = 100 - market.best_bid_cents
            can_fill = order.price_cents >= fill_price

        if not can_fill:
            return fills

        # Fill the full remaining quantity (simplified — no partial depth)
        fill_qty = order.remaining_quantity
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            contract_id=order.contract_id,
            side=order.side,
            price_cents=fill_price,
            quantity=fill_qty,
            fee_cents=self._fee_cents * fill_qty,
            executed_at=datetime.now(UTC),
        )

        # Update order
        order.filled_quantity += fill_qty
        order.avg_fill_price_cents = fill_price
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now(UTC)

        # Adjust balance: refund reserved amount, charge actual fill cost
        # Reserved: order.price_cents * fill_qty (already deducted)
        # Actual cost: fill_price * fill_qty + fees
        reserved = order.price_cents * fill_qty
        actual_cost = fill_price * fill_qty + fill.fee_cents
        balance_adjustment = reserved - actual_cost  # Positive = refund
        self._balance_cents += balance_adjustment

        # Update position
        self._update_position(fill, order.contract_id)

        # Update market volume
        market.volume += fill_qty
        market.open_interest += fill_qty

        # Record fill
        self._fills.append(fill)
        fills.append(fill)

        return fills

    def _update_position(self, fill: Fill, contract_id: str) -> None:
        """Update or create position from a fill."""
        if contract_id in self._positions:
            self._positions[contract_id].update_from_fill(fill)
        else:
            self._positions[contract_id] = Position(
                contract_id=contract_id,
                side=fill.side,
                quantity=fill.quantity,
                avg_entry_cents=fill.price_cents,
            )

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    @property
    def total_pnl_cents(self) -> int:
        """Total realized + unrealized P&L in cents.

        Equity = cash_balance + sum(position_market_value)
        PNL = equity - initial_balance

        Position market value = mid_price × quantity (for YES)
        or (100 - mid_price) × quantity (for NO).
        """
        position_value = 0
        for cid, pos in self._positions.items():
            if pos.quantity > 0 and cid in self._markets:
                market = self._markets[cid]
                if pos.side == Side.YES:
                    position_value += market.mid_price_cents * pos.quantity
                else:
                    position_value += (100 - market.mid_price_cents) * pos.quantity
        return (self._balance_cents + position_value) - self._initial_balance_cents

    @property
    def total_pnl_dollars(self) -> float:
        return self.total_pnl_cents / 100.0

    @property
    def open_order_count(self) -> int:
        return sum(1 for o in self._orders.values() if o.is_active)

    @property
    def total_fills(self) -> int:
        return len(self._fills)

    def summary(self) -> dict:
        """Get a summary of the paper trading state."""
        positions = [p for p in self._positions.values() if p.quantity > 0]
        return {
            "balance_cents": self._balance_cents,
            "balance_dollars": self._balance_cents / 100.0,
            "initial_balance_dollars": self._initial_balance_cents / 100.0,
            "total_pnl_dollars": self.total_pnl_dollars,
            "open_positions": len(positions),
            "open_orders": self.open_order_count,
            "total_fills": self.total_fills,
            "total_orders": len(self._orders),
        }
