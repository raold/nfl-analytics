"""
Kalshi API response → internal dataclass converters.

Maps Kalshi's JSON field names and conventions to our MarketClient
dataclasses. This is the only module that understands Kalshi's response
format — everything else works with the generic dataclasses from base.py.

Kalshi conventions:
    - Prices in cents (int) and dollars (string, 4 decimal places)
    - Quantities: `count` (int) and `count_fp` (string, 2 decimal places)
    - Order status: "resting" / "executed" / "canceled"
    - Sides: "yes" / "no"
    - Actions: "buy" / "sell"
    - Orderbook: [[price, quantity], ...] arrays

Usage:
    from nfl_analytics.markets.kalshi.schemas import parse_order, parse_fill, parse_orderbook

    order = parse_order(kalshi_json["order"])
    fills = [parse_fill(f) for f in kalshi_json["fills"]]
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from nfl_analytics.markets.base import (
    Contract,
    ContractType,
    Fill,
    MarketData,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Sport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

_KALSHI_STATUS_MAP: dict[str, OrderStatus] = {
    "resting": OrderStatus.OPEN,
    "executed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
}


def _parse_status(kalshi_status: str) -> OrderStatus:
    """Map Kalshi order status to our OrderStatus enum."""
    return _KALSHI_STATUS_MAP.get(kalshi_status, OrderStatus.PENDING)


def _parse_side(side_str: str) -> Side:
    return Side.YES if side_str.lower() == "yes" else Side.NO


def _parse_timestamp(ts: str | int | None) -> datetime:
    """Parse Kalshi timestamp (ISO 8601 string or unix seconds)."""
    if ts is None:
        return datetime.now(UTC)
    if isinstance(ts, int):
        return datetime.fromtimestamp(ts, tz=UTC)
    if isinstance(ts, str):
        # Kalshi uses ISO 8601: "2024-01-01T00:00:00Z"
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Order parsing
# ---------------------------------------------------------------------------


def parse_order(data: dict) -> Order:
    """Convert Kalshi order JSON to our Order dataclass.

    Kalshi fields → our fields:
        order_id     → order_id  (exchange-assigned)
        ticker       → contract_id
        side         → side (YES/NO)
        yes_price    → price_cents
        initial_count → quantity
        fill_count   → filled_quantity
        status       → status (mapped)
        client_order_id → order_id if present (our internal ID)
    """
    # Use client_order_id as our order_id if present, else exchange order_id
    our_order_id = data.get("client_order_id") or data["order_id"]
    exchange_order_id = data["order_id"]

    side = _parse_side(data["side"])

    # Price: Kalshi provides yes_price (cents) and no_price (cents)
    # For YES orders, use yes_price; for NO orders, use no_price
    if side == Side.YES:
        price_cents = data.get("yes_price", 50)
    else:
        price_cents = data.get("no_price", 50)

    # Quantity: prefer int fields over _fp string fields
    quantity = data.get("initial_count", 0)
    filled = data.get("fill_count", 0)

    status = _parse_status(data.get("status", "resting"))

    # If partially filled, mark as PARTIAL
    if filled > 0 and status == OrderStatus.OPEN:
        status = OrderStatus.PARTIAL

    # Compute average fill price from cost data
    avg_fill_price = 0
    if filled > 0:
        # taker_fill_cost + maker_fill_cost = total cost in cents
        total_cost = data.get("taker_fill_cost", 0) + data.get("maker_fill_cost", 0)
        avg_fill_price = total_cost // filled if filled else 0

    return Order(
        order_id=our_order_id,
        contract_id=data.get("ticker", ""),
        side=side,
        price_cents=price_cents,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        status=status,
        filled_quantity=filled,
        avg_fill_price_cents=avg_fill_price,
        strategy=data.get("_strategy", ""),  # Not in Kalshi response; set externally
        created_at=_parse_timestamp(data.get("created_time")),
        updated_at=_parse_timestamp(data.get("last_update_time")),
        exchange_order_id=exchange_order_id,
    )


# ---------------------------------------------------------------------------
# Fill parsing
# ---------------------------------------------------------------------------


def parse_fill(data: dict) -> Fill:
    """Convert Kalshi fill JSON to our Fill dataclass."""
    side = _parse_side(data.get("side", "yes"))

    # Price: same logic as orders
    if side == Side.YES:
        price = data.get("yes_price", 50)
    else:
        price = data.get("no_price", 50)

    return Fill(
        fill_id=data.get("fill_id", data.get("trade_id", "")),
        order_id=data.get("order_id", ""),
        contract_id=data.get("ticker", data.get("market_ticker", "")),
        side=side,
        price_cents=price,
        quantity=data.get("count", 0),
        fee_cents=_parse_fee(data),
        executed_at=_parse_timestamp(data.get("created_time")),
    )


def _parse_fee(data: dict) -> int:
    """Extract fee in cents from fill data.

    Kalshi may provide fee as cents (int) or dollars (string).
    """
    # Try cents first
    if "fee_cost" in data:
        val = data["fee_cost"]
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            try:
                return int(float(val) * 100)
            except (ValueError, TypeError):
                return 0
    return 0


# ---------------------------------------------------------------------------
# Position parsing
# ---------------------------------------------------------------------------


def parse_position(data: dict) -> Position:
    """Convert Kalshi position JSON to our Position dataclass.

    Kalshi returns `position` (positive = YES, negative = NO) and
    `market_exposure` (total cost basis in cents).
    """
    raw_position = data.get("position", 0)

    if raw_position >= 0:
        side = Side.YES
        quantity = raw_position
    else:
        side = Side.NO
        quantity = abs(raw_position)

    # Average entry from exposure / quantity
    exposure = data.get("market_exposure", 0)
    avg_entry = exposure // quantity if quantity > 0 else 0

    return Position(
        contract_id=data.get("ticker", ""),
        side=side,
        quantity=quantity,
        avg_entry_cents=avg_entry,
    )


# ---------------------------------------------------------------------------
# OrderBook parsing
# ---------------------------------------------------------------------------


def parse_orderbook(data: dict, contract_id: str) -> OrderBook:
    """Convert Kalshi orderbook JSON to our OrderBook dataclass.

    Kalshi format:
        {"yes": [[price, qty], ...], "no": [[price, qty], ...]}
    Yes bids sorted ascending (highest last), yes asks = no bids inverted.
    """
    ob_data = data.get("orderbook", data)

    # YES side: these are the yes limit orders
    # In Kalshi's response, "yes" array = resting YES orders (bids for YES)
    # "no" array = resting NO orders (which are asks for YES)
    yes_levels = ob_data.get("yes", [])
    no_levels = ob_data.get("no", [])

    # YES bids: from "yes" array, sorted descending (best bid first)
    yes_bids = [
        OrderBookLevel(price_cents=int(level[0]), quantity=int(level[1]))
        for level in reversed(yes_levels)
        if len(level) >= 2
    ]

    # YES asks: derived from NO bids → YES ask = 100 - NO bid price
    yes_asks = [
        OrderBookLevel(price_cents=100 - int(level[0]), quantity=int(level[1]))
        for level in reversed(no_levels)
        if len(level) >= 2
    ]
    # Sort asks ascending (best ask first = lowest price first)
    yes_asks.sort(key=lambda x: x.price_cents)

    return OrderBook(
        contract_id=contract_id,
        yes_bids=yes_bids,
        yes_asks=yes_asks,
    )


def parse_orderbook_snapshot(msg: dict) -> OrderBook:
    """Parse WebSocket orderbook_snapshot message.

    Same format as REST but nested under 'msg' with market_ticker.
    """
    ticker = msg.get("market_ticker", "")
    return parse_orderbook(msg, ticker)


# ---------------------------------------------------------------------------
# Market data parsing
# ---------------------------------------------------------------------------


def parse_market_data(data: dict) -> MarketData:
    """Convert Kalshi market JSON to our MarketData dataclass."""
    # Kalshi provides prices as cents and/or dollar strings
    # REST uses "last_price", WebSocket ticker uses "price"
    last_price = data.get("last_price", data.get("price", 0))
    if last_price == 0 and "last_price_dollars" in data:
        try:
            last_price = int(float(data["last_price_dollars"]) * 100)
        except (ValueError, TypeError):
            pass

    yes_bid = data.get("yes_bid", 0)
    if yes_bid == 0 and "yes_bid_dollars" in data:
        try:
            yes_bid = int(float(data["yes_bid_dollars"]) * 100)
        except (ValueError, TypeError):
            pass

    yes_ask = data.get("yes_ask", 0)
    if yes_ask == 0 and "yes_ask_dollars" in data:
        try:
            yes_ask = int(float(data["yes_ask_dollars"]) * 100)
        except (ValueError, TypeError):
            pass

    volume = data.get("volume", data.get("volume_24h", 0))
    oi = data.get("open_interest", 0)

    return MarketData(
        contract_id=data.get("ticker", data.get("market_ticker", "")),
        last_price_cents=last_price,
        yes_bid_cents=yes_bid,
        yes_ask_cents=yes_ask,
        volume_24h=volume,
        open_interest=oi,
    )


def parse_ticker_message(msg: dict) -> MarketData:
    """Parse WebSocket ticker message into MarketData."""
    return parse_market_data(msg)


# ---------------------------------------------------------------------------
# Contract parsing
# ---------------------------------------------------------------------------

_CONTRACT_TYPE_MAP = {
    "binary": ContractType.CUSTOM,
    "scalar": ContractType.CUSTOM,
}

_SPORT_FROM_TICKER: dict[str, Sport] = {
    "NFL": Sport.NFL,
    "NBA": Sport.NBA,
    "MLB": Sport.MLB,
    "NHL": Sport.NHL,
}


def _infer_sport(ticker: str) -> Sport:
    """Infer sport from contract ticker prefix."""
    upper = ticker.upper()
    for prefix, sport in _SPORT_FROM_TICKER.items():
        if prefix in upper:
            return sport
    return Sport.NFL  # Default


def parse_contract(data: dict) -> Contract:
    """Convert Kalshi market/contract JSON to our Contract dataclass."""
    ticker = data.get("ticker", "")
    status = data.get("status", "active")

    return Contract(
        contract_id=ticker,
        event_id=data.get("event_ticker", ""),
        title=data.get("title", data.get("subtitle", ticker)),
        contract_type=_CONTRACT_TYPE_MAP.get(data.get("market_type", ""), ContractType.CUSTOM),
        sport=_infer_sport(ticker),
        strike=None,
        expiry=_parse_timestamp(data.get("expected_expiration_time")) if data.get("expected_expiration_time") else None,
        last_price_cents=data.get("last_price", 0),
        volume=data.get("volume", 0),
        open_interest=data.get("open_interest", 0),
        is_active=status in ("active", "initialized"),
    )


# ---------------------------------------------------------------------------
# Request builders (internal → Kalshi JSON)
# ---------------------------------------------------------------------------


def build_create_order_body(order: Order) -> dict:
    """Build Kalshi create-order request body from our Order dataclass.

    Maps our Order fields to Kalshi's expected JSON format.
    """
    body: dict = {
        "ticker": order.contract_id,
        "action": "buy",  # We always buy (YES or NO side)
        "side": order.side.value,
        "count": order.quantity,
        "type": "limit",
        "client_order_id": order.order_id,
    }

    # Set price on the correct side
    if order.side == Side.YES:
        body["yes_price"] = order.price_cents
    else:
        body["no_price"] = order.price_cents

    return body


def build_batch_create_body(orders: list[Order]) -> dict:
    """Build Kalshi batch create-orders request body."""
    return {
        "orders": [build_create_order_body(o) for o in orders],
    }


def build_batch_cancel_body(order_ids: list[str]) -> dict:
    """Build Kalshi batch cancel-orders request body."""
    return {
        "orders": [{"order_id": oid} for oid in order_ids],
    }
