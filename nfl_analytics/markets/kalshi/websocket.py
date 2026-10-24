"""
Kalshi WebSocket client for real-time market data and user notifications.

Connects to Kalshi's WebSocket API with RSA-PSS auth and subscribes to:
  - ticker:          Price/volume updates for watched markets
  - orderbook_delta: Order book snapshots + incremental deltas
  - fill:            User fill notifications (authenticated)

The client maintains local order book state from snapshots + deltas and
invokes callbacks when data arrives.

Usage:
    from nfl_analytics.markets.kalshi.websocket import KalshiWebSocket
    from nfl_analytics.markets.kalshi.auth import KalshiAuth

    auth = KalshiAuth.from_env()
    ws = KalshiWebSocket(auth=auth)

    ws.on_ticker = lambda md: print(f"Ticker: {md}")
    ws.on_fill = lambda fill: print(f"Fill: {fill}")

    await ws.connect()
    await ws.subscribe_tickers(["KXNFL-26-SPREAD-KC-BUF-3.5"])
    await ws.run_forever()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import websockets
import websockets.asyncio.client

from nfl_analytics.markets.base import Fill, MarketData, OrderBook, OrderBookLevel
from nfl_analytics.markets.kalshi.auth import KalshiAuth
from nfl_analytics.markets.kalshi.schemas import (
    parse_fill,
    parse_orderbook_snapshot,
    parse_ticker_message,
)

logger = logging.getLogger(__name__)

PROD_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"


# ---------------------------------------------------------------------------
# Local orderbook state
# ---------------------------------------------------------------------------


@dataclass
class LocalOrderBook:
    """Maintains local order book state from snapshots + deltas.

    Kalshi sends:
      1. orderbook_snapshot on subscribe (full book)
      2. orderbook_delta for incremental updates (price + quantity change)

    Each message has a `seq` number. If we detect a gap, we need to
    re-subscribe to get a fresh snapshot.
    """

    contract_id: str
    # {price_cents: quantity} for YES bids and NO bids
    yes_levels: dict[int, int] = field(default_factory=dict)
    no_levels: dict[int, int] = field(default_factory=dict)
    last_seq: int = 0

    def apply_snapshot(self, ob: OrderBook, seq: int) -> None:
        """Replace local state with a full snapshot."""
        self.yes_levels.clear()
        self.no_levels.clear()

        for level in ob.yes_bids:
            self.yes_levels[level.price_cents] = level.quantity

        # Convert our yes_asks back to no_bids for storage
        for level in ob.yes_asks:
            no_price = 100 - level.price_cents
            self.no_levels[no_price] = level.quantity

        self.last_seq = seq

    def apply_delta(self, msg: dict, seq: int) -> bool:
        """Apply an incremental delta. Returns False if seq gap detected."""
        if seq != self.last_seq + 1 and self.last_seq > 0:
            logger.warning(
                "Seq gap for %s: expected %d, got %d",
                self.contract_id,
                self.last_seq + 1,
                seq,
            )
            return False

        self.last_seq = seq

        side = msg.get("side", "yes")
        price = msg.get("price", 0)
        delta = msg.get("delta", 0)

        levels = self.yes_levels if side == "yes" else self.no_levels

        current = levels.get(price, 0)
        new_qty = current + delta

        if new_qty <= 0:
            levels.pop(price, None)
        else:
            levels[price] = new_qty

        return True

    def to_orderbook(self) -> OrderBook:
        """Convert local state to an OrderBook dataclass."""
        # YES bids: sorted descending by price (best bid first)
        yes_bids = [
            OrderBookLevel(price_cents=p, quantity=q)
            for p, q in sorted(self.yes_levels.items(), reverse=True)
            if q > 0
        ]

        # YES asks from NO bids: ask price = 100 - no_bid_price
        yes_asks = [
            OrderBookLevel(price_cents=100 - p, quantity=q)
            for p, q in sorted(self.no_levels.items(), reverse=True)
            if q > 0
        ]
        yes_asks.sort(key=lambda x: x.price_cents)

        return OrderBook(
            contract_id=self.contract_id,
            yes_bids=yes_bids,
            yes_asks=yes_asks,
        )


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------


class KalshiWebSocket:
    """Kalshi WebSocket client with auto-reconnect and local orderbook state.

    Callbacks:
        on_ticker:    Called with MarketData when a ticker update arrives.
        on_orderbook: Called with OrderBook when the book changes.
        on_fill:      Called with Fill when a user fill arrives.
        on_disconnect: Called with no args when the connection drops.
    """

    def __init__(
        self,
        auth: KalshiAuth,
        demo: bool = False,
        reconnect_delay_seconds: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        self._auth = auth
        self._ws_url = DEMO_WS_URL if demo else PROD_WS_URL
        self._reconnect_delay = reconnect_delay_seconds
        self._max_reconnects = max_reconnect_attempts

        self._ws: websockets.asyncio.client.ClientConnection | None = None
        self._running = False
        self._msg_id = 0
        self._subscriptions: dict[int, dict] = {}  # sid → subscription info

        # Local orderbook state per contract
        self._orderbooks: dict[str, LocalOrderBook] = {}

        # Tickers we're subscribed to
        self._ticker_markets: list[str] = []
        self._orderbook_markets: list[str] = []

        # Callbacks
        self.on_ticker: Callable[[MarketData], None] | None = None
        self.on_orderbook: Callable[[OrderBook], None] | None = None
        self.on_fill: Callable[[Fill], None] | None = None
        self.on_disconnect: Callable[[], None] | None = None

    # -------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the Kalshi WebSocket with RSA auth."""
        headers = self._auth.sign_websocket(ws_path="/trade-api/ws/v2")

        self._ws = await websockets.asyncio.client.connect(
            self._ws_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
        )
        self._running = True
        logger.info("Connected to Kalshi WebSocket (%s)", self._ws_url)

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from Kalshi WebSocket")

    async def run_forever(self) -> None:
        """Process messages with auto-reconnect on failure."""
        reconnect_count = 0

        while self._running:
            try:
                if self._ws is None:
                    await self.connect()
                    # Resubscribe after reconnect
                    await self._resubscribe()
                    reconnect_count = 0

                async for raw_msg in self._ws:
                    if not self._running:
                        break
                    self._handle_message(raw_msg)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("WebSocket closed: %s", e)
            except Exception as e:
                logger.error("WebSocket error: %s", e)

            if not self._running:
                break

            # Reconnect logic
            self._ws = None
            reconnect_count += 1

            if reconnect_count > self._max_reconnects:
                logger.error("Max reconnect attempts (%d) exceeded", self._max_reconnects)
                self._running = False
                break

            if self.on_disconnect:
                self.on_disconnect()

            logger.info(
                "Reconnecting in %.1fs (attempt %d/%d)",
                self._reconnect_delay,
                reconnect_count,
                self._max_reconnects,
            )
            await asyncio.sleep(self._reconnect_delay)

    # -------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------

    async def subscribe_tickers(self, market_tickers: list[str]) -> None:
        """Subscribe to ticker (price/volume) updates for given markets."""
        self._ticker_markets = list(set(self._ticker_markets + market_tickers))
        await self._send_subscribe("ticker", market_tickers)

    async def subscribe_orderbooks(self, market_tickers: list[str]) -> None:
        """Subscribe to orderbook snapshots + deltas for given markets."""
        self._orderbook_markets = list(set(self._orderbook_markets + market_tickers))
        for ticker in market_tickers:
            if ticker not in self._orderbooks:
                self._orderbooks[ticker] = LocalOrderBook(contract_id=ticker)
        await self._send_subscribe("orderbook_delta", market_tickers)

    async def subscribe_fills(self) -> None:
        """Subscribe to user fill notifications (authenticated channel)."""
        await self._send_subscribe("fill")

    async def _send_subscribe(self, channel: str, market_tickers: list[str] | None = None) -> None:
        """Send a subscribe command."""
        if not self._ws:
            logger.warning("Cannot subscribe — not connected")
            return

        self._msg_id += 1
        msg: dict = {
            "id": self._msg_id,
            "cmd": "subscribe",
            "params": {"channels": [channel]},
        }
        if market_tickers:
            msg["params"]["market_tickers"] = market_tickers

        await self._ws.send(json.dumps(msg))
        logger.debug("Subscribed to %s (markets=%s)", channel, market_tickers)

    async def _resubscribe(self) -> None:
        """Resubscribe to all channels after reconnect."""
        if self._ticker_markets:
            await self.subscribe_tickers(self._ticker_markets)
        if self._orderbook_markets:
            await self.subscribe_orderbooks(self._orderbook_markets)
        # Always resubscribe to fills
        await self.subscribe_fills()

    # -------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------

    def _handle_message(self, raw: str | bytes) -> None:
        """Route incoming WebSocket messages to handlers."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from WebSocket: %s", raw[:100])
            return

        msg_type = data.get("type", "")
        msg = data.get("msg", {})
        sid = data.get("sid", 0)
        seq = data.get("seq", 0)

        if msg_type == "subscribed":
            self._subscriptions[msg.get("sid", sid)] = msg
            logger.debug("Subscription confirmed: %s (sid=%d)", msg.get("channel", "?"), sid)
            return

        if msg_type == "error":
            logger.error("WebSocket error (code=%s): %s", data.get("code"), data.get("msg"))
            return

        if msg_type in ("ticker", "ticker_v2"):
            self._handle_ticker(msg)
        elif msg_type == "orderbook_snapshot":
            self._handle_orderbook_snapshot(msg, seq)
        elif msg_type == "orderbook_delta":
            self._handle_orderbook_delta(msg, seq)
        elif msg_type == "fill":
            self._handle_fill(msg)

    def _handle_ticker(self, msg: dict) -> None:
        """Process a ticker update message."""
        if self.on_ticker:
            md = parse_ticker_message(msg)
            self.on_ticker(md)

    def _handle_orderbook_snapshot(self, msg: dict, seq: int) -> None:
        """Process a full orderbook snapshot."""
        ticker = msg.get("market_ticker", "")
        if ticker not in self._orderbooks:
            self._orderbooks[ticker] = LocalOrderBook(contract_id=ticker)

        ob = parse_orderbook_snapshot(msg)
        self._orderbooks[ticker].apply_snapshot(ob, seq)

        if self.on_orderbook:
            self.on_orderbook(self._orderbooks[ticker].to_orderbook())

    def _handle_orderbook_delta(self, msg: dict, seq: int) -> None:
        """Process an incremental orderbook delta."""
        ticker = msg.get("market_ticker", "")
        local = self._orderbooks.get(ticker)
        if local is None:
            logger.warning("Delta for unknown ticker %s, ignoring", ticker)
            return

        ok = local.apply_delta(msg, seq)
        if not ok:
            # Seq gap — need fresh snapshot. Resubscribe triggers one.
            logger.warning("Seq gap for %s — resubscribing for fresh snapshot", ticker)
            asyncio.create_task(self.subscribe_orderbooks([ticker]))
            return

        if self.on_orderbook:
            self.on_orderbook(local.to_orderbook())

    def _handle_fill(self, msg: dict) -> None:
        """Process a user fill notification."""
        if self.on_fill:
            fill = parse_fill(msg)
            self.on_fill(fill)

    # -------------------------------------------------------------------
    # State access
    # -------------------------------------------------------------------

    def get_orderbook(self, contract_id: str) -> OrderBook | None:
        """Get the locally-maintained orderbook for a contract."""
        local = self._orderbooks.get(contract_id)
        return local.to_orderbook() if local else None

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running
