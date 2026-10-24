"""
Two-sided quoting market making strategy for prediction markets.

Posts simultaneous YES bids and NO bids (= YES asks) around the model's
fair value estimate. Profits from the bid-ask spread when both sides
fill. Manages inventory risk through quote skewing.

Core formula:
    fair_value = model_probability_cents  (e.g., 62 for 62% prob)
    half_spread = (base + uncertainty_premium) / 2
    skew = net_inventory * skew_per_lot

    bid = fair_value - half_spread - skew
    ask = fair_value + half_spread - skew

When long YES (positive inventory):
    - Both quotes shift DOWN -> easier to sell YES, harder to buy more
When long NO (negative inventory):
    - Both quotes shift UP -> easier to buy YES, harder to sell more

The MM does NOT own a model. Fair values are injected via set_fair_value()
so the pricing model can live in a separate layer (or in HybridStrategy).
This separation keeps the MM focused on quoting mechanics.

Signal output:
    Each on_market_data() returns 0-2 signals:
    - YES bid: Signal(side=YES, price=bid_price)
    - YES ask as NO bid: Signal(side=NO, price=100-ask_price)

Usage:
    from nfl_analytics.trading.market_maker import MarketMaker, MarketMakerConfig

    mm = MarketMaker(config=MarketMakerConfig(base_spread_cents=2))
    mm.set_fair_value("KXNFL-26-WIN-KC", 62.0, confidence=0.8)
    signals = await mm.on_market_data("KXNFL-26-WIN-KC", market_data, orderbook)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nfl_analytics.markets.base import MarketData, OrderBook, Side
from nfl_analytics.trading.base_strategy import StrategyConfig, TradingStrategy
from nfl_analytics.trading.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class MarketMakerConfig(StrategyConfig):
    """Configuration for two-sided market making.

    Spread components:
        total_spread = base_spread + uncertainty_premium
        uncertainty_premium = (1 - confidence) * uncertainty_multiplier
        half_spread = total_spread / 2

    Inventory management:
        skew_cents = net_inventory * skew_cents_per_lot
        At max_inventory, only the inventory-reducing side quotes.
    """

    # Spread parameters
    base_spread_cents: int = 2  # Minimum spread (1 cent each side)
    uncertainty_multiplier: float = 4.0  # Spread scaling for low confidence
    max_spread_cents: int = 10  # Never quote wider than this
    min_spread_cents: int = 2  # Never quote tighter than this

    # Inventory management
    skew_cents_per_lot: float = 0.1  # 1 cent skew per 10 contracts of imbalance
    max_inventory: int = 100  # Max net inventory per contract before one-sided quoting

    # Quote parameters
    quote_size: int = 10  # Contracts per quote
    refresh_threshold_cents: float = 1.0  # Re-quote on fair value change > this
    cooldown_seconds: int = 30  # Re-quote interval (faster than directional's 60s)

    # Edge requirement is lower for MM (spread capture, not directional edge)
    min_edge_cents: int = 1


class MarketMaker(TradingStrategy):
    """Two-sided quoting strategy for prediction market contracts.

    The MM posts bids and asks around a fair value, profiting from the spread
    when both sides fill. Inventory buildup is managed by skewing quotes
    toward the reducing side.

    Fair values must be set externally via set_fair_value() before the MM
    will quote. This decouples the pricing model from the quoting engine.

    Thread-safe: each contract has independent state (fair value, inventory,
    last quotes). No shared mutable state between contracts.
    """

    def __init__(self, config: MarketMakerConfig | None = None):
        cfg = config or MarketMakerConfig()
        super().__init__(cfg)
        self._config: MarketMakerConfig = cfg

        # Per-contract state
        self._fair_values: dict[str, tuple[float, float]] = {}  # contract → (fv_cents, confidence)
        self._inventory: dict[str, int] = {}  # contract → net YES contracts (+ = long YES)
        self._last_fair_value: dict[str, float] = {}  # contract → last quoted fair value
        self._last_quotes: dict[str, tuple[int | None, int | None]] = {}  # → (bid, ask)

        # Stats
        self._stats = {
            "quotes_generated": 0,
            "bid_signals": 0,
            "ask_signals": 0,
            "bid_fills": 0,
            "ask_fills": 0,
            "spread_captured_cents": 0,
        }

    @property
    def name(self) -> str:
        return "market_maker"

    @property
    def stats(self) -> dict:
        """Market making performance statistics."""
        return {**self._stats, "contracts_quoted": len(self._fair_values)}

    # ------------------------------------------------------------------
    # Fair value management
    # ------------------------------------------------------------------

    def set_fair_value(
        self, contract_id: str, value_cents: float, confidence: float = 0.8
    ) -> None:
        """Set the fair value for a contract.

        Args:
            contract_id: The contract to price.
            value_cents: Model's fair value in cents (1-99 scale).
            confidence: Model confidence (0-1). Lower = wider spread.
        """
        value_cents = max(1.0, min(99.0, value_cents))
        confidence = max(0.0, min(1.0, confidence))
        self._fair_values[contract_id] = (value_cents, confidence)

    def clear_fair_value(self, contract_id: str) -> None:
        """Remove fair value for a contract (stops quoting)."""
        self._fair_values.pop(contract_id, None)

    def get_inventory(self, contract_id: str) -> int:
        """Get net YES inventory for a contract.

        Positive = long YES, negative = long NO (short YES).
        """
        return self._inventory.get(contract_id, 0)

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    async def on_market_data(
        self,
        contract_id: str,
        market_data: MarketData,
        orderbook: OrderBook | None = None,
    ) -> list[Signal]:
        """Generate two-sided quotes around the fair value.

        Returns 0-2 signals: a YES bid and/or a NO bid (YES ask).
        Skips quoting if:
        - No fair value set for this contract
        - Fair value hasn't changed enough and cooldown hasn't elapsed
        - Prices would be outside valid range
        """
        fv = self._fair_values.get(contract_id)
        if fv is None:
            return []  # Can't quote without fair value

        fair_value_cents, confidence = fv

        # Check if we should re-quote
        if not self._should_requote(contract_id, fair_value_cents):
            return []

        inventory = self._inventory.get(contract_id, 0)
        bid, ask = self._compute_quotes(fair_value_cents, confidence, inventory)

        signals: list[Signal] = []

        if bid is not None:
            bid_qty = self._quote_size_for_side(inventory, adding_yes=True)
            if bid_qty > 0:
                signals.append(Signal(
                    contract_id=contract_id,
                    side=Side.YES,
                    quantity=bid_qty,
                    price_cents=bid,
                    edge_cents=max(1, round(fair_value_cents - bid)),
                    model_prob=fair_value_cents / 100.0,
                    confidence=confidence,
                    strategy=self.name,
                ))
                self._stats["bid_signals"] += 1

        if ask is not None:
            ask_qty = self._quote_size_for_side(inventory, adding_yes=False)
            no_price = 100 - ask
            if ask_qty > 0 and 1 <= no_price <= 99:
                signals.append(Signal(
                    contract_id=contract_id,
                    side=Side.NO,
                    quantity=ask_qty,
                    price_cents=no_price,
                    edge_cents=max(1, round(ask - fair_value_cents)),
                    model_prob=(100.0 - fair_value_cents) / 100.0,
                    confidence=confidence,
                    strategy=self.name,
                ))
                self._stats["ask_signals"] += 1

        if signals:
            self._record_signal(contract_id)
            self._last_fair_value[contract_id] = fair_value_cents
            self._last_quotes[contract_id] = (bid, ask)
            self._stats["quotes_generated"] += 1

            logger.info(
                "MM %s: bid=%s ask=%s inv=%d fv=%.1f conf=%.2f",
                contract_id,
                bid,
                ask,
                inventory,
                fair_value_cents,
                confidence,
            )

        return signals

    async def on_fill(
        self, contract_id: str, side: str, price_cents: int, quantity: int
    ) -> None:
        """Update inventory on fill.

        YES fill → inventory increases (long more YES).
        NO fill → inventory decreases (sold YES / long NO).
        """
        prev = self._inventory.get(contract_id, 0)

        if side == "yes":
            self._inventory[contract_id] = prev + quantity
            self._stats["bid_fills"] += 1
        else:
            self._inventory[contract_id] = prev - quantity
            self._stats["ask_fills"] += 1

        # Track spread captured when opposite sides fill
        if self._stats["bid_fills"] > 0 and self._stats["ask_fills"] > 0:
            last_bid, last_ask = self._last_quotes.get(contract_id, (None, None))
            if last_bid is not None and last_ask is not None:
                self._stats["spread_captured_cents"] += (last_ask - last_bid) * quantity

        logger.debug(
            "MM fill %s: %s %dx @%d, inventory %d -> %d",
            contract_id,
            side,
            quantity,
            price_cents,
            prev,
            self._inventory.get(contract_id, 0),
        )

    # ------------------------------------------------------------------
    # Internal: quoting logic
    # ------------------------------------------------------------------

    def _should_requote(self, contract_id: str, new_fair_value: float) -> bool:
        """Check if a new quote is needed.

        Requote triggers:
        1. Fair value changed by more than refresh_threshold (price-driven)
        2. Cooldown elapsed since last quote (time-driven)
        3. Never quoted this contract before
        """
        last_fv = self._last_fair_value.get(contract_id)
        if last_fv is None:
            return True  # First time quoting

        # Price-driven: fair value moved enough
        if abs(new_fair_value - last_fv) >= self._config.refresh_threshold_cents:
            return True

        # Time-driven: cooldown elapsed
        return self._check_cooldown(contract_id)

    def _compute_quotes(
        self,
        fair_value_cents: float,
        confidence: float,
        inventory: int,
    ) -> tuple[int | None, int | None]:
        """Compute bid and ask prices.

        Returns (bid_price, ask_price) in YES cents.
        None on a side means don't quote that side.
        """
        # Spread: base + uncertainty premium
        uncertainty_premium = (1.0 - confidence) * self._config.uncertainty_multiplier
        total_spread = self._config.base_spread_cents + uncertainty_premium
        total_spread = max(self._config.min_spread_cents, min(self._config.max_spread_cents, total_spread))
        half_spread = total_spread / 2.0

        # Inventory skew: pushes quotes toward reducing inventory
        skew = inventory * self._config.skew_cents_per_lot

        # Raw prices (both shift by -skew)
        bid_raw = fair_value_cents - half_spread - skew
        ask_raw = fair_value_cents + half_spread - skew

        # Clamp to valid range [1, 99]
        bid = self._clamp_price(bid_raw)
        ask = self._clamp_price(ask_raw)

        # Ensure bid < ask (no crossed quotes)
        if bid is not None and ask is not None and bid >= ask:
            mid = round(fair_value_cents)
            bid = self._clamp_price(mid - 1)
            ask = self._clamp_price(mid + 1)
            if bid is not None and ask is not None and bid >= ask:
                return None, None  # Can't quote at boundaries (fv near 1 or 99)

        # At max inventory, only quote the reducing side
        abs_inv = abs(inventory)
        if abs_inv >= self._config.max_inventory:
            if inventory > 0:
                # Long YES → only ask (sell YES)
                return None, ask
            else:
                # Long NO → only bid (buy YES)
                return bid, None

        return bid, ask

    def _quote_size_for_side(self, inventory: int, adding_yes: bool) -> int:
        """Compute quote size, capping at inventory headroom.

        If the quote would push inventory beyond max, reduce size to
        fit within limits.
        """
        base = self._config.quote_size
        headroom = self._config.max_inventory - abs(inventory)

        if headroom <= 0:
            # At max: only allow reducing quotes
            if adding_yes and inventory > 0:
                return 0  # Would increase long YES
            if not adding_yes and inventory < 0:
                return 0  # Would increase long NO
            return base  # Reducing side gets full size

        # Check if this side increases inventory
        would_increase = (adding_yes and inventory >= 0) or (not adding_yes and inventory <= 0)
        if would_increase:
            return min(base, headroom)

        return base

    @staticmethod
    def _clamp_price(price: float) -> int | None:
        """Clamp price to valid range [1, 99]. Returns None if out of range."""
        rounded = round(price)
        if rounded < 1 or rounded > 99:
            return None
        return rounded
