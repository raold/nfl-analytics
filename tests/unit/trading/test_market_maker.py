"""Tests for two-sided market making strategy."""

from __future__ import annotations

import pytest

from nfl_analytics.markets.base import MarketData, OrderBook, OrderBookLevel, Side
from nfl_analytics.trading.market_maker import MarketMaker, MarketMakerConfig
from nfl_analytics.trading.signal import Signal

pytestmark = pytest.mark.anyio


# ============================================================
# Helpers
# ============================================================


def _market_data(
    contract_id: str = "TEST-01",
    last: int = 55,
    bid: int = 54,
    ask: int = 56,
) -> MarketData:
    return MarketData(
        contract_id=contract_id,
        last_price_cents=last,
        yes_bid_cents=bid,
        yes_ask_cents=ask,
        volume_24h=1000,
        open_interest=500,
    )


def _orderbook(
    contract_id: str = "TEST-01",
    bid: int = 54,
    ask: int = 56,
) -> OrderBook:
    return OrderBook(
        contract_id=contract_id,
        yes_bids=[OrderBookLevel(price_cents=bid, quantity=100)],
        yes_asks=[OrderBookLevel(price_cents=ask, quantity=100)],
    )


def _mm(
    base_spread: int = 2,
    cooldown: int = 0,
    max_inventory: int = 100,
    **kwargs,
) -> MarketMaker:
    """Create a MarketMaker with test-friendly defaults."""
    config = MarketMakerConfig(
        base_spread_cents=base_spread,
        cooldown_seconds=cooldown,  # No cooldown for tests
        max_inventory=max_inventory,
        **kwargs,
    )
    return MarketMaker(config)


# ============================================================
# Two-sided quoting
# ============================================================


class TestTwoSidedQuotes:
    """Basic two-sided quote generation."""

    async def test_two_signals_generated(self):
        """Fair value set -> produces bid + ask signals."""
        mm = _mm()
        mm.set_fair_value("TEST-01", 60.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data())

        assert len(signals) == 2
        sides = {s.side for s in signals}
        assert sides == {Side.YES, Side.NO}

    async def test_bid_below_fair_value(self):
        """YES bid is below fair value (buy cheap)."""
        mm = _mm()
        mm.set_fair_value("TEST-01", 60.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data())
        bid_signal = next(s for s in signals if s.side == Side.YES)

        assert bid_signal.price_cents < 60  # Buy below fair value
        assert bid_signal.edge_cents > 0  # Positive edge

    async def test_ask_above_fair_value(self):
        """YES ask (NO bid) is above fair value (sell dear).

        YES ask at X = NO bid at 100-X.
        If ask = 62, NO price = 38.
        """
        mm = _mm()
        mm.set_fair_value("TEST-01", 60.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data())
        ask_signal = next(s for s in signals if s.side == Side.NO)

        # NO price = 100 - ask_price, so ask_price = 100 - no_price
        ask_price = 100 - ask_signal.price_cents
        assert ask_price > 60  # Sell above fair value
        assert ask_signal.edge_cents > 0  # Positive edge from NO perspective

    async def test_no_signals_without_fair_value(self):
        """No fair value -> no quotes."""
        mm = _mm()
        signals = await mm.on_market_data("TEST-01", _market_data())
        assert signals == []

    async def test_strategy_name(self):
        """Strategy name is 'market_maker'."""
        mm = _mm()
        assert mm.name == "market_maker"

    async def test_signal_strategy_label(self):
        """Signals carry the strategy name."""
        mm = _mm()
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data())
        for s in signals:
            assert s.strategy == "market_maker"


# ============================================================
# Spread calculation
# ============================================================


class TestSpreadCalculation:
    """Spread varies with confidence and config."""

    async def test_high_confidence_tight_spread(self):
        """High confidence -> tight spread (closer to base_spread)."""
        mm = _mm(base_spread=2)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.95)

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))
        bid = next(s for s in signals if s.side == Side.YES).price_cents
        ask = 100 - next(s for s in signals if s.side == Side.NO).price_cents

        spread = ask - bid
        assert spread <= 4  # Tight: base(2) + uncertainty(0.2) ≈ 2-3

    async def test_low_confidence_wide_spread(self):
        """Low confidence -> wider spread."""
        mm = _mm(base_spread=2)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.3)

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))
        bid = next(s for s in signals if s.side == Side.YES).price_cents
        ask = 100 - next(s for s in signals if s.side == Side.NO).price_cents

        spread = ask - bid
        assert spread >= 4  # Wide: base(2) + uncertainty(2.8) = 4.8

    async def test_spread_capped_at_max(self):
        """Spread doesn't exceed max_spread_cents."""
        mm = _mm(base_spread=2, max_spread_cents=6)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.0)  # Maximum uncertainty

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))
        bid = next(s for s in signals if s.side == Side.YES).price_cents
        ask = 100 - next(s for s in signals if s.side == Side.NO).price_cents

        assert (ask - bid) <= 6  # Capped

    async def test_spread_floored_at_min(self):
        """Spread never tighter than min_spread_cents."""
        mm = _mm(base_spread=1, min_spread_cents=2)
        mm.set_fair_value("TEST-01", 50.0, confidence=1.0)

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))
        bid = next(s for s in signals if s.side == Side.YES).price_cents
        ask = 100 - next(s for s in signals if s.side == Side.NO).price_cents

        assert (ask - bid) >= 2


# ============================================================
# Inventory skew
# ============================================================


class TestInventorySkew:
    """Quote skewing based on inventory position."""

    async def test_zero_inventory_symmetric(self):
        """No inventory -> symmetric quotes around fair value."""
        mm = _mm()
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))
        bid = next(s for s in signals if s.side == Side.YES).price_cents
        ask = 100 - next(s for s in signals if s.side == Side.NO).price_cents

        # Symmetric: bid and ask equidistant from 50
        assert (50 - bid) == (ask - 50)

    async def test_long_yes_shifts_down(self):
        """Long YES inventory -> quotes shift DOWN (sell more YES)."""
        mm = _mm(cooldown=0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)
        mm._inventory["TEST-01"] = 30  # Long 30 YES

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))
        bid = next(s for s in signals if s.side == Side.YES).price_cents
        ask = 100 - next(s for s in signals if s.side == Side.NO).price_cents

        # Skew = 30 * 0.1 = 3 cents DOWN
        # Bid shifts down (harder to buy more YES)
        # Ask shifts down (easier to sell YES)
        mid = (bid + ask) / 2
        assert mid < 50  # Midpoint shifted down

    async def test_long_no_shifts_up(self):
        """Long NO (negative YES inventory) -> quotes shift UP."""
        mm = _mm(cooldown=0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)
        mm._inventory["TEST-01"] = -30  # Long 30 NO = short 30 YES

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))
        bid = next(s for s in signals if s.side == Side.YES).price_cents
        ask = 100 - next(s for s in signals if s.side == Side.NO).price_cents

        mid = (bid + ask) / 2
        assert mid > 50  # Midpoint shifted up

    async def test_max_inventory_one_sided_long(self):
        """At max YES inventory, only ask (sell YES) is quoted."""
        mm = _mm(cooldown=0, max_inventory=50)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)
        mm._inventory["TEST-01"] = 50  # At max

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))

        # Should only have NO side (selling YES = buying NO)
        assert len(signals) == 1
        assert signals[0].side == Side.NO

    async def test_max_inventory_one_sided_short(self):
        """At max NO inventory, only bid (buy YES) is quoted."""
        mm = _mm(cooldown=0, max_inventory=50)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)
        mm._inventory["TEST-01"] = -50  # At max short

        signals = await mm.on_market_data("TEST-01", _market_data(last=50))

        assert len(signals) == 1
        assert signals[0].side == Side.YES


# ============================================================
# Fill tracking
# ============================================================


class TestFillTracking:
    """on_fill updates inventory correctly."""

    async def test_yes_fill_increases_inventory(self):
        """YES fill -> inventory goes up."""
        mm = _mm()
        await mm.on_fill("TEST-01", "yes", 55, 10)
        assert mm.get_inventory("TEST-01") == 10

    async def test_no_fill_decreases_inventory(self):
        """NO fill -> inventory goes down."""
        mm = _mm()
        await mm.on_fill("TEST-01", "no", 45, 10)
        assert mm.get_inventory("TEST-01") == -10

    async def test_sequential_fills(self):
        """Multiple fills accumulate correctly."""
        mm = _mm()
        await mm.on_fill("TEST-01", "yes", 55, 10)
        await mm.on_fill("TEST-01", "yes", 56, 5)
        await mm.on_fill("TEST-01", "no", 44, 3)
        assert mm.get_inventory("TEST-01") == 12  # 10 + 5 - 3

    async def test_fill_updates_stats(self):
        """Fills increment bid/ask fill counters."""
        mm = _mm()
        await mm.on_fill("TEST-01", "yes", 55, 10)
        await mm.on_fill("TEST-01", "no", 45, 5)
        assert mm.stats["bid_fills"] == 1
        assert mm.stats["ask_fills"] == 1


# ============================================================
# Price boundaries
# ============================================================


class TestPriceBoundaries:
    """Prices clamped to valid [1, 99] range."""

    async def test_low_fair_value(self):
        """Fair value near 1 -> bid may be None but ask is valid."""
        mm = _mm(base_spread=2, cooldown=0)
        mm.set_fair_value("TEST-01", 2.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data(last=2))
        # Should produce at least one signal (ask side)
        # Bid at ~0.8 would be clamped out
        assert all(1 <= s.price_cents <= 99 for s in signals)

    async def test_high_fair_value(self):
        """Fair value near 99 -> ask may be None but bid is valid."""
        mm = _mm(base_spread=2, cooldown=0)
        mm.set_fair_value("TEST-01", 98.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data(last=98))
        # NO price for ask at ~99.2 would be 100-99 = 1 or invalid
        assert all(1 <= s.price_cents <= 99 for s in signals)

    async def test_fair_value_clamped(self):
        """set_fair_value clamps to [1, 99]."""
        mm = _mm()
        mm.set_fair_value("TEST-01", 150.0, confidence=0.8)
        assert mm._fair_values["TEST-01"][0] == 99.0

        mm.set_fair_value("TEST-01", -10.0, confidence=0.8)
        assert mm._fair_values["TEST-01"][0] == 1.0


# ============================================================
# Requote logic
# ============================================================


class TestRequoteLogic:
    """Quote refresh triggers."""

    async def test_first_quote_always_triggers(self):
        """First call always generates quotes."""
        mm = _mm(cooldown=9999)  # Long cooldown
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data())
        assert len(signals) == 2

    async def test_second_call_within_cooldown_blocked(self):
        """Second call within cooldown and no price change -> no quotes."""
        mm = _mm(cooldown=9999, refresh_threshold_cents=5.0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)

        signals1 = await mm.on_market_data("TEST-01", _market_data())
        assert len(signals1) == 2

        # Same fair value, within cooldown
        signals2 = await mm.on_market_data("TEST-01", _market_data())
        assert len(signals2) == 0

    async def test_price_change_triggers_requote(self):
        """Fair value change > threshold triggers requote despite cooldown."""
        mm = _mm(cooldown=9999, refresh_threshold_cents=2.0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)

        signals1 = await mm.on_market_data("TEST-01", _market_data())
        assert len(signals1) == 2

        # Change fair value by more than threshold
        mm.set_fair_value("TEST-01", 53.0, confidence=0.8)
        signals2 = await mm.on_market_data("TEST-01", _market_data())
        assert len(signals2) == 2


# ============================================================
# Quote sizing
# ============================================================


class TestQuoteSizing:
    """Quote size adjusts near inventory limits."""

    async def test_default_quote_size(self):
        """Quotes use configured quote_size."""
        mm = _mm(quote_size=15, cooldown=0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)

        signals = await mm.on_market_data("TEST-01", _market_data())
        for s in signals:
            assert s.quantity == 15

    async def test_reduced_size_near_limit(self):
        """Quote size reduced when approaching max inventory."""
        mm = _mm(quote_size=20, max_inventory=25, cooldown=0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)
        mm._inventory["TEST-01"] = 15  # 10 contracts of headroom

        signals = await mm.on_market_data("TEST-01", _market_data())
        bid_signal = next(s for s in signals if s.side == Side.YES)

        # Buying YES would increase inventory, capped at headroom (10)
        assert bid_signal.quantity <= 10


# ============================================================
# Stats tracking
# ============================================================


class TestStats:
    """Strategy stats accumulation."""

    async def test_stats_initial(self):
        """Fresh MM has zero stats."""
        mm = _mm()
        assert mm.stats["quotes_generated"] == 0
        assert mm.stats["bid_signals"] == 0
        assert mm.stats["ask_signals"] == 0

    async def test_stats_after_quotes(self):
        """Stats update after generating quotes."""
        mm = _mm(cooldown=0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)

        await mm.on_market_data("TEST-01", _market_data())

        assert mm.stats["quotes_generated"] == 1
        assert mm.stats["bid_signals"] == 1
        assert mm.stats["ask_signals"] == 1
        assert mm.stats["contracts_quoted"] == 1

    async def test_clear_fair_value(self):
        """Clearing fair value stops quoting."""
        mm = _mm(cooldown=0)
        mm.set_fair_value("TEST-01", 50.0, confidence=0.8)
        mm.clear_fair_value("TEST-01")

        signals = await mm.on_market_data("TEST-01", _market_data())
        assert signals == []


# ============================================================
# Multiple contracts
# ============================================================


class TestMultipleContracts:
    """Independent quoting across contracts."""

    async def test_independent_inventories(self):
        """Each contract has its own inventory."""
        mm = _mm(cooldown=0)
        await mm.on_fill("C-1", "yes", 55, 10)
        await mm.on_fill("C-2", "no", 45, 20)

        assert mm.get_inventory("C-1") == 10
        assert mm.get_inventory("C-2") == -20

    async def test_independent_quotes(self):
        """Each contract quotes independently."""
        mm = _mm(cooldown=0)
        mm.set_fair_value("C-1", 40.0, confidence=0.8)
        mm.set_fair_value("C-2", 70.0, confidence=0.8)

        s1 = await mm.on_market_data("C-1", _market_data("C-1", last=40))
        s2 = await mm.on_market_data("C-2", _market_data("C-2", last=70))

        assert len(s1) == 2
        assert len(s2) == 2

        bid1 = next(s for s in s1 if s.side == Side.YES).price_cents
        bid2 = next(s for s in s2 if s.side == Side.YES).price_cents
        assert bid1 < bid2  # C-1 FV=40 should have lower bid than C-2 FV=70
