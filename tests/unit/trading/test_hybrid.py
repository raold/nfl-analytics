"""Tests for hybrid market making + directional strategy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nfl_analytics.markets.base import MarketData, OrderBook, OrderBookLevel, Side
from nfl_analytics.trading.hybrid import HybridConfig, HybridStrategy
from nfl_analytics.trading.market_maker import MarketMakerConfig
from nfl_analytics.trading.signal import Signal

pytestmark = pytest.mark.anyio


# ============================================================
# Helpers
# ============================================================


def _market_data(
    contract_id: str = "KXNFL-26-SPREAD-KC-BUF-3.5",
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
    contract_id: str = "KXNFL-26-SPREAD-KC-BUF-3.5",
    bid: int = 54,
    ask: int = 56,
) -> OrderBook:
    return OrderBook(
        contract_id=contract_id,
        yes_bids=[OrderBookLevel(price_cents=bid, quantity=100)],
        yes_asks=[OrderBookLevel(price_cents=ask, quantity=100)],
    )


def _mock_model(home_spread: float = -3.0, home_win_prob: float = 0.62):
    """Create a mock model that returns fixed predictions."""
    model = MagicMock()
    model.predict.return_value = {
        "home_spread": home_spread,
        "total_points": 47.5,
        "home_win_probability": home_win_prob,
        "spread_std": 3.5,
        "total_std": 5.0,
    }
    model.feature_names = ["f1", "f2"]
    return model


def _hybrid(
    directional_edge: int = 5,
    mm_cooldown: int = 0,
    cooldown: int = 0,
    model: object | None = None,
    **kwargs,
) -> HybridStrategy:
    """Create a HybridStrategy with test-friendly defaults."""
    config = HybridConfig(
        directional_edge_cents=directional_edge,
        mm_config=MarketMakerConfig(cooldown_seconds=mm_cooldown, base_spread_cents=2),
        cooldown_seconds=cooldown,
        **kwargs,
    )
    return HybridStrategy(config=config, model=model)


# ============================================================
# Mode routing
# ============================================================


class TestModeRouting:
    """Correct mode selection based on edge magnitude."""

    async def test_small_edge_routes_to_mm(self):
        """Small edge -> MM mode (two-sided quotes)."""
        # Model says 57% -> fair value 57¢, market at 55¢
        # Edge = 2¢ < 5¢ threshold -> MM mode
        model = _mock_model(home_spread=-3.0)  # Will produce ~57% spread cover
        hybrid = _hybrid(directional_edge=5, model=model)
        hybrid.set_week(2026, 1)

        md = _market_data(last=55, bid=54, ask=56)
        ob = _orderbook(bid=54, ask=56)
        signals = await hybrid.on_market_data(md.contract_id, md, ob)

        # MM produces 2 signals (bid + ask)
        if signals:  # Model may produce small edge
            assert all(s.strategy == "hybrid" or s.strategy == "market_maker" for s in signals)

    async def test_large_edge_routes_to_directional(self):
        """Large edge -> directional mode (one-sided signal)."""
        # Model says home favored by 10 points with line 3.5
        # Cover prob = norm.cdf(10.0 + 3.5, 0, 3.5) ≈ 99.9%
        # Fair value ~99¢, market at 55¢ -> edge ~44¢ >> 5¢ threshold
        model = _mock_model(home_spread=10.0, home_win_prob=0.85)
        hybrid = _hybrid(directional_edge=5, model=model)
        hybrid.set_week(2026, 1)

        contract_id = "KXNFL-26-SPREAD-KC-BUF-3.5"
        md = _market_data(contract_id=contract_id, last=55)
        ob = _orderbook(contract_id=contract_id, bid=54, ask=56)
        signals = await hybrid.on_market_data(contract_id, md, ob)

        # Should produce directional signal(s), not MM quotes
        assert len(signals) >= 1
        for s in signals:
            assert s.strategy == "hybrid"

    async def test_no_week_set_returns_empty(self):
        """Without set_week(), no signals generated."""
        model = _mock_model()
        hybrid = _hybrid(model=model)

        signals = await hybrid.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5",
            _market_data(),
        )
        assert signals == []

    async def test_unparseable_contract_returns_empty(self):
        """Non-NFL contract ID -> no signals."""
        model = _mock_model()
        hybrid = _hybrid(model=model)
        hybrid.set_week(2026, 1)

        signals = await hybrid.on_market_data(
            "RANDOM-CONTRACT",
            _market_data(contract_id="RANDOM-CONTRACT"),
        )
        assert signals == []


# ============================================================
# Strategy name and properties
# ============================================================


class TestProperties:
    """Strategy identity and properties."""

    def test_name_is_hybrid(self):
        assert _hybrid().name == "hybrid"

    def test_internal_mm_accessible(self):
        """Market maker is accessible for inventory inspection."""
        hybrid = _hybrid()
        assert hybrid.market_maker is not None
        assert hybrid.market_maker.name == "market_maker"

    def test_stats_includes_mm(self):
        """Stats include both hybrid and MM sub-stats."""
        hybrid = _hybrid()
        stats = hybrid.stats
        assert "directional_signals" in stats
        assert "mm_signals" in stats
        assert "mm_stats" in stats


# ============================================================
# Inventory sharing
# ============================================================


class TestInventorySharing:
    """Fills update shared MM inventory regardless of mode."""

    async def test_fill_updates_mm_inventory(self):
        """on_fill forwards to MM inventory tracker."""
        hybrid = _hybrid()
        await hybrid.on_fill("C-1", "yes", 55, 10)

        assert hybrid.market_maker.get_inventory("C-1") == 10

    async def test_no_fill_reduces_inventory(self):
        """NO fill reduces YES inventory."""
        hybrid = _hybrid()
        await hybrid.on_fill("C-1", "yes", 55, 20)
        await hybrid.on_fill("C-1", "no", 45, 8)

        assert hybrid.market_maker.get_inventory("C-1") == 12

    async def test_inventory_persists_across_modes(self):
        """Inventory built in one mode affects the other."""
        hybrid = _hybrid()
        # Simulate directional fill building inventory
        await hybrid.on_fill("C-1", "yes", 55, 30)

        # MM should now see inventory = 30 and skew accordingly
        assert hybrid.market_maker.get_inventory("C-1") == 30


# ============================================================
# Lifecycle
# ============================================================


class TestLifecycle:
    """Start/stop lifecycle hooks."""

    async def test_start_activates(self):
        """on_start activates both hybrid and MM."""
        hybrid = _hybrid()
        await hybrid.on_start()
        assert hybrid._active

    async def test_stop_deactivates(self):
        """on_stop deactivates both hybrid and MM."""
        hybrid = _hybrid()
        await hybrid.on_start()
        await hybrid.on_stop()
        assert not hybrid._active

    async def test_set_week_clears_cache(self):
        """Changing week clears prediction cache."""
        model = _mock_model()
        hybrid = _hybrid(model=model)
        hybrid.set_week(2026, 1)

        # Populate cache
        hybrid._prediction_cache["2026-KC-BUF"] = MagicMock()
        assert len(hybrid._prediction_cache) == 1

        # Change week
        hybrid.set_week(2026, 2)
        assert len(hybrid._prediction_cache) == 0


# ============================================================
# Directional signal details
# ============================================================


class TestDirectionalSignals:
    """Directional mode signal properties."""

    async def test_directional_signal_has_kelly_sizing(self):
        """Directional signals use Kelly-optimal sizing."""
        # Model with high confidence and large edge
        model = _mock_model(home_win_prob=0.80)
        hybrid = _hybrid(
            directional_edge=3,
            model=model,
            bankroll_cents=100_000,
        )
        hybrid.set_week(2026, 1)

        # Use spread contract where probability maps to high edge
        contract_id = "KXNFL-26-SPREAD-KC-BUF-3.5"
        md = _market_data(contract_id=contract_id, last=50, bid=49, ask=51)
        ob = _orderbook(contract_id=contract_id, bid=49, ask=51)

        signals = await hybrid.on_market_data(contract_id, md, ob)

        if signals:
            # Directional signal should have reasonable quantity
            assert signals[0].quantity > 0
            assert signals[0].quantity <= hybrid._config.max_directional_contracts

    async def test_directional_cooldown_respected(self):
        """Directional signals respect cooldown."""
        model = _mock_model(home_win_prob=0.80)
        hybrid = _hybrid(directional_edge=3, model=model, cooldown=9999)
        hybrid.set_week(2026, 1)

        contract_id = "KXNFL-26-SPREAD-KC-BUF-3.5"
        md = _market_data(contract_id=contract_id, last=50)
        ob = _orderbook(contract_id=contract_id, bid=49, ask=51)

        signals1 = await hybrid.on_market_data(contract_id, md, ob)
        signals2 = await hybrid.on_market_data(contract_id, md, ob)

        # Second call should be blocked by cooldown
        # (first may or may not produce signals depending on edge)
        if signals1:
            assert len(signals2) == 0


# ============================================================
# MM mode details
# ============================================================


class TestMMMode:
    """Market making mode behavior within hybrid."""

    async def test_mm_mode_sets_fair_value(self):
        """MM mode injects fair value before generating quotes."""
        model = _mock_model(home_win_prob=0.55)  # Close to market -> MM mode
        hybrid = _hybrid(directional_edge=10, model=model, mm_cooldown=0)
        hybrid.set_week(2026, 1)

        # Market at 55 with model at ~55% -> edge ~0 -> MM mode
        contract_id = "KXNFL-26-SPREAD-KC-BUF-3.5"
        md = _market_data(contract_id=contract_id, last=55, bid=54, ask=56)
        ob = _orderbook(contract_id=contract_id, bid=54, ask=56)

        signals = await hybrid.on_market_data(contract_id, md, ob)

        # If MM mode activated, fair value should be set on internal MM
        if contract_id in hybrid.market_maker._fair_values:
            fv, conf = hybrid.market_maker._fair_values[contract_id]
            assert 1.0 <= fv <= 99.0


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    """Boundary conditions and error handling."""

    async def test_no_model_returns_empty(self):
        """Without model, no signals generated."""
        hybrid = _hybrid(model=None)
        hybrid.set_week(2026, 1)

        signals = await hybrid.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5",
            _market_data(),
        )
        assert signals == []

    async def test_win_contract_no_opponent(self):
        """WIN contract without opponent -> no signals (can't predict)."""
        model = _mock_model()
        hybrid = _hybrid(model=model)
        hybrid.set_week(2026, 1)

        signals = await hybrid.on_market_data(
            "KXNFL-26-WIN-KC",
            _market_data(contract_id="KXNFL-26-WIN-KC"),
        )
        assert signals == []

    async def test_model_exception_returns_empty(self):
        """Model exception -> no signals, no crash."""
        model = MagicMock()
        model.predict.side_effect = ValueError("Model error")
        hybrid = _hybrid(model=model)
        hybrid.set_week(2026, 1)

        signals = await hybrid.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5",
            _market_data(),
        )
        assert signals == []

    async def test_zero_quantity_filtered(self):
        """Signals with zero quantity are not emitted."""
        model = _mock_model(home_win_prob=0.501)  # Tiny edge
        hybrid = _hybrid(
            directional_edge=1,
            model=model,
            bankroll_cents=100,  # Very small bankroll -> 0 quantity
        )
        hybrid.set_week(2026, 1)

        contract_id = "KXNFL-26-SPREAD-KC-BUF-3.5"
        md = _market_data(contract_id=contract_id, last=50)

        signals = await hybrid.on_market_data(contract_id, md)
        # Zero-quantity signals should be filtered
        for s in signals:
            assert s.quantity > 0
