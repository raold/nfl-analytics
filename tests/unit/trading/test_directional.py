"""Tests for directional trading strategy.

Tests contract ID parsing, edge calculation, signal generation,
and the full strategy pipeline with mock model components.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from py.markets.base import MarketData, OrderBook, OrderBookLevel, Side

pytestmark = pytest.mark.anyio
from py.trading.directional import (
    DirectionalConfig,
    DirectionalStrategy,
    GamePrediction,
    ParsedContract,
    parse_contract_id,
)
from py.trading.signal import Signal


# ============================================================
# Contract ID parsing
# ============================================================


class TestParseContractId:
    """Parse Kalshi NFL contract IDs into structured data."""

    def test_win_contract(self):
        parsed = parse_contract_id("KXNFL-26-WIN-KC")
        assert parsed is not None
        assert parsed.contract_type == "win"
        assert parsed.season == 2026
        assert parsed.home_team == "KC"

    def test_spread_contract(self):
        parsed = parse_contract_id("KXNFL-26-SPREAD-KC-BUF-3.5")
        assert parsed is not None
        assert parsed.contract_type == "spread"
        assert parsed.season == 2026
        assert parsed.home_team == "KC"
        assert parsed.away_team == "BUF"
        assert parsed.line == 3.5

    def test_spread_negative_line(self):
        parsed = parse_contract_id("KXNFL-26-SPREAD-KC-BUF--3.5")
        # Negative line in contract ID (double dash)
        # Our regex expects [\d.+-]+ so this should work
        assert parsed is not None
        assert parsed.line == -3.5

    def test_total_contract(self):
        parsed = parse_contract_id("KXNFL-26-TOTAL-KC-BUF-47.5")
        assert parsed is not None
        assert parsed.contract_type == "total"
        assert parsed.line == 47.5

    def test_unknown_contract_returns_none(self):
        assert parse_contract_id("CRYPTO-BTC-50K") is None
        assert parse_contract_id("") is None
        assert parse_contract_id("KXNFL-26-UNKNOWN-KC") is None

    def test_case_insensitive(self):
        parsed = parse_contract_id("kxnfl-26-win-kc")
        assert parsed is not None
        assert parsed.home_team == "KC"  # Uppercased

    def test_game_key(self):
        parsed = parse_contract_id("KXNFL-26-SPREAD-KC-BUF-3.5")
        assert parsed.game_key == "2026-KC-BUF"

    def test_two_letter_team(self):
        parsed = parse_contract_id("KXNFL-26-WIN-SF")
        assert parsed is not None
        assert parsed.home_team == "SF"

    def test_three_letter_team(self):
        parsed = parse_contract_id("KXNFL-26-WIN-LAR")
        assert parsed is not None
        assert parsed.home_team == "LAR"


# ============================================================
# GamePrediction
# ============================================================


class TestGamePrediction:
    """GamePrediction dataclass."""

    def test_away_win_probability(self):
        pred = GamePrediction(
            home_team="KC",
            away_team="BUF",
            season=2026,
            week=5,
            home_spread=-3.0,
            total_points=48.5,
            home_win_probability=0.65,
            spread_std=10.0,
            total_std=8.0,
            features={},
        )
        assert pred.away_win_probability == pytest.approx(0.35)


# ============================================================
# Strategy with mock model
# ============================================================


def _mock_model(
    home_spread: float = -3.0,
    total_points: float = 48.5,
    home_win_prob: float = 0.65,
    spread_std: float = 10.0,
    total_std: float = 8.0,
) -> MagicMock:
    """Create a mock GameOutcomeModel."""
    model = MagicMock()
    model.predict.return_value = {
        "home_spread": home_spread,
        "total_points": total_points,
        "home_win_probability": home_win_prob,
        "spread_std": spread_std,
        "total_std": total_std,
    }
    model.feature_extractor = MagicMock()
    model.feature_extractor.extract_game_features.return_value = {"feat_1": 1.0}
    model.feature_names = [f"feat_{i}" for i in range(382)]
    return model


def _market_data(
    last_price: int = 55,
    yes_ask: int = 56,
    volume: int = 100,
) -> MarketData:
    return MarketData(
        contract_id="KXNFL-26-SPREAD-KC-BUF-3.5",
        last_price_cents=last_price,
        yes_bid_cents=last_price - 1,
        yes_ask_cents=yes_ask,
        volume_24h=volume,
        open_interest=0,
    )


def _orderbook(bid: int = 54, ask: int = 56) -> OrderBook:
    return OrderBook(
        contract_id="KXNFL-26-SPREAD-KC-BUF-3.5",
        yes_bids=[OrderBookLevel(price_cents=bid, quantity=50)],
        yes_asks=[OrderBookLevel(price_cents=ask, quantity=50)],
    )


class TestDirectionalStrategy:
    """End-to-end strategy tests with mock model."""

    @pytest.fixture
    def strategy(self):
        config = DirectionalConfig(
            min_edge_cents=5,
            bankroll_cents=100_000,
            use_meta_labeling=False,  # Simplify for unit tests
            kelly_fraction_cap=0.25,
        )
        # home_spread=-7.0 with line=3.5: CDF(-7+3.5, 0, 10) = CDF(-3.5, 0, 10) ≈ 0.36
        # But use a large home advantage to get >55% coverage probability:
        # home_spread=10.0 with line=3.5: CDF(10+3.5, 0, 10) = CDF(13.5, 0, 10) ≈ 0.91
        # At market=55, edge = 91-55 = 36¢ (well above 5¢ threshold)
        model = _mock_model(home_spread=10.0, home_win_prob=0.70)
        strat = DirectionalStrategy(config=config, model=model)
        strat.set_week(2026, 5)
        return strat

    async def test_generates_buy_signal_on_edge(self, strategy):
        """Positive edge generates a YES buy signal."""
        md = _market_data(last_price=55)
        ob = _orderbook(bid=54, ask=56)
        signals = await strategy.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5", md, ob,
        )
        assert len(signals) == 1
        sig = signals[0]
        assert sig.side == Side.YES
        assert sig.edge_cents > 0
        assert sig.quantity > 0
        assert sig.strategy == "directional"

    async def test_no_signal_below_edge_threshold(self):
        """No signal when edge is below threshold."""
        config = DirectionalConfig(
            min_edge_cents=5,
            use_meta_labeling=False,
            bankroll_cents=100_000,
        )
        model = _mock_model(home_win_prob=0.56)  # 56% → market at 55, edge = 1¢
        strat = DirectionalStrategy(config=config, model=model)
        strat.set_week(2026, 5)

        md = _market_data(last_price=55)
        ob = _orderbook(bid=54, ask=56)
        signals = await strat.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5", md, ob,
        )
        assert len(signals) == 0

    async def test_no_buy_signal_on_overpriced(self):
        """Generates NO signal when model prob < market price."""
        config = DirectionalConfig(
            min_edge_cents=5,
            use_meta_labeling=False,
            bankroll_cents=100_000,
        )
        # home_spread=-15.0 with line=3.5: CDF(-15+3.5, 0, 10) = CDF(-11.5, 0, 10) ≈ 0.125
        # At market=55, edge = 12-55 = -43¢ → buy NO
        model = _mock_model(home_spread=-15.0, home_win_prob=0.30)
        strat = DirectionalStrategy(config=config, model=model)
        strat.set_week(2026, 5)

        md = _market_data(last_price=55)
        ob = _orderbook(bid=54, ask=56)
        signals = await strat.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5", md, ob,
        )
        assert len(signals) == 1
        assert signals[0].side == Side.NO  # Buy NO = bet against

    async def test_unknown_contract_ignored(self, strategy):
        """Unknown contract ID produces no signal."""
        md = MarketData(
            contract_id="CRYPTO-BTC",
            last_price_cents=50,
            yes_bid_cents=49,
            yes_ask_cents=51,
            volume_24h=100,
            open_interest=0,
        )
        signals = await strategy.on_market_data("CRYPTO-BTC", md)
        assert len(signals) == 0

    async def test_win_contract_without_opponent_skipped(self, strategy):
        """Win contract can't be predicted without opponent in ID."""
        md = MarketData(
            contract_id="KXNFL-26-WIN-KC",
            last_price_cents=60,
            yes_bid_cents=59,
            yes_ask_cents=61,
            volume_24h=100,
            open_interest=0,
        )
        signals = await strategy.on_market_data("KXNFL-26-WIN-KC", md)
        assert len(signals) == 0  # No opponent → can't predict

    async def test_week_not_set_skips(self):
        """No signals before set_week is called."""
        config = DirectionalConfig(use_meta_labeling=False, bankroll_cents=100_000)
        model = _mock_model(home_win_prob=0.70)
        strat = DirectionalStrategy(config=config, model=model)
        # Don't call set_week

        md = _market_data(last_price=55)
        signals = await strat.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5", md,
        )
        assert len(signals) == 0

    async def test_cooldown_prevents_spam(self, strategy):
        """Cooldown prevents signals on same contract within window."""
        md = _market_data(last_price=55)
        ob = _orderbook(bid=54, ask=56)
        signals1 = await strategy.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5", md, ob,
        )
        assert len(signals1) == 1

        # Immediately try again — should be blocked by cooldown
        signals2 = await strategy.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5", md, ob,
        )
        assert len(signals2) == 0

    async def test_prediction_cached(self, strategy):
        """Prediction is cached and reused across contract types."""
        md = _market_data(last_price=55)
        ob = _orderbook(bid=54, ask=56)
        await strategy.on_market_data(
            "KXNFL-26-SPREAD-KC-BUF-3.5", md, ob,
        )
        # Model should have been called once
        assert strategy._model.predict.call_count == 1

        # Different contract type, same game — should use cache
        md2 = MarketData(
            contract_id="KXNFL-26-TOTAL-KC-BUF-47.5",
            last_price_cents=50,
            yes_bid_cents=49,
            yes_ask_cents=51,
            volume_24h=100,
            open_interest=0,
        )
        await strategy.on_market_data("KXNFL-26-TOTAL-KC-BUF-47.5", md2)
        assert strategy._model.predict.call_count == 1  # Still 1 — cached

    def test_set_week_clears_cache(self, strategy):
        """Changing week clears the prediction cache."""
        strategy._prediction_cache["2026-KC-BUF"] = GamePrediction(
            home_team="KC", away_team="BUF", season=2026, week=5,
            home_spread=-3.0, total_points=48.5, home_win_probability=0.65,
            spread_std=10.0, total_std=8.0, features={},
        )
        assert len(strategy._prediction_cache) == 1
        strategy.set_week(2026, 6)  # New week
        assert len(strategy._prediction_cache) == 0


# ============================================================
# Quantity sizing
# ============================================================


class TestQuantitySizing:
    """Kelly-based position sizing."""

    def test_kelly_sizing_basic(self):
        """Kelly formula produces reasonable quantities."""
        config = DirectionalConfig(
            bankroll_cents=100_000,
            kelly_fraction_cap=0.25,
            use_meta_labeling=False,
        )
        strat = DirectionalStrategy(config=config)
        qty = strat._compute_quantity(model_prob=0.70, price_cents=55, meta_prob=1.0)
        assert qty > 0
        assert qty <= config.max_contracts_per_signal

    def test_no_edge_no_quantity(self):
        """Kelly = 0 when model_prob equals price."""
        config = DirectionalConfig(bankroll_cents=100_000, use_meta_labeling=False)
        strat = DirectionalStrategy(config=config)
        qty = strat._compute_quantity(model_prob=0.55, price_cents=55, meta_prob=1.0)
        assert qty == 0

    def test_negative_edge_no_quantity(self):
        """No contracts when model says losing proposition."""
        config = DirectionalConfig(bankroll_cents=100_000, use_meta_labeling=False)
        strat = DirectionalStrategy(config=config)
        qty = strat._compute_quantity(model_prob=0.40, price_cents=55, meta_prob=1.0)
        assert qty == 0

    def test_meta_labeling_scales_quantity(self):
        """Meta-labeling probability scales Kelly fraction."""
        config = DirectionalConfig(
            bankroll_cents=10_000,  # Small bankroll so cap isn't hit
            kelly_fraction_cap=0.25,
            max_contracts_per_signal=1000,  # High cap to avoid hitting it
            use_meta_labeling=True,
        )
        strat = DirectionalStrategy(config=config)
        # Full meta confidence
        qty_full = strat._compute_quantity(model_prob=0.70, price_cents=55, meta_prob=0.8)
        # Half meta confidence (0.5 is in the linear zone: 0.4-0.6)
        qty_half = strat._compute_quantity(model_prob=0.70, price_cents=55, meta_prob=0.5)
        assert qty_full > qty_half

    def test_quantity_capped(self):
        """Quantity never exceeds max_contracts_per_signal."""
        config = DirectionalConfig(
            bankroll_cents=10_000_000,  # Huge bankroll
            max_contracts_per_signal=50,
            kelly_fraction_cap=0.25,
            use_meta_labeling=False,
        )
        strat = DirectionalStrategy(config=config)
        qty = strat._compute_quantity(model_prob=0.90, price_cents=10, meta_prob=1.0)
        assert qty <= 50

    def test_extreme_prices(self):
        """Edge cases: very high and very low prices."""
        config = DirectionalConfig(bankroll_cents=100_000, use_meta_labeling=False)
        strat = DirectionalStrategy(config=config)
        # Very cheap contract (high payout ratio)
        qty_cheap = strat._compute_quantity(model_prob=0.15, price_cents=5, meta_prob=1.0)
        assert qty_cheap >= 0
        # Very expensive contract
        qty_expensive = strat._compute_quantity(model_prob=0.98, price_cents=95, meta_prob=1.0)
        assert qty_expensive >= 0


# ============================================================
# Strategy name
# ============================================================


class TestStrategyName:
    def test_name(self):
        strat = DirectionalStrategy()
        assert strat.name == "directional"
