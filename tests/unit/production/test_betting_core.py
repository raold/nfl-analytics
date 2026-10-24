"""
Unit tests for production betting core modules.

Tests kelly_sizing.py, kelly_sizing_enhanced.py, and value_bet_detector.py
without any database or external service dependencies.

Covers:
    - Kelly criterion math (full Kelly, fractional Kelly, edge cases)
    - Odds conversion (American to decimal and implied probability)
    - Value bet detection (edge calculation, threshold filtering, confidence)
    - Enhanced Kelly (edge decay, meta-labeling multiplier, regime adjustments)
    - Bankroll management (max bet caps, reserve constraints)
"""

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nfl_analytics.production.kelly_sizing import (
    american_odds_to_implied_prob,
    american_to_decimal_odds,
    decimal_to_american_odds,
    estimate_edge,
    kelly_criterion,
    optimal_bet_size,
    simulate_kelly_growth,
)
from nfl_analytics.production.kelly_sizing_enhanced import (
    BetOpportunity,
    CVaRCalculator,
    EdgeDecayFunction,
    EnhancedKellySizer,
    PortfolioConstraints,
)
from nfl_analytics.production.value_bet_detector import (
    BetType,
    Confidence,
    ValueBet,
    ValueBetDetector,
)


# ============================================================================
# Kelly Sizing - Odds Conversion
# ============================================================================


class TestOddsConversion:
    """Tests for American-to-decimal odds conversion and implied probability."""

    def test_negative_american_to_decimal(self):
        """Standard -110 odds should produce b = 100/110 = 0.909..."""
        dec = american_to_decimal_odds(-110)
        assert dec == pytest.approx(100 / 110, rel=1e-6)

    def test_positive_american_to_decimal(self):
        """+150 odds should produce b = 150/100 = 1.5."""
        dec = american_to_decimal_odds(150)
        assert dec == pytest.approx(1.5, rel=1e-6)

    def test_heavy_favorite_odds(self):
        """-300 odds (heavy favorite) should produce b = 100/300 = 0.333..."""
        dec = american_to_decimal_odds(-300)
        assert dec == pytest.approx(100 / 300, rel=1e-6)

    def test_implied_prob_negative_odds(self):
        """-110 implies 110/(110+100) = 52.38% probability."""
        prob = american_odds_to_implied_prob(-110)
        assert prob == pytest.approx(110 / 210, rel=1e-6)

    def test_implied_prob_positive_odds(self):
        """+200 implies 100/(200+100) = 33.33% probability."""
        prob = american_odds_to_implied_prob(200)
        assert prob == pytest.approx(100 / 300, rel=1e-6)

    def test_decimal_to_american_positive(self):
        """Decimal odds >= 1.0 map to positive American odds."""
        american = decimal_to_american_odds(1.5)
        assert american == 150

    def test_decimal_to_american_negative(self):
        """Decimal odds < 1.0 map to negative American odds."""
        american = decimal_to_american_odds(100 / 110)
        assert american == -110


# ============================================================================
# Kelly Sizing - Core Formula
# ============================================================================


class TestKellyCriterion:
    """Tests for the Kelly criterion bet sizing formula."""

    def test_quarter_kelly_scales_correctly(self):
        """Quarter Kelly (0.25) should be exactly 25% of full Kelly."""
        full = kelly_criterion(win_prob=0.55, decimal_odds=100 / 110, kelly_fraction=1.0,
                               max_bet_fraction=1.0)
        quarter = kelly_criterion(win_prob=0.55, decimal_odds=100 / 110, kelly_fraction=0.25,
                                  max_bet_fraction=1.0)
        assert quarter == pytest.approx(full * 0.25, rel=1e-6)

    def test_full_kelly_at_minus_110_55pct(self):
        """
        At -110 odds with 55% win probability:
            b = 10/11, p = 0.55, q = 0.45
            f* = (b*p - q) / b = (0.5 - 0.45) / (10/11) = 0.05 * 11/10 = 0.055
        """
        b = 10 / 11
        expected_full_kelly = (b * 0.55 - 0.45) / b  # = 0.055
        actual = kelly_criterion(win_prob=0.55, decimal_odds=b, kelly_fraction=1.0,
                                 max_bet_fraction=1.0)
        assert actual == pytest.approx(expected_full_kelly, rel=1e-6)
        assert actual == pytest.approx(0.055, rel=1e-3)

    def test_negative_kelly_returns_zero(self):
        """When model probability is below breakeven, Kelly should be zero (no bet)."""
        # At -110, breakeven is ~52.38%. Using 48% should yield negative Kelly => 0.
        result = kelly_criterion(win_prob=0.48, decimal_odds=100 / 110, kelly_fraction=1.0,
                                 max_bet_fraction=1.0)
        assert result == 0.0

    def test_exactly_breakeven_returns_zero(self):
        """At exactly breakeven probability, Kelly fraction should be zero."""
        # Breakeven for -110: p = 110/210 = 0.52381...
        breakeven_prob = 110 / 210
        result = kelly_criterion(win_prob=breakeven_prob, decimal_odds=100 / 110,
                                 kelly_fraction=1.0, max_bet_fraction=1.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_max_bet_cap_enforcement(self):
        """Kelly bet should never exceed max_bet_fraction regardless of edge."""
        # 70% win probability at +200 odds gives very large Kelly
        result = kelly_criterion(win_prob=0.70, decimal_odds=2.0, kelly_fraction=1.0,
                                 max_bet_fraction=0.05)
        assert result <= 0.05

    def test_half_kelly_fraction(self):
        """Half Kelly (0.5) should be exactly 50% of full Kelly."""
        full = kelly_criterion(win_prob=0.60, decimal_odds=1.0, kelly_fraction=1.0,
                               max_bet_fraction=1.0)
        half = kelly_criterion(win_prob=0.60, decimal_odds=1.0, kelly_fraction=0.5,
                               max_bet_fraction=1.0)
        assert half == pytest.approx(full * 0.5, rel=1e-6)


# ============================================================================
# Kelly Sizing - optimal_bet_size (integrated function)
# ============================================================================


class TestOptimalBetSize:
    """Tests for the optimal_bet_size function that combines edge + Kelly + bankroll."""

    def test_positive_edge_recommends_bet(self):
        """A 55% model prob at -110 has sufficient edge to recommend a bet."""
        result = optimal_bet_size(
            model_prob=0.55,
            market_odds=-110,
            bankroll=10000,
            kelly_fraction=0.25,
            max_bet_fraction=0.05,
            min_edge=0.01,
        )
        assert result["should_bet"] is True
        assert result["edge"] > 0.01
        assert 0 < result["bet_amount"] <= 500  # max 5% of $10K

    def test_insufficient_edge_no_bet(self):
        """When model probability barely exceeds implied, min_edge blocks the bet."""
        # At -110, implied = 52.38%. Model at 53% gives edge ~ 0.6%.
        result = optimal_bet_size(
            model_prob=0.53,
            market_odds=-110,
            bankroll=10000,
            kelly_fraction=0.25,
            max_bet_fraction=0.05,
            min_edge=0.02,  # Require 2% edge
        )
        assert result["should_bet"] is False
        assert "Insufficient edge" in result["reason"]

    def test_bet_amount_scales_with_bankroll(self):
        """Doubling the bankroll should double the bet amount (same fraction)."""
        result_10k = optimal_bet_size(
            model_prob=0.58, market_odds=-110, bankroll=10000,
            kelly_fraction=0.25, min_edge=0.01,
        )
        result_20k = optimal_bet_size(
            model_prob=0.58, market_odds=-110, bankroll=20000,
            kelly_fraction=0.25, min_edge=0.01,
        )
        assert result_10k["should_bet"] is True
        assert result_20k["should_bet"] is True
        assert result_20k["bet_amount"] == pytest.approx(
            result_10k["bet_amount"] * 2, rel=1e-6
        )

    def test_bet_fraction_capped_at_max(self):
        """Even with enormous edge, bet_fraction should not exceed max_bet_fraction."""
        result = optimal_bet_size(
            model_prob=0.80,
            market_odds=+200,
            bankroll=10000,
            kelly_fraction=1.0,  # Full Kelly
            max_bet_fraction=0.05,
            min_edge=0.01,
        )
        assert result["should_bet"] is True
        assert result["bet_fraction"] <= 0.05


# ============================================================================
# Kelly Sizing - Bankroll Simulation
# ============================================================================


class TestSimulateKellyGrowth:
    """Tests for the Monte Carlo bankroll simulation."""

    def test_simulation_returns_expected_keys(self):
        """Simulation result should contain all required metric keys."""
        result = simulate_kelly_growth(
            win_prob=0.55, decimal_odds=100 / 110,
            initial_bankroll=10000, n_bets=100, kelly_fraction=0.25, seed=42,
        )
        expected_keys = {
            "final_bankroll", "total_return", "roi", "n_bets",
            "wins", "losses", "win_rate", "bet_fraction",
            "max_drawdown_pct", "sharpe_ratio", "bankroll_history",
        }
        assert expected_keys.issubset(result.keys())

    def test_simulation_deterministic_with_seed(self):
        """Same seed should produce identical simulation results."""
        r1 = simulate_kelly_growth(
            win_prob=0.55, decimal_odds=1.0, initial_bankroll=10000,
            n_bets=200, kelly_fraction=0.25, seed=123,
        )
        r2 = simulate_kelly_growth(
            win_prob=0.55, decimal_odds=1.0, initial_bankroll=10000,
            n_bets=200, kelly_fraction=0.25, seed=123,
        )
        assert r1["final_bankroll"] == pytest.approx(r2["final_bankroll"], rel=1e-10)

    def test_simulation_bankroll_history_length(self):
        """Bankroll history should have n_bets + 1 entries (initial + each bet)."""
        result = simulate_kelly_growth(
            win_prob=0.55, decimal_odds=1.0, initial_bankroll=10000,
            n_bets=50, kelly_fraction=0.25, seed=42,
        )
        assert len(result["bankroll_history"]) == 51


# ============================================================================
# Value Bet Detector
# ============================================================================


class TestValueBetDetector:
    """Tests for the ValueBetDetector class."""

    def _make_prediction(self, model_prob=0.58, odds=-110, **overrides):
        """Helper to create a prediction dict."""
        pred = {
            "game_id": "2025_16_KC_BAL",
            "home_team": "BAL",
            "away_team": "KC",
            "model_prob": model_prob,
            "spread_line": "-3.5",
            "spread_odds": odds,
        }
        pred.update(overrides)
        return pred

    def test_detects_value_when_edge_exceeds_threshold(self):
        """A prediction with edge above min_edge should be detected."""
        detector = ValueBetDetector(min_edge=0.03, use_llm_explanations=False)
        # 58% model prob at -110 (implied 52.38%) => edge = 5.62%
        predictions = [self._make_prediction(model_prob=0.58)]
        bets = detector.detect_value_bets(predictions)
        assert len(bets) == 1
        assert bets[0].edge >= 0.03

    def test_no_bet_below_threshold(self):
        """A prediction with edge below min_edge should be filtered out."""
        detector = ValueBetDetector(min_edge=0.05, use_llm_explanations=False)
        # 53% model prob at -110 => edge = ~0.6% (well below 5%)
        predictions = [self._make_prediction(model_prob=0.53)]
        bets = detector.detect_value_bets(predictions)
        assert len(bets) == 0

    def test_bets_sorted_by_edge_descending(self):
        """Multiple value bets should be sorted by edge, highest first."""
        detector = ValueBetDetector(min_edge=0.02, use_llm_explanations=False)
        predictions = [
            self._make_prediction(model_prob=0.56, game_id="game_low"),
            self._make_prediction(model_prob=0.65, game_id="game_high"),
            self._make_prediction(model_prob=0.60, game_id="game_mid"),
        ]
        bets = detector.detect_value_bets(predictions)
        edges = [b.edge for b in bets]
        assert edges == sorted(edges, reverse=True)

    def test_max_bets_per_week_limit(self):
        """Detector should cap results at max_bets_per_week."""
        detector = ValueBetDetector(
            min_edge=0.02, max_bets_per_week=2, use_llm_explanations=False
        )
        predictions = [
            self._make_prediction(model_prob=0.58, game_id=f"game_{i}")
            for i in range(5)
        ]
        bets = detector.detect_value_bets(predictions)
        assert len(bets) <= 2

    def test_confidence_high_for_large_edge(self):
        """Edge >= 5% with reasonable probability should yield HIGH confidence."""
        detector = ValueBetDetector(min_edge=0.02, use_llm_explanations=False)
        # 60% model prob at -110 => edge ~ 7.6%
        predictions = [self._make_prediction(model_prob=0.60)]
        bets = detector.detect_value_bets(predictions)
        assert len(bets) == 1
        assert bets[0].confidence == Confidence.HIGH

    def test_confidence_low_for_extreme_probability(self):
        """Very high model probability (outside 0.35-0.75) should yield LOW confidence."""
        detector = ValueBetDetector(min_edge=0.02, use_llm_explanations=False)
        # 80% model prob at -110 => huge edge but extreme probability
        predictions = [self._make_prediction(model_prob=0.80)]
        bets = detector.detect_value_bets(predictions)
        assert len(bets) == 1
        assert bets[0].confidence == Confidence.LOW

    def test_missing_vegas_line_uses_default(self):
        """When no market_odds dict is provided, detector uses prediction's spread_odds."""
        detector = ValueBetDetector(min_edge=0.02, use_llm_explanations=False)
        pred = self._make_prediction(model_prob=0.60, odds=-105)
        bets = detector.detect_value_bets([pred], market_odds=None)
        assert len(bets) == 1
        # At -105, implied = 105/205 = 51.22%, edge = 60% - 51.22% = 8.78%
        assert bets[0].edge == pytest.approx(0.60 - 105 / 205, rel=1e-3)

    def test_value_bet_is_actionable(self):
        """A bet with edge >= 2% and non-LOW confidence should be actionable."""
        bet = ValueBet(
            game_id="test", home_team="BAL", away_team="KC", game_time=None,
            bet_type=BetType.SPREAD, bet_side="BAL -3.5", odds=-110,
            model_prob=0.58, market_implied=0.5238, edge=0.056,
            expected_value=5.0, confidence=Confidence.MEDIUM,
            kelly_fraction=0.05, recommended_size=0.0125,
        )
        assert bet.is_actionable is True

    def test_value_bet_not_actionable_low_confidence(self):
        """A bet with LOW confidence should NOT be actionable, even with decent edge."""
        bet = ValueBet(
            game_id="test", home_team="BAL", away_team="KC", game_time=None,
            bet_type=BetType.SPREAD, bet_side="BAL -3.5", odds=-110,
            model_prob=0.85, market_implied=0.5238, edge=0.03,
            expected_value=2.0, confidence=Confidence.LOW,
            kelly_fraction=0.02, recommended_size=0.005,
        )
        assert bet.is_actionable is False

    def test_llm_explanations_disabled_uses_fallback(self):
        """When use_llm_explanations=False, fallback text explanation is generated."""
        detector = ValueBetDetector(min_edge=0.02, use_llm_explanations=False)
        predictions = [self._make_prediction(model_prob=0.60)]
        bets = detector.detect_value_bets(predictions)
        assert len(bets) == 1
        # Fallback explanation should be non-empty and mention edge
        assert bets[0].explanation == ""  # LLM disabled => no explanation in detect path
        # But fallback can be called directly
        explanation = detector._generate_fallback_explanation(bets[0])
        assert "edge" in explanation.lower() or "Edge" in explanation

    def test_weekly_report_no_bets(self):
        """Weekly report with no bets should return a meaningful message."""
        detector = ValueBetDetector(use_llm_explanations=False)
        report = detector.generate_weekly_report([])
        assert "No value bets" in report

    def test_to_dict_serialization(self):
        """ValueBet.to_dict() should produce a JSON-serializable dictionary."""
        bet = ValueBet(
            game_id="2025_16_KC_BAL", home_team="BAL", away_team="KC",
            game_time=None, bet_type=BetType.SPREAD, bet_side="BAL -3.5",
            odds=-110, model_prob=0.58, market_implied=0.5238, edge=0.056,
            expected_value=5.0, confidence=Confidence.MEDIUM,
            kelly_fraction=0.05, recommended_size=0.0125,
            key_factors=["Solid model edge (3%+)"],
            risks=["Edge near threshold"],
        )
        d = bet.to_dict()
        assert d["game_id"] == "2025_16_KC_BAL"
        assert d["matchup"] == "KC @ BAL"
        assert d["confidence"] == "medium"
        assert isinstance(d["edge"], float)


# ============================================================================
# Enhanced Kelly - Edge Decay
# ============================================================================


class TestEdgeDecay:
    """Tests for EdgeDecayFunction confidence and time decay."""

    def test_high_confidence_no_decay(self):
        """Confidence >= 0.95 should return edge unchanged."""
        edge = EdgeDecayFunction.confidence_decay(0.05, confidence=0.96)
        assert edge == pytest.approx(0.05, rel=1e-6)

    def test_lower_confidence_reduces_edge(self):
        """Confidence < 0.95 should produce a smaller (more conservative) edge."""
        full_edge = 0.05
        decayed = EdgeDecayFunction.confidence_decay(full_edge, confidence=0.70)
        assert decayed < full_edge
        assert decayed > 0  # Should still be positive

    def test_time_decay_far_from_game(self):
        """More than 24 hours out, time decay should have no effect."""
        edge = EdgeDecayFunction.time_decay(0.05, minutes_to_game=2000)
        assert edge == pytest.approx(0.05, rel=1e-6)

    def test_time_decay_close_to_game(self):
        """Within 24 hours, edge should decay (market becomes more efficient)."""
        edge_far = EdgeDecayFunction.time_decay(0.05, minutes_to_game=1440)
        edge_close = EdgeDecayFunction.time_decay(0.05, minutes_to_game=60)
        assert edge_close < edge_far


# ============================================================================
# Enhanced Kelly - Meta-Labeling Multiplier
# ============================================================================


class TestMetaLabelingMultiplier:
    """Tests for BetOpportunity.meta_kelly_multiplier property."""

    def test_low_meta_prob_blocks_bet(self):
        """meta_probability < 0.4 should return multiplier of 0 (no bet)."""
        bet = BetOpportunity(
            game_id="test", side="home", model_prob=0.58, market_odds=-110,
            edge=0.05, confidence=0.85, meta_probability=0.30,
        )
        assert bet.meta_kelly_multiplier == 0.0

    def test_high_meta_prob_full_kelly(self):
        """meta_probability > 0.6 should return multiplier of 1.0 (full Kelly)."""
        bet = BetOpportunity(
            game_id="test", side="home", model_prob=0.58, market_odds=-110,
            edge=0.05, confidence=0.85, meta_probability=0.75,
        )
        assert bet.meta_kelly_multiplier == 1.0

    def test_midrange_meta_prob_interpolates(self):
        """meta_probability at 0.5 (midpoint of 0.4-0.6) should return 0.5."""
        bet = BetOpportunity(
            game_id="test", side="home", model_prob=0.58, market_odds=-110,
            edge=0.05, confidence=0.85, meta_probability=0.50,
        )
        assert bet.meta_kelly_multiplier == pytest.approx(0.5, rel=1e-6)

    def test_default_meta_probability_is_one(self):
        """Default meta_probability=1.0 should give multiplier of 1.0."""
        bet = BetOpportunity(
            game_id="test", side="home", model_prob=0.58, market_odds=-110,
            edge=0.05, confidence=0.85,
        )
        assert bet.meta_probability == 1.0
        assert bet.meta_kelly_multiplier == 1.0


# ============================================================================
# Enhanced Kelly - EnhancedKellySizer
# ============================================================================


class TestEnhancedKellySizer:
    """Tests for the EnhancedKellySizer class."""

    def _make_bet(self, model_prob=0.58, market_odds=-110, edge=0.05,
                  confidence=0.85, meta_probability=1.0):
        return BetOpportunity(
            game_id="KC_vs_BAL", side="KC -3", model_prob=model_prob,
            market_odds=market_odds, edge=edge, confidence=confidence,
            meta_probability=meta_probability,
        )

    def test_base_kelly_positive_for_positive_ev(self):
        """A +EV bet should produce a positive base Kelly fraction."""
        sizer = EnhancedKellySizer(kelly_multiplier=0.25, edge_decay=False,
                                   cvar_constraint=False)
        bet = self._make_bet(model_prob=0.58)
        kelly = sizer.calculate_base_kelly(bet, use_meta_labeling=False)
        assert kelly > 0

    def test_base_kelly_zero_for_negative_ev(self):
        """A -EV bet should produce zero Kelly fraction."""
        sizer = EnhancedKellySizer(edge_decay=False, cvar_constraint=False)
        bet = self._make_bet(model_prob=0.40)  # Below breakeven
        kelly = sizer.calculate_base_kelly(bet, use_meta_labeling=False)
        assert kelly == 0.0

    def test_edge_decay_reduces_kelly(self):
        """Enabling edge decay should reduce Kelly vs disabled decay."""
        bet = self._make_bet(confidence=0.70)
        sizer_no_decay = EnhancedKellySizer(edge_decay=False, cvar_constraint=False)
        sizer_decay = EnhancedKellySizer(edge_decay=True, cvar_constraint=False)

        kelly_no_decay = sizer_no_decay.calculate_base_kelly(bet, use_meta_labeling=False)
        kelly_decay = sizer_decay.calculate_base_kelly(bet, use_meta_labeling=False)
        assert kelly_decay < kelly_no_decay

    def test_regime_adjustment_high_volatility(self):
        """High volatility regime should reduce bet size by 30%."""
        sizer = EnhancedKellySizer(kelly_multiplier=0.25)
        base_size = 0.04
        adjusted = sizer.adjust_for_regime(base_size, volatility_regime="high")
        assert adjusted == pytest.approx(base_size * 0.7, rel=1e-6)

    def test_regime_adjustment_losing_streak(self):
        """A 3+ game losing streak should reduce bet size by 20%."""
        sizer = EnhancedKellySizer(kelly_multiplier=0.25)
        base_size = 0.04
        adjusted = sizer.adjust_for_regime(
            base_size, volatility_regime="normal",
            streak_info={"losing_streak": 4},
        )
        assert adjusted == pytest.approx(base_size * 0.8, rel=1e-6)

    def test_regime_adjustment_capped_at_max(self):
        """Regime adjustments should never exceed max_single_bet constraint."""
        constraints = PortfolioConstraints(max_single_bet=0.03)
        sizer = EnhancedKellySizer(kelly_multiplier=0.25, constraints=constraints)
        # Low volatility + winning streak would try to increase
        adjusted = sizer.adjust_for_regime(
            0.04, volatility_regime="low",
            streak_info={"winning_streak": 5},
        )
        assert adjusted <= 0.03

    def test_portfolio_optimization_without_cvar(self):
        """Portfolio optimization without CVaR should respect exposure limits."""
        sizer = EnhancedKellySizer(
            kelly_multiplier=0.25, edge_decay=False, cvar_constraint=False,
        )
        bets = [
            self._make_bet(model_prob=0.58, edge=0.05),
            self._make_bet(model_prob=0.60, edge=0.07),
        ]
        result = sizer.optimize_portfolio(bets, bankroll=10000)
        assert result["sizes"].sum() <= PortfolioConstraints().max_total_exposure
        assert all(s <= PortfolioConstraints().max_single_bet for s in result["sizes"])
        assert result["expected_value"] > 0

    def test_portfolio_empty_bets(self):
        """Empty bet list should return zero-value portfolio."""
        sizer = EnhancedKellySizer()
        result = sizer.optimize_portfolio([], bankroll=10000)
        assert result["sizes"] == []
        assert result["expected_value"] == 0

    def test_decimal_odds_property(self):
        """BetOpportunity.decimal_odds should correctly convert from American odds."""
        # -110: decimal = 1 + 100/110 = 1.909...
        bet_neg = self._make_bet(market_odds=-110)
        assert bet_neg.decimal_odds == pytest.approx(1 + 100 / 110, rel=1e-6)

        # +150: decimal = 1 + 150/100 = 2.5
        bet_pos = self._make_bet(market_odds=150)
        assert bet_pos.decimal_odds == pytest.approx(2.5, rel=1e-6)


# ============================================================================
# CVaR Calculator
# ============================================================================


class TestCVaRCalculator:
    """Tests for CVaR (Conditional Value at Risk) calculations."""

    def test_var_at_95_percent(self):
        """VaR at 95% should be the 5th percentile of returns."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.05, 10000)
        var = CVaRCalculator.calculate_var(returns, confidence_level=0.95)
        # 5th percentile of N(0.01, 0.05) ~ 0.01 - 1.645*0.05 = -0.07225
        assert var < 0  # Should be negative (a loss)
        assert var == pytest.approx(np.percentile(returns, 5), rel=1e-6)

    def test_cvar_worse_than_var(self):
        """CVaR (expected shortfall) should be more negative than VaR."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.05, 10000)
        var = CVaRCalculator.calculate_var(returns, 0.95)
        cvar = CVaRCalculator.calculate_cvar(returns, 0.95)
        assert cvar < var  # CVaR is the average of the worst losses


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_push_scenario_model_equals_implied(self):
        """When model prob equals implied prob, edge is zero => no bet."""
        implied = american_odds_to_implied_prob(-110)  # ~0.5238
        result = optimal_bet_size(
            model_prob=implied, market_odds=-110, bankroll=10000,
            kelly_fraction=0.25, min_edge=0.01,
        )
        assert result["should_bet"] is False

    def test_extreme_model_prediction_high(self):
        """Model predicting 95% win probability should still be bounded by max bet."""
        result = optimal_bet_size(
            model_prob=0.95, market_odds=-110, bankroll=10000,
            kelly_fraction=0.25, max_bet_fraction=0.05, min_edge=0.01,
        )
        assert result["should_bet"] is True
        assert result["bet_fraction"] <= 0.05
        assert result["bet_amount"] <= 500

    def test_extreme_model_prediction_low(self):
        """Model predicting 5% win probability should never recommend a bet at -110."""
        result = optimal_bet_size(
            model_prob=0.05, market_odds=-110, bankroll=10000,
            kelly_fraction=0.25, min_edge=0.01,
        )
        assert result["should_bet"] is False

    def test_edge_estimation(self):
        """estimate_edge should return model_prob - implied_prob."""
        edge = estimate_edge(model_prob=0.58, market_odds=-110)
        implied = american_odds_to_implied_prob(-110)
        assert edge == pytest.approx(0.58 - implied, rel=1e-6)

    def test_large_positive_odds(self):
        """Large underdog odds (+500) should work correctly through the pipeline."""
        result = optimal_bet_size(
            model_prob=0.30, market_odds=500, bankroll=10000,
            kelly_fraction=0.25, min_edge=0.01,
        )
        # Implied at +500: 100/600 = 16.67%. Model 30% => edge = 13.33%
        assert result["should_bet"] is True
        assert result["edge"] > 0.10
