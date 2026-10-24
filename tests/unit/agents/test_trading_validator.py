"""Tests for pre-trade signal validation agent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nfl_analytics.agents.trading_validator import (
    TradingValidator,
    ValidationResult,
    ValidatorConfig,
)
from nfl_analytics.markets.base import Side
from nfl_analytics.risk.feature_health import FEATURE_GROUPS, HealthCheckResult
from nfl_analytics.risk.risk_manager import RiskManager
from nfl_analytics.trading.signal import Signal

# ============================================================
# Helpers
# ============================================================


def _signal(
    contract_id: str = "KXNFL-26-AFC-KC",
    quantity: int = 10,
    price_cents: int = 50,
    edge_cents: int = 5,
    model_prob: float = 0.55,
) -> Signal:
    return Signal(
        contract_id=contract_id,
        side=Side.YES,
        quantity=quantity,
        price_cents=price_cents,
        edge_cents=edge_cents,
        model_prob=model_prob,
        confidence=0.8,
        strategy="test",
    )


def _healthy_features() -> dict[str, float]:
    """Build a feature dict that passes all health checks."""
    features = {}
    for name, gdef in FEATURE_GROUPS.items():
        prefix = gdef["prefixes"][0]
        for i in range(gdef["min_count"]):
            features[f"{prefix}feat_{i}"] = 1.0
    while len(features) < 210:
        features[f"_pad_{len(features)}"] = 1.0
    return features


# ============================================================
# Valid signal passes
# ============================================================


class TestValidSignal:
    """Normal signals pass all checks."""

    def test_valid_signal_approved(self):
        """A well-formed signal passes validation."""
        validator = TradingValidator()
        result = validator.validate(_signal())
        assert result.approved
        assert result.reason == ""

    def test_valid_signal_no_warnings(self):
        """A signal with small edge produces no warnings."""
        validator = TradingValidator()
        result = validator.validate(_signal(edge_cents=5))
        assert result.approved
        assert result.warnings == []

    def test_valid_signal_all_checks(self):
        """Signal with valid prefix, quantity, price, prob all pass."""
        validator = TradingValidator()
        sig = _signal(
            contract_id="KXNFL-26-SB",
            quantity=50,
            price_cents=62,
            edge_cents=7,
            model_prob=0.69,
        )
        result = validator.validate(sig)
        assert result.approved


# ============================================================
# Edge too large
# ============================================================


class TestEdgeTooLarge:
    """Suspiciously large edge blocks signal."""

    def test_edge_exceeds_max(self):
        """Edge above max_edge_cents blocks the signal."""
        validator = TradingValidator()
        result = validator.validate(_signal(edge_cents=35))
        assert not result.approved
        assert "data error" in result.reason

    def test_negative_edge_exceeds_max(self):
        """Large negative edge also blocks."""
        validator = TradingValidator()
        result = validator.validate(_signal(edge_cents=-35))
        assert not result.approved

    def test_borderline_edge_warns(self):
        """Edge at 25 (>20 but <=30) warns but approves."""
        validator = TradingValidator()
        result = validator.validate(_signal(edge_cents=25))
        assert result.approved
        assert len(result.warnings) == 1
        assert "Large edge" in result.warnings[0]


# ============================================================
# Price out of range
# ============================================================


class TestPriceRange:
    """Price must be 1-99."""

    def test_price_zero(self):
        """Price 0 blocks."""
        validator = TradingValidator()
        sig = Signal(
            contract_id="KXNFL-26-TEST",
            side=Side.YES,
            quantity=10,
            price_cents=0,
            edge_cents=5,
            model_prob=0.55,
            confidence=0.8,
            strategy="test",
        )
        result = validator.validate(sig)
        assert not result.approved
        assert "below minimum" in result.reason

    def test_price_100(self):
        """Price 100 blocks."""
        validator = TradingValidator()
        sig = Signal(
            contract_id="KXNFL-26-TEST",
            side=Side.YES,
            quantity=10,
            price_cents=100,
            edge_cents=5,
            model_prob=0.55,
            confidence=0.8,
            strategy="test",
        )
        result = validator.validate(sig)
        assert not result.approved
        assert "above maximum" in result.reason


# ============================================================
# Unknown prefix
# ============================================================


class TestUnknownPrefix:
    """Unknown contract prefix blocks signal."""

    def test_unknown_prefix_blocks(self):
        """Contract with unrecognized prefix is blocked."""
        validator = TradingValidator()
        result = validator.validate(_signal(contract_id="UNKNOWN-01"))
        assert not result.approved
        assert "Unknown contract prefix" in result.reason

    def test_known_nba_prefix_passes(self):
        """NBA prefix is allowed when in known_prefixes."""
        validator = TradingValidator()
        result = validator.validate(_signal(contract_id="KXNBA-26-FINALS"))
        assert result.approved

    def test_empty_prefixes_allows_all(self):
        """Empty known_prefixes list allows any prefix."""
        config = ValidatorConfig(known_prefixes=[])
        validator = TradingValidator(config=config)
        result = validator.validate(_signal(contract_id="ANYTHING-01"))
        assert result.approved


# ============================================================
# Excessive quantity
# ============================================================


class TestExcessiveQuantity:
    """Quantity above max blocks signal."""

    def test_quantity_over_max(self):
        """Quantity exceeding max_quantity blocks."""
        validator = TradingValidator()
        result = validator.validate(_signal(quantity=300))
        assert not result.approved
        assert "Quantity 300 exceeds maximum" in result.reason


# ============================================================
# Model probability bounds
# ============================================================


class TestModelProbBounds:
    """Model probability must be within [min, max]."""

    def test_prob_too_low(self):
        """Model prob below min blocks."""
        validator = TradingValidator()
        result = validator.validate(_signal(model_prob=0.001))
        assert not result.approved
        assert "below minimum" in result.reason

    def test_prob_too_high(self):
        """Model prob above max blocks."""
        validator = TradingValidator()
        result = validator.validate(_signal(model_prob=0.999))
        assert not result.approved
        assert "above maximum" in result.reason


# ============================================================
# Batch validation
# ============================================================


class TestBatchValidation:
    """Batch validation returns one result per signal."""

    def test_batch_mixed(self):
        """Batch with valid and invalid signals."""
        validator = TradingValidator()
        signals = [
            _signal(edge_cents=5),  # Valid
            _signal(edge_cents=35),  # Too large edge
            _signal(contract_id="BAD-01"),  # Unknown prefix
        ]
        results = validator.validate_batch(signals)
        assert len(results) == 3
        assert results[0].approved
        assert not results[1].approved
        assert not results[2].approved


# ============================================================
# Disabled validator
# ============================================================


class TestDisabledValidator:
    """Disabled validator passes everything."""

    def test_disabled_approves_all(self):
        """When disabled, even bad signals pass."""
        config = ValidatorConfig(enabled=False)
        validator = TradingValidator(config=config)
        result = validator.validate(_signal(edge_cents=100))
        assert result.approved


# ============================================================
# Summary stats
# ============================================================


class TestSummary:
    """Validator tracks statistics."""

    def test_summary_counts(self):
        """Summary reflects validation outcomes."""
        validator = TradingValidator()
        validator.validate(_signal(edge_cents=5))  # Approved
        validator.validate(_signal(edge_cents=35))  # Blocked
        validator.validate(_signal(edge_cents=25))  # Approved with warning
        s = validator.summary()
        assert s["total_validated"] == 3
        assert s["approved"] == 2
        assert s["blocked"] == 1
        assert s["warnings"] == 1


# ============================================================
# Risk manager integration
# ============================================================


class TestRiskManagerIntegration:
    """Integration with RiskManager for feature health checks."""

    def test_features_required_but_missing(self):
        """Blocks when features required but not provided."""
        rm = RiskManager(bankroll_cents=100_000)
        validator = TradingValidator(risk_manager=rm)
        result = validator.validate(_signal(), features=None)
        assert not result.approved
        assert "Features required" in result.reason

    def test_features_healthy_passes(self):
        """Passes when features are healthy."""
        rm = RiskManager(bankroll_cents=100_000)
        validator = TradingValidator(risk_manager=rm)
        result = validator.validate(_signal(), features=_healthy_features())
        assert result.approved

    def test_features_degraded_warns(self):
        """Warns when features are degraded but still approves."""
        rm = RiskManager(bankroll_cents=100_000)
        validator = TradingValidator(risk_manager=rm)
        # Remove critical QBR features
        features = _healthy_features()
        features = {k: v for k, v in features.items() if not k.startswith("home_qbr_")}
        result = validator.validate(_signal(), features=features)
        assert result.approved  # Advisory only — warns but doesn't block
        assert any("Feature health" in w for w in result.warnings)
