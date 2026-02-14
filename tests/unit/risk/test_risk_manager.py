"""Tests for unified risk manager — orchestrates all risk layers."""

from __future__ import annotations

import pytest

from py.markets.base import Position, Side
from py.risk.feature_health import FEATURE_GROUPS, HealthConfig
from py.risk.kill_switch import KillSwitchConfig
from py.risk.position_limits import LimitConfig
from py.risk.risk_manager import RiskCheckResult, RiskManager, RiskManagerConfig
from py.trading.signal import Signal


def _signal(
    contract_id: str = "TEST-01",
    quantity: int = 10,
    price_cents: int = 50,
    edge_cents: int = 5,
) -> Signal:
    return Signal(
        contract_id=contract_id,
        side=Side.YES,
        quantity=quantity,
        price_cents=price_cents,
        edge_cents=edge_cents,
        model_prob=0.55,
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
    # Pad to exceed min_total_features (200)
    while len(features) < 210:
        features[f"_pad_{len(features)}"] = 1.0
    return features


def _position(contract_id: str = "TEST-01", quantity: int = 10) -> Position:
    return Position(
        contract_id=contract_id,
        side=Side.YES,
        quantity=quantity,
        avg_entry_cents=50,
    )


# ============================================================
# Happy path
# ============================================================


class TestHappyPath:
    """All layers pass → signal approved."""

    def test_clean_signal_approved(self):
        """Small signal with healthy features and no positions."""
        rm = RiskManager(bankroll_cents=100_000)
        sig = _signal()
        result = rm.check_signal(sig, features=_healthy_features(), positions=[])
        assert result.approved
        assert result.layer == "all_passed"

    def test_approved_without_features(self):
        """Signal approved when no features provided (skips health check)."""
        rm = RiskManager(bankroll_cents=100_000)
        sig = _signal()
        result = rm.check_signal(sig)
        assert result.approved

    def test_stats_tracked(self):
        """Risk manager tracks approval stats."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.check_signal(_signal(), features=_healthy_features())
        rm.check_signal(_signal(), features=_healthy_features())
        status = rm.status()
        assert status["stats"]["total_checks"] == 2
        assert status["stats"]["approved"] == 2


# ============================================================
# Feature health blocks (Layer 5)
# ============================================================


class TestFeatureHealthBlocking:
    """Layer 5 — degraded features block ALL trading."""

    def test_missing_critical_features_blocks(self):
        """Missing QBR features (critical) blocks the signal."""
        rm = RiskManager(bankroll_cents=100_000)
        features = _healthy_features()
        # Remove all QBR features
        features = {k: v for k, v in features.items() if not k.startswith("home_qbr_")}
        sig = _signal()
        result = rm.check_signal(sig, features=features)
        assert not result.approved
        assert result.layer == "feature_health"
        assert result.health_result is not None

    def test_health_check_cached(self):
        """Health check is cached between signals."""
        rm = RiskManager(bankroll_cents=100_000)
        features = _healthy_features()
        rm.check_signal(_signal(), features=features)  # First check
        # Second check without features uses cached result
        result = rm.check_signal(_signal())
        assert result.approved  # Cached healthy result

    def test_health_bypass_when_disabled(self):
        """Feature health can be disabled for testing."""
        config = RiskManagerConfig(enforce_feature_health=False)
        rm = RiskManager(bankroll_cents=100_000, config=config)
        # Pass garbage features — should be ignored
        result = rm.check_signal(_signal(), features={"single_feat": 1.0})
        assert result.approved


# ============================================================
# Kill switch blocks (Layer 6)
# ============================================================


class TestKillSwitchBlocking:
    """Layer 6 — kill switch blocks ALL trading."""

    def test_kill_switch_blocks_after_daily_loss(self):
        """Daily loss trigger blocks subsequent signals."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.update_equity(90_000)  # 10% loss → triggers
        assert rm.kill_switch.is_triggered

        result = rm.check_signal(_signal(), features=_healthy_features())
        assert not result.approved
        assert result.layer == "kill_switch"
        assert result.kill_switch_active

    def test_kill_switch_blocks_after_manual_trigger(self):
        """Manual trigger blocks signals."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.kill_switch.manual_trigger("suspicious")
        result = rm.check_signal(_signal())
        assert not result.approved
        assert "Kill switch active" in result.reason

    def test_reset_allows_trading(self):
        """Resetting kill switch allows trading again."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.update_equity(90_000)  # Trigger
        rm.reset_kill_switch()  # Re-arm
        result = rm.check_signal(_signal())
        assert result.approved

    def test_kill_switch_bypass(self):
        """Kill switch can be disabled for testing."""
        config = RiskManagerConfig(enforce_kill_switch=False)
        rm = RiskManager(bankroll_cents=100_000, config=config)
        rm.kill_switch.manual_trigger("test")
        result = rm.check_signal(_signal())
        assert result.approved  # Bypassed

    def test_api_errors_trigger_kill_switch(self):
        """Consecutive API errors trigger the kill switch."""
        rm = RiskManager(bankroll_cents=100_000)
        for _ in range(10):
            rm.record_api_error()
        assert rm.kill_switch.is_triggered
        result = rm.check_signal(_signal())
        assert not result.approved

    def test_api_success_resets_counter(self):
        """Successful API call resets error counter."""
        rm = RiskManager(bankroll_cents=100_000)
        for _ in range(9):
            rm.record_api_error()
        rm.record_api_success()
        rm.record_api_error()  # Back to 1
        assert not rm.kill_switch.is_triggered


# ============================================================
# Position limit blocks (Layers 1-3)
# ============================================================


class TestPositionLimitBlocking:
    """Layers 1-3 — position/exposure/correlation limits."""

    def test_large_position_reduced(self):
        """Oversized signal gets reduced by position limits."""
        rm = RiskManager(bankroll_cents=100_000)
        sig = _signal(quantity=200, price_cents=50)  # 10,000¢ > 5% of 100k
        result = rm.check_signal(sig, features=_healthy_features())
        assert result.approved
        assert result.reduced
        assert sig.quantity < 200

    def test_exposure_limit_blocks(self):
        """Full exposure blocks new signals."""
        rm = RiskManager(bankroll_cents=100_000)
        # Fill exposure to 25%
        positions = [_position(f"P-{i}", quantity=50) for i in range(10)]
        sig = _signal(quantity=10, price_cents=50)
        result = rm.check_signal(sig, positions=positions)
        assert not result.approved

    def test_position_limits_bypass(self):
        """Position limits can be disabled for testing."""
        config = RiskManagerConfig(enforce_position_limits=False)
        rm = RiskManager(bankroll_cents=100_000, config=config)
        sig = _signal(quantity=10000, price_cents=50)
        result = rm.check_signal(sig)
        assert result.approved


# ============================================================
# Layer ordering
# ============================================================


class TestLayerOrdering:
    """Risk layers execute in correct priority order."""

    def test_feature_health_checked_before_kill_switch(self):
        """Feature health (Layer 5) runs before kill switch (Layer 6)."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.kill_switch.manual_trigger("test")  # Kill switch active

        # Bad features AND kill switch — should report feature health
        bad_features = {"single_feat": 0.0}  # Far below minimum
        result = rm.check_signal(_signal(), features=bad_features)
        assert not result.approved
        assert result.layer == "feature_health"  # Feature health first

    def test_kill_switch_before_position_limits(self):
        """Kill switch (Layer 6) runs before position limits."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.kill_switch.manual_trigger("test")

        # Skip feature health, trigger kill switch
        config = RiskManagerConfig(enforce_feature_health=False)
        rm2 = RiskManager(bankroll_cents=100_000, config=config)
        rm2.kill_switch.manual_trigger("test")
        sig = _signal(quantity=10000)  # Would also fail position limits

        result = rm2.check_signal(sig)
        assert not result.approved
        assert result.layer == "kill_switch"  # Kill switch before position limits


# ============================================================
# Status and observability
# ============================================================


class TestStatus:
    """Risk manager status reporting."""

    def test_comprehensive_status(self):
        """Status includes all component states."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.check_signal(_signal(), features=_healthy_features())
        status = rm.status()

        assert "kill_switch" in status
        assert "feature_health" in status
        assert "stats" in status
        assert status["stats"]["total_checks"] == 1

    def test_blocks_by_layer_tracking(self):
        """Block reasons tracked per layer."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.kill_switch.manual_trigger("test")
        config = RiskManagerConfig(enforce_feature_health=False)
        rm_no_health = RiskManager(bankroll_cents=100_000, config=config)
        rm_no_health.kill_switch.manual_trigger("test")

        rm_no_health.check_signal(_signal())
        rm_no_health.check_signal(_signal())
        status = rm_no_health.status()
        assert status["stats"]["blocks_by_layer"].get("kill_switch", 0) == 2

    def test_bankroll_update(self):
        """Bankroll update propagates to position limits."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.update_bankroll(200_000)
        # 5% of 200,000 = 10,000. At 50¢ → 200 contracts
        sig = _signal(quantity=200, price_cents=50)
        result = rm.check_signal(sig)
        assert result.approved
        assert not result.reduced

    def test_new_trading_day(self):
        """New trading day resets daily P&L tracking."""
        rm = RiskManager(bankroll_cents=100_000)
        rm.update_equity(95_000)  # 5% daily loss
        rm.new_trading_day()
        assert rm.kill_switch.daily_pnl_pct == 0.0
