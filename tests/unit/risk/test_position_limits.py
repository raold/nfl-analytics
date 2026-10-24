"""Tests for position limit checker — Layers 1-3 of risk hierarchy."""

from __future__ import annotations

import pytest

from nfl_analytics.markets.base import Position, Side
from nfl_analytics.risk.position_limits import LimitCheckResult, LimitConfig, PositionLimitChecker
from nfl_analytics.trading.signal import Signal


def _signal(
    contract_id: str = "TEST-01",
    side: Side = Side.YES,
    quantity: int = 10,
    price_cents: int = 50,
    edge_cents: int = 5,
) -> Signal:
    """Helper to build test signals."""
    return Signal(
        contract_id=contract_id,
        side=side,
        quantity=quantity,
        price_cents=price_cents,
        edge_cents=edge_cents,
        model_prob=0.55,
        confidence=0.8,
        strategy="test",
    )


def _position(
    contract_id: str = "TEST-01",
    side: Side = Side.YES,
    quantity: int = 10,
    avg_entry_cents: int = 50,
) -> Position:
    """Helper to build test positions."""
    return Position(
        contract_id=contract_id,
        side=side,
        quantity=quantity,
        avg_entry_cents=avg_entry_cents,
    )


# ============================================================
# Layer 1: Single position size
# ============================================================


class TestSinglePositionLimit:
    """Layer 1 — max 5% of bankroll per position."""

    def test_small_signal_approved(self):
        """Signal well within limit passes cleanly."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        sig = _signal(quantity=5, price_cents=50)  # 250¢ = 0.25%
        result = checker.check(sig, [])
        assert result.approved
        assert not result.reduced

    def test_exact_limit_approved(self):
        """Signal exactly at the 5% limit passes."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        # 5% of 100,000 = 5,000¢. At 50¢/contract → 100 contracts
        sig = _signal(quantity=100, price_cents=50)
        result = checker.check(sig, [])
        assert result.approved

    def test_over_limit_reduced(self):
        """Signal exceeding limit gets quantity reduced."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        # 5% = 5,000¢. At 50¢/contract → max 100. Ask for 200.
        sig = _signal(quantity=200, price_cents=50)
        result = checker.check(sig, [])
        assert result.approved
        assert result.reduced
        assert sig.quantity <= 100

    def test_existing_position_counted(self):
        """Existing position is counted toward the limit."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        existing = _position(quantity=90, avg_entry_cents=50)  # 4,500¢ used
        sig = _signal(quantity=50, price_cents=50)  # Wants 2,500¢ more → total 7,000 > 5,000
        result = checker.check(sig, [existing])
        assert result.approved
        assert result.reduced
        assert sig.quantity <= 10  # Only room for 10 more contracts

    def test_fully_blocked_when_maxed_out(self):
        """Signal blocked when existing position already at limit."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        existing = _position(quantity=100, avg_entry_cents=50)  # Already at 5,000¢
        sig = _signal(quantity=10, price_cents=50)
        result = checker.check(sig, [existing])
        assert not result.approved
        assert result.layer == "position_size"


# ============================================================
# Layer 2: Total exposure
# ============================================================


class TestTotalExposure:
    """Layer 2 — max 25% total exposure across all positions."""

    def test_no_positions_approved(self):
        """Signal with no existing positions passes easily."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        sig = _signal(quantity=10, price_cents=50)  # 500¢ = 0.5%
        result = checker.check(sig, [])
        assert result.approved

    def test_exposure_near_limit_reduced(self):
        """Signal reduced when total exposure would exceed limit."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        # Current: 24% exposure (24,000¢)
        positions = [_position(f"POS-{i}", quantity=24, avg_entry_cents=100) for i in range(10)]
        # 10 × 24 × 100 = 24,000¢. Max = 25,000¢. Room for 1,000¢.
        sig = _signal(quantity=30, price_cents=50)  # Wants 1,500¢
        result = checker.check(sig, positions)
        assert result.approved
        assert result.reduced
        # Should be reduced to fit within remaining 1,000¢
        assert sig.quantity <= 20  # 20 × 50 = 1,000¢

    def test_exposure_limit_blocks(self):
        """Signal blocked when exposure is already at limit."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        # Already at 25% (25,000¢)
        positions = [_position(f"POS-{i}", quantity=50, avg_entry_cents=50) for i in range(10)]
        sig = _signal(quantity=10, price_cents=50)
        result = checker.check(sig, positions)
        assert not result.approved
        assert result.layer == "total_exposure"


# ============================================================
# Layer 3: Correlation group
# ============================================================


class TestCorrelationGroup:
    """Layer 3 — max 10% in correlated contract group."""

    def test_no_group_skips_check(self):
        """Contract not in any group skips correlation check."""
        checker = PositionLimitChecker(
            bankroll_cents=100_000,
            correlation_groups={"KC-WIN": "KC", "KC-SPREAD": "KC"},
        )
        sig = _signal(contract_id="BUF-WIN", quantity=10, price_cents=50)
        result = checker.check(sig, [])
        assert result.approved

    def test_group_within_limit(self):
        """Contract in group but within 10% limit passes."""
        checker = PositionLimitChecker(
            bankroll_cents=100_000,
            correlation_groups={"KC-WIN": "KC", "KC-SPREAD": "KC"},
        )
        sig = _signal(contract_id="KC-WIN", quantity=10, price_cents=50)  # 500¢ = 0.5%
        result = checker.check(sig, [])
        assert result.approved

    def test_group_over_limit_blocked(self):
        """Signal blocked when correlated group at limit."""
        checker = PositionLimitChecker(
            bankroll_cents=100_000,
            correlation_groups={"KC-WIN": "KC", "KC-SPREAD": "KC"},
        )
        # Existing KC position: 10,000¢ = 10% (at limit)
        existing = _position("KC-SPREAD", quantity=200, avg_entry_cents=50)
        sig = _signal(contract_id="KC-WIN", quantity=10, price_cents=50)
        result = checker.check(sig, [existing])
        assert not result.approved
        assert result.layer == "correlation"


# ============================================================
# Cash reserve (Layer 4 in position_limits)
# ============================================================


class TestCashReserve:
    """Cash reserve — maintain 50% minimum cash."""

    def test_plenty_of_cash(self):
        """Signal approved when plenty of cash available."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        sig = _signal(quantity=10, price_cents=50)  # 500¢
        result = checker.check(sig, [])
        assert result.approved

    def test_cash_reserve_blocks(self):
        """Signal blocked when it would violate cash reserve."""
        config = LimitConfig(
            min_bankroll_reserve_pct=0.50,
            max_position_pct=1.0,  # Disable position limit
            max_exposure_pct=1.0,  # Disable exposure limit
            max_correlation_pct=1.0,  # Disable correlation limit
        )
        checker = PositionLimitChecker(bankroll_cents=100_000, config=config)
        # Reserve = 50,000¢. Available = 100,000 - 50,000 = 50,000¢.
        # Already 49,500¢ in positions → only 500¢ available.
        positions = [_position(f"P-{i}", quantity=99, avg_entry_cents=50) for i in range(10)]
        # 10 × 99 × 50 = 49,500¢ exposure
        sig = _signal(quantity=20, price_cents=50)  # Wants 1,000¢ but only 500 available
        result = checker.check(sig, positions)
        # Should be reduced or blocked
        assert result.approved is True and result.reduced or not result.approved
        if not result.approved:
            assert result.layer == "cash_reserve"


# ============================================================
# Integration
# ============================================================


class TestLimitCheckIntegration:
    """Multi-layer integration tests."""

    def test_bankroll_update(self):
        """update_bankroll changes all limit calculations."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        checker.update_bankroll(200_000)
        # 5% of 200,000 = 10,000. At 50¢ → 200 contracts
        sig = _signal(quantity=200, price_cents=50)
        result = checker.check(sig, [])
        assert result.approved
        assert not result.reduced

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = LimitConfig(
            max_position_pct=0.02,  # Tighter: 2%
            max_exposure_pct=0.10,  # Tighter: 10%
        )
        checker = PositionLimitChecker(bankroll_cents=100_000, config=config)
        # 2% of 100,000 = 2,000. At 50¢ → max 40 contracts
        sig = _signal(quantity=50, price_cents=50)
        result = checker.check(sig, [])
        assert result.approved
        assert result.reduced
        assert sig.quantity <= 40

    def test_result_properties(self):
        """LimitCheckResult tracks modifications correctly."""
        checker = PositionLimitChecker(bankroll_cents=100_000)
        sig = _signal(quantity=200, price_cents=50)
        result = checker.check(sig, [])
        assert result.approved
        assert result.reduced
        assert result.was_modified
        assert result.original_quantity == 200
        assert sig.quantity < 200
