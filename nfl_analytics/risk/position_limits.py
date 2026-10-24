"""
Position and exposure limit enforcement for prediction market trading.

Implements the first three risk layers:
    Layer 1: POSITION SIZE    — Max 5% per contract, Kelly-adjusted
    Layer 2: TOTAL EXPOSURE   — Max 25% of bankroll in open positions
    Layer 3: CORRELATION LIMITS — Max 10% in correlated group (same game, team)

Signals pass through these checks before reaching the executor. Rejected
signals are logged with the reason. Signals can also be REDUCED (quantity
scaled down) to fit within limits.

Usage:
    from nfl_analytics.risk.position_limits import PositionLimitChecker, LimitConfig

    checker = PositionLimitChecker(bankroll_cents=100_000)
    result = checker.check(signal, current_positions)
    if result.approved:
        executor.submit(signal)
    else:
        logger.warning("Blocked: %s", result.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nfl_analytics.markets.base import Position, Side
from nfl_analytics.trading.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class LimitConfig:
    """Configurable position and exposure limits."""

    max_position_pct: float = 0.05  # Max 5% of bankroll per position
    max_exposure_pct: float = 0.25  # Max 25% total exposure
    max_correlation_pct: float = 0.10  # Max 10% in correlated group
    min_bankroll_reserve_pct: float = 0.50  # Keep 50% cash


@dataclass
class LimitCheckResult:
    """Result of a position limit check."""

    approved: bool
    signal: Signal
    reason: str = ""
    original_quantity: int = 0  # Quantity before any reduction
    reduced: bool = False  # True if quantity was reduced to fit
    layer: str = ""  # Which risk layer blocked/reduced

    @property
    def was_modified(self) -> bool:
        return self.reduced and self.signal.quantity != self.original_quantity


class PositionLimitChecker:
    """Enforces position size, total exposure, and correlation limits.

    Each signal is checked against all three layers in order.
    A signal can be:
    - APPROVED: passes all checks unmodified
    - REDUCED: quantity lowered to fit within limits
    - BLOCKED: rejected (quantity would be 0 after reduction)
    """

    def __init__(
        self,
        bankroll_cents: int,
        config: LimitConfig | None = None,
        correlation_groups: dict[str, str] | None = None,
    ):
        """
        Args:
            bankroll_cents: Current bankroll in cents.
            config: Limit thresholds (default: 5%/25%/10%).
            correlation_groups: Mapping of contract_id → group_id.
                Contracts in the same group share the correlation limit.
                Example: {"KC-WIN-SB": "KC", "KC-SPREAD-3": "KC"}
        """
        self.bankroll_cents = bankroll_cents
        self.config = config or LimitConfig()
        self.correlation_groups = correlation_groups or {}

    def update_bankroll(self, bankroll_cents: int) -> None:
        """Update bankroll (call after deposits, withdrawals, settlements)."""
        self.bankroll_cents = bankroll_cents

    def check(
        self,
        signal: Signal,
        positions: list[Position],
    ) -> LimitCheckResult:
        """Run all position limit checks on a signal.

        Checks in order:
        1. Single position size (Layer 1)
        2. Total exposure (Layer 2)
        3. Correlation group (Layer 3)
        4. Cash reserve (minimum reserve)

        Returns LimitCheckResult with approved/rejected/reduced status.
        """
        original_qty = signal.quantity

        # Layer 1: Single position size
        result = self._check_single_position(signal, positions)
        if not result.approved:
            return result

        # Layer 2: Total exposure
        result = self._check_total_exposure(signal, positions)
        if not result.approved:
            return result

        # Layer 3: Correlation group
        result = self._check_correlation_group(signal, positions)
        if not result.approved:
            return result

        # Layer 4: Cash reserve
        result = self._check_cash_reserve(signal, positions)
        if not result.approved:
            return result

        return LimitCheckResult(
            approved=True,
            signal=signal,
            original_quantity=original_qty,
            reduced=signal.quantity != original_qty,
            layer="all_passed",
        )

    def _check_single_position(
        self, signal: Signal, positions: list[Position]
    ) -> LimitCheckResult:
        """Layer 1: Max position size as % of bankroll."""
        max_cost_cents = int(self.bankroll_cents * self.config.max_position_pct)

        # Include existing position in this contract
        existing_qty = 0
        for pos in positions:
            if pos.contract_id == signal.contract_id and pos.side == signal.side:
                existing_qty = pos.quantity

        # Total position cost after this signal
        total_qty = existing_qty + signal.quantity
        total_cost = total_qty * signal.price_cents

        if total_cost <= max_cost_cents:
            return LimitCheckResult(
                approved=True, signal=signal, original_quantity=signal.quantity
            )

        # Try to reduce quantity
        max_new_qty = max(0, (max_cost_cents // signal.price_cents) - existing_qty)
        if max_new_qty <= 0:
            return LimitCheckResult(
                approved=False,
                signal=signal,
                original_quantity=signal.quantity,
                reason=(
                    f"Position limit: {signal.contract_id} would be "
                    f"{total_cost}¢ ({total_cost / self.bankroll_cents:.1%}), "
                    f"max {max_cost_cents}¢ ({self.config.max_position_pct:.0%})"
                ),
                layer="position_size",
            )

        signal.quantity = max_new_qty
        logger.info(
            "Reduced %s qty %d→%d (position limit %.0f%%)",
            signal.contract_id,
            total_qty - existing_qty + max_new_qty - max_new_qty,
            max_new_qty,
            self.config.max_position_pct * 100,
        )
        return LimitCheckResult(
            approved=True,
            signal=signal,
            original_quantity=total_qty - existing_qty,
            reduced=True,
            layer="position_size",
        )

    def _check_total_exposure(
        self, signal: Signal, positions: list[Position]
    ) -> LimitCheckResult:
        """Layer 2: Max total exposure across all positions."""
        max_exposure_cents = int(self.bankroll_cents * self.config.max_exposure_pct)

        # Sum current exposure
        current_exposure = sum(pos.avg_entry_cents * pos.quantity for pos in positions)
        signal_cost = signal.price_cents * signal.quantity
        total_exposure = current_exposure + signal_cost

        if total_exposure <= max_exposure_cents:
            return LimitCheckResult(
                approved=True, signal=signal, original_quantity=signal.quantity
            )

        # Try to reduce
        available = max(0, max_exposure_cents - current_exposure)
        max_qty = available // signal.price_cents if signal.price_cents > 0 else 0

        if max_qty <= 0:
            return LimitCheckResult(
                approved=False,
                signal=signal,
                original_quantity=signal.quantity,
                reason=(
                    f"Exposure limit: total would be {total_exposure}¢ "
                    f"({total_exposure / self.bankroll_cents:.1%}), "
                    f"max {max_exposure_cents}¢ ({self.config.max_exposure_pct:.0%})"
                ),
                layer="total_exposure",
            )

        original = signal.quantity
        signal.quantity = max_qty
        return LimitCheckResult(
            approved=True,
            signal=signal,
            original_quantity=original,
            reduced=True,
            layer="total_exposure",
        )

    def _check_correlation_group(
        self, signal: Signal, positions: list[Position]
    ) -> LimitCheckResult:
        """Layer 3: Max exposure in correlated contract group."""
        group = self.correlation_groups.get(signal.contract_id)
        if group is None:
            # No group assigned, skip check
            return LimitCheckResult(
                approved=True, signal=signal, original_quantity=signal.quantity
            )

        max_group_cents = int(self.bankroll_cents * self.config.max_correlation_pct)

        # Sum exposure in same group
        group_exposure = 0
        for pos in positions:
            pos_group = self.correlation_groups.get(pos.contract_id)
            if pos_group == group:
                group_exposure += pos.avg_entry_cents * pos.quantity

        signal_cost = signal.price_cents * signal.quantity
        total_group = group_exposure + signal_cost

        if total_group <= max_group_cents:
            return LimitCheckResult(
                approved=True, signal=signal, original_quantity=signal.quantity
            )

        available = max(0, max_group_cents - group_exposure)
        max_qty = available // signal.price_cents if signal.price_cents > 0 else 0

        if max_qty <= 0:
            return LimitCheckResult(
                approved=False,
                signal=signal,
                original_quantity=signal.quantity,
                reason=(
                    f"Correlation limit: group '{group}' would be {total_group}¢ "
                    f"({total_group / self.bankroll_cents:.1%}), "
                    f"max {max_group_cents}¢ ({self.config.max_correlation_pct:.0%})"
                ),
                layer="correlation",
            )

        original = signal.quantity
        signal.quantity = max_qty
        return LimitCheckResult(
            approved=True,
            signal=signal,
            original_quantity=original,
            reduced=True,
            layer="correlation",
        )

    def _check_cash_reserve(
        self, signal: Signal, positions: list[Position]
    ) -> LimitCheckResult:
        """Ensure minimum cash reserve is maintained."""
        min_reserve = int(self.bankroll_cents * self.config.min_bankroll_reserve_pct)
        current_exposure = sum(pos.avg_entry_cents * pos.quantity for pos in positions)
        cash_after = self.bankroll_cents - current_exposure - (signal.price_cents * signal.quantity)

        if cash_after >= min_reserve:
            return LimitCheckResult(
                approved=True, signal=signal, original_quantity=signal.quantity
            )

        # Try to reduce
        available_cash = self.bankroll_cents - current_exposure - min_reserve
        max_qty = max(0, available_cash // signal.price_cents) if signal.price_cents > 0 else 0

        if max_qty <= 0:
            return LimitCheckResult(
                approved=False,
                signal=signal,
                original_quantity=signal.quantity,
                reason=(
                    f"Cash reserve: would leave {cash_after}¢, "
                    f"min reserve {min_reserve}¢ ({self.config.min_bankroll_reserve_pct:.0%})"
                ),
                layer="cash_reserve",
            )

        original = signal.quantity
        signal.quantity = max_qty
        return LimitCheckResult(
            approved=True,
            signal=signal,
            original_quantity=original,
            reduced=True,
            layer="cash_reserve",
        )
