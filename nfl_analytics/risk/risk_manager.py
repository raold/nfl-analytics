"""
Unified risk gate orchestrator for prediction market trading.

Runs every signal through the full risk hierarchy before allowing execution:

    Layer 5: FEATURE HEALTH   — Block if data pipeline is degraded
    Layer 6: KILL SWITCH       — Block if emergency shutdown is active
    Layer 1-3: POSITION LIMITS — Check/reduce position size, exposure, correlation
    Layer 4: CVaR (future)     — Portfolio-level tail risk (placeholder)

The risk manager is the single entry point for all risk checks. Strategies
produce Signals, the risk manager validates them, and only approved signals
reach the trade executor.

Usage:
    from nfl_analytics.risk.risk_manager import RiskManager, RiskManagerConfig

    rm = RiskManager(bankroll_cents=100_000)
    result = rm.check_signal(signal, features, positions)
    if result.approved:
        executor.submit(result.signal)  # May have reduced quantity
    else:
        logger.warning("Blocked: %s", result.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nfl_analytics.markets.base import Position
from nfl_analytics.risk.feature_health import (
    FeatureHealthChecker,
    HealthCheckResult,
    HealthConfig,
)
from nfl_analytics.risk.kill_switch import KillSwitch, KillSwitchConfig, KillSwitchEvent
from nfl_analytics.risk.position_limits import LimitCheckResult, LimitConfig, PositionLimitChecker
from nfl_analytics.trading.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskManagerConfig:
    """Configuration for the unified risk manager."""

    # Sub-component configs (use defaults if not provided)
    limit_config: LimitConfig = field(default_factory=LimitConfig)
    kill_switch_config: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    health_config: HealthConfig = field(default_factory=HealthConfig)

    # Feature health check frequency — don't re-check on every signal
    health_check_cache_seconds: int = 60

    # Whether to enforce each layer (allows selective bypass for testing)
    enforce_feature_health: bool = True
    enforce_kill_switch: bool = True
    enforce_position_limits: bool = True


@dataclass
class RiskCheckResult:
    """Result of passing a signal through the full risk hierarchy."""

    approved: bool
    signal: Signal
    reason: str = ""
    layer: str = ""  # Which layer blocked/reduced
    original_quantity: int = 0
    reduced: bool = False

    # Sub-results for observability
    health_result: HealthCheckResult | None = None
    limit_result: LimitCheckResult | None = None
    kill_switch_active: bool = False

    @property
    def was_modified(self) -> bool:
        return self.reduced and self.signal.quantity != self.original_quantity


class RiskManager:
    """Unified risk gate orchestrating all risk layers.

    Holds state for kill switch (persistent), feature health (cached),
    and position limits (stateless per check). Each call to check_signal
    runs the layers in priority order and short-circuits on first hard block.

    The risk manager owns the KillSwitch instance because the kill switch
    persists across trading sessions (once triggered, stays triggered until
    manually reset).
    """

    def __init__(
        self,
        bankroll_cents: int,
        config: RiskManagerConfig | None = None,
        correlation_groups: dict[str, str] | None = None,
    ):
        self.config = config or RiskManagerConfig()

        # Layer 5: Feature health checker
        self._health_checker = FeatureHealthChecker(self.config.health_config)
        self._last_health_result: HealthCheckResult | None = None
        self._last_health_check_time: datetime | None = None

        # Layer 6: Kill switch (persistent state)
        self._kill_switch = KillSwitch(
            initial_equity_cents=bankroll_cents,
            config=self.config.kill_switch_config,
        )

        # Layers 1-3: Position limits
        self._position_checker = PositionLimitChecker(
            bankroll_cents=bankroll_cents,
            config=self.config.limit_config,
            correlation_groups=correlation_groups,
        )

        # Stats
        self._total_checks = 0
        self._total_approved = 0
        self._total_blocked = 0
        self._total_reduced = 0
        self._blocks_by_layer: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kill_switch(self) -> KillSwitch:
        """Direct access to the kill switch for equity updates."""
        return self._kill_switch

    @property
    def health_checker(self) -> FeatureHealthChecker:
        """Direct access to the health checker for inspection."""
        return self._health_checker

    @property
    def position_checker(self) -> PositionLimitChecker:
        """Direct access to position limit checker."""
        return self._position_checker

    @property
    def last_health_result(self) -> HealthCheckResult | None:
        return self._last_health_result

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def check_signal(
        self,
        signal: Signal,
        features: dict[str, float] | None = None,
        positions: list[Position] | None = None,
    ) -> RiskCheckResult:
        """Run a signal through all risk layers.

        Args:
            signal: The trade signal to validate.
            features: Current game features (for health check).
                If None and enforce_feature_health is True, the health
                check uses the last cached result.
            positions: Current open positions (for limit checks).
                If None, assumes no open positions.

        Returns:
            RiskCheckResult with approval status and any modifications.
        """
        self._total_checks += 1
        original_qty = signal.quantity
        positions = positions or []

        # Layer 5: Feature health
        if self.config.enforce_feature_health:
            result = self._check_feature_health(signal, features)
            if result is not None:
                self._record_block("feature_health")
                return result

        # Layer 6: Kill switch
        if self.config.enforce_kill_switch:
            result = self._check_kill_switch(signal)
            if result is not None:
                self._record_block("kill_switch")
                return result

        # Layers 1-3: Position limits
        if self.config.enforce_position_limits:
            limit_result = self._position_checker.check(signal, positions)
            if not limit_result.approved:
                self._record_block(limit_result.layer)
                return RiskCheckResult(
                    approved=False,
                    signal=signal,
                    reason=limit_result.reason,
                    layer=limit_result.layer,
                    original_quantity=original_qty,
                    limit_result=limit_result,
                )
            if limit_result.reduced:
                self._total_reduced += 1

        # All layers passed
        self._total_approved += 1
        return RiskCheckResult(
            approved=True,
            signal=signal,
            layer="all_passed",
            original_quantity=original_qty,
            reduced=signal.quantity != original_qty,
            health_result=self._last_health_result,
        )

    def update_equity(
        self,
        current_equity_cents: int,
        positions_open: int = 0,
        orders_open: int = 0,
    ) -> KillSwitchEvent | None:
        """Update equity tracking for the kill switch.

        Call this on every fill, settlement, or periodic equity snapshot.
        Returns a KillSwitchEvent if the kill switch was triggered.
        """
        return self._kill_switch.update_equity(
            current_equity_cents=current_equity_cents,
            positions_open=positions_open,
            orders_open=orders_open,
        )

    def update_bankroll(self, bankroll_cents: int) -> None:
        """Update bankroll for position limit calculations."""
        self._position_checker.update_bankroll(bankroll_cents)

    def record_api_error(
        self, positions_open: int = 0, orders_open: int = 0
    ) -> KillSwitchEvent | None:
        """Record an API error for consecutive error tracking."""
        return self._kill_switch.record_error(
            positions_open=positions_open,
            orders_open=orders_open,
        )

    def record_api_success(self) -> None:
        """Record a successful API call (resets error counter)."""
        self._kill_switch.record_success()

    def new_trading_day(self, equity_cents: int | None = None) -> None:
        """Reset daily tracking for a new trading day."""
        self._kill_switch.new_trading_day(equity_cents)

    def reset_kill_switch(self) -> None:
        """Re-arm the kill switch after manual review."""
        self._kill_switch.reset()

    def status(self) -> dict:
        """Get comprehensive risk manager status."""
        return {
            "kill_switch": self._kill_switch.status(),
            "feature_health": {
                "last_check": (
                    self._last_health_result.summary()
                    if self._last_health_result
                    else "no check yet"
                ),
                "healthy": self._last_health_result.healthy if self._last_health_result else None,
            },
            "stats": {
                "total_checks": self._total_checks,
                "approved": self._total_approved,
                "blocked": self._total_blocked,
                "reduced": self._total_reduced,
                "blocks_by_layer": dict(self._blocks_by_layer),
            },
        }

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_feature_health(
        self, signal: Signal, features: dict[str, float] | None
    ) -> RiskCheckResult | None:
        """Layer 5: Verify feature pipeline is healthy.

        Uses cached result if available and within TTL. Only blocks
        on critical group failures (configurable).

        Returns RiskCheckResult if blocked, None if passed.
        """
        # Run health check if we have new features or cache expired
        if features is not None:
            self._run_health_check(features)
        elif self._last_health_result is None:
            # No features provided and no cached result — can't check
            logger.debug("No features available for health check, allowing signal")
            return None

        # Check cache freshness
        if self._last_health_check_time is not None:
            age = (datetime.now(UTC) - self._last_health_check_time).total_seconds()
            if age > self.config.health_check_cache_seconds and features is not None:
                self._run_health_check(features)

        # Evaluate result
        if self._last_health_result is not None and not self._last_health_result.healthy:
            critical = self._last_health_result.critical_issues
            return RiskCheckResult(
                approved=False,
                signal=signal,
                reason=f"Feature health degraded: {'; '.join(critical) or 'check details'}",
                layer="feature_health",
                original_quantity=signal.quantity,
                health_result=self._last_health_result,
            )

        return None  # Passed

    def _run_health_check(self, features: dict[str, float]) -> None:
        """Execute a feature health check and cache the result."""
        self._last_health_result = self._health_checker.check_features(features)
        self._last_health_check_time = datetime.now(UTC)

    def _check_kill_switch(self, signal: Signal) -> RiskCheckResult | None:
        """Layer 6: Check if kill switch is active.

        Returns RiskCheckResult if blocked, None if passed.
        """
        if self._kill_switch.is_triggered:
            event = self._kill_switch.trigger_event
            reason = f"Kill switch active: {event.reason}" if event else "Kill switch active"
            return RiskCheckResult(
                approved=False,
                signal=signal,
                reason=reason,
                layer="kill_switch",
                original_quantity=signal.quantity,
                kill_switch_active=True,
            )
        return None  # Passed

    def _record_block(self, layer: str) -> None:
        """Record a blocked signal for stats."""
        self._total_blocked += 1
        self._blocks_by_layer[layer] = self._blocks_by_layer.get(layer, 0) + 1
