"""Pre-trade signal validation agent for the Kalshi trading system.

Catches suspicious signals that might indicate data errors, model bugs,
or market anomalies before they reach the risk pipeline. This is an
advisory layer — it logs warnings and can optionally block signals,
but is NOT part of the RiskManager's check_signal() hierarchy.

Usage:
    from nfl_analytics.agents.trading_validator import TradingValidator, ValidatorConfig

    validator = TradingValidator()
    result = validator.validate(signal, features)
    if not result.approved:
        logger.warning("Signal blocked by validator: %s", result.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from nfl_analytics.risk.risk_manager import RiskManager
from nfl_analytics.trading.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class ValidatorConfig:
    """Configuration for pre-trade signal validation."""

    max_edge_cents: int = 30
    min_price_cents: int = 1
    max_price_cents: int = 99
    known_prefixes: list[str] = field(default_factory=lambda: ["KXNFL", "KXNBA"])
    max_quantity: int = 200
    min_model_prob: float = 0.01
    max_model_prob: float = 0.99
    require_features: bool = True
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result of pre-trade signal validation."""

    approved: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


class TradingValidator:
    """Pre-trade signal validation — advisory layer outside the risk pipeline.

    Checks signals for anomalies before they reach RiskManager.check_signal().
    Catches suspicious edge values, price bounds violations, unknown contract
    prefixes, and excessive quantities.
    """

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        risk_manager: RiskManager | None = None,
    ):
        self.config = config or ValidatorConfig()
        self._risk_manager = risk_manager

        # Stats
        self._total_validated = 0
        self._total_approved = 0
        self._total_blocked = 0
        self._total_warnings = 0

    def validate(
        self,
        signal: Signal,
        features: dict[str, float] | None = None,
    ) -> ValidationResult:
        """Validate a single signal before it enters the risk pipeline.

        Args:
            signal: The trade signal to validate.
            features: Current game features (for health check via risk_manager).

        Returns:
            ValidationResult with approval status, reason if blocked,
            and warnings for borderline cases.
        """
        self._total_validated += 1

        if not self.config.enabled:
            self._total_approved += 1
            return ValidationResult(approved=True)

        warnings: list[str] = []

        # Check price in valid range
        if signal.price_cents < self.config.min_price_cents:
            self._total_blocked += 1
            return ValidationResult(
                approved=False,
                reason=f"Price {signal.price_cents}¢ below minimum {self.config.min_price_cents}¢",
                warnings=warnings,
            )
        if signal.price_cents > self.config.max_price_cents:
            self._total_blocked += 1
            return ValidationResult(
                approved=False,
                reason=f"Price {signal.price_cents}¢ above maximum {self.config.max_price_cents}¢",
                warnings=warnings,
            )

        # Check edge too large (suspicious — might be data error)
        if abs(signal.edge_cents) > self.config.max_edge_cents:
            self._total_blocked += 1
            return ValidationResult(
                approved=False,
                reason=(
                    f"Edge {signal.edge_cents}¢ exceeds maximum {self.config.max_edge_cents}¢ "
                    f"— possible data error"
                ),
                warnings=warnings,
            )

        # Warn on borderline edge (> 20 but <= max)
        if abs(signal.edge_cents) > 20:
            warnings.append(
                f"Large edge: {signal.edge_cents}¢ (threshold: {self.config.max_edge_cents}¢)"
            )

        # Check contract_id prefix
        if self.config.known_prefixes:
            prefix_match = any(
                signal.contract_id.startswith(p) for p in self.config.known_prefixes
            )
            if not prefix_match:
                self._total_blocked += 1
                return ValidationResult(
                    approved=False,
                    reason=(
                        f"Unknown contract prefix: {signal.contract_id} "
                        f"(known: {', '.join(self.config.known_prefixes)})"
                    ),
                    warnings=warnings,
                )

        # Check quantity isn't excessive
        if signal.quantity > self.config.max_quantity:
            self._total_blocked += 1
            return ValidationResult(
                approved=False,
                reason=f"Quantity {signal.quantity} exceeds maximum {self.config.max_quantity}",
                warnings=warnings,
            )

        # Check model_prob within bounds
        if signal.model_prob < self.config.min_model_prob:
            self._total_blocked += 1
            return ValidationResult(
                approved=False,
                reason=(
                    f"Model probability {signal.model_prob:.4f} below minimum "
                    f"{self.config.min_model_prob}"
                ),
                warnings=warnings,
            )
        if signal.model_prob > self.config.max_model_prob:
            self._total_blocked += 1
            return ValidationResult(
                approved=False,
                reason=(
                    f"Model probability {signal.model_prob:.4f} above maximum "
                    f"{self.config.max_model_prob}"
                ),
                warnings=warnings,
            )

        # Check features if required and risk_manager available
        if self.config.require_features and self._risk_manager is not None:
            if features is None:
                self._total_blocked += 1
                return ValidationResult(
                    approved=False,
                    reason="Features required but not provided",
                    warnings=warnings,
                )
            health_result = self._risk_manager.health_checker.check_features(features)
            if not health_result.healthy:
                warnings.append(
                    f"Feature health degraded: {'; '.join(health_result.critical_issues)}"
                )

        # All checks passed
        self._total_approved += 1
        if warnings:
            self._total_warnings += len(warnings)
            logger.warning(
                "Signal %s approved with %d warnings: %s",
                signal.contract_id,
                len(warnings),
                "; ".join(warnings),
            )
        else:
            logger.debug("Signal %s approved by validator", signal.contract_id)

        return ValidationResult(approved=True, warnings=warnings)

    def validate_batch(
        self,
        signals: list[Signal],
        features: dict[str, float] | None = None,
    ) -> list[ValidationResult]:
        """Validate a batch of signals.

        Args:
            signals: List of trade signals to validate.
            features: Shared features for all signals in the batch.

        Returns:
            List of ValidationResult, one per signal (same order).
        """
        return [self.validate(sig, features) for sig in signals]

    def summary(self) -> dict:
        """Get validator summary statistics."""
        return {
            "total_validated": self._total_validated,
            "approved": self._total_approved,
            "blocked": self._total_blocked,
            "warnings": self._total_warnings,
            "enabled": self.config.enabled,
        }
