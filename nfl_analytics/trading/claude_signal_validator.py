"""
Claude API signal validation proof-of-concept.

Uses the Anthropic SDK with extended thinking to review trading signals
before execution. Provides a semantic reasoning layer on top of the
mechanical risk checks.

This is an OPTIONAL, OPT-IN component. Disabled by default. When disabled
or when the API is unavailable, signals pass through unchanged.

Architecture position:
    Strategy -> RiskManager -> [ClaudeSignalValidator] -> TradeExecutor -> Exchange
                                   (optional, opt-in)

Usage:
    from nfl_analytics.trading.claude_signal_validator import (
        ClaudeSignalValidator,
        ClaudeValidatorConfig,
    )

    config = ClaudeValidatorConfig(api_key="sk-ant-...", enabled=True)
    validator = ClaudeSignalValidator(config)
    result = validator.validate(signal, context={"recent_pnl": -500, "win_rate": 0.45})
    if result.decision == ValidationDecision.REJECT:
        logger.warning("Claude rejected signal: %s", result.reasoning)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nfl_analytics.trading.signal import Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ClaudeValidatorConfig:
    """Configuration for Claude signal validation.

    Disabled by default. Set enabled=True and provide api_key to activate.
    """

    api_key: str = ""
    enabled: bool = False
    model: str = "claude-sonnet-4-5-20250929"
    timeout_seconds: float = 10.0
    max_thinking_tokens: int = 1024
    max_output_tokens: int = 256
    cache_ttl_seconds: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ValidationDecision(str, Enum):
    """Possible validator decisions."""

    APPROVE = "approve"
    REDUCE = "reduce"
    REJECT = "reject"


@dataclass
class ClaudeValidationResult:
    """Result of a Claude signal validation.

    Always returned — even on error (decision=APPROVE with error reasoning).
    """

    decision: ValidationDecision
    confidence: float  # 0.0-1.0
    reasoning: str
    suggested_quantity: int | None = None  # Only set if decision is REDUCE
    from_cache: bool = False
    api_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


_PASSTHROUGH_DISABLED = ClaudeValidationResult(
    decision=ValidationDecision.APPROVE,
    confidence=1.0,
    reasoning="validator_disabled",
)


class ClaudeSignalValidator:
    """Optional Claude-powered signal validation.

    Sits between RiskManager and TradeExecutor. When disabled or on any
    error, signals pass through unchanged (APPROVE).

    The ``anthropic`` package is imported lazily and is NOT a required
    dependency. If the package is missing, the validator silently passes
    all signals through.
    """

    def __init__(self, config: ClaudeValidatorConfig | None = None) -> None:
        self._config = config or ClaudeValidatorConfig()
        self._client: Any = None
        self._cache: dict[str, tuple[ClaudeValidationResult, float]] = {}

        # Stats counters
        self._total: int = 0
        self._approved: int = 0
        self._reduced: int = 0
        self._rejected: int = 0
        self._cache_hits: int = 0
        self._api_errors: int = 0
        self._total_latency_ms: float = 0.0

        if self._config.enabled and self._config.api_key:
            try:
                import anthropic  # noqa: F811

                self._client = anthropic.Anthropic(
                    api_key=self._config.api_key,
                    timeout=self._config.timeout_seconds,
                )
                logger.info("ClaudeSignalValidator initialized (model=%s)", self._config.model)
            except ImportError:
                logger.warning("anthropic package not installed — ClaudeSignalValidator disabled")
                self._client = None
        else:
            logger.debug("ClaudeSignalValidator disabled by config")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        signal: Signal,
        context: dict[str, Any] | None = None,
    ) -> ClaudeValidationResult:
        """Validate a trading signal using Claude.

        Returns APPROVE passthrough when:
        - Validator is disabled
        - API key is missing
        - anthropic package not installed
        - API call fails or times out
        - Response cannot be parsed

        Args:
            signal: The trading signal to validate.
            context: Optional dict with recent_pnl, win_rate, etc.

        Returns:
            ClaudeValidationResult with decision, confidence, reasoning.
        """
        self._total += 1

        if self._client is None:
            self._approved += 1
            return _PASSTHROUGH_DISABLED

        # Check cache
        key = self._cache_key(signal)
        cached = self._get_cached(key)
        if cached is not None:
            self._cache_hits += 1
            self._record_decision(cached)
            return cached

        # Call Claude API
        try:
            prompt = self._build_prompt(signal, context)
            start = time.monotonic()
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_thinking_tokens + self._config.max_output_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": self._config.max_thinking_tokens,
                },
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.monotonic() - start) * 1000.0
            self._total_latency_ms += latency_ms

            # Extract text from response content blocks
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text
                    break

            result = self._parse_response(text)
            result.api_latency_ms = latency_ms

            # Cache and track
            self._cache[key] = (result, time.monotonic())
            self._record_decision(result)
            logger.info(
                "Claude validation: %s (conf=%.2f, latency=%.0fms) — %s",
                result.decision.value,
                result.confidence,
                latency_ms,
                result.reasoning[:80],
            )
            return result

        except Exception as e:
            self._api_errors += 1
            self._approved += 1
            logger.warning("Claude API error, passing signal through: %s", e)
            return ClaudeValidationResult(
                decision=ValidationDecision.APPROVE,
                confidence=0.0,
                reasoning=f"api_error: {e}",
            )

    @property
    def stats(self) -> dict[str, Any]:
        """Validation statistics for monitoring."""
        api_calls = self._total - self._cache_hits
        return {
            "total_validations": self._total,
            "approved": self._approved,
            "reduced": self._reduced,
            "rejected": self._rejected,
            "cache_hits": self._cache_hits,
            "api_errors": self._api_errors,
            "avg_latency_ms": (self._total_latency_ms / api_calls if api_calls > 0 else 0.0),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_prompt(self, signal: Signal, context: dict[str, Any] | None) -> str:
        """Build the validation prompt with signal details and context."""
        lines = [
            "You are a trading signal validator for a prediction market system.",
            "Review the following signal and decide whether to APPROVE, REDUCE, or REJECT it.",
            "",
            "Respond with ONLY a JSON object (no markdown, no explanation outside JSON):",
            '{"decision": "approve"|"reduce"|"reject", "confidence": 0.0-1.0, '
            '"reasoning": "brief explanation", "suggested_quantity": null or int}',
            "",
            "Signal Details:",
            f"  Contract: {signal.contract_id}",
            f"  Side: {signal.side.value}",
            f"  Quantity: {signal.quantity}",
            f"  Price: {signal.price_cents}c",
            f"  Edge: {signal.edge_cents}c ({signal.edge_pct:.1%})",
            f"  Model Probability: {signal.model_prob:.1%}",
            f"  Model Confidence: {signal.confidence:.1%}",
            f"  Strategy: {signal.strategy}",
            f"  Sport: {signal.sport}",
            f"  Meta Probability: {signal.meta_prob:.2f}",
            f"  Kelly Fraction: {signal.kelly_fraction:.3f}",
        ]

        if context:
            lines.append("")
            lines.append("Context:")
            for k, v in context.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def _parse_response(self, text: str) -> ClaudeValidationResult:
        """Parse Claude's JSON response into a validation result.

        Falls back to APPROVE if parsing fails.
        """
        try:
            data = json.loads(text)
            decision = ValidationDecision(data["decision"])
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(data.get("reasoning", ""))
            suggested_quantity = data.get("suggested_quantity")
            if suggested_quantity is not None:
                suggested_quantity = int(suggested_quantity)

            return ClaudeValidationResult(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                suggested_quantity=suggested_quantity,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse Claude response: %s (text=%s)", e, text[:200])
            return ClaudeValidationResult(
                decision=ValidationDecision.APPROVE,
                confidence=0.0,
                reasoning=f"parse_error: {e}",
            )

    def _cache_key(self, signal: Signal) -> str:
        """Generate cache key from signal identity."""
        return f"{signal.contract_id}:{signal.side.value}:{signal.price_cents}"

    def _get_cached(self, key: str) -> ClaudeValidationResult | None:
        """Return cached result if within TTL, else None."""
        if key not in self._cache:
            return None
        result, timestamp = self._cache[key]
        if (time.monotonic() - timestamp) > self._config.cache_ttl_seconds:
            del self._cache[key]
            return None
        cached_result = ClaudeValidationResult(
            decision=result.decision,
            confidence=result.confidence,
            reasoning=result.reasoning,
            suggested_quantity=result.suggested_quantity,
            from_cache=True,
            api_latency_ms=0.0,
        )
        return cached_result

    def _record_decision(self, result: ClaudeValidationResult) -> None:
        """Update stats counters based on decision."""
        if result.decision == ValidationDecision.APPROVE:
            self._approved += 1
        elif result.decision == ValidationDecision.REDUCE:
            self._reduced += 1
        elif result.decision == ValidationDecision.REJECT:
            self._rejected += 1
