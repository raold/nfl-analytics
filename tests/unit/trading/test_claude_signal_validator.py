"""
Unit tests for py/trading/claude_signal_validator.py — Claude signal validation PoC.

All tests mock the Anthropic SDK. No real API calls are made.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from py.markets.base import Side
from py.trading.claude_signal_validator import (
    ClaudeSignalValidator,
    ClaudeValidatorConfig,
    ClaudeValidationResult,
    ValidationDecision,
)
from py.trading.signal import Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signal(
    contract_id: str = "KXNFL-26-KC",
    side: Side = Side.YES,
    quantity: int = 10,
    price_cents: int = 55,
    edge_cents: int = 7,
    model_prob: float = 0.62,
    confidence: float = 0.85,
    strategy: str = "directional",
) -> Signal:
    """Helper to create a test signal with reasonable defaults."""
    return Signal(
        contract_id=contract_id,
        side=side,
        quantity=quantity,
        price_cents=price_cents,
        edge_cents=edge_cents,
        model_prob=model_prob,
        confidence=confidence,
        strategy=strategy,
    )


def _mock_claude_response(data: dict) -> MagicMock:
    """Build a mock Anthropic messages.create() response."""
    text_block = MagicMock()
    text_block.text = json.dumps(data)
    response = MagicMock()
    response.content = [text_block]
    return response


def _make_validator_with_mock(response_data: dict | None = None) -> tuple[ClaudeSignalValidator, MagicMock]:
    """Create a validator with a mocked Anthropic client.

    Returns (validator, mock_client) so tests can assert on calls.
    """
    mock_client = MagicMock()
    if response_data is not None:
        mock_client.messages.create.return_value = _mock_claude_response(response_data)

    config = ClaudeValidatorConfig(api_key="sk-ant-test-key", enabled=True)
    # Patch the import so __init__ picks up a mock anthropic module
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        validator = ClaudeSignalValidator(config)

    # The __init__ created a client via mock_anthropic.Anthropic(), which
    # returns mock_client. Verify it landed correctly.
    assert validator._client is mock_client
    return validator, mock_client


# ============================================================
# Passthrough / disabled tests
# ============================================================


class TestPassthrough:
    def test_disabled_passthrough(self):
        """Config enabled=False -> always returns APPROVE with 'validator_disabled'."""
        config = ClaudeValidatorConfig(enabled=False, api_key="sk-ant-test")
        validator = ClaudeSignalValidator(config)
        result = validator.validate(_make_signal())

        assert result.decision == ValidationDecision.APPROVE
        assert result.reasoning == "validator_disabled"
        assert result.confidence == 1.0

    def test_no_api_key_passthrough(self):
        """Enabled=True but empty api_key -> returns APPROVE passthrough."""
        config = ClaudeValidatorConfig(enabled=True, api_key="")
        validator = ClaudeSignalValidator(config)
        result = validator.validate(_make_signal())

        assert result.decision == ValidationDecision.APPROVE
        assert result.reasoning == "validator_disabled"

    def test_missing_anthropic_package(self):
        """If the anthropic package is not importable -> APPROVE passthrough."""
        config = ClaudeValidatorConfig(enabled=True, api_key="sk-ant-test")

        with patch.dict("sys.modules", {"anthropic": None}):
            # Importing None causes ImportError in the try block
            validator = ClaudeSignalValidator(config)

        result = validator.validate(_make_signal())
        assert result.decision == ValidationDecision.APPROVE
        assert result.reasoning == "validator_disabled"


# ============================================================
# API response handling
# ============================================================


class TestAPIResponses:
    def test_approve_response(self):
        """Mock Claude returning approve JSON -> APPROVE result."""
        validator, _ = _make_validator_with_mock({
            "decision": "approve",
            "confidence": 0.85,
            "reasoning": "Signal looks reasonable given current edge",
        })
        result = validator.validate(_make_signal())

        assert result.decision == ValidationDecision.APPROVE
        assert result.confidence == 0.85
        assert "reasonable" in result.reasoning
        assert result.from_cache is False
        assert result.api_latency_ms > 0.0

    def test_reduce_response(self):
        """Mock Claude returning reduce JSON with suggested_quantity -> REDUCE."""
        validator, _ = _make_validator_with_mock({
            "decision": "reduce",
            "confidence": 0.70,
            "reasoning": "Edge is marginal, reduce position size",
            "suggested_quantity": 5,
        })
        result = validator.validate(_make_signal(quantity=10))

        assert result.decision == ValidationDecision.REDUCE
        assert result.confidence == 0.70
        assert result.suggested_quantity == 5
        assert "marginal" in result.reasoning

    def test_reject_response(self):
        """Mock Claude returning reject JSON -> REJECT result."""
        validator, _ = _make_validator_with_mock({
            "decision": "reject",
            "confidence": 0.92,
            "reasoning": "Key player injury not reflected in model probability",
        })
        result = validator.validate(_make_signal())

        assert result.decision == ValidationDecision.REJECT
        assert result.confidence == 0.92
        assert "injury" in result.reasoning

    def test_api_timeout(self):
        """API raises exception -> returns APPROVE with api_error reason."""
        validator, mock_client = _make_validator_with_mock()
        mock_client.messages.create.side_effect = TimeoutError("Request timed out")

        result = validator.validate(_make_signal())

        assert result.decision == ValidationDecision.APPROVE
        assert "api_error" in result.reasoning
        assert "timed out" in result.reasoning

    def test_invalid_json_response(self):
        """Claude returns garbage text -> APPROVE fallback."""
        validator, mock_client = _make_validator_with_mock()
        # Override the response with non-JSON text
        text_block = MagicMock()
        text_block.text = "I think this signal is probably fine but I'm not sure"
        response = MagicMock()
        response.content = [text_block]
        mock_client.messages.create.return_value = response

        result = validator.validate(_make_signal())

        assert result.decision == ValidationDecision.APPROVE
        assert "parse_error" in result.reasoning


# ============================================================
# Caching
# ============================================================


class TestCaching:
    def test_cache_hit(self):
        """Validate same signal twice -> second is from_cache=True."""
        validator, mock_client = _make_validator_with_mock({
            "decision": "approve",
            "confidence": 0.80,
            "reasoning": "Looks good",
        })
        signal = _make_signal()

        result1 = validator.validate(signal)
        result2 = validator.validate(signal)

        assert result1.from_cache is False
        assert result2.from_cache is True
        assert result2.decision == ValidationDecision.APPROVE
        assert result2.api_latency_ms == 0.0
        # API should only be called once
        assert mock_client.messages.create.call_count == 1

    def test_cache_miss_different_signal(self):
        """Different contract_id -> cache miss, separate API call."""
        validator, mock_client = _make_validator_with_mock({
            "decision": "approve",
            "confidence": 0.80,
            "reasoning": "OK",
        })

        validator.validate(_make_signal(contract_id="KXNFL-26-KC"))
        validator.validate(_make_signal(contract_id="KXNFL-26-BUF"))

        assert mock_client.messages.create.call_count == 2


# ============================================================
# Prompt building
# ============================================================


class TestPromptBuilding:
    def test_build_prompt_includes_signal(self):
        """Prompt should contain signal contract_id, edge, model_prob."""
        validator, _ = _make_validator_with_mock()
        signal = _make_signal(
            contract_id="KXNFL-26-AFC-KC",
            edge_cents=9,
            model_prob=0.71,
        )
        prompt = validator._build_prompt(signal, context=None)

        assert "KXNFL-26-AFC-KC" in prompt
        assert "9c" in prompt
        assert "71.0%" in prompt

    def test_build_prompt_includes_context(self):
        """Context dict should appear in the prompt."""
        validator, _ = _make_validator_with_mock()
        signal = _make_signal()
        context = {"recent_pnl": -500, "win_rate": 0.45, "streak": "3 losses"}
        prompt = validator._build_prompt(signal, context)

        assert "recent_pnl" in prompt
        assert "-500" in prompt
        assert "win_rate" in prompt
        assert "0.45" in prompt
        assert "3 losses" in prompt


# ============================================================
# Stats tracking
# ============================================================


class TestStats:
    def test_stats_tracking(self):
        """Validate multiple signals -> stats reflect counts."""
        validator, mock_client = _make_validator_with_mock()

        # First call: approve
        mock_client.messages.create.return_value = _mock_claude_response({
            "decision": "approve",
            "confidence": 0.8,
            "reasoning": "OK",
        })
        validator.validate(_make_signal(contract_id="A"))

        # Second call: reject
        mock_client.messages.create.return_value = _mock_claude_response({
            "decision": "reject",
            "confidence": 0.9,
            "reasoning": "Bad",
        })
        validator.validate(_make_signal(contract_id="B"))

        # Third call: reduce
        mock_client.messages.create.return_value = _mock_claude_response({
            "decision": "reduce",
            "confidence": 0.7,
            "reasoning": "Cut",
            "suggested_quantity": 3,
        })
        validator.validate(_make_signal(contract_id="C"))

        # Fourth call: API error
        mock_client.messages.create.side_effect = ConnectionError("lost connection")
        validator.validate(_make_signal(contract_id="D"))

        stats = validator.stats
        assert stats["total_validations"] == 4
        assert stats["approved"] == 2  # 1 approve + 1 api_error passthrough
        assert stats["rejected"] == 1
        assert stats["reduced"] == 1
        assert stats["api_errors"] == 1
        assert stats["cache_hits"] == 0
        assert stats["avg_latency_ms"] >= 0.0


# ============================================================
# Parse response
# ============================================================


class TestParseResponse:
    def test_parse_response_valid(self):
        """_parse_response with valid JSON returns correct result."""
        validator, _ = _make_validator_with_mock()
        text = json.dumps({
            "decision": "reduce",
            "confidence": 0.65,
            "reasoning": "Edge is thin",
            "suggested_quantity": 3,
        })
        result = validator._parse_response(text)

        assert result.decision == ValidationDecision.REDUCE
        assert result.confidence == 0.65
        assert result.reasoning == "Edge is thin"
        assert result.suggested_quantity == 3

    def test_parse_response_invalid(self):
        """_parse_response with invalid JSON -> APPROVE fallback."""
        validator, _ = _make_validator_with_mock()
        result = validator._parse_response("this is not json {{{")

        assert result.decision == ValidationDecision.APPROVE
        assert result.confidence == 0.0
        assert "parse_error" in result.reasoning
