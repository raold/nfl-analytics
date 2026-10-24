# ADR-019: Claude API Integration for Trading Signal Validation

**Status:** Proposed
**Date:** 2026-02-13
**Deciders:** User, Claude Code
**Tags:** architecture, claude-api, trading, ai-integration

---

## Context

### Problem Statement

The trading system generates signals mechanistically: the directional strategy computes model probability vs market price, and any signal with sufficient edge passes through the risk manager to execution. While the risk hierarchy (feature health, kill switch, position limits) catches systemic problems, it lacks a semantic or contextual validation layer. There is no mechanism to ask "does this signal make sense given the broader context?" before committing capital.

### Background

- The Kalshi trading system (ADR-018) produces binary contract signals from Game Outcome Model v2
- Signals pass through a 3-layer risk hierarchy but all checks are mechanical (thresholds, counts, staleness)
- Claude's extended thinking capability enables structured reasoning about multi-factor decisions
- The Anthropic SDK provides synchronous and async Python clients with extended thinking support
- Question: Can LLM reasoning improve signal quality enough to justify API costs and latency?

### Decision Drivers

- **Signal quality:** Some mechanically valid signals are contextually poor (e.g., stale injury news not yet in features)
- **Cost efficiency:** Per-signal API cost must be justified by preventing bad trades
- **Latency:** Trading signals are time-sensitive; validation cannot block execution indefinitely
- **Graceful degradation:** The system must work identically when the API is unavailable

---

## Decision

Build an opt-in `ClaudeSignalValidator` that uses the Anthropic SDK with extended thinking to review trading signals before execution. The validator sits between `RiskManager` and `TradeExecutor` as an optional filter. Disabled by default. Gracefully degrades (passes signals through) if the API is unavailable or the `anthropic` package is not installed.

### Details

**Architecture Position:**

```
Strategy -> RiskManager -> [ClaudeSignalValidator] -> TradeExecutor -> Exchange
                               (optional, opt-in)
```

**Model Choice:** `claude-sonnet-4-5-20250929` (cost-effective for per-signal calls with extended thinking support).

**Extended Thinking:** Enabled for deeper reasoning about signal quality. Budget of 1024 thinking tokens provides sufficient depth without excessive cost.

**Structured Output:** The validator expects a JSON response:

```json
{
    "decision": "approve" | "reduce" | "reject",
    "confidence": 0.0-1.0,
    "reasoning": "...",
    "suggested_quantity": null | int
}
```

**Prompt Context:** Each validation request includes signal details (contract, side, edge, model probability), recent P&L performance, market context, and model track record.

**Response Timeout:** 10 seconds. Trading is time-sensitive; if the API does not respond within 10 seconds, the signal passes through unchanged.

**Caching:** Recent validations are cached by `(contract_id, side, price_cents)` with a 5-minute TTL to avoid duplicate API calls for the same opportunity.

**Cost Analysis:**

| Item | Cost |
|------|------|
| Sonnet 4.5 input tokens | ~$3/M tokens |
| Sonnet 4.5 output tokens | ~$15/M tokens |
| Extended thinking tokens | ~$15/M tokens |
| Typical signal context | ~800 input tokens |
| Typical response | ~200 output tokens |
| Typical thinking | ~1000 tokens |
| **Per-signal cost** | **~$0.02-0.05** |
| At 50 signals/day | ~$1-2.50/day |
| Monthly estimate | ~$30-75/month |

**Break-even:** The validator needs to prevent approximately 1-2 bad trades per month to justify its cost, assuming average trade size of $50-100.

---

## Alternatives Considered

### Alternative 1: Local LLM (Ollama/qwen3)

**Description:** Run signal validation through a locally-hosted model (qwen3:8b via Ollama or MLX).

**Pros:**
- Zero marginal API cost
- No external dependency
- Full data privacy

**Cons:**
- Significantly worse reasoning capability for nuanced financial decisions
- 2.4-28.6 seconds inference time (MLX vs Ollama benchmarks from Dec 2025)
- No extended thinking support
- Cannot reason about novel situations as effectively

**Evaluation:** Rejected. The reasoning quality gap is too large for a financial decision-making component. The existing semantic stack already uses Ollama for feature extraction where speed matters more than depth.

### Alternative 2: Rule-Based Heuristics

**Description:** Extend the existing `TradingValidator` agent with additional rule-based checks (e.g., "reject signals on game day if injury report changed in last 2 hours").

**Pros:**
- Zero latency, zero cost
- Deterministic behavior
- Already have infrastructure (`RiskManager`)

**Cons:**
- Cannot capture semantic context ("this team's offensive line is depleted")
- Requires manual rule authoring for each scenario
- Brittle to novel situations

**Evaluation:** Not mutually exclusive. Rule-based heuristics remain in the risk manager. Claude adds a semantic layer on top for situations rules cannot anticipate.

### Alternative 3: Fine-Tuned Binary Classifier

**Description:** Train a classifier on historical signal outcomes (profitable/unprofitable) to filter signals.

**Pros:**
- Fast inference (~10ms)
- Tuned to specific trading patterns
- No ongoing API cost

**Cons:**
- Requires substantial labeled training data (profitable vs unprofitable signals)
- We have no such labeled data yet (system is new)
- Cannot adapt to novel market conditions without retraining
- Black-box decisions without reasoning

**Evaluation:** Rejected for now. Insufficient training data. Could be revisited after 1-2 seasons of live trading history.

---

## Consequences

### Positive

- **Semantic validation:** Catches contextually poor signals that pass mechanical checks
- **Reasoning trail:** Every validation includes human-readable reasoning for audit
- **Zero-risk deployment:** Disabled by default; graceful degradation means no impact on existing system
- **Low cost:** $30-75/month is negligible relative to potential drawdown prevention

### Negative

- **API dependency:** Introduces external service dependency (mitigated by passthrough on failure)
- **Latency:** Adds up to 10 seconds per signal in the worst case (mitigated by caching and timeout)
- **Non-determinism:** LLM responses may vary for identical inputs (mitigated by structured output + cache)

### Neutral

- **New dependency:** `anthropic` Python package (optional, not added to requirements)
- **New module:** `nfl_analytics/trading/claude_signal_validator.py` (~150 lines)
- **Configuration surface:** New `ClaudeValidatorConfig` dataclass

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API latency causes missed trades | Medium | Medium | 10s timeout + passthrough on failure |
| API costs exceed edge improvement | Medium | Low | Opt-in, monitor cost/benefit ratio via stats |
| API unavailability during trading hours | Low | Low | Graceful degradation (pass signals through) |
| Over-reliance on LLM for financial decisions | Low | High | Advisory only, risk manager still authoritative |
| Prompt injection via market data | Very Low | Medium | Signal data is structured, not user-supplied text |

---

## Implementation

### Approach

Proof-of-concept implementation as a single module (`nfl_analytics/trading/claude_signal_validator.py`) with full unit test coverage. The module is self-contained with no modifications to existing files. Integration into the `LiveTrader` pipeline is a future step.

### Timeline

- **Proposed:** 2026-02-13
- **Decided:** Pending
- **Implemented:** 2026-02-13 (PoC)
- **Validated:** Pending (requires live signal data)

### Success Criteria

- [ ] PoC module passes all unit tests without calling the Claude API
- [ ] Graceful degradation confirmed (disabled config, missing package, API errors all pass through)
- [ ] Response parsing handles malformed JSON without crashing
- [ ] Cache prevents redundant API calls within TTL window
- [ ] Stats tracking provides cost/benefit visibility
- [ ] Integration test with Anthropic sandbox API (future)
- [ ] A/B test: validator-enabled vs validator-disabled over 100+ signals (future)

### Validation

Track validator decisions alongside trade outcomes over a 30-day paper trading window. Compare P&L of signals the validator would have rejected against signals it approved. The validator justifies its cost if rejected signals have negative expected value.

---

## Related Decisions

- [ADR-018: Kalshi Trading System Architecture](./018-kalshi-trading-system.md) — The trading system this validator integrates into
- [ADR-017: 2026 Season Roadmap](./017-2026-season-roadmap.md) — Broader improvement roadmap

---

## References

### Documentation

- [Anthropic SDK Documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [CLAUDE.md — Kalshi Trading System section](../../CLAUDE.md)

### Code

- `nfl_analytics/trading/claude_signal_validator.py` — PoC implementation
- `nfl_analytics/trading/signal.py` — Signal dataclass
- `nfl_analytics/risk/` — Risk management hierarchy (upstream of validator)
- `nfl_analytics/execution/` — Trade executor (downstream of validator)

---

**Version:** 1.0
**Last Updated:** 2026-02-13
