# ADR-018: Kalshi Trading System Architecture

**Status:** Superseded by the separate Sibyl repository; code removed in the 2026-10 prune
**Date:** 2026-02-13

> **Note (2026-10-24), [retraction ledger](../investigations/2026-10-final-evaluation/retraction_ledger.md)
> row C03.** The "mature prediction capabilities" below did not exist: Game Outcome v2's 56.5%
> win-probability accuracy is 162 of 284 decided games (57.0%), against 65.8% for the closing
> favorite, and its log-loss is worse than the market's (0.6871 against 0.6070). No model in this
> project beats the closing line, and v2 is frozen (DO NOT DEPLOY). The meta-labeling gate was
> trained on wrong-sign labels. `nfl_analytics/{trading,markets,risk,execution,monitoring}/` are at
> tag `archive/pre-prune-2026-10`.
**Deciders:** User, Claude Code
**Tags:** architecture, trading, kalshi, prediction-markets

---

## Context

### Problem Statement

The NFL analytics system has mature prediction capabilities (Game Outcome Model v2, 56.5% win probability, ensemble of 5 models) but no mechanism to convert predictions into automated trades. Prediction markets (specifically Kalshi, the only CFTC-regulated exchange) offer binary contracts on NFL outcomes with tighter spreads than traditional sportsbooks, making them suitable for systematic trading.

### Background

- Game Outcome Model v2 produces spread, total, and win probability predictions with calibrated uncertainty
- Kalshi offers NFL contracts: spreads, moneylines, totals, and player props
- Binary contracts trade in cents (1-99¢ = 1-99% probability)
- Kalshi provides REST API + WebSocket with RSA-PSS authentication
- The system needs paper trading for validation before live deployment

### Decision Drivers

- **Monetization:** Convert prediction edge into realized P&L
- **Risk management:** Prevent catastrophic losses through automated guardrails
- **Extensibility:** Support multiple sports and strategy types beyond NFL directional
- **Testability:** Every component must be independently testable with >90% coverage
- **Async-first:** Market data and order management are inherently concurrent

---

## Decision

Build a 6-phase modular trading system with strict separation between strategy (signal generation), risk (signal gating), execution (order management), and monitoring (P&L/alerting). All market interaction goes through an abstract `MarketClient` interface enabling paper/live switching.

### Details

**Phase 1 — Foundation:** Abstract base classes for `MarketClient`, `TradingStrategy`, `Signal`, and `PaperClient`. Establishes the contract that all downstream code depends on.

**Phase 2 — NFL Strategy:** `DirectionalStrategy` integrates Game Outcome Model v2 for NFL spread/moneyline/total contracts. `RiskManager` orchestrates a 3-layer risk hierarchy (feature health → kill switch → position limits).

**Phase 3 — Kalshi Live:** `KalshiClient` implements `MarketClient` with RSA-PSS auth, token-bucket rate limiting, exponential backoff retry, and batch order placement. WebSocket support for real-time market data.

**Phase 4 — Market Making:** `MarketMaker` strategy quotes both sides of a contract with configurable half-spread. `HybridStrategy` combines directional edge with market-making spread capture.

**Phase 5 — Monitoring & Safety:** `PnLTracker` (FIFO matching), `DrawdownMonitor` (Sharpe/Sortino/recovery factor), `TradeLogger` (JSONL), `AlertManager` (webhook + cooldown), and `LiveTrader` (async orchestration loop).

**Phase 6 — Multi-Sport:** `MarketRouter` routes contracts to sport-specific strategies via longest-prefix matching on contract IDs (e.g., `KXNFL` → NFL, `KXNBA` → NBA).

### Architecture

```
MarketClient.get_market_data() → Strategy.on_market_data() → Signal[]
                                                                 │
                                          RiskManager.check_signal()
                                          ├── Layer 5: FeatureHealthChecker
                                          ├── Layer 6: KillSwitch
                                          └── Layers 1-3: PositionLimitChecker
                                                                 │
                                          TradeExecutor.submit_signal() → Exchange
                                                                 │
                                                    on_fill callback
                                          ├── PnLTracker.record_fill()
                                          ├── RiskManager.update_equity()
                                          ├── DrawdownMonitor.record_equity()
                                          ├── TradeLogger.log_fill()
                                          └── AlertManager.alert_fill()
```

### Key Design Decisions

1. **FIFO P&L matching:** Position closes match against oldest opens (standard for tax and audit). Simpler than LIFO/specific-lot and correct for binary contracts.

2. **Prefix-based contract routing:** Contract IDs encode sport (`KXNFL-*`), enabling O(1) routing without API calls. Longest-prefix-first handles overlapping prefixes.

3. **Layered risk with short-circuit:** Risk checks execute in priority order. If feature health fails, skip kill switch and position limits. Reduces latency on blocked signals.

4. **Cents-native pricing:** All prices and P&L tracked in integer cents, avoiding floating-point precision issues inherent in financial calculations.

5. **Callback-driven fill handling:** `TradeExecutor.on_fill` triggers a cascade of updates (P&L, risk, logging, alerts) without the executor needing to know about downstream consumers.

6. **Graceful degradation:** If `AlertManager` webhook fails, alerts still log locally. If meta-labeling model unavailable, `DirectionalStrategy` falls back to raw edge.

---

## Alternatives Considered

### Alternative 1: Monolithic Trading Engine

**Description:** Single class handling strategy, risk, execution, and monitoring.

**Pros:**
- Simpler initial implementation
- No inter-module communication overhead

**Cons:**
- Untestable (can't mock exchange without mocking everything)
- Can't swap strategies without rewriting engine
- Risk logic tightly coupled to execution

**Evaluation:** Rejected — testability and extensibility are non-negotiable for a financial system.

### Alternative 2: Event-Driven Architecture (Message Queue)

**Description:** Components communicate via an event bus (Redis Streams, RabbitMQ).

**Pros:**
- Natural decoupling
- Easy to add new consumers
- Built-in audit trail

**Cons:**
- Infrastructure overhead (requires message broker)
- Latency added by serialization/deserialization
- Complexity disproportionate to current scale (single-user system)

**Evaluation:** Rejected for now — asyncio callbacks provide sufficient decoupling at current scale. Can migrate to event-driven if multi-user or multi-exchange support needed.

### Alternative 3: Third-Party Framework (Zipline, Backtrader)

**Description:** Build on existing backtesting/trading frameworks.

**Pros:**
- Mature ecosystem, built-in backtesting
- Community support

**Cons:**
- Designed for equities, not binary contracts
- No Kalshi integration
- Heavy dependencies that conflict with existing stack

**Evaluation:** Rejected — binary contract mechanics (1-99¢ pricing, settlement at 0 or 100) are fundamentally different from equity trading. Custom implementation required.

---

## Consequences

### Positive

- **Full test coverage:** 437 tests across 5 modules, all passing in <6s
- **Paper trading:** `PaperClient` enables end-to-end testing without exchange connection
- **Multi-strategy:** Can run directional + market making + hybrid simultaneously
- **Extensible:** Adding a new sport requires only a `TradingStrategy` subclass and a `MarketRouter.register_sport()` call

### Negative

- **No backtesting yet:** Historical Kalshi data replay not implemented (roadmap item)
- **Single exchange:** Only Kalshi supported; adding Polymarket/PredictIt would require new `MarketClient` implementations
- **No portfolio optimization:** Position sizing uses Kelly criterion per-signal, not portfolio-level optimization

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Kalshi API changes break client | Medium | High | Schema validation, version pinning, integration tests |
| Kill switch fails to trigger | Low | Critical | Multiple independent checks (daily loss + drawdown + API errors) |
| Feature staleness causes bad signals | Medium | Medium | FeatureHealthChecker blocks signals when data is stale |
| Rate limiting causes missed trades | Low | Low | Token bucket with configurable limits, batch API for orders |

---

## Implementation

### Approach

Implemented in 6 sequential phases over a single development session. Each phase built on the previous, with tests written alongside production code (test-first for risk and execution modules).

### Timeline

- **Proposed:** 2026-02-13
- **Decided:** 2026-02-13
- **Implemented:** 2026-02-13
- **Validated:** 2026-02-13 (437 tests passing)

### Success Criteria

- [x] All 6 phases implemented with full test coverage
- [x] Paper trading end-to-end pipeline works
- [x] Risk manager correctly gates signals through all layers
- [x] Kill switch triggers on daily loss, drawdown, and API errors
- [x] Multi-sport routing via MarketRouter
- [ ] Integration test with Kalshi sandbox (next phase)
- [ ] Backtest on historical Kalshi data (next phase)

---

## Related Decisions

- [ADR-017: 2026 Season Roadmap](./017-2026-season-roadmap.md) — Trading system is the primary monetization vector for 2026 improvements

---

## References

### Documentation

- [CLAUDE.md — Kalshi Trading System section](../../CLAUDE.md)
- AGENT_INFRASTRUCTURE.md — Phase 6 (`docs/agents/AGENT_INFRASTRUCTURE.md`, removed in the 2026-10 prune; tag `archive/pre-prune-2026-10`)
- `docs/TRADING_ROADMAP.md` (removed in the 2026-10 prune; tag `archive/pre-prune-2026-10`)

### Code

- `nfl_analytics/trading/` — Strategy implementations
- `nfl_analytics/markets/` — Market client abstractions and Kalshi integration
- `nfl_analytics/risk/` — Risk management hierarchy
- `nfl_analytics/execution/` — Order management and live trading
- `nfl_analytics/monitoring/` — P&L, logging, and alerting

---

**Version:** 1.0
**Last Updated:** 2026-02-13
