# Trading System Development Roadmap

**Last Updated:** February 13, 2026

## Completed (Feb 13, 2026)

- [x] Phase 1: Foundation (base.py, signal, strategy ABC, paper client) — 67 tests
- [x] Phase 2: NFL Strategy (directional, risk hierarchy, drawdown) — 23 tests
- [x] Phase 3: Kalshi Live (auth, client, schemas, websocket) — 99 tests
- [x] Phase 4: Market Making (market_maker, hybrid) — 52 tests
- [x] Phase 5: Monitoring & Safety (PnL, logger, alerts, live_trader) — 69 tests
- [x] Phase 6: Multi-Sport (market_router) — 15 tests
- [x] Documentation audit + CLAUDE.md update (ADR-018)
- [x] Context preservation (Claude memory, Serena, knowledge graph)
- [x] Trading agents (validator, monitor, auditor)
- [x] Claude signal validator PoC (ADR-019)

**Total:** 437 tests, 37 production files, ~9,200 LOC

## Next Up: Integration & Hardening

- [ ] **Integration tests:** Paper trading end-to-end (strategy -> risk -> executor -> paper_client)
- [ ] **Backtest harness:** Replay historical Kalshi data through the pipeline
- [ ] **Kalshi sandbox testing:** Connect to Kalshi demo API (`demo-api.kalshi.co`)
- [ ] **Position sizing calibration:** Fit Kelly model on historical edges
- [ ] **WebSocket integration test:** Real-time market data via `KalshiWebSocket`
- [ ] **Multi-strategy live test:** Run directional + market maker simultaneously via `MarketRouter`

## Future: Production Deployment

- [ ] **Paper trading burn-in:** 2+ weeks of simulated trading with `PaperClient`
- [ ] **Demo API testing:** Connect to Kalshi demo with real auth, validate order flow
- [ ] **Live deployment checklist:**
  - Kill switch thresholds validated on paper trading data
  - Reconciliation agent running automatically
  - Alert webhook configured and tested
  - Logging pipeline verified (JSONL -> analysis)
  - Regulatory compliance audit trail complete

## Future: Claude API Integration (ADR-019)

- [ ] **ClaudeSignalValidator PoC:** Extended thinking for signal quality assessment
- [ ] **Kalshi MCP server adapter:** Direct market data access from Claude
- [ ] **Meta-strategy evaluator:** Ensemble weight drift detection via Claude reasoning
- [ ] **Semantic feature validation:** Claude analysis of feature pipeline health

## Future: Advanced Features

- [ ] **Portfolio optimization:** Move from per-signal Kelly to portfolio-level CVaR optimization
- [ ] **Multi-exchange:** Add Polymarket/PredictIt `MarketClient` implementations
- [ ] **Feature pipeline v2:** Real-time feature computation (currently batch/cached)
- [ ] **Model retraining loop:** Automated model updates based on prediction performance
- [ ] **Dashboard:** Live P&L dashboard using `performance_dashboard.py` + web UI

## Architecture Decisions

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-018](adr/018-kalshi-trading-system.md) | 6-phase modular trading system | Accepted |
| [ADR-019](adr/019-claude-api-trading-integration.md) | Claude API integration roadmap | Proposed |

## Resumption Guide

To resume development from a new session:

1. **Read this file** to understand what's done and what's next
2. **Run tests** to verify baseline:
   ```bash
   uv run pytest tests/unit/risk/ tests/unit/trading/ tests/unit/markets/ tests/unit/execution/ tests/unit/monitoring/ -o "addopts=" -q
   ```
3. **Read CLAUDE.md** "Kalshi Trading System" section for architecture overview
4. **Key entry points:**
   - `LiveTrader` (`nfl_analytics/execution/live_trader.py`) — async orchestrator
   - `DirectionalStrategy` (`nfl_analytics/trading/directional.py`) — NFL signal generation
   - `RiskManager` (`nfl_analytics/risk/risk_manager.py`) — universal signal gate
   - `MarketRouter` (`nfl_analytics/markets/market_router.py`) — multi-sport dispatch
5. **Context systems** have detailed memories (Serena: `trading-architecture`, `trading-risk-hierarchy`, `trading-testing-patterns`)
