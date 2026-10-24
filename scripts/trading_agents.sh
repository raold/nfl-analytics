#!/usr/bin/env bash
# CLI wrapper for trading system agents.
# Usage: ./scripts/trading_agents.sh {validate|monitor|audit|all}

set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-help}" in
    validate)
        echo "Running signal validator..."
        uv run python -c "
from nfl_analytics.agents.trading_validator import TradingValidator
v = TradingValidator()
print(v.summary())
"
        ;;
    monitor)
        echo "Running portfolio monitor..."
        uv run python -c "
from nfl_analytics.agents.portfolio_monitor import PortfolioMonitor
from nfl_analytics.monitoring.pnl_tracker import PnLTracker
pnl = PnLTracker(initial_equity_cents=100_000)
m = PortfolioMonitor(pnl_tracker=pnl)
report = m.check_health()
print(report.summary())
"
        ;;
    audit)
        echo "Running execution auditor..."
        uv run python -c "
import asyncio
from nfl_analytics.agents.execution_auditor import ExecutionAuditor
from nfl_analytics.markets.paper_client import PaperMarketClient
client = PaperMarketClient()
auditor = ExecutionAuditor(client=client)
report = asyncio.run(auditor.run_audit())
print(report.summary())
"
        ;;
    all)
        $0 validate
        $0 monitor
        $0 audit
        ;;
    *)
        echo "Usage: $0 {validate|monitor|audit|all}"
        exit 1
        ;;
esac
