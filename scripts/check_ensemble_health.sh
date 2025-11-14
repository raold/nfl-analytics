#!/bin/bash
#
# Ensemble Health Check - Pre-Backtest Validation
#
# CRITICAL: Run this BEFORE expensive backtests to prevent R model loading failures.
#
# Usage:
#   ./scripts/check_ensemble_health.sh                    # Quick health check
#   ./scripts/check_ensemble_health.sh --save-report      # Save report to file
#   ./scripts/check_ensemble_health.sh --monitor file.pkl # Monitor from history
#
# Examples:
#   # Pre-backtest validation (CRITICAL)
#   ./scripts/check_ensemble_health.sh
#   if [ $? -eq 0 ]; then
#       echo "Safe to run backtest"
#       python py/backtest/historical_validation_4way.py
#   else
#       echo "DO NOT run backtest - fix issues first"
#       exit 1
#   fi
#
#   # Save health report
#   ./scripts/check_ensemble_health.sh --save-report ensemble_health.pkl
#
#   # Monitor predictions during backtest
#   ./scripts/check_ensemble_health.sh --monitor prediction_history.pkl
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 14, 2025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Parse arguments
MODE="--check"
OUTPUT=""
HISTORY_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --save-report)
            OUTPUT="${2:-ensemble_health_report_$(date +%Y%m%d_%H%M%S).pkl}"
            shift
            ;;
        --monitor)
            MODE="--monitor"
            HISTORY_FILE="$2"
            shift
            ;;
        --report)
            MODE="--report"
            HISTORY_FILE="$2"
            shift
            ;;
        --help)
            echo "Usage: $0 [--save-report OUTPUT] [--monitor FILE] [--report FILE]"
            echo ""
            echo "Options:"
            echo "  (no args)           Run pre-backtest health check"
            echo "  --save-report FILE  Save health report to FILE"
            echo "  --monitor FILE      Monitor predictions from history file"
            echo "  --report FILE       Generate report from history file"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Build command
CMD="python py/utils/ensemble_health_monitor.py $MODE"

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

if [ -n "$HISTORY_FILE" ]; then
    CMD="$CMD --history-file $HISTORY_FILE"
fi

echo "========================================"
echo "Ensemble Health Monitor"
echo "========================================"
echo "Mode: $MODE"
echo "Project: $PROJECT_ROOT"
echo ""

# Run health check
$CMD

exit $?
