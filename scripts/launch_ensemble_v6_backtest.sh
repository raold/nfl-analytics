#!/bin/bash
# Ensemble v6.0 Historical Backtest Launch Script
# ==================================================
#
# Launches comprehensive overnight backtest validating ensemble v6.0
# vs baseline v5.1 on 2022-2024 historical data (~148 games).
#
# Expected Runtime: 18 hours (~7.3 min/game)
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 15, 2025

set -euo pipefail

# Configuration
BACKTEST_SCRIPT="nfl_analytics/backtest/historical_validation_v6.py"
LOG_FILE="/tmp/ensemble_v6_backtest.log"
PID_FILE="/tmp/ensemble_v6_backtest.pid"
CHECKPOINT_FILE="/tmp/ensemble_v6_checkpoint.pkl"
OUTPUT_DIR="results"
OUTPUT_FILE="${OUTPUT_DIR}/ensemble_v6_backtest_2022_2024.csv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Ensemble v6.0 Historical Backtest - Launch Script"
echo "================================================================================"
echo ""

# Check if script exists
if [ ! -f "$BACKTEST_SCRIPT" ]; then
    echo -e "${RED}ERROR: Backtest script not found at $BACKTEST_SCRIPT${NC}"
    exit 1
fi

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: Backtest already running (PID: $OLD_PID)${NC}"
        echo "To kill: kill $OLD_PID"
        echo "To monitor: tail -f $LOG_FILE"
        exit 1
    else
        echo -e "${YELLOW}Removing stale PID file${NC}"
        rm "$PID_FILE"
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Show configuration
echo "Configuration:"
echo "  Script:     $BACKTEST_SCRIPT"
echo "  Log:        $LOG_FILE"
echo "  Checkpoint: $CHECKPOINT_FILE"
echo "  Output:     $OUTPUT_FILE"
echo "  PID file:   $PID_FILE"
echo ""

# Ask for confirmation
read -p "Launch 18-hour backtest? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Launch backtest in background
echo -e "${GREEN}Launching backtest...${NC}"

nohup python "$BACKTEST_SCRIPT" \
    --start-season 2022 \
    --end-season 2024 \
    --output "$OUTPUT_FILE" \
    --checkpoint "$CHECKPOINT_FILE" \
    > "$LOG_FILE" 2>&1 &

BACKTEST_PID=$!
echo "$BACKTEST_PID" > "$PID_FILE"

echo -e "${GREEN}✅ Backtest launched successfully!${NC}"
echo ""
echo "Process Information:"
echo "  PID: $BACKTEST_PID"
echo "  Status: Running in background"
echo ""
echo "Monitor Progress:"
echo "  ${YELLOW}tail -f $LOG_FILE${NC}"
echo ""
echo "Check Process:"
echo "  ${YELLOW}ps -p $BACKTEST_PID${NC}"
echo ""
echo "Kill Process:"
echo "  ${YELLOW}kill $BACKTEST_PID${NC}"
echo ""
echo "Expected Completion:"
echo "  ~18 hours from now ($(date -v+18H '+%Y-%m-%d %H:%M'))"
echo ""
echo "Results will be saved to:"
echo "  - ${OUTPUT_FILE}"
echo "  - ${OUTPUT_DIR}/ensemble_v6_backtest_summary_*.json"
echo "  - results/milestones/ENSEMBLE_V6_BACKTEST_COMPLETE.md (if approved)"
echo ""
echo "================================================================================"
