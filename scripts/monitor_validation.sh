#!/bin/bash
# Monitor ensemble v6.0 validation progress

echo "====================================================================="
echo "Ensemble v6.0 Validation Monitor"
echo "====================================================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "validate_ensemble_v6_recovered_xgb.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ Validation process is NOT running"
    echo ""
    echo "Checking for completion..."
    if [ -f "results/validation_v6_recovered_xgb/validation_summary.json" ]; then
        echo "✅ Validation COMPLETE - results available"
        echo ""
        echo "Summary:"
        cat results/validation_v6_recovered_xgb/validation_summary.json | python3 -m json.tool | head -30
    else
        echo "⚠️  No results found - validation may have failed"
        echo ""
        echo "Last 20 lines of log:"
        tail -20 validation_v6_recovered_xgb.log
    fi
    exit 0
fi

# Process is running - show stats
echo "✅ Validation is RUNNING (PID: $PID)"
echo ""

# CPU and memory
CPU=$(ps aux | grep "$PID" | grep -v grep | awk '{print $3}')
MEM=$(ps aux | grep "$PID" | grep -v grep | awk '{print $4}')
echo "Resource Usage:"
echo "  CPU: ${CPU}%"
echo "  Memory: ${MEM}%"
echo ""

# Log stats
LOG_LINES=$(wc -l validation_v6_recovered_xgb.log | awk '{print $1}')
WARNINGS=$(grep -c "Component state_space_v2 failed" validation_v6_recovered_xgb.log)
echo "Progress Indicators:"
echo "  Log lines: $LOG_LINES"
echo "  Predictions processed: ~$WARNINGS (approx)"
echo ""

# Estimate progress (assuming ~540 games per season, 3 seasons = 1620 total)
TOTAL_EXPECTED=1632
PROGRESS_PCT=$(awk "BEGIN {printf \"%.1f\", ($WARNINGS / $TOTAL_EXPECTED) * 100}")
echo "  Estimated progress: ${PROGRESS_PCT}% of 1,632 games"
echo ""

# Latest activity
echo "Latest activity (last 10 warnings):"
tail -100 validation_v6_recovered_xgb.log | grep "WARNING\|INFO" | tail -10
echo ""

echo "====================================================================="
echo "To view full log: tail -f validation_v6_recovered_xgb.log"
echo "To re-run monitor: bash scripts/monitor_validation.sh"
echo "====================================================================="
