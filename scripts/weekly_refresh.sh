#!/bin/bash
#
# Weekly Prediction Refresh Script
#
# Designed to run automatically every Tuesday morning (after Monday Night Football)
# to refresh predictions for upcoming weeks.
#
# Cron schedule (Tuesdays at 3 AM):
# 0 3 * * 2 cd /Users/dro/rice/nfl-analytics && ./scripts/weekly_refresh.sh >> logs/weekly_refresh.log 2>&1
#
# Or use launchd on macOS - see scripts/com.nfl-analytics.weekly-refresh.plist

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/Users/dro/rice/nfl-analytics"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/weekly_refresh_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Start logging
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "================================================================================"
echo "NFL WEEKLY PREDICTION REFRESH"
echo "================================================================================"
echo "Started at: $(date)"
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Activate virtual environment (if using venv instead of uv)
# source .venv/bin/activate

# Run the refresh pipeline
echo "[1/1] Running prediction refresh pipeline..."
uv run python py/pipeline/refresh_predictions.py --auto --weeks-ahead 2

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ WEEKLY REFRESH COMPLETED SUCCESSFULLY"
else
    echo "✗ WEEKLY REFRESH FAILED (exit code: $EXIT_CODE)"
fi
echo "Finished at: $(date)"
echo "================================================================================"

exit $EXIT_CODE
