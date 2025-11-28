#!/bin/bash
#
# Automated Weekly Prediction Generation
#
# Schedule: Tuesday 12:00 PM ET (after R model regeneration at 3 AM)
# Purpose: Generate ensemble predictions for upcoming NFL week using fresh R model predictions
#
# Cron entry:
# 0 12 * * 2 cd /Users/dro/rice/nfl-analytics && ./scripts/generate_weekly_predictions_automated.sh >> /tmp/weekly_predictions.log 2>&1
#
# Created: November 22, 2025
# Author: Claude Code (Sonnet 4.5)

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
PROJECT_DIR="/Users/dro/rice/nfl-analytics"
LOG_DIR="$PROJECT_DIR/logs/predictions"
PYTHON_VENV="$PROJECT_DIR/.venv/bin/python"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Timestamp for logging
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "=============================================================================="
echo "[$TIMESTAMP] Starting automated weekly prediction generation"
echo "=============================================================================="

# Determine current NFL season and week
# Note: This is a simplified version - assumes season runs Sep-Feb
CURRENT_YEAR=$(date +%Y)
CURRENT_MONTH=$(date +%m)

# Determine season (NFL season spans two calendar years)
if [ "$CURRENT_MONTH" -lt 3 ]; then
    # Jan-Feb: Previous year's season
    SEASON=$((CURRENT_YEAR - 1))
else
    # Mar-Dec: Current year's season
    SEASON=$CURRENT_YEAR
fi

# Determine week (simplified - assumes Week 1 starts first week of September)
# For production, query database or NFL API for accurate week number
WEEK_OF_YEAR=$(date +%U)
FIRST_WEEK_OF_SEP=35  # Approximate week number for first week of September

if [ "$WEEK_OF_YEAR" -lt "$FIRST_WEEK_OF_SEP" ]; then
    # Before September: Offseason or previous season playoffs
    echo "[$TIMESTAMP] WARNING: Currently in offseason (before September)"
    echo "[$TIMESTAMP] Skipping prediction generation"
    exit 0
fi

# Calculate NFL week (Week 1 = first week of September)
NFL_WEEK=$((WEEK_OF_YEAR - FIRST_WEEK_OF_SEP + 1))

# Cap at Week 18 (regular season)
if [ "$NFL_WEEK" -gt 18 ]; then
    echo "[$TIMESTAMP] WARNING: Calculated week $NFL_WEEK exceeds regular season"
    echo "[$TIMESTAMP] Setting to Week 18 (playoffs)"
    NFL_WEEK=18
fi

echo "[$TIMESTAMP] Detected Season: $SEASON, Week: $NFL_WEEK"

# Step 1: Verify R model predictions are fresh
echo "[$TIMESTAMP] Checking R model prediction freshness..."

# Query database for latest R prediction timestamp
LATEST_R_PREDICTION=$(PGPASSWORD="sicillionbillions" psql -h localhost -p 5544 -U dro -d devdb01 -t -c \
    "SELECT MAX(generated_at) FROM ml.r_model_predictions WHERE season = $SEASON;" | xargs)

if [ -z "$LATEST_R_PREDICTION" ]; then
    echo "[$TIMESTAMP] ERROR: No R model predictions found for season $SEASON"
    echo "[$TIMESTAMP] Run: Rscript R/batch_generate_r_predictions_optimized.R --seasons $SEASON"
    exit 1
fi

# Calculate age of R predictions (hours)
R_PREDICTION_AGE=$(PGPASSWORD="sicillionbillions" psql -h localhost -p 5544 -U dro -d devdb01 -t -c \
    "SELECT EXTRACT(EPOCH FROM (NOW() - MAX(generated_at)))/3600 FROM ml.r_model_predictions WHERE season = $SEASON;" | xargs)

R_PREDICTION_AGE_INT=$(printf "%.0f" "$R_PREDICTION_AGE")

echo "[$TIMESTAMP] Latest R prediction: $LATEST_R_PREDICTION (${R_PREDICTION_AGE_INT}h old)"

if [ "$R_PREDICTION_AGE_INT" -gt 24 ]; then
    echo "[$TIMESTAMP] WARNING: R predictions are ${R_PREDICTION_AGE_INT}h old (>24h)"
    echo "[$TIMESTAMP] Consider regenerating: Rscript R/batch_generate_r_predictions_optimized.R --seasons $SEASON"
fi

# Step 2: Generate ensemble predictions
echo "[$TIMESTAMP] Generating ensemble predictions for Season $SEASON, Week $NFL_WEEK..."

cd "$PROJECT_DIR"

# Activate virtual environment and run prediction generation
source .venv/bin/activate

# Generate predictions (assuming script exists - create if needed)
if [ -f "scripts/generate_weekly_predictions.py" ]; then
    uv run python scripts/generate_weekly_predictions.py \
        --season "$SEASON" \
        --week "$NFL_WEEK" \
        2>&1 | tee -a "$LOG_DIR/predictions_S${SEASON}W${NFL_WEEK}_$(date +%Y%m%d_%H%M%S).log"
else
    echo "[$TIMESTAMP] WARNING: scripts/generate_weekly_predictions.py not found"
    echo "[$TIMESTAMP] Using game predictions script instead..."

    chmod +x scripts/generate_game_predictions_v6.py
    uv run python scripts/generate_game_predictions_v6.py \
        --season "$SEASON" \
        --week "$NFL_WEEK" \
        2>&1 | tee -a "$LOG_DIR/predictions_S${SEASON}W${NFL_WEEK}_$(date +%Y%m%d_%H%M%S).log"
fi

PREDICTION_STATUS=$?

if [ $PREDICTION_STATUS -eq 0 ]; then
    echo "[$TIMESTAMP] ✅ Prediction generation completed successfully"

    # Step 3: Verify predictions were saved to database
    PREDICTION_COUNT=$(PGPASSWORD="sicillionbillions" psql -h localhost -p 5544 -U dro -d devdb01 -t -c \
        "SELECT COUNT(*) FROM ml.game_predictions WHERE season = $SEASON AND week = $NFL_WEEK;" | xargs)

    echo "[$TIMESTAMP] Saved $PREDICTION_COUNT game predictions to database"

    # Optional: Send notification (Slack, email, etc.)
    # Uncomment and configure if desired:
    # curl -X POST -H 'Content-type: application/json' \
    #     --data "{\"text\":\"✅ Week $NFL_WEEK predictions generated ($PREDICTION_COUNT games)\"}" \
    #     YOUR_SLACK_WEBHOOK_URL

else
    echo "[$TIMESTAMP] ❌ Prediction generation failed with exit code $PREDICTION_STATUS"

    # Optional: Send error notification
    # curl -X POST -H 'Content-type: application/json' \
    #     --data "{\"text\":\"❌ Week $NFL_WEEK prediction generation FAILED\"}" \
    #     YOUR_SLACK_WEBHOOK_URL

    exit $PREDICTION_STATUS
fi

echo "=============================================================================="
echo "[$TIMESTAMP] Automated prediction generation complete"
echo "=============================================================================="

exit 0
