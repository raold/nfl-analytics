#!/bin/bash
# Refresh all materialized views
# Schedule via cron: 0 3 * * * during season (daily at 3 AM)

set -e  # Exit on error

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Database connection parameters
PGHOST=${POSTGRES_HOST:-localhost}
PGPORT=${POSTGRES_PORT:-5544}
PGDATABASE=${POSTGRES_DB:-devdb01}
PGUSER=${POSTGRES_USER:-dro}
PGPASSWORD=${POSTGRES_PASSWORD}

# Export for psql
export PGPASSWORD

# Log file
LOG_FILE="logs/mv_refresh_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "========================================" | tee -a "$LOG_FILE"
echo "Materialized View Refresh - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run refresh function
echo "Refreshing all materialized views..." | tee -a "$LOG_FILE"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" "$PGDATABASE" -c "SELECT * FROM refresh_all_materialized_views();" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo "" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Materialized views refreshed successfully" | tee -a "$LOG_FILE"
else
    echo "❌ Materialized view refresh failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi
echo "========================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
