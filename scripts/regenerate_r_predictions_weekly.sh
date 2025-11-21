#!/bin/bash
# ==============================================================================
# Weekly R Model Predictions Regeneration Script
# ==============================================================================
# Purpose: Automated weekly regeneration of R model predictions for offline
#          ensemble inference. Runs after Monday Night Football concludes.
#
# Schedule: Tuesday 3:00 AM (after MNF)
# Cron: 0 3 * * 2 /path/to/regenerate_r_predictions_weekly.sh
#
# Author: Claude Code (Sonnet 4.5)
# Created: November 20, 2025
# ==============================================================================

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT="/Users/dro/rice/nfl-analytics"
LOG_DIR="${PROJECT_ROOT}/logs/r_predictions"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/regeneration_${TIMESTAMP}.log"

# Database connection
export DB_HOST="localhost"
export DB_PORT="5544"
export DB_NAME="devdb01"
export DB_USER="dro"
export DB_PASSWORD="sicillionbillions"

# R script path
R_SCRIPT="${PROJECT_ROOT}/R/batch_generate_r_predictions.R"

# Current season (adjust each year)
CURRENT_SEASON=$(date +%Y)

# Docker container names
FASTAPI_CONTAINER="nfl-analytics-api"
STREAMLIT_CONTAINER="nfl-analytics-dashboard"

# Notification settings (optional - configure for production)
NOTIFY_EMAIL=""  # Set to your email for notifications
STALENESS_THRESHOLD_DAYS=7

# ==============================================================================
# Functions
# ==============================================================================

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $*"
    exit 1
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check R is installed
    if ! command -v Rscript &> /dev/null; then
        error "Rscript not found. Please install R."
    fi

    # Check R script exists
    if [[ ! -f "$R_SCRIPT" ]]; then
        error "R script not found: $R_SCRIPT"
    fi

    # Check PostgreSQL is accessible
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &> /dev/null; then
        error "Cannot connect to PostgreSQL database"
    fi

    log "✅ All prerequisites met"
}

run_r_predictions() {
    log "Starting R model prediction generation for season $CURRENT_SEASON..."

    # Run R batch script
    Rscript "$R_SCRIPT" --seasons "$CURRENT_SEASON" --model all 2>&1 | tee -a "$LOG_FILE"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        error "R script execution failed"
    fi

    log "✅ R predictions generated successfully"
}

check_staleness() {
    log "Checking for stale predictions..."

    # Query staleness check function
    STALENESS_RESULT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*)
        FROM ml.check_r_prediction_staleness();
    ")

    STALE_COUNT=$(echo "$STALENESS_RESULT" | xargs)

    if [[ "$STALE_COUNT" -gt 0 ]]; then
        log "⚠️  WARNING: $STALE_COUNT model-season combinations have stale predictions (>$STALENESS_THRESHOLD_DAYS days old)"

        # Get details
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
            SELECT * FROM ml.check_r_prediction_staleness();
        " | tee -a "$LOG_FILE"
    else
        log "✅ No stale predictions detected"
    fi
}

check_coverage() {
    log "Checking prediction coverage..."

    # Query coverage for current season
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT
            model_name,
            n_players,
            n_predictions,
            oldest_prediction,
            newest_prediction,
            avg_generation_time_ms
        FROM ml.r_prediction_coverage
        WHERE season = $CURRENT_SEASON
        ORDER BY model_name;
    " | tee -a "$LOG_FILE"

    log "✅ Coverage check complete"
}

restart_docker_services() {
    log "Restarting Docker services to clear Python cache..."

    # Check if Docker containers exist
    if docker ps -a --format '{{.Names}}' | grep -q "^${FASTAPI_CONTAINER}$"; then
        log "Restarting FastAPI container..."
        docker restart "$FASTAPI_CONTAINER" >> "$LOG_FILE" 2>&1
        log "✅ FastAPI restarted"
    else
        log "⚠️  FastAPI container not found: $FASTAPI_CONTAINER"
    fi

    if docker ps -a --format '{{.Names}}' | grep -q "^${STREAMLIT_CONTAINER}$"; then
        log "Restarting Streamlit container..."
        docker restart "$STREAMLIT_CONTAINER" >> "$LOG_FILE" 2>&1
        log "✅ Streamlit restarted"
    else
        log "⚠️  Streamlit container not found: $STREAMLIT_CONTAINER"
    fi
}

send_notification() {
    local status=$1
    local message=$2

    if [[ -n "$NOTIFY_EMAIL" ]]; then
        log "Sending notification to $NOTIFY_EMAIL..."

        # Use mail command if available
        if command -v mail &> /dev/null; then
            echo "$message" | mail -s "R Predictions Regeneration: $status" "$NOTIFY_EMAIL"
            log "✅ Notification sent"
        else
            log "⚠️  mail command not found, skipping notification"
        fi
    fi
}

cleanup_old_logs() {
    log "Cleaning up old log files (>30 days)..."

    find "$LOG_DIR" -name "regeneration_*.log" -mtime +30 -delete

    log "✅ Old logs cleaned up"
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"

    log "======================================================================"
    log "Starting Weekly R Predictions Regeneration"
    log "======================================================================"
    log "Season: $CURRENT_SEASON"
    log "Log file: $LOG_FILE"
    log ""

    # Run regeneration workflow
    check_prerequisites
    run_r_predictions
    check_staleness
    check_coverage
    restart_docker_services
    cleanup_old_logs

    log ""
    log "======================================================================"
    log "✅ Weekly R Predictions Regeneration Complete"
    log "======================================================================"

    # Send success notification
    send_notification "SUCCESS" "R predictions regenerated successfully for season $CURRENT_SEASON. See log: $LOG_FILE"
}

# Error handling
trap 'error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"
