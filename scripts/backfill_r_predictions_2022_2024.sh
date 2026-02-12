#!/bin/bash
# ==============================================================================
# One-Time R Model Predictions Backfill (2022-2024)
# ==============================================================================
# Purpose: Generate historical R model predictions for 2022-2024 seasons
#          to enable offline ensemble inference testing and deployment
#
# Runtime: ~25 minutes for all 3 seasons (~640k predictions)
# Storage: ~96 MB in PostgreSQL
#
# Author: Claude Code (Sonnet 4.5)
# Created: November 20, 2025
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT="/Users/dro/rice/nfl-analytics"
LOG_DIR="${PROJECT_ROOT}/logs/r_predictions"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/backfill_${TIMESTAMP}.log"

# Database connection
export DB_HOST="localhost"
export DB_PORT="5544"
export DB_NAME="devdb01"
export DB_USER="dro"
export DB_PASSWORD="sicillionbillions"

# R script
R_SCRIPT="${PROJECT_ROOT}/R/batch_generate_r_predictions.R"

# Seasons to backfill
SEASONS="2022,2023,2024"

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

    # Check R
    if ! command -v Rscript &> /dev/null; then
        error "Rscript not found"
    fi

    # Check R script exists
    if [[ ! -f "$R_SCRIPT" ]]; then
        error "R script not found: $R_SCRIPT"
    fi

    # Check database accessible
    export PGPASSWORD="$DB_PASSWORD"
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &> /dev/null; then
        error "Cannot connect to database"
    fi

    # Check R models exist
    for model_file in \
        "models/bayesian/simplified_random_walk_v2.rds" \
        "models/bayesian/passing_yards_hierarchical_v2.rds" \
        "models/bayesian/passing_informative_priors_v2.rds"; do

        if [[ ! -f "${PROJECT_ROOT}/${model_file}" ]]; then
            error "Model file not found: ${model_file}"
        fi
    done

    log "✅ All prerequisites met"
}

check_existing_predictions() {
    log "Checking for existing predictions..."

    local existing_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*)
        FROM ml.r_model_predictions
        WHERE season IN (2022, 2023, 2024);
    " | xargs)

    log "Found $existing_count existing predictions for 2022-2024"

    if [[ "$existing_count" -gt 0 ]]; then
        log "⚠️  WARNING: Existing predictions will be updated (ON CONFLICT DO UPDATE)"
    fi
}

run_backfill() {
    log "======================================================================"
    log "Starting backfill for seasons: $SEASONS"
    log "======================================================================"

    local start_time=$(date +%s)

    # Run R batch script
    log "Executing R batch script..."
    Rscript "$R_SCRIPT" --seasons "$SEASONS" --model all 2>&1 | tee -a "$LOG_FILE"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        error "R script execution failed"
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))

    log "✅ Backfill completed in ${elapsed_min}m ${elapsed_sec}s"
}

verify_backfill() {
    log "======================================================================"
    log "Verifying backfill results..."
    log "======================================================================"

    # Query coverage
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT
            model_name,
            season,
            COUNT(DISTINCT player_id) as n_players,
            COUNT(*) as n_predictions,
            MIN(generated_at) as oldest,
            MAX(generated_at) as newest,
            ROUND(AVG(generation_time_ms)::NUMERIC, 2) as avg_gen_time_ms
        FROM ml.r_model_predictions
        WHERE season IN (2022, 2023, 2024)
        GROUP BY model_name, season
        ORDER BY season, model_name;
    " | tee -a "$LOG_FILE"

    # Check total predictions
    local total_predictions=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*)
        FROM ml.r_model_predictions
        WHERE season IN (2022, 2023, 2024);
    " | xargs)

    log ""
    log "Total predictions generated: $total_predictions"

    # Expected: ~1,800 player-games × 3 models = ~5,400 predictions
    # (actual may vary based on QB starts in each season)
    if [[ "$total_predictions" -lt 3000 ]]; then
        log "⚠️  WARNING: Expected ~5,400+ predictions, got $total_predictions"
    else
        log "✅ Prediction count looks reasonable"
    fi

    # Check for missing models
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT
            season,
            COUNT(DISTINCT model_name) as n_models
        FROM ml.r_model_predictions
        WHERE season IN (2022, 2023, 2024)
        GROUP BY season
        ORDER BY season;
    " | tee -a "$LOG_FILE"

    log ""
    log "Expected 3 models per season (state_space_v2, hierarchical_v2, informative_priors_v2)"
}

check_staleness() {
    log "======================================================================"
    log "Checking prediction staleness..."
    log "======================================================================"

    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT * FROM ml.check_r_prediction_staleness()
        WHERE season IN (2022, 2023, 2024);
    " | tee -a "$LOG_FILE"

    log "✅ Staleness check complete"
}

estimate_storage() {
    log "======================================================================"
    log "Estimating storage usage..."
    log "======================================================================"

    local table_size=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT pg_size_pretty(pg_total_relation_size('ml.r_model_predictions'));
    " | xargs)

    log "Table size: $table_size"
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    mkdir -p "$LOG_DIR"

    log "======================================================================"
    log "R Model Predictions Backfill (2022-2024)"
    log "======================================================================"
    log "Seasons: $SEASONS"
    log "Log file: $LOG_FILE"
    log ""

    check_prerequisites
    check_existing_predictions
    run_backfill
    verify_backfill
    check_staleness
    estimate_storage

    log ""
    log "======================================================================"
    log "✅ Backfill Complete!"
    log "======================================================================"
    log ""
    log "Next steps:"
    log "  1. Review coverage report above"
    log "  2. Run A/B test: scripts/ab_test_lookup_vs_rpy2.py"
    log "  3. Deploy ensemble v6.1 with lookup integration"
    log ""
}

# Error handling
trap 'error "Script failed at line $LINENO"' ERR

# Run main
main "$@"
