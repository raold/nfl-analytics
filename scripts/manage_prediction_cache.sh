#!/bin/bash
# Prediction Cache Manager CLI - Shell wrapper for prediction_cache_manager.py
#
# Usage:
#   ./scripts/manage_prediction_cache.sh stats
#   ./scripts/manage_prediction_cache.sh invalidate --model state_space_v2 --stat-type passing_yards
#   ./scripts/manage_prediction_cache.sh optimize
#   ./scripts/manage_prediction_cache.sh validate --sample-size 10000
#
# Author: Claude Code (Sonnet 4.5)
# Created: November 22, 2025

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python environment
if [ ! -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    log_error "Virtual environment not found at $PROJECT_ROOT/.venv"
    log_info "Run: uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt"
    exit 1
fi

PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Check database connection
check_database() {
    log_info "Checking database connection..."

    if ! pg_isready -h localhost -p 5544 -U dro -d devdb01 > /dev/null 2>&1; then
        log_error "PostgreSQL not running on localhost:5544"
        log_info "Start database: docker compose up -d pg"
        exit 1
    fi

    log_success "Database connection OK"
}

# Show usage
show_usage() {
    cat <<EOF
Prediction Cache Manager - PostgreSQL cache management for R model predictions

USAGE:
    $0 <command> [options]

COMMANDS:
    stats                    Show cache statistics (hit rate, size, predictions by model)
    invalidate               Clear cached predictions matching criteria
    optimize                 Optimize indexes and run VACUUM ANALYZE
    validate                 Validate cache data integrity
    backfill                 Backfill cache from rpy2 model (requires implementation)

OPTIONS:
    --model <name>           Filter by model name
    --stat-type <type>       Filter by stat type (passing_yards, rushing_yards, etc.)
    --season <year>          Filter by season year
    --older-than <hours>     Filter by age (hours)
    --sample-size <n>        Validation sample size (default: all predictions)
    -h, --help               Show this help message

EXAMPLES:
    # Show cache statistics
    $0 stats

    # Invalidate all state_space_v2 predictions
    $0 invalidate --model state_space_v2

    # Invalidate passing_yards predictions older than 24 hours
    $0 invalidate --stat-type passing_yards --older-than 24

    # Optimize indexes and vacuum
    $0 optimize

    # Validate random sample of 10,000 predictions
    $0 validate --sample-size 10000

    # Validate all predictions (may be slow for large caches)
    $0 validate

CACHE STATISTICS METRICS:
    - Total Predictions: Number of cached predictions
    - Unique Players: Number of distinct players
    - Unique Models: Number of distinct models
    - Cache Size: Disk space used (MB)
    - Hit Rate: Percentage of successful lookups
    - Avg Lookup Time: Average query time (ms, target <1ms)
    - Predictions by Model: Breakdown by model name
    - Predictions by Stat Type: Breakdown by stat type

VALIDATION CHECKS:
    ✅ NaN detection (no null/NaN predictions)
    ✅ Range validation (passing 0-600, rushing 0-300, etc.)
    ✅ Duplicate detection (unique composite keys)
    ✅ Orphaned predictions (player exists in database)
    ✅ Timestamp consistency (created_at <= last_updated)

PERFORMANCE TARGETS:
    - Lookup time: <1ms (composite index on player_id, stat_type, season, week)
    - Hit rate: >95% (for production ensemble predictions)
    - Cache size: <500MB (efficient storage of predictions)
    - Insert rate: >1000 predictions/sec (bulk insert optimization)

DATABASE SCHEMA:
    Table: ml.r_model_predictions
    Columns:
        - player_id (TEXT)
        - season (INTEGER)
        - week (INTEGER)
        - model_name (TEXT)
        - stat_type (TEXT)
        - prediction (NUMERIC)
        - created_at (TIMESTAMP)
        - last_updated (TIMESTAMP)

    Composite Primary Key: (player_id, season, week, model_name, stat_type)

    Indexes:
        - idx_r_predictions_lookup (player_id, stat_type, season, week)
        - idx_r_predictions_model (model_name, stat_type)
        - idx_r_predictions_updated (last_updated)
        - idx_r_predictions_season (season, week)

For detailed documentation, see: docs/guides/PREDICTION_CACHE_MANAGER.md

EOF
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi

    COMMAND=$1
    shift

    case "$COMMAND" in
        stats)
            check_database
            log_info "Fetching cache statistics..."
            $PYTHON "$PROJECT_ROOT/py/utils/prediction_cache_manager.py" stats "$@"
            ;;

        invalidate)
            check_database
            log_warning "This will DELETE cached predictions matching the criteria"
            read -p "Continue? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Invalidating cache..."
                $PYTHON "$PROJECT_ROOT/py/utils/prediction_cache_manager.py" invalidate "$@"
                log_success "Cache invalidation complete"
            else
                log_info "Cancelled"
                exit 0
            fi
            ;;

        optimize)
            check_database
            log_info "Optimizing cache indexes and running VACUUM ANALYZE..."
            log_warning "This may take several minutes for large caches"
            $PYTHON "$PROJECT_ROOT/py/utils/prediction_cache_manager.py" optimize "$@"
            log_success "Cache optimization complete"
            ;;

        validate)
            check_database
            log_info "Validating cache integrity..."
            $PYTHON "$PROJECT_ROOT/py/utils/prediction_cache_manager.py" validate "$@"
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                log_success "Cache validation PASSED"
            else
                log_error "Cache validation FAILED (exit code $EXIT_CODE)"
                exit $EXIT_CODE
            fi
            ;;

        backfill)
            check_database
            log_error "Backfill command requires custom implementation"
            log_info "See py/utils/prediction_cache_manager.py:backfill_from_rpy2()"
            exit 1
            ;;

        -h|--help|help)
            show_usage
            exit 0
            ;;

        *)
            log_error "Unknown command: $COMMAND"
            echo
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
