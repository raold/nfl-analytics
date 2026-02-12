#!/usr/bin/env bash
# Cache Invalidation CLI Tool
# Usage: ./scripts/clear_cache.sh [OPTIONS]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Help message
show_help() {
    cat << EOF
Cache Invalidation CLI Tool

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --pattern <pattern>     Invalidate keys matching Redis pattern
                           Examples: "model:bnn:*", "query:games_*"

    --model <name>          Invalidate specific model
                           Example: bnn_passing_v1

    --model-type <type>     Filter by model type (bnn, xgboost, r_model)
                           Use with --model flag

    --player <player_id>    Invalidate player-specific caches
                           Example: 00-0019596 (Patrick Mahomes)

    --season <year>         Filter by season (use with --player)
    --week <num>            Filter by week (use with --player or --season)

    --queries <key1,key2>   Invalidate specific query caches
                           Example: games_season_2024,player_stats_2024_week_10

    --streamlit             Invalidate all Streamlit dashboard caches

    --all                   ⚠️  Clear ENTIRE cache (requires --confirm)
    --confirm               Confirm dangerous operations (--all)

    --dry-run               Show what would be deleted without actually deleting
    --stats                 Show cache statistics

    -h, --help              Show this help message

EXAMPLES:
    # Clear all BNN models
    $0 --pattern "model:bnn:*"

    # Clear specific model
    $0 --model bnn_passing_v1 --model-type bnn

    # Clear player cache for Patrick Mahomes (2024 Week 10)
    $0 --player 00-0019596 --season 2024 --week 10

    # Clear dashboard cache after data refresh
    $0 --streamlit

    # Clear specific queries
    $0 --queries games_season_2024,player_stats_week_10

    # See what would be deleted (dry run)
    $0 --pattern "model:*" --dry-run

    # Nuclear option: clear everything
    $0 --all --confirm

    # Show cache statistics
    $0 --stats

COMMON WORKFLOWS:
    # After model retraining
    $0 --model bnn_passing_v2 --model-type bnn

    # After data ingestion (Week 10, 2024)
    $0 --pattern "player_features:*:s2024w10"
    $0 --pattern "query:*"
    $0 --streamlit

    # Before production deployment
    $0 --stats
    $0 --pattern "model:*" --dry-run

EOF
}

# Parse arguments
PATTERN=""
MODEL=""
MODEL_TYPE=""
PLAYER=""
SEASON=""
WEEK=""
QUERIES=""
STREAMLIT=false
ALL=false
CONFIRM=false
DRY_RUN=false
STATS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --player)
            PLAYER="$2"
            shift 2
            ;;
        --season)
            SEASON="$2"
            shift 2
            ;;
        --week)
            WEEK="$2"
            shift 2
            ;;
        --queries)
            QUERIES="$2"
            shift 2
            ;;
        --streamlit)
            STREAMLIT=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --confirm)
            CONFIRM=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --stats)
            STATS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Build Python command
PYTHON_CMD="python3 -c \"
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator, invalidate_after_model_training, invalidate_after_data_ingestion

invalidator = CacheInvalidator()

# Execute requested operation
"

# Handle --stats flag
if [ "$STATS" = true ]; then
    echo -e "${BLUE}📊 Cache Statistics${NC}"
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator

invalidator = CacheInvalidator()
stats = invalidator.stats()

print('Cache Statistics:')
for key, value in stats.items():
    print(f'  {key}: {value}')
"
    exit 0
fi

# Handle --pattern flag
if [ -n "$PATTERN" ]; then
    echo -e "${YELLOW}🔍 Invalidating pattern: $PATTERN${NC}"
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}(DRY RUN - no keys will be deleted)${NC}"
    fi

    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator

invalidator = CacheInvalidator()
result = invalidator.invalidate_pattern('$PATTERN', dry_run=$DRY_RUN)

print(f'Matched: {result.keys_matched} keys')
print(f'Deleted: {result.keys_deleted} keys')
print(f'Time: {result.execution_time_ms:.2f}ms')
if result.errors:
    print(f'Errors: {result.errors}')
"
    exit 0
fi

# Handle --model flag
if [ -n "$MODEL" ]; then
    echo -e "${YELLOW}🤖 Invalidating model: $MODEL${NC}"
    if [ -n "$MODEL_TYPE" ]; then
        echo -e "${BLUE}Model type filter: $MODEL_TYPE${NC}"
    fi

    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator

invalidator = CacheInvalidator()
result = invalidator.invalidate_model('$MODEL', '$MODEL_TYPE' if '$MODEL_TYPE' else None)

print(f'Matched: {result.keys_matched} keys')
print(f'Deleted: {result.keys_deleted} keys')
print(f'Time: {result.execution_time_ms:.2f}ms')
if result.errors:
    print(f'Errors: {result.errors}')
"
    exit 0
fi

# Handle --player flag
if [ -n "$PLAYER" ]; then
    echo -e "${YELLOW}👤 Invalidating player: $PLAYER${NC}"
    FILTER_MSG=""
    if [ -n "$SEASON" ]; then
        FILTER_MSG="${FILTER_MSG}Season: $SEASON "
    fi
    if [ -n "$WEEK" ]; then
        FILTER_MSG="${FILTER_MSG}Week: $WEEK"
    fi
    if [ -n "$FILTER_MSG" ]; then
        echo -e "${BLUE}Filters: $FILTER_MSG${NC}"
    fi

    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator

invalidator = CacheInvalidator()
season = int('$SEASON') if '$SEASON' else None
week = int('$WEEK') if '$WEEK' else None
result = invalidator.invalidate_player('$PLAYER', season=season, week=week)

print(f'Patterns: {result.patterns_used}')
print(f'Matched: {result.keys_matched} keys')
print(f'Deleted: {result.keys_deleted} keys')
print(f'Time: {result.execution_time_ms:.2f}ms')
if result.errors:
    print(f'Errors: {result.errors}')
"
    exit 0
fi

# Handle --queries flag
if [ -n "$QUERIES" ]; then
    echo -e "${YELLOW}📝 Invalidating queries: $QUERIES${NC}"

    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator

invalidator = CacheInvalidator()
query_list = '$QUERIES'.split(',')
result = invalidator.invalidate_queries(query_list)

print(f'Matched: {result.keys_matched} keys')
print(f'Deleted: {result.keys_deleted} keys')
print(f'Time: {result.execution_time_ms:.2f}ms')
if result.errors:
    print(f'Errors: {result.errors}')
"
    exit 0
fi

# Handle --streamlit flag
if [ "$STREAMLIT" = true ]; then
    echo -e "${YELLOW}📊 Invalidating Streamlit dashboard cache${NC}"

    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator

invalidator = CacheInvalidator()
result = invalidator.invalidate_streamlit()

print(f'Matched: {result.keys_matched} keys')
print(f'Deleted: {result.keys_deleted} keys')
print(f'Time: {result.execution_time_ms:.2f}ms')
if result.errors:
    print(f'Errors: {result.errors}')
"
    exit 0
fi

# Handle --all flag
if [ "$ALL" = true ]; then
    if [ "$CONFIRM" != true ]; then
        echo -e "${RED}⚠️  ERROR: --all requires --confirm flag${NC}"
        echo "This will delete the ENTIRE cache!"
        echo "Usage: $0 --all --confirm"
        exit 1
    fi

    echo -e "${RED}⚠️  WARNING: Clearing ENTIRE cache!${NC}"
    echo -e "${YELLOW}This will delete ALL cached data.${NC}"
    read -p "Are you absolutely sure? (type 'yes' to confirm): " CONFIRMATION

    if [ "$CONFIRMATION" != "yes" ]; then
        echo -e "${GREEN}✅ Aborted - cache not cleared${NC}"
        exit 0
    fi

    echo -e "${YELLOW}🔥 Clearing entire cache...${NC}"

    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from py.utils.cache_invalidator import CacheInvalidator

invalidator = CacheInvalidator()
result = invalidator.invalidate_all(confirm=True)

print(f'Patterns: {len(result.patterns_used)}')
print(f'Matched: {result.keys_matched} keys')
print(f'Deleted: {result.keys_deleted} keys')
print(f'Time: {result.execution_time_ms:.2f}ms')
if result.errors:
    print(f'Errors: {result.errors}')
"
    echo -e "${GREEN}✅ Cache cleared${NC}"
    exit 0
fi

# If no flags provided, show help
echo -e "${RED}Error: No operation specified${NC}"
echo "Run '$0 --help' for usage information"
exit 1
