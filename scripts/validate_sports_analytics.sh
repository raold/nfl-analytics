#!/bin/bash

# Sports Analytics Validator CLI Wrapper
# Validates NFL data against domain-specific rules

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function show_usage() {
    cat << EOF
Sports Analytics Validator - Validate NFL data against domain rules

Usage:
    $0 --validate-df <file.csv>     Validate DataFrame
    $0 --check-query <query.sql>    Check SQL query logic
    $0 --report                     Generate validation report
    $0 --help                       Show this help

Examples:
    # Validate feature DataFrame
    $0 --validate-df data/features/player_props.csv

    # Check SQL query for violations
    $0 --check-query queries/player_stats.sql

    # Validate and save report
    $0 --validate-df data/features.csv > validation_report.txt

Validation Categories:
    - Position eligibility (QB/RB/WR/TE stat validity)
    - Stat ranges (NFL physical limits)
    - Schedule constraints (weeks, games per team)
    - Team abbreviations (LAR/LAC enforcement)
    - Game results (score consistency)

Exit Codes:
    0 - Validation passed (no errors)
    1 - Validation failed (errors found)
EOF
}

# Check if Python is available (prefer python3, fallback to python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

case "$1" in
    --validate-df)
        if [ -z "$2" ]; then
            echo -e "${RED}ERROR: No file specified${NC}"
            show_usage
            exit 1
        fi
        echo -e "${YELLOW}Validating DataFrame: $2${NC}"
        $PYTHON_CMD "$PROJECT_ROOT/nfl_analytics/utils/sports_analytics_validator.py" --validate-df "$2"
        ;;

    --check-query)
        if [ -z "$2" ]; then
            echo -e "${RED}ERROR: No query file specified${NC}"
            show_usage
            exit 1
        fi
        echo -e "${YELLOW}Checking SQL query: $2${NC}"
        $PYTHON_CMD "$PROJECT_ROOT/nfl_analytics/utils/sports_analytics_validator.py" --check-query "$2"
        ;;

    --report)
        $PYTHON_CMD "$PROJECT_ROOT/nfl_analytics/utils/sports_analytics_validator.py" --report
        ;;

    --help|-h)
        show_usage
        exit 0
        ;;

    *)
        echo -e "${RED}ERROR: Unknown option: $1${NC}"
        show_usage
        exit 1
        ;;
esac
