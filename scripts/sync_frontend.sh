#!/bin/bash
#
# Frontend Sync CLI - Wrapper for FrontendEngineeringAgent
#
# Provides convenient commands for schema validation, API contract enforcement,
# and frontend-backend synchronization.
#
# Usage:
#   ./scripts/sync_frontend.sh validate-schema    # Validate dashboard SQL queries
#   ./scripts/sync_frontend.sh validate-api       # Check API contract compliance
#   ./scripts/sync_frontend.sh check-drift        # Detect schema drift
#   ./scripts/sync_frontend.sh generate-types     # Generate Pydantic validators
#   ./scripts/sync_frontend.sh apply-updates      # Suggest migration updates
#   ./scripts/sync_frontend.sh save-snapshot      # Save current schema snapshot
#   ./scripts/sync_frontend.sh startup-check      # Run all startup validations
#
# Author: Claude Code (Sonnet 4.5)
# Created: November 22, 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python environment
PYTHON="${PYTHON:-python3}"

# Agent module path
AGENT_MODULE="nfl_analytics.utils.frontend_engineering_agent"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

check_python() {
    if ! command -v "${PYTHON}" &> /dev/null; then
        print_error "Python not found. Set PYTHON environment variable."
        exit 1
    fi
}

check_dependencies() {
    # Check if required Python packages are installed
    if ! "${PYTHON}" -c "import psycopg" 2>/dev/null; then
        print_warning "psycopg not installed - database validation will fail"
        echo "Install: pip install psycopg[binary]"
    fi

    if ! "${PYTHON}" -c "import requests" 2>/dev/null; then
        print_warning "requests not installed - API validation will fail"
        echo "Install: pip install requests"
    fi
}

# ============================================================================
# Command Functions
# ============================================================================

cmd_validate_schema() {
    print_header "VALIDATING DASHBOARD SCHEMA"

    cd "${PROJECT_ROOT}"

    if "${PYTHON}" -m ${AGENT_MODULE} validate-schema; then
        print_success "Schema validation passed"
        return 0
    else
        print_error "Schema validation failed"
        return 1
    fi
}

cmd_validate_api() {
    print_header "VALIDATING API CONTRACT"

    cd "${PROJECT_ROOT}"

    if "${PYTHON}" -m ${AGENT_MODULE} validate-api; then
        print_success "API contract validation passed"
        return 0
    else
        print_error "API contract validation failed"
        return 1
    fi
}

cmd_check_drift() {
    print_header "DETECTING SCHEMA DRIFT"

    cd "${PROJECT_ROOT}"

    if "${PYTHON}" -m ${AGENT_MODULE} check-drift; then
        print_success "No breaking schema drift detected"
        return 0
    else
        print_warning "Schema drift detected - review required"
        return 1
    fi
}

cmd_generate_types() {
    print_header "GENERATING TYPE VALIDATORS"

    cd "${PROJECT_ROOT}"

    if "${PYTHON}" -m ${AGENT_MODULE} generate-types; then
        print_success "Type validators generated"
        return 0
    else
        print_error "Type generation failed"
        return 1
    fi
}

cmd_apply_updates() {
    print_header "SUGGESTING FRONTEND UPDATES"

    cd "${PROJECT_ROOT}"

    if "${PYTHON}" -m ${AGENT_MODULE} apply-updates; then
        print_success "Update suggestions generated"
        return 0
    else
        print_warning "Manual updates required"
        return 1
    fi
}

cmd_save_snapshot() {
    print_header "SAVING SCHEMA SNAPSHOT"

    cd "${PROJECT_ROOT}"

    "${PYTHON}" <<EOF
from nfl_analytics.utils.frontend_engineering_agent import FrontendEngineeringAgent

agent = FrontendEngineeringAgent()
snapshot_path = agent.save_schema_snapshot()
print(f"✅ Schema snapshot saved: {snapshot_path}")
EOF

    if [ $? -eq 0 ]; then
        print_success "Schema snapshot saved"
        return 0
    else
        print_error "Failed to save schema snapshot"
        return 1
    fi
}

cmd_startup_check() {
    print_header "RUNNING STARTUP VALIDATION CHECKS"

    local exit_code=0

    # Check 1: Schema validation
    echo "1/3: Validating dashboard schema..."
    if cmd_validate_schema; then
        print_success "Schema validation: PASSED"
    else
        print_error "Schema validation: FAILED"
        exit_code=1
    fi

    echo ""

    # Check 2: API contract
    echo "2/3: Validating API contract..."
    if cmd_validate_api; then
        print_success "API contract: PASSED"
    else
        print_error "API contract: FAILED"
        exit_code=1
    fi

    echo ""

    # Check 3: Schema drift
    echo "3/3: Checking schema drift..."
    if cmd_check_drift; then
        print_success "Schema drift: NONE"
    else
        print_warning "Schema drift: DETECTED (non-blocking)"
        # Don't fail startup on drift - just warn
    fi

    echo ""
    print_header "STARTUP VALIDATION SUMMARY"

    if [ ${exit_code} -eq 0 ]; then
        print_success "All critical checks passed - safe to start dashboard"
        return 0
    else
        print_error "Critical checks failed - DO NOT start dashboard"
        return 1
    fi
}

cmd_help() {
    cat <<EOF
Frontend Sync CLI - Schema Validation and Frontend-Backend Synchronization

USAGE:
    ./scripts/sync_frontend.sh <command> [options]

COMMANDS:
    validate-schema     Validate all dashboard SQL queries against database schema
    validate-api        Check API contract compliance (FastAPI ↔ Streamlit)
    check-drift         Detect schema drift since last snapshot
    generate-types      Generate Pydantic validators from database schema
    apply-updates       Suggest automated migration updates
    save-snapshot       Save current schema snapshot for drift detection
    startup-check       Run all startup validation checks (blocking)
    help                Show this help message

EXAMPLES:
    # Pre-commit validation
    ./scripts/sync_frontend.sh validate-schema

    # Check if API is compatible
    ./scripts/sync_frontend.sh validate-api

    # Detect breaking schema changes
    ./scripts/sync_frontend.sh check-drift

    # Generate type-safe validators
    ./scripts/sync_frontend.sh generate-types

    # Save baseline for drift detection
    ./scripts/sync_frontend.sh save-snapshot

    # Run all checks before dashboard startup
    ./scripts/sync_frontend.sh startup-check

INTEGRATION:
    # Pre-commit hook (add to .git/hooks/pre-commit)
    ./scripts/sync_frontend.sh validate-schema || exit 1

    # Dashboard startup (add to dashboard/app.py)
    ./scripts/sync_frontend.sh startup-check || exit 1

    # CI/CD pipeline (add to .github/workflows/test.yml)
    - name: Validate Frontend
      run: ./scripts/sync_frontend.sh validate-schema

ENVIRONMENT:
    DB_HOST         Database host (default: localhost)
    DB_PORT         Database port (default: 5544)
    DB_NAME         Database name (default: devdb01)
    DB_USER         Database user (default: dro)
    DB_PASSWORD     Database password (required)
    PYTHON          Python interpreter (default: python3)

DOCUMENTATION:
    See docs/FRONTEND_ENGINEERING_AGENT.md for comprehensive guide

AUTHOR:
    Claude Code (Sonnet 4.5)
    Created: November 22, 2025
EOF
}

# ============================================================================
# Main CLI Router
# ============================================================================

main() {
    check_python
    check_dependencies

    if [ $# -eq 0 ]; then
        cmd_help
        exit 0
    fi

    local command=$1
    shift

    case "${command}" in
        validate-schema)
            cmd_validate_schema "$@"
            ;;
        validate-api)
            cmd_validate_api "$@"
            ;;
        check-drift)
            cmd_check_drift "$@"
            ;;
        generate-types)
            cmd_generate_types "$@"
            ;;
        apply-updates)
            cmd_apply_updates "$@"
            ;;
        save-snapshot)
            cmd_save_snapshot "$@"
            ;;
        startup-check)
            cmd_startup_check "$@"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            print_error "Unknown command: ${command}"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
