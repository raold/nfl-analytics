#!/bin/bash
#
# Model Registry Auditor - Shell Wrapper
#
# Comprehensive automated auditing and registration of ML models.
#
# Usage:
#   ./scripts/model_registry_auditor.sh --scan
#   ./scripts/model_registry_auditor.sh --register --dry-run
#   ./scripts/model_registry_auditor.sh --audit
#   ./scripts/model_registry_auditor.sh --convergence
#   ./scripts/model_registry_auditor.sh --sync-check
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 14, 2025

set -euo pipefail

# Use python3 explicitly
PYTHON=python3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Python script
PYTHON_SCRIPT="nfl_analytics/utils/model_registry_auditor.py"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: $PYTHON_SCRIPT not found${NC}"
    exit 1
fi

# Check for Python API (click library not required for Python API)
# Note: Import check disabled due to R initialization issues
# Dependencies will be checked at runtime

# Help message
show_help() {
    echo "Model Registry Auditor - Automated model management"
    echo ""
    echo "Usage:"
    echo "  ./scripts/model_registry_auditor.sh --scan           # Scan for pending registrations"
    echo "  ./scripts/model_registry_auditor.sh --register       # Auto-register models (dry-run)"
    echo "  ./scripts/model_registry_auditor.sh --audit          # Full audit report"
    echo "  ./scripts/model_registry_auditor.sh --convergence    # Check Bayesian convergence"
    echo "  ./scripts/model_registry_auditor.sh --sync-check     # Database-registry sync"
    echo "  ./scripts/model_registry_auditor.sh --help           # Show this help"
    echo ""
    echo "Options:"
    echo "  --output FILE       Save report to file (for --scan, --audit)"
    echo "  --no-dry-run        Actually register models (for --register)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/model_registry_auditor.sh --scan --output pending.json"
    echo "  ./scripts/model_registry_auditor.sh --register --no-dry-run"
    echo "  ./scripts/model_registry_auditor.sh --audit --output audit_report.json"
}

# Parse arguments
COMMAND=""
OUTPUT=""
DRY_RUN="--dry-run"

while [[ $# -gt 0 ]]; do
    case $1 in
        --scan)
            COMMAND="scan"
            shift
            ;;
        --register)
            COMMAND="register"
            shift
            ;;
        --audit)
            COMMAND="audit"
            shift
            ;;
        --convergence)
            COMMAND="convergence"
            shift
            ;;
        --sync-check)
            COMMAND="sync_check"
            shift
            ;;
        --output|-o)
            OUTPUT="--output $2"
            shift 2
            ;;
        --no-dry-run)
            DRY_RUN="--no-dry-run"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if command specified
if [ -z "$COMMAND" ]; then
    echo -e "${YELLOW}No command specified. Use --help for usage.${NC}"
    show_help
    exit 1
fi

# Check if click is available for CLI
if ! $PYTHON -c "import click" 2>/dev/null; then
    echo -e "${YELLOW}Warning: click library not installed. Using Python API fallback.${NC}"
    echo ""

    # Use Python API directly
    if [ "$COMMAND" = "scan" ]; then
        $PYTHON << EOF
from nfl_analytics.utils.model_registry_auditor import ModelRegistryAuditor
import json

auditor = ModelRegistryAuditor()
filesystem_models = auditor.scan_filesystem_models()
database_models = auditor.scan_database_models()
candidates = auditor.identify_pending_registrations(filesystem_models, database_models)

print("=" * 80)
print("PENDING MODEL REGISTRATIONS")
print("=" * 80)
print(f"Total filesystem models: {len(filesystem_models)}")
print(f"Total database models: {len(database_models)}")
print(f"Pending registrations: {len(candidates)}")
print(f"  - Should register: {sum(1 for c in candidates if c.should_register)}")
print(f"  - Blocked: {sum(1 for c in candidates if not c.should_register)}")

valid = [c for c in candidates if c.should_register]
if valid:
    print()
    print("READY TO REGISTER (first 10):")
    for c in valid[:10]:
        print(f"  ✅ {c.model_name} ({c.model_type})")
EOF
    elif [ "$COMMAND" = "audit" ]; then
        $PYTHON << EOF
from nfl_analytics.utils.model_registry_auditor import ModelRegistryAuditor
import logging
logging.getLogger('nfl_analytics.utils.model_registry_auditor').setLevel(logging.ERROR)

auditor = ModelRegistryAuditor()
report = auditor.audit_full()
print(report.summary())
EOF
    elif [ "$COMMAND" = "convergence" ]; then
        $PYTHON << EOF
from nfl_analytics.utils.model_registry_auditor import ModelRegistryAuditor
import logging
logging.getLogger('nfl_analytics.utils.model_registry_auditor').setLevel(logging.ERROR)

auditor = ModelRegistryAuditor()
registry_models = auditor.scan_registry_yaml()
reports = auditor.validate_convergence(registry_models)

print("=" * 80)
print("BAYESIAN MODEL CONVERGENCE")
print("=" * 80)

for r in reports:
    status = "✅ PASS" if r.converged else "⛔ FAIL"
    rhat = f"{r.rhat_max:.4f}" if r.rhat_max else "N/A"
    ess = str(r.ess_min) if r.ess_min else "N/A"
    div = str(r.divergences) if r.divergences else "N/A"
    print(f"{r.model_name:<40} {status:<10} {rhat:<10} {ess:<10} {div:<15}")

failures = [r for r in reports if not r.converged]
if failures:
    print()
    print(f"{len(failures)} FAILURES:")
    for r in failures:
        print(f"  ⛔ {r.model_name}")
        for issue in r.issues:
            print(f"      - {issue}")
EOF
    else
        echo -e "${RED}Command $COMMAND requires click library for CLI${NC}"
        echo "Install with: pip install click"
        exit 1
    fi

    exit 0
fi

# Run with click CLI
if [ "$COMMAND" = "register" ]; then
    $PYTHON "$PYTHON_SCRIPT" "$COMMAND" $DRY_RUN
else
    $PYTHON "$PYTHON_SCRIPT" "$COMMAND" $OUTPUT
fi
