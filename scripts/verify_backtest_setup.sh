#!/bin/bash
# Verify Ensemble v6.0 Backtest Setup
# ====================================
#
# Pre-flight checks before launching 18-hour backtest.
# Validates all dependencies, models, and configurations.
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 15, 2025

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CHECKS_PASSED=0
CHECKS_FAILED=0

echo "================================================================================"
echo "Ensemble v6.0 Backtest - Setup Verification"
echo "================================================================================"
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ PASS${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}❌ FAIL${NC} $1"
    echo -e "   ${YELLOW}Fix: $2${NC}"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC} $1"
    echo -e "   ${YELLOW}Note: $2${NC}"
}

# Check 1: Python environment
echo "Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    check_pass "Python found: $PYTHON_VERSION"
else
    check_fail "Python not found" "Install Python 3.9+"
fi

# Check 2: Required Python packages
echo ""
echo "Checking required packages..."
REQUIRED_PACKAGES=("pandas" "numpy" "scipy" "psycopg2" "rpy2")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        check_pass "Package '$pkg' installed"
    else
        check_fail "Package '$pkg' missing" "pip install $pkg"
    fi
done

# Check 3: Database connection
echo ""
echo "Checking database connection..."
if docker ps | grep -q pg; then
    check_pass "PostgreSQL container running"

    # Test connection
    if psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1" &>/dev/null; then
        check_pass "Database connection successful"
    else
        check_fail "Database connection failed" "Check PostgreSQL credentials"
    fi
else
    check_fail "PostgreSQL container not running" "docker compose up -d pg"
fi

# Check 4: Model files existence
echo ""
echo "Checking model files..."

# XGBoost v4 Ridge Residuals
if [ -f "models/xgboost/xgb_v4_ridge_residuals.json" ]; then
    check_pass "XGBoost v4 Ridge model found"
else
    check_fail "XGBoost v4 Ridge model missing" "python scripts/xgb_exp7_backtest_2024.py"
fi

if [ -f "models/xgboost/ridge_opponent_baseline.pkl" ]; then
    check_pass "Ridge opponent baseline found"
else
    check_fail "Ridge opponent baseline missing" "python scripts/xgb_exp7_backtest_2024.py"
fi

# Hybrid Static-Incremental
if [ -f "models/online_learning/hybrid/static_model.pkl" ]; then
    check_pass "Hybrid static model found"
else
    check_fail "Hybrid static model missing" "python scripts/validate_hybrid_model.py"
fi

# R models
R_MODELS=("hierarchical_v2.rds" "state_space_v2.rds" "informative_priors_v2.rds")
for model in "${R_MODELS[@]}"; do
    if [ -f "models/bayesian/$model" ]; then
        check_pass "R model '$model' found"
    else
        check_warn "R model '$model' missing" "Ensemble will use fallback predictions"
    fi
done

# Temperature scaler
if [ -f "models/calibration/temperature_scaler_4way.pkl" ]; then
    check_pass "Temperature scaler found"
else
    check_warn "Temperature scaler missing" "Predictions will be uncalibrated"
fi

# Check 5: Backtest script
echo ""
echo "Checking backtest script..."
if [ -f "nfl_analytics/backtest/historical_validation_v6.py" ]; then
    check_pass "Backtest script found"
else
    check_fail "Backtest script missing" "Script should be at nfl_analytics/backtest/historical_validation_v6.py"
fi

# Check 6: Output directory
echo ""
echo "Checking output directory..."
if [ -d "results" ]; then
    check_pass "Output directory 'results/' exists"
else
    mkdir -p results
    check_pass "Created output directory 'results/'"
fi

# Check 7: Dry run test
echo ""
echo "Running dry run test (5 games)..."
echo "This may take 30-60 seconds..."

if timeout 120 python nfl_analytics/backtest/historical_validation_v6.py --dry-run &>/tmp/backtest_dryrun_test.log; then
    if grep -q "DRY RUN COMPLETE" /tmp/backtest_dryrun_test.log; then
        V6_PREDS=$(grep "v6.0 predictions:" /tmp/backtest_dryrun_test.log | awk '{print $4}')
        V5_PREDS=$(grep "v5.1 predictions:" /tmp/backtest_dryrun_test.log | awk '{print $4}')

        if [ "$V6_PREDS" -gt 0 ] && [ "$V5_PREDS" -gt 0 ]; then
            check_pass "Dry run successful (v6: $V6_PREDS, v5: $V5_PREDS predictions)"
        else
            check_fail "Dry run generated no predictions" "Check log: /tmp/backtest_dryrun_test.log"
        fi
    else
        check_fail "Dry run did not complete" "Check log: /tmp/backtest_dryrun_test.log"
    fi
else
    check_fail "Dry run timeout or error" "Check log: /tmp/backtest_dryrun_test.log"
fi

# Check 8: Disk space
echo ""
echo "Checking disk space..."
AVAILABLE_GB=$(df -H . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "${AVAILABLE_GB%.*}" -gt 10 ]; then
    check_pass "Sufficient disk space (${AVAILABLE_GB}GB available)"
else
    check_warn "Low disk space (${AVAILABLE_GB}GB available)" "Recommend > 10GB free"
fi

# Check 9: Memory
echo ""
echo "Checking available memory..."
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    AVAILABLE_MB=$(vm_stat | awk '/Pages free/ {print $3}' | sed 's/\.//' | awk '{print $1 * 4096 / 1048576}')
    if [ "${AVAILABLE_MB%.*}" -gt 4000 ]; then
        check_pass "Sufficient memory (${AVAILABLE_MB%.*}MB available)"
    else
        check_warn "Low memory (${AVAILABLE_MB%.*}MB available)" "Recommend > 4GB free"
    fi
fi

# Summary
echo ""
echo "================================================================================"
echo "VERIFICATION SUMMARY"
echo "================================================================================"
echo ""
echo -e "${GREEN}Checks Passed: $CHECKS_PASSED${NC}"
echo -e "${RED}Checks Failed: $CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED - Ready to launch backtest!${NC}"
    echo ""
    echo "Launch commands:"
    echo "  bash scripts/launch_ensemble_v6_backtest.sh"
    echo ""
    exit 0
else
    echo -e "${RED}❌ $CHECKS_FAILED CHECKS FAILED - Fix issues before launching${NC}"
    echo ""
    echo "Fix issues above, then re-run verification:"
    echo "  bash scripts/verify_backtest_setup.sh"
    echo ""
    exit 1
fi
