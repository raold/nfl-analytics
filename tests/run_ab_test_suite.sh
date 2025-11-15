#!/bin/bash
#
# Comprehensive A/B Test Suite Runner
# =====================================
#
# Runs all A/B test framework tests with coverage reporting.
#
# Usage:
#   bash tests/run_ab_test_suite.sh [OPTIONS]
#
# Options:
#   --unit          Run only unit tests
#   --integration   Run only integration tests
#   --e2e           Run only end-to-end tests
#   --coverage      Generate coverage report
#   --verbose       Verbose output
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 14, 2025
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}A/B Test Framework - Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Parse arguments
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_E2E=false
RUN_ALL=true
COVERAGE=false
VERBOSE=false

for arg in "$@"; do
    case $arg in
        --unit)
            RUN_UNIT=true
            RUN_ALL=false
            ;;
        --integration)
            RUN_INTEGRATION=true
            RUN_ALL=false
            ;;
        --e2e)
            RUN_E2E=true
            RUN_ALL=false
            ;;
        --coverage)
            COVERAGE=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

# If no specific test type selected, run all
if [ "$RUN_ALL" = true ]; then
    RUN_UNIT=true
    RUN_INTEGRATION=true
    RUN_E2E=true
fi

# Build pytest command
PYTEST_CMD="pytest"
PYTEST_ARGS=""

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -vv"
else
    PYTEST_ARGS="$PYTEST_ARGS -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=py.production.ab_test_router --cov=py.production.real_time_api --cov-report=term-missing --cov-report=html"
fi

# Test results summary
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    echo -e "${YELLOW}Running Unit Tests...${NC}"
    echo "----------------------------------------"

    if $PYTEST_CMD tests/production/test_ab_test_router_comprehensive.py $PYTEST_ARGS; then
        echo -e "${GREEN}✓ Unit tests passed${NC}"
        UNIT_RESULT=0
    else
        echo -e "${RED}✗ Unit tests failed${NC}"
        UNIT_RESULT=1
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
fi

# Run integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${YELLOW}Running Integration Tests...${NC}"
    echo "----------------------------------------"

    if $PYTEST_CMD tests/integration/test_ab_test_api_integration.py $PYTEST_ARGS; then
        echo -e "${GREEN}✓ Integration tests passed${NC}"
        INTEGRATION_RESULT=0
    else
        echo -e "${RED}✗ Integration tests failed${NC}"
        INTEGRATION_RESULT=1
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
fi

# Run end-to-end tests
if [ "$RUN_E2E" = true ]; then
    echo -e "${YELLOW}Running End-to-End Tests...${NC}"
    echo "----------------------------------------"

    if $PYTEST_CMD tests/e2e/test_ab_test_e2e.py $PYTEST_ARGS; then
        echo -e "${GREEN}✓ End-to-end tests passed${NC}"
        E2E_RESULT=0
    else
        echo -e "${RED}✗ End-to-end tests failed${NC}"
        E2E_RESULT=1
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
fi

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Suite Summary${NC}"
echo -e "${GREEN}========================================${NC}"

if [ "$RUN_UNIT" = true ]; then
    if [ $UNIT_RESULT -eq 0 ]; then
        echo -e "Unit Tests:        ${GREEN}PASSED${NC}"
    else
        echo -e "Unit Tests:        ${RED}FAILED${NC}"
    fi
fi

if [ "$RUN_INTEGRATION" = true ]; then
    if [ $INTEGRATION_RESULT -eq 0 ]; then
        echo -e "Integration Tests: ${GREEN}PASSED${NC}"
    else
        echo -e "Integration Tests: ${RED}FAILED${NC}"
    fi
fi

if [ "$RUN_E2E" = true ]; then
    if [ $E2E_RESULT -eq 0 ]; then
        echo -e "E2E Tests:         ${GREEN}PASSED${NC}"
    else
        echo -e "E2E Tests:         ${RED}FAILED${NC}"
    fi
fi

echo ""

if [ "$COVERAGE" = true ]; then
    echo -e "${YELLOW}Coverage report generated:${NC}"
    echo "  HTML: htmlcov/index.html"
    echo ""
fi

# Exit with failure if any tests failed
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}✗ $FAILED_TESTS test suite(s) failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All test suites passed${NC}"
    exit 0
fi
