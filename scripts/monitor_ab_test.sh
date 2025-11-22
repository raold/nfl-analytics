#!/bin/bash
#
# A/B Test Monitor - CLI wrapper for real-time experiment monitoring
#
# Usage:
#   ./monitor_ab_test.sh status    - Print full A/B test report
#   ./monitor_ab_test.sh metrics   - Print current metrics (MAE, ROI, win rate)
#   ./monitor_ab_test.sh check     - Check guardrail violations
#   ./monitor_ab_test.sh stop      - Auto-stop if guardrails violated
#   ./monitor_ab_test.sh promote   - Promote treatment to production
#
# Author: Claude Code (Sonnet 4.5)
# Created: November 22, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Python environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Change to project root
cd "$PROJECT_ROOT"

# ============================================================================
# Helper Functions
# ============================================================================

print_usage() {
    cat <<EOF
${BLUE}A/B Test Monitor${NC} - Real-time experiment tracking and guardrails

${GREEN}Usage:${NC}
  $0 <command>

${GREEN}Commands:${NC}
  ${YELLOW}status${NC}    - Print full A/B test report with recommendations
  ${YELLOW}metrics${NC}   - Print current metrics (MAE, ROI, win rate, sample size)
  ${YELLOW}check${NC}     - Check guardrail violations (returns exit code 1 if violated)
  ${YELLOW}stop${NC}      - Auto-stop test if guardrails violated
  ${YELLOW}promote${NC}   - Promote treatment to production (if recommended)

${GREEN}Examples:${NC}
  # Quick status check
  $0 status

  # Pre-backtest guardrail check
  $0 check && python py/backtest/historical_validation_4way.py

  # Auto-stop on violations (cron job)
  */15 * * * * $0 stop

  # Promote if ready
  $0 promote

${GREEN}Monitoring:${NC}
  Prometheus metrics: http://localhost:9090
  Grafana dashboard: http://localhost:3000
  Database: ml.ab_test_results, ml.ab_test_status

${GREEN}Configuration:${NC}
  config/ab_test_config.yaml - Test configuration and guardrails
  .env - Database credentials

EOF
}

check_dependencies() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found${NC}"
        exit 1
    fi

    if [ ! -f "config/ab_test_config.yaml" ]; then
        echo -e "${RED}Error: A/B test config not found${NC}"
        echo "Expected: config/ab_test_config.yaml"
        exit 1
    fi
}

# ============================================================================
# Command Handlers
# ============================================================================

cmd_status() {
    echo -e "${BLUE}Fetching A/B test status...${NC}\n"
    python py/utils/ab_test_monitor.py status
}

cmd_metrics() {
    echo -e "${BLUE}Current A/B test metrics:${NC}\n"
    python py/utils/ab_test_monitor.py metrics
}

cmd_check() {
    echo -e "${BLUE}Checking guardrails...${NC}\n"
    python py/utils/ab_test_monitor.py check

    # Exit with error code if violations detected (for CI/CD)
    if python -c "
from py.utils.ab_test_monitor import ABTestMonitor
monitor = ABTestMonitor()
status = monitor.check_guardrails()
exit(1 if status.violated else 0)
    "; then
        echo -e "\n${GREEN}✅ All guardrails passed${NC}"
        exit 0
    else
        echo -e "\n${RED}⛔ Guardrail violations detected${NC}"
        exit 1
    fi
}

cmd_stop() {
    echo -e "${BLUE}Checking for auto-stop conditions...${NC}\n"
    python py/utils/ab_test_monitor.py stop

    # Check if test was stopped
    if python -c "
from py.utils.ab_test_monitor import ABTestMonitor
monitor = ABTestMonitor()
exit(0 if monitor.auto_stop_if_violated() else 1)
    "; then
        echo -e "${RED}⛔ A/B TEST STOPPED DUE TO GUARDRAIL VIOLATIONS${NC}"
        echo -e "${YELLOW}Action required:${NC}"
        echo "  1. Review violations: $0 status"
        echo "  2. Rollback treatment: ./scripts/rollback_ensemble.sh"
        echo "  3. Investigate root cause"
        exit 0
    else
        echo -e "${GREEN}✅ No violations. Test continues.${NC}"
        exit 0
    fi
}

cmd_promote() {
    echo -e "${BLUE}Checking promotion eligibility...${NC}\n"
    python py/utils/ab_test_monitor.py promote

    # Get recommendation
    recommendation=$(python -c "
from py.utils.ab_test_monitor import ABTestMonitor
monitor = ABTestMonitor()
print(monitor.recommend_decision().value)
    ")

    if [ "$recommendation" == "promote_treatment" ]; then
        echo -e "\n${GREEN}✅ Treatment promoted to production${NC}"
        echo -e "${YELLOW}Next steps:${NC}"
        echo "  1. Update enhanced_ensemble_v6.py configuration"
        echo "  2. Deploy to production: docker-compose up -d api"
        echo "  3. Monitor for 1 week: ./scripts/monitor_production.sh"
    else
        echo -e "\n${YELLOW}Current recommendation: $recommendation${NC}"
        echo "Run '$0 status' for detailed analysis"
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Check dependencies
    check_dependencies

    # Parse command
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi

    case "$1" in
        status)
            cmd_status
            ;;
        metrics)
            cmd_metrics
            ;;
        check)
            cmd_check
            ;;
        stop)
            cmd_stop
            ;;
        promote)
            cmd_promote
            ;;
        help|-h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown command '$1'${NC}\n"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
