#!/bin/bash
#
# Betting Strategy Validator - Validate betting strategies before live deployment
#
# Usage:
#   ./scripts/validate_strategy.sh RETURNS.pkl               # Validate backtest results
#   ./scripts/validate_strategy.sh --stress RETURNS.pkl      # Run stress tests
#   ./scripts/validate_strategy.sh --kelly EDGE PROB         # Calculate Kelly bet size
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ "$1" == "--stress" ]; then
    if [ -z "$2" ]; then
        echo "Error: Please provide returns file"
        echo "Usage: ./scripts/validate_strategy.sh --stress RETURNS.pkl"
        exit 1
    fi

    RETURNS_FILE="$2"

    if [ ! -f "$RETURNS_FILE" ]; then
        echo "Error: Returns file not found: $RETURNS_FILE"
        exit 1
    fi

    echo "=== Betting Strategy Stress Testing ==="
    echo "Returns: $RETURNS_FILE"
    echo ""

    python3 -c "
import sys
import pickle
from nfl_analytics.utils.betting_validator import BettingValidator

validator = BettingValidator()

# Load returns
with open('$RETURNS_FILE', 'rb') as f:
    returns = pickle.load(f)

print('Running stress tests...')
print()

# Run stress tests
report = validator.validate_backtest_results(returns, initial_bankroll=10000)

# Focus on stress test results
if report.stress_test_results:
    print('Stress Test Results:')
    for scenario, result in report.stress_test_results.items():
        print(f'  {scenario}:')
        print(f'    Final bankroll: \${result[\"final_bankroll\"]:.2f}')
        print(f'    ROI: {result[\"roi\"]:.2%}')
        print(f'    Max drawdown: {result[\"max_drawdown\"]:.2%}')
        print()

    if all(r['roi'] > -0.10 for r in report.stress_test_results.values()):
        print('✅ Passed all stress tests (max loss <10%)')
    else:
        print('⚠️  Failed some stress tests (losses >10%)')
else:
    print('⚠️  No stress test results available')

print()
print('Full report:')
print(validator.generate_report(report))
"

elif [ "$1" == "--kelly" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Error: Please provide edge and probability"
        echo "Usage: ./scripts/validate_strategy.sh --kelly EDGE PROBABILITY"
        echo ""
        echo "Example: ./scripts/validate_strategy.sh --kelly 0.05 0.60"
        echo "  (5% edge, 60% win probability)"
        exit 1
    fi

    EDGE="$2"
    PROB="$3"

    echo "=== Kelly Criterion Calculator ==="
    echo ""

    python3 -c "
edge = float('$EDGE')
prob = float('$PROB')

# Full Kelly
full_kelly = edge / (1 - prob)

# Fractional Kelly (0.25 = quarter Kelly for safety)
frac_kelly = 0.25 * full_kelly

print(f'Edge: {edge:.2%}')
print(f'Win Probability: {prob:.2%}')
print()
print(f'Full Kelly: {full_kelly:.2%} of bankroll')
print(f'Fractional Kelly (0.25): {frac_kelly:.2%} of bankroll')
print()

# Example bet sizes
bankroll = 10000
print(f'Example with \${bankroll} bankroll:')
print(f'  Full Kelly bet: \${bankroll * full_kelly:.2f}')
print(f'  Fractional Kelly bet: \${bankroll * frac_kelly:.2f} (recommended)')
print()
print('⚠️  Use fractional Kelly (0.25) for safety')
print('   Full Kelly can lead to excessive drawdowns')
"

elif [ -n "$1" ]; then
    # Validate backtest results from returns file
    RETURNS_FILE="$1"

    if [ ! -f "$RETURNS_FILE" ]; then
        echo "Error: Returns file not found: $RETURNS_FILE"
        exit 1
    fi

    INITIAL_BANKROLL="${2:-10000}"

    echo "=== Betting Strategy Validation ==="
    echo "Returns: $RETURNS_FILE"
    echo "Initial Bankroll: \$$INITIAL_BANKROLL"
    echo ""

    python3 -c "
import sys
import pickle
import numpy as np
from nfl_analytics.utils.betting_validator import BettingValidator

validator = BettingValidator()

# Load returns
with open('$RETURNS_FILE', 'rb') as f:
    returns = pickle.load(f)

# Convert to numpy if needed
if not isinstance(returns, np.ndarray):
    returns = np.array(returns)

print(f'Loaded {len(returns)} bets')
print()

# Validate
report = validator.validate_backtest_results(returns, initial_bankroll=float('$INITIAL_BANKROLL'))

# Print report
print(validator.generate_report(report))
print()

# Decision
if report.approved_for_deployment:
    print('=' * 60)
    print('✅ BETTING STRATEGY APPROVED FOR PRODUCTION DEPLOYMENT')
    print('=' * 60)
    print()
    print('All validation criteria met:')
    print(f'  ✅ ROI: {report.roi:.2%} (target: >5%)')
    print(f'  ✅ Win Rate: {report.win_rate:.2%} (target: >59%)')
    print(f'  ✅ Sharpe Ratio: {report.sharpe_ratio:.2f} (target: >2.2)')
    print(f'  ✅ Max Drawdown: {report.max_drawdown:.2%} (limit: <-8%)')
    print()
    print('Next steps:')
    print('  1. Deploy to production API')
    print('  2. Start with paper trading')
    print('  3. Monitor performance closely')
    print('  4. Gradually increase bet sizes')
    sys.exit(0)
else:
    print('=' * 60)
    print('⛔ BETTING STRATEGY REJECTED - DO NOT DEPLOY')
    print('=' * 60)
    print()
    print('Failed criteria:')

    critical_issues = [issue for issue in report.issues if issue.severity.value == 'critical']

    for issue in critical_issues:
        print(f'  ❌ {issue.description}')

    print()
    print('Do NOT deploy this strategy to production')
    print()
    print('Recommendations:')
    print('  - Retrain models with more data')
    print('  - Adjust risk parameters')
    print('  - Improve calibration')
    print('  - Re-run backtest on different time period')
    sys.exit(1)
"

else
    echo "=== Betting Strategy Validator ==="
    echo ""
    echo "Validates betting strategies before live deployment using risk gates:"
    echo "  - ROI >5% (target: 7%+)"
    echo "  - Win Rate >59% (target: 62%+)"
    echo "  - Sharpe Ratio >2.2 (target: 2.5+)"
    echo "  - Max Drawdown <-8%"
    echo ""
    echo "Usage:"
    echo "  ./scripts/validate_strategy.sh RETURNS.pkl [BANKROLL]  # Validate backtest"
    echo "  ./scripts/validate_strategy.sh --stress RETURNS.pkl    # Stress tests"
    echo "  ./scripts/validate_strategy.sh --kelly EDGE PROB       # Kelly calculator"
    echo ""
    echo "Examples:"
    echo "  ./scripts/validate_strategy.sh backtest_returns.pkl 10000"
    echo "  ./scripts/validate_strategy.sh --stress backtest_returns.pkl"
    echo "  ./scripts/validate_strategy.sh --kelly 0.05 0.60"
    echo ""
    echo "⛔ Strategies that fail validation will be BLOCKED from deployment"
    exit 1
fi

echo ""
echo "✅ Validation complete"
