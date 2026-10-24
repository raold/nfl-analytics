#!/bin/bash
#
# Performance Optimizer - Analyze bottlenecks in models, SQL, and backtests
#
# Usage:
#   ./scripts/optimize_performance.sh                    # Full analysis
#   ./scripts/optimize_performance.sh --mcmc             # Check MCMC config
#   ./scripts/optimize_performance.sh --sql query.sql    # Analyze SQL query
#   ./scripts/optimize_performance.sh --backtest         # Estimate backtest time
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ "$1" == "--mcmc" ]; then
    echo "=== MCMC Configuration Check ==="
    python3 -c "
from nfl_analytics.utils.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer()

# Check default MCMC configuration
issues = optimizer.check_mcmc_config(chains=4, tune=1000, draws=2000)

for issue in issues:
    optimizer.report.add_issue(issue)

print(optimizer.generate_report())
"

elif [ "$1" == "--sql" ]; then
    if [ -z "$2" ]; then
        echo "Error: Please provide SQL file path"
        echo "Usage: ./scripts/optimize_performance.sh --sql query.sql"
        exit 1
    fi

    echo "=== SQL Query Analysis ==="
    python3 -c "
from nfl_analytics.utils.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer()
sql_code = open('$2').read()

issues = optimizer.check_sql_query(sql_code)

for issue in issues:
    optimizer.report.add_issue(issue)

print(optimizer.generate_report())
"

elif [ "$1" == "--backtest" ]; then
    echo "=== Backtest Time Estimation ==="
    python3 -c "
from nfl_analytics.utils.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer()

# Estimate for 2022-2024 backtest (148 games, 5 models)
estimate = optimizer.estimate_backtest_time(
    num_games=148,
    models_per_game=5,
    avg_prediction_time=1.0
)

print(f'Estimated backtest time: {estimate[\"estimated_hours\"]:.2f} hours')
print(f'Total predictions: {estimate[\"total_predictions\"]}')
print(f'Average time per prediction: {estimate[\"avg_prediction_time\"]}s')

if estimate['estimated_hours'] > 1.0:
    print('\\n⚠️  WARNING: Backtest exceeds 1 hour target')
    print('Consider:')
    print('  - Caching predictions')
    print('  - Parallel model execution')
    print('  - Reducing model complexity')
else:
    print('\\n✅ Backtest time within 1 hour target')
"

else
    echo "=== Performance Optimization Analysis ==="
    echo ""
    echo "Analyzing project for performance bottlenecks..."
    echo ""

    python3 -c "
from nfl_analytics.utils.performance_optimizer import PerformanceOptimizer
from pathlib import Path

optimizer = PerformanceOptimizer()

# Check Python files for performance issues
print('Scanning Python files for bottlenecks...')
py_files = list(Path('nfl_analytics/').rglob('*.py'))

for py_file in py_files[:5]:  # Sample first 5 files
    code = py_file.read_text()
    issues = optimizer.check_pandas_performance(code)

    if issues:
        print(f'\\n{py_file}:')
        for issue in issues:
            print(f'  - {issue.description}')

# Check MCMC configuration
print('\\n\\nChecking MCMC configuration...')
mcmc_issues = optimizer.check_mcmc_config(chains=4, tune=1000, draws=2000)

if mcmc_issues:
    for issue in mcmc_issues:
        print(f'  - {issue.description}')
else:
    print('  ✅ MCMC configuration looks good')

# Estimate backtest time
print('\\n\\nEstimating backtest time...')
estimate = optimizer.estimate_backtest_time(148, 5, 1.0)
print(f'  Estimated: {estimate[\"estimated_hours\"]:.2f} hours ({estimate[\"total_predictions\"]} predictions)')

if estimate['estimated_hours'] <= 1.0:
    print('  ✅ Within 1 hour target')
else:
    print('  ⚠️  Exceeds 1 hour target')

print('\\n\\nFor detailed analysis:')
print('  ./scripts/optimize_performance.sh --mcmc      # MCMC configuration')
print('  ./scripts/optimize_performance.sh --sql FILE  # SQL query analysis')
print('  ./scripts/optimize_performance.sh --backtest  # Backtest estimation')
"
fi

echo ""
echo "✅ Performance analysis complete"
