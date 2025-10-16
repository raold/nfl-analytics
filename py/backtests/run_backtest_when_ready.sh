#!/bin/bash
# Wait for Bayesian ratings to be available, then run backtest

echo "Waiting for Bayesian player ratings..."

while true; do
    count=$(PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -tAc "SELECT COUNT(*) FROM mart.bayesian_player_ratings WHERE stat_type = 'passing_yards'")

    if [ "$count" -gt 0 ]; then
        echo "✓ Found $count passing yards ratings in database"
        echo "Starting multi-year backtest..."

        cd /Users/dro/rice/nfl-analytics
        uv run python py/backtests/bayesian_props_multiyear_backtest.py 2>&1 | tee reports/multiyear_backtest_with_bayesian.log

        echo ""
        echo "✅ Backtest complete! Results saved to:"
        echo "   - reports/multiyear_backtest_with_bayesian.log"
        echo "   - reports/bayesian_backtest_multiyear/"

        break
    fi

    echo "  Waiting... (checking every 30 seconds)"
    sleep 30
done
