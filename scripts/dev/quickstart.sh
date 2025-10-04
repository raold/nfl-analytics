#!/usr/bin/env bash
set -euo pipefail

# One-command startup for development
echo "🚀 NFL Analytics Quick Start"

# Start database if not running
if ! docker ps | grep -q nfl-analytics-pg-1; then
    echo "Starting PostgreSQL..."
    docker compose -f infrastructure/docker/docker-compose.yaml up -d pg
    sleep 3
else
    echo "✅ PostgreSQL already running"
fi

# Activate Python environment (prefer uv if available)
if [ -d .venv-uv ]; then
    source .venv-uv/bin/activate
    echo "✅ Activated uv environment"
elif [ -d .venv ]; then
    source .venv/bin/activate
    echo "✅ Activated pip environment"
else
    echo "❌ No Python environment found. Run: uv venv .venv-uv"
    exit 1
fi

# Check database connection
echo "Testing database connection..."
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT version();" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Database connected"
else
    echo "❌ Database connection failed"
    exit 1
fi

# Show stats
echo ""
echo "📊 Database Statistics:"
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -t -c "
SELECT
    'Games: ' || COUNT(*) || ' (' || MIN(season) || '-' || MAX(season) || ')' as stats
FROM games
UNION ALL
SELECT 'Plays: ' || TO_CHAR(COUNT(*), '999,999,999') FROM plays
UNION ALL
SELECT 'Odds: ' || TO_CHAR(COUNT(*), '999,999,999') FROM odds_history
UNION ALL
SELECT 'DB Size: ' || pg_size_pretty(pg_database_size('devdb01'));"

echo ""
echo "Ready for development! 🏈"
echo ""
echo "Quick commands:"
echo "  make features    - Generate features"
echo "  make backtest    - Run backtests"
echo "  make weekly      - Weekly data update"