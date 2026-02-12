#!/bin/bash
#
# Setup NFL Analytics Cron Jobs
#
# This script installs automated weekly workflows for:
#   - Prediction generation (Tuesday 12pm ET)
#   - Odds tracking (3x daily: 10am, 4pm, 10pm ET)
#   - Outcome backfill (Wednesday 6am ET)
#
# Prerequisites:
#   1. The Odds API key (free tier: https://the-odds-api.com)
#   2. PostgreSQL database running (localhost:5544)
#   3. Python environment with uv and dependencies installed
#
# Usage:
#   bash scripts/setup_cron.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "NFL Analytics Cron Job Setup"
echo "========================================"
echo ""

# Get project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project directory: $PROJECT_DIR"

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# 1. Check for The Odds API key
if [ -z "$ODDS_API_KEY" ]; then
    echo -e "${YELLOW}⚠️ ODDS_API_KEY not set${NC}"
    echo "Please set your API key as an environment variable:"
    echo "  export ODDS_API_KEY='your_key_here'"
    echo ""
    echo "Get a free key at: https://the-odds-api.com/liveapi/guides/v4/#get-access"
    echo ""
    read -p "Enter your Odds API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        export ODDS_API_KEY="$api_key"
        echo "ODDS_API_KEY set for this session"
    else
        echo -e "${RED}❌ Skipping odds tracking cron jobs (API key required)${NC}"
    fi
else
    echo -e "${GREEN}✅ ODDS_API_KEY found${NC}"
fi

# 2. Check database connection
echo ""
echo "Testing database connection..."
if psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Database connection successful${NC}"
else
    echo -e "${RED}❌ Database connection failed${NC}"
    echo "Please ensure PostgreSQL is running on localhost:5544"
    exit 1
fi

# 3. Check Python environment
echo ""
echo "Checking Python environment..."
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✅ uv found${NC}"
else
    echo -e "${RED}❌ uv not found${NC}"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# 4. Test prediction script
echo ""
echo "Testing prediction script..."
if uv run python scripts/generate_weekly_predictions.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Prediction script OK${NC}"
else
    echo -e "${RED}❌ Prediction script failed${NC}"
    exit 1
fi

# 5. Test odds fetcher script
echo ""
echo "Testing odds fetcher script..."
if uv run python scripts/fetch_odds.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Odds fetcher script OK${NC}"
else
    echo -e "${RED}❌ Odds fetcher script failed${NC}"
    exit 1
fi

# 6. Test outcome backfill script
echo ""
echo "Testing outcome backfill script..."
if uv run python scripts/backfill_outcomes.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Outcome backfill script OK${NC}"
else
    echo -e "${RED}❌ Outcome backfill script failed${NC}"
    exit 1
fi

# Create temporary crontab file with proper paths
echo ""
echo "Generating crontab configuration..."

TEMP_CRONTAB=$(mktemp)

# Update paths in crontab template
sed "s|NFL_ANALYTICS_DIR=.*|NFL_ANALYTICS_DIR=$PROJECT_DIR|g" \
    "$PROJECT_DIR/config/nfl_analytics_crontab.txt" > "$TEMP_CRONTAB"

# Add ODDS_API_KEY if set
if [ -n "$ODDS_API_KEY" ]; then
    sed -i '' "s|ODDS_API_KEY=.*|ODDS_API_KEY=$ODDS_API_KEY|g" "$TEMP_CRONTAB"
fi

echo "Crontab configuration:"
echo "========================================"
cat "$TEMP_CRONTAB"
echo "========================================"
echo ""

# Ask for confirmation
read -p "Install these cron jobs? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Setup cancelled"
    rm "$TEMP_CRONTAB"
    exit 0
fi

# Backup existing crontab
echo ""
echo "Backing up existing crontab..."
crontab -l > "$PROJECT_DIR/config/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true

# Install new crontab
echo "Installing cron jobs..."
crontab "$TEMP_CRONTAB"

# Cleanup
rm "$TEMP_CRONTAB"

echo ""
echo -e "${GREEN}✅ Cron jobs installed successfully!${NC}"
echo ""
echo "Verify installation:"
echo "  crontab -l"
echo ""
echo "View logs:"
echo "  tail -f /tmp/weekly_predictions.log"
echo "  tail -f /tmp/odds_fetcher.log"
echo "  tail -f /tmp/backfill_outcomes.log"
echo ""
echo "Schedule:"
echo "  - Predictions: Tuesday 12pm ET"
echo "  - Odds: 3x daily (10am, 4pm, 10pm ET)"
echo "  - Outcomes: Wednesday 6am ET"
echo ""
echo "To remove cron jobs:"
echo "  crontab -r"
echo ""
