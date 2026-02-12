#!/usr/bin/env bash
# Emergency Native PostgreSQL Startup Script
# Starts PostgreSQL using existing pgdata without Docker
# ESTIMATED TIME: 5-10 minutes (after Homebrew install)
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "EMERGENCY NATIVE POSTGRESQL STARTUP"
echo "=========================================="
echo ""

# Configuration
PGDATA="$(cd "$(dirname "")/.." \&\& pwd)/pgdata"
PGLOG="$(cd "$(dirname "")/.." \&\& pwd)/logs/postgres.log"
PGPORT="5544"
PG_VERSION="16"

# Ensure logs directory exists
mkdir -p "$(dirname $PGLOG)"

# Step 1: Check if PostgreSQL@16 is installed
echo "[1/6] Checking PostgreSQL@16 installation..."
if ! brew list postgresql@16 &>/dev/null; then
  echo -e "${YELLOW}⚠️  PostgreSQL@16 not installed${NC}"
  echo ""
  echo "Installing PostgreSQL@16 via Homebrew (this will take 2-3 minutes)..."
  echo ""

  if ! command -v brew &>/dev/null; then
    echo -e "${RED}❌ ERROR: Homebrew not installed${NC}"
    echo "Install Homebrew first: https://brew.sh"
    exit 1
  fi

  brew install postgresql@16

  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PostgreSQL@16 installed successfully${NC}"
  else
    echo -e "${RED}❌ ERROR: PostgreSQL@16 installation failed${NC}"
    exit 1
  fi
else
  echo -e "${GREEN}✅ PostgreSQL@16 already installed${NC}"
fi
echo ""

# Step 2: Check pgdata directory
echo "[2/6] Verifying pgdata directory..."
if [ ! -d "$PGDATA" ]; then
  echo -e "${RED}❌ ERROR: pgdata directory not found: $PGDATA${NC}"
  echo "Cannot proceed without existing database data"
  exit 1
fi

if [ ! -f "$PGDATA/PG_VERSION" ]; then
  echo -e "${RED}❌ ERROR: pgdata appears corrupted (PG_VERSION missing)${NC}"
  exit 1
fi

PGDATA_VERSION=$(cat "$PGDATA/PG_VERSION")
echo -e "${GREEN}✅ pgdata exists (PostgreSQL version $PGDATA_VERSION)${NC}"

if [ "$PGDATA_VERSION" != "$PG_VERSION" ]; then
  echo -e "${YELLOW}⚠️  WARNING: pgdata version ($PGDATA_VERSION) != installed version ($PG_VERSION)${NC}"
  echo "This may cause compatibility issues. Proceeding anyway..."
fi

PGDATA_SIZE=$(du -sh "$PGDATA" | awk '{print $1}')
echo "   Size: $PGDATA_SIZE"
echo ""

# Step 3: Stop Homebrew PostgreSQL service (if running)
echo "[3/6] Checking for conflicting PostgreSQL services..."
if brew services list | grep postgresql@16 | grep started &>/dev/null; then
  echo -e "${YELLOW}⚠️  Stopping Homebrew PostgreSQL@16 service${NC}"
  brew services stop postgresql@16
  sleep 2
fi

# Check if another PostgreSQL is running on port 5544
if lsof -i :$PGPORT &>/dev/null; then
  echo -e "${YELLOW}⚠️  Port $PGPORT is already in use:${NC}"
  lsof -i :$PGPORT
  echo ""
  read -p "Attempt to stop existing process? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    PID=$(lsof -ti :$PGPORT)
    echo "Stopping PID $PID..."
    kill -TERM $PID || true
    sleep 3
  else
    echo "Cannot proceed with port in use. Exiting."
    exit 1
  fi
fi

echo -e "${GREEN}✅ Port $PGPORT is available${NC}"
echo ""

# Step 4: Start PostgreSQL
echo "[4/6] Starting PostgreSQL on port $PGPORT..."
PG_CTL="/opt/homebrew/opt/postgresql@16/bin/pg_ctl"

if [ ! -x "$PG_CTL" ]; then
  echo -e "${RED}❌ ERROR: pg_ctl not found at $PG_CTL${NC}"
  echo "PostgreSQL@16 may not be properly installed"
  exit 1
fi

# Check if already running
if "$PG_CTL" -D "$PGDATA" status &>/dev/null; then
  echo -e "${YELLOW}⚠️  PostgreSQL already running with this pgdata${NC}"
  echo "Current status:"
  "$PG_CTL" -D "$PGDATA" status
  echo ""
else
  # Start PostgreSQL
  echo "Command: $PG_CTL -D $PGDATA -l $PGLOG -o '-p $PGPORT' start"
  echo ""

  "$PG_CTL" \
    -D "$PGDATA" \
    -l "$PGLOG" \
    -o "-p $PGPORT" \
    start

  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PostgreSQL start command issued${NC}"
  else
    echo -e "${RED}❌ ERROR: PostgreSQL failed to start${NC}"
    echo "Check logs: $PGLOG"
    tail -20 "$PGLOG"
    exit 1
  fi
fi
echo ""

# Step 5: Wait for PostgreSQL to be ready
echo "[5/6] Waiting for PostgreSQL to be ready..."
ATTEMPTS=0
MAX_ATTEMPTS=30

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
  if pg_isready -h localhost -p "$PGPORT" -U dro &>/dev/null; then
    echo -e "${GREEN}✅ PostgreSQL is ready!${NC}"
    break
  fi

  ATTEMPTS=$((ATTEMPTS + 1))
  echo -n "."
  sleep 1
done
echo ""

if [ $ATTEMPTS -eq $MAX_ATTEMPTS ]; then
  echo -e "${RED}❌ ERROR: PostgreSQL did not become ready in time${NC}"
  echo "Check logs: $PGLOG"
  echo ""
  echo "Last 30 lines of log:"
  tail -30 "$PGLOG"
  exit 1
fi
echo ""

# Step 6: Verify connection and data
echo "[6/6] Verifying database connection..."
DB_URL="postgresql://dro:sicillionbillions@localhost:$PGPORT/devdb01"

if psql "$DB_URL" -c "SELECT version();" &>/dev/null; then
  echo -e "${GREEN}✅ Database connection successful!${NC}"
  echo ""

  # Get PostgreSQL version
  echo "PostgreSQL version:"
  psql "$DB_URL" -t -c "SELECT version();" | sed 's/^[ \t]*/  /'
  echo ""

  # Check data integrity
  echo "Data verification:"
  psql "$DB_URL" -t -c "
    SELECT 'Games: ' || COUNT(*)::text FROM public.games
    UNION ALL
    SELECT 'Plays: ' || COUNT(*)::text FROM public.plays
    UNION ALL
    SELECT 'Player Stats: ' || COUNT(*)::text FROM public.player_stats;
  " | sed 's/^[ \t]*/  /'
  echo ""

  # Check TimescaleDB extension
  echo "TimescaleDB extension status:"
  if psql "$DB_URL" -t -c "\dx" | grep -q timescaledb; then
    TIMESCALE_VERSION=$(psql "$DB_URL" -t -c "SELECT extversion FROM pg_extension WHERE extname='timescaledb';" | tr -d ' ')
    echo -e "  ${GREEN}✅ TimescaleDB $TIMESCALE_VERSION installed${NC}"
  else
    echo -e "  ${YELLOW}⚠️  TimescaleDB extension not found${NC}"
    echo "  This may affect time-series features"
  fi
  echo ""

else
  echo -e "${RED}❌ ERROR: Database connection failed${NC}"
  echo "Connection string: $DB_URL"
  echo ""
  psql "$DB_URL" -c "SELECT 1;" 2>&1 | head -10
  exit 1
fi

# Success!
echo "=========================================="
echo -e "${GREEN}✅ SUCCESS - DATABASE OPERATIONAL${NC}"
echo "=========================================="
echo ""
echo "PostgreSQL is now running natively on port $PGPORT"
echo ""
echo "Connection string:"
echo "  $DB_URL"
echo ""
echo "Management commands:"
echo "  Status:  $PG_CTL -D $PGDATA status"
echo "  Stop:    $PG_CTL -D $PGDATA stop"
echo "  Restart: $PG_CTL -D $PGDATA restart"
echo "  Logs:    tail -f $PGLOG"
echo ""
echo "Next steps:"
echo "  1. Test Python connection:"
echo "     python -c \"import psycopg2; conn=psycopg2.connect('$DB_URL'); print('OK')\""
echo ""
echo "  2. Test R connection:"
echo "     Rscript -e \"library(RPostgres); con=dbConnect(Postgres(), host='localhost', port=$PGPORT, dbname='devdb01', user='dro', password='sicillionbillions'); print(dbGetQuery(con, 'SELECT 1'))\""
echo ""
echo "  3. Resume production work:"
echo "     - XGBoost BFS exhaustive search"
echo "     - A/B test API deployment"
echo "     - 2025 season predictions"
echo ""
echo "NOTE: This PostgreSQL instance will NOT auto-start on boot."
echo "To start on boot, use: brew services start postgresql@16"
echo "(But this may conflict with Docker. Keep manual control.)"
echo ""
