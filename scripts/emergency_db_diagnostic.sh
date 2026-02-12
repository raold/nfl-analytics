#!/usr/bin/env bash
# Emergency Database Diagnostic Script
# Quickly assess database availability and Docker state
set -euo pipefail

echo "=========================================="
echo "NFL ANALYTICS - EMERGENCY DIAGNOSTIC"
echo "=========================================="
echo ""
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

test_pass() {
  echo -e "${GREEN}✅ PASS${NC}: $1"
  TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_fail() {
  echo -e "${RED}❌ FAIL${NC}: $1"
  TESTS_FAILED=$((TESTS_FAILED + 1))
}

test_warn() {
  echo -e "${YELLOW}⚠️  WARN${NC}: $1"
}

# 1. Check Docker Desktop availability
echo "=========================================="
echo "1. DOCKER DESKTOP STATUS"
echo "=========================================="
if docker ps &>/dev/null; then
  test_pass "Docker daemon responding"
  echo "Running containers:"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
  test_fail "Docker daemon NOT responding"
  echo "Error: $(docker ps 2>&1 || echo 'Cannot connect to Docker daemon')"
fi
echo ""

# 2. Check Docker socket
echo "=========================================="
echo "2. DOCKER SOCKET"
echo "=========================================="
DOCKER_SOCKET="${DOCKER_HOST:-/var/run/docker.sock}"
DOCKER_SOCKET="${DOCKER_SOCKET#unix://}"  # Strip unix:// prefix if present
if [ -S "$DOCKER_SOCKET" ]; then
  test_pass "Docker socket exists: $DOCKER_SOCKET"
else
  test_fail "Docker socket missing: $DOCKER_SOCKET"
fi
echo ""

# 3. Check PostgreSQL port 5544
echo "=========================================="
echo "3. POSTGRESQL PORT 5544"
echo "=========================================="
if lsof -i :5544 &>/dev/null; then
  test_pass "Port 5544 is listening"
  echo "Process:"
  lsof -i :5544 | grep -v COMMAND || true
else
  test_fail "Port 5544 NOT listening"
  echo "No process bound to port 5544"
fi
echo ""

# 4. Check pgdata directory
echo "=========================================="
echo "4. PGDATA DIRECTORY"
echo "=========================================="
PGDATA_DIR="$(cd "$(dirname "")/.." \&\& pwd)/pgdata"
if [ -d "$PGDATA_DIR" ]; then
  test_pass "pgdata directory exists"
  echo "Size: $(du -sh $PGDATA_DIR 2>/dev/null || echo 'Unknown')"
  echo "Contents:"
  ls -lh "$PGDATA_DIR" | head -10

  # Check for critical PostgreSQL files
  if [ -f "$PGDATA_DIR/PG_VERSION" ]; then
    PG_VERSION=$(cat "$PGDATA_DIR/PG_VERSION")
    test_pass "PostgreSQL version: $PG_VERSION"
  else
    test_fail "PG_VERSION file missing (corrupted data?)"
  fi
else
  test_fail "pgdata directory NOT found at $PGDATA_DIR"
fi
echo ""

# 5. Check PostgreSQL client availability
echo "=========================================="
echo "5. POSTGRESQL CLIENT TOOLS"
echo "=========================================="
if command -v psql &>/dev/null; then
  test_pass "psql available: $(which psql)"
  PSQL_VERSION=$(psql --version | head -1)
  echo "Version: $PSQL_VERSION"
else
  test_warn "psql NOT installed (will need for testing)"
  echo "Install: brew install postgresql@16"
fi

if command -v pg_isready &>/dev/null; then
  test_pass "pg_isready available"
else
  test_warn "pg_isready NOT installed"
fi

if command -v pg_ctl &>/dev/null; then
  test_pass "pg_ctl available: $(which pg_ctl)"
else
  test_warn "pg_ctl NOT installed"
fi
echo ""

# 6. Check Homebrew PostgreSQL installation
echo "=========================================="
echo "6. HOMEBREW POSTGRESQL"
echo "=========================================="
if brew list postgresql@16 &>/dev/null; then
  test_pass "PostgreSQL@16 installed via Homebrew"
  PG16_PATH=$(brew --prefix postgresql@16)
  echo "Location: $PG16_PATH"
  echo "Binary: $PG16_PATH/bin/postgres"

  # Check if Homebrew PostgreSQL is running
  if brew services list | grep postgresql@16 | grep started &>/dev/null; then
    test_warn "Homebrew PostgreSQL@16 service is RUNNING (may conflict)"
    echo "Consider stopping: brew services stop postgresql@16"
  else
    test_pass "Homebrew PostgreSQL@16 service not running"
  fi
else
  test_warn "PostgreSQL@16 NOT installed via Homebrew"
  echo "Install: brew install postgresql@16"
fi
echo ""

# 7. Test database connection (if port is open)
echo "=========================================="
echo "7. DATABASE CONNECTION TEST"
echo "=========================================="
DB_URL="postgresql://dro:sicillionbillions@localhost:5544/devdb01"
if command -v psql &>/dev/null && lsof -i :5544 &>/dev/null; then
  if psql "$DB_URL" -c "SELECT 1;" &>/dev/null; then
    test_pass "Database connection successful"

    # Get row counts
    echo "Data verification:"
    psql "$DB_URL" -t -c "
      SELECT 'Games: ' || COUNT(*)::text FROM public.games
      UNION ALL
      SELECT 'Plays: ' || COUNT(*)::text FROM public.plays
      UNION ALL
      SELECT 'Player Stats: ' || COUNT(*)::text FROM public.player_stats;
    " 2>/dev/null | sed 's/^[ \t]*/  /' || echo "  (Query failed)"
  else
    test_fail "Database connection FAILED"
    echo "Connection string: $DB_URL"
    echo "Error: $(psql "$DB_URL" -c 'SELECT 1;' 2>&1 | head -3)"
  fi
else
  test_warn "Cannot test connection (port not listening or psql missing)"
fi
echo ""

# 8. Check Docker alternatives
echo "=========================================="
echo "8. DOCKER ALTERNATIVES"
echo "=========================================="
if command -v colima &>/dev/null; then
  test_pass "Colima installed: $(which colima)"
  if colima status &>/dev/null; then
    echo "Colima status: RUNNING"
  else
    echo "Colima status: STOPPED"
  fi
else
  test_warn "Colima NOT installed (option: brew install colima)"
fi

if [ -d "/Applications/OrbStack.app" ]; then
  test_pass "OrbStack installed"
else
  test_warn "OrbStack NOT installed (option: https://orbstack.dev)"
fi
echo ""

# 9. Check Docker Compose file
echo "=========================================="
echo "9. DOCKER COMPOSE CONFIGURATION"
echo "=========================================="
COMPOSE_FILE="$(cd "$(dirname "")/.." \&\& pwd)/docker-compose.yml"
if [ -f "$COMPOSE_FILE" ]; then
  test_pass "docker-compose.yml exists"

  # Extract PostgreSQL service config
  echo "PostgreSQL service configuration:"
  grep -A 15 "^  pg-1:" "$COMPOSE_FILE" | sed 's/^/  /' || true
else
  test_fail "docker-compose.yml NOT found"
fi
echo ""

# 10. Check environment variables
echo "=========================================="
echo "10. ENVIRONMENT VARIABLES"
echo "=========================================="
ENV_FILE="$(cd "$(dirname "")/.." \&\& pwd)/.env"
if [ -f "$ENV_FILE" ]; then
  test_pass ".env file exists"
  echo "Database variables:"
  grep -E "^(DB_|POSTGRES_|DATABASE_)" "$ENV_FILE" | sed 's/PASSWORD=.*/PASSWORD=***/' | sed 's/^/  /' || echo "  (None found)"
else
  test_warn ".env file NOT found (using defaults from docker-compose.yml)"
fi
echo ""

# 11. Check system resources
echo "=========================================="
echo "11. SYSTEM RESOURCES"
echo "=========================================="
echo "macOS Version: $(sw_vers -productVersion)"
echo "Chip: $(sysctl -n machdep.cpu.brand_string)"
echo "Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"
echo "Disk space (NFL analytics dir):"
df -h $(cd "$(dirname "")/.." \&\& pwd) | tail -1 | awk '{print "  Available: "$4" / "$2" ("$5" used)"}'
echo ""

# 12. Check recent Docker logs (if Docker is running)
echo "=========================================="
echo "12. DOCKER CONTAINER LOGS (Last 20 lines)"
echo "=========================================="
if docker ps -a --filter "name=nfl-analytics-pg-1" --format "{{.Names}}" | grep -q "nfl-analytics-pg-1"; then
  test_pass "PostgreSQL container exists (nfl-analytics-pg-1)"
  echo "Container status:"
  docker ps -a --filter "name=nfl-analytics-pg-1" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

  echo ""
  echo "Recent logs:"
  docker logs nfl-analytics-pg-1 --tail 20 2>&1 | sed 's/^/  /' || echo "  (Cannot read logs)"
else
  test_warn "PostgreSQL container NOT found"
fi
echo ""

# Summary
echo "=========================================="
echo "DIAGNOSTIC SUMMARY"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

# Recommendations
echo "=========================================="
echo "RECOMMENDATIONS"
echo "=========================================="

if [ "$TESTS_FAILED" -eq 0 ] && lsof -i :5544 &>/dev/null; then
  echo -e "${GREEN}✅ Database appears to be operational!${NC}"
  echo ""
  echo "Next steps:"
  echo "  1. Test Python connection: python -c 'import psycopg2; ...'"
  echo "  2. Run validation query: psql $DB_URL -c 'SELECT COUNT(*) FROM public.games;'"
  echo "  3. Resume production work"
else
  echo -e "${RED}❌ Database is NOT operational${NC}"
  echo ""
  echo "Immediate action required. Choose recovery option:"
  echo ""
  echo "FASTEST (10-15 min): Option 1 - Native PostgreSQL"
  echo "  brew install postgresql@16"
  echo "  /opt/homebrew/opt/postgresql@16/bin/pg_ctl \\"
  echo "    -D $(cd "$(dirname "")/.." \&\& pwd)/pgdata \\"
  echo "    -l $(cd "$(dirname "")/.." \&\& pwd)/logs/postgres.log \\"
  echo "    -o '-p 5544' \\"
  echo "    start"
  echo ""
  echo "ALTERNATIVE (15-20 min): Option 2 - Colima"
  echo "  brew install colima"
  echo "  colima start --cpu 4 --memory 8"
  echo "  docker compose up -d pg-1"
  echo ""
  echo "See EMERGENCY_DATABASE_RECOVERY.md for full details"
fi

echo ""
echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="
