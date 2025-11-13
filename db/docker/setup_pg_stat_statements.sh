#!/bin/bash
# ============================================================
# SETUP pg_stat_statements IN DOCKER CONTAINER
# ============================================================
#
# PURPOSE: Configure PostgreSQL to enable query performance monitoring
#
# USAGE:
#   bash db/docker/setup_pg_stat_statements.sh
#
# REQUIREMENTS:
#   - Docker container running PostgreSQL 14+
#   - TimescaleDB already installed
#
# Author: NFL Analytics DevOps Team
# Date: November 13, 2025
# ============================================================

set -e

echo ""
echo "============================================================"
echo "SETTING UP pg_stat_statements IN DOCKER"
echo "============================================================"
echo ""

# Configuration
CONTAINER_NAME="nfl-analytics-pg-1"  # Adjust if your container name differs
DB_NAME="devdb01"
DB_USER="dro"
DB_PASSWORD="sicillionbillions"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================
# STEP 1: CHECK IF CONTAINER IS RUNNING
# ============================================================
echo "Step 1: Checking if PostgreSQL container is running..."
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${RED}✗ Container $CONTAINER_NAME is not running${NC}"
    echo "Available containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}"
    exit 1
fi
echo -e "${GREEN}✓ Container is running${NC}"
echo ""

# ============================================================
# STEP 2: CHECK CURRENT shared_preload_libraries
# ============================================================
echo "Step 2: Checking current shared_preload_libraries..."
CURRENT_LIBS=$(docker exec -it "$CONTAINER_NAME" \
    psql -U "$DB_USER" -d "$DB_NAME" -tAc "SHOW shared_preload_libraries;")

echo "Current value: $CURRENT_LIBS"

if [[ "$CURRENT_LIBS" == *"pg_stat_statements"* ]]; then
    echo -e "${GREEN}✓ pg_stat_statements already enabled${NC}"
    NEEDS_RESTART=false
else
    echo -e "${YELLOW}⚠ pg_stat_statements NOT enabled - will configure${NC}"
    NEEDS_RESTART=true
fi
echo ""

# ============================================================
# STEP 3: BACKUP CURRENT CONFIGURATION
# ============================================================
if [ "$NEEDS_RESTART" = true ]; then
    echo "Step 3: Backing up current postgresql.conf..."
    docker exec -it "$CONTAINER_NAME" \
        cp /var/lib/postgresql/data/postgresql.conf \
           /var/lib/postgresql/data/postgresql.conf.backup
    echo -e "${GREEN}✓ Backup created: postgresql.conf.backup${NC}"
    echo ""
fi

# ============================================================
# STEP 4: UPDATE shared_preload_libraries
# ============================================================
if [ "$NEEDS_RESTART" = true ]; then
    echo "Step 4: Updating shared_preload_libraries..."

    # Add pg_stat_statements to shared_preload_libraries
    docker exec -it "$CONTAINER_NAME" bash -c "
        grep -q 'shared_preload_libraries' /var/lib/postgresql/data/postgresql.conf && \
        sed -i \"s/shared_preload_libraries = 'timescaledb'/shared_preload_libraries = 'timescaledb,pg_stat_statements'/\" /var/lib/postgresql/data/postgresql.conf || \
        echo \"shared_preload_libraries = 'timescaledb,pg_stat_statements'\" >> /var/lib/postgresql/data/postgresql.conf
    "

    # Add pg_stat_statements configuration
    docker exec -it "$CONTAINER_NAME" bash -c "
        cat >> /var/lib/postgresql/data/postgresql.conf <<'EOF'

# pg_stat_statements configuration (added by setup script)
pg_stat_statements.track = all
pg_stat_statements.max = 10000
pg_stat_statements.track_utility = on
pg_stat_statements.save = on
EOF
    "

    echo -e "${GREEN}✓ Configuration updated${NC}"
    echo ""
fi

# ============================================================
# STEP 5: RESTART POSTGRESQL (if needed)
# ============================================================
if [ "$NEEDS_RESTART" = true ]; then
    echo "Step 5: Restarting PostgreSQL container..."
    echo -e "${YELLOW}⚠ This will briefly interrupt database connections${NC}"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker restart "$CONTAINER_NAME"
        echo "Waiting for PostgreSQL to start..."
        sleep 5

        # Wait for PostgreSQL to be ready
        until docker exec -it "$CONTAINER_NAME" \
            pg_isready -U "$DB_USER" -d "$DB_NAME" > /dev/null 2>&1; do
            echo "Waiting for PostgreSQL..."
            sleep 2
        done

        echo -e "${GREEN}✓ PostgreSQL restarted successfully${NC}"
    else
        echo -e "${RED}✗ Restart cancelled - changes will take effect on next restart${NC}"
        exit 1
    fi
    echo ""
fi

# ============================================================
# STEP 6: CREATE pg_stat_statements EXTENSION
# ============================================================
echo "Step 6: Creating pg_stat_statements extension..."
docker exec -it "$CONTAINER_NAME" \
    psql -U "$DB_USER" -d "$DB_NAME" -c \
    "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

echo -e "${GREEN}✓ Extension created${NC}"
echo ""

# ============================================================
# STEP 7: VERIFY INSTALLATION
# ============================================================
echo "Step 7: Verifying installation..."
docker exec -it "$CONTAINER_NAME" \
    psql -U "$DB_USER" -d "$DB_NAME" -c \
    "SELECT extname, extversion FROM pg_extension WHERE extname = 'pg_stat_statements';"

echo ""

# ============================================================
# STEP 8: TEST QUERY TRACKING
# ============================================================
echo "Step 8: Testing query tracking..."
docker exec -it "$CONTAINER_NAME" \
    psql -U "$DB_USER" -d "$DB_NAME" -c \
    "SELECT COUNT(*) as tracked_queries FROM pg_stat_statements;"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}pg_stat_statements SETUP COMPLETE${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Next steps:"
echo "1. Run: psql -f db/monitoring_queries_performance.sql"
echo "2. Review slow queries and optimize"
echo "3. Create missing indexes: psql -f db/optimize_indexes.sql"
echo ""
