#!/bin/bash
# Database Sync Script for Cross-Machine Workflow
# Windows PC → Mac Laptop
# Created: November 2, 2025
#
# Purpose: Export PostgreSQL database from Windows PC and prepare for Mac import
# Usage: bash scripts/sync_db_to_laptop.sh

set -e  # Exit on error

# Configuration
DB_HOST="localhost"
DB_PORT="5544"
DB_USER="dro"
DB_NAME="devdb01"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="db/backups"
BACKUP_FILE="${BACKUP_DIR}/devdb01_${TIMESTAMP}.sql"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== NFL Analytics Database Sync ===${NC}"
echo -e "${BLUE}Exporting database for Mac laptop...${NC}\n"

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Export database
echo -e "${YELLOW}[1/4] Exporting PostgreSQL database...${NC}"
pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
  --clean --if-exists --no-owner --no-acl \
  -f "${BACKUP_FILE}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database exported: ${BACKUP_FILE}${NC}"
    echo -e "${GREEN}✓ Size: $(du -h "${BACKUP_FILE}" | cut -f1)${NC}\n"
else
    echo -e "${RED}✗ Database export failed${NC}"
    exit 1
fi

# Add to git
echo -e "${YELLOW}[2/4] Adding to git...${NC}"
git add "${BACKUP_FILE}"
echo -e "${GREEN}✓ Added to git staging${NC}\n"

# Commit
echo -e "${YELLOW}[3/4] Committing database snapshot...${NC}"
git commit -m "chore: database snapshot before Mac transition (${TIMESTAMP})

Includes:
- All tables, views, and materialized views
- Recent data additions (combine stats, enhanced features)
- Schema updates from Windows development session

To restore on Mac:
1. git pull
2. dropdb -h localhost -p 5544 -U dro devdb01
3. createdb -h localhost -p 5544 -U dro devdb01
4. psql -h localhost -p 5544 -U dro -d devdb01 -f ${BACKUP_FILE}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Committed to git${NC}\n"
else
    echo -e "${YELLOW}! Nothing to commit (no changes since last sync)${NC}\n"
fi

# Push to GitHub
echo -e "${YELLOW}[4/4] Pushing to GitHub...${NC}"
git push origin main

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Pushed to GitHub${NC}\n"
else
    echo -e "${RED}✗ Push failed${NC}"
    exit 1
fi

# Summary
echo -e "${BLUE}=== Sync Complete ===${NC}"
echo -e "${GREEN}✓ Database exported and pushed to GitHub${NC}"
echo -e "\n${BLUE}Next steps on Mac laptop:${NC}"
echo -e "  1. ${YELLOW}git pull${NC}"
echo -e "  2. ${YELLOW}dropdb -h localhost -p 5544 -U dro devdb01${NC}"
echo -e "  3. ${YELLOW}createdb -h localhost -p 5544 -U dro devdb01${NC}"
echo -e "  4. ${YELLOW}psql -h localhost -p 5544 -U dro -d devdb01 -f ${BACKUP_FILE}${NC}"
echo -e "\n${GREEN}Database will be fully synchronized!${NC}\n"
