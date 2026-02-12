#!/bin/bash
#
# Frontend Engineering Agent - Installation Script
#
# Installs the Frontend Engineering Agent and sets up all integrations:
# - Pre-commit hook
# - Initial schema snapshot
# - Python dependencies
#
# Usage:
#   ./scripts/install_frontend_agent.sh
#
# Author: Claude Code (Sonnet 4.5)
# Created: November 22, 2025

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Frontend Engineering Agent - Installation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# Step 1: Check Python Dependencies
# ============================================================================

echo -e "${YELLOW}Step 1/5: Checking Python dependencies...${NC}"

MISSING_DEPS=()

if ! python3 -c "import psycopg" 2>/dev/null; then
    MISSING_DEPS+=("psycopg[binary]")
fi

if ! python3 -c "import requests" 2>/dev/null; then
    MISSING_DEPS+=("requests")
fi

if ! python3 -c "import pydantic" 2>/dev/null; then
    MISSING_DEPS+=("pydantic")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo "Installing..."
    pip install "${MISSING_DEPS[@]}"
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ All dependencies present${NC}"
fi

echo ""

# ============================================================================
# Step 2: Make Scripts Executable
# ============================================================================

echo -e "${YELLOW}Step 2/5: Making scripts executable...${NC}"

chmod +x scripts/sync_frontend.sh
chmod +x hooks/pre-commit-frontend-validation

echo -e "${GREEN}✅ Scripts executable${NC}"
echo ""

# ============================================================================
# Step 3: Install Pre-commit Hook
# ============================================================================

echo -e "${YELLOW}Step 3/5: Installing pre-commit hook...${NC}"

if [ -f ".git/hooks/pre-commit" ]; then
    echo -e "${YELLOW}⚠️  Existing pre-commit hook found${NC}"
    echo "Backing up to .git/hooks/pre-commit.backup"
    mv .git/hooks/pre-commit .git/hooks/pre-commit.backup
fi

cp hooks/pre-commit-frontend-validation .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

echo -e "${GREEN}✅ Pre-commit hook installed${NC}"
echo ""

# ============================================================================
# Step 4: Save Initial Schema Snapshot
# ============================================================================

echo -e "${YELLOW}Step 4/5: Saving initial schema snapshot...${NC}"

# Check if database is available
if psql "${DB_CONNECTION_STRING:-postgresql://dro:sicillionbillions@localhost:5544/devdb01}" -c "SELECT 1" >/dev/null 2>&1; then
    ./scripts/sync_frontend.sh save-snapshot
    echo -e "${GREEN}✅ Schema snapshot saved${NC}"
else
    echo -e "${YELLOW}⚠️  Database not available - schema snapshot skipped${NC}"
    echo "Run './scripts/sync_frontend.sh save-snapshot' after database is up"
fi

echo ""

# ============================================================================
# Step 5: Run Initial Validation
# ============================================================================

echo -e "${YELLOW}Step 5/5: Running initial validation...${NC}"

if ./scripts/sync_frontend.sh validate-schema; then
    echo -e "${GREEN}✅ Schema validation passed${NC}"
else
    echo -e "${YELLOW}⚠️  Schema validation found issues${NC}"
    echo "Review errors and fix before committing"
fi

echo ""

# ============================================================================
# Installation Complete
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Frontend Engineering Agent Installation Complete${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Next steps:"
echo ""
echo "  1. Review validation results:"
echo "     ./scripts/sync_frontend.sh validate-schema"
echo ""
echo "  2. Check API contract:"
echo "     ./scripts/sync_frontend.sh validate-api"
echo ""
echo "  3. Run all startup checks:"
echo "     ./scripts/sync_frontend.sh startup-check"
echo ""
echo "  4. Read documentation:"
echo "     cat docs/FRONTEND_ENGINEERING_AGENT.md"
echo ""
echo "Pre-commit hook is now active - validates schemas before every commit."
echo ""
