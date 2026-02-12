#!/bin/bash
# ============================================================================
# Odds API Integration Quick Start Script
# ============================================================================
#
# This script helps you get the odds fetching service up and running quickly.
#
# Usage:
#   bash scripts/odds_quickstart.sh
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 8, 2025
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ============================================================================
# Step 1: Check Prerequisites
# ============================================================================

print_header "STEP 1: Checking Prerequisites"

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker installed: $(docker --version)"
else
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if command -v docker compose &> /dev/null; then
    print_success "Docker Compose installed: $(docker compose version)"
else
    print_error "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    print_success "Python installed: $(python3 --version)"
else
    print_warning "Python not found. Some tests will be skipped."
fi

# Check psql
if command -v psql &> /dev/null; then
    print_success "PostgreSQL client installed: $(psql --version | head -1)"
else
    print_warning "psql not found. Database checks will be skipped."
fi

# ============================================================================
# Step 2: Check Environment Variables
# ============================================================================

print_header "STEP 2: Checking Environment Variables"

# Check for .env file
if [ -f .env ]; then
    print_success ".env file exists"
    source .env
else
    print_warning ".env file not found. Creating template..."
    cat > .env << 'EOF'
# The Odds API Configuration
ODDS_API_KEY=your_api_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5544
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=sicillionbillions

# Service Configuration
ODDS_POLL_INTERVAL=900
LINE_MOVE_THRESHOLD=5.0
LOG_LEVEL=INFO

# Alerting (optional)
# ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
EOF
    print_warning "Created .env template. Please edit and add your ODDS_API_KEY."
    echo ""
    echo "Get your API key from: https://the-odds-api.com"
    echo ""
    read -p "Press Enter after setting ODDS_API_KEY in .env..."
    source .env
fi

# Check API key
if [ -z "$ODDS_API_KEY" ] || [ "$ODDS_API_KEY" = "your_api_key_here" ]; then
    print_error "ODDS_API_KEY not set in .env file"
    echo ""
    echo "Please:"
    echo "  1. Sign up at https://the-odds-api.com"
    echo "  2. Get your API key"
    echo "  3. Set ODDS_API_KEY in .env file"
    echo "  4. Run this script again"
    exit 1
else
    print_success "ODDS_API_KEY set: ${ODDS_API_KEY:0:8}..."
fi

# ============================================================================
# Step 3: Test API Connection
# ============================================================================

print_header "STEP 3: Testing The Odds API Connection"

echo "Testing API key with The Odds API..."
API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey=$ODDS_API_KEY")

if [ "$API_RESPONSE" = "200" ]; then
    print_success "API key valid and working"

    # Get quota info
    QUOTA_INFO=$(curl -s -D - \
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey=$ODDS_API_KEY" \
        | grep -i "x-requests-remaining")

    if [ ! -z "$QUOTA_INFO" ]; then
        print_success "API quota: $QUOTA_INFO"
    fi
elif [ "$API_RESPONSE" = "401" ]; then
    print_error "API key invalid or unauthorized"
    exit 1
elif [ "$API_RESPONSE" = "429" ]; then
    print_error "API rate limit exceeded"
    exit 1
else
    print_error "API request failed (HTTP $API_RESPONSE)"
    exit 1
fi

# ============================================================================
# Step 4: Check Database
# ============================================================================

print_header "STEP 4: Checking Database"

if command -v psql &> /dev/null; then
    echo "Checking PostgreSQL connection..."

    if psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
        -c "SELECT 1;" &> /dev/null; then
        print_success "Database connection successful"

        # Check if odds tables exist
        TABLES=$(psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
            -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN ('odds_history', 'prop_lines_history', 'api_usage_tracker');")

        if [ "$TABLES" -eq 3 ]; then
            print_success "All required tables exist"
        else
            print_warning "Some tables missing. Run migrations?"
            read -p "Run db/migrations/009_odds_optimization.sql? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
                    -f db/migrations/009_odds_optimization.sql
                print_success "Migrations completed"
            fi
        fi

        # Check existing data
        ODDS_COUNT=$(psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" \
            -t -c "SELECT COUNT(*) FROM odds_history;")
        print_success "odds_history has $ODDS_COUNT rows"
    else
        print_error "Database connection failed"
        echo ""
        echo "Please ensure:"
        echo "  1. PostgreSQL is running"
        echo "  2. Database credentials in .env are correct"
        echo "  3. Database 'devdb01' exists"
        exit 1
    fi
else
    print_warning "Skipping database checks (psql not installed)"
fi

# ============================================================================
# Step 5: Build Docker Image
# ============================================================================

print_header "STEP 5: Building Docker Image"

echo "Building nfl-analytics-odds-fetcher image..."
if docker build -f containers/Dockerfile.odds-fetcher -t nfl-analytics-odds-fetcher . > /tmp/docker_build.log 2>&1; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed. Check /tmp/docker_build.log"
    exit 1
fi

# ============================================================================
# Step 6: Run Test
# ============================================================================

print_header "STEP 6: Running Test (Dry Run)"

echo "Testing odds service with --once --dry-run..."
if python3 py/data/odds_update_service.py --once --dry-run > /tmp/odds_test.log 2>&1; then
    print_success "Test run successful"

    # Show test results
    echo ""
    echo "Test output:"
    tail -20 /tmp/odds_test.log | grep -E "Fetched|Poll complete|inserted|ERROR" || true
else
    print_error "Test run failed. Check /tmp/odds_test.log"
    tail -50 /tmp/odds_test.log
    exit 1
fi

# ============================================================================
# Step 7: Choose Deployment Method
# ============================================================================

print_header "STEP 7: Deployment Options"

echo ""
echo "The odds service is ready! Choose how to run it:"
echo ""
echo "Option 1: Docker Compose (Recommended)"
echo "  docker compose --profile odds up -d"
echo "  docker compose logs -f odds-fetcher"
echo ""
echo "Option 2: Standalone Docker"
echo "  docker run -d --name odds-fetcher --env-file .env --network nfl-analytics nfl-analytics-odds-fetcher"
echo "  docker logs -f odds-fetcher"
echo ""
echo "Option 3: Direct Python"
echo "  python3 py/data/odds_update_service.py"
echo ""
echo "For testing (run once, no database writes):"
echo "  python3 py/data/odds_update_service.py --once --dry-run"
echo ""

read -p "Start service with Docker Compose now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "Starting Odds Service with Docker Compose"

    # Check if network exists
    if ! docker network inspect nfl-analytics &> /dev/null; then
        echo "Creating nfl-analytics network..."
        docker network create nfl-analytics
    fi

    # Start service
    docker compose --profile odds up -d

    print_success "Service started!"
    echo ""
    echo "View logs:"
    echo "  docker compose logs -f odds-fetcher"
    echo ""
    echo "Check status:"
    echo "  docker compose ps odds-fetcher"
    echo ""
    echo "Stop service:"
    echo "  docker compose --profile odds down"
    echo ""

    # Wait a few seconds and show logs
    sleep 3
    echo "Recent logs:"
    docker compose logs --tail=20 odds-fetcher || true
fi

# ============================================================================
# Summary
# ============================================================================

print_header "SETUP COMPLETE"

echo ""
print_success "Odds API integration is ready!"
echo ""
echo "Documentation:"
echo "  📖 docs/ODDS_API_INTEGRATION.md"
echo ""
echo "Monitoring:"
echo "  📊 Grafana: http://localhost:3000 (admin/sicillionbillions)"
echo "  📈 Prometheus: http://localhost:9090"
echo "  🔍 Logs: logs/odds/odds_service_YYYYMMDD.log"
echo ""
echo "Database queries:"
echo "  📋 db/queries/odds_monitoring_dashboard.sql"
echo ""
echo "Tests:"
echo "  🧪 pytest tests/integration/test_odds_service.py -v"
echo ""
echo "Troubleshooting:"
echo "  • Check logs: docker compose logs odds-fetcher"
echo "  • Run tests: python3 py/data/odds_update_service.py --once --dry-run"
echo "  • View API usage: SELECT * FROM api_usage_tracker;"
echo "  • Check odds coverage: SELECT * FROM mart.odds_coverage;"
echo ""

print_success "Happy betting! 🎲"
