#!/bin/bash
#
# Quick-Start Monitoring Stack Deployment
# NFL Analytics Production System
#
# Usage:
#   bash scripts/deploy_monitoring.sh
#
# What this does:
#   1. Creates Docker network if needed
#   2. Starts Prometheus + Grafana + exporters
#   3. Verifies services are healthy
#   4. Opens Grafana dashboard in browser
#   5. Prints access URLs
#
# Estimated time: 5 minutes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/Users/dro/rice/nfl-analytics"
cd "$PROJECT_ROOT"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  NFL Analytics - Monitoring Stack Deployment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Verify prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found. Please install Docker Desktop.${NC}"
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose found${NC}"

# Step 2: Create network
echo -e "${YELLOW}[2/6] Creating Docker network...${NC}"

if docker network inspect nfl-analytics &> /dev/null; then
    echo -e "${GREEN}✓ Network 'nfl-analytics' already exists${NC}"
else
    docker network create nfl-analytics
    echo -e "${GREEN}✓ Network 'nfl-analytics' created${NC}"
fi

# Step 3: Start services
echo -e "${YELLOW}[3/6] Starting monitoring services...${NC}"
echo "  This may take 2-3 minutes on first run (downloading images)..."

docker compose up -d prometheus grafana postgres_exporter node_exporter

echo -e "${GREEN}✓ Services started${NC}"

# Step 4: Wait for services to be healthy
echo -e "${YELLOW}[4/6] Waiting for services to be healthy...${NC}"

# Wait for Prometheus
echo -n "  Waiting for Prometheus..."
for i in {1..30}; do
    if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo -e " ${GREEN}healthy${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for Grafana
echo -n "  Waiting for Grafana..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e " ${GREEN}healthy${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Step 5: Verify targets
echo -e "${YELLOW}[5/6] Verifying Prometheus targets...${NC}"

sleep 5  # Give Prometheus time to scrape targets

# Check if jq is available for JSON parsing
if command -v jq &> /dev/null; then
    echo ""
    curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | "  \(.labels.job): \(.health)"' 2>/dev/null || true
    echo ""
else
    echo -e "${YELLOW}  (Install 'jq' to see target status)${NC}"
fi

# Step 6: Print access information
echo -e "${YELLOW}[6/6] Deployment complete!${NC}"
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Monitoring Stack is Ready${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
echo ""
echo -e "  📊 ${YELLOW}Grafana Dashboards:${NC}"
echo -e "     http://localhost:3000"
echo -e "     Login: admin / sicillionbillions"
echo ""
echo -e "  📈 ${YELLOW}Prometheus:${NC}"
echo -e "     http://localhost:9090"
echo ""
echo -e "  🔔 ${YELLOW}Alertmanager (optional):${NC}"
echo -e "     http://localhost:9093"
echo -e "     (Start with: docker compose --profile alerts up -d alertmanager)"
echo ""
echo -e "  📡 ${YELLOW}Metrics Endpoint (to be deployed):${NC}"
echo -e "     http://localhost:8000/metrics"
echo ""

# Step 7: Print dashboard URLs
echo -e "${BLUE}Dashboard Quick Links:${NC}"
echo ""
echo "  1. Performance Dashboard (ROI, Win Rate, P&L)"
echo "     http://localhost:3000/d/performance"
echo ""
echo "  2. Model Quality Dashboard (Calibration, MAE)"
echo "     http://localhost:3000/d/model_quality"
echo ""
echo "  3. Risk Dashboard (Drawdown, Exposure)"
echo "     http://localhost:3000/d/risk"
echo ""
echo "  4. Operations Dashboard (Latency, Errors)"
echo "     http://localhost:3000/d/operations"
echo ""

# Step 8: Print next steps
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "  1. Open Grafana and verify dashboards load:"
echo "     open http://localhost:3000"
echo ""
echo "  2. Deploy metrics endpoint (test server):"
echo "     python scripts/start_test_metrics_server.py"
echo ""
echo "  3. Simulate test data:"
echo "     python scripts/simulate_test_bets.py"
echo ""
echo "  4. Monitor dashboards for real-time updates"
echo ""
echo "  5. Review deployment guide:"
echo "     docs/MONITORING_DEPLOYMENT_GUIDE.md"
echo ""

# Step 9: Print troubleshooting
echo -e "${BLUE}Troubleshooting:${NC}"
echo ""
echo "  Check service status:"
echo "    docker compose ps"
echo ""
echo "  View logs:"
echo "    docker logs nfl-analytics-prometheus"
echo "    docker logs nfl-analytics-grafana"
echo ""
echo "  Restart services:"
echo "    docker compose restart prometheus grafana"
echo ""
echo "  Stop all services:"
echo "    docker compose down"
echo ""

# Step 10: Auto-open Grafana (optional, macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Opening Grafana in browser...${NC}"
    sleep 2
    open "http://localhost:3000" 2>/dev/null || true
fi

echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Deployment Successful! 🚀${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
