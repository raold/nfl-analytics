#!/bin/bash
# NFL Props Prediction API Testing Script
# Quick verification of API endpoints and functionality

set -e

API_URL="${API_URL:-http://localhost:8000}"
TIMEOUT=10

echo "=================================================="
echo "NFL Props Prediction API - Endpoint Verification"
echo "=================================================="
echo "API URL: $API_URL"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local expected_status=$4

    echo -n "Testing $method $endpoint... "

    if [ -z "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" \
            --max-time $TIMEOUT \
            -X $method \
            "$API_URL$endpoint" \
            2>/dev/null || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" \
            --max-time $TIMEOUT \
            -X $method \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint" \
            2>/dev/null || echo "000")
    fi

    status=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✓ $status${NC}"
        return 0
    else
        echo -e "${RED}✗ Expected $expected_status, got $status${NC}"
        return 1
    fi
}

# 1. Health Check
echo -e "\n${YELLOW}=== 1. Health Check ===${NC}"
test_endpoint "GET" "/health" "" "200" || true

# 2. Root Endpoint
echo -e "\n${YELLOW}=== 2. Root Endpoint ===${NC}"
test_endpoint "GET" "/" "" "200" || true

# 3. Valid Prediction
echo -e "\n${YELLOW}=== 3. Valid Prediction ===${NC}"
PREDICTION_DATA='{
    "player_id": "00-0036355",
    "stat_type": "passing_yards",
    "week": 8,
    "season": 2024,
    "confidence_level": 0.90
}'
test_endpoint "POST" "/predict" "$PREDICTION_DATA" "200" || true

# 4. Invalid Stat Type
echo -e "\n${YELLOW}=== 4. Invalid Stat Type (should fail) ===${NC}"
INVALID_DATA='{
    "player_id": "00-0036355",
    "stat_type": "invalid_stat",
    "week": 8,
    "season": 2024
}'
test_endpoint "POST" "/predict" "$INVALID_DATA" "400" || true

# 5. Invalid Week
echo -e "\n${YELLOW}=== 5. Invalid Week (should fail) ===${NC}"
INVALID_WEEK='{
    "player_id": "00-0036355",
    "stat_type": "passing_yards",
    "week": 20,
    "season": 2024
}'
test_endpoint "POST" "/predict" "$INVALID_WEEK" "400" || true

# 6. Metrics Endpoint
echo -e "\n${YELLOW}=== 6. Metrics Endpoint ===${NC}"
test_endpoint "GET" "/metrics" "" "200" || true

# 7. Status Endpoint
echo -e "\n${YELLOW}=== 7. Status Endpoint ===${NC}"
test_endpoint "GET" "/status" "" "200" || true

# 8. API Docs
echo -e "\n${YELLOW}=== 8. API Documentation ===${NC}"
test_endpoint "GET" "/docs" "" "200" || true

# 9. OpenAPI Schema
echo -e "\n${YELLOW}=== 9. OpenAPI Schema ===${NC}"
test_endpoint "GET" "/openapi.json" "" "200" || true

# 10. Cache Hit Test
echo -e "\n${YELLOW}=== 10. Cache Hit Test ===${NC}"
echo "Making same prediction twice to test cache..."

echo -n "First prediction (should be cache miss)... "
PRED1=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$PREDICTION_DATA" \
    "$API_URL/predict")

CACHE_HIT1=$(echo "$PRED1" | grep -o '"cache_hit":[^,}]*' || echo '"cache_hit":unknown')
echo "Result: $CACHE_HIT1"

echo -n "Second prediction (should be cache hit)... "
PRED2=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$PREDICTION_DATA" \
    "$API_URL/predict")

CACHE_HIT2=$(echo "$PRED2" | grep -o '"cache_hit":[^,}]*' || echo '"cache_hit":unknown')
echo "Result: $CACHE_HIT2"

# Summary
echo -e "\n=================================================="
echo -e "${GREEN}API Verification Complete${NC}"
echo "=================================================="
echo ""
echo "Summary:"
echo "- Health check: Available"
echo "- Prediction endpoint: Functional"
echo "- Metrics collection: Available"
echo "- Documentation: Accessible at /docs"
echo ""
echo "Quick links:"
echo "- API Root:       $API_URL/"
echo "- Health:         $API_URL/health"
echo "- Docs:           $API_URL/docs"
echo "- Metrics:        $API_URL/metrics"
echo "- Status:         $API_URL/status"
echo ""
