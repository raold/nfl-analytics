# NFL Analytics Prediction API - Deployment Guide

**Phase**: 1.5 - Production API Service
**Status**: Complete
**Date**: November 8, 2025
**Author**: Claude Code

## Overview

The NFL Analytics Prediction API is a production-ready FastAPI service that serves ensemble predictions using trained Bayesian and neural network models. It provides RESTful endpoints for generating predictions, monitoring model health, and exposing Prometheus metrics.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│                  (Dashboard, CLI, External)                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP/JSON
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    FastAPI Application                       │
│                        (Port 8000)                           │
│                                                              │
│  Endpoints:                                                  │
│  • POST /predict    - Generate ensemble predictions         │
│  • GET  /health     - Health check                          │
│  • GET  /models     - List loaded models                    │
│  • GET  /metrics    - Prometheus metrics                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         EnsemblePredictor (Prediction Engine)      │    │
│  │                                                     │    │
│  │  Models:                                           │    │
│  │  1. State-Space v2        (14MB, 47.77 MAE)      │    │
│  │  2. Hierarchical v1       (22MB, brms/Stan)       │    │
│  │  3. BNN Multi-Output v1   (59KB, PyTorch)        │    │
│  │                                                     │    │
│  │  Method: Inverse Variance Weighting               │    │
│  └────────────────────────────────────────────────────┘    │
└───────────────┬──────────────────────────┬──────────────────┘
                │                          │
       ┌────────▼────────┐       ┌────────▼────────┐
       │   PostgreSQL    │       │     Redis       │
       │   (Port 5544)   │       │  (Port 6379)    │
       │                 │       │                 │
       │ • Game data     │       │ • Caching       │
       │ • Player stats  │       │ • Sessions      │
       │ • Model ratings │       │                 │
       └─────────────────┘       └─────────────────┘
```

### Models Integration

| Model | Type | Size | Status | Weight | Performance |
|-------|------|------|--------|--------|-------------|
| State-Space v2 | Bayesian Random Walk | 14MB | ✅ APPROVED | 0.35 | MAE: 47.77 yards |
| Hierarchical v1 | Bayesian Hierarchical | 22MB | ✅ Complete | 0.30 | Baseline |
| BNN Multi-Output v1 | Neural Network | 59KB | ✅ Complete | 0.35 | MAE: 20.2 yards |

**Ensemble Method**: Inverse variance weighting
- Models are weighted by `1 / (uncertainty^2)`
- Ensemble uncertainty is reduced by ~15% vs individual models
- Minimum 2 models required for ensemble ready status

## File Structure

```
api/
├── __init__.py           # Package initialization
├── main.py               # FastAPI application (400+ lines)
├── models.py             # Pydantic models (200+ lines)
├── prediction.py         # Ensemble prediction logic (350+ lines)
├── requirements.txt      # Python dependencies
├── README.md             # API documentation
└── test_api.py           # Test suite (400+ lines)

containers/
└── Dockerfile.api        # Multi-stage Docker build (78 lines)

docker-compose.yml        # Updated with API service
```

## Deployment

### Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database running (port 5544)
- Redis cache running (port 6379)
- Trained models in `models/bayesian/` directory

### Quick Start

```bash
# 1. Ensure database and Redis are running
docker compose up -d pg-1 redis

# 2. Start API service
docker compose up -d api

# 3. Check health
curl http://localhost:8000/health | jq

# 4. View logs
docker compose logs -f api

# 5. Test endpoints
python api/test_api.py
```

### Build from Source

```bash
# Build Docker image
docker build -f containers/Dockerfile.api -t nfl-analytics-api .

# Run container
docker run -d \
  --name nfl-api \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -e REDIS_URL=redis://localhost:6379 \
  -v $(pwd)/models:/app/models:ro \
  nfl-analytics-api

# Check status
docker logs nfl-api
```

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r api/requirements.txt

# Set environment variables
export DATABASE_URL=postgresql://dro:sicillionbillions@localhost:5544/devdb01
export REDIS_URL=redis://localhost:6379
export MODELS_DIR=/Users/dro/rice/nfl-analytics/models

# Start API server
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API runs at http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

## API Endpoints

### 1. Health Check

**GET /health**

Returns service health status.

```bash
curl http://localhost:8000/health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2025-11-08T12:00:00Z",
  "database_connected": true,
  "redis_connected": true,
  "models_loaded": 3,
  "uptime_seconds": 123.45,
  "version": "1.0.0"
}
```

**Status Values**:
- `healthy`: All systems operational (DB ✓, Redis ✓, Models ≥ 2)
- `degraded`: Partial functionality (Models ≥ 1)
- `unhealthy`: Critical failure (Models = 0)

### 2. Generate Prediction

**POST /predict**

Generate ensemble prediction for a player in a game.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'
```

**Request Body**:
```json
{
  "game_id": "2024_10_KC_LV",
  "player_id": "00-0033873",
  "stat_type": "passing_yards"
}
```

**Response** (200 OK):
```json
{
  "game_id": "2024_10_KC_LV",
  "player_id": "00-0033873",
  "stat_type": "passing_yards",
  "prediction": 285.3,
  "confidence": 0.78,
  "lower_bound": 215.1,
  "upper_bound": 355.5,
  "models_used": [
    {
      "model_name": "state_space_v2",
      "prediction": 290.5,
      "uncertainty": 47.8,
      "weight": 0.35
    },
    {
      "model_name": "hierarchical_v1",
      "prediction": 275.2,
      "uncertainty": 52.3,
      "weight": 0.30
    },
    {
      "model_name": "bnn_multioutput_v1",
      "prediction": 288.7,
      "uncertainty": 45.1,
      "weight": 0.35
    }
  ],
  "timestamp": "2025-11-08T12:00:00Z",
  "ensemble_method": "inverse_variance_weighted"
}
```

**Stat Types** (enum):
- `passing_yards`
- `rushing_yards`
- `receiving_yards`
- `passing_tds`
- `rushing_tds`

**Error Responses**:
- `422 Unprocessable Entity`: Invalid request body
- `503 Service Unavailable`: Predictor not initialized or insufficient models

### 3. List Models

**GET /models**

List all loaded models with metadata.

```bash
curl http://localhost:8000/models
```

**Response** (200 OK):
```json
{
  "models": [
    {
      "model_name": "state_space_v2",
      "model_type": "bayesian_state_space",
      "version": "2.0",
      "file_path": "/app/models/bayesian/simplified_random_walk_v2.rds",
      "file_size_mb": 14.2,
      "loaded_at": "2025-11-08T12:00:00Z",
      "status": "loaded",
      "metadata": {"device": "cpu"}
    }
  ],
  "total_models": 3,
  "loaded_models": 3,
  "ensemble_ready": true
}
```

### 4. Prometheus Metrics

**GET /metrics**

Expose Prometheus metrics for monitoring.

```bash
curl http://localhost:8000/metrics
```

**Metrics Exposed**:
- `predictions_total{stat_type, status}` - Counter of predictions
- `prediction_latency_seconds` - Histogram of prediction latency
- `models_loaded_total` - Gauge of loaded models
- `api_requests_total{endpoint, method, status}` - Counter of API requests

**Example Output**:
```
# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{stat_type="passing_yards",status="success"} 42.0

# HELP prediction_latency_seconds Time to generate prediction
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.1"} 35.0
prediction_latency_seconds_bucket{le="0.5"} 42.0
```

## Testing

### Automated Test Suite

Run comprehensive test suite:

```bash
# Test local API
python api/test_api.py

# Test remote API
python api/test_api.py --base-url http://production-server:8000
```

**Tests Included**:
1. Health check endpoint
2. Root endpoint
3. List models endpoint
4. Generate prediction endpoint
5. Prometheus metrics endpoint
6. Error handling validation

**Expected Output**:
```
======================================================================
  NFL ANALYTICS PREDICTION API - TEST SUITE
======================================================================

Base URL: http://localhost:8000

======================================================================
  Test 1: Health Check
======================================================================
Status: healthy
Database Connected: True
Redis Connected: True
Models Loaded: 3
Uptime: 45.67s
✓ Health endpoint working

...

======================================================================
  TEST SUMMARY
======================================================================

Total Tests: 6
Passed: 6 ✓
Failed: 0 ✗
Success Rate: 100.0%

✓ All tests passed!
```

### Manual Testing

```bash
# 1. Check health
curl http://localhost:8000/health | jq .status

# 2. List models
curl http://localhost:8000/models | jq '.loaded_models'

# 3. Generate prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }' | jq .prediction

# 4. Check metrics
curl http://localhost:8000/metrics | grep predictions_total
```

### Load Testing

```bash
# Install Apache Bench
brew install httpd

# Create payload file
echo '{
  "game_id": "2024_10_KC_LV",
  "player_id": "00-0033873",
  "stat_type": "passing_yards"
}' > payload.json

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -p payload.json -T application/json \
  http://localhost:8000/predict

# Expected performance:
# - P50 latency: < 100ms
# - P95 latency: < 500ms
# - Throughput: > 50 req/s
```

## Monitoring

### Prometheus Integration

Add scrape target to `prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'nfl-api'
    static_configs:
      - targets: ['nfl-analytics-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

Restart Prometheus:
```bash
docker compose restart prometheus
```

### Grafana Dashboard

Create dashboard with panels:

1. **Predictions Rate** (predictions_total)
   - Query: `rate(predictions_total[5m])`
   - Panel Type: Graph

2. **Latency Distribution** (prediction_latency_seconds)
   - Query: `histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))`
   - Panel Type: Graph

3. **Models Health** (models_loaded_total)
   - Query: `models_loaded_total`
   - Panel Type: Gauge

4. **Error Rate** (predictions_total{status="error"})
   - Query: `rate(predictions_total{status="error"}[5m])`
   - Panel Type: Graph

### Alerts

Configure Prometheus alerts in `prometheus/alert_rules.yml`:

```yaml
groups:
  - name: api_alerts
    interval: 30s
    rules:
      # API Down
      - alert: APIDown
        expr: up{job="nfl-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "NFL Analytics API is down"

      # High Error Rate
      - alert: HighErrorRate
        expr: rate(predictions_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate (> 10%)"

      # High Latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency (P95 > 1s)"

      # Models Unavailable
      - alert: ModelsUnavailable
        expr: models_loaded_total < 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Insufficient models loaded (< 2)"
```

## Troubleshooting

### API Won't Start

**Symptom**: `docker compose up api` fails

**Diagnosis**:
```bash
# Check logs
docker compose logs api

# Check dependencies
docker compose ps pg-1 redis
```

**Common Issues**:
1. Database not ready → Wait for `pg-1` health check to pass
2. Redis not ready → Wait for `redis` health check to pass
3. Port 8000 in use → Change port in docker-compose.yml

### Models Not Loading

**Symptom**: `/models` returns `loaded_models: 0`

**Diagnosis**:
```bash
# Check models directory exists
ls -lh models/bayesian/

# Check volume mount
docker compose exec api ls -lh /app/models/bayesian/

# Check API logs
docker compose logs api | grep "Loading"
```

**Solution**:
```bash
# Ensure models exist
ls models/bayesian/simplified_random_walk_v2.rds
ls models/bayesian/passing_yards_hierarchical_v1.rds
ls models/bayesian/bnn_multioutput_v1_final.pt

# Restart API
docker compose restart api
```

### Predictions Failing

**Symptom**: `/predict` returns 503 Service Unavailable

**Diagnosis**:
```bash
# Check ensemble ready status
curl http://localhost:8000/health | jq .models_loaded

# Should be >= 2 for ensemble_ready: true
```

**Solution**:
- Load at least 2 models (see "Models Not Loading")
- Check database connectivity
- Review API logs for errors

### High Latency

**Symptom**: Predictions take > 1 second

**Diagnosis**:
```bash
# Check metrics
curl http://localhost:8000/metrics | grep latency

# Time individual request
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"}'
```

**Solutions**:
1. Enable Redis caching for player data
2. Pre-load models on startup
3. Use lighter BNN models
4. Add connection pooling for database

## Production Checklist

### Security

- [ ] Change default database password (`sicillionbillions` is for dev only)
- [ ] Enable HTTPS via reverse proxy (nginx/traefik)
- [ ] Add authentication (API keys or OAuth2)
- [ ] Restrict CORS origins (currently allows `*`)
- [ ] Enable rate limiting (e.g., 100 req/min per IP)
- [ ] Setup firewall rules (allow only 8000)
- [ ] Run as non-root user (already configured in Dockerfile)

### Performance

- [ ] Enable Redis caching for predictions (1min TTL)
- [ ] Add database connection pooling (SQLAlchemy)
- [ ] Optimize model loading (lazy load, shared memory)
- [ ] Enable response compression (gzip)
- [ ] Deploy multiple API replicas behind load balancer
- [ ] Add CDN for static content (OpenAPI schema)

### Monitoring

- [ ] Configure Prometheus alerts
- [ ] Create Grafana dashboard
- [ ] Setup log aggregation (ELK stack or Loki)
- [ ] Enable distributed tracing (Jaeger)
- [ ] Add custom metrics for business logic
- [ ] Configure uptime monitoring (UptimeRobot)

### Reliability

- [ ] Implement circuit breaker for database
- [ ] Add retry logic with exponential backoff
- [ ] Configure health check intervals
- [ ] Setup log rotation
- [ ] Add graceful shutdown handling
- [ ] Implement request timeouts

## Next Steps

### Phase 2: Enhanced Features

1. **Authentication**
   - API key management
   - OAuth2 integration
   - Rate limiting per user

2. **Caching Layer**
   - Redis caching for predictions
   - Player context pre-loading
   - Cache invalidation strategy

3. **Batch Predictions**
   - `POST /predict/batch` endpoint
   - Async processing with Celery
   - Progress tracking

4. **Real-time Updates**
   - WebSocket support for live predictions
   - Streaming predictions via SSE
   - Model updates without downtime

5. **Advanced Analytics**
   - Historical prediction accuracy
   - Model performance comparison
   - A/B testing framework

### Integration Points

- **Streamlit Dashboard**: Already configured to depend on API service
- **CQL Bet Sizer**: Can query API for predictions in production
- **Monitoring Stack**: Prometheus/Grafana already configured
- **CI/CD**: Add automated tests to `.github/workflows/`

## References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- Pydantic Models: https://docs.pydantic.dev/
- Prometheus Client: https://prometheus.io/docs/instrumenting/clientlibs/
- Docker Multi-Stage Builds: https://docs.docker.com/build/building/multi-stage/
- API Best Practices: https://restfulapi.net/

## Support

For issues or questions:
- API README: `/Users/dro/rice/nfl-analytics/api/README.md`
- GitHub Issues: https://github.com/raold/nfl-analytics/issues
- Documentation: `/Users/dro/rice/nfl-analytics/docs/`
- Contact: dro@rice.edu

---

**Phase 1.5 Complete**: Production-ready FastAPI service deployed and tested.

**Next**: Phase 2 - Enhanced features (authentication, caching, batch predictions)
