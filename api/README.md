# NFL Analytics Prediction API

Production-ready FastAPI service for serving ensemble NFL player props predictions.

## Overview

The Prediction API provides a RESTful interface for generating ensemble predictions using trained models. It combines predictions from multiple Bayesian and neural network models using inverse variance weighting.

**Models Integrated:**
1. State-Space Random Walk v2 (APPROVED, 47.77 MAE)
2. Bayesian Hierarchical v1 (22MB, brms/Stan)
3. BNN Multi-Output v1 (59KB, PyTorch)

**Key Features:**
- Ensemble predictions with uncertainty quantification
- Prometheus metrics for monitoring
- Redis caching for performance
- Health checks and graceful degradation
- Docker containerized deployment

## Quick Start

### Local Development (without Docker)

```bash
# Install dependencies
cd /Users/dro/rice/nfl-analytics
pip install -r api/requirements.txt

# Start API server
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
```

### Docker Deployment

```bash
# Build and start all services (includes database, redis, API)
docker compose up -d

# View API logs
docker compose logs -f api

# Check API health
curl http://localhost:8000/health

# Stop services
docker compose down
```

### Start only API service

```bash
# Start dependencies first
docker compose up -d pg-1 redis

# Start API
docker compose up -d api

# Check status
docker compose ps
```

## API Endpoints

### Health Check

**GET /health**

Check service health and component status.

```bash
curl http://localhost:8000/health
```

Response:
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

### Generate Prediction

**POST /predict**

Generate ensemble prediction for a player in a specific game.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'
```

Request body:
- `game_id` (string): Game identifier (e.g., "2024_10_KC_LV")
- `player_id` (string): NFL player ID (e.g., "00-0033873" for Patrick Mahomes)
- `stat_type` (enum): One of:
  - `passing_yards`
  - `rushing_yards`
  - `receiving_yards`
  - `passing_tds`
  - `rushing_tds`

Response:
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

### List Models

**GET /models**

List all loaded models with metadata.

```bash
curl http://localhost:8000/models
```

Response:
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

### Prometheus Metrics

**GET /metrics**

Expose Prometheus metrics for monitoring.

```bash
curl http://localhost:8000/metrics
```

Metrics exposed:
- `predictions_total` - Total predictions by stat type and status
- `prediction_latency_seconds` - Prediction latency histogram
- `models_loaded_total` - Number of models loaded
- `api_requests_total` - Total API requests by endpoint

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

### Ensemble Prediction Flow

```
1. Client sends POST /predict request
2. API validates request (Pydantic)
3. EnsemblePredictor loads game/player context
4. Individual models generate predictions:
   - State-Space v2: Dynamic skill trajectories
   - Hierarchical v1: Partial pooling across players
   - BNN Multi-Output: Neural network predictions
5. Inverse variance weighting combines predictions
6. Response includes ensemble mean + 90% CI
7. Prometheus metrics recorded
```

### Model Loading Strategy

Models are loaded on startup using lazy loading pattern:

- **PyTorch models (.pt)**: Direct torch.load()
- **R models (.rds)**: Stub loader (production: rpy2)
- **Unavailable models**: Graceful degradation with warning logs

Ensemble requires at least 2 models to be considered "ready".

### Caching Strategy

Redis is used for caching:
- Player context data (5min TTL)
- Recent predictions (1min TTL)
- Model metadata (10min TTL)

## Configuration

Environment variables (set in docker-compose.yml):

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql://dro:sicillionbillions@localhost:5544/devdb01 | PostgreSQL connection string |
| REDIS_URL | redis://localhost:6379 | Redis connection string |
| API_HOST | 0.0.0.0 | API bind host |
| API_PORT | 8000 | API port |
| LOG_LEVEL | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| MODELS_DIR | /app/models | Models directory path |

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health | jq

# List models
curl http://localhost:8000/models | jq

# Generate prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }' | jq

# Check metrics
curl http://localhost:8000/metrics | grep predictions_total
```

### Load Testing

```bash
# Install Apache Bench (if not installed)
brew install httpd  # macOS

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 -p payload.json -T application/json \
  http://localhost:8000/predict

# payload.json
echo '{"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"}' > payload.json
```

## Monitoring

### Prometheus Integration

Add to `prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'nfl-api'
    static_configs:
      - targets: ['nfl-analytics-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Import dashboard JSON from `grafana/dashboards/api-performance.json` (TBD).

Key panels:
- Predictions per second
- P50/P95/P99 latency
- Error rate
- Models health

## Troubleshooting

### API won't start

```bash
# Check logs
docker compose logs api

# Common issues:
# 1. Database not ready: Wait for pg-1 health check
# 2. Redis not ready: Wait for redis health check
# 3. Models not found: Ensure models/ directory is mounted
```

### Models not loading

```bash
# Check models directory
ls -lh models/bayesian/

# Expected files:
# - simplified_random_walk_v2.rds (14MB)
# - passing_yards_hierarchical_v1.rds (22MB)
# - bnn_multioutput_v1_final.pt (59KB)

# Check API logs for model loading errors
docker compose logs api | grep "Loading"
```

### Predictions failing

```bash
# Check health endpoint
curl http://localhost:8000/health | jq .models_loaded

# Should be >= 2 for ensemble_ready: true

# Check individual model status
curl http://localhost:8000/models | jq '.models[] | select(.status != "loaded")'
```

## Development

### Adding a New Model

1. Train and save model to `models/` directory
2. Add loader to `api/prediction.py`:
   ```python
   {
       "name": "new_model_v1",
       "type": "model_type",
       "path": self.models_dir / "bayesian" / "new_model_v1.pkl",
       "loader": self._load_new_model,
       "version": "1.0"
   }
   ```
3. Implement loader function
4. Restart API: `docker compose restart api`

### Code Style

- Black formatter (line length 100)
- Type hints for all functions
- Docstrings (Google style)
- Pydantic for validation

### Dependencies

Update `api/requirements.txt` and rebuild:

```bash
docker compose build api
docker compose up -d api
```

## Production Deployment

### Security Checklist

- [ ] Change default database password
- [ ] Enable HTTPS (reverse proxy)
- [ ] Restrict CORS origins
- [ ] Add authentication (API keys/OAuth)
- [ ] Enable rate limiting
- [ ] Setup monitoring alerts
- [ ] Configure log rotation

### Performance Optimization

- **Horizontal scaling**: Deploy multiple API replicas behind load balancer
- **Model caching**: Pre-load models into Redis on startup
- **Async processing**: Use background workers for slow predictions
- **CDN**: Cache static content (OpenAPI schema)

## Support

For issues or questions:
- GitHub Issues: https://github.com/raold/nfl-analytics/issues
- Documentation: See `docs/` directory
- Contact: dro@rice.edu

## License

See LICENSE file in project root.
