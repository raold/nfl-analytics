# NFL Props Prediction API - Quick Start Guide

**Version**: 1.0.0
**Created**: November 8, 2025
**Location**: `py/api/main.py`

## 30-Second Setup

```bash
# 1. Install dependencies
pip install fastapi uvicorn pydantic redis prometheus-client

# 2. Start development server
uvicorn py.api.main:app --host 0.0.0.0 --port 8000

# 3. Visit documentation
# Browser: http://localhost:8000/docs
```

## Quick Test

```bash
# Health check
curl -X GET http://localhost:8000/health | jq

# Make prediction
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "player_id": "00-0036355",
        "stat_type": "passing_yards",
        "week": 8,
        "season": 2024
    }' | jq

# Run automated tests
bash scripts/test_api.sh
```

## Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service status |
| `/predict` | POST | Generate prediction |
| `/metrics` | GET | Prometheus metrics |
| `/status` | GET | Service summary |
| `/docs` | GET | Interactive documentation |

## Production Deployment

```bash
# Install production server
pip install gunicorn

# Start with 4 workers
gunicorn py.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000

# Or with Docker
docker build -t nfl-api .
docker run -p 8000:8000 -e REDIS_HOST=redis nfl-api
```

## Configuration

```bash
# Set environment variables
export API_LOG_LEVEL=INFO
export MODELS_PATH=/models
export REDIS_HOST=localhost
export REDIS_PORT=6379
export CACHE_TTL_SECONDS=3600
export CORS_ORIGINS="http://localhost:3000,https://example.com"

# Start server with config
uvicorn py.api.main:app --host 0.0.0.0 --port 8000
```

## Example Request

```python
import requests
import json

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "player_id": "00-0036355",
        "stat_type": "passing_yards",
        "week": 8,
        "season": 2024,
        "confidence_level": 0.90
    }
)

prediction = response.json()

# Print results
print(f"Prediction: {prediction['prediction']:.1f} yards")
print(f"Uncertainty: ±{prediction['uncertainty']:.1f} yards")
print(f"Interval: [{prediction['confidence_interval']['lower']:.1f}, "
      f"{prediction['confidence_interval']['upper']:.1f}]")
print(f"Cache hit: {prediction['cache_hit']}")
print(f"Processing: {prediction['processing_ms']:.1f}ms")
```

## Expected Response

```json
{
    "player_id": "00-0036355",
    "player_name": "Player-36355",
    "stat_type": "passing_yards",
    "week": 8,
    "season": 2024,
    "prediction": 268.5,
    "prediction_unit": "yards",
    "uncertainty": 32.4,
    "confidence_level": 0.90,
    "confidence_interval": {
        "lower": 210.2,
        "upper": 326.8,
        "level": 0.90
    },
    "model_versions": {
        "bayesian": "state_space_v2.0",
        "bnn": "multioutput_v1_final",
        "xgboost": "v3.0",
        "state_space": "simplified_random_walk_v2",
        "ensemble": "v3.1"
    },
    "ensemble_weights": {
        "bayesian": 0.35,
        "bnn": 0.25,
        "xgboost": 0.25,
        "state_space": 0.15
    },
    "timestamp": "2024-11-08T14:30:45.123Z",
    "cache_hit": false,
    "processing_ms": 145.3
}
```

## Valid Stat Types

```
passing_yards    - QB passing yards
passing_tds      - QB touchdown passes
interceptions    - QB interceptions
rushing_yards    - RB/WR rushing yards
rushing_tds      - Rushing touchdowns
receiving_yards  - WR/TE receiving yards
receptions       - Total receptions
receiving_tds    - Receiving touchdowns
field_goals      - Kicker field goals made
extra_points     - Kicker extra points made
```

## Monitoring

```bash
# Health status
curl http://localhost:8000/health | jq '.status'

# Cache performance
curl http://localhost:8000/health | jq '.cache_stats'

# Service metrics
curl http://localhost:8000/status | jq

# Prometheus metrics
curl http://localhost:8000/metrics | grep nfl_
```

## Troubleshooting

**Port already in use?**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Redis not connecting?**
```bash
# Check Redis is running
redis-cli ping

# Verify connection
redis-cli -h localhost -p 6379 ping
```

**Models not loading?**
```bash
# Check models directory
ls -la /models/

# Verify MODELS_PATH env var
echo $MODELS_PATH

# Check startup logs
tail -f api.log | grep -i model
```

## Next Steps

1. **Read Full Documentation**: See `docs/API_SERVICE_GUIDE.md`
2. **Configure for Production**: Set environment variables
3. **Setup Monitoring**: Configure Prometheus + Grafana
4. **Integrate Cache**: Ensure Redis is running
5. **Test Under Load**: Use `locust` for load testing
6. **Deploy**: Use Docker or Kubernetes

## Key Files

- **API Code**: `py/api/main.py` (29 KB, 1,000+ lines)
- **Documentation**: `docs/API_SERVICE_GUIDE.md` (comprehensive guide)
- **Testing**: `scripts/test_api.sh` (automated verification)
- **This Guide**: `API_QUICKSTART.md`

## Support

- **API Docs**: Visit http://localhost:8000/docs
- **Project Docs**: See `CLAUDE.md`
- **Issues**: Check `docs/API_SERVICE_GUIDE.md` troubleshooting section

---

**Version**: 1.0.0
**Status**: Production-Ready
**Last Updated**: November 8, 2025
