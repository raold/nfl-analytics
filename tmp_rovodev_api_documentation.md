# NFL Analytics API Documentation

**Version:** 1.0  
**Base URL:** `http://localhost:8000` (development)  
**Generated:** 2025-12-06

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [Request/Response Models](#requestresponse-models)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)
8. [Client Libraries](#client-libraries)

---

## 1. Overview

The NFL Analytics API provides access to machine learning-powered predictions for NFL games, including:
- Game outcome predictions (spreads, totals, win probability)
- Player prop predictions (passing, rushing, receiving yards)
- Model performance metrics
- System health status

### Key Features

- **5-Way Ensemble Model:** Combines Bayesian, ML, and frequentist approaches
- **Uncertainty Quantification:** Returns confidence intervals and standard deviations
- **Real-time Predictions:** Generate predictions on-demand or retrieve cached results
- **Batch Operations:** Predict entire weeks of games in single request
- **Production-Ready:** Rate limiting, authentication, monitoring

### Technology Stack

- **Framework:** FastAPI 0.104.0+
- **Server:** Uvicorn (ASGI)
- **Database:** PostgreSQL 16 + TimescaleDB
- **Cache:** Redis 7
- **Monitoring:** Prometheus metrics

---

## 2. Authentication

### API Key Authentication

All endpoints (except `/health` and `/docs`) require API key authentication.

**Method:** Header-based authentication

**Header:**
```
X-API-Key: your-api-key-here
```

**Example Request:**
```bash
curl -H "X-API-Key: your-api-key-here" \
     http://localhost:8000/models
```

### Obtaining an API Key

Contact your system administrator or generate via:
```bash
# Generate new API key (admin only)
python scripts/generate_api_key.py --user "your-name"
```

**Security Notes:**
- Never commit API keys to version control
- Rotate keys quarterly
- Use different keys for dev/staging/prod

---

## 3. Endpoints

### 3.1 Health Check

**GET** `/health`

Check API health and service availability.

**Authentication:** None required

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-06T10:30:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "r_bridge": "operational"
  },
  "models_loaded": 5,
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - All services healthy
- `503 Service Unavailable` - One or more services down

---

### 3.2 List Models

**GET** `/models`

List all approved models in the ensemble.

**Authentication:** Required

**Response:**
```json
{
  "ensemble_version": "v6.0",
  "models": [
    {
      "name": "state_space_v2",
      "type": "bayesian",
      "weight": 0.27,
      "status": "approved",
      "approved_date": "2025-11-08"
    },
    {
      "name": "hierarchical_v2",
      "type": "bayesian",
      "weight": 0.27,
      "status": "approved",
      "approved_date": "2025-11-09"
    },
    {
      "name": "xgboost_v4_ridge",
      "type": "gradient_boosting",
      "weight": 0.23,
      "status": "approved",
      "approved_date": "2025-11-15"
    }
  ]
}
```

---

### 3.3 Single Game Prediction

**POST** `/predict/game/full`

Generate full prediction for a single game with uncertainty quantification.

**Authentication:** Required

**Request Body:**
```json
{
  "home_team": "KC",
  "away_team": "BUF",
  "game_date": "2024-12-15T20:20:00Z",
  "venue": "GEHA Field at Arrowhead Stadium",
  "weather": {
    "temperature": 28,
    "wind_speed": 12,
    "precipitation": 0
  }
}
```

**Response:**
```json
{
  "game_id": "2024_15_BUF_KC",
  "prediction": {
    "home_score": 27.3,
    "away_score": 24.1,
    "spread": -3.2,
    "total": 51.4,
    "home_win_prob": 0.62
  },
  "uncertainty": {
    "home_score_std": 7.8,
    "away_score_std": 8.1,
    "spread_std": 11.2,
    "confidence_interval_95": {
      "spread": [-25.2, 18.8],
      "total": [36.2, 66.6]
    }
  },
  "component_predictions": [
    {
      "model": "state_space_v2",
      "weight": 0.27,
      "spread": -2.8,
      "total": 50.1
    },
    {
      "model": "hierarchical_v2",
      "weight": 0.27,
      "spread": -3.5,
      "total": 52.3
    }
  ],
  "metadata": {
    "generated_at": "2024-12-06T10:35:00Z",
    "models_version": "v6.0",
    "cache_hit": false
  }
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid input
- `404 Not Found` - Team not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Prediction failed

---

### 3.4 Batch Week Predictions

**POST** `/predict/week`

Generate predictions for all games in a specific week.

**Authentication:** Required

**Request Body:**
```json
{
  "season": 2024,
  "week": 15,
  "include_uncertainty": true
}
```

**Response:**
```json
{
  "season": 2024,
  "week": 15,
  "games_count": 16,
  "predictions": [
    {
      "game_id": "2024_15_BUF_KC",
      "home_team": "KC",
      "away_team": "BUF",
      "prediction": {
        "spread": -3.2,
        "total": 51.4,
        "home_win_prob": 0.62
      }
    }
  ],
  "metadata": {
    "generated_at": "2024-12-06T10:40:00Z",
    "processing_time_ms": 2340
  }
}
```

---

### 3.5 Retrieve Stored Prediction

**GET** `/predictions/game/{game_id}`

Retrieve a previously generated prediction from database.

**Authentication:** Required

**Path Parameters:**
- `game_id` (string): Game identifier (format: `YYYY_WW_AWAY_HOME`)

**Example:**
```
GET /predictions/game/2024_15_BUF_KC
```

**Response:**
```json
{
  "game_id": "2024_15_BUF_KC",
  "prediction": {
    "spread": -3.2,
    "total": 51.4,
    "home_win_prob": 0.62
  },
  "generated_at": "2024-12-06T10:35:00Z",
  "model_version": "v6.0"
}
```

---

### 3.6 Prometheus Metrics

**GET** `/metrics`

Export Prometheus-compatible metrics for monitoring.

**Authentication:** None required

**Response Format:** Prometheus text format

**Sample Metrics:**
```
# HELP prediction_requests_total Total prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total{endpoint="/predict/game/full",status="success"} 1523

# HELP prediction_duration_seconds Prediction duration
# TYPE prediction_duration_seconds histogram
prediction_duration_seconds_bucket{le="0.5"} 1200
prediction_duration_seconds_bucket{le="1.0"} 1500
```

---

## 4. Request/Response Models

### 4.1 PredictionRequest

**Model Definition:**
```python
class PredictionRequest(BaseModel):
    home_team: str                    # 2-3 letter team abbreviation
    away_team: str                    # 2-3 letter team abbreviation
    game_date: datetime               # ISO 8601 format
    venue: Optional[str] = None       # Stadium name
    weather: Optional[WeatherData] = None
```

**Validation Rules:**
- `home_team`, `away_team`: Must be valid NFL team abbreviations
- `game_date`: Must be future date (for predictions) or historical (for validation)
- `venue`: Optional, used for travel distance features

**Example:**
```json
{
  "home_team": "KC",
  "away_team": "BUF",
  "game_date": "2024-12-15T20:20:00Z",
  "venue": "GEHA Field at Arrowhead Stadium"
}
```

---

### 4.2 WeatherData

**Model Definition:**
```python
class WeatherData(BaseModel):
    temperature: Optional[float]      # Fahrenheit
    wind_speed: Optional[float]       # mph
    precipitation: Optional[float]    # inches
    humidity: Optional[float]         # percentage
    dome: bool = False                # Indoor stadium
```

---

### 4.3 PredictionResponse

**Model Definition:**
```python
class PredictionResponse(BaseModel):
    game_id: str
    prediction: GamePrediction
    uncertainty: UncertaintyQuantification
    component_predictions: List[ComponentPrediction]
    metadata: PredictionMetadata
```

---

### 4.4 GamePrediction

**Model Definition:**
```python
class GamePrediction(BaseModel):
    home_score: float
    away_score: float
    spread: float                     # Negative = home favored
    total: float                      # Over/under
    home_win_prob: float             # 0.0 to 1.0
```

---

### 4.5 UncertaintyQuantification

**Model Definition:**
```python
class UncertaintyQuantification(BaseModel):
    home_score_std: float
    away_score_std: float
    spread_std: float
    total_std: float
    confidence_interval_95: ConfidenceInterval
    quantiles: Dict[str, float]      # e.g., {"10": 45.2, "90": 58.1}
```

---

### 4.6 Team Abbreviations

**Valid NFL Team Codes:**
```
AFC East:  BUF, MIA, NE, NYJ
AFC North: BAL, CIN, CLE, PIT
AFC South: HOU, IND, JAX, TEN
AFC West:  DEN, KC, LV, LAC

NFC East:  DAL, NYG, PHI, WAS
NFC North: CHI, DET, GB, MIN
NFC South: ATL, CAR, NO, TB
NFC West:  ARI, LAR, SF, SEA
```

---

## 5. Error Handling

### 5.1 Error Response Format

All errors return a consistent JSON structure:

```json
{
  "error": {
    "code": "INVALID_TEAM",
    "message": "Team abbreviation 'XXX' not found",
    "details": {
      "valid_teams": ["BUF", "KC", ...]
    }
  },
  "timestamp": "2024-12-06T10:45:00Z"
}
```

### 5.2 Error Codes

| HTTP Status | Error Code | Description | Solution |
|------------|------------|-------------|----------|
| 400 | `INVALID_TEAM` | Team abbreviation not recognized | Use valid 2-3 letter code |
| 400 | `INVALID_DATE` | Date format incorrect | Use ISO 8601: `YYYY-MM-DDTHH:MM:SSZ` |
| 400 | `MISSING_REQUIRED_FIELD` | Required field missing | Check request schema |
| 401 | `UNAUTHORIZED` | Missing or invalid API key | Include `X-API-Key` header |
| 404 | `PREDICTION_NOT_FOUND` | Prediction doesn't exist | Generate prediction first |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry (see headers) |
| 500 | `MODEL_ERROR` | Model prediction failed | Contact support |
| 503 | `SERVICE_UNAVAILABLE` | Database or Redis down | Check `/health` endpoint |

### 5.3 Rate Limit Headers

When rate limited, response includes:

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1701864300
Retry-After: 60
```

---

## 6. Rate Limiting

### 6.1 Default Limits

**Per API Key:**
- **100 requests per minute** for prediction endpoints
- **1000 requests per minute** for read-only endpoints (`/models`, `/predictions/`)
- **Unlimited** for `/health` and `/metrics`

### 6.2 Rate Limit Algorithm

Uses **Token Bucket** algorithm with Redis backend:
- Tokens refill at constant rate
- Burst capacity = limit
- Distributed across API instances

### 6.3 Handling Rate Limits

**Best Practices:**
1. Implement exponential backoff
2. Cache predictions (use `/predictions/{game_id}` to retrieve)
3. Use batch endpoints (`/predict/week`) instead of individual calls
4. Monitor `X-RateLimit-Remaining` header

**Example Retry Logic:**
```python
import time
import requests

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited, waiting {retry_after}s...")
            time.sleep(retry_after)
            continue
            
        return response
    
    raise Exception("Max retries exceeded")
```

---

## 7. Examples

### 7.1 Python Client

**Installation:**
```bash
pip install requests
```

**Example Code:**
```python
import requests
from datetime import datetime

API_BASE = "http://localhost:8000"
API_KEY = "your-api-key-here"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Single game prediction
request_data = {
    "home_team": "KC",
    "away_team": "BUF",
    "game_date": "2024-12-15T20:20:00Z",
    "venue": "GEHA Field at Arrowhead Stadium",
    "weather": {
        "temperature": 28,
        "wind_speed": 12,
        "precipitation": 0
    }
}

response = requests.post(
    f"{API_BASE}/predict/game/full",
    json=request_data,
    headers=headers
)

if response.status_code == 200:
    prediction = response.json()
    print(f"Spread: {prediction['prediction']['spread']:.1f}")
    print(f"Total: {prediction['prediction']['total']:.1f}")
    print(f"Home Win Prob: {prediction['prediction']['home_win_prob']:.1%}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### 7.2 cURL Examples

**Health Check:**
```bash
curl http://localhost:8000/health
```

**List Models:**
```bash
curl -H "X-API-Key: your-key" \
     http://localhost:8000/models
```

**Single Prediction:**
```bash
curl -X POST \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "KC",
    "away_team": "BUF",
    "game_date": "2024-12-15T20:20:00Z"
  }' \
  http://localhost:8000/predict/game/full
```

**Batch Week Predictions:**
```bash
curl -X POST \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "season": 2024,
    "week": 15,
    "include_uncertainty": true
  }' \
  http://localhost:8000/predict/week
```

### 7.3 JavaScript/Node.js Client

**Installation:**
```bash
npm install axios
```

**Example Code:**
```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:8000';
const API_KEY = 'your-api-key-here';

const client = axios.create({
  baseURL: API_BASE,
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
  }
});

async function predictGame(homeTeam, awayTeam, gameDate) {
  try {
    const response = await client.post('/predict/game/full', {
      home_team: homeTeam,
      away_team: awayTeam,
      game_date: gameDate
    });
    
    const { prediction } = response.data;
    console.log(`Spread: ${prediction.spread.toFixed(1)}`);
    console.log(`Total: ${prediction.total.toFixed(1)}`);
    console.log(`Win Prob: ${(prediction.home_win_prob * 100).toFixed(1)}%`);
    
    return response.data;
  } catch (error) {
    if (error.response) {
      console.error(`Error ${error.response.status}:`, error.response.data);
    } else {
      console.error('Network error:', error.message);
    }
    throw error;
  }
}

// Usage
predictGame('KC', 'BUF', '2024-12-15T20:20:00Z');
```

---

## 8. Client Libraries

### 8.1 Official Client (Python)

**Location:** `dashboard/utils/api_client.py`

**Features:**
- Automatic retry with exponential backoff
- Connection pooling
- Request batching
- Type hints

**Usage:**
```python
from dashboard.utils.api_client import NFLAPIClient

client = NFLAPIClient(
    base_url="http://localhost:8000",
    api_key="your-key"
)

# Single prediction
prediction = client.get_game_prediction(game_id="2024_15_BUF_KC")

# Batch predictions
week_predictions = client.get_week_predictions(season=2024, week=15)

# Model status
models = client.get_model_status()
```

### 8.2 Batched Client (Python)

**Location:** `dashboard/utils/api_client_batched.py`

**Features:**
- Request coalescing (combines multiple requests)
- Reduces API calls by 70%+
- Automatic batching window (100ms)

**Usage:**
```python
from dashboard.utils.api_client_batched import BatchedNFLAPIClient

client = BatchedNFLAPIClient(
    base_url="http://localhost:8000",
    api_key="your-key"
)

# These requests will be batched automatically
predictions = await asyncio.gather(
    client.get_game_prediction("2024_15_BUF_KC"),
    client.get_game_prediction("2024_15_PHI_DAL"),
    client.get_game_prediction("2024_15_SF_SEA")
)
# Sent as single batch request
```

---

## 9. Performance

### 9.1 Response Times (p95)

| Endpoint | Response Time | Notes |
|----------|---------------|-------|
| `/health` | <10ms | No database queries |
| `/models` | <50ms | Cached in Redis (60s TTL) |
| `/predict/game/full` | <500ms | Includes ensemble computation |
| `/predict/week` | <3s | Batch of 16 games |
| `/predictions/{id}` | <100ms | Database lookup with index |

### 9.2 Caching Strategy

**Redis Caching:**
- Predictions: 300 seconds TTL
- Model list: 60 seconds TTL
- Game data: 30 seconds TTL

**Cache Keys:**
```
prediction:{game_id}
models:list
game:{game_id}:data
```

### 9.3 Performance Tips

1. **Use Batch Endpoints:** `/predict/week` is 5x faster than individual calls
2. **Cache Predictions:** Retrieve from `/predictions/{id}` instead of re-predicting
3. **Use Batched Client:** Reduces API calls by 70%
4. **Monitor Rate Limits:** Check `X-RateLimit-Remaining` header

---

## 10. Monitoring & Observability

### 10.1 Prometheus Metrics

**Available Metrics:**

**Request Metrics:**
```
prediction_requests_total{endpoint, status}
prediction_duration_seconds{endpoint}
prediction_errors_total{endpoint, error_type}
```

**Model Metrics:**
```
model_load_time_seconds{model_name}
model_prediction_duration_seconds{model_name}
ensemble_weight{model_name}
```

**System Metrics:**
```
api_uptime_seconds
database_connections_active
redis_cache_hit_rate
```

### 10.2 Grafana Dashboard

**Pre-built Dashboard:** `grafana/dashboards/api_performance.json`

**Panels:**
- Request rate (req/sec)
- Response time (p50, p95, p99)
- Error rate
- Model performance
- Cache hit ratio

**Access:** http://localhost:3000 (Grafana)

---

## 11. OpenAPI/Swagger Documentation

### 11.1 Interactive Documentation

**Swagger UI:** http://localhost:8000/docs

Features:
- Interactive API explorer
- "Try it out" functionality
- Request/response examples
- Schema validation

**ReDoc:** http://localhost:8000/redoc

Features:
- Cleaner, more readable layout
- Better for documentation reference
- Printable format

### 11.2 OpenAPI Spec

**Download OpenAPI JSON:**
```bash
curl http://localhost:8000/openapi.json > api_spec.json
```

**Use for Code Generation:**
```bash
# Generate Python client
openapi-generator-cli generate \
  -i api_spec.json \
  -g python \
  -o ./generated-client
```

---

## 12. Deployment

### 12.1 Docker Deployment

**Start API:**
```bash
docker-compose up api
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

### 12.2 Environment Variables

**Required:**
```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://:password@host:6379/0
API_HOST=0.0.0.0
API_PORT=8000
```

**Optional:**
```bash
MODELS_DIR=/app/models
ENSEMBLE_VERSION=v6.0
LOG_LEVEL=INFO
ENABLE_CORS=true
RATE_LIMIT_ENABLED=true
```

### 12.3 Production Checklist

- [ ] Set API keys in environment (not hard-coded)
- [ ] Configure Redis password
- [ ] Enable HTTPS (use reverse proxy like nginx)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation
- [ ] Set up health check alerts
- [ ] Test rate limiting
- [ ] Load test (see TICKET-10)
- [ ] Document runbook

---

## 13. Troubleshooting

### 13.1 Common Issues

**Issue:** `503 Service Unavailable`
- **Cause:** Database or Redis connection failed
- **Solution:** Check `/health` endpoint, verify services running

**Issue:** `401 Unauthorized`
- **Cause:** Missing or invalid API key
- **Solution:** Include `X-API-Key` header with valid key

**Issue:** `500 Internal Server Error` on `/predict/*`
- **Cause:** Model loading failed or R bridge issue
- **Solution:** Check logs, verify models exist in `models/` directory

**Issue:** Slow response times (>2s)
- **Cause:** Database connection pool exhausted or cache miss
- **Solution:** Check `database_connections_active` metric, verify Redis running

### 13.2 Debug Mode

**Enable Debug Logging:**
```bash
export LOG_LEVEL=DEBUG
uvicorn api.main:app --reload
```

**Check Logs:**
```bash
docker-compose logs -f api
```

### 13.3 Health Check Interpretation

**Healthy Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "redis": "connected",
    "r_bridge": "operational"
  },
  "models_loaded": 5
}
```

**Unhealthy Response:**
```json
{
  "status": "unhealthy",
  "services": {
    "database": "disconnected",  ← Issue here
    "redis": "connected",
    "r_bridge": "operational"
  },
  "models_loaded": 0
}
```

---

## 14. Changelog

### Version 1.0.0 (2025-11-14)
- Initial production release
- 5-way ensemble (v6.0)
- API key authentication
- Rate limiting
- Prometheus metrics

### Version 0.9.0 (2025-11-08)
- Beta release
- 4-way ensemble (v5.1)
- Basic authentication

---

## 15. Support

### 15.1 Documentation

- **Architecture:** `docs/api/ARCHITECTURE.md`
- **Deployment:** `docs/deployment/PRODUCTION_DEPLOYMENT_COMPLETE.md`
- **Comprehensive Review:** `tmp_rovodev_comprehensive_codebase_review.md`

### 15.2 Contact

For API issues or feature requests, contact:
- Email: support@nfl-analytics.local
- GitHub Issues: [repo]/issues

---

**Generated:** 2025-12-06  
**Maintainer:** NFL Analytics Team  
**API Version:** 1.0.0

