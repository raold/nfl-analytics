# NFL Analytics API Reference

**Last Updated:** December 11, 2025
**API Version:** 3.0.0
**Base URL:** `http://localhost:8000`
**OpenAPI Spec:** `http://localhost:8000/openapi.json`
**Interactive Docs:** `http://localhost:8000/docs`

---

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)
  - [Root & Health](#root--health)
  - [Game Predictions](#game-predictions)
  - [Player Props](#player-props)
  - [Cache Management](#cache-management)
  - [Monitoring](#monitoring)
- [Models & Versions](#models--versions)
- [Request/Response Schemas](#requestresponse-schemas)
- [Error Codes](#error-codes)
- [Performance](#performance)
- [Examples](#examples)

---

## Overview

The NFL Analytics API provides real-time predictions for NFL games and player props using a production-grade ensemble of Bayesian and machine learning models.

### Key Features

- **Game Predictions**: Spreads, totals, scores, win probabilities
- **Player Props**: Passing/rushing/receiving yards, touchdown probabilities
- **Ensemble Models**: 5-way ensemble (State-Space, Hierarchical, XGBoost, Informative Priors, Hybrid)
- **A/B Testing**: Dirichlet weight optimization for player props
- **Redis Caching**: Sub-100ms latency on cache hits
- **Prometheus Metrics**: Built-in observability

### Model Performance

| Model Type | Metric | Value |
|------------|--------|-------|
| Game Spreads | Cover Rate | 54.5% |
| Game Spreads | ROI | +4.0% |
| Player Props (v8.0) | MAE Improvement | 31.5% |
| Player Props (Equal Weight) | MAE | 44.25 yards |
| Inference Time | Warm Cache | ~105ms |
| Inference Time | Cold Start | ~4s |

---

## Authentication

**Current Status:** No authentication required (development mode)

**Planned:** API key authentication for production deployment
- Header: `X-API-Key: your-api-key`
- Rate limiting per API key
- Role-based access control

---

## Rate Limiting

**Current Status:** No rate limits (development mode)

**Planned Production Limits:**
- Free tier: 100 requests/hour
- Standard tier: 1,000 requests/hour
- Premium tier: 10,000 requests/hour
- Burst allowance: 10 requests/second

Response headers will include:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1702345678
```

---

## Endpoints

### Root & Health

#### `GET /`
Get API information and links to key endpoints.

**Response:** `200 OK`
```json
{
  "name": "NFL Props Prediction API",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2025-12-11T12:00:00Z",
  "uptime_seconds": 3600.5,
  "ensemble_version": "v5.1",
  "links": {
    "docs": "/docs",
    "openapi": "/openapi.json",
    "health": "/health",
    "predict_player": "/predict",
    "predict_game": "/predict/game",
    "metrics": "/metrics"
  }
}
```

---

#### `GET /health`
Service health status with component availability.

**Response:** `200 OK` (Healthy) | `503 Service Unavailable` (Unhealthy)
```json
{
  "status": "healthy",
  "timestamp": "2025-12-11T12:00:00Z",
  "version": "1.0.0",
  "models_loaded": 1,
  "cache_available": true,
  "cache_stats": {
    "hits": 12345,
    "misses": 678,
    "hit_rate": 0.948,
    "total_keys": 450
  },
  "uptime_seconds": 86400.0
}
```

**Status Values:**
- `healthy`: All components operational
- `degraded`: Some components unavailable
- `unhealthy`: Critical failures

---

#### `GET /models`
List all loaded models with metadata.

**Response:** `200 OK`
```json
{
  "models": [
    {
      "model_name": "state_space_v2",
      "model_type": "bayesian",
      "version": "v2",
      "file_path": "models/bayesian/simplified_random_walk_v2.rds",
      "file_size_mb": 1.5,
      "loaded_at": "2025-12-11T10:00:00Z",
      "status": "loaded",
      "metadata": {
        "mae": 47.77,
        "training_window": "2018-2024",
        "method": "Stan"
      }
    }
  ],
  "total_models": 4,
  "loaded_models": 4,
  "ensemble_ready": true
}
```

---

#### `GET /status`
Service status and metrics summary.

**Response:** `200 OK`
```json
{
  "total_predictions": 15234,
  "cache_hit_rate": 0.948,
  "avg_processing_ms": 145.3,
  "models_loaded": 1,
  "uptime_seconds": 86400.0
}
```

---

#### `GET /metrics`
Prometheus metrics endpoint (text/plain format).

**Response:** `200 OK`
```
# HELP nfl_api_requests_total Total API requests
# TYPE nfl_api_requests_total counter
nfl_api_requests_total{endpoint="/predict",method="POST",status="200"} 1234
...
```

---

### Game Predictions

#### `POST /api/v1/predict/game/full`
Generate full game prediction with scores, spreads, totals, and win probabilities.

**Request Body:**
```json
{
  "game_id": "2024_12_BUF_KC",
  "season": 2024,
  "week": 12
}
```

**Response:** `200 OK`
```json
{
  "game_id": "2024_12_BUF_KC",
  "season": 2024,
  "week": 12,
  "home_team": "KC",
  "away_team": "BUF",
  "predicted_home_score": 27.5,
  "predicted_away_score": 24.2,
  "predicted_total": 51.7,
  "predicted_spread": 3.3,
  "predicted_home_win_prob": 0.612,
  "spread_std": 7.1,
  "total_std": 9.5,
  "sportsbook_spread": -3.5,
  "sportsbook_total": 52.5,
  "sportsbook_home_ml": -180,
  "sportsbook_away_ml": 155,
  "spread_edge": 0.2,
  "total_edge": -0.8,
  "ml_edge": 0.022,
  "predicted_at": "2025-12-11T12:00:00Z",
  "ensemble_version": "v6.1",
  "processing_ms": 105.3
}
```

**Edge Calculation:**
- `spread_edge`: Our spread - sportsbook spread (positive = home favored)
- `total_edge`: Our total - sportsbook total (positive = over value)
- `ml_edge`: Our win prob - implied prob from moneyline (positive = home value)

---

#### `POST /api/v1/predict/week`
Batch predict all games in a given week.

**Request Body:**
```json
{
  "season": 2024,
  "week": 12
}
```

**Response:** `200 OK`
```json
{
  "season": 2024,
  "week": 12,
  "total_games": 14,
  "predictions": [
    {
      "game_id": "2024_12_BUF_KC",
      "season": 2024,
      "week": 12,
      "home_team": "KC",
      "away_team": "BUF",
      "predicted_home_score": 27.5,
      "predicted_away_score": 24.2,
      "predicted_total": 51.7,
      "predicted_spread": 3.3,
      "predicted_home_win_prob": 0.612,
      "spread_std": 7.1,
      "total_std": 9.5,
      "sportsbook_spread": -3.5,
      "sportsbook_total": 52.5,
      "spread_edge": 0.2,
      "total_edge": -0.8,
      "predicted_at": "2025-12-11T12:00:00Z",
      "ensemble_version": "v6.1",
      "processing_ms": 105.3
    }
  ],
  "processing_ms": 3456.7
}
```

**Note:** Uses Redis cache for 50-200x speedup on repeated requests.

---

#### `POST /api/v1/predict/games/batch`
Batch predict 1-50 games in a single request.

**Request Body:**
```json
{
  "games": [
    {"game_id": "2024_12_BUF_KC", "season": 2024, "week": 12},
    {"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12},
    {"game_id": "2024_12_PHI_LAR", "season": 2024, "week": 12}
  ]
}
```

**Response:** `200 OK`
```json
{
  "total_requested": 3,
  "total_successful": 3,
  "total_failed": 0,
  "predictions": [...],
  "errors": [],
  "processing_ms": 320.5,
  "cache_hit_rate": 0.67
}
```

**Partial Failures:** Returns all successful predictions plus error details for failures.

**Max Games:** 50 per request

---

#### `GET /api/v1/predictions/game/{game_id}`
Retrieve stored game prediction from database.

**Path Parameters:**
- `game_id` (required): Game ID (e.g., `2024_12_BUF_KC`)

**Query Parameters:**
- `ensemble_version` (optional): Filter by ensemble version

**Response:** `200 OK` (same schema as `/predict/game/full`)

**Example:**
```bash
curl http://localhost:8000/api/v1/predictions/game/2024_12_BUF_KC
curl http://localhost:8000/api/v1/predictions/game/2024_12_BUF_KC?ensemble_version=v6.1
```

---

#### `POST /api/v1/predict/game`
Generate QB passing yards predictions for both teams.

**Request Body:**
```json
{
  "game_id": "2024_12_BAL_LAC",
  "season": 2024,
  "week": 12
}
```

**Response:** `200 OK`
```json
{
  "game_id": "2024_12_BAL_LAC",
  "home_prediction": {
    "predicted_passing_yards": 245.3,
    "predicted_passing_yards_std": 48.2
  },
  "away_prediction": {
    "predicted_passing_yards": 268.7,
    "predicted_passing_yards_std": 52.1
  },
  "total_predicted_yards": 514.0,
  "processing_ms": 280.5
}
```

**Use Case:** Dashboard game-level predictions showing QB performance.

---

### Player Props

#### `POST /api/v1/predict/passing_yards/{player_id}`
Predict passing yards for a quarterback.

**Path Parameters:**
- `player_id` (required): Player GSIS ID (e.g., `00-0036355` for Patrick Mahomes)

**Request Body:**
```json
{
  "player_id": "00-0036355",
  "stat_type": "passing_yards",
  "week": 12,
  "season": 2025,
  "confidence_level": 0.90
}
```

**Response:** `200 OK`
```json
{
  "player_id": "00-0036355",
  "player_name": "Patrick Mahomes",
  "stat_type": "passing_yards",
  "week": 12,
  "season": 2025,
  "prediction": 268.5,
  "prediction_unit": "yards",
  "uncertainty": 32.4,
  "confidence_level": 0.90,
  "confidence_interval_lower": 210.2,
  "confidence_interval_upper": 326.8,
  "model_used": "ensemble_v6.1_dirichlet",
  "ab_group": "treatment",
  "calibration_score": 0.895,
  "timestamp": "2025-12-11T12:00:00Z",
  "processing_ms": 145.3
}
```

**A/B Testing:** 50% of requests use Dirichlet v8.0 weights (31.5% MAE improvement).

---

#### `POST /api/v1/predict/rushing_yards/{player_id}`
Predict rushing yards for a running back or QB.

**Path Parameters:**
- `player_id` (required): Player GSIS ID (e.g., `00-0034844` for Derrick Henry)

**Request Body:**
```json
{
  "player_id": "00-0034844",
  "stat_type": "rushing_yards",
  "week": 12,
  "season": 2025,
  "confidence_level": 0.90
}
```

**Response:** `200 OK` (same schema as passing yards)

**Model:** EnhancedEnsemble5Way v6.1 (no A/B test for rushing)

---

#### `POST /api/v1/predict/receiving_yards/{player_id}`
Predict receiving yards for a receiver or tight end.

**Path Parameters:**
- `player_id` (required): Player GSIS ID

**Request Body:**
```json
{
  "player_id": "00-0038120",
  "stat_type": "receiving_yards",
  "week": 12,
  "season": 2025,
  "confidence_level": 0.90
}
```

**Response:** `200 OK` (same schema as passing yards)

**Model:** EnhancedEnsemble5Way v6.1

---

#### `POST /api/v1/predict/touchdown_prob/{player_id}`
Predict touchdown probability for any skill position player.

**Path Parameters:**
- `player_id` (required): Player GSIS ID

**Request Body:**
```json
{
  "player_id": "00-0036355",
  "stat_type": "touchdowns",
  "week": 12,
  "season": 2025,
  "confidence_level": 0.90
}
```

**Response:** `200 OK`
```json
{
  "player_id": "00-0036355",
  "player_name": "Patrick Mahomes",
  "stat_type": "touchdowns",
  "week": 12,
  "season": 2025,
  "prediction": 0.65,
  "prediction_unit": "probability",
  "uncertainty": 0.10,
  "confidence_level": 0.90,
  "confidence_interval_lower": 0.50,
  "confidence_interval_upper": 0.80,
  "model_used": "xgboost_touchdowns_v1",
  "ab_group": null,
  "calibration_score": null,
  "timestamp": "2025-12-11T12:00:00Z",
  "processing_ms": 85.3
}
```

**Model:** XGBoost binary classifier v1

---

#### `GET /api/v1/props/health`
Health check for player props prediction models.

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "production_mode": true,
  "mock_data": false,
  "models": {
    "ensemble_v6.1": {
      "available": true,
      "description": "5-way ensemble (State-Space, Hierarchical, XGBoost, Informative Priors, Hybrid)",
      "stat_types": ["passing_yards", "rushing_yards", "receiving_yards"]
    },
    "ab_test_router": {
      "available": true,
      "description": "Dirichlet weights A/B test (v6.0 vs v6.1)",
      "split_ratio": "50/50"
    },
    "xgboost_touchdowns_v1": {
      "available": true,
      "description": "XGBoost binary classifier for TD probability",
      "stat_types": ["touchdowns"]
    }
  },
  "timestamp": "2025-12-11T12:00:00Z"
}
```

---

### Cache Management

#### `POST /api/v1/predict/cache/invalidate`
Invalidate cached game predictions.

**Request Body:**
```json
{
  "game_id": "2024_12_BAL_LAC",
  "season": 2024,
  "week": 12,
  "invalidate_all": false
}
```

**Use Cases:**
- **Specific game:** Provide `game_id`, `season`, `week`
- **Week refresh:** Provide `season`, `week`
- **Season refresh:** Provide `season`
- **Full clear:** Set `invalidate_all: true`

**Response:** `200 OK`
```json
{
  "invalidated_count": 1,
  "message": "Invalidated 1 cache entries for game 2024_12_BAL_LAC"
}
```

---

#### `GET /api/v1/predict/cache/stats`
Get cache statistics for monitoring.

**Response:** `200 OK`
```json
{
  "total_keys": 450,
  "redis_memory_mb": 12.5,
  "cache_available": true,
  "ensemble_version": "v6.1",
  "default_ttl_seconds": 3600
}
```

---

### Monitoring

#### `POST /predict`
Legacy single player prediction endpoint (main.py).

**Note:** Prefer `/api/v1/predict/{stat_type}/{player_id}` for new integrations.

**Request Body:**
```json
{
  "player_id": "00-0036355",
  "stat_type": "passing_yards",
  "week": 12,
  "season": 2024,
  "confidence_level": 0.90
}
```

**Response:** `200 OK` (same schema as new endpoints)

---

## Models & Versions

### Ensemble v6.1 (5-Way)

| Component | Weight | Description |
|-----------|--------|-------------|
| State-Space v2 | 27% | Bayesian state-space (Stan/cmdstanr) |
| Hierarchical v2 | 27% | Bayesian hierarchical (brms) |
| XGBoost v4 Ridge | 23% | Ridge-boosted XGBoost (4.70 MAE) |
| Informative Priors v2 | 13% | Bayesian informative priors (brms) |
| Hybrid Incremental | 10% | 70% static + 30% incremental blend |

### Dirichlet v8.0 Weights (Props A/B Test)

| Component | Weight | Notes |
|-----------|--------|-------|
| Hierarchical v2 | 39.6% | Up from 27% |
| Informative Priors v2 | 39.5% | Up from 13% |
| Hybrid Incremental | 10.1% | Similar |
| State-Space v2 | 8.2% | Down from 27% |
| XGBoost v4 Ridge | 2.6% | Down from 23% |

**Performance:** 31.5% MAE improvement on validation set

**Note:** Dec 9 holdout tests showed equal-weight averaging (44.25 MAE) outperformed Dirichlet v8.0 (44.91 MAE) on true out-of-sample 2025 data.

### Temperature Scaling

- **T = 0.7083** (v6.1, corrected Nov 23)
- **Calibration:** 90% coverage on holdout

---

## Request/Response Schemas

### GamePredictionRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `game_id` | string | Yes | Game ID (format: `YYYY_WW_AWAY_HOME`) |
| `season` | integer | Yes | NFL season (1999-2030) |
| `week` | integer | Yes | Week number (1-22) |

### GamePredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `game_id` | string | Game ID |
| `season` | integer | Season year |
| `week` | integer | Week number |
| `home_team` | string | Home team abbreviation |
| `away_team` | string | Away team abbreviation |
| `predicted_home_score` | float | Predicted home team score |
| `predicted_away_score` | float | Predicted away team score |
| `predicted_total` | float | Predicted total points |
| `predicted_spread` | float | Predicted spread (positive = home favored) |
| `predicted_home_win_prob` | float | Home team win probability (0-1) |
| `spread_std` | float | Spread uncertainty (yards) |
| `total_std` | float | Total uncertainty (points) |
| `sportsbook_spread` | float? | Sportsbook closing spread |
| `sportsbook_total` | float? | Sportsbook closing total |
| `sportsbook_home_ml` | float? | Home moneyline (American odds) |
| `sportsbook_away_ml` | float? | Away moneyline (American odds) |
| `spread_edge` | float? | Our spread - sportsbook spread |
| `total_edge` | float? | Our total - sportsbook total |
| `ml_edge` | float? | Our win prob - implied prob |
| `predicted_at` | datetime | Prediction timestamp (ISO 8601) |
| `ensemble_version` | string? | Ensemble version used |
| `processing_ms` | float | Processing time (milliseconds) |

### PropsPredictionRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `player_id` | string | Yes | Player GSIS ID |
| `stat_type` | string | Yes | `passing_yards`, `rushing_yards`, `receiving_yards`, `touchdowns` |
| `week` | integer | Yes | Week number (1-18) |
| `season` | integer | Yes | Season year (2018-2025) |
| `confidence_level` | float | No | Confidence level (0.5-0.99, default: 0.90) |

### PropsPredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `player_id` | string | Player GSIS ID |
| `player_name` | string? | Player display name |
| `stat_type` | string | Statistic type |
| `week` | integer | Week number |
| `season` | integer | Season year |
| `prediction` | float | Point prediction |
| `prediction_unit` | string | Unit (`yards` or `probability`) |
| `uncertainty` | float | Standard deviation |
| `confidence_level` | float | Confidence level used |
| `confidence_interval_lower` | float | Lower bound |
| `confidence_interval_upper` | float | Upper bound |
| `model_used` | string | Model version |
| `ab_group` | string? | A/B test group (`control`, `treatment`, or null) |
| `calibration_score` | float? | Calibration score (if available) |
| `timestamp` | string | Prediction timestamp (ISO 8601) |
| `processing_ms` | float | Processing time (milliseconds) |

---

## Error Codes

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid `game_id` format, unsupported `stat_type` |
| 404 | Not Found | Game or player not found in database |
| 500 | Internal Server Error | Model prediction failed, database error |
| 503 | Service Unavailable | Models not loaded, cache unavailable |

### Error Response Format

```json
{
  "status_code": 400,
  "detail": "Invalid game_id format: BAD_ID. Expected: YYYY_WW_AWAY_HOME",
  "timestamp": "2025-12-11T12:00:00Z"
}
```

---

## Performance

### Latency Benchmarks

| Scenario | Latency | Notes |
|----------|---------|-------|
| Cache Hit | <1ms | Redis cached response |
| Warm Prediction | ~105ms | Models loaded, no cache |
| Cold Start | ~4s | First prediction after restart |
| Batch (16 games, cached) | <2s | Parallel processing |
| Week prediction | ~3-5s | 14-16 games |

### Optimization Tips

1. **Use batch endpoints** for multiple games
2. **Leverage caching** - TTL is 1 hour by default
3. **Warm up critical predictions** during off-peak hours
4. **Monitor cache hit rates** via `/api/v1/predict/cache/stats`

---

## Examples

### Full Game Prediction (curl)

```bash
curl -X POST http://localhost:8000/api/v1/predict/game/full \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_12_BUF_KC",
    "season": 2024,
    "week": 12
  }'
```

### Player Props - Passing Yards (curl)

```bash
curl -X POST http://localhost:8000/api/v1/predict/passing_yards/00-0036355 \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0036355",
    "stat_type": "passing_yards",
    "week": 12,
    "season": 2025,
    "confidence_level": 0.90
  }'
```

### Week Batch Prediction (Python)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict/week",
    json={
        "season": 2024,
        "week": 12
    }
)

data = response.json()
print(f"Generated {data['total_games']} predictions in {data['processing_ms']:.1f}ms")

for pred in data['predictions']:
    print(f"{pred['away_team']}@{pred['home_team']}: {pred['predicted_spread']:.1f}")
```

### Invalidate Cache (curl)

```bash
curl -X POST http://localhost:8000/api/v1/predict/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{
    "season": 2024,
    "week": 12
  }'
```

### Get Cache Stats (curl)

```bash
curl http://localhost:8000/api/v1/predict/cache/stats
```

### Health Checks (curl)

```bash
# Main API health
curl http://localhost:8000/health

# Props models health
curl http://localhost:8000/api/v1/props/health

# Overall status
curl http://localhost:8000/status
```

---

## Client SDKs

### Python Example

```python
from typing import Dict, List
import requests

class NFLAnalyticsClient:
    """NFL Analytics API client."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def predict_game(self, game_id: str, season: int, week: int) -> Dict:
        """Get full game prediction."""
        response = self.session.post(
            f"{self.base_url}/api/v1/predict/game/full",
            json={"game_id": game_id, "season": season, "week": week}
        )
        response.raise_for_status()
        return response.json()

    def predict_passing_yards(
        self,
        player_id: str,
        week: int,
        season: int,
        confidence_level: float = 0.90
    ) -> Dict:
        """Get passing yards prediction."""
        response = self.session.post(
            f"{self.base_url}/api/v1/predict/passing_yards/{player_id}",
            json={
                "player_id": player_id,
                "stat_type": "passing_yards",
                "week": week,
                "season": season,
                "confidence_level": confidence_level
            }
        )
        response.raise_for_status()
        return response.json()

    def predict_week(self, season: int, week: int) -> List[Dict]:
        """Get predictions for all games in a week."""
        response = self.session.post(
            f"{self.base_url}/api/v1/predict/week",
            json={"season": season, "week": week}
        )
        response.raise_for_status()
        return response.json()["predictions"]

    def health_check(self) -> Dict:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage
client = NFLAnalyticsClient()

# Game prediction
game = client.predict_game("2024_12_BUF_KC", 2024, 12)
print(f"Spread: {game['predicted_spread']:.1f}, Total: {game['predicted_total']:.1f}")

# Player prediction
mahomes = client.predict_passing_yards("00-0036355", 12, 2024)
print(f"Mahomes: {mahomes['prediction']:.1f} ± {mahomes['uncertainty']:.1f} yards")
```

### JavaScript Example

```javascript
class NFLAnalyticsClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  async predictGame(gameId, season, week) {
    const response = await fetch(`${this.baseURL}/api/v1/predict/game/full`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId, season, week })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }

  async predictPassingYards(playerId, week, season, confidenceLevel = 0.90) {
    const response = await fetch(
      `${this.baseURL}/api/v1/predict/passing_yards/${playerId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          player_id: playerId,
          stat_type: 'passing_yards',
          week,
          season,
          confidence_level: confidenceLevel
        })
      }
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    return await response.json();
  }
}

// Usage
const client = new NFLAnalyticsClient();

(async () => {
  const game = await client.predictGame('2024_12_BUF_KC', 2024, 12);
  console.log(`Spread: ${game.predicted_spread}, Total: ${game.predicted_total}`);

  const mahomes = await client.predictPassingYards('00-0036355', 12, 2024);
  console.log(`Mahomes: ${mahomes.prediction} ± ${mahomes.uncertainty} yards`);
})();
```

---

## Support

**Documentation:** `http://localhost:8000/docs` (Swagger UI)
**OpenAPI Spec:** `http://localhost:8000/openapi.json`
**Repository:** [github.com/raold/nfl-analytics](https://github.com/raold/nfl-analytics)

**Version:** 3.0.0
**Last Updated:** December 11, 2025
**Author:** Claude Code (Sonnet 4.5)
