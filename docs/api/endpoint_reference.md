# API Endpoint Reference

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

Complete reference documentation for all NFL Analytics Prediction API endpoints. Includes request/response schemas, authentication requirements, rate limits, error responses, and client examples.

## Table of Contents

1. [Base URL](#base-url)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
   - [GET /health](#get-health)
   - [GET /models](#get-models)
   - [POST /predict](#post-predict)
   - [POST /predict/batch](#post-predictbatch)
   - [POST /admin/refresh](#post-adminrefresh)
4. [Common Patterns](#common-patterns)
5. [Performance Characteristics](#performance-characteristics)

---

## Base URL

**Production:** `https://api.nfl-analytics.com`
**Development:** `http://localhost:8000`

All endpoints use the base URL prefix. HTTPS is required in production.

---

## Authentication

### Overview

Most endpoints require API key authentication via the `X-API-Key` header.

**Exceptions (no auth required):**
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /` - API root
- `GET /docs` - OpenAPI documentation
- `GET /redoc` - ReDoc documentation

### Header Format

```http
X-API-Key: your_64_character_api_key_here
```

### Example

```bash
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: dev_test_key_1234567890abcdef..." \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"}'
```

**See also:** [Authentication Guide](authentication_guide.md)

---

## Endpoints

### GET /health

Health check endpoint returning service status.

**Authentication:** None required
**Rate Limit:** 10,000 requests/hour
**Latency (p95):** <50ms

#### Request

```http
GET /health HTTP/1.1
Host: api.nfl-analytics.com
```

#### Response Schema

```json
{
  "status": "healthy | degraded | unhealthy",
  "timestamp": "2025-11-15T12:00:00Z",
  "database_connected": true,
  "redis_connected": true,
  "models_loaded": 4,
  "uptime_seconds": 3600.5,
  "version": "1.1.0"
}
```

**Status Values:**
- `healthy` - All systems operational (DB + Redis + 2+ models)
- `degraded` - Partial functionality (1+ models loaded)
- `unhealthy` - Service unavailable (0 models loaded)

#### Example: Healthy Response

**Request:**
```bash
curl -X GET "https://api.nfl-analytics.com/health"
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-15T12:00:00Z",
  "database_connected": true,
  "redis_connected": true,
  "models_loaded": 4,
  "uptime_seconds": 3600.5,
  "version": "1.1.0"
}
```

#### Example: Degraded Response

**Response (200 OK):**
```json
{
  "status": "degraded",
  "timestamp": "2025-11-15T12:00:00Z",
  "database_connected": true,
  "redis_connected": false,
  "models_loaded": 2,
  "uptime_seconds": 1200.3,
  "version": "1.1.0"
}
```

#### Client Examples

**Python:**
```python
import requests

response = requests.get("https://api.nfl-analytics.com/health")
health = response.json()

if health["status"] == "healthy":
    print(f"✓ API healthy ({health['models_loaded']} models loaded)")
else:
    print(f"⚠ API {health['status']}")
```

**R:**
```r
library(httr)
library(jsonlite)

response <- GET("https://api.nfl-analytics.com/health")
health <- content(response, as = "parsed")

if (health$status == "healthy") {
  cat(sprintf("✓ API healthy (%d models loaded)\n", health$models_loaded))
} else {
  cat(sprintf("⚠ API %s\n", health$status))
}
```

**JavaScript:**
```javascript
const axios = require('axios');

async function checkHealth() {
  const response = await axios.get('https://api.nfl-analytics.com/health');
  const health = response.data;

  if (health.status === 'healthy') {
    console.log(`✓ API healthy (${health.models_loaded} models loaded)`);
  } else {
    console.log(`⚠ API ${health.status}`);
  }
}
```

---

### GET /models

List all loaded models with metadata.

**Authentication:** Required (X-API-Key)
**Rate Limit:** 1,000 requests/hour
**Latency (p95):** <100ms

#### Request

```http
GET /models HTTP/1.1
Host: api.nfl-analytics.com
X-API-Key: your_api_key_here
```

#### Response Schema

```json
{
  "models": [
    {
      "model_name": "state_space_v2",
      "model_type": "bayesian_state_space",
      "version": "2.0",
      "file_path": "/app/models/bayesian/state_space_v2.rds",
      "file_size_mb": 12.5,
      "loaded_at": "2025-11-15T10:00:00Z",
      "status": "loaded",
      "metadata": {
        "training_window": "2018-2024",
        "mae": 47.77,
        "calibration": 0.925
      }
    }
  ],
  "total_models": 4,
  "loaded_models": 4,
  "ensemble_ready": true
}
```

**Model Status Values:**
- `loaded` - Model successfully loaded and ready
- `failed` - Model failed to load (check logs)
- `unavailable` - Model file not found

**Ensemble Ready:** True if ≥2 models loaded successfully

#### Example: Success Response

**Request:**
```bash
curl -X GET "https://api.nfl-analytics.com/models" \
  -H "X-API-Key: dev_test_key_1234567890abcdef..."
```

**Response (200 OK):**
```json
{
  "models": [
    {
      "model_name": "state_space_v2",
      "model_type": "bayesian_state_space",
      "version": "2.0",
      "file_path": "/app/models/bayesian/state_space_v2.rds",
      "file_size_mb": 12.5,
      "loaded_at": "2025-11-15T10:00:00Z",
      "status": "loaded",
      "metadata": {
        "training_window": "2018-2024",
        "mae": 47.77
      }
    },
    {
      "model_name": "hierarchical_v2",
      "model_type": "bayesian_hierarchical",
      "version": "2.0",
      "file_path": "/app/models/bayesian/hierarchical_v2.rds",
      "file_size_mb": 18.3,
      "loaded_at": "2025-11-15T10:00:05Z",
      "status": "loaded",
      "metadata": null
    },
    {
      "model_name": "hybrid_xgboost",
      "model_type": "xgboost",
      "version": "4.0",
      "file_path": "/app/models/xgboost/xgb_ridge_v4.pkl",
      "file_size_mb": 3.2,
      "loaded_at": "2025-11-15T10:00:10Z",
      "status": "loaded",
      "metadata": null
    },
    {
      "model_name": "informative_priors_v2",
      "model_type": "bayesian_informative_priors",
      "version": "2.0",
      "file_path": "/app/models/bayesian/informative_priors_v2.rds",
      "file_size_mb": 15.7,
      "loaded_at": "2025-11-15T10:00:15Z",
      "status": "loaded",
      "metadata": null
    }
  ],
  "total_models": 4,
  "loaded_models": 4,
  "ensemble_ready": true
}
```

#### Example: Authentication Error

**Response (401 Unauthorized):**
```json
{
  "error": "Unauthorized",
  "message": "API key required. Include 'X-API-Key' header in request.",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Headers:**
```http
WWW-Authenticate: ApiKey
```

#### Client Examples

**Python:**
```python
import requests
import os

API_KEY = os.getenv("NFL_API_KEY")
BASE_URL = "https://api.nfl-analytics.com"

response = requests.get(
    f"{BASE_URL}/models",
    headers={"X-API-Key": API_KEY}
)

if response.status_code == 200:
    data = response.json()
    print(f"Loaded: {data['loaded_models']}/{data['total_models']} models")
    print(f"Ensemble ready: {data['ensemble_ready']}")

    for model in data['models']:
        status_icon = "✓" if model['status'] == 'loaded' else "✗"
        print(f"  {status_icon} {model['model_name']} ({model['model_type']})")
else:
    print(f"Error {response.status_code}: {response.json()['message']}")
```

**R:**
```r
library(httr)
library(jsonlite)

API_KEY <- Sys.getenv("NFL_API_KEY")
BASE_URL <- "https://api.nfl-analytics.com"

response <- GET(
  url = paste0(BASE_URL, "/models"),
  add_headers("X-API-Key" = API_KEY)
)

if (status_code(response) == 200) {
  data <- content(response, as = "parsed")
  cat(sprintf("Loaded: %d/%d models\n", data$loaded_models, data$total_models))
  cat(sprintf("Ensemble ready: %s\n", data$ensemble_ready))

  for (model in data$models) {
    status_icon <- ifelse(model$status == "loaded", "✓", "✗")
    cat(sprintf("  %s %s (%s)\n", status_icon, model$model_name, model$model_type))
  }
} else {
  error <- content(response, as = "parsed")
  cat(sprintf("Error %d: %s\n", status_code(response), error$message))
}
```

---

### POST /predict

Generate ensemble prediction for a single player/game combination.

**Authentication:** Required (X-API-Key)
**Rate Limit:** 100 requests/hour (default, configurable per key)
**Latency (p95):** <500ms

#### Request Schema

```json
{
  "game_id": "string (format: YYYY_WW_HOME_AWAY)",
  "player_id": "string (nflverse format)",
  "stat_type": "passing_yards | rushing_yards | receiving_yards | passing_tds | rushing_tds"
}
```

**Field Descriptions:**

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `game_id` | string | Yes | Game identifier (season_week_home_away) | "2024_10_KC_LV" |
| `player_id` | string | Yes | NFL player ID (nflverse format) | "00-0033873" |
| `stat_type` | enum | Yes | Statistic to predict | "passing_yards" |

**Stat Type Values:**
- `passing_yards` - Passing yards (QB only)
- `rushing_yards` - Rushing yards (all positions)
- `receiving_yards` - Receiving yards (WR/TE/RB)
- `passing_tds` - Passing touchdowns (QB only)
- `rushing_tds` - Rushing touchdowns (all positions)

#### Response Schema

```json
{
  "game_id": "string",
  "player_id": "string",
  "stat_type": "string",
  "prediction": 285.3,
  "confidence": 0.78,
  "lower_bound": 215.1,
  "upper_bound": 355.5,
  "models_used": [
    {
      "model_name": "state_space_v2",
      "prediction": 290.5,
      "uncertainty": 47.8,
      "weight": 0.30
    }
  ],
  "timestamp": "2025-11-15T12:00:00Z",
  "ensemble_method": "inverse_variance_weighted"
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | float | Ensemble mean prediction |
| `confidence` | float | Model confidence (0-1) |
| `lower_bound` | float | 90% credible interval lower bound |
| `upper_bound` | float | 90% credible interval upper bound |
| `models_used` | array | Individual model contributions |
| `ensemble_method` | string | Combination method (always "inverse_variance_weighted") |

#### Example: Success Response

**Request:**
```bash
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: dev_test_key_1234567890abcdef..." \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'
```

**Response (200 OK):**
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
      "weight": 0.30
    },
    {
      "model_name": "hierarchical_v2",
      "prediction": 275.2,
      "uncertainty": 52.3,
      "weight": 0.30
    },
    {
      "model_name": "hybrid_xgboost",
      "prediction": 288.7,
      "uncertainty": 45.1,
      "weight": 0.25
    },
    {
      "model_name": "informative_priors_v2",
      "prediction": 282.1,
      "uncertainty": 50.2,
      "weight": 0.15
    }
  ],
  "timestamp": "2025-11-15T12:00:00Z",
  "ensemble_method": "inverse_variance_weighted"
}
```

**Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1731678000
```

#### Example: Validation Error

**Request:**
```bash
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: dev_test_key_1234567890abcdef..." \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "invalid_stat"
  }'
```

**Response (422 Unprocessable Entity):**
```json
{
  "error": "ValidationError",
  "message": "Invalid stat_type",
  "details": {
    "stat_type": "Must be one of [passing_yards, rushing_yards, receiving_yards, passing_tds, rushing_tds]"
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

#### Example: Rate Limit Exceeded

**Response (429 Too Many Requests):**
```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Limit: 100 requests per 1 hour(s). Try again in 3456 seconds.",
  "details": {
    "limit": 100,
    "remaining": 0,
    "reset_at": "2025-11-15T13:00:00Z"
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Headers:**
```http
Retry-After: 3456
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1731678000
```

#### Example: Service Unavailable

**Response (503 Service Unavailable):**
```json
{
  "error": "ServiceUnavailable",
  "message": "Predictor not initialized",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

#### Client Examples

**Python:**
```python
import requests
import os

API_KEY = os.getenv("NFL_API_KEY")
BASE_URL = "https://api.nfl-analytics.com"

def predict_player_stats(game_id, player_id, stat_type):
    """Generate prediction with error handling"""

    response = requests.post(
        f"{BASE_URL}/predict",
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "game_id": game_id,
            "player_id": player_id,
            "stat_type": stat_type
        },
        timeout=10
    )

    # Check rate limit
    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
    if remaining < 10:
        print(f"⚠ Warning: Only {remaining} requests remaining")

    if response.status_code == 200:
        prediction = response.json()
        return {
            "prediction": prediction["prediction"],
            "lower": prediction["lower_bound"],
            "upper": prediction["upper_bound"],
            "confidence": prediction["confidence"]
        }
    elif response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 3600))
        raise Exception(f"Rate limited. Retry in {retry_after}s")
    else:
        error = response.json()
        raise Exception(f"API error ({response.status_code}): {error['message']}")

# Usage
try:
    result = predict_player_stats(
        game_id="2024_10_KC_LV",
        player_id="00-0033873",
        stat_type="passing_yards"
    )
    print(f"Prediction: {result['prediction']:.1f} yards")
    print(f"90% CI: [{result['lower']:.1f}, {result['upper']:.1f}]")
    print(f"Confidence: {result['confidence']:.1%}")
except Exception as e:
    print(f"Error: {e}")
```

**R:**
```r
library(httr)
library(jsonlite)

API_KEY <- Sys.getenv("NFL_API_KEY")
BASE_URL <- "https://api.nfl-analytics.com"

predict_player_stats <- function(game_id, player_id, stat_type) {
  # Make prediction request
  response <- POST(
    url = paste0(BASE_URL, "/predict"),
    add_headers("X-API-Key" = API_KEY),
    body = list(
      game_id = game_id,
      player_id = player_id,
      stat_type = stat_type
    ),
    encode = "json"
  )

  # Check rate limit
  remaining <- as.numeric(headers(response)$`x-ratelimit-remaining`)
  if (!is.na(remaining) && remaining < 10) {
    warning(sprintf("Only %d requests remaining", remaining))
  }

  if (status_code(response) == 200) {
    prediction <- content(response, as = "parsed")

    list(
      prediction = prediction$prediction,
      lower = prediction$lower_bound,
      upper = prediction$upper_bound,
      confidence = prediction$confidence
    )
  } else if (status_code(response) == 429) {
    retry_after <- as.numeric(headers(response)$`retry-after`)
    stop(sprintf("Rate limited. Retry in %d seconds", retry_after))
  } else {
    error <- content(response, as = "parsed")
    stop(sprintf("API error (%d): %s", status_code(response), error$message))
  }
}

# Usage
tryCatch({
  result <- predict_player_stats(
    game_id = "2024_10_KC_LV",
    player_id = "00-0033873",
    stat_type = "passing_yards"
  )

  cat(sprintf("Prediction: %.1f yards\n", result$prediction))
  cat(sprintf("90%% CI: [%.1f, %.1f]\n", result$lower, result$upper))
  cat(sprintf("Confidence: %.1f%%\n", result$confidence * 100))
}, error = function(e) {
  cat(sprintf("Error: %s\n", e$message))
})
```

**JavaScript:**
```javascript
const axios = require('axios');

const API_KEY = process.env.NFL_API_KEY;
const BASE_URL = 'https://api.nfl-analytics.com';

async function predictPlayerStats(gameId, playerId, statType) {
  try {
    const response = await axios.post(
      `${BASE_URL}/predict`,
      {
        game_id: gameId,
        player_id: playerId,
        stat_type: statType
      },
      {
        headers: {
          'X-API-Key': API_KEY,
          'Content-Type': 'application/json'
        },
        timeout: 10000
      }
    );

    // Check rate limit
    const remaining = parseInt(response.headers['x-ratelimit-remaining']);
    if (remaining < 10) {
      console.warn(`⚠ Warning: Only ${remaining} requests remaining`);
    }

    const prediction = response.data;
    return {
      prediction: prediction.prediction,
      lower: prediction.lower_bound,
      upper: prediction.upper_bound,
      confidence: prediction.confidence
    };

  } catch (error) {
    if (error.response) {
      if (error.response.status === 429) {
        const retryAfter = parseInt(error.response.headers['retry-after']);
        throw new Error(`Rate limited. Retry in ${retryAfter}s`);
      } else {
        const errorData = error.response.data;
        throw new Error(`API error (${error.response.status}): ${errorData.message}`);
      }
    } else {
      throw new Error(`Request failed: ${error.message}`);
    }
  }
}

// Usage
(async () => {
  try {
    const result = await predictPlayerStats(
      '2024_10_KC_LV',
      '00-0033873',
      'passing_yards'
    );

    console.log(`Prediction: ${result.prediction.toFixed(1)} yards`);
    console.log(`90% CI: [${result.lower.toFixed(1)}, ${result.upper.toFixed(1)}]`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
  } catch (error) {
    console.error(`Error: ${error.message}`);
  }
})();
```

---

### POST /predict/batch

Generate predictions for multiple player/game combinations in a single request.

**Authentication:** Required (X-API-Key)
**Rate Limit:** 10 batch requests/hour (configurable per key)
**Batch Size Limit:** 100 predictions per request
**Latency (p95):** <5000ms (depends on batch size)

#### Request Schema

```json
{
  "predictions": [
    {
      "game_id": "2024_10_KC_LV",
      "player_id": "00-0033873",
      "stat_type": "passing_yards"
    },
    {
      "game_id": "2024_10_KC_LV",
      "player_id": "00-0034857",
      "stat_type": "rushing_yards"
    }
  ]
}
```

**Constraints:**
- Maximum 100 predictions per batch
- Each prediction counted separately against hourly rate limit
- Batch request itself counted as 1 batch request

#### Response Schema

```json
{
  "predictions": [
    {
      "game_id": "2024_10_KC_LV",
      "player_id": "00-0033873",
      "stat_type": "passing_yards",
      "prediction": 285.3,
      "confidence": 0.78,
      "lower_bound": 215.1,
      "upper_bound": 355.5,
      "models_used": [...],
      "timestamp": "2025-11-15T12:00:00Z",
      "ensemble_method": "inverse_variance_weighted",
      "status": "success"
    },
    {
      "game_id": "2024_10_KC_LV",
      "player_id": "00-0034857",
      "stat_type": "rushing_yards",
      "status": "error",
      "error": "Player not found in database",
      "timestamp": "2025-11-15T12:00:00Z"
    }
  ],
  "total_predictions": 2,
  "successful_predictions": 1,
  "failed_predictions": 1,
  "batch_timestamp": "2025-11-15T12:00:00Z"
}
```

**Status Values:**
- `success` - Prediction generated successfully
- `error` - Prediction failed (see `error` field)

#### Example: Batch Request

**Request:**
```bash
curl -X POST "https://api.nfl-analytics.com/predict/batch" \
  -H "X-API-Key: dev_test_key_1234567890abcdef..." \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0033873",
        "stat_type": "passing_yards"
      },
      {
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0034857",
        "stat_type": "rushing_yards"
      },
      {
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0036355",
        "stat_type": "receiving_yards"
      }
    ]
  }'
```

**Response (200 OK):**
```json
{
  "predictions": [
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
          "weight": 0.30
        },
        {
          "model_name": "hierarchical_v2",
          "prediction": 275.2,
          "uncertainty": 52.3,
          "weight": 0.30
        },
        {
          "model_name": "hybrid_xgboost",
          "prediction": 288.7,
          "uncertainty": 45.1,
          "weight": 0.25
        },
        {
          "model_name": "informative_priors_v2",
          "prediction": 282.1,
          "uncertainty": 50.2,
          "weight": 0.15
        }
      ],
      "timestamp": "2025-11-15T12:00:00Z",
      "ensemble_method": "inverse_variance_weighted",
      "status": "success"
    },
    {
      "game_id": "2024_10_KC_LV",
      "player_id": "00-0034857",
      "stat_type": "rushing_yards",
      "prediction": 92.1,
      "confidence": 0.65,
      "lower_bound": 35.2,
      "upper_bound": 149.0,
      "models_used": [...],
      "timestamp": "2025-11-15T12:00:01Z",
      "ensemble_method": "inverse_variance_weighted",
      "status": "success"
    },
    {
      "game_id": "2024_10_KC_LV",
      "player_id": "00-0036355",
      "stat_type": "receiving_yards",
      "prediction": 68.4,
      "confidence": 0.58,
      "lower_bound": 15.3,
      "upper_bound": 121.5,
      "models_used": [...],
      "timestamp": "2025-11-15T12:00:02Z",
      "ensemble_method": "inverse_variance_weighted",
      "status": "success"
    }
  ],
  "total_predictions": 3,
  "successful_predictions": 3,
  "failed_predictions": 0,
  "batch_timestamp": "2025-11-15T12:00:02Z"
}
```

#### Example: Batch Size Exceeded

**Request:**
```bash
curl -X POST "https://api.nfl-analytics.com/predict/batch" \
  -H "X-API-Key: dev_test_key_1234567890abcdef..." \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [/* 101 predictions */]
  }'
```

**Response (422 Unprocessable Entity):**
```json
{
  "error": "ValidationError",
  "message": "Batch size exceeds maximum",
  "details": {
    "batch_size": 101,
    "max_batch_size": 100
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

#### Client Examples

**Python:**
```python
import requests
import os

API_KEY = os.getenv("NFL_API_KEY")
BASE_URL = "https://api.nfl-analytics.com"

def predict_batch(predictions):
    """Generate batch predictions"""

    if len(predictions) > 100:
        raise ValueError(f"Batch size {len(predictions)} exceeds maximum 100")

    response = requests.post(
        f"{BASE_URL}/predict/batch",
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        },
        json={"predictions": predictions},
        timeout=30
    )

    if response.status_code == 200:
        return response.json()
    else:
        error = response.json()
        raise Exception(f"Batch prediction failed: {error['message']}")

# Usage
predictions_input = [
    {
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0033873",
        "stat_type": "passing_yards"
    },
    {
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0034857",
        "stat_type": "rushing_yards"
    },
    {
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0036355",
        "stat_type": "receiving_yards"
    }
]

try:
    results = predict_batch(predictions_input)

    print(f"Batch completed: {results['successful_predictions']}/{results['total_predictions']} successful")

    for pred in results['predictions']:
        if pred['status'] == 'success':
            print(f"  ✓ {pred['player_id']} {pred['stat_type']}: {pred['prediction']:.1f}")
        else:
            print(f"  ✗ {pred['player_id']} {pred['stat_type']}: {pred['error']}")

except Exception as e:
    print(f"Error: {e}")
```

**R:**
```r
library(httr)
library(jsonlite)

API_KEY <- Sys.getenv("NFL_API_KEY")
BASE_URL <- "https://api.nfl-analytics.com"

predict_batch <- function(predictions) {
  if (length(predictions) > 100) {
    stop(sprintf("Batch size %d exceeds maximum 100", length(predictions)))
  }

  response <- POST(
    url = paste0(BASE_URL, "/predict/batch"),
    add_headers("X-API-Key" = API_KEY),
    body = list(predictions = predictions),
    encode = "json",
    timeout(30)
  )

  if (status_code(response) == 200) {
    content(response, as = "parsed")
  } else {
    error <- content(response, as = "parsed")
    stop(sprintf("Batch prediction failed: %s", error$message))
  }
}

# Usage
predictions_input <- list(
  list(game_id = "2024_10_KC_LV", player_id = "00-0033873", stat_type = "passing_yards"),
  list(game_id = "2024_10_KC_LV", player_id = "00-0034857", stat_type = "rushing_yards"),
  list(game_id = "2024_10_KC_LV", player_id = "00-0036355", stat_type = "receiving_yards")
)

tryCatch({
  results <- predict_batch(predictions_input)

  cat(sprintf("Batch completed: %d/%d successful\n",
    results$successful_predictions, results$total_predictions))

  for (pred in results$predictions) {
    if (pred$status == "success") {
      cat(sprintf("  ✓ %s %s: %.1f\n", pred$player_id, pred$stat_type, pred$prediction))
    } else {
      cat(sprintf("  ✗ %s %s: %s\n", pred$player_id, pred$stat_type, pred$error))
    }
  }
}, error = function(e) {
  cat(sprintf("Error: %s\n", e$message))
})
```

---

### POST /admin/refresh

Refresh model cache and reload models from disk. Admin-only endpoint.

**Authentication:** Required (Admin API key)
**Rate Limit:** 10 requests/hour
**Latency (p95):** <30000ms (depends on model count and size)

#### Request

```http
POST /admin/refresh HTTP/1.1
Host: api.nfl-analytics.com
X-API-Key: admin_api_key_here
Content-Type: application/json
```

**Body (optional):**
```json
{
  "force_reload": true,
  "models_to_reload": ["state_space_v2", "hierarchical_v2"]
}
```

**Parameters:**
- `force_reload` (boolean, default: false) - Force reload even if models already loaded
- `models_to_reload` (array, optional) - Specific models to reload (empty = all)

#### Response Schema

```json
{
  "status": "success",
  "models_reloaded": 4,
  "models_failed": 0,
  "reload_time_seconds": 25.3,
  "models": [
    {
      "model_name": "state_space_v2",
      "status": "loaded",
      "reload_time_seconds": 8.2
    }
  ],
  "timestamp": "2025-11-15T12:00:00Z"
}
```

#### Example: Success Response

**Request:**
```bash
curl -X POST "https://api.nfl-analytics.com/admin/refresh" \
  -H "X-API-Key: admin_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "force_reload": true
  }'
```

**Response (200 OK):**
```json
{
  "status": "success",
  "models_reloaded": 4,
  "models_failed": 0,
  "reload_time_seconds": 25.3,
  "models": [
    {
      "model_name": "state_space_v2",
      "status": "loaded",
      "reload_time_seconds": 8.2
    },
    {
      "model_name": "hierarchical_v2",
      "status": "loaded",
      "reload_time_seconds": 10.1
    },
    {
      "model_name": "hybrid_xgboost",
      "status": "loaded",
      "reload_time_seconds": 2.5
    },
    {
      "model_name": "informative_priors_v2",
      "status": "loaded",
      "reload_time_seconds": 4.5
    }
  ],
  "timestamp": "2025-11-15T12:00:25Z"
}
```

#### Example: Forbidden (Non-Admin Key)

**Response (403 Forbidden):**
```json
{
  "error": "Forbidden",
  "message": "API key does not have permission for endpoint: /admin/refresh",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

---

## Common Patterns

### Exponential Backoff

Implement exponential backoff when handling rate limits:

```python
import time
import random

def predict_with_backoff(payload, max_retries=3):
    """Make prediction with exponential backoff"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                headers={"X-API-Key": API_KEY},
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    delay = min(retry_after, 2 ** attempt * 10)
                    jitter = random.uniform(0, delay * 0.1)
                    sleep_time = delay + jitter

                    print(f"Rate limited. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    raise
            else:
                raise

    raise Exception("Max retries exceeded")
```

### Connection Pooling

Use connection pooling for better performance:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create session with connection pooling
session = requests.Session()

# Configure retries
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    backoff_factor=1
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,
    pool_maxsize=20
)

session.mount("https://", adapter)
session.mount("http://", adapter)

# Make requests using session
response = session.post(
    f"{BASE_URL}/predict",
    headers={"X-API-Key": API_KEY},
    json=payload
)
```

### Rate Limit Monitoring

Monitor rate limit headers to avoid hitting limits:

```python
class RateLimitTracker:
    """Track rate limit usage"""

    def __init__(self):
        self.limit = None
        self.remaining = None
        self.reset_time = None

    def update_from_headers(self, headers):
        """Update tracking from response headers"""
        self.limit = int(headers.get("X-RateLimit-Limit", 0))
        self.remaining = int(headers.get("X-RateLimit-Remaining", 0))
        self.reset_time = int(headers.get("X-RateLimit-Reset", 0))

    def should_throttle(self, threshold=0.1):
        """Check if approaching rate limit (default: 10% remaining)"""
        if self.limit and self.remaining is not None:
            return self.remaining < (self.limit * threshold)
        return False

    def time_until_reset(self):
        """Seconds until rate limit resets"""
        if self.reset_time:
            return max(0, self.reset_time - int(time.time()))
        return 0

# Usage
tracker = RateLimitTracker()

response = requests.post(
    f"{BASE_URL}/predict",
    headers={"X-API-Key": API_KEY},
    json=payload
)

tracker.update_from_headers(response.headers)

if tracker.should_throttle():
    print(f"⚠ Warning: Only {tracker.remaining} requests remaining")
    print(f"   Resets in {tracker.time_until_reset()} seconds")
```

---

## Performance Characteristics

### Latency Targets (95th percentile)

| Endpoint | p50 | p95 | p99 |
|----------|-----|-----|-----|
| GET /health | 10ms | 50ms | 100ms |
| GET /models | 30ms | 100ms | 200ms |
| POST /predict | 200ms | 500ms | 1000ms |
| POST /predict/batch (10 items) | 1000ms | 2500ms | 5000ms |
| POST /admin/refresh | 10s | 30s | 60s |

### Timeout Recommendations

```python
TIMEOUTS = {
    "/health": 1,      # 1 second
    "/models": 5,      # 5 seconds
    "/predict": 10,    # 10 seconds
    "/predict/batch": 30,  # 30 seconds
    "/admin/refresh": 120  # 2 minutes
}
```

### Caching Recommendations

**Cache TTL by endpoint:**
- `/health` - 30 seconds (frequently updated)
- `/models` - 5 minutes (rarely changes)
- `/predict` - 1 hour (game-specific, stable)
- `/predict/batch` - 1 hour

**Cache invalidation:**
- Model cache: Invalidate on `/admin/refresh`
- Prediction cache: Invalidate on game status change

---

## Support

**Documentation:** https://docs.nfl-analytics.com
**API Status:** https://status.nfl-analytics.com
**Support Email:** support@nfl-analytics.com

**Response Times:**
- Critical (security, outage): 1 hour
- High (rate limit, auth issues): 4 hours
- Normal (questions, feature requests): 24 hours
