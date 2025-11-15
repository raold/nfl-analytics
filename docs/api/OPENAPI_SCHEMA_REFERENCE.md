# OpenAPI 3.0 Schema Reference

**Date:** November 15, 2025
**Purpose:** Visual reference for OpenAPI spec structure
**Status:** Template (to be filled from actual FastAPI extraction)

---

## OpenAPI 3.0 Specification Structure

```yaml
# openapi.json (generated from FastAPI)

openapi: 3.0.0

info:
  title: NFL Analytics Prediction API
  version: 1.0.0
  description: Production API for ensemble NFL player props predictions
  contact:
    name: API Support
    url: https://github.com/raold/nfl-analytics
    email: support@nfl-analytics.local
  license:
    name: Proprietary
  x-custom:
    environment: production
    ensemble_version: v5.1
    last_updated: 2025-11-15

servers:
  - url: http://localhost:8000
    description: Development (local)
    variables:
      host:
        default: localhost
      port:
        default: '8000'
  - url: https://api-dev.nfl-analytics.local/api/v1
    description: Development (cloud)
  - url: https://api-staging.nfl-analytics.local/api/v1
    description: Staging
  - url: https://api.nfl-analytics.local/api/v1
    description: Production

tags:
  - name: Health
    description: Service health and status
  - name: Predictions
    description: Ensemble predictions for player stats
  - name: Models
    description: Loaded models and metadata
  - name: Monitoring
    description: Prometheus metrics and monitoring
  - name: Root
    description: API root endpoints

paths:
  /health:
    get:
      summary: Health check endpoint
      description: |
        Returns service health status including:
        - Database connectivity
        - Redis connectivity
        - Models loaded
        - Uptime
      operationId: health_check
      tags:
        - Health
      responses:
        '200':
          description: Service health status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'
              example:
                status: healthy
                timestamp: 2025-11-15T14:30:45.123Z
                database_connected: true
                redis_connected: true
                models_loaded: 4
                uptime_seconds: 3600.45
                version: 1.0.0

  /predict:
    post:
      summary: Generate ensemble prediction
      description: |
        Combines predictions from all loaded models using inverse variance weighting.

        The response includes:
        - Ensemble prediction (mean)
        - Confidence interval (90% credible bounds)
        - Individual model contributions
        - Prediction metadata
      operationId: predict
      tags:
        - Predictions
      security:
        - api_key: []  # Required after P0 implementation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PredictionRequest'
            example:
              game_id: "2024_10_KC_BUF"
              player_id: "00-0033873"
              stat_type: passing_yards
      responses:
        '200':
          description: Ensemble prediction with model breakdown
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PredictionResponse'
              example:
                game_id: "2024_10_KC_BUF"
                player_id: "00-0033873"
                stat_type: passing_yards
                prediction: 285.3
                confidence: 0.78
                lower_bound: 215.1
                upper_bound: 355.5
                models_used:
                  - model_name: state_space_v2
                    prediction: 290.5
                    uncertainty: 47.8
                    weight: 0.30
                  - model_name: hierarchical_v2
                    prediction: 275.2
                    uncertainty: 52.3
                    weight: 0.25
                timestamp: 2025-11-15T14:30:45.123Z
                ensemble_method: inverse_variance_weighted
        '400':
          description: Validation error (bad request)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: ValidationError
                message: 1 validation error for PredictionRequest
                details:
                  loc: [body, game_id]
                  msg: Invalid game_id format
                timestamp: 2025-11-15T14:30:45.123Z
        '503':
          description: Service unavailable (models not loaded)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
              example:
                error: HTTPException
                message: Insufficient models loaded (need at least 2)
                details:
                  status_code: 503
                timestamp: 2025-11-15T14:30:45.123Z
        '500':
          description: Internal server error (prediction failed)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /models:
    get:
      summary: List loaded models
      description: |
        Returns all loaded models with metadata including:
        - Model name, type, version
        - File path and size
        - Load status and timestamp
        - Ensemble readiness
      operationId: list_models
      tags:
        - Models
      security:
        - api_key: []  # Required after P0 implementation
      responses:
        '200':
          description: List of loaded models
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ModelsListResponse'
              example:
                models:
                  - model_name: state_space_v2
                    model_type: bayesian_state_space
                    version: 2.0
                    file_path: /app/models/bayesian/simplified_random_walk_v2.rds
                    file_size_mb: 45.2
                    loaded_at: 2025-11-15T14:30:00.000Z
                    status: loaded
                    metadata:
                      device: cpu
                      weight: 0.30
                      ensemble_version: 5.0
                total_models: 4
                loaded_models: 4
                ensemble_ready: true
        '503':
          description: Service unavailable

  /metrics:
    get:
      summary: Prometheus metrics
      description: |
        Exposes application metrics in Prometheus format:
        - predictions_total - Total predictions by stat type and status
        - prediction_latency_seconds - Prediction latency histogram
        - models_loaded_total - Number of models loaded
        - api_requests_total - Total API requests by endpoint
      operationId: metrics
      tags:
        - Monitoring
      responses:
        '200':
          description: Prometheus metrics (text format)
          content:
            text/plain:
              schema:
                type: string
              example: |
                # HELP predictions_total Total number of predictions made
                # TYPE predictions_total counter
                predictions_total{stat_type="passing_yards",status="success"} 1234.0

  /:
    get:
      summary: API root
      description: Redirect to API documentation
      operationId: root
      tags:
        - Root
      responses:
        '200':
          description: API information
          content:
            application/json:
              schema:
                type: object
                properties:
                  name:
                    type: string
                  version:
                    type: string
                  status:
                    type: string
                  docs:
                    type: string
                    format: uri
                  health:
                    type: string
                    format: uri
                  metrics:
                    type: string
                    format: uri
              example:
                name: NFL Analytics Prediction API
                version: 1.0.0
                status: running
                docs: /docs
                health: /health
                metrics: /metrics

components:
  schemas:
    StatType:
      type: string
      enum:
        - passing_yards
        - rushing_yards
        - receiving_yards
        - passing_tds
        - rushing_tds
      description: Type of stat to predict

    PredictionRequest:
      type: object
      required:
        - game_id
        - player_id
        - stat_type
      properties:
        game_id:
          type: string
          description: Game identifier (format YYYY_WW_HOME_AWAY)
          example: "2024_10_KC_BUF"
          pattern: '^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$'
        player_id:
          type: string
          description: NFL player ID (format 00-XXXXXXX)
          example: "00-0033873"
          pattern: '^\d{2}-\d{7}$'
        stat_type:
          $ref: '#/components/schemas/StatType'

    ModelContribution:
      type: object
      properties:
        model_name:
          type: string
          description: Model identifier
          example: state_space_v2
        prediction:
          type: number
          format: float
          description: Model prediction value
          example: 290.5
        uncertainty:
          type: number
          format: float
          nullable: true
          description: Prediction uncertainty (standard deviation)
          example: 47.8
        weight:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: Weight in ensemble (0-1)
          example: 0.30

    PredictionResponse:
      type: object
      properties:
        game_id:
          type: string
          example: "2024_10_KC_BUF"
        player_id:
          type: string
          example: "00-0033873"
        stat_type:
          $ref: '#/components/schemas/StatType'
        prediction:
          type: number
          format: float
          description: Ensemble mean prediction
          example: 285.3
        confidence:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: Model confidence (0-1)
          example: 0.78
        lower_bound:
          type: number
          format: float
          description: 90% credible interval lower bound (Q10)
          example: 215.1
        upper_bound:
          type: number
          format: float
          description: 90% credible interval upper bound (Q90)
          example: 355.5
        models_used:
          type: array
          items:
            $ref: '#/components/schemas/ModelContribution'
        timestamp:
          type: string
          format: date-time
          description: Prediction timestamp (UTC)
          example: 2025-11-15T14:30:45.123Z
        ensemble_method:
          type: string
          description: Ensemble combination method
          example: inverse_variance_weighted
      required:
        - game_id
        - player_id
        - stat_type
        - prediction
        - confidence
        - lower_bound
        - upper_bound
        - models_used
        - timestamp
        - ensemble_method

    ModelInfo:
      type: object
      properties:
        model_name:
          type: string
          example: state_space_v2
        model_type:
          type: string
          description: Model architecture (e.g., bayesian_state_space)
          example: bayesian_state_space
        version:
          type: string
          example: "2.0"
        file_path:
          type: string
          format: path
          example: /app/models/bayesian/simplified_random_walk_v2.rds
        file_size_mb:
          type: number
          format: float
          example: 45.2
        loaded_at:
          type: string
          format: date-time
          example: 2025-11-15T14:30:00.000Z
        status:
          type: string
          enum: [loaded, failed, unavailable]
          example: loaded
        metadata:
          type: object
          nullable: true
          example:
            device: cpu
            weight: 0.30
            ensemble_version: 5.0

    ModelsListResponse:
      type: object
      properties:
        models:
          type: array
          items:
            $ref: '#/components/schemas/ModelInfo'
        total_models:
          type: integer
          example: 4
        loaded_models:
          type: integer
          example: 4
        ensemble_ready:
          type: boolean
          example: true
      required:
        - models
        - total_models
        - loaded_models
        - ensemble_ready

    HealthResponse:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
          example: healthy
        timestamp:
          type: string
          format: date-time
          example: 2025-11-15T14:30:45.123Z
        database_connected:
          type: boolean
          example: true
        redis_connected:
          type: boolean
          example: true
        models_loaded:
          type: integer
          example: 4
        uptime_seconds:
          type: number
          format: float
          example: 3600.45
        version:
          type: string
          example: 1.0.0
      required:
        - status
        - timestamp
        - database_connected
        - redis_connected
        - models_loaded
        - uptime_seconds
        - version

    ErrorResponse:
      type: object
      properties:
        error:
          type: string
          description: Error type (exception class)
          example: ValidationError
        message:
          type: string
          description: Human-readable error message
          example: Invalid game_id format
        details:
          type: object
          nullable: true
          description: Additional error details
          example:
            loc: [body, game_id]
            msg: value error, invalid game_id format
        timestamp:
          type: string
          format: date-time
          example: 2025-11-15T14:30:45.123Z
      required:
        - error
        - message
        - timestamp

  securitySchemes:
    api_key:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for authentication
      # Note: Add to all protected endpoints after P0 implementation

  responses:
    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'

    Unauthorized:
      description: Invalid or missing authentication
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'

    RateLimited:
      description: Rate limit exceeded
      headers:
        X-RateLimit-Limit:
          schema:
            type: integer
            example: 100
        X-RateLimit-Remaining:
          schema:
            type: integer
            example: 0
        X-RateLimit-Reset:
          schema:
            type: integer
            example: 1731688260
        Retry-After:
          schema:
            type: integer
            example: 30
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'

  examples:
    SuccessfulPrediction:
      summary: Successful ensemble prediction
      value:
        game_id: "2024_10_KC_BUF"
        player_id: "00-0033873"
        stat_type: passing_yards
        prediction: 285.3
        confidence: 0.78
        lower_bound: 215.1
        upper_bound: 355.5
        models_used:
          - model_name: state_space_v2
            prediction: 290.5
            uncertainty: 47.8
            weight: 0.30
          - model_name: hierarchical_v2
            prediction: 275.2
            uncertainty: 52.3
            weight: 0.25
        timestamp: 2025-11-15T14:30:45.123Z
        ensemble_method: inverse_variance_weighted

    HealthyService:
      summary: Service in healthy state
      value:
        status: healthy
        timestamp: 2025-11-15T14:30:45.123Z
        database_connected: true
        redis_connected: true
        models_loaded: 4
        uptime_seconds: 3600.45
        version: 1.0.0

x-metrics:
  - name: predictions_total
    type: counter
    labels: [stat_type, status]
    help: Total number of predictions made
  - name: prediction_latency_seconds
    type: histogram
    buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    help: Time to generate prediction
  - name: models_loaded_total
    type: gauge
    help: Number of models successfully loaded
  - name: api_requests_total
    type: counter
    labels: [endpoint, method, status]
    help: Total API requests
```

---

## Key Schema Elements

### 1. StatType Enum

```python
enum StatType(str, Enum):
    PASSING_YARDS = "passing_yards"      # QB: 0-600 yards
    RUSHING_YARDS = "rushing_yards"      # RB/WR: 0-300 yards
    RECEIVING_YARDS = "receiving_yards"  # WR/TE: 0-350 yards
    PASSING_TDS = "passing_tds"          # QB: 0-8 TDs
    RUSHING_TDS = "rushing_tds"          # RB/WR: 0-6 TDs
```

### 2. Request Body Validation

```python
# Game ID format: YYYY_WW_HOME_AWAY
pattern: '^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$'
# Example: 2024_10_KC_BUF

# Player ID format: 00-XXXXXXX
pattern: '^\d{2}-\d{7}$'
# Example: 00-0033873
```

### 3. Response Structure

```
PredictionResponse:
├── game_id (string)
├── player_id (string)
├── stat_type (enum)
├── prediction (float)         ← Ensemble mean
├── confidence (0-1 float)     ← Model confidence
├── lower_bound (float)        ← Q10 (90% CI lower)
├── upper_bound (float)        ← Q90 (90% CI upper)
├── models_used[] (array)
│  ├── model_name (string)
│  ├── prediction (float)      ← Individual model prediction
│  ├── uncertainty (float)     ← Model standard deviation
│  └── weight (0-1 float)      ← Weight in ensemble
├── timestamp (ISO8601)
└── ensemble_method (string)   ← "inverse_variance_weighted"
```

### 4. Error Status Codes

| Code | Meaning | Cause | Example |
|------|---------|-------|---------|
| 200 | Success | Prediction successful | `prediction: 285.3` |
| 400 | Bad Request | Invalid input | Invalid game_id format |
| 401 | Unauthorized | Invalid API key | Missing X-API-Key header |
| 403 | Forbidden | API key disabled | Account suspended |
| 429 | Too Many Requests | Rate limit exceeded | Exceeded 100 req/min |
| 503 | Service Unavailable | Models not loaded | Startup in progress |
| 500 | Internal Error | Prediction failed | Database connection error |

---

## Prometheus Metrics

### 1. predictions_total (Counter)

```
predictions_total{stat_type="passing_yards",status="success"} 1234.0
predictions_total{stat_type="passing_yards",status="error"} 5.0
predictions_total{stat_type="rushing_yards",status="success"} 456.0
```

**Labels:**
- `stat_type`: passing_yards, rushing_yards, receiving_yards, passing_tds, rushing_tds
- `status`: success, error

### 2. prediction_latency_seconds (Histogram)

```
prediction_latency_seconds_bucket{le="0.01"} 100.0
prediction_latency_seconds_bucket{le="0.05"} 450.0
prediction_latency_seconds_bucket{le="0.1"} 850.0
prediction_latency_seconds_bucket{le="0.5"} 1100.0
prediction_latency_seconds_bucket{le="1.0"} 1220.0
prediction_latency_seconds_bucket{le="2.0"} 1230.0
prediction_latency_seconds_bucket{le="5.0"} 1234.0
prediction_latency_seconds_bucket{le="+Inf"} 1234.0
prediction_latency_seconds_sum 245.67
prediction_latency_seconds_count 1234.0
```

**Buckets:** 10ms, 50ms, 100ms, 500ms, 1s, 2s, 5s

### 3. models_loaded_total (Gauge)

```
models_loaded_total 4.0
```

**Values:** 0-4 (state_space_v2, hierarchical_v2, xgboost_v2, informative_priors_v2)

### 4. api_requests_total (Counter)

```
api_requests_total{endpoint="/health",method="GET",status="success"} 1500.0
api_requests_total{endpoint="/predict",method="POST",status="success"} 1234.0
api_requests_total{endpoint="/predict",method="POST",status="error"} 5.0
api_requests_total{endpoint="/models",method="GET",status="success"} 234.0
api_requests_total{endpoint="/metrics",method="GET",status="success"} 50.0
```

**Labels:**
- `endpoint`: /health, /predict, /models, /metrics, /
- `method`: GET, POST
- `status`: success, error

---

## Generation Commands

### Extract OpenAPI Spec

```bash
# Option 1: Using FastAPI utilities
python -c "
from api.main import app
import json
spec = app.openapi()
with open('api/openapi.json', 'w') as f:
    json.dump(spec, f, indent=2)
print('OpenAPI spec saved to api/openapi.json')
"

# Option 2: Using curl (if server running)
curl http://localhost:8000/openapi.json > api/openapi.json

# Option 3: Using openapi-generator-cli
openapi-generator-cli generate \
  -i api/openapi.json \
  -g openapi-yaml \
  -o api/specs/
```

### Validate OpenAPI Spec

```bash
# Install validator
npm install -g @apidevtools/swagger-cli

# Validate
swagger-cli validate api/openapi.json

# Lint with Spectral
npm install -g @stoplight/spectral-cli
spectral lint api/openapi.json
```

### Generate Client Code

```bash
# Python
openapi-generator-cli generate \
  -i api/openapi.json \
  -g python \
  -o sdk/python

# R
openapi-generator-cli generate \
  -i api/openapi.json \
  -g r \
  -o sdk/r

# Go
openapi-generator-cli generate \
  -i api/openapi.json \
  -g go \
  -o sdk/go
```

---

## References

- **OpenAPI Specification:** https://spec.openapis.org/oas/v3.0.0
- **JSON Schema:** https://json-schema.org/
- **Pydantic Docs:** https://docs.pydantic.dev/latest/
- **FastAPI Docs:** https://fastapi.tiangolo.com/advanced/openapi-schema/
- **Swagger UI:** https://swagger.io/tools/swagger-ui/
- **ReDoc:** https://redoc.ly/

---

**Next Steps:**
1. Extract actual spec from running FastAPI app
2. Compare with this template
3. Validate against OpenAPI 3.0 standard
4. Publish to SwaggerHub or API registry

