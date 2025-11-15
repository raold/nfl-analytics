# OpenAPI Specification & API Documentation Plan

**Date Created:** November 15, 2025
**Status:** Plan Ready for Implementation
**Priority:** P0 (Documentation Critical for Production)
**Estimated Effort:** 40-60 hours across all priorities

---

## Executive Summary

The NFL Analytics Prediction API (`api/main.py`) is production-ready but lacks comprehensive external documentation. FastAPI auto-generates OpenAPI 3.0 specs at `/docs` (Swagger UI) and `/redoc` (ReDoc), but we lack:

1. **External OpenAPI spec** (publishable, versioned)
2. **Comprehensive endpoint documentation** with real examples
3. **Client library generators** (Python, R, Go)
4. **Integration guide** for consumers
5. **Rate limiting & authentication docs** (currently missing)
6. **Changelog & versioning policy**

This plan establishes a living documentation strategy aligned with production deployment.

---

## Current State Analysis

### API Implementation Status

**Endpoints Implemented (5 total):**
| Endpoint | Method | Status | Purpose |
|----------|--------|--------|---------|
| `/` | GET | ✅ Operational | Root redirect to docs |
| `/health` | GET | ✅ Operational | Service health check |
| `/predict` | POST | ✅ Operational | Single prediction |
| `/models` | GET | ✅ Operational | List loaded models |
| `/metrics` | GET | ✅ Operational | Prometheus metrics |

**Key Implementation Details:**
- **Framework:** FastAPI 0.104+
- **OpenAPI Version:** 3.0.0 (auto-generated)
- **Response Validation:** Pydantic v2.5+
- **Authentication:** None (currently public, will need API keys for production)
- **Rate Limiting:** Not implemented
- **Caching:** Redis-backed (7-day TTL planned)
- **Monitoring:** Prometheus metrics (5 metrics, labels by stat_type/status/endpoint)

**Current API Quality Score:** 65/100
- ✅ Auto-generated OpenAPI spec
- ✅ Type-safe Pydantic models
- ✅ Proper HTTP status codes
- ✅ Error handling with context
- ⚠️ Limited request/response examples
- ⚠️ No rate limiting documentation
- ❌ No authentication (needed for production betting)
- ❌ No versioning strategy
- ❌ No breaking change policy
- ❌ No client libraries

### Pydantic Models (Comprehensive)

**Request Models:**
```python
- PredictionRequest (game_id, player_id, stat_type)
- StatType enum (5 values: PASSING_YARDS, RUSHING_YARDS, RECEIVING_YARDS, PASSING_TDS, RUSHING_TDS)
```

**Response Models:**
```python
- PredictionResponse (prediction, confidence, bounds, model contributions)
- ModelContribution (model_name, prediction, uncertainty, weight)
- ModelInfo (name, type, version, status, metadata)
- ModelsListResponse (models[], total_models, loaded_models, ensemble_ready)
- HealthResponse (status, timestamp, db/redis/models, uptime)
- ErrorResponse (error, message, details, timestamp)
```

**Quality:** Excellent - all models have field descriptions, examples, validation constraints.

### Documentation Gaps Identified

| Gap | Impact | Priority |
|-----|--------|----------|
| No external OpenAPI spec file (openapi.json) | Can't publish to API portals, no versioning | P0 |
| No request/response examples from real tests | Difficult for consumers to write clients | P1 |
| No authentication documentation | Can't control access or rate limit | P0 |
| No rate limiting policy | Prevents DoS protection, no SLA | P1 |
| No client library stubs | Slow adoption, more support burden | P2 |
| No changelog/versioning | Breaking changes not documented | P1 |
| No performance/SLA docs | Consumers can't plan integrations | P1 |
| No error catalog | Difficult error handling for clients | P1 |
| No migration guide | Will need when v2 released | P2 |

---

## Priority 0: OpenAPI Specification (P0) - 12-16 hours

### Objective
Generate and publish versioned OpenAPI 3.0 spec with all required metadata.

### Deliverables

#### 1. Extract & Export OpenAPI Spec

**File:** `api/openapi.json`

**Generation Script:** `scripts/export_openapi.py`
```bash
python scripts/export_openapi.py --output api/openapi.json
```

**Content:**
- Complete OpenAPI 3.0 spec from FastAPI
- All 5 endpoints with detailed definitions
- All request/response schemas
- 5 Prometheus metrics documented
- Example values for all response types
- Server URLs (dev, staging, prod)

**Spec Structure:**
```yaml
openapi: 3.0.0
info:
  title: NFL Analytics Prediction API
  version: 1.0.0
  description: Production API for ensemble NFL player props predictions
  contact:
    name: API Support
    url: https://github.com/raold/nfl-analytics
servers:
  - url: http://localhost:8000
    description: Development
  - url: https://api-dev.nfl-analytics.local/api/v1
    description: Development (cloud)
  - url: https://api-staging.nfl-analytics.local/api/v1
    description: Staging
  - url: https://api.nfl-analytics.local/api/v1
    description: Production
paths:
  /health:
    get: ...
  /predict:
    post: ...
  /models:
    get: ...
  /metrics:
    get: ...
components:
  schemas: [All Pydantic models]
  responses: [Reusable response definitions]
  examples: [Real-world examples]
```

**Tools:**
- Use `datamodel-json` or manual extraction from FastAPI schema
- Validate with `spectral lint` (OpenAPI linter)

#### 2. Versioned Spec Management

**File Structure:**
```
api/specs/
├── openapi.json          (Current, v1.0.0)
├── openapi.v1.0.0.json   (Archived)
├── openapi.v1.1.0.json   (Archived)
└── SPEC_VERSIONS.md      (Version history)
```

**Version Numbering:**
- `MAJOR.MINOR.PATCH` (semantic versioning)
- `MAJOR`: Breaking changes (new required fields, removed endpoints)
- `MINOR`: Additive changes (new optional fields, new endpoints)
- `PATCH`: Fixes (example corrections, description updates)

**Spec History File:** `SPEC_VERSIONS.md`
```markdown
# OpenAPI Spec Versions

## v1.0.0 (2025-11-15) - Initial Release
- 5 endpoints (health, predict, models, metrics, root)
- 4-way ensemble support
- Prometheus metrics export

## v1.1.0 (2025-11-20) - Rate Limiting
- Added rate limiting headers
- New RateLimitError response

## v2.0.0 (2025-12-01) - Breaking Changes
- Moved to /api/v2 prefix
- Changed prediction response format
```

#### 3. Interactive API Documentation

**Locations:**
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

**Enhancements:**
- Add custom logo/branding to Swagger UI
- Add server selector dropdown (dev/staging/prod)
- Try-it-out examples for each endpoint
- Schema examples from real backtest results

**Implementation:**
```python
# In api/main.py
app = FastAPI(
    title="NFL Analytics Prediction API",
    description="...",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_init_oauth={},  # OAuth2 integration (future)
)
```

### Implementation Steps

1. **Extract OpenAPI spec** from running FastAPI app
   ```bash
   python scripts/export_openapi.py --format json --output api/openapi.json
   ```

2. **Validate spec** against OpenAPI 3.0 standard
   ```bash
   spectral lint api/openapi.json --ruleset https://unpkg.com/@stoplight/spectral-rulesets
   ```

3. **Publish to API portals:**
   - SwaggerHub (free tier: https://app.swaggerhub.com/organizations/yourorg)
   - Postman API Registry
   - GitHub Pages (for internal docs)

4. **Version control:**
   ```bash
   git add api/specs/openapi.v1.0.0.json
   git add api/SPEC_VERSIONS.md
   ```

### Acceptance Criteria
- ✅ `api/openapi.json` generated and committed
- ✅ All 5 endpoints documented in spec
- ✅ All Pydantic models as OpenAPI schemas
- ✅ Spec validates against OpenAPI 3.0.0 standard
- ✅ Server URLs include dev/staging/prod
- ✅ Version metadata (1.0.0) in info section

---

## Priority 1: Endpoint Documentation (P1) - 16-20 hours

### Objective
Create comprehensive documentation for each endpoint with real examples.

### Deliverables

#### 1. Endpoint Documentation File

**File:** `docs/api/ENDPOINT_REFERENCE.md` (2,500+ lines)

**Structure per endpoint:**
```markdown
## POST /predict

**Description:** Generate ensemble prediction for single player stat.

### Request

**Method:** POST
**Content-Type:** application/json

**Path Parameters:** None

**Query Parameters:** None

**Request Body:**

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| game_id | string | ✅ | Game identifier (YYYY_WW_HOME_AWAY) | "2024_10_KC_LV" |
| player_id | string | ✅ | NFL player ID (00-XXXXXXX) | "00-0033873" |
| stat_type | enum | ✅ | Stat to predict | "passing_yards" |

**Enum Values (stat_type):**
```json
[
  "passing_yards",      // QB passing yards (0-600)
  "rushing_yards",      // RB/WR rushing yards (0-300)
  "receiving_yards",    // WR/TE receiving yards (0-350)
  "passing_tds",        // QB passing TDs (0-8)
  "rushing_tds"         // RB/WR rushing TDs (0-6)
]
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_BUF",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'
```

### Response

**HTTP Status:** 200 OK

**Response Body:**

| Field | Type | Description |
|-------|------|-------------|
| game_id | string | Echoed from request |
| player_id | string | Echoed from request |
| stat_type | string | Echoed from request |
| prediction | float | Ensemble mean prediction (yards) |
| confidence | float | Model confidence (0.0-1.0, higher = more confident) |
| lower_bound | float | 90% credible interval lower (Q10) |
| upper_bound | float | 90% credible interval upper (Q90) |
| models_used | array | Individual model contributions |
| timestamp | datetime | Prediction time (UTC ISO8601) |
| ensemble_method | string | "inverse_variance_weighted" |

**Model Contribution Object:**
```json
{
  "model_name": "state_space_v2",
  "prediction": 290.5,
  "uncertainty": 47.8,
  "weight": 0.30
}
```

**Example Response (Success):**
```json
{
  "game_id": "2024_10_KC_BUF",
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
      "weight": 0.25
    },
    {
      "model_name": "xgboost_v4",
      "prediction": 292.1,
      "uncertainty": 44.5,
      "weight": 0.20
    },
    {
      "model_name": "informative_priors_v2",
      "prediction": 280.8,
      "uncertainty": 56.2,
      "weight": 0.25
    }
  ],
  "timestamp": "2025-11-15T14:30:45.123Z",
  "ensemble_method": "inverse_variance_weighted"
}
```

### Error Responses

| Status | Error | Scenario |
|--------|-------|----------|
| 400 | ValidationError | Invalid game_id format or missing required field |
| 503 | ServiceUnavailable | Models not loaded or insufficient models (need 2+) |
| 500 | InternalServerError | Ensemble prediction failed (data not found, computation error) |

**Example Error Response (400 - Bad Request):**
```json
{
  "error": "ValidationError",
  "message": "1 validation error for PredictionRequest",
  "details": {
    "loc": ["body", "game_id"],
    "msg": "value error, invalid game_id format (expected YYYY_WW_HOME_AWAY)",
    "type": "value_error"
  },
  "timestamp": "2025-11-15T14:30:45.123Z"
}
```

**Example Error Response (503 - Service Unavailable):**
```json
{
  "error": "HTTPException",
  "message": "Insufficient models loaded (need at least 2)",
  "details": {
    "status_code": 503
  },
  "timestamp": "2025-11-15T14:30:45.123Z"
}
```

### Performance Notes

- **Latency:** 50-200ms p50, 500ms p95 (cached), 1-2s p95 (uncached)
- **Throughput:** 1,000+ predictions/second on single instance
- **Memory:** ~2GB for 4-way ensemble (state-space + hierarchical + XGBoost + IP)
- **Database:** ~10ms for feature lookups (cached)

### Real-World Examples

**Example 1: Patrick Mahomes Passing Yards (High Confidence)**
```json
Request: {
  "game_id": "2024_10_KC_LV",
  "player_id": "00-0033873",
  "stat_type": "passing_yards"
}

Response: {
  "prediction": 285.3,
  "confidence": 0.78,
  "lower_bound": 215.1,
  "upper_bound": 355.5
}
```

**Example 2: Backup QB (Low Confidence)**
```json
Request: {
  "game_id": "2024_10_KC_LV",
  "player_id": "00-0099999",
  "stat_type": "passing_yards"
}

Response: {
  "prediction": null,
  "confidence": 0.0,
  "lower_bound": null,
  "upper_bound": null,
  "models_used": []
}
```

### Consuming This Endpoint

See [API Client Examples](#client-examples) for Python, R, Go implementations.
```

**Repeat for:**
- `/health` (health check status)
- `/models` (list loaded models)
- `/metrics` (Prometheus metrics)

#### 2. API Reference Quick Card

**File:** `docs/api/API_QUICK_REFERENCE.md` (500 lines)

Single-page reference card for developers:

```markdown
# API Quick Reference Card

## Endpoints

### GET /health - Health Check
```bash
curl http://localhost:8000/health
```
Response: `{status, timestamp, db_connected, redis_connected, models_loaded, uptime_seconds}`

### POST /predict - Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{game_id: "2024_10_KC_BUF", player_id: "00-0033873", stat_type: "passing_yards"}'
```
Response: `{prediction, confidence, bounds, models_used[]}`

### GET /models - List Models
```bash
curl http://localhost:8000/models
```
Response: `{models[], total_models, loaded_models, ensemble_ready}`

### GET /metrics - Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```
Response: Prometheus format text

## Stat Types
- passing_yards (0-600)
- rushing_yards (0-300)
- receiving_yards (0-350)
- passing_tds (0-8)
- rushing_tds (0-6)

## HTTP Status Codes
- 200: Success
- 400: Bad request (validation error)
- 503: Service unavailable (models not loaded)
- 500: Internal error
```

### Implementation Steps

1. **Generate examples from test cases**
   ```bash
   pytest tests/integration/test_api.py --export-examples --format json
   # Output: tests/integration/api_examples.json
   ```

2. **Extract real backtest results**
   ```python
   # From historical backtest runs
   import json
   from py.backtest.historical_validation_4way import BacktestResults

   results = BacktestResults.load('results/backtest_2024.pkl')
   real_examples = results.sample_predictions(n=10)
   json.dump(real_examples, open('docs/api/real_world_examples.json', 'w'))
   ```

3. **Document error cases**
   - Invalid game_id format
   - Missing player in database
   - Insufficient models loaded
   - Database connection failure
   - Redis cache miss (should still work)

4. **Performance benchmarks**
   ```bash
   python scripts/benchmark_api.py --duration 5 --concurrency 100
   # Output: api_performance_benchmark.json
   ```

### Acceptance Criteria
- ✅ All 5 endpoints documented with full details
- ✅ Request/response examples (both success and error)
- ✅ Real examples from backtest results
- ✅ Stat type enum values and valid ranges
- ✅ Error codes and troubleshooting
- ✅ Performance/latency expectations
- ✅ Database query patterns documented

---

## Priority 2: Authentication & Rate Limiting (P0 High) - 12-16 hours

### Objective
Implement and document authentication/rate limiting for production API.

**This is P0 HIGH for production deployment** (currently public = security risk).

### Deliverables

#### 1. API Key Authentication

**File:** `api/auth.py` (new, 200 lines)

**Implementation:**
```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key from request header"""
    # Validate against database or environment
    if not validate_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key
```

**Update `api/main.py`:**
```python
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(request: PredictionRequest):
    # Protected endpoint
    ...
```

**Key Management:**
- Store hashed API keys in database (no plaintext)
- Rate limit per key
- Automatic expiration
- Usage tracking

#### 2. Rate Limiting

**File:** `api/rate_limiting.py` (new, 150 lines)

**Implementation:** Redis-based token bucket
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")  # 100 predictions per minute
async def predict(request: PredictionRequest):
    ...
```

**Rate Limits (tiered):**
| Tier | Requests/Min | Requests/Hour | Requests/Day | Price |
|------|--------------|---------------|--------------|-------|
| Free | 10 | 100 | 1,000 | $0 |
| Standard | 100 | 5,000 | 100,000 | $99/mo |
| Professional | 1,000 | 50,000 | 1,000,000 | $999/mo |
| Enterprise | Unlimited | Custom | Custom | Contact |

**Rate Limit Headers (RFC 6585):**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1731688260
Retry-After: 30
```

#### 3. Authentication Documentation

**File:** `docs/api/AUTHENTICATION_GUIDE.md` (800 lines)

```markdown
# API Authentication & Rate Limiting Guide

## Getting API Keys

1. Register account: https://api.nfl-analytics.local/register
2. Generate key: https://api.nfl-analytics.local/console
3. Copy key (shown once): sk_live_xxxxxxxxxxxxxx

## Using API Keys

Include in every request header:
```bash
curl -H "X-API-Key: sk_live_xxxxxxxxxxxxxx" \
  -X POST http://localhost:8000/predict \
  -d '{...}'
```

## Rate Limits

Your account tier: Standard (100/min)

**Current Usage:**
- This minute: 45/100
- This hour: 2,450/5,000
- This day: 23,500/100,000

## Handling Rate Limits

When you exceed limit:
```http
HTTP 429 Too Many Requests
X-RateLimit-Reset: 1731688260
Retry-After: 30
```

Exponential backoff:
```python
import time
import requests

for attempt in range(5):
    response = requests.post(...)
    if response.status_code == 429:
        wait_seconds = 2 ** attempt
        time.sleep(wait_seconds)
    else:
        break
```
```

### Implementation Steps

1. **Create API key management system**
   - Database table: `api_keys(id, user_id, key_hash, tier, created_at, expires_at)`
   - CLI: `python scripts/manage_api_keys.py --generate --user=test_user`

2. **Implement rate limiting middleware**
   ```bash
   pip install slowapi
   ```

3. **Add authentication to endpoints**
   - `/predict` - Protected (requires key)
   - `/health` - Public (no auth required)
   - `/metrics` - Public but restricted to internal IPs
   - `/models` - Protected (requires key)

4. **Document in OpenAPI spec**
   ```yaml
   components:
     securitySchemes:
       api_key:
         type: apiKey
         in: header
         name: X-API-Key
   security:
     - api_key: []
   ```

### Acceptance Criteria
- ✅ API key generation and validation
- ✅ Rate limiting per tier (10/100/1000/unlimited)
- ✅ Rate limit headers in responses
- ✅ 429 responses with Retry-After
- ✅ Authentication in OpenAPI spec
- ✅ Documentation with examples

---

## Priority 2: Error Catalog & Troubleshooting (P1) - 8-12 hours

### Objective
Document all error codes and troubleshooting steps.

### Deliverables

#### Error Reference Table

**File:** `docs/api/ERROR_REFERENCE.md`

| Code | Status | Error Type | Cause | Solution |
|------|--------|-----------|-------|----------|
| 400 | Bad Request | ValidationError | Missing/invalid game_id | Use format YYYY_WW_HOME_AWAY (e.g., 2024_10_KC_BUF) |
| 400 | Bad Request | ValidationError | Invalid player_id | Use format 00-XXXXXXX (8 digits) |
| 400 | Bad Request | ValidationError | Unknown stat_type | Use: passing_yards, rushing_yards, receiving_yards, passing_tds, rushing_tds |
| 503 | Service Unavailable | ModelLoadError | Models not loaded | Wait 30s for startup, check logs |
| 503 | Service Unavailable | InsufficientModels | <2 models loaded | Check ensemble health: GET /health |
| 500 | Internal Error | PredictionFailed | Data not in database | Ensure game/player exists in database |
| 500 | Internal Error | DatabaseError | Connection failed | Check PostgreSQL at 5544 |
| 500 | Internal Error | RegressionError | NaN prediction | Check for missing features |

#### Troubleshooting Guide

**File:** `docs/api/TROUBLESHOOTING.md`

**Issues covered:**
1. "Insufficient models loaded (need at least 2)" → Check `/models` endpoint, wait for startup
2. "Prediction returned null" → Check if game/player in database
3. "API returns 500" → Check logs: `docker logs nfl-analytics-api`
4. "Latency > 2 seconds" → Check Redis connectivity and cache hit rate
5. "Rate limited (429)" → Wait for reset or upgrade tier

### Acceptance Criteria
- ✅ All error codes documented
- ✅ Common issues and solutions
- ✅ Example error responses
- ✅ Troubleshooting flowchart

---

## Priority 3: Client Libraries & SDKs (P2) - 20-30 hours

### Objective
Generate client libraries for Python, R, and Go.

### Deliverables

#### 1. Python SDK

**File:** `sdk/python/nfl_analytics_api/` (new package)

**Generated from:** OpenAPI spec using OpenAPI Generator

```bash
openapi-generator-cli generate \
  -i api/openapi.json \
  -g python \
  -o sdk/python \
  --package-name nfl_analytics_api
```

**Package structure:**
```
sdk/python/
├── nfl_analytics_api/
│   ├── __init__.py
│   ├── api_client.py
│   ├── prediction_api.py
│   ├── models_api.py
│   ├── health_api.py
│   └── models/
│       ├── prediction_request.py
│       ├── prediction_response.py
│       ├── model_info.py
│       └── ...
├── setup.py
├── requirements.txt
└── examples/
    ├── predict_single.py
    ├── predict_batch.py
    └── list_models.py
```

**Usage example:**
```python
from nfl_analytics_api import ApiClient, PredictionApi, PredictionRequest, StatType

# Initialize client
client = ApiClient(api_key="sk_live_xxxxx")
api = PredictionApi(client)

# Single prediction
request = PredictionRequest(
    game_id="2024_10_KC_BUF",
    player_id="00-0033873",
    stat_type=StatType.PASSING_YARDS
)
response = api.predict(request)
print(f"Prediction: {response.prediction:.1f} yards")
print(f"Confidence: {response.confidence:.0%}")
print(f"90% CI: [{response.lower_bound:.1f}, {response.upper_bound:.1f}]")
```

**Installation:**
```bash
pip install nfl-analytics-api
```

#### 2. R SDK

**File:** `sdk/r/nflAnalyticsApi/` (new package)

**Generated using:** OpenAPI Generator

```bash
openapi-generator-cli generate \
  -i api/openapi.json \
  -g r \
  -o sdk/r
```

**Usage example:**
```r
library(nflAnalyticsApi)

# Initialize client
client <- ApiClient$new(api_key = "sk_live_xxxxx")
api <- PredictionApi$new(client)

# Single prediction
request <- PredictionRequest$new(
  game_id = "2024_10_KC_BUF",
  player_id = "00-0033873",
  stat_type = "passing_yards"
)
response <- api$predict(request)
cat(sprintf("Prediction: %.1f yards\n", response$prediction))
cat(sprintf("90%% CI: [%.1f, %.1f]\n", response$lower_bound, response$upper_bound))
```

**Installation:**
```r
# From GitHub
devtools::install_github("raold/nfl-analytics/sdk/r")

# Or from CRAN (future)
install.packages("nflAnalyticsApi")
```

#### 3. Go SDK

**File:** `sdk/go/` (new module)

**Generated using:** OpenAPI Generator

```bash
openapi-generator-cli generate \
  -i api/openapi.json \
  -g go \
  -o sdk/go
```

**Usage example:**
```go
package main

import (
    "fmt"
    "github.com/raold/nfl-analytics/sdk/go"
)

func main() {
    // Initialize client
    client := openapi.NewAPIClient(
        openapi.NewConfiguration(),
    )
    client.GetConfig().AddDefaultHeader("X-API-Key", "sk_live_xxxxx")

    // Single prediction
    req := *openapi.NewPredictionRequest(
        "2024_10_KC_BUF",
        "00-0033873",
        openapi.PASSING_YARDS,
    )

    response, _, err := client.PredictionApi.Predict(context.Background()).
        PredictionRequest(req).
        Execute()

    if err != nil {
        panic(err)
    }

    fmt.Printf("Prediction: %.1f yards\n", response.Prediction)
    fmt.Printf("90%% CI: [%.1f, %.1f]\n", response.LowerBound, response.UpperBound)
}
```

### Implementation Steps

1. **Install OpenAPI Generator**
   ```bash
   npm install @openapitools/openapi-generator-cli
   ```

2. **Generate Python SDK**
   ```bash
   openapi-generator-cli generate \
     -i api/openapi.json \
     -g python \
     -o sdk/python \
     --package-name nfl_analytics_api \
     --additional-properties=packageVersion=1.0.0
   ```

3. **Generate R SDK**
   ```bash
   openapi-generator-cli generate \
     -i api/openapi.json \
     -g r \
     -o sdk/r
   ```

4. **Generate Go SDK**
   ```bash
   openapi-generator-cli generate \
     -i api/openapi.json \
     -g go \
     -o sdk/go
   ```

5. **Test SDKs with real API**
   ```bash
   cd sdk/python && python -m pytest tests/
   cd sdk/r && R CMD check .
   cd sdk/go && go test ./...
   ```

6. **Publish to package registries**
   - PyPI: `python setup.py upload`
   - CRAN: Submit to https://cran.r-project.org
   - Go Modules: Auto-discovered from GitHub

### Acceptance Criteria
- ✅ Python SDK generated and tested
- ✅ R SDK generated and tested
- ✅ Go SDK generated and tested
- ✅ All 5 endpoints accessible via SDK
- ✅ Request/response models match OpenAPI spec
- ✅ Auth/rate limiting integrated
- ✅ Published to registries (PyPI, CRAN, Go Modules)
- ✅ Usage examples in each SDK

---

## Priority 3: Integration Guide (P2) - 12-16 hours

### Objective
Help consumers integrate the API into betting systems.

### Deliverables

#### 1. Integration Architecture Guide

**File:** `docs/api/INTEGRATION_GUIDE.md` (2,000 lines)

**Sections:**
1. **Consuming Predictions** - How to call API and use predictions
2. **Risk Management** - Using confidence intervals for bet sizing
3. **Caching Strategy** - When to cache vs. call API
4. **Batch Processing** - Future endpoint for bulk predictions
5. **Error Handling** - Graceful degradation on API errors
6. **Monitoring** - Health checks and alerting
7. **Performance** - Latency expectations, optimization

**Example: Bet Sizing Using Confidence Intervals**
```python
import requests
from scipy.stats import norm

def calculate_bet_size(prediction, confidence, true_probability, bankroll, kelly=0.25):
    """
    Kelly criterion bet sizing using API prediction and confidence.

    Args:
        prediction: API prediction (yards)
        confidence: Model confidence (0-1)
        true_probability: Actual probability of event
        bankroll: Current betting bankroll
        kelly: Kelly fraction (0.25 = 1/4 Kelly for safety)

    Returns:
        Bet size as fraction of bankroll
    """
    # Adjust probability by confidence
    # Low confidence → closer to 50% (no edge)
    # High confidence → trust the model more
    adjusted_prob = 0.5 + (true_probability - 0.5) * confidence

    # Kelly criterion
    edge = adjusted_prob - 0.5
    bet_size = kelly * edge

    return bet_size

# Example usage
response = requests.post('http://localhost:8000/predict', json={
    'game_id': '2024_10_KC_BUF',
    'player_id': '00-0033873',
    'stat_type': 'passing_yards'
})

pred = response['prediction']
conf = response['confidence']
bounds = (response['lower_bound'], response['upper_bound'])

# Assume market odds imply 55% probability at over/under 280
true_prob = 0.55

bet_size = calculate_bet_size(pred, conf, true_prob, bankroll=10000)
print(f"Predicted: {pred:.1f} yards (90% CI: {bounds})")
print(f"Confidence: {conf:.0%}")
print(f"Recommended bet: {bet_size:.0%} of bankroll = ${bet_size * 10000:.2f}")
```

#### 2. Example Implementations

**Files:** `docs/api/examples/` (5 complete examples)

**Example 1:** `simple_prediction.py` - Single prediction
**Example 2:** `batch_prediction.py` - Multiple predictions in parallel
**Example 3:** `real_time_odds_integration.py` - Fetch odds, compare to prediction
**Example 4:** `backtesting_integration.py` - Use API in backtest
**Example 5:** `dashboard_integration.py` - Display predictions in dashboard

### Acceptance Criteria
- ✅ Architecture diagram (system integration)
- ✅ 5+ complete, runnable examples
- ✅ Risk management guide (bet sizing, Kelly criterion)
- ✅ Error handling patterns
- ✅ Monitoring/alerting setup
- ✅ Performance tuning guide

---

## Priority 3: Changelog & Versioning Policy (P2) - 6-8 hours

### Objective
Establish versioning strategy and document breaking changes.

### Deliverables

#### 1. API Versioning Policy

**File:** `docs/api/VERSIONING_POLICY.md`

**Semantic Versioning:** MAJOR.MINOR.PATCH

**Examples:**
- **v1.0.0** → Initial release
- **v1.1.0** → New endpoint (backward compatible)
- **v1.1.1** → Bug fix (backward compatible)
- **v2.0.0** → Breaking change (old API deprecated)

**Deprecation Policy:**
- Breaking changes require 6-month notice
- Old version supported for 1 year minimum
- Gradual migration path provided

**Change Categories:**

| Type | Semver | Example | Backward Compatible |
|------|--------|---------|---------------------|
| Bug Fix | PATCH | Fix NaN in confidence calculation | ✅ Yes |
| New Field (optional) | MINOR | Add `model_versions` to response | ✅ Yes |
| New Field (required) | MAJOR | Add required `timestamp` field | ❌ No |
| New Endpoint | MINOR | Add `/predict/batch` endpoint | ✅ Yes |
| Remove Endpoint | MAJOR | Remove `/metrics` endpoint | ❌ No |
| Change Field Type | MAJOR | Change prediction from float to int | ❌ No |

#### 2. Changelog Format

**File:** `docs/api/CHANGELOG.md`

**Format:** Keep a Changelog (https://keepachangelog.com/)

```markdown
# Changelog - NFL Analytics Prediction API

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2025-12-15 (Planned)

### Added
- [BREAKING] New `/api/v2/predict` endpoint with improved response format
- Batch prediction endpoint `/api/v2/predict/batch`
- Confidence score refinements (now 0.0-1.0 scale)
- Model version tracking in response

### Changed
- [BREAKING] `/predict` moved to `/api/v2/predict` (v1 sunset Dec 31, 2025)
- Response format: `model_predictions` renamed to `models_used`
- Ensemble weights now dynamic per season

### Deprecated
- `/api/v1/predict` (EOL: 2025-12-31)
- Static ensemble weights (use `/models` to get current weights)

### Fixed
- Confidence interval edge case when ensemble_std = 0

## [1.1.0] - 2025-11-20

### Added
- Rate limiting documentation with tier information
- API key authentication
- New `X-RateLimit-*` response headers

### Changed
- Health check now returns model versions

### Fixed
- NaN in confidence calculation when uncertainty = 0

## [1.0.0] - 2025-11-15

### Added
- Initial release
- 5 endpoints (health, predict, models, metrics, root)
- 4-way ensemble support
- Prometheus metrics export
```

### Acceptance Criteria
- ✅ Semantic versioning policy documented
- ✅ Deprecation timeline (6-month notice)
- ✅ CHANGELOG.md with all v1.0.0 changes
- ✅ Migration guide for v1 → v2
- ✅ Backward compatibility matrix

---

## Implementation Timeline & Effort Estimates

### Phase 1: P0 Priorities (Weeks 1-2) - 28-36 hours

| Priority | Task | Hours | Owner | Week |
|----------|------|-------|-------|------|
| P0 | Extract OpenAPI spec | 4 | Doc Agent | Week 1 |
| P0 | Validate against standard | 2 | Doc Agent | Week 1 |
| P0 | Publish to API portals | 3 | Doc Agent | Week 1 |
| P0 | Implement API key auth | 8 | API Agent | Week 1 |
| P0 | Implement rate limiting | 6 | API Agent | Week 1 |
| P0 | Document authentication | 4 | Doc Agent | Week 2 |
| P0 | Document rate limiting | 4 | Doc Agent | Week 2 |
| **Subtotal P0** | | **31-35 hours** | | |

### Phase 2: P1 Priorities (Weeks 3-4) - 32-44 hours

| Priority | Task | Hours | Owner | Week |
|----------|------|-------|-------|------|
| P1 | Document all 5 endpoints | 12 | Doc Agent | Week 3 |
| P1 | Generate error catalog | 6 | Doc Agent | Week 3 |
| P1 | Create troubleshooting guide | 4 | Doc Agent | Week 3 |
| P1 | Extract real examples | 4 | Doc Agent | Week 4 |
| P1 | Write quick reference card | 3 | Doc Agent | Week 4 |
| P1 | Changelog & versioning | 6 | Doc Agent | Week 4 |
| **Subtotal P1** | | **35-40 hours** | | |

### Phase 3: P2 Priorities (Weeks 5-6) - 40-60 hours

| Priority | Task | Hours | Owner | Week |
|----------|------|-------|-------|------|
| P2 | Generate Python SDK | 6 | Dev Agent | Week 5 |
| P2 | Generate R SDK | 6 | Dev Agent | Week 5 |
| P2 | Generate Go SDK | 6 | Dev Agent | Week 5 |
| P2 | Test all SDKs | 8 | Dev Agent | Week 5 |
| P2 | Publish to registries | 4 | Dev Agent | Week 6 |
| P2 | Integration guide | 10 | Doc Agent | Week 6 |
| P2 | Example implementations | 12 | Doc Agent | Week 6 |
| **Subtotal P2** | | **52-70 hours** | | |

### **Total Effort Estimate: 118-149 hours across 6 weeks**

**Breakdown by role:**
- **Documentation Agent:** 65-80 hours (endpoint docs, examples, guides, troubleshooting)
- **API Agent:** 14-18 hours (authentication, rate limiting)
- **Developer Agent:** 32-48 hours (SDKs, testing, publishing)

---

## Success Criteria & Metrics

### Documentation Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| OpenAPI spec completeness | 100% | All endpoints + schemas documented |
| Endpoint documentation | 100% | All 5 endpoints with examples |
| Error coverage | 90%+ | All error codes documented |
| Example quality | 100% | Real examples from actual backtest |
| Code examples work | 100% | All SDK examples tested and passing |

### Adoption Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API client download | 100+ | PyPI, CRAN, GitHub |
| Swagger UI usage | Daily | Analytics on /docs hits |
| Integration partners | 3+ | Teams using API for production bets |
| Support tickets | <5/week | From API consumers |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API latency (p50) | <100ms | Monitoring dashboard |
| API latency (p95) | <500ms | Monitoring dashboard |
| Uptime | 99.9% | SLA compliance |
| Error rate | <0.1% | Prometheus metrics |

---

## Dependencies & Prerequisites

### Tools Required
- OpenAPI Generator CLI
- Spectral (OpenAPI linter)
- SwaggerHub or Postman API
- Git + GitHub
- Docker (for API testing)

### Knowledge Required
- OpenAPI 3.0 specification
- RESTful API design
- FastAPI framework
- Rate limiting patterns
- Authentication schemes

### Access Required
- FastAPI app (`api/main.py`)
- Test database with sample data
- Production API instance (for testing)

---

## Recommended Execution Order

1. **Week 1 (P0):** Extract OpenAPI spec + implement authentication
2. **Week 2 (P0):** Document auth/rate limiting + validate spec
3. **Week 3 (P1):** Write endpoint documentation + error catalog
4. **Week 4 (P1):** Create examples + quick reference
5. **Week 5 (P2):** Generate SDKs (Python, R, Go)
6. **Week 6 (P2):** Write integration guide + publish

---

## File Checklist

### To Be Created

- [ ] `api/openapi.json` - OpenAPI spec (auto-generated)
- [ ] `api/specs/SPEC_VERSIONS.md` - Version history
- [ ] `api/auth.py` - API key authentication
- [ ] `api/rate_limiting.py` - Rate limiting middleware
- [ ] `scripts/export_openapi.py` - OpenAPI export script
- [ ] `scripts/benchmark_api.py` - Performance testing
- [ ] `docs/api/ENDPOINT_REFERENCE.md` - Full endpoint docs
- [ ] `docs/api/API_QUICK_REFERENCE.md` - Quick card
- [ ] `docs/api/AUTHENTICATION_GUIDE.md` - Auth documentation
- [ ] `docs/api/ERROR_REFERENCE.md` - Error catalog
- [ ] `docs/api/TROUBLESHOOTING.md` - Troubleshooting guide
- [ ] `docs/api/INTEGRATION_GUIDE.md` - Integration guide
- [ ] `docs/api/VERSIONING_POLICY.md` - Versioning strategy
- [ ] `docs/api/CHANGELOG.md` - Changelog
- [ ] `docs/api/examples/` - Example implementations (5+)
- [ ] `sdk/python/nfl_analytics_api/` - Python SDK (auto-generated)
- [ ] `sdk/r/nflAnalyticsApi/` - R SDK (auto-generated)
- [ ] `sdk/go/` - Go SDK (auto-generated)

### To Be Modified

- [ ] `api/main.py` - Add auth/rate limiting decorators
- [ ] `CLAUDE.md` - Link to API documentation

---

## Conclusion

This plan provides a comprehensive roadmap for API documentation from production-ready to fully documented, discoverable, and easy-to-consume. Implementation across P0, P1, P2 priorities ensures:

1. **Production Security** (P0): Authentication and rate limiting
2. **Developer Experience** (P1): Clear documentation and examples
3. **Ecosystem Growth** (P2): Client libraries and integration partners

Total effort: **118-149 hours** over **6 weeks**

---

**Next Steps:**
1. Assign Documentation Agent to P0 tasks (Week 1)
2. Assign API Agent to authentication implementation
3. Schedule review meetings at end of each phase
4. Track progress via this document (update weekly)
