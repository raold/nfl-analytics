# API Documentation Index

**Status:** Phase 2 Complete (All Core Documentation)
**Date:** November 15, 2025
**Owner:** Documentation Specialist Agent

---

## Quick Links

### Core Documentation (✅ COMPLETE)

**Phase 2 Part A (Authentication, Errors, Troubleshooting):**
- **[Endpoint Reference](./endpoint_reference.md)** - Complete endpoint documentation (5 endpoints)
- **[Error Catalog](./error_catalog.md)** - Comprehensive error codes and remediation (50+ errors)
- **[Troubleshooting Guide](./troubleshooting_guide.md)** - Integration issues and solutions (20+ scenarios)
- **[Authentication Guide](./authentication_guide.md)** - API key management and security
- **[Rate Limiting Guide](./rate_limiting_guide.md)** - Rate limits and optimization strategies
- **[OpenAPI Specification](./openapi.yaml)** - OpenAPI 3.0 spec file

**Phase 2 Part B (Advanced Features):** ✅ NEW
- **[Advanced Examples](./advanced_examples.md)** - Batch optimization, webhooks, real-time odds, backtesting
- **[Performance Tuning](./performance_tuning.md)** - Latency optimization, connection pooling, caching
- **[Webhooks](./webhooks.md)** - Event-driven integration (future feature, v1.2)
- **[Migration Guide](./migration_guide.md)** - Version migration, breaking changes, rollback plans

### Implementation Plan
- **[OpenAPI Specification Plan](./OPENAPI_SPECIFICATION_PLAN.md)** - Complete implementation roadmap (40-60 hours)
- **[API Audit Findings](./API_AUDIT_FINDINGS.md)** - Current state analysis + gaps

---

## Overview

The NFL Analytics API is a production-grade REST API serving ensemble predictions for NFL player props betting. This directory contains comprehensive documentation for API consumers including endpoint references, error handling, troubleshooting, integration guides, performance optimization, and migration strategies.

### API Status

**Implementation:** ✅ Production-Ready
**Documentation:** ✅ Phase 2 Complete (Part A + Part B)
**Security:** ✅ API Key Authentication + Rate Limiting

---

## Documentation Files

### 1. Endpoint Reference (`endpoint_reference.md`)

**Status:** ✅ COMPLETE (1,362 lines)
**Last Updated:** November 15, 2025

**Contents:**
- Complete documentation for all 5 endpoints
- Request/response schemas with examples
- Authentication requirements
- Rate limiting information
- Error responses (401, 403, 422, 429, 500, 503)
- cURL, Python, R, and JavaScript examples
- Performance characteristics (p95 latency targets)

**Endpoints Documented:**
1. `GET /health` - Health check (no auth required)
2. `GET /models` - List loaded models (auth required)
3. `POST /predict` - Single prediction (auth + rate limit)
4. `POST /predict/batch` - Batch predictions (auth + special limits)
5. `POST /admin/refresh` - Refresh model cache (admin only)

**Use Cases:**
- API integration reference
- Request/response validation
- Client code examples
- Performance benchmarking

---

### 2. Error Catalog (`error_catalog.md`)

**Status:** ✅ COMPLETE (854 lines)
**Last Updated:** November 15, 2025

**Contents:**
- 50+ error codes with full documentation
- HTTP status codes (400, 401, 403, 404, 422, 429, 500, 503)
- Common causes for each error
- Step-by-step remediation
- Example error responses
- Quick lookup table

**Error Categories:**
- **Authentication (AUTH-001 to AUTH-005):** API key issues
- **Rate Limiting (RATE-001 to RATE-003):** Quota management
- **Validation (VAL-001 to VAL-004):** Input validation
- **Server Errors (SRV-001 to SRV-003):** Service issues
- **Client Errors (CLI-001 to CLI-002):** Request formatting

**Use Cases:**
- Error diagnosis
- Debugging failed requests
- Building robust error handlers
- Support escalation

---

### 3. Troubleshooting Guide (`troubleshooting_guide.md`)

**Status:** ✅ COMPLETE (1,138 lines)
**Last Updated:** November 15, 2025

**Contents:**
- 20+ common integration issues with solutions
- Diagnostic procedures
- Performance optimization strategies
- Monitoring and alerting setup
- Error handling patterns
- Production deployment checklist

**Topics Covered:**
1. **Authentication Issues** - Missing/invalid/expired API keys
2. **Rate Limiting** - Handling 429 errors, batch optimization
3. **Response Interpretation** - Understanding uncertainty bounds
4. **Performance** - Connection pooling, caching, parallelization
5. **Error Handling** - Comprehensive exception handling
6. **Monitoring** - Usage tracking and alerting
7. **Common Mistakes** - Anti-patterns to avoid
8. **Debugging Tools** - cURL generators, request loggers

**Use Cases:**
- Resolving integration problems
- Optimizing API performance
- Setting up monitoring
- Pre-launch validation

---

### 4. Authentication Guide (`authentication_guide.md`)

**Status:** ✅ COMPLETE (514 lines)
**Last Updated:** November 15, 2025

**Contents:**
- API key provisioning process
- Header-based authentication
- Key rotation procedures
- Security best practices
- Client examples (Python, R, JavaScript)
- Troubleshooting auth failures

---

### 5. Rate Limiting Guide (`rate_limiting_guide.md`)

**Status:** ✅ COMPLETE (657 lines)
**Last Updated:** November 15, 2025

**Contents:**
- Rate limit policies (hourly/daily)
- Sliding window algorithm
- Checking quota via headers
- Handling 429 errors
- Efficient batching strategies
- Requesting limit increases

---

### 6. OpenAPI Specification (`openapi.yaml`)

**Status:** ✅ COMPLETE (610 lines)
**Last Updated:** November 15, 2025

**Contents:**
- OpenAPI 3.0.3 compliant specification
- Complete schemas for all request/response models
- Authentication security scheme
- Error response definitions
- Interactive documentation support

---

### 7. Advanced Examples (`advanced_examples.md`) ✅ NEW

**Status:** ✅ COMPLETE (612 lines)
**Last Updated:** November 15, 2025

**Contents:**
- **Batch Prediction Optimization** - 10-50x speedup patterns
- **Streaming Responses** - Server-sent events for large batches (future)
- **Webhook Integration** - Async predictions with callbacks
- **Real-Time Odds Integration** - Combine predictions with sportsbook odds
- **Multi-Season Backtesting** - Validate accuracy across multiple seasons
- **Rate Limit Optimization** - Adaptive throttling
- **Parallel Request Handling** - ThreadPool and ProcessPool patterns
- **Production-Grade Integration** - Complete client with all optimizations

**Performance Gains:**
- Batch processing: 10x speedup
- Parallel batches (5 workers): 2-5x additional speedup
- Connection pooling: 1.5x speedup
- Redis caching: 100x for repeated requests

**Use Cases:**
- High-volume backtesting
- Production betting systems
- Real-time odds arbitrage
- Multi-season validation

---

### 8. Performance Tuning Guide (`performance_tuning.md`) ✅ NEW

**Status:** ✅ COMPLETE (583 lines)
**Last Updated:** November 15, 2025

**Contents:**
- **Performance Targets** - Latency benchmarks (p50, p95, p99)
- **Latency Optimization** - Connection pooling, timeouts, HTTP/2 (future)
- **Connection Pooling** - Configuration recommendations
- **Request Batching** - Optimal batch sizes, adaptive batching
- **Caching Strategies** - In-memory LRU, Redis, cache invalidation
- **Parallel Request Handling** - ThreadPool vs ProcessPool
- **Database Query Optimization** - Client-side preparation
- **Monitoring and Profiling** - Request timing, Prometheus metrics

**Performance Summary:**

| Optimization | Speedup | Use Case |
|--------------|---------|----------|
| Batch API (100 items) | 10x | Multiple predictions per game |
| Parallel batches (5 workers) | 2-5x | High rate limit, many predictions |
| Connection pooling | 1.5x | Eliminates connection overhead |
| Redis caching | 100x | Repeated predictions (same inputs) |

**Use Cases:**
- Reducing API latency
- Maximizing throughput
- Implementing caching
- Production monitoring

---

### 9. Webhooks (`webhooks.md`) ✅ NEW

**Status:** ✅ COMPLETE (446 lines) - **Planned Feature (v1.2)**
**Last Updated:** November 15, 2025

**Contents:**
- **Webhook Overview** - Event-driven integration patterns
- **Event Types** - prediction.complete, prediction.failed, model.updated, error.occurred
- **Webhook Registration** - Register, list, update, delete webhooks
- **Payload Schemas** - Complete event payload documentation
- **Security and Verification** - HMAC-SHA256 signature verification
- **Retry Logic** - Exponential backoff (5 retries)
- **Testing Webhooks** - ngrok, mock servers, test tools
- **Best Practices** - Async processing, idempotency, circuit breakers

**Example Use Cases:**
- Async batch processing (1,000+ predictions)
- Model update notifications
- Error monitoring and alerting
- Real-time betting triggers

**Note:** Webhooks are a planned feature for API v1.2 (Q1 2026). This documentation describes the proposed implementation.

---

### 10. Migration Guide (`migration_guide.md`) ✅ NEW

**Status:** ✅ COMPLETE (536 lines)
**Last Updated:** November 15, 2025

**Contents:**
- **Version History** - v1.0, v1.1 (current), v1.2 (planned)
- **v1.0 → v1.1 Migration** - Breaking changes, authentication required
- **Deprecated Features** - EOL dates, replacement endpoints
- **Authentication Migration** - Step-by-step key provisioning
- **Response Format Changes** - Field renames (lower/upper → lower_bound/upper_bound)
- **Rate Limiting Changes** - Default limits, checking status
- **Migration Checklist** - Pre-migration, development, staging, production
- **Rollback Plan** - Quick rollback, hybrid approach, gradual migration

**Breaking Changes (v1.0 → v1.1):**
1. Authentication now required (X-API-Key header)
2. Rate limiting introduced (100 req/hour default)
3. Error response format standardized
4. Response field renames

**Migration Timeline:**
- Release: November 15, 2025
- Migration Window: 4 months
- v1.0 EOL: March 31, 2026

**Use Cases:**
- Planning API upgrades
- Understanding breaking changes
- Emergency rollback procedures
- Version compatibility

---

## API Overview

### Endpoints Summary

| Endpoint | Method | Auth | Rate Limit | Latency (p95) |
|----------|--------|------|------------|---------------|
| `/health` | GET | No | 10,000/hour | <50ms |
| `/models` | GET | Yes | 1,000/hour | <100ms |
| `/predict` | POST | Yes | 100/hour | <500ms |
| `/predict/batch` | POST | Yes | 10/hour | <5000ms |
| `/admin/refresh` | POST | Admin | 10/hour | <30000ms |

### Authentication

All prediction endpoints require API key authentication via `X-API-Key` header:

```bash
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"}'
```

**See:** [Authentication Guide](./authentication_guide.md)

### Rate Limiting

**Default Limits:**
- **Hourly:** 100 requests
- **Daily:** 1,000 requests
- **Sliding window** algorithm (not fixed window)

**Rate limit headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1731678000
```

**See:** [Rate Limiting Guide](./rate_limiting_guide.md)

---

## Quick Start

### 1. Health Check (No Auth)

```bash
curl -X GET "https://api.nfl-analytics.com/health"
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 4,
  "version": "1.1.0"
}
```

### 2. List Models (Auth Required)

```bash
curl -X GET "https://api.nfl-analytics.com/models" \
  -H "X-API-Key: $NFL_API_KEY"
```

### 3. Single Prediction (Auth Required)

```bash
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: $NFL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'
```

**Response:**
```json
{
  "prediction": 285.3,
  "confidence": 0.78,
  "lower_bound": 215.1,
  "upper_bound": 355.5,
  "models_used": [
    {"model_name": "state_space_v2", "prediction": 290.5, "weight": 0.30},
    {"model_name": "hierarchical_v2", "prediction": 275.2, "weight": 0.30},
    {"model_name": "hybrid_xgboost", "prediction": 288.7, "weight": 0.25},
    {"model_name": "informative_priors_v2", "prediction": 282.1, "weight": 0.15}
  ],
  "ensemble_method": "inverse_variance_weighted"
}
```

---

## Client Examples

### Python

```python
import requests
import os

API_KEY = os.getenv("NFL_API_KEY")
BASE_URL = "https://api.nfl-analytics.com"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    headers={"X-API-Key": API_KEY},
    json={
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0033873",
        "stat_type": "passing_yards"
    }
)

prediction = response.json()
print(f"Prediction: {prediction['prediction']:.1f} yards")
print(f"90% CI: [{prediction['lower_bound']:.1f}, {prediction['upper_bound']:.1f}]")
```

**See:** [Endpoint Reference](./endpoint_reference.md) for complete examples

### R

```r
library(httr)

API_KEY <- Sys.getenv("NFL_API_KEY")
BASE_URL <- "https://api.nfl-analytics.com"

response <- POST(
  url = paste0(BASE_URL, "/predict"),
  add_headers("X-API-Key" = API_KEY),
  body = list(
    game_id = "2024_10_KC_LV",
    player_id = "00-0033873",
    stat_type = "passing_yards"
  ),
  encode = "json"
)

prediction <- content(response, as = "parsed")
cat(sprintf("Prediction: %.1f yards\n", prediction$prediction))
```

---

## Error Handling

### Common Errors

**401 Unauthorized:**
```json
{
  "error": "Unauthorized",
  "message": "API key required. Include 'X-API-Key' header in request."
}
```
→ Solution: Add `X-API-Key` header with valid API key

**429 Too Many Requests:**
```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Limit: 100 requests per 1 hour(s)."
}
```
→ Solution: Wait until `X-RateLimit-Reset` timestamp or request higher limit

**422 Validation Error:**
```json
{
  "error": "ValidationError",
  "message": "Invalid stat_type",
  "details": {
    "stat_type": "Must be one of [passing_yards, rushing_yards, receiving_yards, passing_tds, rushing_tds]"
  }
}
```
→ Solution: Fix input validation error

**See:** [Error Catalog](./error_catalog.md) for complete error reference

---

## Performance Optimization

### Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter

session = requests.Session()
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)
session.mount("https://", adapter)

# Reuse session for all requests
response = session.post(url, headers=headers, json=payload)
```

### Caching

```python
import redis
import json

cache = redis.Redis(host='localhost', port=6379)

def get_or_predict(game_id, player_id, stat_type):
    cache_key = f"prediction:{game_id}:{player_id}:{stat_type}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Call API
    prediction = predict(game_id, player_id, stat_type)

    # Cache for 1 hour
    cache.setex(cache_key, 3600, json.dumps(prediction))

    return prediction
```

**See:** [Performance Tuning Guide](./performance_tuning.md) for complete optimization strategies

---

## Production Checklist

### Before Going Live

- [ ] **Authentication**
  - [ ] API keys stored in environment variables
  - [ ] Keys rotated within last 90 days
  - [ ] Separate keys for dev/staging/production

- [ ] **Error Handling**
  - [ ] Exponential backoff for 429 errors
  - [ ] Timeout set on all requests (10s)
  - [ ] Comprehensive try/except blocks

- [ ] **Performance**
  - [ ] Connection pooling enabled
  - [ ] Caching implemented
  - [ ] Batch endpoint used where applicable

- [ ] **Monitoring**
  - [ ] Logging configured
  - [ ] Metrics tracked (requests, errors, latency)
  - [ ] Alerts configured for errors

**See:** [Troubleshooting Guide - Production Checklist](./troubleshooting_guide.md#production-checklist)

---

## Documentation Stats

### Coverage Summary

| Category | Files | Lines | Status |
|----------|-------|-------|--------|
| **Phase 2 Part A** | | | |
| Endpoint Reference | 1 | 1,362 | ✅ Complete |
| Error Catalog | 1 | 854 | ✅ Complete |
| Troubleshooting | 1 | 1,138 | ✅ Complete |
| Authentication | 1 | 514 | ✅ Complete |
| Rate Limiting | 1 | 657 | ✅ Complete |
| OpenAPI Spec | 1 | 610 | ✅ Complete |
| **Phase 2 Part B** | | | |
| Advanced Examples | 1 | 612 | ✅ Complete |
| Performance Tuning | 1 | 583 | ✅ Complete |
| Webhooks | 1 | 446 | ✅ Complete (planned feature) |
| Migration Guide | 1 | 536 | ✅ Complete |
| **Total** | **10** | **7,312** | **✅ Phase 2 Complete** |

### Documentation Quality

| Aspect | Coverage | Notes |
|--------|----------|-------|
| Endpoints | 5/5 (100%) | All endpoints documented |
| Error Codes | 50+ | Comprehensive catalog |
| Examples | 30+ | Python, R, JavaScript, cURL |
| Troubleshooting | 20+ scenarios | Common issues covered |
| Performance | ✅ Complete | Optimization strategies |
| Security | ✅ Complete | Best practices documented |
| Advanced Patterns | ✅ Complete | Batch, parallel, webhooks, migration |

---

## Phase 2 Deliverables Summary

### Part A (32 hours) ✅ COMPLETE
- Endpoint Reference (1,362 lines)
- Error Catalog (854 lines)
- Troubleshooting Guide (1,138 lines)
- Authentication Guide (514 lines)
- Rate Limiting Guide (657 lines)
- OpenAPI Specification (610 lines)

### Part B (24 hours) ✅ COMPLETE
- Advanced Examples (612 lines)
- Performance Tuning (583 lines)
- Webhooks (446 lines - planned feature)
- Migration Guide (536 lines)

**Total:** 7,312 lines of comprehensive API documentation across 10 files

---

## Support

**Documentation:** https://docs.nfl-analytics.com
**API Status:** https://status.nfl-analytics.com
**Support Email:** support@nfl-analytics.com

**Response Times:**
- Critical (security, outage): 1 hour
- High (rate limit, auth): 4 hours
- Normal (questions): 24 hours

---

## Version History

| Date | Version | Status | Changes |
|------|---------|--------|---------|
| 2025-11-15 | 1.0 | ✅ Complete | Initial API audit and documentation plan |
| 2025-11-15 | 2.0 | ✅ Complete | Phase 2 Part A - Authentication, Errors, Troubleshooting |
| 2025-11-15 | 3.0 | ✅ Complete | Phase 2 Part B - Advanced Examples, Performance, Webhooks, Migration |

---

**Last Updated:** November 15, 2025
**Status:** Phase 2 Complete (Part A + Part B delivered)
**Next Review:** After v1.2 release (Q1 2026)
