# API Error Catalog

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

Comprehensive catalog of all API error codes, messages, causes, and remediation steps. Use this guide to diagnose and resolve API errors quickly.

## Table of Contents

1. [Error Response Format](#error-response-format)
2. [Authentication Errors (401, 403)](#authentication-errors-401-403)
3. [Rate Limiting Errors (429)](#rate-limiting-errors-429)
4. [Validation Errors (422)](#validation-errors-422)
5. [Server Errors (500, 503)](#server-errors-500-503)
6. [Client Errors (400, 404)](#client-errors-400-404)
7. [Error Code Reference](#error-code-reference)

---

## Error Response Format

All API errors follow a consistent JSON structure:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional context"
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Fields:**
- `error` (string) - Error type/code (e.g., "Unauthorized", "ValidationError")
- `message` (string) - Human-readable description
- `details` (object, optional) - Additional error context
- `timestamp` (string) - ISO 8601 timestamp when error occurred

---

## Authentication Errors (401, 403)

### AUTH-001: Missing API Key

**HTTP Status:** 401 Unauthorized

**Error Response:**
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

**Common Causes:**
1. Missing `X-API-Key` header entirely
2. Empty API key value
3. Typo in header name (e.g., "API-Key" instead of "X-API-Key")

**Remediation:**
```bash
# ❌ WRONG - Missing header
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_10_KC_LV", ...}'

# ✅ CORRECT - Include X-API-Key header
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_10_KC_LV", ...}'
```

**Python Fix:**
```python
# ❌ WRONG
response = requests.post(url, json=payload)

# ✅ CORRECT
headers = {"X-API-Key": API_KEY}
response = requests.post(url, headers=headers, json=payload)
```

---

### AUTH-002: Invalid API Key

**HTTP Status:** 401 Unauthorized

**Error Response:**
```json
{
  "error": "Unauthorized",
  "message": "Invalid API key",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. API key doesn't exist in database
2. Typo in API key (wrong character, truncated, extra spaces)
3. Using development key in production environment
4. Key was deleted from database

**Remediation:**
1. **Verify API key is correct:**
   - Check all 64 characters are present
   - Remove leading/trailing whitespace
   - Ensure no line breaks in key

2. **Check environment:**
   - Development keys only work on `http://localhost:8000`
   - Production keys only work on `https://api.nfl-analytics.com`

3. **Test with known good key:**
   ```bash
   # Test key validity
   curl -X GET "https://api.nfl-analytics.com/models" \
     -H "X-API-Key: $NFL_API_KEY"
   ```

4. **Request new key if invalid:**
   - Email: support@nfl-analytics.com
   - Include: Original key prefix (first 8 characters)

---

### AUTH-003: Inactive or Revoked Key

**HTTP Status:** 403 Forbidden

**Error Response:**
```json
{
  "error": "Forbidden",
  "message": "API key is inactive or revoked. Contact administrator.",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Key manually deactivated by administrator
2. Key revoked due to security incident
3. Account suspended due to terms of service violation
4. Key marked inactive pending investigation

**Remediation:**
1. **Contact support immediately:**
   - Email: support@nfl-analytics.com
   - Subject: "API Key Inactive - [Key Prefix]"
   - Include: Key prefix (first 8 characters), reason for inquiry

2. **Check for security notifications:**
   - Review email for security alerts
   - Check dashboard for account status

3. **Request reactivation or new key:**
   - Provide business justification
   - Confirm terms of service compliance

**Response Time:** 1-4 hours during business hours

---

### AUTH-004: Expired API Key

**HTTP Status:** 403 Forbidden

**Error Response:**
```json
{
  "error": "Forbidden",
  "message": "API key expired on 2025-11-01. Please rotate your key.",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. API key passed expiration date
2. Key not rotated on schedule (recommended: every 90 days)

**Remediation:**
1. **Request new API key:**
   - Email: support@nfl-analytics.com
   - Subject: "API Key Rotation - [Key Name]"
   - Include: Expired key prefix, desired expiration (90 days recommended)

2. **Update application with new key:**
   ```bash
   # Update .env file
   echo "NFL_API_KEY=new_api_key_here" >> .env
   ```

3. **Test new key:**
   ```bash
   curl -X GET "https://api.nfl-analytics.com/health" \
     -H "X-API-Key: $NEW_API_KEY"
   ```

4. **Revoke old key after 7-day grace period:**
   - Ensures all systems updated
   - Email support to revoke old key

**Automation:**
```python
# Check key expiration
from datetime import datetime, timedelta

KEY_CREATED_AT = datetime.fromisoformat(os.getenv("KEY_CREATED_AT"))
ROTATION_THRESHOLD = timedelta(days=90)

if datetime.utcnow() - KEY_CREATED_AT > ROTATION_THRESHOLD:
    print("⚠️  API key rotation recommended")
    print("Contact support@nfl-analytics.com")
```

---

### AUTH-005: Endpoint Permission Denied

**HTTP Status:** 403 Forbidden

**Error Response:**
```json
{
  "error": "Forbidden",
  "message": "API key does not have permission for endpoint: /predict",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. API key restricted to specific endpoints (e.g., read-only key)
2. Attempting to access admin endpoint with non-admin key
3. Key provisioned with limited permissions

**Remediation:**
1. **Check key permissions:**
   - Contact support to verify allowed endpoints
   - Review API key provisioning document

2. **Request endpoint access:**
   - Email: support@nfl-analytics.com
   - Subject: "Endpoint Access Request - [Endpoint]"
   - Include: Key prefix, endpoint(s) needed, business justification

3. **Use appropriate key:**
   - Admin endpoints require admin-level API keys
   - Separate keys for different permission levels

**Example Restricted Key:**
```json
{
  "key_name": "readonly_dashboard",
  "allowed_endpoints": ["/health", "/models"],
  "forbidden_endpoints": ["/predict", "/predict/batch", "/admin/refresh"]
}
```

---

## Rate Limiting Errors (429)

### RATE-001: Hourly Rate Limit Exceeded

**HTTP Status:** 429 Too Many Requests

**Error Response:**
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

**Common Causes:**
1. Exceeded 100 requests in rolling 1-hour window
2. Burst traffic (many requests in short time)
3. Polling loop without rate limiting
4. Multiple clients sharing same API key

**Remediation:**

**1. Implement exponential backoff:**
```python
import time
import requests

def predict_with_backoff(payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                if attempt < max_retries - 1:
                    print(f"Rate limited. Retrying in {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    raise
            else:
                raise
```

**2. Monitor rate limit headers:**
```python
response = requests.post(url, headers=headers, json=payload)

remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
if remaining < 10:
    print(f"⚠ Warning: Only {remaining} requests remaining")
```

**3. Request higher rate limit:**
- Email: support@nfl-analytics.com
- Subject: "Rate Limit Increase Request"
- Include: Current usage, required limit, business justification

**4. Use batch endpoint:**
```python
# Instead of 100 individual requests
for player in players:
    predict(player)  # 100 requests

# Use batch endpoint
predict_batch(players)  # 1 batch request
```

---

### RATE-002: Daily Rate Limit Exceeded

**HTTP Status:** 429 Too Many Requests

**Error Response:**
```json
{
  "error": "RateLimitExceeded",
  "message": "Daily rate limit exceeded. Limit: 1000 requests per 24 hour(s). Try again in 43200 seconds.",
  "details": {
    "limit": 1000,
    "remaining": 0,
    "reset_at": "2025-11-16T00:00:00Z"
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Headers:**
```http
Retry-After: 43200
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1731715200
```

**Common Causes:**
1. Exceeded 1,000 requests in rolling 24-hour window
2. Large backfill or batch operation
3. Inefficient polling
4. Multiple integrations sharing same key

**Remediation:**

**1. Wait until reset:**
- Check `Retry-After` header for exact reset time
- Schedule retry after rate limit window expires

**2. Upgrade tier:**
| Tier | Daily Limit | Hourly Limit | Cost |
|------|-------------|--------------|------|
| Free | 1,000 | 100 | $0 |
| Standard | 5,000 | 500 | $99/mo |
| Premium | 20,000 | 2,000 | $299/mo |
| Enterprise | Custom | Custom | Contact sales |

**3. Optimize request patterns:**
```python
# ❌ INEFFICIENT - Polling every 5 seconds
while True:
    predict(player)
    time.sleep(5)  # 17,280 requests/day

# ✅ EFFICIENT - Pre-compute and cache
predictions = predict_batch(all_players)  # 1 request
cache.set("predictions", predictions, ttl=3600)
```

**4. Use separate keys per integration:**
- Dashboard: 1 key
- Batch processing: 1 key
- Live betting: 1 key
- Each gets independent rate limits

---

### RATE-003: Batch Size Limit Exceeded

**HTTP Status:** 422 Unprocessable Entity

**Error Response:**
```json
{
  "error": "ValidationError",
  "message": "Batch size exceeds maximum",
  "details": {
    "batch_size": 150,
    "max_batch_size": 100
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Batch request with >100 predictions
2. Misunderstanding of batch limits

**Remediation:**

**Split large batches:**
```python
def predict_large_batch(predictions, batch_size=100):
    """Split large batch into smaller chunks"""
    results = []

    for i in range(0, len(predictions), batch_size):
        batch = predictions[i:i + batch_size]
        result = predict_batch(batch)
        results.extend(result["predictions"])

        # Rate limiting between batches
        if i + batch_size < len(predictions):
            time.sleep(5)  # 5 seconds between batches

    return results

# Usage
all_predictions = [...]  # 500 predictions
results = predict_large_batch(all_predictions)  # 5 batch requests
```

---

## Validation Errors (422)

### VAL-001: Invalid Stat Type

**HTTP Status:** 422 Unprocessable Entity

**Error Response:**
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

**Common Causes:**
1. Typo in stat_type (e.g., "passing_yard" instead of "passing_yards")
2. Using unsupported stat type (e.g., "interceptions")
3. Case mismatch (e.g., "Passing_Yards" instead of "passing_yards")

**Remediation:**

**Valid stat types:**
- `passing_yards` - Passing yards (QB only)
- `rushing_yards` - Rushing yards (all positions)
- `receiving_yards` - Receiving yards (WR/TE/RB)
- `passing_tds` - Passing touchdowns (QB only)
- `rushing_tds` - Rushing touchdowns (all positions)

**Python validation:**
```python
from enum import Enum

class StatType(str, Enum):
    PASSING_YARDS = "passing_yards"
    RUSHING_YARDS = "rushing_yards"
    RECEIVING_YARDS = "receiving_yards"
    PASSING_TDS = "passing_tds"
    RUSHING_TDS = "rushing_tds"

# Validate before sending
stat_type = "passing_yards"
if stat_type not in [s.value for s in StatType]:
    raise ValueError(f"Invalid stat_type: {stat_type}")
```

---

### VAL-002: Invalid Game ID Format

**HTTP Status:** 422 Unprocessable Entity

**Error Response:**
```json
{
  "error": "ValidationError",
  "message": "Invalid game_id format",
  "details": {
    "game_id": "Expected format: YYYY_WW_HOME_AWAY (e.g., '2024_10_KC_LV')"
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Wrong separator (e.g., "2024-10-KC-LV" instead of "2024_10_KC_LV")
2. Missing components (e.g., "2024_KC_LV")
3. Invalid team abbreviations (e.g., "LA" instead of "LAR" or "LAC")
4. Leading zeros removed (e.g., "2024_1_KC_LV" instead of "2024_01_KC_LV")

**Remediation:**

**Correct format:**
```
YYYY_WW_HOME_AWAY
2024_10_KC_LV
```

**Components:**
- `YYYY` - Season year (4 digits)
- `WW` - Week number (2 digits, zero-padded)
- `HOME` - Home team abbreviation (2-3 letters)
- `AWAY` - Away team abbreviation (2-3 letters)

**Python validation:**
```python
import re

def validate_game_id(game_id):
    """Validate game_id format"""
    pattern = r'^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$'

    if not re.match(pattern, game_id):
        raise ValueError(
            f"Invalid game_id format: {game_id}. "
            f"Expected: YYYY_WW_HOME_AWAY (e.g., '2024_10_KC_LV')"
        )

    return True

# Usage
validate_game_id("2024_10_KC_LV")  # ✅ Valid
validate_game_id("2024-10-KC-LV")  # ❌ Raises ValueError
```

**Team Abbreviations:**
```
ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE,
DAL, DEN, DET, GB, HOU, IND, JAX, KC,
LAC, LAR, LV, MIA, MIN, NE, NO, NYG,
NYJ, PHI, PIT, SEA, SF, TB, TEN, WAS
```

---

### VAL-003: Invalid Player ID Format

**HTTP Status:** 422 Unprocessable Entity

**Error Response:**
```json
{
  "error": "ValidationError",
  "message": "Invalid player_id format",
  "details": {
    "player_id": "Expected nflverse format: XX-XXXXXXX (e.g., '00-0033873')"
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Using old GSIS ID format instead of nflverse
2. Missing leading zeros
3. Wrong separator (e.g., "00_0033873" instead of "00-0033873")

**Remediation:**

**Correct format:**
```
XX-XXXXXXX
00-0033873
```

**Python validation:**
```python
import re

def validate_player_id(player_id):
    """Validate player_id format (nflverse)"""
    pattern = r'^\d{2}-\d{7}$'

    if not re.match(pattern, player_id):
        raise ValueError(
            f"Invalid player_id format: {player_id}. "
            f"Expected: XX-XXXXXXX (e.g., '00-0033873')"
        )

    return True
```

**Convert GSIS to nflverse:**
```python
def gsis_to_nflverse(gsis_id):
    """Convert GSIS ID to nflverse format"""
    # GSIS: 00-0012345 (7 digits)
    # nflverse: 00-0012345 (7 digits with leading zeros)
    return f"{gsis_id[:2]}-{gsis_id[2:].zfill(7)}"
```

---

### VAL-004: Missing Required Field

**HTTP Status:** 422 Unprocessable Entity

**Error Response:**
```json
{
  "error": "ValidationError",
  "message": "Missing required field",
  "details": {
    "missing_fields": ["player_id"]
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Request body missing required field
2. Field name typo (e.g., "playerid" instead of "player_id")
3. Field set to null instead of omitted

**Remediation:**

**Required fields for /predict:**
- `game_id` (string)
- `player_id` (string)
- `stat_type` (string)

**Python validation:**
```python
def validate_prediction_request(data):
    """Validate prediction request has required fields"""
    required = ["game_id", "player_id", "stat_type"]

    missing = [field for field in required if field not in data or data[field] is None]

    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return True
```

---

## Server Errors (500, 503)

### SRV-001: Predictor Not Initialized

**HTTP Status:** 503 Service Unavailable

**Error Response:**
```json
{
  "error": "ServiceUnavailable",
  "message": "Predictor not initialized",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. API server starting up (models still loading)
2. All models failed to load
3. Model files missing from disk
4. Database connection failed during startup

**Remediation:**

**1. Check service health:**
```bash
curl -X GET "https://api.nfl-analytics.com/health"
```

Expected response if starting up:
```json
{
  "status": "unhealthy",
  "models_loaded": 0
}
```

**2. Wait for models to load:**
- Typical startup time: 30-60 seconds
- Retry request after 60 seconds

**3. Check status page:**
- https://status.nfl-analytics.com
- Monitor for ongoing incidents

**4. Contact support if persists >5 minutes:**
- Email: support@nfl-analytics.com
- Subject: "Service Unavailable - Predictor Not Initialized"

---

### SRV-002: Insufficient Models Loaded

**HTTP Status:** 503 Service Unavailable

**Error Response:**
```json
{
  "error": "ServiceUnavailable",
  "message": "Insufficient models loaded (need at least 2)",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Partial model loading failure (only 1/4 models loaded)
2. Model file corruption
3. Insufficient memory to load all models

**Remediation:**

**1. Check models status:**
```bash
curl -X GET "https://api.nfl-analytics.com/models" \
  -H "X-API-Key: $API_KEY"
```

**2. Review model status:**
```json
{
  "models": [
    {"model_name": "state_space_v2", "status": "loaded"},
    {"model_name": "hierarchical_v2", "status": "failed"}
  ],
  "loaded_models": 1,
  "ensemble_ready": false
}
```

**3. Report failed models:**
- Email: support@nfl-analytics.com
- Subject: "Model Loading Failure - [Model Name]"
- Include: Model name(s) failing to load

---

### SRV-003: Internal Server Error

**HTTP Status:** 500 Internal Server Error

**Error Response:**
```json
{
  "error": "InternalServerError",
  "message": "Prediction failed: [error details]",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Unexpected error during prediction
2. Database query failure
3. Model inference error
4. Memory exhaustion

**Remediation:**

**1. Retry request once:**
- Transient errors may resolve on retry
- Wait 5 seconds before retry

**2. Simplify request:**
- Try single prediction instead of batch
- Test with different game/player combination

**3. Report persistent errors:**
- Email: support@nfl-analytics.com
- Subject: "Internal Server Error - /predict"
- Include: Request payload, error message, timestamp

---

## Client Errors (400, 404)

### CLI-001: Bad Request

**HTTP Status:** 400 Bad Request

**Error Response:**
```json
{
  "error": "BadRequest",
  "message": "Invalid JSON in request body",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Malformed JSON (missing comma, extra comma, unquoted keys)
2. Wrong Content-Type header
3. Empty request body

**Remediation:**

**1. Validate JSON:**
```python
import json

# Validate JSON before sending
payload = {
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
}

# Test JSON encoding
try:
    json.dumps(payload)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

**2. Set Content-Type header:**
```bash
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \  # Required
  -d '{"game_id": "2024_10_KC_LV", ...}'
```

---

### CLI-002: Not Found

**HTTP Status:** 404 Not Found

**Error Response:**
```json
{
  "error": "NotFound",
  "message": "Endpoint not found",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Common Causes:**
1. Typo in endpoint URL (e.g., "/predicts" instead of "/predict")
2. Wrong HTTP method (e.g., GET /predict instead of POST /predict)
3. Wrong base URL

**Remediation:**

**Valid endpoints:**
- `GET /health`
- `GET /models`
- `POST /predict`
- `POST /predict/batch`
- `POST /admin/refresh`
- `GET /metrics`

**Check endpoint spelling and method:**
```bash
# ❌ WRONG
curl -X GET "https://api.nfl-analytics.com/predict"  # Wrong method

# ✅ CORRECT
curl -X POST "https://api.nfl-analytics.com/predict"
```

---

## Error Code Reference

### Quick Lookup Table

| Code | HTTP | Error Type | Severity | Retry? |
|------|------|------------|----------|--------|
| AUTH-001 | 401 | Missing API Key | High | No - Fix auth |
| AUTH-002 | 401 | Invalid API Key | High | No - Fix key |
| AUTH-003 | 403 | Inactive/Revoked Key | Critical | No - Contact support |
| AUTH-004 | 403 | Expired API Key | High | No - Rotate key |
| AUTH-005 | 403 | Endpoint Permission Denied | Medium | No - Request access |
| RATE-001 | 429 | Hourly Rate Limit | Medium | Yes - After reset |
| RATE-002 | 429 | Daily Rate Limit | Medium | Yes - After reset |
| RATE-003 | 422 | Batch Size Limit | Low | No - Split batch |
| VAL-001 | 422 | Invalid Stat Type | Low | No - Fix input |
| VAL-002 | 422 | Invalid Game ID | Low | No - Fix format |
| VAL-003 | 422 | Invalid Player ID | Low | No - Fix format |
| VAL-004 | 422 | Missing Required Field | Low | No - Add field |
| SRV-001 | 503 | Predictor Not Initialized | High | Yes - Wait 60s |
| SRV-002 | 503 | Insufficient Models | High | Yes - Wait 60s |
| SRV-003 | 500 | Internal Server Error | Critical | Yes - Once |
| CLI-001 | 400 | Bad Request | Low | No - Fix JSON |
| CLI-002 | 404 | Not Found | Low | No - Fix URL |

### Error Severity Levels

**Critical:**
- Service outage or data corruption
- Immediate attention required
- Contact support within 1 hour

**High:**
- Blocking operations
- Requires manual intervention
- Contact support within 4 hours

**Medium:**
- Degraded functionality
- Workarounds available
- Can self-remediate

**Low:**
- Client-side validation errors
- Self-service remediation
- No support contact needed

---

## Support

**Documentation:** https://docs.nfl-analytics.com
**API Status:** https://status.nfl-analytics.com
**Support Email:** support@nfl-analytics.com

**When Reporting Errors:**
1. Error code (e.g., AUTH-003)
2. Full error message
3. Request payload (sanitize API keys)
4. Timestamp
5. Steps to reproduce

**Response Times:**
- Critical: 1 hour
- High: 4 hours
- Medium: 8 hours
- Low: 24 hours
