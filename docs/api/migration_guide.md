# API Migration Guide

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

Step-by-step guide for migrating between NFL Analytics API versions. Includes breaking changes, deprecated endpoints, authentication updates, and migration checklists.

## Table of Contents

1. [Version History](#version-history)
2. [v1.0 → v1.1 Migration](#v10--v11-migration)
3. [Deprecated Features](#deprecated-features)
4. [Authentication Migration](#authentication-migration)
5. [Response Format Changes](#response-format-changes)
6. [Rate Limiting Changes](#rate-limiting-changes)
7. [Migration Checklist](#migration-checklist)
8. [Rollback Plan](#rollback-plan)

---

## Version History

| Version | Release Date | Status | Breaking Changes |
|---------|--------------|--------|------------------|
| v1.0 | October 2025 | Deprecated (EOL: March 2026) | N/A (initial release) |
| v1.1 | November 2025 | Current | Authentication required |
| v1.2 | Q1 2026 (planned) | Upcoming | Webhooks, HTTP/2 |

**Support Policy:**
- Current version: Full support
- Previous version: Security patches only (6 months)
- EOL versions: No support

---

## v1.0 → v1.1 Migration

### Overview

**Release Date:** November 15, 2025
**Migration Window:** 4 months (Nov 2025 - Feb 2026)
**End of Life (v1.0):** March 31, 2026

**Major Changes:**
1. **Authentication now required** for all prediction endpoints
2. **Rate limiting** introduced (100 req/hour default)
3. **Error response format** standardized
4. **New endpoint:** `/admin/refresh` (admin only)
5. **Performance improvements:** 30-40% latency reduction

### Breaking Changes

#### 1. Authentication Required

**Before (v1.0):**
```python
# No authentication
response = requests.post(
    "https://api.nfl-analytics.com/predict",
    json={
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0033873",
        "stat_type": "passing_yards"
    }
)
```

**After (v1.1):**
```python
# API key required
response = requests.post(
    "https://api.nfl-analytics.com/predict",
    headers={
        "X-API-Key": os.getenv("NFL_API_KEY")  # ⚠️ REQUIRED
    },
    json={
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0033873",
        "stat_type": "passing_yards"
    }
)
```

**Migration Steps:**
1. Provision API key (contact administrator or use dashboard)
2. Store API key in environment variable: `export NFL_API_KEY=your_key_here`
3. Add `X-API-Key` header to all requests
4. Test with `/health` endpoint (no auth required)

**Error if missing authentication:**
```json
{
  "error": "Unauthorized",
  "message": "API key required. Include 'X-API-Key' header in request.",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

#### 2. Rate Limiting Introduced

**Before (v1.0):** Unlimited requests

**After (v1.1):** 100 requests/hour default

**New Response Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1731678000
```

**Migration Steps:**
1. Monitor `X-RateLimit-Remaining` header
2. Implement exponential backoff for 429 errors
3. Use batch endpoint for multiple predictions
4. Request limit increase if needed

**Example: Handle 429 errors**
```python
import time

def predict_with_retry(payload, max_retries=3):
    """Predict with rate limit handling"""

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            # Rate limited
            retry_after = int(response.headers.get("Retry-After", 60))
            if attempt < max_retries - 1:
                print(f"Rate limited. Retrying in {retry_after}s...")
                time.sleep(retry_after)
            else:
                raise Exception("Rate limit exceeded after retries")
        else:
            response.raise_for_status()

    raise Exception("Max retries exceeded")
```

#### 3. Error Response Format Standardized

**Before (v1.0):**
```json
{
  "error": "Prediction failed",
  "status": 500
}
```

**After (v1.1):**
```json
{
  "error": "InternalServerError",
  "message": "Prediction failed",
  "details": {
    "component": "ensemble",
    "reason": "model_unavailable"
  },
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Migration Steps:**
1. Update error handling to parse new format
2. Check `error` field (not `status`)
3. Use `details` for diagnostic information

**Example: Updated error handling**
```python
# Before (v1.0)
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
except requests.HTTPError as e:
    error_msg = e.response.json().get("error")
    print(f"Error: {error_msg}")

# After (v1.1)
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
except requests.HTTPError as e:
    error_data = e.response.json()
    error_type = error_data.get("error")
    error_msg = error_data.get("message")
    details = error_data.get("details", {})

    print(f"Error [{error_type}]: {error_msg}")
    if details:
        print(f"Details: {details}")
```

### Non-Breaking Changes

#### 1. New Endpoint: `/admin/refresh`

**Purpose:** Refresh model cache (admin API keys only)

**Usage:**
```python
response = requests.post(
    "https://api.nfl-analytics.com/admin/refresh",
    headers={"X-API-Key": admin_api_key},
    json={"force_reload": true}
)
```

**Note:** Regular API keys will receive 403 Forbidden.

#### 2. Performance Improvements

- 30-40% latency reduction via database connection pooling
- Improved caching (70-85% hit rate)
- Batch endpoint optimization (10-50x speedup)

**No code changes required** - automatically available in v1.1.

---

## Deprecated Features

### v1.0 Deprecated Endpoints

| Endpoint | Status | Replacement | EOL Date |
|----------|--------|-------------|----------|
| `/predict` (no auth) | Deprecated | `/predict` (with auth) | March 2026 |
| `/models` (no auth) | Deprecated | `/models` (with auth) | March 2026 |

### Deprecation Warnings

v1.0 endpoints return deprecation headers:

```http
Warning: 299 - "API v1.0 is deprecated. Migrate to v1.1 by March 2026. See https://docs.nfl-analytics.com/migration"
```

**Migration Deadline:** March 31, 2026

After this date, v1.0 endpoints will return:
```json
{
  "error": "EndpointDeprecated",
  "message": "API v1.0 is no longer supported. Please upgrade to v1.1+",
  "migration_guide": "https://docs.nfl-analytics.com/migration"
}
```

---

## Authentication Migration

### Step 1: Provision API Key

**Option A: Dashboard (Recommended)**
1. Login to https://dashboard.nfl-analytics.com
2. Navigate to Settings → API Keys
3. Click "Generate New Key"
4. Copy key (shown once only)
5. Set permissions (read, write, admin)

**Option B: Contact Administrator**
1. Email support@nfl-analytics.com with:
   - Organization name
   - Use case (production, development, testing)
   - Expected request volume
2. Receive API key via secure channel

### Step 2: Store API Key Securely

**❌ BAD: Hardcoded in source code**
```python
API_KEY = "dev_test_key_1234567890abcdef..."  # Never do this!
```

**✅ GOOD: Environment variable**
```bash
# .env file (add to .gitignore)
NFL_API_KEY=dev_test_key_1234567890abcdef...

# Load in Python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("NFL_API_KEY")
```

**✅ GOOD: Secrets manager (production)**
```python
import boto3

def get_api_key():
    """Retrieve API key from AWS Secrets Manager"""
    client = boto3.client('secretsmanager', region_name='us-east-1')

    response = client.get_secret_value(SecretId='nfl-api-key')
    return response['SecretString']

API_KEY = get_api_key()
```

### Step 3: Update All API Calls

**Find all API calls:**
```bash
# Search codebase
grep -r "api.nfl-analytics.com" . --include="*.py"
grep -r "requests.post" . --include="*.py"
```

**Add authentication header:**
```python
# Create session with default headers
import requests

session = requests.Session()
session.headers.update({
    "X-API-Key": os.getenv("NFL_API_KEY")
})

# All requests now include auth
response = session.post(url, json=payload)
```

### Step 4: Test Authentication

```python
def test_authentication():
    """Verify API key works"""

    # Test health endpoint (no auth required)
    response = requests.get("https://api.nfl-analytics.com/health")
    assert response.status_code == 200
    print("✓ Health check passed (no auth)")

    # Test models endpoint (auth required)
    response = requests.get(
        "https://api.nfl-analytics.com/models",
        headers={"X-API-Key": os.getenv("NFL_API_KEY")}
    )

    if response.status_code == 200:
        print("✓ Authentication successful")
        data = response.json()
        print(f"  Models loaded: {data['loaded_models']}")
    elif response.status_code == 401:
        print("✗ Authentication failed (invalid key)")
    elif response.status_code == 403:
        print("✗ Authentication failed (key inactive/expired)")
    else:
        print(f"✗ Unexpected status: {response.status_code}")

    return response.status_code == 200

# Run test
test_authentication()
```

---

## Response Format Changes

### Prediction Response (v1.0 vs v1.1)

**v1.0:**
```json
{
  "player_id": "00-0033873",
  "prediction": 285.3,
  "lower": 215.1,
  "upper": 355.5
}
```

**v1.1:**
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
    }
  ],
  "timestamp": "2025-11-15T12:00:00Z",
  "ensemble_method": "inverse_variance_weighted"
}
```

**Migration:**
```python
# v1.0
lower = response_data['lower']
upper = response_data['upper']

# v1.1 (renamed fields)
lower = response_data['lower_bound']
upper = response_data['upper_bound']

# New fields available
confidence = response_data['confidence']
models = response_data['models_used']
```

### Error Response Changes

**v1.0:**
```json
{
  "error": "Invalid input",
  "status": 422
}
```

**v1.1:**
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

---

## Rate Limiting Changes

### Default Limits (v1.1)

| Limit Type | Default | Configurable |
|------------|---------|--------------|
| Hourly | 100 requests | Yes (per key) |
| Daily | 1,000 requests | Yes (per key) |
| Burst | 10 requests/second | No |

### Checking Rate Limit Status

```python
def check_rate_limit():
    """Check current rate limit status"""

    response = requests.get(
        "https://api.nfl-analytics.com/health",
        headers={"X-API-Key": os.getenv("NFL_API_KEY")}
    )

    # Parse rate limit headers
    limit = int(response.headers.get("X-RateLimit-Limit", 0))
    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
    reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))

    # Calculate reset time
    from datetime import datetime
    reset_time = datetime.fromtimestamp(reset_timestamp)

    print(f"Rate Limit Status:")
    print(f"  Limit: {limit} requests/hour")
    print(f"  Remaining: {remaining}")
    print(f"  Resets at: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        "limit": limit,
        "remaining": remaining,
        "reset_at": reset_time
    }
```

### Requesting Limit Increase

**Eligibility:**
- Production use case
- Historical usage >80% of limit
- No rate limit violations

**Process:**
1. Email support@nfl-analytics.com with:
   - API key ID
   - Current limit and requested limit
   - Use case justification
   - Expected request pattern
2. Approval within 2 business days
3. Limit updated automatically

---

## Migration Checklist

### Pre-Migration (Planning Phase)

- [ ] **Review Changes**
  - [ ] Read migration guide
  - [ ] Identify affected code
  - [ ] Review breaking changes

- [ ] **API Key Provisioning**
  - [ ] Provision API keys (dev, staging, production)
  - [ ] Store keys securely (environment variables, secrets manager)
  - [ ] Test keys in development

- [ ] **Code Audit**
  - [ ] Find all API calls: `grep -r "api.nfl-analytics.com"`
  - [ ] Identify error handling code
  - [ ] Check response parsing logic

### Development Phase

- [ ] **Update Code**
  - [ ] Add `X-API-Key` header to all requests
  - [ ] Update error handling (new format)
  - [ ] Rename `lower`/`upper` → `lower_bound`/`upper_bound`
  - [ ] Implement rate limit handling

- [ ] **Testing**
  - [ ] Unit tests pass
  - [ ] Integration tests pass
  - [ ] Manual testing in development
  - [ ] Load testing (if high volume)

- [ ] **Rate Limiting**
  - [ ] Monitor `X-RateLimit-*` headers
  - [ ] Implement exponential backoff
  - [ ] Use batch endpoint where applicable
  - [ ] Request limit increase if needed

### Staging Phase

- [ ] **Deploy to Staging**
  - [ ] Update environment variables
  - [ ] Deploy code changes
  - [ ] Smoke test critical paths

- [ ] **Validation**
  - [ ] Run end-to-end tests
  - [ ] Check error handling
  - [ ] Verify rate limiting behavior
  - [ ] Review logs for issues

### Production Phase

- [ ] **Pre-Deployment**
  - [ ] Schedule maintenance window
  - [ ] Notify stakeholders
  - [ ] Prepare rollback plan

- [ ] **Deployment**
  - [ ] Update production API keys
  - [ ] Deploy code changes
  - [ ] Monitor for errors

- [ ] **Post-Deployment**
  - [ ] Verify functionality
  - [ ] Monitor error rates
  - [ ] Check performance metrics
  - [ ] Confirm rate limits adequate

### Post-Migration

- [ ] **Cleanup**
  - [ ] Remove v1.0 compatibility code
  - [ ] Update documentation
  - [ ] Archive old API keys

- [ ] **Monitoring**
  - [ ] Set up alerts for 401/403 errors
  - [ ] Monitor rate limit usage
  - [ ] Track API latency

---

## Rollback Plan

### If Migration Fails

**Option 1: Quick Rollback (Emergency)**

1. Revert code to previous version
2. Redeploy previous version
3. Continue using v1.0 (EOL March 2026)

```bash
# Git rollback
git revert <migration-commit-hash>
git push origin main

# Redeploy
./deploy.sh production
```

**Option 2: Hybrid Approach (Temporary)**

Support both v1.0 and v1.1:

```python
def predict(game_id, player_id, stat_type, api_version="1.1"):
    """Support both API versions"""

    if api_version == "1.1":
        # Use authentication
        headers = {"X-API-Key": os.getenv("NFL_API_KEY")}
    else:
        # v1.0 (no auth)
        headers = {}

    response = requests.post(url, headers=headers, json=payload)

    if api_version == "1.1":
        # New response format
        return {
            "prediction": response.json()['prediction'],
            "lower": response.json()['lower_bound'],
            "upper": response.json()['upper_bound']
        }
    else:
        # Old response format
        return response.json()
```

**Option 3: Gradual Migration**

Migrate one component at a time:

1. Week 1: Migrate development environment
2. Week 2: Migrate staging environment
3. Week 3: Migrate 10% of production traffic (canary)
4. Week 4: Migrate 50% of production traffic
5. Week 5: Migrate 100% of production traffic

---

## Common Migration Issues

### Issue 1: Missing API Key

**Symptom:** 401 Unauthorized errors

**Solution:**
```python
# Check API key is loaded
api_key = os.getenv("NFL_API_KEY")
if not api_key:
    raise ValueError("NFL_API_KEY environment variable not set")

# Verify key format (64 characters)
if len(api_key) != 64:
    raise ValueError(f"Invalid API key format (expected 64 chars, got {len(api_key)})")
```

### Issue 2: Rate Limit Exceeded

**Symptom:** 429 Too Many Requests

**Solution:**
```python
# Use batch endpoint
predictions = [...]  # 100 predictions
chunks = [predictions[i:i+100] for i in range(0, len(predictions), 100)]

for chunk in chunks:
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        headers={"X-API-Key": API_KEY},
        json={"predictions": chunk}
    )
    time.sleep(1)  # Rate limit protection
```

### Issue 3: Field Name Changes

**Symptom:** KeyError: 'lower' or 'upper'

**Solution:**
```python
# Compatibility function
def get_bounds(response_data, api_version="1.1"):
    """Get bounds (compatible with both versions)"""
    if api_version == "1.1":
        return response_data['lower_bound'], response_data['upper_bound']
    else:
        return response_data['lower'], response_data['upper']
```

---

## Support

### Migration Support

**Email:** support@nfl-analytics.com
**Subject:** API v1.1 Migration Support

**Include:**
- Current API version
- Migration issue description
- Error messages/logs
- Code snippet (if applicable)

**Response Time:**
- Critical (production down): 1 hour
- High (blocking migration): 4 hours
- Normal (questions): 24 hours

### Resources

- **Migration Guide:** https://docs.nfl-analytics.com/migration
- **API Reference:** https://docs.nfl-analytics.com/api
- **Troubleshooting:** https://docs.nfl-analytics.com/troubleshooting
- **Status Page:** https://status.nfl-analytics.com

---

## Future Migrations (v1.2 Preview)

### Planned Changes (Q1 2026)

**v1.2 Features:**
1. **Webhooks** - Async event notifications
2. **HTTP/2** - Multiplexing, header compression
3. **GraphQL endpoint** - Flexible queries
4. **Streaming predictions** - Server-sent events

**Breaking Changes:**
- None expected (fully backward compatible with v1.1)

**Migration Effort:** Low (opt-in features only)

---

**Last Updated:** November 15, 2025
**Version:** 1.1.0
**Next Review:** January 2026
