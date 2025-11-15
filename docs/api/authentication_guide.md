# API Authentication Guide

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

The NFL Analytics Prediction API uses API key authentication to secure access to prediction endpoints. All prediction requests require a valid API key in the `X-API-Key` header.

## Table of Contents

1. [Obtaining an API Key](#obtaining-an-api-key)
2. [Using Your API Key](#using-your-api-key)
3. [API Key Management](#api-key-management)
4. [Key Rotation](#key-rotation)
5. [Security Best Practices](#security-best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Obtaining an API Key

### Request Access

Contact your administrator to provision an API key:

**Email:** support@nfl-analytics.com
**Subject:** API Key Request

Include in your request:
- **Organization name**
- **Use case description**
- **Expected request volume**
- **Contact email for key delivery**

### Key Attributes

Each API key includes:

| Attribute | Description | Example |
|-----------|-------------|---------|
| **API Key** | 64-character unique identifier | `dev_test_key_12345...` |
| **Key Name** | Descriptive identifier | "Production Dashboard" |
| **Owner Email** | Contact for key | admin@company.com |
| **Rate Limit** | Requests per hour | 100/hour, 1000/day |
| **Endpoints** | Allowed endpoints | ["/predict", "/models"] |
| **Expiration** | Optional expiry date | 2026-11-15 |

### Key Delivery

Your API key will be delivered securely via:
- Encrypted email
- Secure credential management system (1Password, Vault)

**⚠️ WARNING:** Never share your API key publicly. Treat it like a password.

---

## Using Your API Key

### Header-Based Authentication

Include your API key in the `X-API-Key` header for all requests to protected endpoints:

#### curl Example

```bash
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'
```

#### Python Example

```python
import requests

API_KEY = "your_api_key_here"  # Load from environment variable
BASE_URL = "https://api.nfl-analytics.com"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
}

response = requests.post(
    f"{BASE_URL}/predict",
    headers=headers,
    json=payload
)

if response.status_code == 200:
    prediction = response.json()
    print(f"Prediction: {prediction['prediction']:.1f} yards")
    print(f"90% CI: [{prediction['lower_bound']:.1f}, {prediction['upper_bound']:.1f}]")
else:
    print(f"Error {response.status_code}: {response.json()['message']}")
```

#### R Example

```r
library(httr)
library(jsonlite)

API_KEY <- Sys.getenv("NFL_API_KEY")  # Load from environment
BASE_URL <- "https://api.nfl-analytics.com"

# Make prediction request
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

# Parse response
if (status_code(response) == 200) {
  prediction <- content(response, as = "parsed")
  cat(sprintf("Prediction: %.1f yards\n", prediction$prediction))
  cat(sprintf("90%% CI: [%.1f, %.1f]\n",
    prediction$lower_bound,
    prediction$upper_bound
  ))
} else {
  error <- content(response, as = "parsed")
  cat(sprintf("Error %d: %s\n", status_code(response), error$message))
}
```

#### JavaScript Example

```javascript
const axios = require('axios');

const API_KEY = process.env.NFL_API_KEY;
const BASE_URL = 'https://api.nfl-analytics.com';

async function getPrediction() {
  try {
    const response = await axios.post(
      `${BASE_URL}/predict`,
      {
        game_id: '2024_10_KC_LV',
        player_id: '00-0033873',
        stat_type: 'passing_yards'
      },
      {
        headers: {
          'X-API-Key': API_KEY,
          'Content-Type': 'application/json'
        }
      }
    );

    const { prediction, lower_bound, upper_bound } = response.data;
    console.log(`Prediction: ${prediction.toFixed(1)} yards`);
    console.log(`90% CI: [${lower_bound.toFixed(1)}, ${upper_bound.toFixed(1)}]`);

  } catch (error) {
    if (error.response) {
      console.error(`Error ${error.response.status}: ${error.response.data.message}`);
    } else {
      console.error(`Request failed: ${error.message}`);
    }
  }
}

getPrediction();
```

### Response Headers

All API responses include rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1731678000
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Total requests allowed per hour |
| `X-RateLimit-Remaining` | Requests remaining in current window |
| `X-RateLimit-Reset` | Unix timestamp when limit resets |

---

## API Key Management

### Viewing Your Keys

Contact your administrator to view:
- Active API keys
- Usage statistics
- Rate limit quotas
- Expiration dates

### Key Status

API keys can have the following statuses:

| Status | Description | Action |
|--------|-------------|--------|
| **Active** | Key is valid and usable | Continue using |
| **Inactive** | Key is temporarily disabled | Contact admin to reactivate |
| **Revoked** | Key is permanently disabled | Request new key |
| **Expired** | Key has passed expiration date | Rotate key |

### Revoking a Key

If your API key is compromised:

1. **Immediately contact support:** support@nfl-analytics.com
2. **Subject:** URGENT: Revoke API Key
3. **Include:** Key prefix (first 8 characters) and reason

Keys are revoked within 1 hour during business hours.

---

## Key Rotation

### Rotation Schedule

**Recommended:** Rotate API keys every 90 days for security.

### Rotation Process

1. **Request new key** from administrator
2. **Update applications** with new key
3. **Test new key** in development/staging
4. **Deploy to production**
5. **Revoke old key** after 7-day grace period

### Automated Rotation Script

```python
import os
import requests
from datetime import datetime, timedelta

# Check if key needs rotation (90 days)
KEY_CREATED_AT = datetime.fromisoformat(os.getenv("KEY_CREATED_AT"))
ROTATION_THRESHOLD = timedelta(days=90)

if datetime.utcnow() - KEY_CREATED_AT > ROTATION_THRESHOLD:
    print("⚠️  API key rotation recommended")
    print("Contact support@nfl-analytics.com to request new key")
```

---

## Security Best Practices

### ✅ DO

1. **Store keys in environment variables**
   ```bash
   export NFL_API_KEY="your_key_here"
   ```

2. **Use secret management systems**
   - 1Password, Vault, AWS Secrets Manager

3. **Rotate keys regularly**
   - Every 90 days minimum

4. **Monitor usage**
   - Check rate limit headers
   - Review usage statistics monthly

5. **Use HTTPS only**
   - Never send keys over unencrypted connections

6. **Implement request signing** (advanced)
   - Add HMAC signatures for additional security

### ❌ DON'T

1. **Never commit keys to version control**
   ```bash
   # Add to .gitignore
   .env
   secrets.yaml
   ```

2. **Never share keys publicly**
   - Don't post in Slack, GitHub issues, etc.

3. **Never use keys in frontend code**
   - Keys should only be used server-side

4. **Never use the same key across environments**
   - Use separate keys for dev/staging/production

5. **Never ignore security alerts**
   - Respond immediately to compromise notifications

### Environment Variable Setup

**.env file** (add to .gitignore):
```bash
# NFL Analytics API
NFL_API_KEY="your_api_key_here"
NFL_API_BASE_URL="https://api.nfl-analytics.com"
```

**Load in Python:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NFL_API_KEY")
```

**Load in R:**
```r
library(dotenv)

load_dot_env()
API_KEY <- Sys.getenv("NFL_API_KEY")
```

---

## Troubleshooting

### 401 Unauthorized

**Error:**
```json
{
  "error": "Unauthorized",
  "message": "API key required. Include 'X-API-Key' header in request.",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Causes:**
- Missing `X-API-Key` header
- Empty API key value
- Malformed header name

**Solution:**
```python
# ❌ WRONG
headers = {"API-Key": API_KEY}

# ✅ CORRECT
headers = {"X-API-Key": API_KEY}
```

### 403 Forbidden - Invalid Key

**Error:**
```json
{
  "error": "Forbidden",
  "message": "Invalid API key",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Causes:**
- API key doesn't exist in database
- Typo in API key value
- Using development key in production

**Solution:**
- Verify API key is correct (all 64 characters)
- Check for leading/trailing whitespace
- Confirm key was provisioned by administrator

### 403 Forbidden - Inactive/Revoked

**Error:**
```json
{
  "error": "Forbidden",
  "message": "API key is inactive or revoked. Contact administrator.",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Causes:**
- Key manually deactivated by admin
- Key revoked due to security incident
- Account suspended

**Solution:**
- Contact support@nfl-analytics.com
- Request new API key

### 403 Forbidden - Expired

**Error:**
```json
{
  "error": "Forbidden",
  "message": "API key expired on 2025-11-01. Please rotate your key.",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Causes:**
- API key passed expiration date

**Solution:**
- Request new API key
- Follow key rotation process

### 403 Forbidden - Endpoint Permission

**Error:**
```json
{
  "error": "Forbidden",
  "message": "API key does not have permission for endpoint: /predict",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Causes:**
- API key has restricted endpoint access
- Attempting to access unauthorized endpoint

**Solution:**
- Contact admin to update endpoint permissions
- Use authorized endpoints only

### 429 Rate Limit Exceeded

**Error:**
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

**Solution:**
- Wait until `reset_at` timestamp
- Implement exponential backoff
- Request higher rate limit from admin

**Exponential Backoff Example:**
```python
import time
import requests
from requests.exceptions import HTTPError

def predict_with_retry(payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                headers={"X-API-Key": API_KEY},
                json=payload
            )
            response.raise_for_status()
            return response.json()

        except HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))
                if attempt < max_retries - 1:
                    print(f"Rate limited. Retrying in {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    raise
            else:
                raise

    raise Exception("Max retries exceeded")
```

---

## Support

**Email:** support@nfl-analytics.com
**Documentation:** https://docs.nfl-analytics.com
**Status Page:** https://status.nfl-analytics.com

**Response Times:**
- Critical (security): 1 hour
- High (key rotation): 4 hours
- Normal (questions): 24 hours
