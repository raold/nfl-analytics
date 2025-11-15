# Quick Start: API Security

**TL;DR for developers who want to start using the API immediately**

## 1. Get API Key (30 seconds)

```bash
# For development, use the test key (already provisioned)
export NFL_API_KEY="dev_test_key_12345678901234567890123456789012"
```

**Production:** Email support@nfl-analytics.com to request your API key.

## 2. Make Your First Request (30 seconds)

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Prediction (auth required)
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $NFL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'
```

## 3. Check Your Rate Limit (10 seconds)

Every response includes rate limit headers:

```bash
curl -i http://localhost:8000/models \
  -H "X-API-Key: $NFL_API_KEY" \
  | grep X-RateLimit

# Output:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 99
# X-RateLimit-Reset: 1731678000
```

## 4. Python Client Example (2 minutes)

```python
import os
import requests

# Load from environment
API_KEY = os.getenv("NFL_API_KEY")
BASE_URL = "http://localhost:8000"

def predict(game_id, player_id, stat_type):
    """Make prediction with automatic rate limit handling"""
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
        }
    )

    # Check rate limit
    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
    if remaining < 10:
        print(f"⚠️  Only {remaining} requests remaining!")

    # Handle errors
    if response.status_code == 401:
        raise Exception("Invalid API key")
    elif response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        raise Exception(f"Rate limited. Retry in {retry_after}s")
    elif response.status_code != 200:
        raise Exception(f"API error: {response.status_code}")

    return response.json()

# Usage
prediction = predict("2024_10_KC_LV", "00-0033873", "passing_yards")
print(f"Prediction: {prediction['prediction']:.1f} yards")
print(f"90% CI: [{prediction['lower_bound']:.1f}, {prediction['upper_bound']:.1f}]")
```

## 5. Error Handling (2 minutes)

```python
import time

def predict_with_retry(game_id, player_id, stat_type, max_retries=3):
    """Predict with exponential backoff on rate limit"""

    for attempt in range(max_retries):
        try:
            return predict(game_id, player_id, stat_type)

        except Exception as e:
            if "Rate limited" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    raise Exception("Max retries exceeded")
```

## 6. Security Checklist

Before deploying to production:

```bash
# ✅ Never commit API keys
grep -r "api_key.*=" . --exclude-dir=.git | grep -v "os.getenv"

# ✅ Add .env to .gitignore
echo ".env" >> .gitignore

# ✅ Load from environment variables
# Python: os.getenv("NFL_API_KEY")
# R: Sys.getenv("NFL_API_KEY")
# Node: process.env.NFL_API_KEY

# ✅ Use HTTPS in production
BASE_URL = "https://api.nfl-analytics.com"  # Not http://
```

## 7. Common Errors and Fixes

### 401 Unauthorized

```
Error: API key required
Fix: Add X-API-Key header
```

```bash
# ❌ WRONG
curl http://localhost:8000/predict

# ✅ CORRECT
curl -H "X-API-Key: $NFL_API_KEY" http://localhost:8000/predict
```

### 403 Forbidden

```
Error: API key is inactive or revoked
Fix: Request new API key from support
```

### 429 Rate Limited

```
Error: Rate limit exceeded
Fix: Implement exponential backoff or request limit increase
```

```python
# Check Retry-After header
retry_after = int(response.headers.get("Retry-After", 60))
time.sleep(retry_after)
```

## 8. Rate Limits by Endpoint

| Endpoint | Limit | Use For |
|----------|-------|---------|
| `/predict` | 100/hour | Predictions |
| `/models` | 1,000/hour | Metadata |
| `/health` | 10,000/hour | Monitoring |

**Need Higher Limits?** Email support@nfl-analytics.com

## 9. Full Documentation

- **Authentication:** [docs/api/authentication_guide.md](authentication_guide.md)
- **Rate Limiting:** [docs/api/rate_limiting_guide.md](rate_limiting_guide.md)
- **Security:** [docs/api/security_best_practices.md](security_best_practices.md)
- **OpenAPI Spec:** [docs/api/openapi.yaml](openapi.yaml)
- **Interactive Docs:** http://localhost:8000/docs

## 10. Getting Help

**Email:** support@nfl-analytics.com
**Response Time:** 24 hours

**Urgent Security Issues:** security@nfl-analytics.com
**Response Time:** 1 hour

---

**That's it!** You're ready to use the NFL Analytics API securely.
