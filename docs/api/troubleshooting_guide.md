# API Troubleshooting Guide

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

Practical troubleshooting guide for common API integration issues. Includes step-by-step diagnostics, debugging techniques, and performance optimization strategies.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication Issues](#authentication-issues)
3. [Rate Limiting Challenges](#rate-limiting-challenges)
4. [Prediction Response Interpretation](#prediction-response-interpretation)
5. [Performance Optimization](#performance-optimization)
6. [Error Handling Patterns](#error-handling-patterns)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Common Integration Mistakes](#common-integration-mistakes)
9. [Debugging Tools](#debugging-tools)
10. [Production Checklist](#production-checklist)

---

## Getting Started

### Initial Connection Test

Before integrating, verify basic API connectivity:

```bash
# Test 1: Health check (no auth required)
curl -X GET "https://api.nfl-analytics.com/health"

# Expected: HTTP 200 with {"status": "healthy"}
```

**If health check fails:**

1. **Check network connectivity:**
   ```bash
   ping api.nfl-analytics.com
   nslookup api.nfl-analytics.com
   ```

2. **Verify firewall rules:**
   - Outbound HTTPS (port 443) allowed
   - No corporate proxy blocking requests

3. **Check API status:**
   - Visit: https://status.nfl-analytics.com
   - Subscribe to status updates

### API Key Validation Test

```bash
# Test 2: Authenticated endpoint
curl -X GET "https://api.nfl-analytics.com/models" \
  -H "X-API-Key: $NFL_API_KEY"

# Expected: HTTP 200 with models list
```

**If authentication fails:**

1. **Verify header format:**
   ```bash
   # ❌ WRONG
   -H "API-Key: $NFL_API_KEY"
   -H "ApiKey: $NFL_API_KEY"

   # ✅ CORRECT
   -H "X-API-Key: $NFL_API_KEY"
   ```

2. **Check key length:**
   ```bash
   echo -n "$NFL_API_KEY" | wc -c
   # Expected: 64 characters
   ```

3. **Remove whitespace:**
   ```bash
   # Clean API key
   NFL_API_KEY=$(echo "$NFL_API_KEY" | tr -d ' \n\r')
   ```

### Simple Prediction Test

```bash
# Test 3: Single prediction
curl -X POST "https://api.nfl-analytics.com/predict" \
  -H "X-API-Key: $NFL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
  }'

# Expected: HTTP 200 with prediction
```

**If prediction fails:**

See [Error Catalog](error_catalog.md) for specific error codes and remediation.

---

## Authentication Issues

### Issue 1: 401 Unauthorized - Missing API Key

**Symptoms:**
```json
{
  "error": "Unauthorized",
  "message": "API key required. Include 'X-API-Key' header in request."
}
```

**Diagnosis:**

```python
import requests

# Test if header is being sent
response = requests.post(
    "https://api.nfl-analytics.com/predict",
    headers={"X-API-Key": API_KEY},  # Check this is set
    json=payload
)

# Debug: Print request headers
print("Request headers:", response.request.headers)
# Should show: X-API-Key: your_key_here
```

**Solution:**

```python
# ❌ COMMON MISTAKE - Missing header
response = requests.post(url, json=payload)

# ✅ CORRECT - Include header
headers = {"X-API-Key": os.getenv("NFL_API_KEY")}
response = requests.post(url, headers=headers, json=payload)
```

**R Fix:**
```r
# ❌ WRONG
response <- POST(url, body = payload, encode = "json")

# ✅ CORRECT
response <- POST(
  url,
  add_headers("X-API-Key" = Sys.getenv("NFL_API_KEY")),
  body = payload,
  encode = "json"
)
```

---

### Issue 2: 401 Unauthorized - Invalid API Key

**Symptoms:**
```json
{
  "error": "Unauthorized",
  "message": "Invalid API key"
}
```

**Diagnosis:**

```bash
# Step 1: Verify key format
echo "$NFL_API_KEY" | wc -c
# Expected: 64 characters

# Step 2: Check for whitespace
echo "$NFL_API_KEY" | od -c | head
# Look for \n, \r, \t, spaces

# Step 3: Test key in isolation
curl -X GET "https://api.nfl-analytics.com/models" \
  -H "X-API-Key: $(echo $NFL_API_KEY | tr -d ' \n\r')"
```

**Common Causes:**

1. **Line breaks in .env file:**
   ```bash
   # ❌ WRONG
   NFL_API_KEY="abcd1234...
   ...xyz9876"

   # ✅ CORRECT
   NFL_API_KEY="abcd1234...xyz9876"
   ```

2. **Copy-paste artifacts:**
   ```python
   # ❌ WRONG - Extra whitespace
   API_KEY = "  abcd1234...xyz9876  "

   # ✅ CORRECT - Strip whitespace
   API_KEY = os.getenv("NFL_API_KEY").strip()
   ```

3. **Environment variable not set:**
   ```python
   import os

   # Debug: Check if env var exists
   api_key = os.getenv("NFL_API_KEY")
   if not api_key:
       raise ValueError("NFL_API_KEY environment variable not set")

   print(f"API Key loaded: {api_key[:8]}...")  # Show prefix only
   ```

**Solution:**

```python
# Robust API key loading
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get and validate API key
API_KEY = os.getenv("NFL_API_KEY")
if not API_KEY:
    raise ValueError("NFL_API_KEY not found in environment")

# Strip whitespace
API_KEY = API_KEY.strip()

# Validate length
if len(API_KEY) != 64:
    raise ValueError(f"Invalid API key length: {len(API_KEY)} (expected 64)")

print("✓ API key validated")
```

---

### Issue 3: 403 Forbidden - Expired Key

**Symptoms:**
```json
{
  "error": "Forbidden",
  "message": "API key expired on 2025-11-01. Please rotate your key."
}
```

**Diagnosis:**

```python
# Check key metadata
from datetime import datetime, timedelta

KEY_CREATED_AT = datetime.fromisoformat(os.getenv("KEY_CREATED_AT"))
KEY_AGE = datetime.utcnow() - KEY_CREATED_AT

print(f"Key age: {KEY_AGE.days} days")

if KEY_AGE > timedelta(days=90):
    print("⚠️  Key older than 90 days - rotation recommended")
```

**Solution:**

1. **Request new key:**
   - Email: support@nfl-analytics.com
   - Subject: "API Key Rotation - [Key Name]"

2. **Update environment:**
   ```bash
   # Update .env
   echo "NFL_API_KEY=new_key_here" >> .env
   echo "KEY_CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> .env
   ```

3. **Test new key:**
   ```bash
   source .env
   curl -X GET "https://api.nfl-analytics.com/models" \
     -H "X-API-Key: $NFL_API_KEY"
   ```

4. **Revoke old key after 7 days**

---

## Rate Limiting Challenges

### Issue 4: 429 Too Many Requests

**Symptoms:**
```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Limit: 100 requests per 1 hour(s)."
}
```

**Diagnosis:**

```python
# Monitor rate limit headers
import requests

response = requests.post(url, headers=headers, json=payload)

limit = int(response.headers.get("X-RateLimit-Limit", 0))
remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
reset = int(response.headers.get("X-RateLimit-Reset", 0))

print(f"Rate limit: {remaining}/{limit} remaining")
print(f"Resets at: {datetime.fromtimestamp(reset)}")

# Calculate usage
usage_pct = ((limit - remaining) / limit) * 100
print(f"Usage: {usage_pct:.1f}%")
```

**Solution 1: Implement Exponential Backoff**

```python
import time
import random

def predict_with_backoff(payload, max_retries=3):
    """Make prediction with exponential backoff"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 60))

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    base_delay = min(retry_after, 2 ** attempt * 10)
                    jitter = random.uniform(0, base_delay * 0.1)
                    sleep_time = base_delay + jitter

                    print(f"⏳ Rate limited. Retrying in {sleep_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    print(f"❌ Max retries ({max_retries}) exceeded")
                    raise
            else:
                # Other HTTP error - don't retry
                raise

    raise Exception("Failed after all retries")
```

**Solution 2: Client-Side Rate Limiting**

```python
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    """Client-side rate limiter using sliding window"""

    def __init__(self, limit=100, window_seconds=3600):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = deque()

    def can_make_request(self):
        """Check if we can make a request"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        return len(self.requests) < self.limit

    def wait_if_needed(self):
        """Wait until we can make a request"""
        while not self.can_make_request():
            if self.requests:
                oldest = self.requests[0]
                wait_until = oldest + timedelta(seconds=self.window_seconds)
                wait_seconds = (wait_until - datetime.utcnow()).total_seconds()

                if wait_seconds > 0:
                    print(f"⏳ Rate limit reached. Waiting {wait_seconds:.1f}s...")
                    time.sleep(wait_seconds + 1)
            else:
                break

    def record_request(self):
        """Record a request"""
        self.requests.append(datetime.utcnow())

    def make_request(self, func, *args, **kwargs):
        """Execute function with rate limiting"""
        self.wait_if_needed()
        self.record_request()
        return func(*args, **kwargs)

# Usage
limiter = RateLimiter(limit=100, window_seconds=3600)

for player in players:
    prediction = limiter.make_request(
        predict_player,
        game_id=game.id,
        player_id=player.id,
        stat_type="passing_yards"
    )
```

**Solution 3: Use Batch Endpoint**

```python
# ❌ INEFFICIENT - 100 individual requests
predictions = []
for player in players:
    pred = predict_single(player)  # 100 API calls
    predictions.append(pred)

# ✅ EFFICIENT - 1 batch request
batch_input = [
    {"game_id": game_id, "player_id": p.id, "stat_type": "passing_yards"}
    for p in players
]
predictions = predict_batch(batch_input)  # 1 API call
```

**Solution 4: Request Higher Limit**

Email support@nfl-analytics.com with:
1. Current usage patterns
2. Business justification
3. Required limit (e.g., 500/hour)
4. Willingness to upgrade tier

---

### Issue 5: Batch Size Exceeded

**Symptoms:**
```json
{
  "error": "ValidationError",
  "message": "Batch size exceeds maximum",
  "details": {
    "batch_size": 150,
    "max_batch_size": 100
  }
}
```

**Solution:**

```python
def predict_large_batch(predictions, batch_size=100, delay=5):
    """Split large batch into chunks"""

    results = []
    total = len(predictions)

    for i in range(0, total, batch_size):
        batch = predictions[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} predictions)...")

        try:
            response = predict_batch(batch)
            results.extend(response["predictions"])

            # Rate limiting between batches
            if i + batch_size < total:
                print(f"⏳ Waiting {delay}s before next batch...")
                time.sleep(delay)

        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            # Optionally: continue with next batch or fail fast
            raise

    return results

# Usage
all_predictions = [...]  # 500 predictions
results = predict_large_batch(all_predictions, batch_size=100, delay=5)
# Makes 5 batch requests with 5s delay between each
```

---

## Prediction Response Interpretation

### Issue 6: Understanding Uncertainty Bounds

**Symptoms:**
- Wide confidence intervals (e.g., [150, 400])
- Low confidence scores (<0.6)
- Predictions seem unreliable

**Diagnosis:**

```python
def analyze_prediction_uncertainty(prediction):
    """Analyze prediction uncertainty"""

    pred = prediction["prediction"]
    lower = prediction["lower_bound"]
    upper = prediction["upper_bound"]
    confidence = prediction["confidence"]

    # Calculate interval width
    interval_width = upper - lower

    # Calculate relative uncertainty
    relative_uncertainty = interval_width / pred if pred > 0 else float('inf')

    print(f"Prediction: {pred:.1f}")
    print(f"90% CI: [{lower:.1f}, {upper:.1f}]")
    print(f"Interval width: {interval_width:.1f}")
    print(f"Relative uncertainty: {relative_uncertainty:.1%}")
    print(f"Confidence: {confidence:.1%}")

    # Interpret uncertainty
    if relative_uncertainty > 0.5:
        print("⚠️  HIGH UNCERTAINTY - Wide interval relative to prediction")
    elif relative_uncertainty > 0.3:
        print("⚠️  MODERATE UNCERTAINTY - Typical for player props")
    else:
        print("✓ LOW UNCERTAINTY - Narrow interval")

    return {
        "interval_width": interval_width,
        "relative_uncertainty": relative_uncertainty,
        "uncertainty_category": (
            "high" if relative_uncertainty > 0.5 else
            "moderate" if relative_uncertainty > 0.3 else
            "low"
        )
    }

# Usage
prediction = {
    "prediction": 285.3,
    "lower_bound": 215.1,
    "upper_bound": 355.5,
    "confidence": 0.78
}

analysis = analyze_prediction_uncertainty(prediction)
```

**Interpretation:**

**Confidence Score (0-1):**
- **>0.8** - High confidence (models agree closely)
- **0.6-0.8** - Moderate confidence (typical)
- **<0.6** - Low confidence (models disagree)

**Credible Interval:**
- **90% CI** - 90% probability true value falls within bounds
- **Narrow interval** - Low uncertainty, high precision
- **Wide interval** - High uncertainty, less reliable

**Use Cases by Uncertainty:**

```python
def decision_strategy(prediction, analysis):
    """Betting strategy based on uncertainty"""

    if analysis["uncertainty_category"] == "low":
        return {
            "bet_confidence": "high",
            "suggested_stake": "standard",
            "strategy": "Good value - models agree"
        }
    elif analysis["uncertainty_category"] == "moderate":
        return {
            "bet_confidence": "moderate",
            "suggested_stake": "reduced (50-75%)",
            "strategy": "Typical uncertainty - standard bet"
        }
    else:  # high
        return {
            "bet_confidence": "low",
            "suggested_stake": "minimal or skip",
            "strategy": "High uncertainty - avoid or hedge"
        }

strategy = decision_strategy(prediction, analysis)
print(f"Recommended strategy: {strategy['strategy']}")
```

---

### Issue 7: Model Disagreement

**Symptoms:**
- Individual models have very different predictions
- Large spread in `models_used` array

**Diagnosis:**

```python
def analyze_model_agreement(prediction):
    """Analyze agreement among ensemble models"""

    models = prediction["models_used"]

    # Extract predictions
    model_preds = [m["prediction"] for m in models]

    # Calculate statistics
    mean_pred = sum(model_preds) / len(model_preds)
    max_pred = max(model_preds)
    min_pred = min(model_preds)
    spread = max_pred - min_pred

    # Calculate standard deviation
    variance = sum((p - mean_pred) ** 2 for p in model_preds) / len(model_preds)
    std_dev = variance ** 0.5

    print(f"Model predictions: {[f'{p:.1f}' for p in model_preds]}")
    print(f"Mean: {mean_pred:.1f}")
    print(f"Spread: {spread:.1f} (range: {min_pred:.1f} - {max_pred:.1f})")
    print(f"Std Dev: {std_dev:.1f}")

    # Coefficient of variation
    cv = std_dev / mean_pred if mean_pred > 0 else float('inf')
    print(f"Coefficient of variation: {cv:.1%}")

    # Interpret agreement
    if cv < 0.1:
        print("✓ STRONG AGREEMENT - Models converge")
    elif cv < 0.2:
        print("⚠️  MODERATE AGREEMENT - Typical variance")
    else:
        print("⚠️  WEAK AGREEMENT - Models diverge significantly")

    return {
        "spread": spread,
        "std_dev": std_dev,
        "cv": cv,
        "agreement": (
            "strong" if cv < 0.1 else
            "moderate" if cv < 0.2 else
            "weak"
        )
    }

# Usage
prediction = {
    "prediction": 285.3,
    "models_used": [
        {"model_name": "state_space_v2", "prediction": 290.5},
        {"model_name": "hierarchical_v2", "prediction": 275.2},
        {"model_name": "hybrid_xgboost", "prediction": 288.7},
        {"model_name": "informative_priors_v2", "prediction": 282.1}
    ]
}

agreement = analyze_model_agreement(prediction)
```

**Interpretation:**

**When Models Disagree:**
1. **High spread (>50 yards)** - Different models capture different patterns
2. **Check model weights** - Which models dominate ensemble?
3. **Consider uncertainty** - Wider CI indicates disagreement
4. **Bet conservatively** - Reduce stake or skip bet

---

## Performance Optimization

### Issue 8: Slow Response Times

**Symptoms:**
- Predictions taking >2 seconds
- Timeouts on batch requests

**Diagnosis:**

```python
import time

def measure_latency(func, *args, **kwargs):
    """Measure function latency"""

    start = time.time()
    result = func(*args, **kwargs)
    latency = time.time() - start

    print(f"Latency: {latency*1000:.1f}ms")
    return result, latency

# Test single prediction
prediction, latency = measure_latency(
    predict_single,
    game_id="2024_10_KC_LV",
    player_id="00-0033873",
    stat_type="passing_yards"
)

if latency > 1.0:
    print("⚠️  Slow prediction (>1s)")
```

**Solution 1: Use Connection Pooling**

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create session with connection pooling
session = requests.Session()

# Configure connection pool
adapter = HTTPAdapter(
    pool_connections=10,  # Pool size
    pool_maxsize=20,      # Max connections
    max_retries=Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
)

session.mount("https://", adapter)
session.mount("http://", adapter)

# Reuse session for all requests
def predict_single(game_id, player_id, stat_type):
    response = session.post(  # Use session, not requests
        f"{BASE_URL}/predict",
        headers={"X-API-Key": API_KEY},
        json={
            "game_id": game_id,
            "player_id": player_id,
            "stat_type": stat_type
        },
        timeout=10
    )
    return response.json()

# Performance improvement: ~30-50% faster on repeated requests
```

**Solution 2: Implement Caching**

```python
import redis
import json
from functools import wraps

# Connect to Redis
cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_prediction(ttl=3600):
    """Cache predictions for TTL seconds"""

    def decorator(func):
        @wraps(func)
        def wrapper(game_id, player_id, stat_type):
            # Generate cache key
            cache_key = f"prediction:{game_id}:{player_id}:{stat_type}"

            # Check cache
            cached = cache.get(cache_key)
            if cached:
                print(f"✓ Cache hit: {cache_key}")
                return json.loads(cached)

            # Cache miss - call API
            print(f"✗ Cache miss: {cache_key}")
            prediction = func(game_id, player_id, stat_type)

            # Store in cache
            cache.setex(cache_key, ttl, json.dumps(prediction))

            return prediction

        return wrapper
    return decorator

@cache_prediction(ttl=3600)
def predict_single(game_id, player_id, stat_type):
    # Original API call
    ...

# Performance improvement:
# - Cache hit: <1ms
# - Cache miss: ~500ms (normal API latency)
```

**Solution 3: Parallel Batch Processing**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def predict_parallel(predictions, max_workers=5):
    """Process predictions in parallel"""

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all predictions
        futures = {
            executor.submit(
                predict_single,
                p["game_id"],
                p["player_id"],
                p["stat_type"]
            ): p
            for p in predictions
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Prediction failed: {e}")

    return results

# Performance improvement:
# - Sequential: 100 predictions × 500ms = 50 seconds
# - Parallel (5 workers): 100 predictions / 5 × 500ms = 10 seconds
```

---

### Issue 9: High Memory Usage

**Symptoms:**
- Memory growing unbounded
- Out of memory errors on large batches

**Solution:**

```python
def process_large_dataset(predictions, batch_size=100):
    """Process large dataset with memory management"""

    total = len(predictions)

    for i in range(0, total, batch_size):
        batch = predictions[i:i + batch_size]

        # Process batch
        results = predict_batch(batch)

        # Process results immediately (don't accumulate)
        for result in results["predictions"]:
            # Save to database, file, etc.
            save_prediction(result)

        # Free memory
        del results
        del batch

        # Progress tracking
        progress = min(i + batch_size, total)
        print(f"Progress: {progress}/{total} ({progress/total*100:.1f}%)")

    print("✓ Processing complete")

# Memory stays constant regardless of dataset size
```

---

## Error Handling Patterns

### Issue 10: Unhandled Exceptions

**Solution: Comprehensive Error Handler**

```python
import requests
from requests.exceptions import (
    ConnectionError,
    Timeout,
    HTTPError,
    RequestException
)

def predict_robust(game_id, player_id, stat_type, max_retries=3):
    """Make prediction with comprehensive error handling"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                headers={"X-API-Key": API_KEY},
                json={
                    "game_id": game_id,
                    "player_id": player_id,
                    "stat_type": stat_type
                },
                timeout=10
            )

            # Check HTTP status
            response.raise_for_status()

            # Parse JSON
            return response.json()

        except ConnectionError as e:
            print(f"⚠️  Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise Exception("Failed to connect to API after retries")

        except Timeout as e:
            print(f"⚠️  Request timeout (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise Exception("Request timed out after retries")

        except HTTPError as e:
            status_code = e.response.status_code

            if status_code == 401:
                # Auth error - don't retry
                raise Exception("Authentication failed. Check API key.")

            elif status_code == 403:
                # Forbidden - don't retry
                error_msg = e.response.json().get("message", "Forbidden")
                raise Exception(f"Access denied: {error_msg}")

            elif status_code == 422:
                # Validation error - don't retry
                details = e.response.json().get("details", {})
                raise ValueError(f"Invalid request: {details}")

            elif status_code == 429:
                # Rate limit - retry with backoff
                retry_after = int(e.response.headers.get("Retry-After", 60))
                if attempt < max_retries - 1:
                    print(f"⏳ Rate limited. Retrying in {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    raise Exception("Rate limit exceeded after retries")

            elif status_code >= 500:
                # Server error - retry
                if attempt < max_retries - 1:
                    print(f"⚠️  Server error {status_code}. Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise Exception("Server error persists after retries")

            else:
                # Unknown error - don't retry
                raise Exception(f"Unexpected HTTP error: {status_code}")

        except RequestException as e:
            # Generic request error
            print(f"⚠️  Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    raise Exception("Failed after all retries")

# Usage
try:
    prediction = predict_robust(
        game_id="2024_10_KC_LV",
        player_id="00-0033873",
        stat_type="passing_yards"
    )
    print(f"✓ Prediction: {prediction['prediction']:.1f}")

except ValueError as e:
    print(f"❌ Validation error: {e}")
    # Handle validation errors

except Exception as e:
    print(f"❌ Error: {e}")
    # Handle other errors
```

---

## Monitoring and Alerting

### Issue 11: No Visibility into API Usage

**Solution: Comprehensive Monitoring**

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_usage.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class APIMonitor:
    """Monitor API usage and performance"""

    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0

    def track_request(self, endpoint, status_code, latency):
        """Track API request metrics"""

        self.total_requests += 1
        self.total_latency += latency

        if status_code >= 400:
            self.failed_requests += 1

        # Log request
        logger.info(
            f"API Request: {endpoint} - "
            f"Status: {status_code} - "
            f"Latency: {latency*1000:.1f}ms"
        )

        # Alert on errors
        if status_code >= 500:
            logger.error(f"Server error on {endpoint}: {status_code}")

        # Alert on slow requests
        if latency > 2.0:
            logger.warning(f"Slow request on {endpoint}: {latency*1000:.1f}ms")

    def get_stats(self):
        """Get usage statistics"""

        if self.total_requests == 0:
            return {"total_requests": 0}

        avg_latency = self.total_latency / self.total_requests
        error_rate = self.failed_requests / self.total_requests

        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency * 1000
        }

# Global monitor
monitor = APIMonitor()

def predict_monitored(game_id, player_id, stat_type):
    """Make prediction with monitoring"""

    start_time = time.time()

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": API_KEY},
            json={
                "game_id": game_id,
                "player_id": player_id,
                "stat_type": stat_type
            },
            timeout=10
        )

        latency = time.time() - start_time
        monitor.track_request("/predict", response.status_code, latency)

        response.raise_for_status()
        return response.json()

    except requests.HTTPError as e:
        latency = time.time() - start_time
        monitor.track_request("/predict", e.response.status_code, latency)
        raise

    except Exception as e:
        latency = time.time() - start_time
        monitor.track_request("/predict", 0, latency)
        raise

# Periodic stats reporting
def report_stats():
    """Report usage statistics"""

    stats = monitor.get_stats()

    logger.info("="*50)
    logger.info("API Usage Statistics")
    logger.info(f"  Total requests: {stats['total_requests']}")
    logger.info(f"  Failed requests: {stats.get('failed_requests', 0)}")
    logger.info(f"  Error rate: {stats.get('error_rate', 0):.1%}")
    logger.info(f"  Avg latency: {stats.get('avg_latency_ms', 0):.1f}ms")
    logger.info("="*50)

# Call every hour
import schedule

schedule.every().hour.do(report_stats)
```

---

## Common Integration Mistakes

### Mistake 1: Hardcoded API Keys

**❌ WRONG:**
```python
API_KEY = "abcd1234...xyz9876"  # Hardcoded secret
```

**✅ CORRECT:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NFL_API_KEY")

if not API_KEY:
    raise ValueError("NFL_API_KEY not set in environment")
```

---

### Mistake 2: No Rate Limit Handling

**❌ WRONG:**
```python
# Will fail on 429
for player in players:
    predict(player)
```

**✅ CORRECT:**
```python
# Handles rate limits gracefully
limiter = RateLimiter(limit=100)
for player in players:
    limiter.make_request(predict, player)
```

---

### Mistake 3: No Timeouts

**❌ WRONG:**
```python
# Can hang indefinitely
response = requests.post(url, json=payload)
```

**✅ CORRECT:**
```python
# Fails fast on slow responses
response = requests.post(url, json=payload, timeout=10)
```

---

### Mistake 4: Ignoring Error Responses

**❌ WRONG:**
```python
response = requests.post(url, json=payload)
prediction = response.json()  # Crashes on error
```

**✅ CORRECT:**
```python
response = requests.post(url, json=payload)
response.raise_for_status()  # Check status first
prediction = response.json()
```

---

### Mistake 5: Not Validating Inputs

**❌ WRONG:**
```python
# No validation
predict(user_input_game_id, user_input_player_id, user_input_stat_type)
```

**✅ CORRECT:**
```python
def validate_and_predict(game_id, player_id, stat_type):
    # Validate game_id format
    if not re.match(r'^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$', game_id):
        raise ValueError(f"Invalid game_id format: {game_id}")

    # Validate player_id format
    if not re.match(r'^\d{2}-\d{7}$', player_id):
        raise ValueError(f"Invalid player_id format: {player_id}")

    # Validate stat_type
    valid_stats = ['passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds']
    if stat_type not in valid_stats:
        raise ValueError(f"Invalid stat_type: {stat_type}")

    return predict(game_id, player_id, stat_type)
```

---

## Debugging Tools

### cURL Command Generator

```python
def generate_curl_command(endpoint, method="POST", payload=None):
    """Generate curl command for debugging"""

    cmd = f"curl -X {method} \"{BASE_URL}{endpoint}\" \\\n"
    cmd += f"  -H \"X-API-Key: $NFL_API_KEY\" \\\n"
    cmd += f"  -H \"Content-Type: application/json\""

    if payload:
        import json
        cmd += f" \\\n  -d '{json.dumps(payload)}'"

    return cmd

# Usage
payload = {
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
}

curl_cmd = generate_curl_command("/predict", "POST", payload)
print("Test with curl:")
print(curl_cmd)
```

### Request/Response Logger

```python
import json

def log_request_response(response):
    """Log full request and response for debugging"""

    print("="*50)
    print("REQUEST")
    print("="*50)
    print(f"Method: {response.request.method}")
    print(f"URL: {response.request.url}")
    print(f"Headers:")
    for k, v in response.request.headers.items():
        if k.lower() != 'x-api-key':  # Don't log API key
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: ***REDACTED***")
    print(f"Body: {response.request.body}")

    print("\n" + "="*50)
    print("RESPONSE")
    print("="*50)
    print(f"Status: {response.status_code}")
    print(f"Headers:")
    for k, v in response.headers.items():
        print(f"  {k}: {v}")

    try:
        print(f"Body: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Body: {response.text}")

    print("="*50)

# Usage
response = requests.post(url, headers=headers, json=payload)
log_request_response(response)
```

---

## Production Checklist

### Pre-Launch Checklist

- [ ] **Authentication**
  - [ ] API keys stored in environment variables (not hardcoded)
  - [ ] Keys rotated within last 90 days
  - [ ] Separate keys for dev/staging/production

- [ ] **Error Handling**
  - [ ] Exponential backoff for rate limits
  - [ ] Timeout set on all requests (10s recommended)
  - [ ] Comprehensive try/except blocks
  - [ ] Graceful degradation on errors

- [ ] **Performance**
  - [ ] Connection pooling enabled
  - [ ] Caching implemented (Redis recommended)
  - [ ] Batch endpoint used where applicable
  - [ ] Parallel processing for large datasets

- [ ] **Monitoring**
  - [ ] Logging configured
  - [ ] Metrics tracked (requests, errors, latency)
  - [ ] Alerts configured for errors
  - [ ] Rate limit usage monitored

- [ ] **Testing**
  - [ ] Integration tests passing
  - [ ] Load testing completed
  - [ ] Error scenarios tested (401, 429, 500)
  - [ ] Latency benchmarks established

- [ ] **Documentation**
  - [ ] API integration documented
  - [ ] Runbook created for on-call
  - [ ] Error handling procedures documented

---

## Support

**Documentation:** https://docs.nfl-analytics.com
**API Status:** https://status.nfl-analytics.com
**Support Email:** support@nfl-analytics.com

**When Requesting Support:**
1. API endpoint and method
2. Request payload (sanitize API keys)
3. Full error response
4. Timestamp
5. Steps to reproduce
6. curl command if possible

**Response Times:**
- Critical (outage): 1 hour
- High (blocking issue): 4 hours
- Normal (questions): 24 hours
