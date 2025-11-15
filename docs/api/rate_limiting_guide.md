# Rate Limiting Guide

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

The NFL Analytics Prediction API implements rate limiting to ensure fair usage and system stability. All requests are subject to rate limits based on API key and endpoint.

## Table of Contents

1. [Rate Limit Policies](#rate-limit-policies)
2. [Checking Your Quota](#checking-your-quota)
3. [When Limits Are Exceeded](#when-limits-are-exceeded)
4. [Requesting Limit Increases](#requesting-limit-increases)
5. [Efficient Batching Strategies](#efficient-batching-strategies)
6. [Best Practices](#best-practices)

---

## Rate Limit Policies

### Default Limits

| Endpoint | Limit | Window | Notes |
|----------|-------|--------|-------|
| `/predict` | 100 requests | 1 hour | Ensemble prediction (most expensive) |
| `/models` | 1,000 requests | 1 hour | Read-only metadata |
| `/health` | 10,000 requests | 1 hour | Monitoring (very cheap) |
| Default (all others) | 100 requests | 1 hour | Conservative default |

### Daily Limits

In addition to hourly limits, daily limits apply:

| Tier | Hourly Limit | Daily Limit |
|------|-------------|-------------|
| **Free** | 100 | 1,000 |
| **Standard** | 500 | 5,000 |
| **Premium** | 2,000 | 20,000 |
| **Enterprise** | Custom | Custom |

### Sliding Window Algorithm

Rate limits use a **sliding window** algorithm (not fixed window):

- **Fixed Window (less accurate):**
  ```
  00:00-01:00 → 100 requests allowed
  01:00-02:00 → 100 requests allowed
  ```
  Problem: 200 requests possible in 1 minute (00:59:30 - 01:00:30)

- **Sliding Window (more accurate):**
  ```
  12:00:00 → Request 1
  12:30:00 → Request 50
  13:00:00 → Check last 60 minutes (allows 50 more)
  ```
  Guarantees: No more than 100 requests in ANY 60-minute window

### Rate Limit Buckets

Limits are tracked independently per:

1. **API Key** - Each key has its own quota
2. **Endpoint** - Different limits per endpoint
3. **Time Window** - Rolling 1-hour window

Example:
- API Key `A` can make 100 requests to `/predict` per hour
- API Key `B` can also make 100 requests to `/predict` per hour
- Same API Key can make 1,000 requests to `/models` per hour

---

## Checking Your Quota

### Response Headers

Every API response includes rate limit headers:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1731678000
```

| Header | Description | Example |
|--------|-------------|---------|
| `X-RateLimit-Limit` | Total requests allowed in window | 100 |
| `X-RateLimit-Remaining` | Requests remaining in current window | 95 |
| `X-RateLimit-Reset` | Unix timestamp when limit resets | 1731678000 |

### Parsing Headers

#### Python

```python
import requests
from datetime import datetime

response = requests.get(
    "https://api.nfl-analytics.com/models",
    headers={"X-API-Key": API_KEY}
)

# Extract rate limit info
limit = int(response.headers.get("X-RateLimit-Limit", 0))
remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))

# Convert reset timestamp to datetime
reset_time = datetime.fromtimestamp(reset_timestamp)

print(f"Rate Limit: {remaining}/{limit} remaining")
print(f"Resets at: {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate percentage used
usage_pct = ((limit - remaining) / limit) * 100
print(f"Usage: {usage_pct:.1f}%")

# Warn if approaching limit
if remaining < limit * 0.1:  # Less than 10% remaining
    print("⚠️  WARNING: Approaching rate limit!")
```

#### R

```r
library(httr)

response <- GET(
  "https://api.nfl-analytics.com/models",
  add_headers("X-API-Key" = API_KEY)
)

# Extract rate limit headers
limit <- as.numeric(headers(response)$`x-ratelimit-limit`)
remaining <- as.numeric(headers(response)$`x-ratelimit-remaining`)
reset_timestamp <- as.numeric(headers(response)$`x-ratelimit-reset`)

# Convert reset timestamp
reset_time <- as.POSIXct(reset_timestamp, origin = "1970-01-01")

cat(sprintf("Rate Limit: %d/%d remaining\n", remaining, limit))
cat(sprintf("Resets at: %s\n", format(reset_time, "%Y-%m-%d %H:%M:%S")))

# Warn if approaching limit
if (remaining < limit * 0.1) {
  warning("Approaching rate limit!")
}
```

#### JavaScript

```javascript
const axios = require('axios');

async function checkRateLimit() {
  const response = await axios.get(
    'https://api.nfl-analytics.com/models',
    {
      headers: { 'X-API-Key': process.env.NFL_API_KEY }
    }
  );

  const limit = parseInt(response.headers['x-ratelimit-limit']);
  const remaining = parseInt(response.headers['x-ratelimit-remaining']);
  const resetTimestamp = parseInt(response.headers['x-ratelimit-reset']);

  const resetTime = new Date(resetTimestamp * 1000);

  console.log(`Rate Limit: ${remaining}/${limit} remaining`);
  console.log(`Resets at: ${resetTime.toLocaleString()}`);

  // Warn if approaching limit
  if (remaining < limit * 0.1) {
    console.warn('⚠️  WARNING: Approaching rate limit!');
  }

  return { limit, remaining, resetTime };
}
```

---

## When Limits Are Exceeded

### 429 Response

When rate limit is exceeded, API returns HTTP 429:

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

### Retry-After Header

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 3456
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1731678000
```

Use `Retry-After` header to determine when to retry.

### Handling 429 Errors

#### Exponential Backoff (Recommended)

```python
import time
import requests
from requests.exceptions import HTTPError

def predict_with_backoff(payload, max_retries=3, base_delay=1):
    """Make prediction with exponential backoff on rate limit"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.nfl-analytics.com/predict",
                headers={"X-API-Key": API_KEY},
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited
                retry_after = int(e.response.headers.get("Retry-After", 60))

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(retry_after, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, delay * 0.1)
                    sleep_time = delay + jitter

                    print(f"Rate limited. Retrying in {sleep_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    print(f"Max retries ({max_retries}) exceeded")
                    raise
            else:
                # Other HTTP error
                raise

    raise Exception("Failed after all retries")


# Usage
try:
    prediction = predict_with_backoff({
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0033873",
        "stat_type": "passing_yards"
    })
    print(f"Prediction: {prediction['prediction']:.1f}")
except Exception as e:
    print(f"Error: {e}")
```

#### Respect Retry-After

```python
def respect_retry_after(response):
    """Wait for Retry-After duration if rate limited"""

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        reset_at = datetime.fromtimestamp(
            int(response.headers.get("X-RateLimit-Reset", 0))
        )

        print(f"⚠️  Rate limit exceeded")
        print(f"   Limit resets at: {reset_at.strftime('%H:%M:%S')}")
        print(f"   Waiting {retry_after} seconds...")

        time.sleep(retry_after)
        return True

    return False
```

---

## Requesting Limit Increases

### When to Request

Request limit increase if:

- ✅ Consistently hitting limit during normal usage
- ✅ Business requirements justify higher volume
- ✅ Willing to upgrade to paid tier

### How to Request

**Email:** support@nfl-analytics.com
**Subject:** Rate Limit Increase Request

**Include:**
1. **API Key Prefix** (first 8 characters)
2. **Current Limit** (e.g., 100/hour)
3. **Requested Limit** (e.g., 500/hour)
4. **Justification:**
   - Current usage patterns
   - Business use case
   - Expected request volume
5. **Tier Upgrade** (if applicable)

### Approval Process

- **Response Time:** 1-2 business days
- **Approval Criteria:**
  - Valid business justification
  - No abuse history
  - Compliance with terms of service
- **Implementation:** Limits updated within 4 hours of approval

### Temporary Increases

For one-time events (e.g., live games, special analysis):

**Email:** support@nfl-analytics.com
**Subject:** Temporary Rate Limit Increase

**Include:**
- Event date/time
- Expected duration
- Estimated request volume
- Justification

Approval typically granted within 24 hours.

---

## Efficient Batching Strategies

### Batch Prediction Requests

Instead of making many individual requests, batch predictions when possible.

#### Strategy 1: Pre-compute and Cache

```python
import redis
import json

# Connect to Redis cache
cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_or_predict(game_id, player_id, stat_type):
    """Get prediction from cache or API"""

    # Check cache first
    cache_key = f"prediction:{game_id}:{player_id}:{stat_type}"
    cached = cache.get(cache_key)

    if cached:
        print(f"Cache hit: {cache_key}")
        return json.loads(cached)

    # Cache miss - call API
    print(f"Cache miss: {cache_key}")
    response = requests.post(
        "https://api.nfl-analytics.com/predict",
        headers={"X-API-Key": API_KEY},
        json={
            "game_id": game_id,
            "player_id": player_id,
            "stat_type": stat_type
        }
    )

    if response.status_code == 200:
        prediction = response.json()

        # Cache for 1 hour
        cache.setex(cache_key, 3600, json.dumps(prediction))

        return prediction
    else:
        raise Exception(f"API error: {response.status_code}")


# Usage - only makes API call once per unique combination
predictions = []
for player in ["00-0033873", "00-0034857"]:
    for stat in ["passing_yards", "passing_tds"]:
        pred = get_or_predict("2024_10_KC_LV", player, stat)
        predictions.append(pred)

print(f"Made {len(predictions)} predictions (some from cache)")
```

#### Strategy 2: Rate Limit Aware Queue

```python
import time
from collections import deque
from datetime import datetime, timedelta

class RateLimitedQueue:
    """Queue that respects API rate limits"""

    def __init__(self, limit=100, window_seconds=3600):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = deque()

    def can_make_request(self):
        """Check if we can make a request without exceeding limit"""

        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Remove old requests outside window
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        return len(self.requests) < self.limit

    def wait_if_needed(self):
        """Wait until we can make a request"""

        while not self.can_make_request():
            # Calculate how long to wait
            if self.requests:
                oldest = self.requests[0]
                wait_until = oldest + timedelta(seconds=self.window_seconds)
                wait_seconds = (wait_until - datetime.utcnow()).total_seconds()

                if wait_seconds > 0:
                    print(f"Rate limit reached. Waiting {wait_seconds:.1f}s...")
                    time.sleep(wait_seconds + 1)  # +1 for safety
            else:
                break

    def make_request(self, func, *args, **kwargs):
        """Execute function with rate limiting"""

        self.wait_if_needed()
        self.requests.append(datetime.utcnow())
        return func(*args, **kwargs)


# Usage
queue = RateLimitedQueue(limit=100, window_seconds=3600)

predictions = []
for game in games:
    for player in players:
        prediction = queue.make_request(
            predict_player,
            game_id=game.id,
            player_id=player.id,
            stat_type="passing_yards"
        )
        predictions.append(prediction)

print(f"Completed {len(predictions)} predictions without rate limit errors")
```

#### Strategy 3: Parallel Processing with Limit

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class RateLimitedExecutor:
    """Thread pool executor with rate limiting"""

    def __init__(self, max_workers=5, requests_per_second=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.requests_per_second = requests_per_second
        self.lock = threading.Lock()
        self.last_request_time = None

    def submit(self, func, *args, **kwargs):
        """Submit task with rate limiting"""

        def rate_limited_func():
            with self.lock:
                if self.last_request_time:
                    elapsed = time.time() - self.last_request_time
                    min_interval = 1.0 / self.requests_per_second

                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)

                self.last_request_time = time.time()

            return func(*args, **kwargs)

        return self.executor.submit(rate_limited_func)


# Usage - 5 workers, max 2 requests/second
executor = RateLimitedExecutor(max_workers=5, requests_per_second=2)

# Submit all prediction tasks
futures = []
for game in games:
    for player in players:
        future = executor.submit(
            predict_player,
            game_id=game.id,
            player_id=player.id,
            stat_type="passing_yards"
        )
        futures.append(future)

# Collect results
predictions = []
for future in as_completed(futures):
    try:
        prediction = future.result()
        predictions.append(prediction)
    except Exception as e:
        print(f"Prediction failed: {e}")

print(f"Completed {len(predictions)} predictions")
```

---

## Best Practices

### ✅ DO

1. **Monitor rate limit headers**
   - Check `X-RateLimit-Remaining` after each request
   - Stop making requests when close to limit

2. **Implement exponential backoff**
   - Respect `Retry-After` header
   - Add jitter to avoid thundering herd

3. **Cache predictions**
   - Use Redis/Memcached for frequently accessed predictions
   - Set appropriate TTL (e.g., 1 hour for live games)

4. **Batch requests during off-peak**
   - Pre-compute predictions overnight
   - Avoid rush hour (game time ± 2 hours)

5. **Use rate-limited queues**
   - Implement client-side rate limiting
   - Prevent 429 errors proactively

6. **Request limit increases proactively**
   - Don't wait until you're blocked
   - Plan for peak usage (playoffs, primetime games)

### ❌ DON'T

1. **Don't ignore 429 responses**
   - Retrying immediately will fail
   - Respect `Retry-After` header

2. **Don't make redundant requests**
   - Cache predictions that haven't changed
   - Avoid polling for same prediction

3. **Don't use tight polling loops**
   - Use webhooks/SSE if available
   - Minimum 5-second intervals for polling

4. **Don't share API keys across teams**
   - Each team should have own key
   - Easier to track usage and set appropriate limits

5. **Don't hammer API on errors**
   - Implement circuit breaker pattern
   - Back off on repeated failures

### Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit tripped, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for API calls"""

    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""

        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if (datetime.utcnow() - self.last_failure_time).total_seconds() > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)

            # Success - reset circuit
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.utcnow()

            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"Circuit breaker OPEN after {self.failures} failures")

            raise
```

---

## Support

**Email:** support@nfl-analytics.com
**Documentation:** https://docs.nfl-analytics.com
**Rate Limit Dashboard:** https://dashboard.nfl-analytics.com/rate-limits

**Monitoring:**
- Real-time usage: https://dashboard.nfl-analytics.com/api-keys/{key_id}
- Historical usage: Available on request
