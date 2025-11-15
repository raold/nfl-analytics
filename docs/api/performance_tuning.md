# API Performance Tuning Guide

**Version:** 1.1.0
**Last Updated:** November 15, 2025
**Status:** Production

## Overview

Comprehensive guide to optimizing NFL Analytics API performance. Target: <500ms p95 latency for predictions, >100 req/sec throughput under load.

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Latency Optimization](#latency-optimization)
3. [Connection Pooling](#connection-pooling)
4. [Request Batching](#request-batching)
5. [Caching Strategies](#caching-strategies)
6. [Parallel Request Handling](#parallel-request-handling)
7. [Database Query Optimization](#database-query-optimization)
8. [Monitoring and Profiling](#monitoring-and-profiling)

---

## Performance Targets

### Latency Benchmarks (95th Percentile)

| Endpoint | Current (p95) | Target (p95) | Optimized (p95) |
|----------|---------------|--------------|-----------------|
| GET /health | 35ms | <50ms | 15ms |
| GET /models | 78ms | <100ms | 45ms |
| POST /predict | 425ms | <500ms | 280ms |
| POST /predict/batch (10) | 2,100ms | <2,500ms | 1,400ms |
| POST /predict/batch (100) | 4,800ms | <5,000ms | 3,200ms |

**Optimization Results:** 30-40% latency reduction achievable through connection pooling + caching.

### Throughput Targets

| Configuration | Requests/Second | Use Case |
|---------------|-----------------|----------|
| Single client (sequential) | 2-3 | Testing, small workloads |
| Single client (pooled) | 8-12 | Production client |
| 5 parallel clients (pooled) | 35-50 | High-volume backtesting |
| 10 parallel clients (pooled) | 60-85 | Maximum throughput |

**Note:** Rate limits still apply. High throughput requires increased rate limits.

---

## Latency Optimization

### 1. Minimize Network Overhead

#### Use Connection Pooling (1.5x speedup)

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ❌ BAD: New connection per request
def predict_slow(player_id, game_id):
    response = requests.post(url, json=payload)  # Creates new TCP connection each time
    return response.json()

# ✅ GOOD: Connection pooling
session = requests.Session()

retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    backoff_factor=1
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,  # Keep 10 connections alive
    pool_maxsize=20       # Maximum 20 connections in pool
)

session.mount("https://", adapter)
session.mount("http://", adapter)

# Now all requests reuse connections
def predict_fast(player_id, game_id):
    response = session.post(url, json=payload)  # Reuses existing connection
    return response.json()
```

**Performance Impact:**
- First request: ~500ms (includes TCP handshake + SSL)
- Subsequent requests: ~300ms (connection reused)
- **Speedup:** 1.5-2x for high-frequency requests

#### Set Appropriate Timeouts

```python
# ❌ BAD: No timeout (hangs indefinitely)
response = requests.post(url, json=payload)

# ❌ BAD: Timeout too short (fails prematurely)
response = requests.post(url, json=payload, timeout=1)

# ✅ GOOD: Appropriate timeouts
TIMEOUTS = {
    "/health": 1,           # Health check: 1 second
    "/predict": 10,         # Single prediction: 10 seconds
    "/predict/batch": 30,   # Batch: 30 seconds
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=payload,
    timeout=TIMEOUTS["/predict"]
)
```

### 2. Optimize Request Payloads

#### Minimize JSON Size

```python
# ❌ BAD: Verbose payload (unnecessary fields)
payload = {
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards",
    "extra_field_1": None,
    "extra_field_2": None,
    "verbose_mode": False
}

# ✅ GOOD: Minimal payload
payload = {
    "game_id": "2024_10_KC_LV",
    "player_id": "00-0033873",
    "stat_type": "passing_yards"
}
```

**Impact:** Reduces payload size by 30-40%, negligible latency improvement but cleaner code.

### 3. Enable HTTP/2 (Future)

**Note:** API currently uses HTTP/1.1. HTTP/2 support planned for v1.2.

HTTP/2 benefits:
- Multiplexing (multiple requests over single connection)
- Header compression (HPACK)
- Server push (proactive resource delivery)

**Expected improvement:** 20-30% latency reduction for batch requests.

---

## Connection Pooling

### Configuration Recommendations

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

def create_optimized_session(
    pool_connections: int = 10,
    pool_maxsize: int = 20,
    max_retries: int = 3
) -> requests.Session:
    """
    Create optimized session with connection pooling.

    Args:
        pool_connections: Connections to keep alive (default: 10)
        pool_maxsize: Maximum pool size (default: 20)
        max_retries: Retry attempts (default: 3)

    Returns:
        Configured requests.Session
    """
    session = requests.Session()

    # Retry strategy
    retry_strategy = Retry(
        total=max_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        backoff_factor=1,
        respect_retry_after_header=True
    )

    # Connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        pool_block=False  # Don't block if pool full, create new connection
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

# Usage
session = create_optimized_session()

# All requests reuse connections
for player in players:
    response = session.post(url, json={"player_id": player})
```

### Tuning Parameters

| Parameter | Low Volume | Medium Volume | High Volume |
|-----------|------------|---------------|-------------|
| `pool_connections` | 2-5 | 10-20 | 20-50 |
| `pool_maxsize` | 5-10 | 20-40 | 50-100 |
| Use case | <10 req/min | 10-100 req/min | >100 req/min |

**Rule of thumb:** `pool_maxsize` = 2 × `pool_connections`

---

## Request Batching

### Batch Size Optimization

```python
import time
from typing import List, Dict

def benchmark_batch_size(predictions: List[Dict], batch_sizes: List[int]):
    """Find optimal batch size for your workload"""

    results = []

    for batch_size in batch_sizes:
        # Time batched requests
        start = time.time()

        chunks = [predictions[i:i+batch_size] for i in range(0, len(predictions), batch_size)]

        for chunk in chunks:
            response = requests.post(
                f"{BASE_URL}/predict/batch",
                headers={"X-API-Key": API_KEY},
                json={"predictions": chunk}
            )

        elapsed = time.time() - start

        results.append({
            "batch_size": batch_size,
            "total_time": elapsed,
            "time_per_prediction": elapsed / len(predictions),
            "throughput": len(predictions) / elapsed
        })

        print(f"Batch size {batch_size:3d}: {elapsed:.2f}s ({len(predictions)/elapsed:.1f} pred/sec)")

    return results

# Test different batch sizes
predictions = generate_test_predictions(1000)  # 1,000 predictions
batch_sizes = [10, 25, 50, 75, 100]

results = benchmark_batch_size(predictions, batch_sizes)

# Find optimal batch size
optimal = max(results, key=lambda x: x['throughput'])
print(f"\nOptimal batch size: {optimal['batch_size']} ({optimal['throughput']:.1f} pred/sec)")
```

**Typical Results:**

| Batch Size | Throughput (pred/sec) | Notes |
|------------|----------------------|-------|
| 1 (individual) | 2-3 | No batching |
| 10 | 15-20 | Good for small jobs |
| 25 | 25-35 | Balanced |
| 50 | 30-45 | High throughput |
| 100 | 25-40 | Optimal for large jobs |

**Recommendation:** Use batch size 50-100 for maximum throughput.

### Adaptive Batching

```python
class AdaptiveBatcher:
    """Dynamically adjust batch size based on latency"""

    def __init__(self, initial_batch_size: int = 50, target_latency: float = 3.0):
        self.batch_size = initial_batch_size
        self.target_latency = target_latency  # Target: 3 seconds per batch
        self.latencies = []

    def process_batch(self, chunk: List[Dict]) -> List[Dict]:
        """Process batch and measure latency"""
        start = time.time()

        response = requests.post(
            f"{BASE_URL}/predict/batch",
            headers={"X-API-Key": API_KEY},
            json={"predictions": chunk},
            timeout=30
        )

        latency = time.time() - start
        self.latencies.append(latency)

        # Adjust batch size
        self._adjust_batch_size(latency)

        return response.json()['predictions']

    def _adjust_batch_size(self, latency: float):
        """Adjust batch size based on latency"""
        if latency > self.target_latency * 1.2:
            # Too slow, reduce batch size
            self.batch_size = max(10, int(self.batch_size * 0.8))
            print(f"Batch too slow ({latency:.2f}s), reducing to {self.batch_size}")
        elif latency < self.target_latency * 0.8:
            # Fast, increase batch size
            self.batch_size = min(100, int(self.batch_size * 1.2))
            print(f"Batch fast ({latency:.2f}s), increasing to {self.batch_size}")

    def predict_all(self, predictions: List[Dict]) -> List[Dict]:
        """Process all predictions with adaptive batching"""
        results = []
        i = 0

        while i < len(predictions):
            chunk = predictions[i:i + self.batch_size]
            batch_results = self.process_batch(chunk)
            results.extend(batch_results)
            i += self.batch_size

        return results

# Usage
batcher = AdaptiveBatcher(initial_batch_size=50, target_latency=3.0)
results = batcher.predict_all(predictions)

print(f"\nFinal batch size: {batcher.batch_size}")
print(f"Average latency: {sum(batcher.latencies)/len(batcher.latencies):.2f}s")
```

---

## Caching Strategies

### 1. In-Memory Caching (100x speedup for repeated requests)

```python
from functools import lru_cache
import hashlib
import json

# Simple LRU cache (Python built-in)
@lru_cache(maxsize=1000)
def predict_cached(game_id: str, player_id: str, stat_type: str) -> Dict:
    """Cache predictions in memory (LRU eviction)"""
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"X-API-Key": API_KEY},
        json={
            "game_id": game_id,
            "player_id": player_id,
            "stat_type": stat_type
        }
    )
    return response.json()

# Usage
prediction1 = predict_cached("2024_10_KC_LV", "00-0033873", "passing_yards")  # API call
prediction2 = predict_cached("2024_10_KC_LV", "00-0033873", "passing_yards")  # Cached (instant)
```

### 2. Redis Caching (Persistent, Multi-Process)

```python
import redis
import json
import hashlib

class RedisCache:
    """Redis-backed prediction cache"""

    def __init__(self, redis_host: str = "localhost", ttl: int = 3600):
        self.redis = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.ttl = ttl  # 1 hour TTL

    def _make_key(self, game_id: str, player_id: str, stat_type: str) -> str:
        """Generate cache key"""
        key_str = f"prediction:{game_id}:{player_id}:{stat_type}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, game_id: str, player_id: str, stat_type: str) -> Optional[Dict]:
        """Get cached prediction"""
        key = self._make_key(game_id, player_id, stat_type)
        cached = self.redis.get(key)

        if cached:
            return json.loads(cached)
        return None

    def set(self, game_id: str, player_id: str, stat_type: str, prediction: Dict):
        """Cache prediction"""
        key = self._make_key(game_id, player_id, stat_type)
        self.redis.setex(key, self.ttl, json.dumps(prediction))

    def predict(self, game_id: str, player_id: str, stat_type: str) -> Dict:
        """Get prediction with caching"""
        # Check cache
        cached = self.get(game_id, player_id, stat_type)
        if cached:
            print("Cache hit")
            return cached

        # Call API
        print("Cache miss - calling API")
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": API_KEY},
            json={
                "game_id": game_id,
                "player_id": player_id,
                "stat_type": stat_type
            }
        )
        prediction = response.json()

        # Cache result
        self.set(game_id, player_id, stat_type, prediction)

        return prediction

# Usage
cache = RedisCache(ttl=3600)  # 1-hour cache

prediction = cache.predict("2024_10_KC_LV", "00-0033873", "passing_yards")
# Subsequent calls instant (from cache)
```

### Cache Invalidation Strategy

**When to invalidate cache:**
1. Game starts (predictions may change with latest data)
2. Model refresh (`POST /admin/refresh`)
3. Manual cache clear for specific game/player

```python
class SmartCache(RedisCache):
    """Cache with intelligent invalidation"""

    def invalidate_game(self, game_id: str):
        """Invalidate all predictions for a game"""
        pattern = f"prediction:{game_id}:*"
        keys = self.redis.keys(pattern)

        if keys:
            self.redis.delete(*keys)
            print(f"Invalidated {len(keys)} predictions for game {game_id}")

    def invalidate_player(self, player_id: str):
        """Invalidate all predictions for a player"""
        pattern = f"prediction:*:{player_id}:*"
        keys = self.redis.keys(pattern)

        if keys:
            self.redis.delete(*keys)
            print(f"Invalidated {len(keys)} predictions for player {player_id}")

    def invalidate_all(self):
        """Clear all cached predictions"""
        keys = self.redis.keys("prediction:*")
        if keys:
            self.redis.delete(*keys)
            print(f"Invalidated all {len(keys)} cached predictions")

# Usage
cache = SmartCache()

# Cache predictions
cache.predict("2024_10_KC_LV", "00-0033873", "passing_yards")

# Game started, invalidate all predictions for game
cache.invalidate_game("2024_10_KC_LV")
```

### Cache Performance Metrics

```python
class CacheMonitor(RedisCache):
    """Monitor cache hit rate"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hits = 0
        self.misses = 0

    def get(self, *args, **kwargs):
        cached = super().get(*args, **kwargs)

        if cached:
            self.hits += 1
        else:
            self.misses += 1

        return cached

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "hit_rate_percent": f"{hit_rate * 100:.1f}%"
        }

# Usage
cache = CacheMonitor()

# Make 100 requests
for _ in range(100):
    cache.predict("2024_10_KC_LV", "00-0033873", "passing_yards")

stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}")
# Output: Cache hit rate: 99.0% (99 hits, 1 miss)
```

---

## Parallel Request Handling

### Thread Pool Optimization

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def optimal_worker_count() -> int:
    """
    Calculate optimal number of worker threads.

    Rule of thumb: 2-5 workers per CPU core for I/O-bound tasks
    """
    import os
    cpu_count = os.cpu_count() or 4
    return min(cpu_count * 3, 20)  # Max 20 workers

def predict_parallel_optimized(
    predictions: List[Dict],
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Parallel predictions with optimal worker count.

    Args:
        predictions: List of prediction requests
        max_workers: Worker threads (default: auto-calculated)

    Returns:
        List of prediction results
    """
    if max_workers is None:
        max_workers = optimal_worker_count()

    print(f"Using {max_workers} worker threads")

    results = []

    def make_prediction(pred: Dict) -> Dict:
        """Worker function"""
        try:
            response = session.post(  # Use shared session
                f"{BASE_URL}/predict",
                json=pred,
                timeout=10
            )
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Create shared session for all workers
    session = create_optimized_session()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(make_prediction, pred): pred for pred in predictions}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    finally:
        session.close()

    return results

# Usage
results = predict_parallel_optimized(predictions)  # Auto-tuned worker count
```

### Process Pool (CPU-Bound Workloads)

**Note:** For API requests (I/O-bound), use ThreadPool. ProcessPool only for CPU-intensive post-processing.

```python
from concurrent.futures import ProcessPoolExecutor

def process_prediction_cpu_intensive(prediction: Dict) -> Dict:
    """
    CPU-intensive prediction processing.

    Example: Monte Carlo simulation, complex calculations
    """
    import numpy as np

    # Simulate CPU-intensive work
    mean = prediction['prediction']
    std = (prediction['upper_bound'] - prediction['lower_bound']) / (2 * 1.645)

    # Monte Carlo simulation (CPU-intensive)
    samples = np.random.normal(mean, std, 10000)
    percentiles = np.percentile(samples, [5, 25, 50, 75, 95])

    return {
        **prediction,
        "monte_carlo": {
            "p5": percentiles[0],
            "p25": percentiles[1],
            "p50": percentiles[2],
            "p75": percentiles[3],
            "p95": percentiles[4]
        }
    }

def parallel_cpu_processing(predictions: List[Dict]) -> List[Dict]:
    """Process predictions with ProcessPool"""
    import os

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_prediction_cpu_intensive, predictions))

    return results

# Usage
# 1. Get predictions (I/O-bound - use ThreadPool)
predictions = predict_parallel_optimized(prediction_requests)

# 2. Process predictions (CPU-bound - use ProcessPool)
processed = parallel_cpu_processing(predictions)
```

---

## Database Query Optimization

### Client-Side Preparation

**Note:** Database queries happen server-side, but you can optimize by reducing redundant requests.

```python
def batch_by_game(predictions: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group predictions by game_id for efficient batching.

    Reduces API calls by batching same-game predictions.
    """
    from collections import defaultdict

    by_game = defaultdict(list)

    for pred in predictions:
        by_game[pred['game_id']].append(pred)

    return dict(by_game)

# Usage
predictions = [
    {"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"},
    {"game_id": "2024_10_KC_LV", "player_id": "00-0034857", "stat_type": "rushing_yards"},
    {"game_id": "2024_10_BUF_IND", "player_id": "00-0036389", "stat_type": "passing_yards"},
    # ... more
]

grouped = batch_by_game(predictions)

for game_id, game_preds in grouped.items():
    # Batch predictions for same game
    results = predict_batch(game_preds)
    print(f"Game {game_id}: {len(results)} predictions")
```

---

## Monitoring and Profiling

### Request Timing

```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimedClient:
    """API client with request timing"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = create_optimized_session()
        self.timings = []

    def predict(self, game_id: str, player_id: str, stat_type: str) -> Dict:
        """Timed prediction"""
        start = time.time()

        response = self.session.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": self.api_key},
            json={
                "game_id": game_id,
                "player_id": player_id,
                "stat_type": stat_type
            }
        )

        elapsed = time.time() - start

        self.timings.append({
            "endpoint": "/predict",
            "latency": elapsed,
            "status": response.status_code
        })

        logger.info(
            f"Prediction: {player_id} {stat_type} @ {game_id} "
            f"({response.status_code}, {elapsed*1000:.0f}ms)"
        )

        return response.json()

    def get_percentiles(self) -> Dict:
        """Calculate latency percentiles"""
        import numpy as np

        latencies = [t['latency'] for t in self.timings]

        if not latencies:
            return {}

        return {
            "count": len(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "mean": np.mean(latencies),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }

# Usage
client = TimedClient(api_key=API_KEY)

# Make 100 requests
for player in players:
    client.predict(player['game_id'], player['player_id'], 'passing_yards')

# Analyze performance
stats = client.get_percentiles()
print(f"\nLatency Statistics (ms):")
print(f"  Count: {stats['count']}")
print(f"  Mean:  {stats['mean']*1000:.0f}ms")
print(f"  p50:   {stats['p50']*1000:.0f}ms")
print(f"  p95:   {stats['p95']*1000:.0f}ms")
print(f"  p99:   {stats['p99']*1000:.0f}ms")
```

### Prometheus Metrics Integration

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
api_requests = Counter(
    'nfl_api_requests_total',
    'Total API requests',
    ['endpoint', 'status']
)

api_latency = Histogram(
    'nfl_api_latency_seconds',
    'API request latency',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

class MonitoredClient:
    """API client with Prometheus metrics"""

    def predict(self, game_id: str, player_id: str, stat_type: str) -> Dict:
        """Prediction with metrics"""
        start = time.time()

        try:
            response = session.post(
                f"{BASE_URL}/predict",
                json={
                    "game_id": game_id,
                    "player_id": player_id,
                    "stat_type": stat_type
                }
            )

            # Record metrics
            api_requests.labels(endpoint='/predict', status=response.status_code).inc()
            api_latency.labels(endpoint='/predict').observe(time.time() - start)

            return response.json()

        except Exception as e:
            api_requests.labels(endpoint='/predict', status='error').inc()
            raise

# Start Prometheus metrics server
start_http_server(8001)

# Use monitored client
client = MonitoredClient()

# Metrics available at http://localhost:8001/metrics
```

---

## Performance Checklist

### Before Production

- [ ] **Connection Pooling**
  - [ ] HTTPAdapter configured with pool_connections=10+
  - [ ] Session reused across all requests
  - [ ] Retry strategy configured

- [ ] **Batching**
  - [ ] Batch size optimized (50-100)
  - [ ] Adaptive batching for variable workloads
  - [ ] Grouped by game_id where possible

- [ ] **Caching**
  - [ ] Redis cache for repeated predictions
  - [ ] TTL configured (1 hour recommended)
  - [ ] Cache invalidation strategy defined

- [ ] **Parallel Processing**
  - [ ] Worker count optimized (CPU count × 3)
  - [ ] Shared session across workers
  - [ ] Rate limits respected

- [ ] **Monitoring**
  - [ ] Request timing logged
  - [ ] Latency percentiles tracked (p50, p95, p99)
  - [ ] Error rates monitored
  - [ ] Cache hit rate tracked

---

## Support

**Performance Issues?** Contact support@nfl-analytics.com with:
- Latency percentiles (p50, p95, p99)
- Request volume (req/min)
- Error rates
- Client configuration

**Monitoring:** Prometheus metrics available at `/metrics` endpoint

---

**Last Updated:** November 15, 2025
**Status:** Production-Ready
**Next:** [Monitoring Guide](./monitoring_guide.md)
