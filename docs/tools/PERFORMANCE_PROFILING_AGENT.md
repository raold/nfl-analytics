# Performance Profiling Agent

**Status:** ✅ Production-ready
**Created:** November 22, 2025
**Author:** Claude Code (Sonnet 4.5)
**Purpose:** Enhanced performance profiling agent for comprehensive bottleneck detection across the entire NFL analytics stack

## Overview

The Performance Profiling Agent is an advanced profiling system that identifies performance bottlenecks across:

- **Database queries** (PostgreSQL, pg_stat_statements, EXPLAIN ANALYZE)
- **Python code** (cProfile, memory_profiler, tracemalloc)
- **R code** (Rprof integration via subprocess)
- **API endpoints** (Prometheus metrics, latency tracking)
- **Backtest performance** (prediction time, model loading)

### Key Features

1. **Comprehensive Profiling**
   - Database query profiling with pg_stat_statements
   - Python code profiling (CPU, memory, hot paths)
   - R code profiling via Rprof
   - API endpoint latency tracking

2. **Bottleneck Detection**
   - N+1 query pattern detection
   - Memory leak detection
   - Expensive computation identification (>1s)
   - Missing index detection

3. **Optimization Recommendations**
   - Actionable suggestions with code examples
   - Expected improvement estimates
   - Implementation difficulty ratings
   - Reference documentation links

4. **Visualization & Export**
   - HTML timeline visualization
   - JSON report export
   - Flamegraph support (with py-spy)
   - Prometheus/Grafana integration

## Installation

The agent is included in the standard project installation:

```bash
# Python dependencies already installed
pip install psycopg2-binary pandas numpy

# Optional: Enhanced profiling tools
pip install memory-profiler line-profiler py-spy
```

**Database Setup (Required):**

```sql
-- Enable pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Verify enabled
SELECT * FROM pg_stat_statements LIMIT 1;
```

## Quick Start

### 1. Database Query Profiling

```bash
# Profile slow queries (>100ms)
./scripts/profile_performance.sh database

# Profile very slow queries (>500ms)
./scripts/profile_performance.sh database --min-duration 500

# Export top 100 queries
./scripts/profile_performance.sh database --top-n 100
```

**Python API:**

```python
from py.utils.performance_profiling_agent import PerformanceProfilingAgent

agent = PerformanceProfilingAgent()

# Profile database queries
report = agent.profile_database_queries(min_duration_ms=100, top_n=50)

print(f"Slow queries: {len(report.slow_queries)}")
print(f"N+1 patterns: {len(report.n_plus_one_queries)}")
print(f"Missing indexes: {len(report.missing_indexes)}")
print(f"P95 latency: {report.p95_query_time_ms:.1f}ms")
```

### 2. Python Code Profiling

```bash
# Profile specific Python file
./scripts/profile_performance.sh code py/ensemble/enhanced_ensemble_v5.py

# Profile with output
./scripts/profile_performance.sh code py/backtest/historical_validation_4way.py \
    --output backtest_profile.txt
```

**Python API:**

```python
# Profile entire file
report = agent.profile_python_code(
    "py/ensemble/enhanced_ensemble_v5.py",
    track_memory=True
)

print(f"Total time: {report.total_time_s:.2f}s")
print(f"Memory used: {report.memory_usage_mb:.1f}MB")
print(f"Peak memory: {report.peak_memory_mb:.1f}MB")
print(f"Hot paths: {len(report.hot_paths)}")

# Profile specific function
def expensive_operation(n):
    return sum(i**2 for i in range(n))

report = agent.profile_function(expensive_operation, 1000000)
```

### 3. R Code Profiling

```bash
# Profile R script
./scripts/profile_performance.sh code R/train_hierarchical_v2_canonical.R
```

**Python API:**

```python
# Profile R script
result = agent.profile_r_code("R/train_state_space_v2_canonical.R")

if 'error' not in result:
    print(f"R profiling output: {result['output_file']}")
else:
    print(f"Error: {result['error']}")
```

### 4. API Endpoint Profiling

```bash
# Profile API endpoints (last 1 hour)
./scripts/profile_performance.sh api

# Profile with HTML report
./scripts/profile_performance.sh api --format html --output api_report.html
```

**Python API:**

```python
# Profile API endpoints
report = agent.profile_api_endpoints(
    duration_hours=1,
    min_latency_ms=100
)

print(f"Total endpoints: {len(report.endpoints)}")
print(f"Slow endpoints: {len(report.slow_endpoints)}")

for endpoint in report.slow_endpoints[:5]:
    print(f"  {endpoint['method']} {endpoint['endpoint']}: {endpoint['avg_latency_ms']:.1f}ms")
```

### 5. Full Stack Profiling

```bash
# Profile entire stack
./scripts/profile_performance.sh full

# Export JSON report
./scripts/profile_performance.sh full --format json --output performance_report.json

# Export HTML timeline
./scripts/profile_performance.sh full --format html --output timeline.html
```

**Python API:**

```python
# Detect all bottlenecks
bottlenecks = agent.detect_bottlenecks(
    profile_database=True,
    profile_python=True,
    profile_api=True,
)

print(f"Found {len(bottlenecks)} bottlenecks")

# Generate recommendations
recommendations = agent.generate_recommendations(bottlenecks)

print(f"Top 5 Recommendations:")
for i, rec in enumerate(recommendations[:5], 1):
    print(f"{i}. Priority {rec.priority}: {rec.recommendation}")
    print(f"   Expected improvement: {rec.expected_improvement}")
    print(f"   Difficulty: {rec.implementation_difficulty}")

# Export reports
agent.export_report("performance_report.json")
agent.export_timeline("timeline.html")

# Print summary
print(agent.generate_summary_report())
```

## Profiling Modes

### Database Profiling

**Detects:**
- Slow queries (configurable threshold, default: >100ms)
- N+1 query patterns (high call count, low avg time)
- Missing indexes (sequential scans on large tables)
- Cache hit ratios

**Metrics Tracked:**
- Average query time
- P95/P99 query latency
- Total execution time
- Number of calls
- Rows returned
- Buffer cache hit ratio

**Example Output:**

```
Slow Queries: 15
N+1 Patterns: 3
Missing Indexes: 5
Average Query Time: 250.5ms
P95 Query Time: 1250.0ms
P99 Query Time: 3500.0ms

Top 10 Slowest Queries:
----------------------------------------------------------------------

1. Avg: 3500.0ms, Calls: 50, Total: 175.0s
   SELECT * FROM mart.player_game_stats pgs JOIN games g ON ...

2. Avg: 1250.0ms, Calls: 200, Total: 250.0s
   SELECT * FROM plays WHERE game_id IN (SELECT ...)
```

### Python Code Profiling

**Detects:**
- Expensive computations (>1s cumulative time)
- Hot paths (frequently called functions)
- Memory usage and leaks
- Inefficient algorithms

**Metrics Tracked:**
- Total execution time
- Per-function cumulative time
- Per-function total time
- Number of calls
- Memory usage (MB)
- Peak memory (MB)

**Example Output:**

```
Total Time: 45.23s
Memory Usage: 250.5MB
Peak Memory: 512.0MB
Hot Paths: 8

Top 10 Functions by Cumulative Time:
----------------------------------------------------------------------
1. 25.500s - train_model (py/models/train.py)
2. 12.300s - compute_features (py/data/feature_engineering.py)
3. 5.200s - predict_batch (py/ensemble/enhanced_ensemble_v5.py)
```

### R Code Profiling

**Uses:** Rprof via subprocess

**Detects:**
- Slow R functions
- Memory allocation patterns
- brms/Stan sampling bottlenecks

**Note:** Requires Rscript in PATH

### API Endpoint Profiling

**Uses:** Prometheus metrics (requires FastAPI instrumented with prometheus_client)

**Detects:**
- Slow endpoints (>100ms average latency)
- High-traffic endpoints
- Response time trends

**Example Output:**

```
Total Endpoints: 5
Slow Endpoints: 2

Slow Endpoints:
----------------------------------------------------------------------
1. POST /predict: 250.5ms
2. GET /models/ensemble: 150.2ms
```

## Bottleneck Types

The agent detects the following bottleneck types:

| Type | Description | Typical Cause |
|------|-------------|---------------|
| `DATABASE_QUERY` | Slow database query | Missing index, complex join, large result set |
| `N_PLUS_ONE` | N+1 query pattern | Loop with individual queries instead of JOIN |
| `MEMORY_LEAK` | Memory not freed | Circular references, unused objects |
| `CPU_BOUND` | CPU-intensive computation | Inefficient algorithm, missing vectorization |
| `IO_BOUND` | I/O bottleneck | Slow disk/network, large file reads |
| `INEFFICIENT_ALGORITHM` | Suboptimal algorithm | O(n²) when O(n) possible |
| `MISSING_INDEX` | Database missing index | Sequential scan on large table |
| `MISSING_CACHE` | Cacheable query not cached | Repeated identical queries |
| `SUBOPTIMAL_PANDAS` | Inefficient Pandas operation | iterrows(), chained indexing |
| `EXPENSIVE_COMPUTATION` | Long-running computation | >1s execution time |

## Severity Levels

Bottlenecks are classified by severity:

| Severity | Threshold | Impact | Priority |
|----------|-----------|--------|----------|
| `CRITICAL` | >10s | Blocking production | P0 - Immediate action |
| `HIGH` | >3s | Significant impact | P1 - Fix within 1 week |
| `MEDIUM` | >1s | Noticeable impact | P2 - Fix within 1 month |
| `LOW` | >500ms | Minor impact | P3 - Fix when convenient |
| `INFO` | <500ms | Optimization opportunity | P4 - Optional |

## Recommendations

The agent generates actionable recommendations with:

1. **Priority** (1-5, 1=highest)
2. **Recommendation** (what to do)
3. **Expected improvement** (performance gain)
4. **Implementation difficulty** (Easy/Medium/Hard)
5. **Code example** (how to implement)
6. **Reference docs** (where to learn more)

**Example Recommendation:**

```
Priority 1: Use JOIN or batch query to eliminate N+1 pattern
Expected Improvement: 10-100x faster
Difficulty: Easy

Code Example:
# BAD (N+1)
for player_id in player_ids:
    stats = query(f"SELECT * FROM stats WHERE player_id = {player_id}")

# GOOD (single query)
stats = query("SELECT * FROM stats WHERE player_id = ANY(%s)", (player_ids,))

Reference: docs/database/SQL_PATTERNS.md
```

## Visualization

### HTML Timeline

The agent generates an interactive HTML timeline showing:

- All bottlenecks sorted by severity
- Time, location, and impact
- Color-coded by severity (red=critical, orange=high, yellow=medium, green=low)

**Generate:**

```bash
./scripts/profile_performance.sh full --format html
# Opens: performance_timeline.html
```

### JSON Report

Complete profiling data exported as JSON:

```json
{
  "generated_at": "2025-11-22T10:30:00",
  "project_root": "\",
  "bottlenecks": [
    {
      "severity": "HIGH",
      "type": "database_query",
      "title": "Slow query: 2500.0ms average",
      "description": "Query called 100 times, total 250000.0ms",
      "location": "PostgreSQL",
      "time_ms": 2500.0,
      "impact": "250.0s total time",
      "detected_at": "2025-11-22T10:30:00",
      "details": {
        "query": "SELECT * FROM ...",
        "calls": 100,
        "avg_time_ms": 2500.0
      }
    }
  ],
  "profile_data": {}
}
```

### Flamegraph (Optional)

**Requires:** py-spy or speedscope

```bash
# Install py-spy
pip install py-spy

# Generate flamegraph
py-spy record -o flamegraph.svg -- python py/backtest/historical_validation_4way.py
```

## Integration with Existing Tools

### 1. Performance Optimizer

The Performance Profiling Agent **extends** the existing Performance Optimizer:

```python
from py.utils.performance_optimizer import PerformanceOptimizer
from py.utils.performance_profiling_agent import PerformanceProfilingAgent

# Use optimizer for basic checks
optimizer = PerformanceOptimizer()
optimizer.check_mcmc_config(chains=4, tune=1000, draws=2000)

# Use agent for deep profiling
agent = PerformanceProfilingAgent()
agent.profile_database_queries()
agent.detect_bottlenecks()
```

### 2. Database Pool

The agent works seamlessly with database pooling:

```python
from py.utils.database_pool import get_default_pool
from py.utils.performance_profiling_agent import PerformanceProfilingAgent

# Use pooled connections for profiling
pool = get_default_pool()
agent = PerformanceProfilingAgent(
    db_connection_string=pool.pool.dsn
)
```

### 3. Prometheus/Grafana

**API endpoint profiling requires:**

1. FastAPI instrumented with prometheus_client
2. Prometheus scraping /metrics endpoint
3. Prometheus accessible at http://localhost:9090

**Example instrumentation:**

```python
from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import FastAPI

app = FastAPI()

# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

@app.middleware("http")
async def track_requests(request, call_next):
    with REQUEST_DURATION.labels(method=request.method, endpoint=request.url.path).time():
        response = await call_next(request)
    return response
```

## Common Use Cases

### 1. Pre-Production Checklist

Before deploying to production, run full profiling:

```bash
# 1. Profile database queries
./scripts/profile_performance.sh database --min-duration 100

# 2. Profile ensemble code
./scripts/profile_performance.sh code py/ensemble/enhanced_ensemble_v5.py

# 3. Profile backtest performance
./scripts/profile_performance.sh backtest

# 4. Generate full report
./scripts/profile_performance.sh full --format json
```

**Acceptance Criteria:**
- No CRITICAL bottlenecks (>10s)
- <5 HIGH bottlenecks (>3s)
- P95 query latency <500ms
- Backtest time <1h (148 games)

### 2. Weekly Performance Audit

```bash
# Run weekly audit
./scripts/profile_performance.sh full --format html --output weekly_audit.html

# Review report
open weekly_audit.html
```

**Track trends:**
- Query time degradation
- Memory usage growth
- New N+1 patterns
- API latency increases

### 3. Backtest Optimization

Profile backtest before optimization:

```python
agent = PerformanceProfilingAgent()

# Profile backtest code
report = agent.profile_python_code("py/backtest/historical_validation_4way.py")

print(f"Backtest time: {report.total_time_s:.1f}s")
print(f"Memory used: {report.memory_usage_mb:.1f}MB")

# Profile database queries during backtest
query_report = agent.profile_database_queries()
print(f"Slow queries: {len(query_report.slow_queries)}")

# Get optimization recommendations
bottlenecks = agent.detect_bottlenecks()
recommendations = agent.generate_recommendations(bottlenecks)

for rec in recommendations[:5]:
    print(f"- {rec.recommendation}")
```

**Expected improvements:**
- Database pooling: 50-100x faster queries
- Feature caching: 70-85% hit rate
- Result: 2.7x speedup (Nov 13 achievement)

### 4. Model Training Optimization

Profile model training to identify bottlenecks:

```python
# Profile MCMC sampling
report = agent.profile_python_code("py/models/train_bnn_passing.py")

# Check for slow operations
for func in report.function_stats[:10]:
    if func['cumtime'] > 10:
        print(f"SLOW: {func['function']} took {func['cumtime']:.1f}s")
```

**Common bottlenecks:**
- Insufficient MCMC chains (<4)
- Too few tuning steps (<1000)
- Inefficient data loading (iterrows)
- Missing vectorization

## Troubleshooting

### pg_stat_statements Not Enabled

**Error:**
```
pg_stat_statements not enabled. Run: CREATE EXTENSION pg_stat_statements;
```

**Solution:**
```sql
-- Connect to database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01

-- Enable extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Verify
SELECT * FROM pg_stat_statements LIMIT 1;
```

### Prometheus Not Accessible

**Error:**
```
Prometheus query failed: Connection refused
```

**Solution:**
```bash
# Check Prometheus is running
curl http://localhost:9090/api/v1/status/buildinfo

# If not running, start Docker services
docker compose up -d prometheus
```

### R Profiling Timeout

**Error:**
```
R profiling timeout: R/train_hierarchical_v2_canonical.R
```

**Solution:**
- R script takes >5 minutes (timeout limit)
- Profile manually with Rprof:

```r
Rprof("output.out")
source("R/train_hierarchical_v2_canonical.R")
Rprof(NULL)
summaryRprof("output.out")
```

### Memory Profiling Overhead

**Note:** Memory profiling adds overhead (~10-20% slower)

**Solution:**
- Disable for production: `track_memory=False`
- Use only for development profiling

## Performance Impact

The agent itself has minimal performance impact:

| Operation | Overhead | Impact |
|-----------|----------|--------|
| Database profiling (pg_stat_statements) | <1% | Negligible |
| Python profiling (cProfile) | ~10% | Acceptable for dev |
| Memory profiling (tracemalloc) | ~20% | Disable in production |
| API profiling (Prometheus) | <1% | Negligible |

**Best Practice:** Run profiling in development/staging, not production.

## Advanced Usage

### Custom Bottleneck Detection

```python
from py.utils.performance_profiling_agent import (
    Bottleneck,
    BottleneckSeverity,
    BottleneckType,
)

agent = PerformanceProfilingAgent()

# Add custom bottleneck
custom_bottleneck = Bottleneck(
    severity=BottleneckSeverity.HIGH,
    bottleneck_type=BottleneckType.EXPENSIVE_COMPUTATION,
    title="Custom expensive operation",
    description="My custom operation is slow",
    location="my_script.py:42",
    current_time_ms=5000,
    impact="5s execution time",
    details={"custom_metric": 100},
)

agent.bottlenecks.append(custom_bottleneck)

# Generate recommendations
recommendations = agent.generate_recommendations()
```

### Continuous Monitoring

Integrate with CI/CD to track performance trends:

```yaml
# .github/workflows/performance.yml
name: Performance Profiling

on:
  pull_request:
    branches: [main]

jobs:
  profile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Profile performance
        run: |
          ./scripts/profile_performance.sh full --format json

      - name: Check for regressions
        run: |
          python scripts/check_performance_regression.py \
            performance_report.json \
            baseline_report.json
```

## References

- **Performance Optimizer:** `py/utils/performance_optimizer.py`
- **Database Pool:** `py/utils/database_pool.py`
- **Feature Cache:** `py/utils/feature_cache.py`
- **SQL Patterns:** `docs/database/SQL_PATTERNS.md`
- **Testing Guide:** `tests/unit/test_performance_profiling_agent.py`

## Changelog

**November 22, 2025 - v1.0 (Initial Release)**
- Comprehensive database, Python, R, API profiling
- Bottleneck detection and recommendations
- HTML timeline and JSON export
- CLI wrapper script
- Unit tests (30+ test methods)
- Complete documentation

## Future Enhancements

1. **Flamegraph Integration:** Native flamegraph generation (py-spy/speedscope)
2. **ML-Based Predictions:** Predict performance degradation before it happens
3. **Automatic Optimization:** Apply simple optimizations automatically
4. **Historical Tracking:** Track performance trends over time in database
5. **Real-Time Alerts:** Slack/email alerts for critical bottlenecks

---

**Status:** ✅ Production-ready
**Last Updated:** November 22, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
