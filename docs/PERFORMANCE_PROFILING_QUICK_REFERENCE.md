# Performance Profiling Agent - Quick Reference

**One-page reference for common profiling tasks**

## Quick Start

```bash
# Database profiling
./scripts/profile_performance.sh database

# Python code profiling
./scripts/profile_performance.sh code py/ensemble/enhanced_ensemble_v5.py

# API profiling
./scripts/profile_performance.sh api

# Full stack profiling
./scripts/profile_performance.sh full --format html
```

## Python API

```python
from py.utils.performance_profiling_agent import PerformanceProfilingAgent

agent = PerformanceProfilingAgent()

# Database profiling
report = agent.profile_database_queries(min_duration_ms=100)
print(f"Slow queries: {len(report.slow_queries)}")

# Python profiling
report = agent.profile_python_code("file.py", track_memory=True)
print(f"Time: {report.total_time_s:.2f}s, Memory: {report.memory_usage_mb:.1f}MB")

# Function profiling
report = agent.profile_function(my_function, arg1, arg2)

# Full bottleneck detection
bottlenecks = agent.detect_bottlenecks()
recommendations = agent.generate_recommendations(bottlenecks)

# Export reports
agent.export_report("report.json")
agent.export_timeline("timeline.html")
print(agent.generate_summary_report())
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--min-duration MS` | Minimum query duration | 100ms |
| `--output FILE` | Output file path | Auto-generated |
| `--format FORMAT` | Output format (text, json, html) | text |
| `--top-n N` | Number of top results | 50 |

## Bottleneck Types

| Type | Description | Fix |
|------|-------------|-----|
| `DATABASE_QUERY` | Slow query | Add index, rewrite query |
| `N_PLUS_ONE` | N+1 pattern | Use JOIN or batch query |
| `MEMORY_LEAK` | Memory not freed | Fix references, gc.collect() |
| `EXPENSIVE_COMPUTATION` | Long computation | Vectorize, parallelize |
| `MISSING_INDEX` | No index | CREATE INDEX |
| `MISSING_CACHE` | Not cached | Add Redis/feature cache |

## Severity Levels

| Severity | Threshold | Action |
|----------|-----------|--------|
| CRITICAL | >10s | Fix immediately (P0) |
| HIGH | >3s | Fix within 1 week (P1) |
| MEDIUM | >1s | Fix within 1 month (P2) |
| LOW | >500ms | Fix when convenient (P3) |

## Common Use Cases

### Pre-Production Checklist
```bash
./scripts/profile_performance.sh database --min-duration 100
./scripts/profile_performance.sh code py/ensemble/enhanced_ensemble_v5.py
./scripts/profile_performance.sh full --format json
```

**Acceptance Criteria:**
- No CRITICAL bottlenecks
- <5 HIGH bottlenecks
- P95 query latency <500ms

### Weekly Audit
```bash
./scripts/profile_performance.sh full --format html --output weekly_audit.html
open weekly_audit.html
```

### Backtest Optimization
```python
agent = PerformanceProfilingAgent()
report = agent.profile_python_code("py/backtest/historical_validation_4way.py")
query_report = agent.profile_database_queries()
bottlenecks = agent.detect_bottlenecks()
recommendations = agent.generate_recommendations(bottlenecks)
```

## Setup Requirements

### Database Profiling (pg_stat_statements)
```sql
-- Add to postgresql.conf
shared_preload_libraries = 'pg_stat_statements'

-- Restart PostgreSQL, then:
CREATE EXTENSION pg_stat_statements;
```

### API Profiling (Prometheus)
```python
# FastAPI instrumentation
from prometheus_client import make_asgi_app
app.mount("/metrics", make_asgi_app())
```

## Troubleshooting

**pg_stat_statements not enabled:**
```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

**Prometheus not accessible:**
```bash
docker compose up -d prometheus
curl http://localhost:9090/api/v1/status/buildinfo
```

**Memory profiling slow:**
```python
# Disable in production
report = agent.profile_python_code(file, track_memory=False)
```

## Integration

```python
# With Performance Optimizer
from py.utils.performance_optimizer import PerformanceOptimizer
optimizer = PerformanceOptimizer()
optimizer.check_mcmc_config(chains=4, tune=1000, draws=2000)

# With Database Pool
from py.utils.database_pool import get_default_pool
pool = get_default_pool()
agent = PerformanceProfilingAgent(db_connection_string=pool.pool.dsn)

# With Ensemble Health Monitor
from py.utils.ensemble_health_monitor import EnsembleHealthMonitor
monitor = EnsembleHealthMonitor()
if monitor.check_ensemble_health().is_healthy:
    agent.profile_python_code("py/backtest/historical_validation_4way.py")
```

## Example Recommendation

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

## References

- **Full Documentation:** `docs/tools/PERFORMANCE_PROFILING_AGENT.md`
- **Examples:** `examples/performance_profiling_examples.py`
- **Tests:** `tests/unit/test_performance_profiling_agent.py`
- **CLI Script:** `scripts/profile_performance.sh`
- **Core Implementation:** `py/utils/performance_profiling_agent.py`

---

**Last Updated:** November 22, 2025
