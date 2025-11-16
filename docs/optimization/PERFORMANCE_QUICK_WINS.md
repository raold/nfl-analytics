# Performance Quick Wins - Immediate Optimizations

**Last Updated:** November 15, 2025
**Time to Implement:** 2-4 hours
**Expected Impact:** 3-5x speedup on critical paths

---

## 🚀 Quick Win 1: Reduce API Latency (4s → 1.2s)

**Current:** 4 seconds per prediction with R models
**Target:** <2 seconds
**Effort:** 1 hour
**Impact:** 3.3x speedup

### Step 1: Reduce R Posterior Draws (15 min)

```python
# File: py/ensemble/enhanced_ensemble_v6_production.py
# Line 669 (hierarchical) and 742 (informative priors)

# BEFORE:
result = self.r_interface.predict_hierarchical(
    player_id, season, week, self.hierarchical_path,
    ndraws=1000  # ← Too many draws
)

# AFTER:
result = self.r_interface.predict_hierarchical(
    player_id, season, week, self.hierarchical_path,
    ndraws=500  # ← 2x faster, minimal accuracy loss
)
```

**Test:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"player_id":"00-0019596","season":2024,"week":9,"stat_type":"passing_yards"}'

# Measure response time (should be ~2s instead of 4s)
```

### Step 2: Implement Database Connection Pooling (30 min)

```python
# File: py/ensemble/enhanced_ensemble_v6_production.py
# Replace psycopg2 with psycopg (v3) and connection pool

from psycopg_pool import ConnectionPool

class EnhancedEnsemble5Way:
    def __init__(self, ...):
        # Create connection pool (singleton)
        self.db_pool = ConnectionPool(
            conninfo=f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            min_size=2,
            max_size=10,
            timeout=30.0
        )

    def _fetch_features(self, player_id, season, week):
        # Use connection from pool
        with self.db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()
```

**Expected:** 200ms → 50ms (4x speedup)

### Step 3: Parallel R Model Execution (15 min)

```python
# File: py/ensemble/enhanced_ensemble_v6_production.py
# Add to predict() method

from concurrent.futures import ThreadPoolExecutor, as_completed

def predict(self, player_id, season, week, ...):
    # ... existing code ...

    # Run R models in parallel (instead of sequential)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(self._predict_state_space, player_id, season, week, features): "state_space_v2",
            executor.submit(self._predict_hierarchical, player_id, season, week, features): "hierarchical_v2",
            executor.submit(self._predict_informative_priors, player_id, season, week, features): "informative_priors_v2",
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                pred, std = future.result()
                components[model_name] = {"mean": pred, "std": std}
            except Exception as e:
                logger.warning(f"Component {model_name} failed: {e}")

    # ... rest of code ...
```

**Expected:** 2.5s → 1.0s (2.5x speedup on R models)

---

## 🔥 Quick Win 2: Redis Prediction Caching (Cache hit → 3ms)

**Current:** Every prediction takes 4 seconds
**Target:** 85% cache hit rate (3ms on hit)
**Effort:** 1 hour
**Impact:** 1333x speedup on cache hits

### Step 1: Add Caching to API Endpoint (30 min)

```python
# File: api/main.py
# Modify predict() endpoint

import json

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: APIKey = Depends(require_api_key)):
    start_time = time.time()

    # 1. Check cache first
    cache_key = f"nfl:prediction:{request.player_id}:{request.season}:{request.week}:{request.stat_type.value}"

    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Cache HIT: {cache_key}")
                prediction = PredictionResponse(**json.loads(cached))

                # Record metrics
                latency = time.time() - start_time
                prediction_latency.observe(latency)
                prediction_counter.labels(stat_type=request.stat_type.value, status="cache_hit").inc()

                return prediction
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

    # 2. Cache MISS - generate prediction
    logger.info(f"Cache MISS: {cache_key}")
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    prediction = predictor.predict(request)

    # 3. Cache result (1 hour TTL)
    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                prediction.json()
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    # Record metrics
    latency = time.time() - start_time
    prediction_latency.observe(latency)
    prediction_counter.labels(stat_type=request.stat_type.value, status="cache_miss").inc()

    return prediction
```

### Step 2: Create Cache Warmup Script (30 min)

```python
# File: py/api/cache_warmup.py
#!/usr/bin/env python3
"""
Cache Warmup Script - Pre-populate Redis cache for backtest
"""

import argparse
import logging
from redis import Redis
from api.models import PredictionRequest, StatType
from api.prediction import EnsemblePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def warmup_cache(season: int, weeks: list[int], stat_type: str = "passing_yards"):
    """Pre-warm Redis cache with predictions"""

    # Connect to Redis
    redis_client = Redis.from_url("redis://localhost:6379", decode_responses=True)
    logger.info(f"Connected to Redis: {redis_client.ping()}")

    # Load predictor
    predictor = EnsemblePredictor(models_dir="/app/models", device="cpu")
    predictor.load_models()
    logger.info("Predictor loaded")

    # Fetch QB list for season
    import psycopg
    conn = psycopg.connect("postgresql://dro:sicillionbillions@localhost:5544/devdb01")
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT player_id
        FROM mart.player_game_stats
        WHERE season = %s AND stat_category = 'passing' AND stat_attempts >= 10
        ORDER BY stat_yards DESC
        LIMIT 32
    """, (season,))
    player_ids = [row[0] for row in cur.fetchall()]
    logger.info(f"Found {len(player_ids)} QBs for season {season}")

    # Warmup cache
    total = len(player_ids) * len(weeks)
    count = 0
    for player_id in player_ids:
        for week in weeks:
            try:
                request = PredictionRequest(
                    player_id=player_id,
                    season=season,
                    week=week,
                    stat_type=StatType[stat_type]
                )
                prediction = predictor.predict(request)

                # Cache result
                cache_key = f"nfl:prediction:{player_id}:{season}:{week}:{stat_type}"
                redis_client.setex(cache_key, 3600, prediction.json())

                count += 1
                if count % 10 == 0:
                    logger.info(f"Progress: {count}/{total} ({100*count/total:.1f}%)")

            except Exception as e:
                logger.warning(f"Failed to cache {player_id} W{week}: {e}")

    logger.info(f"✅ Cache warmup complete: {count}/{total} predictions cached")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--weeks", type=str, default="1-10", help="Week range (e.g. 1-10)")
    parser.add_argument("--stat-type", type=str, default="passing_yards")
    args = parser.parse_args()

    # Parse weeks
    if "-" in args.weeks:
        start, end = args.weeks.split("-")
        weeks = list(range(int(start), int(end) + 1))
    else:
        weeks = [int(args.weeks)]

    warmup_cache(args.season, weeks, args.stat_type)
```

**Usage:**
```bash
# Pre-warm cache for 2024 weeks 1-10
python py/api/cache_warmup.py --season 2024 --weeks 1-10

# Expected: 5-10 minutes to cache 320 predictions (32 QBs × 10 weeks)
```

---

## ⚡ Quick Win 3: Optimize Pre-commit Hooks (15s → 6s)

**Current:** 10-15 seconds per commit
**Target:** <6 seconds
**Effort:** 30 min
**Impact:** 2.5x speedup

### Step 1: Enable MyPy Incremental Mode (10 min)

```yaml
# File: .pre-commit-config.yaml
# Line 19-25

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.5.0
  hooks:
    - id: mypy
      additional_dependencies: [types-requests, types-python-dateutil]
      files: ^py/
      args: [--cache-dir, .mypy_cache, --ignore-missing-imports]  # ← Add --cache-dir
```

**Expected:** 3s → 0.5s (6x speedup)

### Step 2: Skip Security Checks for Tests (10 min)

```yaml
# File: .pre-commit-config.yaml
# Line 54-60

- repo: https://github.com/PyCQA/bandit
  rev: 1.7.5
  hooks:
    - id: bandit
      args: [-c, pyproject.toml]
      additional_dependencies: ["bandit[toml]", "pbr"]
      files: ^py/
      exclude: ^(tests/|examples/|scripts/test_)  # ← Add exclude
```

**Expected:** 2s → 0.5s (4x speedup)

### Step 3: Test Pre-commit Speed (10 min)

```bash
# Before optimization
time git commit -m "test"
# Expected: ~15 seconds

# After optimization
time git commit -m "test"
# Expected: ~6 seconds (2.5x speedup)
```

---

## 🎯 Quick Win 4: MCMC Parallelization (87 min → 59 min)

**Current:** 4 chains, 40% CPU utilization
**Target:** 6 chains, 60% CPU utilization
**Effort:** 1 hour (includes validation)
**Impact:** 32% speedup on training

### Step 1: Run Optimization Script (5 min)

```bash
# Check if script exists
ls scripts/optimize_mcmc_m4.sh

# If exists, run optimization
bash scripts/optimize_mcmc_m4.sh

# If not exists, manual edit required (see Step 2)
```

### Step 2: Manual Configuration (if script missing) (20 min)

```r
# File: R/state_space/train_state_space_v2_canonical.R
# Find the brm() or cmdstan_model$sample() call

# BEFORE:
fit <- brm(
    formula = ...,
    data = train_data,
    family = gaussian(),
    prior = ...,
    chains = 4,  # ← Increase to 6
    cores = 4,   # ← Increase to 6
    iter = 6000,
    warmup = 2000,
    ...
)

# AFTER:
fit <- brm(
    formula = ...,
    data = train_data,
    family = gaussian(),
    prior = ...,
    chains = 6,  # ← 6 chains for M4 10-core
    cores = 6,   # ← Parallel execution
    iter = 6000,
    warmup = 2000,
    ...
)
```

**Apply to all R training scripts:**
1. `R/state_space/train_state_space_v2_canonical.R`
2. `R/train_hierarchical_v2_production.R`
3. `R/train_informative_priors_v2_production.R`

### Step 3: Validate Convergence (30 min)

```bash
# Run test training cycle
Rscript R/state_space/train_state_space_v2_canonical.R

# Check convergence diagnostics
# Look for:
# ✅ R-hat < 1.01 (all parameters)
# ✅ ESS > 400 (bulk and tail)
# ✅ Divergences < 10
# ✅ Training time reduced (22 min → ~15 min)

# If convergence fails, revert
# sed -i 's/chains = 6/chains = 4/g' R/*.R
# sed -i 's/cores = 6/cores = 4/g' R/*.R
```

**Expected Speedup:**
- State-Space v2: 22 min → 15 min (32% faster)
- Hierarchical v2: 35 min → 24 min (31% faster)
- Informative Priors v2: 30 min → 20 min (33% faster)
- **Total:** 87 min → 59 min (32% faster)

---

## 📊 Expected Total Impact

### API Performance

**Before:**
- Prediction latency: 4 seconds (slow)
- Cache hit rate: 0% (no caching)
- Average latency: 4 seconds

**After:**
- Prediction latency: 1.2 seconds (optimized)
- Cache hit rate: 85% (Redis caching)
- Average latency: 1.2s × 0.15 + 0.003s × 0.85 = **0.18s + 0.003s ≈ 183ms** ✅

**Speedup:** 4s → 183ms (21.8x faster on average)

### Backtest Performance

**Before:**
- Training per cycle: 87 minutes
- Data loading: 2 minutes
- Predictions: 8 minutes
- Total per cycle: 97 minutes
- Full backtest (12 cycles): **19.4 hours**

**After:**
- Training per cycle: 59 minutes (MCMC optimization)
- Data loading: 1.5 minutes (SQL indexes)
- Predictions: 0.5 minutes (Redis cache)
- Total per cycle: 61 minutes
- Full backtest (12 cycles): **12.2 hours** ✅

**Speedup:** 19.4h → 12.2h (1.59x faster)

### Developer Experience

**Before:**
- Pre-commit hooks: 15 seconds per commit
- Commits per day: 20
- Total time: 5 minutes/day wasted

**After:**
- Pre-commit hooks: 6 seconds per commit
- Commits per day: 20
- Total time: 2 minutes/day

**Time Saved:** 3 minutes/day × 5 days/week = 15 minutes/week

---

## ✅ Validation Checklist

### API Optimization
- [ ] R posterior draws reduced to 500
- [ ] Database connection pooling implemented
- [ ] Parallel R execution implemented
- [ ] Test prediction latency <2s
- [ ] Redis caching implemented
- [ ] Cache warmup script created
- [ ] Cache hit rate >85% during backtest

### MCMC Optimization
- [ ] 6 chains configured in all R scripts
- [ ] Test training cycle completed
- [ ] Convergence validated (R-hat <1.01)
- [ ] ESS >400 for all parameters
- [ ] Divergences <10
- [ ] Training time reduced by 30%+

### Pre-commit Optimization
- [ ] MyPy incremental mode enabled
- [ ] Security checks exclude tests
- [ ] Custom validators optimized
- [ ] Commit time <6s

### Monitoring
- [ ] Grafana dashboard accessible
- [ ] pg_stat_statements enabled
- [ ] Redis metrics visible
- [ ] API latency metrics tracked

---

## 🚨 Rollback Plan

### If API Performance Degrades

```bash
# Revert R posterior draws
sed -i 's/ndraws=500/ndraws=1000/g' py/ensemble/enhanced_ensemble_v6_production.py

# Disable Redis caching
# Comment out cache check/write in api/main.py

# Revert to sequential R execution
# Remove ThreadPoolExecutor code
```

### If MCMC Convergence Fails

```bash
# Revert to 4 chains
sed -i 's/chains = 6/chains = 4/g' R/**/*.R
sed -i 's/cores = 6/cores = 4/g' R/**/*.R
```

### If Pre-commit Too Slow

```bash
# Disable specific hooks
# Edit .pre-commit-config.yaml and comment out slow hooks
```

---

## 📝 Summary

**Total Implementation Time:** 4 hours
**Expected Speedup:**
- API: 21.8x average (with caching)
- Backtest: 1.59x (19.4h → 12.2h)
- Pre-commit: 2.5x (15s → 6s)

**Status:** ✅ READY TO IMPLEMENT

**Next Steps:**
1. Implement API optimizations (1-2 hours)
2. Apply MCMC configuration (1 hour)
3. Optimize pre-commit hooks (30 min)
4. Validate and test (1 hour)
5. Launch backtest

**Risk Level:** LOW (all changes reversible)

---

**Last Updated:** November 15, 2025
**Author:** Performance Optimization Team
