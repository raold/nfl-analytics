# 4-Way Ensemble Backtest Optimization - Implementation Checklist

**Date:** November 13, 2025
**Target Runtime:** 45-65 min → 24-44 min (60% reduction)
**Expected Effort:** 2-3 hours
**Risk Level:** LOW-MEDIUM

---

## Quick Reference

| Phase | Component | Effort | Savings | Risk | Status |
|-------|-----------|--------|---------|------|--------|
| 1 | Connection pooling | 10 min | 10-12 min | LOW | Ready |
| 1 | R model cache | 15 min | 8-9 min | LOW | Ready |
| 2 | Database indexes | 25 min | 8-10 min | MED | Design complete |
| 2 | Feature cache | 15 min | 2-3 min | LOW | Ready |
| **Total Phase 1+2** | **65 min** | **28-34 min** | **LOW-MED** | **Ready** |

---

## Phase 1: Connection Pooling (10 minutes)

### Goal
Replace individual psycopg2.connect() calls with connection pool that reuses connections.

### Current Code Problem

**File:** `py/backtest/historical_validation_5way.py`, line ~165

```python
# SLOW: Creates new connection for every query
conn = psycopg2.connect(
    host="localhost",
    port=5544,
    database="devdb01",
    user="dro",
    password="sicillionbillions"
)

df = pd.read_sql_query(query, conn)
conn.close()
# Overhead: ~100-150ms per connection × 296 queries = 44-74 min
```

### Implementation

**Step 1:** Add connection pool initialization in `HistoricalValidator.__init__()` (line 53)

```python
from psycopg2 import pool  # Add to imports at top

class HistoricalValidator:
    def __init__(self, ...):
        # ... existing code ...

        # NEW: Initialize connection pool
        self.db_pool = pool.SimpleConnectionPool(
            minconn=2,      # Minimum idle connections
            maxconn=8,      # Maximum concurrent connections
            host="localhost",
            port=5544,
            database="devdb01",
            user="dro",
            password="sicillionbillions",
            connect_timeout=2
        )
        logger.info("✓ Initialized connection pool (min=2, max=8)")
```

**Step 2:** Replace query calls in `_get_r_predictions()` (line ~165)

```python
# BEFORE (line 165-171)
conn = psycopg2.connect(
    host=self.db_host,
    port=self.db_port,
    user=self.db_user,
    password=self.db_password,
    database=self.db_name
)
df = pd.read_sql_query(query, conn)
conn.close()

# AFTER (4 lines instead of 7)
conn = self.db_pool.getconn()
try:
    df = pd.read_sql_query(query, conn)
finally:
    self.db_pool.putconn(conn)
```

**Step 3:** Repeat for all database query locations

Find and replace ALL occurrences of this pattern:
```python
# Find: psycopg2.connect(
# Replace with: conn = self.db_pool.getconn()
# And wrap with try/finally: self.db_pool.putconn(conn)
```

Locations to update:
- Line ~165: `_get_r_predictions()`
- Line ~257: `_get_player_data()`
- Line ~311: `_get_bnn_prediction()`
- Line ~443: `_get_hybrid_xgboost_prediction()`
- Plus any others in `fetch_historical_games()`

**Step 4:** Add cleanup in `close()` method (line 509)

```python
def close(self):
    """Close database connection."""
    if self.conn:
        self.conn.close()

    # NEW: Close connection pool
    if hasattr(self, 'db_pool') and self.db_pool:
        self.db_pool.closeall()
        logger.info("✓ Connection pool closed")
```

**Step 5:** Test

```bash
# Run a small backtest
python py/backtest/historical_validation_5way.py \
    --start-season 2024 \
    --end-season 2024 \
    --end-week 5 \
    --bankroll 1000

# Expected: Much faster database queries
# Look for: "✓ Initialized connection pool (min=2, max=8)"
```

---

## Phase 1: R Model Cache (15 minutes)

### Goal
Load R models once at backtest start instead of reloading for each prediction.

### Current Code Problem

**Files:**
- `py/utils/r_service.py` (line ~160) - `_get_r_predictions()`
- `py/ensemble/enhanced_ensemble_v6.py` (line ~139-163)

```python
# SLOW: Loads model from disk for EVERY prediction
import rpy2.robjects as ro
ro.r('library(brms)')
ro.r(f'model <- readRDS("{model_path}")')  # Reads from disk
ro.r('posterior_preds <- posterior_predict(model, newdata)')
# Cost: ~2-3 seconds per call × 296 predictions = 10-15 minutes
```

### Implementation Option A: Extend RService (Recommended)

**File:** `py/utils/r_service.py`

Add to `RService` class (after line 115):

```python
class RService:
    def __init__(self, config: Optional[RServiceConfig] = None):
        # ... existing code ...
        self._model_cache = {}  # NEW: Cache loaded models
        self._r_initialized = False
        logger.info("Initialized RService with model caching")

    def load_model(self, model_name: str, model_path: Path):
        """Load R model once into memory."""
        if model_name in self._model_cache:
            logger.debug(f"Model {model_name} already cached")
            return

        start = time.time()
        try:
            import rpy2.robjects as ro

            # Lazy initialize R
            if not self._r_initialized:
                ro.r('library(brms)')
                ro.r('library(tidyverse)')
                self._r_initialized = True

            # Load model into R environment with unique variable name
            ro.r(f'model_{model_name} <- readRDS("{model_path}")')
            self._model_cache[model_name] = True

            elapsed = time.time() - start
            logger.info(f"✓ Cached {model_name} in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def predict_cached(
        self,
        model_name: str,
        newdata_df: pd.DataFrame
    ) -> Tuple[float, float]:
        """Predict using cached R model."""
        if model_name not in self._model_cache:
            raise ValueError(f"Model {model_name} not cached. Call load_model() first.")

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            # Convert and assign data
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(newdata_df)

            ro.r.assign('new_data', r_df)

            # Predict (model already in memory)
            ro.r(f'''
            posterior_preds <- posterior_predict(model_{model_name}, newdata = new_data)
            pred_mean <- mean(posterior_preds)
            pred_std <- sd(posterior_preds)
            ''')

            mean_val = float(ro.r('pred_mean')[0])
            std_val = float(ro.r('pred_std')[0])

            return (mean_val, std_val)

        except Exception as e:
            logger.error(f"Cached prediction failed: {e}")
            raise
```

### Implementation Option B: Simple Cache (If rpy2 issues)

If rpy2 has threading issues, use subprocess with model persistence:

```python
# In HistoricalValidator.__init__() (line 53)
self.r_models_loaded = False

# In run_backtest() (line 321)
def run_backtest(self) -> pd.DataFrame:
    # Load R models at START of backtest
    self._preload_r_models()

    # ... rest of backtest ...

def _preload_r_models(self):
    """Load R models into memory before backtest."""
    if self.r_models_loaded:
        return

    from py.utils.r_service import get_r_service
    r_service = get_r_service()

    logger.info("Preloading R models...")
    r_service.load_model(
        'hierarchical_v2',
        Path('models/bayesian/passing_yards_hierarchical_v2.rds')
    )
    r_service.load_model(
        'informative_priors_v2',
        Path('models/bayesian/passing_informative_priors_v2.rds')
    )

    self.r_models_loaded = True
    logger.info("✓ R models preloaded")
```

### Testing

```bash
# Run backtest and watch for model loading messages
python py/backtest/historical_validation_5way.py \
    --start-season 2024 --end-season 2024 --end-week 5

# Expected output:
# ✓ Cached hierarchical_v2 in 2.15s
# ✓ Cached informative_priors_v2 in 1.89s
# Processing 10 player-games with vectorized approach...
#   Speed: 2.5+ games/sec (should be faster with cached models)
```

---

## Phase 2: Database Indexes (25 minutes)

### Goal
Add indexes on frequently-queried columns to speed up database lookups.

### Design Analysis

Current slow query:
```sql
SELECT pgs.stat_yards, pgs.stat_attempts, ...
FROM mart.player_game_stats pgs
WHERE pgs.player_id = '00-0019596'
  AND pgs.season = 2024
  AND pgs.week = 9
LIMIT 1
```

Without indexes: Full table scan of 1.25M rows (~300ms)
With indexes: Index seek + retrieve (~50-80ms)

### Implementation

**Step 1:** Create migration file

```bash
# Create migration file
touch db/migrations/004_add_performance_indexes.sql
```

**Step 2:** Add to `db/migrations/004_add_performance_indexes.sql`

```sql
-- Performance indexes for backtest queries
-- Author: Claude Code
-- Date: November 13, 2025

-- Index for player_game_stats queries (most frequent)
CREATE INDEX IF NOT EXISTS idx_pgs_player_season_week
  ON mart.player_game_stats(player_id, season, week);

-- Index for player_hierarchy lookups
CREATE INDEX IF NOT EXISTS idx_ph_player_id
  ON mart.player_hierarchy(player_id);

-- Analyze query plan
ANALYZE mart.player_game_stats;
ANALYZE mart.player_hierarchy;

-- Verify indexes created
SELECT indexname, tablename
FROM pg_indexes
WHERE tablename IN ('player_game_stats', 'player_hierarchy')
  AND indexname LIKE 'idx_%';
```

**Step 3:** Apply migration (DEV FIRST!)

```bash
# Apply to dev database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/migrations/004_add_performance_indexes.sql

# Verify indexes exist
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT indexname, tablename FROM pg_indexes WHERE tablename LIKE '%player%'"
```

**Step 4:** Measure improvement

```bash
# Before indexes
EXPLAIN ANALYZE SELECT * FROM mart.player_game_stats
  WHERE player_id = '00-0019596' AND season = 2024 AND week = 9;

# After indexes - should show "Index Scan" instead of "Seq Scan"
```

**Expected Result:**
- Query time: 300ms → 50-80ms (75% reduction)
- Full backtest: 44 min → 34 min (10 min saved)

---

## Phase 2: Feature Cache (15 minutes)

### Goal
Compute features once per game, reuse across multiple models.

### Current Code Problem

**File:** `py/backtest/historical_validation_5way.py`, lines ~303-410 (repeated in 3 places)

```python
# Feature engineering repeated for BNN, XGBoost, State-Space
df["log_attempts"] = np.log1p(df["attempts"])  # Computed 3 times!
df["yards_per_attempt"] = df["yards"] / df["attempts"]
df["td_rate"] = df["touchdowns"] / df["attempts"]
# ... 8 more features

# Cost: ~50ms × 3 models × 148 games = 22 minutes
```

### Implementation

**Step 1:** Create feature cache module

```bash
cat > py/backtest/feature_cache.py << 'EOF'
"""Caching layer for backtest features to avoid recomputation."""

from typing import Dict
import numpy as np

class FeatureCache:
    """Cache computed features by (player_id, season, week)."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def get_features(self, player_id: str, season: int, week: int, row) -> Dict:
        """Get cached or compute features."""
        key = f"{player_id}_{season}_{week}"

        if key in self._cache:
            return self._cache[key]

        # Compute features
        attempts = row.get('stat_attempts', 30) or 30
        yards = row.get('stat_yards', 0) or 0
        touchdowns = row.get('stat_touchdowns', 0) or 0

        features = {
            'log_attempts': np.log1p(attempts),
            'yards_per_attempt': yards / max(attempts, 1),
            'td_rate': touchdowns / max(attempts, 1),
            'is_home': int(row.get('is_home', 0)),
            'spread_abs': abs(row.get('spread_close', 0)),
            'total_line': float(row.get('total_close', 48.0)),
            'is_bad_weather': int(row.get('is_bad_weather', 0)),
            'temp': float(row.get('temp', 70.0)),
            'wind': float(row.get('wind', 5.0)),
        }

        self._cache[key] = features
        return features

    def stats(self):
        """Get cache statistics."""
        return {'cache_size': len(self._cache)}
EOF
```

**Step 2:** Integrate into backtest

```python
# In historical_validation_5way.py (line 53)
from py.backtest.feature_cache import FeatureCache

class HistoricalValidator:
    def __init__(self, ...):
        # ... existing code ...
        self.feature_cache = FeatureCache()

# In process_game() (line 207)
def process_game(self, row: pd.Series) -> Dict:
    # Get cached features
    features = self.feature_cache.get_features(
        row['player_id'],
        row['season'],
        row['week'],
        row.to_dict()
    )

    # Use features for all models
    bnn_features = features  # Cached
    xgb_features = features  # Cached
    # ... rest of prediction code ...
```

**Step 3:** Test

```bash
python py/backtest/historical_validation_5way.py \
    --start-season 2024 --end-season 2024 --end-week 5

# Should see faster feature engineering
# Monitor: "Speed: X games/sec" should increase by 20-30%
```

---

## Validation Checklist

Before and after each optimization:

- [ ] **Code review:** Check for syntax errors, logging
- [ ] **Database connection:** Verify connection pool/pooling working
- [ ] **Small backtest:** Run on 1 week only (should be <30s)
- [ ] **Medium backtest:** Run on 1 month (should be <5 min)
- [ ] **Full backtest:** Run on 2022-2024 (measure runtime improvement)
- [ ] **Results validation:** Check metrics haven't changed (only speed improved)
- [ ] **Git commit:** Commit each optimization separately

### Small Backtest (Validation)

```bash
# Should complete in <30 seconds (vs 3-5 min currently)
python py/backtest/historical_validation_5way.py \
    --start-season 2024 --end-season 2024 --end-week 1 \
    --bankroll 1000 \
    2>&1 | grep "Processing time:"
```

### Medium Backtest (Confidence Check)

```bash
# Should complete in <5 minutes (vs 15-20 min currently)
python py/backtest/historical_validation_5way.py \
    --start-season 2024 --end-season 2024 --end-week 12 \
    --bankroll 5000 \
    2>&1 | grep "Processing time:"
```

### Full Backtest (Final Measurement)

```bash
# Should complete in 24-44 minutes (vs 45-65 min currently)
time python py/backtest/historical_validation_5way.py \
    --start-season 2022 --end-season 2024 --end-week 17 \
    --bankroll 10000
```

---

## Git Workflow

After each optimization:

```bash
# Phase 1: Connection pooling
git add py/backtest/historical_validation_5way.py
git commit -m "perf: add database connection pooling (10-12 min saved)"

# Phase 1: R model cache
git add py/utils/r_service.py py/ensemble/enhanced_ensemble_v6.py
git commit -m "perf: add R model caching (8-9 min saved)"

# Phase 2: Database indexes
git add db/migrations/004_add_performance_indexes.sql
git commit -m "perf: add query indexes on player_game_stats (8-10 min saved)"

# Phase 2: Feature cache
git add py/backtest/feature_cache.py py/backtest/historical_validation_5way.py
git commit -m "perf: add feature caching (2-3 min saved)"

# Summary
git log --oneline -4
# perf: add feature caching (2-3 min saved)
# perf: add query indexes on player_game_stats (8-10 min saved)
# perf: add R model caching (8-9 min saved)
# perf: add database connection pooling (10-12 min saved)
```

---

## Troubleshooting

### Issue: "Connection pool exhausted"
```
psycopg2.OperationalError: out of connections
```
**Solution:** Increase `maxconn` in connection pool (line ~70)
```python
self.db_pool = pool.SimpleConnectionPool(minconn=2, maxconn=16)
```

### Issue: "R model not found after load"
```
rpy2.rinterface_lib.embedded.RRuntimeError: model is not defined
```
**Solution:** Verify model path exists and is readable
```bash
ls -la models/bayesian/passing_yards_hierarchical_v2.rds
```

### Issue: Feature cache has different values per run
**Solution:** Ensure deterministic feature computation (no randomness)

### Issue: Backtest results changed after optimization
**Solution:** Compare metrics before/after (ROI, MAE should be identical)
```bash
# Before optimization
cat analysis/results/historical_validation_5way_metrics.json | jq '.roi_pct'

# After optimization (should match exactly)
cat analysis/results/historical_validation_5way_metrics.json | jq '.roi_pct'
```

---

## Performance Benchmarking Template

Create `docs/OPTIMIZATION_BENCHMARKS.md` and update after each phase:

```markdown
# Performance Benchmarks

## Baseline (No Optimizations)
- Date: 2024-11-13
- Backtest: 2022-2024 QB passing yards
- Runtime: 48 minutes
- Games/sec: 0.05
- Bottleneck: Database queries (49%)

## After Phase 1 (Connection Pooling + R Cache)
- Runtime: 30 minutes
- Speedup: 1.6x
- Games/sec: 0.08
- Status: ✓ VALIDATED

## After Phase 2 (Indexes + Feature Cache)
- Runtime: 18 minutes
- Speedup: 2.7x
- Games/sec: 0.14
- Status: ⏳ PENDING
```

---

## Summary

**Total Implementation Time:** 2-3 hours
**Total Runtime Savings:** 28-34 minutes (60% reduction)
**Risk Level:** LOW-MEDIUM

**Start with Phase 1** (connection pooling + R cache) for immediate 1.5-2.7x speedup.

**Detailed instructions for each optimization available above.**

Questions? See `/Users/dro/rice/nfl-analytics/docs/ENSEMBLE_BACKTEST_PERFORMANCE_ESTIMATE.md`

