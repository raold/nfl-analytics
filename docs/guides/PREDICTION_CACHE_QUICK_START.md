# Prediction Cache Manager - Quick Start Guide

**5-Minute Guide to PostgreSQL Cache for R Model Predictions**

## What Is It?

A high-performance PostgreSQL cache that replaces slow rpy2 calls (50-200ms) with fast database lookups (<1ms), delivering 50-200x speedup.

## Quick Setup (3 Steps)

### 1. Run Database Migration

```bash
psql -U dro -d devdb01 -f db/migrations/047_create_r_model_predictions.sql
```

### 2. Create Indexes

```bash
./scripts/manage_prediction_cache.sh optimize
```

### 3. Verify Setup

```bash
./scripts/manage_prediction_cache.sh stats
```

Expected output: Empty cache (Total Predictions: 0)

## Common Operations

### Cache Predictions (After Model Training)

```python
from py.utils.prediction_cache_manager import PredictionCacheManager
import pandas as pd

manager = PredictionCacheManager()

# Prepare predictions DataFrame
predictions = pd.DataFrame({
    'player_id': ['P001', 'P002', 'P003'],
    'season': [2024, 2024, 2024],
    'week': [10, 10, 10],
    'prediction': [250.5, 180.3, 120.7]
})

# Bulk insert
count = manager.cache_predictions(
    predictions,
    model_name='state_space_v2',
    stat_type='passing_yards',
    validate=True  # Range checks, NaN detection
)

print(f"Cached {count} predictions")
```

### Fast Lookup (In Ensemble)

```python
# Try cache first (<1ms)
cached = manager.get_cached_prediction(
    player_id='P001',
    stat_type='passing_yards',
    season=2024,
    week=10,
    model_name='state_space_v2',
    max_age_hours=24  # Only fresh predictions
)

if cached is not None:
    return cached  # Cache hit
else:
    # Cache miss - compute via rpy2 (slow)
    return compute_prediction(...)
```

### Check Statistics

```bash
./scripts/manage_prediction_cache.sh stats
```

**Key Metrics:**
- Hit Rate: Target >95%
- Avg Lookup Time: Target <1ms
- Cache Size: Should be <500MB

### Invalidate Stale Predictions

```bash
# After model retraining
./scripts/manage_prediction_cache.sh invalidate --model state_space_v2

# Remove old predictions (>7 days)
./scripts/manage_prediction_cache.sh invalidate --older-than 168
```

### Validate Cache Integrity

```bash
# Quick validation (random sample)
./scripts/manage_prediction_cache.sh validate --sample-size 10000

# Full validation (may be slow)
./scripts/manage_prediction_cache.sh validate
```

## Ensemble Integration Pattern

```python
from py.utils.prediction_cache_manager import PredictionCacheManager

class EnhancedEnsemble:
    def __init__(self):
        self.cache = PredictionCacheManager()

    def get_state_space_prediction(self, player_id, season, week):
        # Cache-first pattern
        cached = self.cache.get_cached_prediction(
            player_id, 'passing_yards', season, week,
            model_name='state_space_v2', max_age_hours=24
        )

        if cached is not None:
            return cached  # <1ms

        # Cache miss - compute and cache
        prediction = self._compute_rpy2_prediction(player_id, season, week)

        self.cache.cache_predictions(
            pd.DataFrame([{
                'player_id': player_id,
                'season': season,
                'week': week,
                'prediction': prediction
            }]),
            model_name='state_space_v2',
            stat_type='passing_yards'
        )

        return prediction
```

## Weekly Maintenance

```bash
# 1. Invalidate stale predictions
./scripts/manage_prediction_cache.sh invalidate --older-than 168

# 2. Optimize performance
./scripts/manage_prediction_cache.sh optimize

# 3. Validate integrity
./scripts/manage_prediction_cache.sh validate --sample-size 10000

# 4. Check statistics
./scripts/manage_prediction_cache.sh stats
```

## Troubleshooting

### Low Hit Rate (<90%)

**Cause:** Missing predictions in cache

**Solution:**
```bash
# Check coverage
./scripts/manage_prediction_cache.sh stats

# Backfill missing predictions
python -c "
from py.utils.prediction_cache_manager import PredictionCacheManager
# ... backfill code ...
"
```

### Slow Lookups (>2ms)

**Cause:** Missing indexes or fragmentation

**Solution:**
```bash
./scripts/manage_prediction_cache.sh optimize
```

### Validation Failures

**Cause:** Invalid predictions (NaN, out of range, duplicates)

**Solution:**
```bash
# Get detailed report
./scripts/manage_prediction_cache.sh validate --sample-size 10000

# Fix identified issues
./scripts/manage_prediction_cache.sh invalidate --model <problematic_model>

# Re-cache with validation
# (Python code to retrain and cache)
```

## Performance Targets

| Metric | Target | Check Command |
|--------|--------|---------------|
| Lookup Time | <1ms | `stats` → "Avg Lookup Time" |
| Hit Rate | >95% | `stats` → "Hit Rate" |
| Cache Size | <500MB | `stats` → "Cache Size" |
| Validation | 100% pass | `validate` → exit code 0 |

## Migration from rpy2

```python
from py.utils.prediction_cache_manager import PredictionCacheManager

manager = PredictionCacheManager()

# Define rpy2 wrapper
def get_rpy2_predictions(season, stat_type, player_ids=None):
    # Call R model via rpy2
    return predictions_df

# Backfill 3 seasons
total, cached = manager.backfill_from_rpy2(
    model_name='state_space_v2',
    stat_type='passing_yards',
    seasons=[2022, 2023, 2024],
    rpy2_prediction_function=get_rpy2_predictions
)

print(f"Migrated {cached}/{total} predictions")
```

## Documentation

- **Comprehensive Guide:** `docs/guides/PREDICTION_CACHE_MANAGER.md`
- **Integration Examples:** `examples/prediction_cache_integration.py`
- **Milestone Summary:** `docs/milestones/PREDICTION_CACHE_MANAGER_COMPLETE.md`

## CLI Reference

```bash
# Show all options
./scripts/manage_prediction_cache.sh --help

# Common commands
./scripts/manage_prediction_cache.sh stats
./scripts/manage_prediction_cache.sh invalidate --model <name>
./scripts/manage_prediction_cache.sh optimize
./scripts/manage_prediction_cache.sh validate --sample-size 10000
```

## Expected Results

**Before (rpy2):**
- Backtest: 18 hours
- Prediction latency: 50-200ms

**After (PostgreSQL Cache):**
- Backtest: 30-60 minutes (18-36x faster)
- Prediction latency: <1ms (50-200x faster)
- Cache hit rate: >95%

---

**Status:** Production-Ready ✅
**For Help:** See `docs/guides/PREDICTION_CACHE_MANAGER.md`
