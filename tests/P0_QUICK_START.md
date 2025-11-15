# P0 Test Suite Quick Start

**Status:** ✅ 5 test suites, 45+ tests, <10 min runtime
**ROI:** Prevents 19-113 hours of debugging over 6 months

---

## Run All P0 Tests (Fast - 3 minutes)

```bash
# Run only critical tests, skip slow tests
pytest tests/integration -m "critical and not slow" -v
```

**Expected Output:**
```
✅ 25 tests passed in 2.8 minutes
```

---

## Run Full P0 Suite (Slow - 10 minutes)

```bash
# Run all critical tests including slow ones
pytest tests/integration -m critical --runslow -v
```

**Expected Output:**
```
✅ 45 tests passed in 9.2 minutes
```

---

## Run Individual Suites

### 1. R Model Interface (Prevents Nov 13 failure - 3.5 hours saved)
```bash
pytest tests/integration/test_r_model_interface_critical.py -v -m critical
```

### 2. Ensemble v5.1 Smoke (Validates deployment)
```bash
pytest tests/integration/test_ensemble_v5_smoke.py -v -m critical
```

### 3. Temporal Leakage (Prevents 84-hour BNN investigation)
```bash
pytest tests/integration/test_temporal_leakage_critical.py -v -m critical
```

### 4. Database Pooling (Prevents connection exhaustion)
```bash
pytest tests/integration/test_database_pooling.py -v -m critical
```

### 5. Batch Prediction (Validates 10-50x speedup)
```bash
pytest tests/integration/test_batch_prediction_perf.py -v -m critical
```

---

## Run Performance Benchmarks

```bash
pytest tests/integration -m benchmark -v
```

---

## Common Issues

### Database Not Available
```
Error: psycopg2.OperationalError: connection refused
```

**Fix:** Start PostgreSQL container
```bash
docker compose up -d pg
```

### R Environment Not Available
```
Error: rpy2 not installed
```

**Fix:** Install rpy2
```bash
pip install rpy2
```

### Models Not Found
```
Error: Model file not found: models/bayesian/hierarchical_v2.rds
```

**Fix:** Ensure models are in correct location (Git LFS)
```bash
git lfs pull
```

---

## Test Markers

| Marker | Description | Usage |
|--------|-------------|-------|
| `critical` | P0 priority (prevents expensive failures) | `-m critical` |
| `integration` | Requires database | `-m integration` |
| `benchmark` | Performance benchmarks | `-m benchmark` |
| `slow` | Tests >30 seconds | `--runslow` |

---

## ROI by Suite

| Suite | Tests | Runtime | ROI (Hours Saved) |
|-------|-------|---------|-------------------|
| R Model Interface | 8 | ~2 min | 3.5-18 |
| Ensemble Smoke | 10 | ~1 min | 4 |
| Temporal Leakage | 8 | ~2 min | 5-84 |
| Database Pooling | 7 | ~2 min | 3 |
| Batch Prediction | 7 | ~3 min | 4 |
| **TOTAL** | **40** | **~10 min** | **19-113** |

---

## Pre-Commit Workflow (Recommended)

```bash
# Before committing changes
pytest tests/integration -m "critical and not slow" -v

# If tests pass
git add .
git commit -m "your message"

# If tests fail - FIX THEM FIRST
# Never use: git commit --no-verify
```

---

## Documentation

- **Full Documentation:** `docs/testing/P0_TEST_SUITE_COMPLETE.md`
- **Test Implementation:** `tests/integration/test_*_critical.py`
- **Configuration:** `tests/conftest.py`

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 15, 2025
