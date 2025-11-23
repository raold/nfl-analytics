# Dashboard R Models for Future Games - Root Cause & Fix

**Date:** November 23, 2025
**Status:** ❌ CRITICAL - R models can't predict future games → 2/5 components fail → placeholder fallback
**Impact:** 13/14 Week 12 games show placeholders (240 ± 50) instead of real predictions

---

## Root Cause Analysis

### Issue 1: R Models Require Historical Play Data

**Error in logs:**
```
Component state_space_v2 failed: State-space v2 prediction not found in database
Component hierarchical_v2 failed: Hierarchical v2 prediction not found in database
Component informative_priors_v2 failed: Informative priors v2 prediction not found in database
Insufficient ensemble components loaded: 2 valid (minimum 3 required)
```

**Why it fails:**
1. **Today**: November 23, 2025 (Week 12 games in progress)
2. **R models** (State-Space v2, Hierarchical v2, Informative Priors v2) generate predictions from **historical play-by-play data**
3. **Future games** (13/14 Week 12 games) have `home_score = NULL` → haven't been played yet → **no play data** → R can't make predictions
4. **XGBoost** and **Hybrid** models CAN predict (use rolling 3-game features, not play-level data)
5. **Ensemble validation**: Requires **≥3/5 components** (`enhanced_ensemble_v6_production.py:552`)
6. **Result**: Only 2/5 components work (XGBoost v4 Ridge 23% + Hybrid 10% = 33%) → raises RuntimeError → falls back to placeholder

### Issue 2: Database Only Has R Predictions Through Week 6

**Query result:**
```sql
SELECT model_name, season, week, COUNT(*) as predictions
FROM ml.r_model_predictions
GROUP BY model_name, season, week
ORDER BY season DESC, week DESC LIMIT 20;

 model_name            | season | week | predictions
-----------------------+--------+------+-------------
 hierarchical_v2       |   2025 |    6 |           2
 informative_priors_v2 |   2025 |    6 |           2
 state_space_v2        |   2025 |    6 |           2
```

The batch R prediction script (`R/batch_generate_r_predictions_optimized.R`) **cannot** generate predictions for Week 12 because:
- It needs historical play data (FROM plays table)
- Week 12 games haven't finished yet → no plays data
- Attempting to run it returns: `⚠️ No player-games found for given criteria`

---

## Current Behavior

**Game:** SEA @ TEN (2025 Week 12, kickoff 18:00 today)

1. Dashboard calls `/api/v1/predict/game` with game_id `2025_12_SEA_TEN`
2. API identifies starting QBs from season stats (C.Ward for TEN: 330 attempts)
3. Ensemble tries to load predictions:
   - **XGBoost v4 Ridge**: ✅ SUCCESS (has 3-game rolling features)
   - **Hybrid Static-Incremental**: ✅ SUCCESS (has rolling features)
   - **State-Space v2**: ❌ FAIL (no prediction in database)
   - **Hierarchical v2**: ❌ FAIL (no prediction in database)
   - **Informative Priors v2**: ❌ FAIL (no prediction in database)
4. Ensemble validation: `len(valid_predictions) = 2 < 3` → raises RuntimeError
5. API catches error, falls back to: `home_yards = 240.0, home_uncertainty = 50.0`
6. Dashboard displays: **"TEN QB: 240 ± 50 yds"** (PLACEHOLDER)

**Result**: Mixed predictions across Week 12 games:
- Some QBs get real predictions (178 ± 24)
- Some QBs get placeholders (240 ± 50)
- Inconsistent user experience

---

## Proposed Fix

### Option A: Lower Minimum Component Threshold (RECOMMENDED)

**File:** `py/ensemble/enhanced_ensemble_v6_production.py`
**Line:** 552

**Change:**
```python
# BEFORE
if len(valid_predictions) < 3:
    raise RuntimeError(...)

# AFTER
if len(valid_predictions) < 2:
    raise RuntimeError(...)
```

**Rationale:**
- For **future games**, R models physically cannot provide predictions (no play data yet)
- **XGBoost + Hybrid** (33% combined weight) can still make reasonable predictions
- **Weight renormalization** (lines 562-579) handles the missing components:
  - XGBoost: 23% → 70% (renormalized)
  - Hybrid: 10% → 30% (renormalized)
- Better to show **Python-only predictions** than **hardcoded placeholders**

**Expected behavior after fix:**
- SEA @ TEN: Shows XGBoost + Hybrid prediction (e.g., "TEN QB: 172 ± 30 yds")
- Uncertainty increases (30 vs 24) to reflect missing R components
- NO MORE PLACEHOLDERS for future games

### Option B: Pre-generate R Predictions from Projected Stats (COMPLEX)

**Approach:**
1. Create projected stats for Week 12 QBs (rolling averages, trend extrapolation)
2. Feed projected stats into R models to generate "synthetic" posterior distributions
3. Store in `ml.r_model_predictions` table
4. Ensemble loads these synthetic predictions

**Pros:**
- Maintains full 5/5 component ensemble
- Preserves uncertainty quantification from R models

**Cons:**
- Requires building a "projection engine" for future weeks
- Synthetic predictions may overfit to recent trends
- More engineering complexity
- Still can't capture true Bayesian priors for unseen game contexts

### Option C: Separate Future/Historical Prediction Modes (OVERKILL)

**Approach:**
1. Detect if game is in future (`kickoff > NOW()`)
2. Use Python-only ensemble for future games
3. Use full 5-way ensemble for historical games

**Pros:**
- Clear separation of concerns
- Can optimize each mode independently

**Cons:**
- Code duplication
- Different prediction distributions for historical vs future
- Harder to maintain consistency

---

## Recommendation

**Implement Option A**: Lower threshold to 2 components

**Why:**
1. **Simple**: One-line code change
2. **Safe**: Weight renormalization prevents bias
3. **Fast**: No additional infrastructure needed
4. **Consistent**: Same prediction logic for all games
5. **Transparent**: User sees uncertainty increases when R models unavailable

**Implementation:**
```bash
# Edit ensemble file
vim py/ensemble/enhanced_ensemble_v6_production.py  # Line 552: change "< 3" to "< 2"

# Rebuild API container
docker-compose build api && docker-compose restart api

# Test Week 12 predictions
curl -X POST 'http://localhost:8000/api/v1/predict/game' \
  -H 'Content-Type: application/json' \
  -d '{"game_id": "2025_12_SEA_TEN", "season": 2025, "week": 12}'

# Expected: Real prediction (not 240 ± 50), higher uncertainty
```

---

## Next Steps

1. ✅ Root cause identified (R models need play data, future games don't have it)
2. ✅ Implement Option A fix (lower threshold to 2 components)
3. ✅ Rebuild API container (discovered duplicate validation in API endpoint)
4. ✅ Test predictions for all Week 12 games (all showing real values)
5. ⏳ User verifies dashboard shows real predictions (not placeholders)
6. ✅ Monitor uncertainty values (16-24 yards std, varying by QB/game)
7. ✅ Document trade-off (Python-only for future, full ensemble for historical)

**Status:** ✅ COMPLETE - See `docs/fixes/DASHBOARD_THRESHOLD_FIX_COMPLETE_NOV23.md` for full summary

---

## Files Involved

- **Ensemble Logic:** `py/ensemble/enhanced_ensemble_v6_production.py:552`
- **API Endpoint:** `py/api/routes/game_predictions.py:1280-1337`
- **Dashboard Display:** `dashboard/app.py:1887-1912`
- **R Batch Script:** `R/batch_generate_r_predictions_optimized.R` (can't run for future weeks)

---

**Impact:** ❌ CRITICAL - Dashboard completely non-functional for future games (core use case)
**Effort:** 5 minutes (1-line fix + rebuild)
**Priority:** P0 - Blocking all Week 12 predictions
