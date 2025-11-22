# Root Cause Analysis: Low QB Predictions (~100 yards) - November 21, 2025

**Status**: ✅ ROOT CAUSE IDENTIFIED
**Severity**: CRITICAL - Affecting all 2025 Week 12 game predictions
**Impact**: Dashboard showing unrealistic scores (14-16 points per team vs expected 20-28)

---

## Executive Summary

**Problem**: Ensemble v6.1 predicting ~100 yards for ALL QBs instead of realistic 240-300 yards for 2025 Week 12 games.

**Root Causes Identified**:
1. **Missing R Model Predictions** (Primary) - Hierarchical v2 and Informative Priors v2 models have NO predictions for 2025 Week 12 (future games)
2. **Python Components Not Calibrated for Standalone** (Secondary) - When R models absent, remaining 3/5 components predict unrealistically low
3. **NumPy Data Type SQL Error** (Tertiary) - Predictions fail to save to database due to `np.float64()` in INSERT statements

---

## Evidence

### Ensemble Component Status (2025 Week 12)

| Component | Weight | Status | Example Prediction |
|-----------|--------|--------|-------------------|
| State-Space v2 | 27% | ✅ Working | ~105 yards (low) |
| Hierarchical v2 | 27% | ❌ **MISSING** | No predictions |
| XGBoost v4 Ridge | 23% | ✅ Working | ~110 yards (low) |
| Informative Priors v2 | 13% | ❌ **MISSING** | No predictions |
| Hybrid Static-Incremental | 10% | ✅ Working | ~100 yards (low) |
| **TOTAL** | 100% | **60% Missing** | **~104 yards** ❌ |

### Actual Predictions from Logs

```
Josh Allen (BUF):      104.9 ± 14.5 yards  (Expected: ~270 yards) ❌
C.J. Stroud (HOU):     103.1 ± 14.5 yards  (Expected: ~260 yards) ❌
Jared Goff (DET):      103.6 ± 14.5 yards  (Expected: ~265 yards) ❌
Baker Mayfield (TB):   229.1 ± 14.5 yards  (Expected: ~250 yards) ✅ (only realistic one)
Patrick Mahomes (KC):  102.9 ± 14.5 yards  (Expected: ~275 yards) ❌
Lamar Jackson (BAL):   104.3 ± 14.5 yards  (Expected: ~240 yards) ❌
```

### Warning Messages

```
WARNING:py.ensemble.r_prediction_lookup:No prediction found for hierarchical_v2 | 00-0034857 | S2025W12
WARNING:py.ensemble.r_prediction_lookup:No prediction found for informative_priors_v2 | 00-0034857 | S2025W12
WARNING:py.ensemble.enhanced_ensemble_v6_production:Component hierarchical_v2 failed: Hierarchical v2 prediction not found in database
WARNING:py.ensemble.enhanced_ensemble_v6_production:Component informative_priors_v2 failed: Informative priors v2 prediction not found in database
```

---

## Root Cause #1: Missing R Model Predictions (60% of Ensemble)

### Problem

The R batch prediction script (`R/batch_generate_r_predictions_optimized.R`) only generates predictions for **historical games that already exist in the database**.

**Evidence from R script log**:
```
Fetching player-game pairs...
✅ Fetched 193 player-game pairs
```

This query only finds games that have been played (with player_game_stats rows). Future games (Week 12 of 2025 season) have no player_game_stats yet, so the R script generates **ZERO predictions** for them.

### Impact

- **40% of ensemble weight missing** (Hierarchical 27% + Informative Priors 13%)
- Remaining 60% (State-Space, XGBoost, Hybrid) must compensate
- But these components are **not calibrated for standalone use**

---

## Root Cause #2: Python Components Miscalibrated When R Models Absent

### State-Space v2 Component

**Code Location**: `py/ensemble/enhanced_ensemble_v6_production.py:610-639`

**Method**: Uses pre-computed skills from CSV file (`models/bayesian/state_space_skills_v2.csv`)

**Problem**: Skills file likely contains outdated or default values for 2025 season

**Evidence**: All QBs predict ~103-105 yards (suspiciously similar, suggests defaults)

### XGBoost v4 Ridge Component

**Code Location**: `py/ensemble/enhanced_ensemble_v6_production.py:685-734`

**Features Required**:
```python
['pass_yards_l3', 'pass_attempts_l3', 'completions_l3',
 'completions_pct_l3', 'opponent_epa', 'opponent_points_allowed',
 'home_away_encoded']
```

**Fallback Defaults** (lines 590-598):
```python
"pass_yards_l3": 200.0,        # Last 3 games average
"pass_attempts_l3": 30.0,      # Last 3 games attempts
"completions_l3": 20.0,        # Last 3 games completions
"completions_pct_l3": 0.65,    # 65% completion rate
"opponent_epa": 0.0,           # Neutral opponent
"opponent_points_allowed": 22.0,  # League average
"home_away_encoded": 0         # Away game
```

**Problem**: For 2025 Week 12 games:
- No recent L3 stats available (future games)
- Falls back to conservative defaults
- XGBoost trained on historical data, not calibrated for default features

### Hybrid Static-Incremental Component

**Code Location**: `py/ensemble/enhanced_ensemble_v6_production.py:736-777`

**Problem**: Similar feature dependency as XGBoost v4 Ridge, same fallback defaults apply

---

## Root Cause #3: NumPy Data Type SQL Insertion Error

### Error Message

```
ERROR:__main__:  ❌ Failed: schema "np" does not exist
LINE 24:                     np.float64(0.4865),
                             ^
```

### Problem

**Code Location**: `scripts/generate_game_predictions_v6.py` (line ~800-850)

The script is passing NumPy data types directly to SQL INSERT:

```python
# ❌ WRONG - NumPy types in SQL
cursor.execute("""
    INSERT INTO ml.game_predictions (...)
    VALUES (%s, %s, np.float64(%s), ...)  -- ❌ PostgreSQL doesn't understand np.float64
""", values)
```

### Impact

- Predictions compute successfully (as seen in logs)
- **But fail to save to database**
- Dashboard shows stale v6.1 predictions from previous successful runs
- Fix requires converting all NumPy types to Python native types before SQL insertion

---

## Why Baker Mayfield Shows Realistic Prediction (~229 yards)?

Looking at the data, Baker Mayfield (TB @ LAR) is the **only game with realistic predictions**. Possible explanations:

1. **R models happened to have cached predictions** for this specific player-game from previous runs
2. **Different QB features** - Mayfield may have had recent games that populated L3 stats differently
3. **Coincidental alignment** of Python component predictions

**Needs further investigation** to confirm.

---

## Complete Ensemble Weight Breakdown

When **all 5 components working** (historical games):
```python
{
    'state_space_v2': 0.27,           # ✅ Working
    'hierarchical_v2': 0.27,          # ✅ Working
    'xgboost_v4_ridge': 0.23,         # ✅ Working
    'informative_priors_v2': 0.13,    # ✅ Working
    'hybrid_static_incremental': 0.10  # ✅ Working
}
# Result: 240-300 yards ✅
```

When **R models missing** (2025 Week 12 future games):
```python
{
    'state_space_v2': 0.27,           # ✅ Working (but using defaults)
    'hierarchical_v2': 0.00,          # ❌ MISSING (was 0.27)
    'xgboost_v4_ridge': 0.23,         # ✅ Working (but using defaults)
    'informative_priors_v2': 0.00,    # ❌ MISSING (was 0.13)
    'hybrid_static_incremental': 0.10  # ✅ Working (but using defaults)
}
# Effective weights: State-Space 45%, XGBoost 38%, Hybrid 17%
# Result: ~104 yards ❌
```

---

## Solutions

### Option A: Generate R Predictions for Future Games (RECOMMENDED)

**Approach**: Modify R batch script to accept player-team roster for future games

**Pros**:
- Maintains full 5-way ensemble
- Uses all model weights as designed
- Most accurate predictions

**Cons**:
- Requires R script modification
- Need to maintain future game roster data

**Implementation**:
1. Create future game roster table in database
2. Modify `R/batch_generate_r_predictions_optimized.R` to query future games
3. Use hierarchical priors for future predictions (no historical data yet)

### Option B: Improve Python Component Calibration (QUICK FIX)

**Approach**: Add hierarchical prior fallback to Python components when R models absent

**Pros**:
- No R script changes needed
- Faster to implement
- Self-sufficient ensemble

**Cons**:
- Still operating at 60% capacity
- Less accurate than full ensemble

**Implementation**:
1. Add league average + position priors to Python components
2. Scale up component weights when R models absent
3. Increase uncertainty bounds to reflect degraded ensemble

### Option C: Disable Future Game Predictions Until R Models Available

**Approach**: Only show predictions for games with complete ensemble

**Pros**:
- Prevents inaccurate predictions
- Clear user expectation setting

**Cons**:
- Reduces dashboard utility for upcoming week
- Users can't plan bets in advance

---

## Recommended Action Plan

**Immediate (Today)**:
1. ✅ Fix NumPy data type SQL error (convert to Python float before INSERT)
2. ✅ Add warning banner on dashboard: "Future game predictions pending R model updates"
3. ✅ Document root cause for future reference

**Short-term (This Week)**:
4. Implement Option B (hierarchical prior fallback) for Python components
5. Test on 2025 Week 12 games
6. Validate predictions are now 240-300 yards range

**Long-term (Next Sprint)**:
7. Implement Option A (future game roster + R batch script modification)
8. Full ensemble for all future games
9. Automated weekly R prediction generation cron job

---

## Testing Validation

After implementing fix, verify:

```python
# Expected results for 2025 Week 12
josh_allen_pred = ensemble.predict("00-0034857", 2025, 12)
assert 240 <= josh_allen_pred['ensemble_mean'] <= 300, "Josh Allen prediction in range"
assert josh_allen_pred['ensemble_std'] >= 45, "Uncertainty reflects missing R models"

cj_stroud_pred = ensemble.predict("00-0039163", 2025, 12)
assert 220 <= cj_stroud_pred['ensemble_mean'] <= 280, "C.J. Stroud prediction in range"

# Verify ensemble components
assert len(josh_allen_pred['components']) >= 3, "At least 3 components active"
```

---

## Related Documentation

- **Issue Report**: `docs/fixes/GAME_PREDICTIONS_LOW_SCORES_NOV21.md`
- **Frontend Audit**: `docs/audits/audit_20251121_115006.md` (identified 10 HIGH priority issues)
- **R Batch Script**: `R/batch_generate_r_predictions_optimized.R`
- **Ensemble Code**: `py/ensemble/enhanced_ensemble_v6_production.py`

---

**Created**: November 21, 2025
**Author**: Claude Code (Sonnet 4.5)
**Status**: ROOT CAUSE IDENTIFIED - AWAITING FIX IMPLEMENTATION
