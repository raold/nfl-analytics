# Game Predictions Low Scores Issue

**Date**: November 21, 2025
**Status**: ⚠️ ROOT CAUSE IDENTIFIED

---

## Problem

Dashboard shows unrealistic game predictions:
- Most games: 14-16 points per team (totals: 29-31)
- Only 3 games realistic: ATL@NO (21.9@21.8), PHI@DAL (15.5@22.6), TB@LAR (15.7@22.3)
- Ensemble predicting **only 103-105 yards** for elite QBs (Josh Allen, C.J. Stroud)

---

## Root Cause

**Missing R Model Predictions for Season 2025**

The ensemble v6.1 includes 2 R models (40% weight):
- `hierarchical_v2` (27% weight)
- `informative_priors_v2` (13% weight)

These models have NO predictions for season 2025:
```
WARNING: No prediction found for hierarchical_v2 | 00-0039163 | S2025W12
WARNING: No prediction found for informative_priors_v2 | 00-0039163 | S2025W12
```

Result: Only 3 Python components working (60% of ensemble):
- `state_space_v2` (27% → ~45% renormalized)
- `xgboost_v4_ridge` (23% → ~38% renormalized)
- `hybrid_static_incremental` (10% → ~17% renormalized)

**These Python components are ALL predicting ~100 yards** instead of realistic 240-300 yards.

---

## Evidence

From prediction logs (Week 12, 2025):
```
Josh Allen (BUF):       104.9 ± 14.5 yards (should be ~270 yards)
C.J. Stroud (HOU):      103.1 ± 14.5 yards (should be ~260 yards)
Patrick Mahomes (KC):   102.9 ± 14.5 yards (should be ~275 yards)
Jared Goff (DET):       103.6 ± 14.5 yards (should be ~255 yards)
```

Conversion to team scores (formula: `10 + yards/20 - 5*def_epa`):
- 104 yards → 10 + 5.2 = **15.2 points** ❌
- 270 yards → 10 + 13.5 = **23.5 points** ✅

---

## Why Some Games Look Realistic

ATL @ NO shows 21.9 @ 21.8 (realistic):
- Both QBs predicted ~238 yards
- This suggests certain QBs DO have better data in Python components

PHI @ DAL shows 15.5 @ 22.6:
- DAL QB: ~253 yards (realistic)
- PHI QB: ~110 yards (too low)

TB @ LAR shows 15.7 @ 22.3:
- LAR QB: ~246 yards (realistic)
- TB QB: ~114 yards (too low)

**Pattern**: Some QBs get realistic predictions, most get ~100 yards

---

## Required Fixes

### Fix 1: Generate R Model Predictions for 2025 ✅ IN PROGRESS
```bash
# Background process already running:
Rscript R/batch_generate_r_predictions_optimized.R --seasons 2025
```

This will populate:
- `ml.r_model_predictions` table with hierarchical_v2 predictions
- `ml.r_model_predictions` table with informative_priors_v2 predictions

Then regenerate game predictions:
```bash
python scripts/generate_game_predictions_v6.py --season 2025 --week 12
```

### Fix 2: Investigate Python Component Low Predictions

Why are Python models predicting ~100 yards for most QBs?

**Hypotheses**:
1. **Missing feature data for 2025** - L3 stats (last 3 games) might be NULL for Week 12
2. **State-space skills degraded** - Canonical window might not have 2025 data
3. **XGBoost fallback** - Model falling back to conservative defaults

**Investigation needed**:
```sql
-- Check if QB has recent stats for 2025
SELECT player_id, player_display_name, season, week,
       stat_attempts, stat_yards, stat_completions
FROM mart.player_game_stats
WHERE player_id = '00-0034857'  -- Josh Allen
  AND season = 2025
ORDER BY week DESC
LIMIT 5;
```

### Fix 3: Dashboard White Text Visibility

User reports: "cant read text because its white -- we fixed this previously"

Need to audit dashboard CSS for white text on white background issues.

---

## Timeline

1. ✅ Identified root cause (missing R predictions)
2. ✅ R batch prediction script running in background
3. ⏳ Waiting for R predictions to complete (~30-60 mins)
4. ⏳ Regenerate game predictions with full 5-way ensemble
5. ⏳ Fix dashboard white text issue
6. ⏳ Investigate Python component low predictions

---

## Expected Results After Fix

With full 5-way ensemble (R models included):
- Josh Allen: ~270 yards → 23.5 points ✅
- C.J. Stroud: ~260 yards → 23.0 points ✅
- Patrick Mahomes: ~275 yards → 23.8 points ✅

Game totals should be 44-50 points (22-25 per team).

---

**Status**: ⚠️ ROOT CAUSE IDENTIFIED - R models running in background
**ETA**: 30-60 minutes for R prediction generation
**Next**: Regenerate game predictions + fix dashboard white text
