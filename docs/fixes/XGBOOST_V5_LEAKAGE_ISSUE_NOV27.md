# XGBoost v5 Data Leakage Issue - Investigation Report

**Date:** November 27, 2025
**Status:** BLOCKED - Requires retraining before production deployment
**Impact:** Models demoted to staging status

---

## Executive Summary

XGBoost v5 models (v5_integrated and v5_tuned) contain **data leakage** that makes them unsuitable for production inference. The models were trained with same-game features (attempts, completions, TDs) that are **not available at prediction time**.

**Result:** Artificially low MAE (1.86-3.74 yards vs realistic ~40 yards) that cannot be achieved in production.

---

## The Problem

### Leaky Features Identified

The v5 models include these features that represent **outcomes** we're trying to predict, not inputs:

```
attempts           # Same-game passing attempts
completions        # Same-game completions
passing_tds        # Same-game touchdowns
interceptions      # Same-game interceptions
yards_per_completion  # Derived from same-game data
avg_air_yards      # Same-game air yards
```

### Why This Is Leakage

**At training time:** Model learns: yards ≈ f(attempts, completions, ...)
- This is nearly deterministic: yards ≈ completions × yards_per_completion

**At inference time:** We don't know attempts/completions before the game!
- We want to predict yards for a FUTURE game
- Model requires features that only exist AFTER the game

### Impact on Metrics

| Model | Reported MAE | Realistic MAE | Inflation |
|-------|-------------|---------------|-----------|
| v5_tuned | 1.86 | ~35-40 | ~20x |
| v5_integrated | 3.74 | ~35-40 | ~10x |
| v4_ridge (no leakage) | 4.70 | 4.70 | 1x |

The v4 model uses only lag features (pass_yards_l3) and opponent stats - information available BEFORE the game.

---

## Root Cause Analysis

### Training Script Issue

`scripts/train_xgboost_player_agg_v5.py` lines 126-171:

```python
# This query pulls SAME-GAME stats (not lagged)
SELECT
    pgs.stat_yards as yards,        # TARGET
    pgs.stat_attempts as attempts,  # LEAKY - same game!
    pgs.stat_completions as completions,  # LEAKY - same game!
    pgs.stat_touchdowns as touchdowns,    # LEAKY - same game!
    ...
```

### What Should Have Been Used

```python
# CORRECT: Use LAG features only
SELECT
    AVG(pgs.stat_yards) OVER (
        PARTITION BY pgs.player_id
        ORDER BY pgs.season, pgs.week
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as yards_l3,  # Safe - prior games only
    ...
```

---

## Resolution Plan

### Option A: Retrain v5.1 Without Leaky Features (Recommended)

1. Remove same-game features from training query
2. Keep valuable features:
   - Player aggregation features (from Bayesian projections)
   - Travel/weather features
   - Coaching (OC/DC) features
   - Causal features (DID, PSM)
   - Lag features (L3 averages)
3. Expected MAE: ~30-35 yards (realistic improvement over v4's 4.70 Ridge + ~35 ensemble)

### Option B: Use Expected Values at Inference

1. Keep v5 model as-is
2. At inference, estimate "expected_attempts" from:
   - Player historical average
   - Opponent defense adjustment
   - Vegas implied totals
3. Risk: Adds another layer of prediction uncertainty

### Option C: Keep v4 (Current Decision)

1. v4 Ridge Residuals remains production model
2. No leakage, honest 4.70 MAE
3. Wait for v5.1 retraining

---

## Registry Updates Made

```json
{
  "xgboost_passing_v5_integrated": {
    "status": "staging",  // Changed from "prod"
    "tags": ["leakage-warning", "needs-retraining"],
    "metrics": {
      "leakage_warning": "same-game features unavailable at inference"
    }
  }
}
```

---

## Lessons Learned

### 1. Feature Availability at Inference Time

**Rule:** Before including any feature, ask: "Is this available BEFORE the game starts?"

| Feature Type | Available at Inference? | Example |
|-------------|------------------------|---------|
| Lag features | ✅ Yes | yards_l3 (prior 3 games) |
| Opponent stats | ✅ Yes | opp_def_epa_l4 |
| Context features | ✅ Yes | is_home, temp, wind |
| Same-game stats | ❌ No | attempts, completions |
| Outcome features | ❌ No | yards (target), TDs |

### 2. Suspiciously Low Error = Red Flag

When MAE drops dramatically (40 → 1.86), investigate:
- Is the model "cheating" with future information?
- Are features correlated with the target by definition?
- Would a simple regression achieve similar results?

### 3. Data Leakage Checklist

Before training any model:
- [ ] All features use only data from PRIOR weeks
- [ ] No same-game statistics in features
- [ ] Target variable excluded from features
- [ ] Temporal split respects game order
- [ ] Run `py/validation/data_leakage_detector.py`

---

## Next Steps

1. ~~Document the issue~~ **DONE** (this file)
2. ~~Update registry status~~ **DONE** (staging + tags)
3. Create v5.1 training script without leaky features
4. Retrain and validate
5. Compare v5.1 MAE to v4 baseline (expect ~30-35 yards)
6. If v5.1 shows real improvement, promote to production

---

## References

- Registry: `models/registry.json` (xgboost_passing_v5_integrated)
- Training script: `scripts/train_xgboost_player_agg_v5.py`
- v5 metadata: `models/xgboost/xgboost_v5_integrated_metadata.json`
- Leakage detector: `py/validation/data_leakage_detector.py`
- Ensemble code: `py/ensemble/enhanced_ensemble_v6_production.py`
