# XGBoost v3 with Opponent Strength Features - Training Results

**Date:** November 14, 2025
**Author:** NFL Analytics Research Team
**Status:** Model Trained Successfully, Performance Analysis Complete

---

## Executive Summary

Trained XGBoost v3 with 3 new opponent strength features. Model trained successfully but showed **unexpected regression** compared to v2 baseline.

**Key Finding:** Adding opponent strength features **decreased performance by 6.6%** (40.20 → 42.86 MAE).

---

## Model Comparison

| Metric | v2 Baseline (13 features) | v3 with Opponent Strength (16 features) | Change |
|--------|---------------------------|------------------------------------------|--------|
| **MAE** | **40.20 yards** | **42.86 yards** | **+2.66 yards (+6.6%)** ⚠️ |
| **RMSE** | 123.14 yards | 127.09 yards | +3.95 yards (+3.2%) |
| **R²** | 0.8545 | 0.8450 | -0.0095 (-1.1%) |
| **Features** | 13 | 16 | +3 |
| **Training Time** | 0.012 min | 0.020 min | +62% |
| **Best Iteration** | 416 | 637 | +53% |
| **Model Size** | 2.2 MB | 3.4 MB | +55% |

**Verdict:** v2 baseline remains superior. v3 opponent strength features did not improve performance.

---

## New Features Added (v3)

1. **opp_def_epa_exp** - Exponentially weighted opponent defensive EPA (α=0.8)
   - Feature importance rank: #11 (gain: 170.1)
   - Coverage: 65.0% of samples (missing for 2016-2017, week 1 games)

2. **opp_pass_def_rank** - Opponent pass defense percentile rank (0-100)
   - Feature importance rank: #12 (gain: 100.0)
   - Coverage: 65.0% of samples

3. **matchup_difficulty** - Composite difficulty index (0-100)
   - Feature importance rank: #14 (gain: 91.1)
   - Coverage: 65.0% of samples

**Note:** All 3 new features ranked in bottom 6 of 16 total features by gain importance.

---

## Feature Importance Analysis

### Top 10 Features (by gain)

| Rank | Feature | Gain (v3) | Gain (v2) | Change | Notes |
|------|---------|-----------|-----------|--------|-------|
| 1 | log_attempts | 3,305,988 | 4,724,878 | -30% | Still #1 but weakened |
| 2 | week_in_season | 1,876,403 | 1,937,953 | -3% | Stable |
| 3 | total_line | 148,052 | 286,197 | -48% | Significant drop |
| 4 | spread_abs | 86,134 | 156,197 | -45% | Large decline |
| 5 | yards_per_attempt | 69,384 | 70,944 | -2% | Stable |
| 6 | td_rate | 37,081 | 80,800 | -54% | Major drop |
| 7 | yards_L3 | 4,799 | 5,120 | -6% | Slight decline |
| 8 | td_rate_L3 | 3,618 | 6,892 | -48% | Large drop |
| 9 | yards_per_attempt_L3 | 2,264 | 3,087 | -27% | Moderate decline |
| 10 | is_home | 2,056 | 2,531 | -19% | Decline |

### New Opponent Strength Features (bottom tier)

| Rank | Feature | Gain | % of Top Feature |
|------|---------|------|------------------|
| 11 | opp_def_epa_exp [NEW] | 170.1 | 0.005% |
| 12 | opp_pass_def_rank [NEW] | 100.0 | 0.003% |
| 14 | matchup_difficulty [NEW] | 91.1 | 0.003% |

**Observation:** All 3 opponent strength features have negligible importance compared to top features (2-3 orders of magnitude lower).

---

## Hypothesis: Why Did Performance Decline?

### 1. Feature Coverage Issues (35% Missing Data)
- Opponent strength features only available for 2018-2024 games (week 2+)
- Missing for:
  - 2016-2017 seasons (573 + 575 = 1,148 games, 21.5% of training data)
  - Week 1 of every season (no prior defensive history)
  - Total missing: ~1,931 / 5,510 games (35%)
- Filled with median values (neutral stance), may introduce noise

### 2. Redundancy with Existing Features
- `opp_def_epa` already exists in base query (from `games.away_def_epa_last_4`)
- New exponentially weighted version (`opp_def_epa_exp`) may be redundant
- Model may not benefit from similar signals

### 3. Overfitting to Weak Signals
- Low feature importance suggests weak predictive power
- More features + longer training (637 vs 416 iterations) = potential overfitting
- R² drop (-1.1%) supports overfitting hypothesis

### 4. Model Complexity Without Benefit
- 55% larger model size (2.2 → 3.4 MB)
- 62% longer training time
- 53% more boosting rounds
- No accuracy gain to justify complexity

---

## Data Quality Analysis

### Opponent Strength Feature Coverage

```
Total QB games: 5,510
Opponent strength records: 3,710 (67.3%)

Feature coverage:
- opp_def_epa_exp: 3,579 / 5,510 (65.0%)
- opp_pass_def_rank: 3,579 / 5,510 (65.0%)
- matchup_difficulty: 3,579 / 5,510 (65.0%)

Missing data filled with median:
- opp_def_epa_exp: -0.0102
- opp_pass_def_rank: 50.0000
- matchup_difficulty: 50.0000
```

**Issue:** Filling 35% of samples with median values creates "average opponent" signal for 1/3 of training data, diluting predictive value.

---

## Training Details

### v3 Configuration
- **Training window:** 2018-2024 (full weight) + 2015-2017 (exponential decay H=4)
- **Test set:** 2025 season (182 games)
- **Hyperparameters:** Same as v2 (max_depth=7, lr=0.05, etc.)
- **Early stopping:** 75 rounds
- **Best iteration:** 637 (vs v2: 416)

### Training Set Distribution
```
2015: 0 games, weight=0.472
2016: 573 games, weight=0.607
2017: 575 games, weight=0.779
2018-2024: 4,180 games, weight=1.000

Total train samples: 5,328
Mean weight: 0.934
```

---

## Recommendations

### Short-term Actions

1. **Use v2 Baseline for Production**
   - v2 MAE: 40.20 yards (SUPERIOR)
   - v3 MAE: 42.86 yards (INFERIOR)
   - Keep 30% ensemble weight on v2, not v3

2. **Archive v3 Results**
   - Document as unsuccessful experiment
   - Save model for future reference but do NOT deploy

3. **Investigate Alternative Approaches**
   - Option A: Use opponent strength features ONLY for 2018+ games (exclude 2016-2017)
   - Option B: Impute missing values with better method (e.g., team average EPA)
   - Option C: Create derived features (e.g., QB strength - opponent strength)

### Long-term Research Directions

1. **Better Opponent Metrics**
   - Per-QB opponent history (not team-level)
   - Strength-of-schedule adjustments
   - Matchup-specific features (QB vs specific defensive schemes)

2. **Temporal Alignment**
   - Ensure opponent strength features available for ALL training samples
   - Consider dropping 2016-2017 from training if opponent data unavailable

3. **Feature Interaction**
   - Test QB skill × opponent strength interactions
   - Non-linear transformations (e.g., QB percentile - opponent percentile)

4. **Ensemble Integration Alternative**
   - Train separate model for post-2018 games with full opponent data
   - Use v2 for pre-2018, v3 for post-2018 in ensemble

---

## Files Generated

### Model Artifacts
- **Model:** `/Users/dro/rice/nfl-analytics/models/xgboost/passing_xgboost_v3_with_opponent_strength.pkl` (3.4 MB)
- **Metadata:** `/Users/dro/rice/nfl-analytics/models/xgboost/passing_xgboost_v3_with_opponent_strength_metadata.json` (2.3 KB)

### Training Script
- **Script:** `/Users/dro/rice/nfl-analytics/py/models/train_xgboost_v3_with_opponent_strength.py` (733 lines)

### Input Data
- **Opponent Features:** `/Users/dro/rice/nfl-analytics/data/processed/features/opponent_strength_features.csv` (3,710 records)

---

## Conclusion

**The addition of 3 opponent strength features to XGBoost v3 resulted in a 6.6% performance degradation compared to v2 baseline.**

**Root Cause:** High missing data rate (35%) combined with weak feature importance suggests opponent strength signals are either redundant or too noisy to improve predictions.

**Decision:** **Revert to v2 baseline** for production ensemble. Explore alternative opponent modeling approaches before attempting v4.

**Status:** ✅ Training complete, ⚠️ Performance regression identified, 📋 v2 baseline recommended

---

**Next Steps:**
1. Verify v2 baseline loaded correctly in ensemble
2. Run full 4-way ensemble backtest with v2 (not v3)
3. Research alternative opponent strength modeling approaches
4. Consider dropping 2016-2017 from training to improve opponent data coverage

---

**Training Log:** See full output above
**Comparison Baseline:** v2 MAE = 40.20 yards (Nov 9, 2025)
**v3 Result:** MAE = 42.86 yards (Nov 14, 2025)
**Recommendation:** Use v2, not v3
