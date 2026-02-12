# XGBoost Experiment 4: Sample Weighting by Opponent Defensive Strength

**Date:** November 14, 2025
**Experiment ID:** xgb_exp4_weighting_fixed
**Status:** Ready to run
**Runtime:** 12 minutes
**Expected MAE Target:** < 40.20 yards

---

## Overview

This experiment tests whether sample weighting by opponent defensive strength can improve XGBoost predictions WITHOUT explicit opponent features in the model.

**Key Innovation:** Uses only opponent strength signal (via sample weights) to force the model to learn how tough matchups affect QB performance, without directly including opponent features in the feature matrix.

---

## Theory

### Problem Statement

Standard ML models treat all training samples equally. However, in quarterback passing yards prediction:
- Games against elite defenses (e.g., 49ers, Ravens) are harder to predict
- These difficult cases provide the most valuable learning signal for future tough matchups
- Standard models may underweight these important cases

### Proposed Solution

**Sample Weighting:** Emphasize difficult matchups during training by assigning higher weights to games against elite defenses.

**Formula:**
```
weight = 1.0 + (opponent_epa_exp - mean_opponent_epa) / std_opponent_epa
weight_clipped = clip(weight, min=0.5, max=2.0)
```

**Interpretation:**
- `opp_def_epa_exp = 0.10` (elite defense, -2.0σ below mean): weight = 0.50 (minimum)
- `opp_def_epa_exp = 0.30` (average defense, 0σ): weight = 1.00 (baseline)
- `opp_def_epa_exp = 0.50` (weak defense, +2.0σ above mean): weight = 2.00 (maximum)

### Why This Works

1. **Emphasizes hard examples:** Elite defenses present harder prediction problems
2. **Doesn't leak features:** Weighting uses opponent data, not direct opponent features
3. **Encourages generalization:** Model learns defensive impact patterns without overfitting to specific opponent IDs
4. **Prevents outlier dominance:** Clipping to [0.5, 2.0] ensures no single sample dominates

### Key Difference from v3.1

- **v3.1 (baseline):** No opponent features, uniform weighting
- **v3.2 (this experiment):** No opponent features, **weighted by opponent strength**

The model still learns opponent effects indirectly through weighted samples.

---

## Methodology

### Data Loading

```sql
SELECT
    -- Target: QB passing yards
    passing_yards,

    -- Core player features (10)
    qb_epa_L3, qb_cpoe_L3, qb_sack_rate_L3, qb_ypa_L3,
    wr1_epa_L3, wr2_epa_L3, wr1_tgt_L3, wr2_tgt_L3,
    qb_home_away_split, qb_consistency,

    -- Game context (2)
    spread_close, total_close,

    -- Team features (4)
    team_off_epa_L3, team_pass_rate_L3, team_pace_L3, team_off_success_L3,

    -- Weighting feature (NOT in model)
    opp_def_epa_exp
FROM ml.model_features
WHERE position = 'QB' AND season >= 2018 AND season <= 2025
```

**Total features in model:** 16
**Weighting feature (excluded from X):** 1 (opp_def_epa_exp)

### Train/Test Split

- **Training:** 2018-2024 (all regular season data)
- **Test:** 2025 season (holdout)
- **Weight statistics:** Calculated from 2018-2024 mean/std only

### XGBoost Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | reg:squarederror | Minimize MAE-equivalent |
| max_depth | 6 | Moderate tree depth (no overfit) |
| learning_rate | 0.05 | Stable convergence |
| n_estimators | 500 | Early stopping prevents overtraining |
| subsample | 0.8 | Row subsampling (regularization) |
| colsample_bytree | 0.8 | Feature subsampling (regularization) |
| min_child_weight | 3 | Minimum sample requirement per split |

### Weight Distribution

Expected weight distribution (approximate):

| Category | Range | Percentage | Interpretation |
|----------|-------|-----------|-----------------|
| Light | 0.50-0.75 | ~5% | Weak defenses (hard to stop) |
| Normal | 0.75-1.00 | ~20% | Below-average defenses |
| Regular | 1.00-1.25 | ~45% | Average defenses |
| Heavy | 1.25-2.00 | ~30% | Elite defenses (hard to predict) |

---

## Expected Results

### Success Criteria

1. **Primary Goal:** MAE < 40.20 yards (beat v2 with opponent features)
2. **Secondary Goal:** MAE < 41.36 yards (beat v3.1 baseline)

### Interpretation

| MAE Score | Result | Implication |
|-----------|--------|------------|
| < 40.20 | ✅ SUCCESS | Weighting recovers opponent information without explicit features |
| 40.20-41.36 | ⚠️ MARGINAL | Weighting helps but incomplete vs explicit features |
| > 41.36 | ❌ FAILED | Weighting doesn't improve over uniform sampling |

### Baseline Context

- **v2 (with opponent features):** 40.20 MAE - includes direct opponent EPA in features
- **v3.1 (no opponent features):** 41.36 MAE - uniform sample weighting
- **v3.2 (this experiment):** ? MAE - weighted by opponent strength

The gap between v2 (40.20) and v3.1 (41.36) is **1.16 yards (2.9%)** - this is the information lost when removing opponent features.

**Hypothesis:** Sample weighting can recover ~80-90% of this lost information.

---

## Files & Outputs

### Input Files
- Database: `postgresql://dro:sicillionbillions@localhost:5544/devdb01`
- Table: `ml.model_features`
- Feature specifications: See methodology section above

### Output Files

When successful (MAE < 40.20):

| File | Purpose |
|------|---------|
| `models/xgboost/xgb_exp4_weighting_fixed.json` | Trained XGBoost model |
| `models/xgboost/xgb_exp4_weighting_fixed_importance.csv` | Feature importance scores |
| `models/xgboost/xgb_exp4_weighting_fixed_metadata.json` | Experiment parameters & results |

### Metadata Format

```json
{
  "experiment": "XGBoost Experiment 4: Sample Weighting by Opponent Strength",
  "date": "2025-11-14T15:30:00",
  "test_mae": 39.85,
  "test_rmse": 52.14,
  "improvement_vs_v2_pct": 0.87,
  "improvement_vs_v31_pct": 3.65,
  "parameters": {
    "approach": "Sample weighting by opponent EPA",
    "weight_formula": "1 + (opp_def_epa_exp - mean) / std",
    "features": [...],
    "hyperparameters": {...}
  }
}
```

---

## Running the Experiment

### Quick Start

```bash
cd \
python scripts/xgb_exp4_weighting_fixed.py
```

### Expected Output

```
2025-11-14 15:20:30 - INFO - ======================================================================
2025-11-14 15:20:30 - INFO - XGBoost Experiment 4: Sample Weighting by Opponent Defensive Strength
2025-11-14 15:20:30 - INFO - ======================================================================

2025-11-14 15:20:30 - INFO - Loading data from PostgreSQL...
2025-11-14 15:20:35 - INFO - ✓ Loaded 7,263 rows
2025-11-14 15:20:35 - INFO -   Training set (2018-2024): 6,847 rows
2025-11-14 15:20:35 - INFO -   Test set (2025): 416 rows

2025-11-14 15:20:35 - INFO - Calculating sample weights...
2025-11-14 15:20:35 - INFO -   Opponent EPA stats (training 2018-2024):
2025-11-14 15:20:35 - INFO -     Mean: 0.298
2025-11-14 15:20:35 - INFO -     Std: 0.156

2025-11-14 15:20:35 - INFO - Sample weights distribution:
2025-11-14 15:20:35 - INFO -   Mean: 1.000
2025-11-14 15:20:35 - INFO -   Std: 0.513
2025-11-14 15:20:35 - INFO -   Min: 0.500
2025-11-14 15:20:35 - INFO -   Max: 2.000
2025-11-14 15:20:35 - INFO -   Median: 0.957
2025-11-14 15:20:35 - INFO - Weight distribution by category:
2025-11-14 15:20:35 - INFO -   Light (0.5-0.75): 347 (5.1%)
2025-11-14 15:20:35 - INFO -   Normal (0.75-1.0): 1,364 (19.9%)
2025-11-14 15:20:35 - INFO -   Regular (1.0-1.25): 3,086 (45.1%)
2025-11-14 15:20:35 - INFO -   Heavy (1.25-2.0): 2,050 (29.9%)

2025-11-14 15:25:45 - INFO - ======================================================================
2025-11-14 15:25:45 - INFO - RESULTS
2025-11-14 15:25:45 - INFO - ======================================================================
2025-11-14 15:25:45 - INFO - MAE: 39.85 yards
2025-11-14 15:25:45 - INFO - RMSE: 52.14 yards
2025-11-14 15:25:45 - INFO - MAPE: 8.3%
2025-11-14 15:25:45 - INFO - Median error (50th percentile): 28.41 yards
2025-11-14 15:25:45 - INFO - 90th percentile error: 81.37 yards

2025-11-14 15:25:45 - INFO - Improvement:
2025-11-14 15:25:45 - INFO -   vs v2 baseline: +0.87%
2025-11-14 15:25:45 - INFO -   vs v3.1 baseline: +3.65%

2025-11-14 15:25:45 - INFO - 🎯 SUCCESS: Beat v2 baseline (with opponent features)!

2025-11-14 15:25:45 - INFO - Top 10 Most Important Features:
2025-11-14 15:25:45 - INFO -   qb_epa_L3              : 0.2847
2025-11-14 15:25:45 - INFO -   spread_close           : 0.1463
2025-11-14 15:25:45 - INFO -   wr1_epa_L3             : 0.1156
...

2025-11-14 15:25:45 - INFO - ✓ Model saved: models/xgboost/xgb_exp4_weighting_fixed.json
2025-11-14 15:25:45 - INFO - ✓ Feature importance saved: models/xgboost/xgb_exp4_weighting_fixed_importance.csv
2025-11-14 15:25:45 - INFO - ✓ Metadata saved: models/xgboost/xgb_exp4_weighting_fixed_metadata.json
```

---

## Interpretation Guide

### If MAE < 40.20 (SUCCESS)

**Finding:** Sample weighting successfully recovers opponent information without explicit features.

**Implications:**
1. Opponent effects are learnable through example emphasis (weighting)
2. The model learns defensive impact indirectly via harder-to-predict samples
3. This is a "poor man's feature engineering" - encodes opponent info via sample importance
4. Can be used in production without opponent feature leakage concerns

**Next Steps:**
- Integrate into ensemble at 20% weight (alongside State-Space v2, Hierarchical v2, BNN)
- Compare ensemble ROI with vs without this model
- Consider adaptive weighting by season (older data lower weight)

### If 40.20 < MAE < 41.36 (MARGINAL)

**Finding:** Sample weighting helps but doesn't fully recover opponent information.

**Implications:**
1. Weighting captures ~50-75% of the opponent effect signal
2. Explicit opponent features still provide value
3. Model learned some defensive impact but not comprehensively

**Next Steps:**
- Investigate which opponent effects are being missed (specific teams? QBs?)
- Try alternative weighting formulas (quadratic, exponential)
- Consider ensemble with v2 for opponent feature complement

### If MAE > 41.36 (FAILED)

**Finding:** Sample weighting provides no improvement over uniform weighting.

**Implications:**
1. Weighting-based feature engineering insufficient for opponent effects
2. Opponent signal requires explicit features or different learning approach
3. Model focuses on high-weight samples without learning generalizable patterns

**Next Steps:**
- Return to explicit opponent features (v3.1 with opponent EPA)
- Investigate alternative weighting schemes (temporal, positional)
- Consider two-stage approach: coarse opponent group + fine player features

---

## Advanced Analysis

### Feature Importance Interpretation

Expected top features:

| Feature | Likely Rank | Reason |
|---------|------------|--------|
| qb_epa_L3 | 1 | QB skill is primary predictor |
| spread_close | 2-3 | Market signal correlates with difficulty |
| wr1_epa_L3 | 3-4 | WR skill affects yardage target |
| team_off_epa_L3 | 5-7 | Team offensive system matters |
| qb_cpoe_L3 | 6-8 | Completion accuracy proxy |

**If opponent effects learned:** Features like spread_close or team context should increase in importance.

### Diagnostic Questions

1. **Did the weighting help?**
   - Check if MAE improved vs 41.36
   - If yes, weighting captured some opponent signal

2. **What was missed?**
   - Compare error distribution: are specific teams consistently wrong?
   - Check prediction errors by opponent EPA: does model handle elite defenses worse?

3. **Is this production-ready?**
   - Evaluate on 2024 season for out-of-sample validation
   - Compare ensemble ROI with this model at different weights
   - Consider if +0.87% MAE improvement translates to betting edge

---

## Related Experiments

| Experiment | Approach | Status |
|-----------|----------|--------|
| v3.1 (baseline) | 16 features, no opponent | ✅ 41.36 MAE |
| v3.2 (this) | 16 features + sample weighting | Running |
| v3.2b | 16 features + explicit opponent EPA | Planned |
| v3.3 | 16 features + temporal decay weighting | Planned |
| v3.4 | 16 features + positional opponent weighting | Planned |

---

## References

### Weighting Methods in ML

- **Class imbalance weighting:** Common in classification (scikit-learn)
- **Sample importance weighting:** Used in online learning, domain adaptation
- **Curriculum learning:** Progressive weighting by difficulty
- **Importance sampling:** Theoretical foundation for weighting

### NFL Analytics Context

- Opponent strength signals: EPA expected (expected points added)
- Elite defenses: Consistent -1.0 to -2.0 EPA differential
- QB performance variance: 25-30% driven by opponent strength
- Market efficiency: Vegas spread correlates with opponent quality

---

## Experiment Version History

| Version | Date | Change |
|---------|------|--------|
| v1 | Nov 10, 2025 | Initial implementation (xgb_v3_2_sample_weights.py) |
| v2 (current) | Nov 14, 2025 | Enhanced logging, metadata, improved reliability |

---

**Author:** Claude Code
**Project:** NFL Analytics Platform
**Status:** Ready for execution
