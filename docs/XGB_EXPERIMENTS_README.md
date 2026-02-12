# XGBoost Experiments Framework

**Date:** November 14, 2025
**Status:** Defensive Clustering Experiment (Exp 2) - COMPLETE & READY

---

## Overview

This directory contains a systematic framework for XGBoost experiments to improve NFL QB passing yards predictions. Each experiment tests a specific hypothesis about feature engineering, clustering, or regularization.

**Current Status:**
- ✅ Exp 2: Defensive Clustering (COMPLETE, ready to run)
- v2 Baseline: 40.20 MAE (13 features)
- v3.1 Reference: ~39-40 MAE (16 features with continuous opponent metrics)

---

## Experiment 2: Defensive Clustering (THIS EXPERIMENT)

### Quick Summary

**Hypothesis:** K-means clustering of opponent defenses into categorical tiers reduces noise while capturing non-linear effects better than continuous opponent metrics.

**Approach:**
1. Cluster defenses (k=4) using EPA + rank
2. Create tier dummies: def_tier_1, def_tier_2, def_tier_3, def_tier_4
3. Train XGBoost with 20 features (tiers replace continuous opponent metrics)
4. Evaluate on 2025 data

**Expected Outcome:** MAE < 40.20 yards (beat v2 baseline)

**File:** `scripts/xgb_exp2_clustering_fixed.py`

### Running the Experiment

```bash
cd \

# Syntax check
python -m py_compile scripts/xgb_exp2_clustering_fixed.py

# Execute (~20 minutes)
python scripts/xgb_exp2_clustering_fixed.py

# Review results
cat models/xgboost/xgb_exp2_clustering_summary.json
```

### Output Files

```
models/xgboost/
├─ xgb_exp2_clustering_summary.json      # Metrics + success status
├─ xgb_exp2_clustering_importance.csv    # Feature importance ranking
└─ xgb_exp2_clustering.json              # Trained model (if MAE < 40.20)
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| MAE | Primary metric: Mean Absolute Error (yards) |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination (0-1) |
| Improvement | % vs v2 baseline (40.20 MAE) |

### Success Criteria

✅ **SUCCESS** if:
- MAE < 40.20 yards
- Tier dummies appear in top 15 features
- Model saves to `xgb_exp2_clustering.json`

❌ **FAILED** if:
- MAE ≥ 40.20 yards
- Tier dummies ranked 15th+
- Model not saved

### Interpreting Results

**Tier Dummies in Top 5 Features:**
→ Categorical clustering captures important non-linear effects
→ Trees benefit from explicit tier boundaries
→ Consider expanding: Add tier interactions, temporal tiers, etc.

**Tier Dummies Ranked 10-15:**
→ Tiers provide modest value
→ Continuous metrics also important
→ Consider hybrid: Keep both tier dummies + continuous features

**Tier Dummies Ranked 15+:**
→ Discretization loses important signal
→ Continuous metrics sufficient
→ Stick with v3.1 approach

---

## Experiment Framework Architecture

### Feature Categories

All experiments use 5 core feature categories:

1. **Core Player Features (10):**
   - QB metrics: EPA (3-game), CPOE, sack rate, YPA
   - WR metrics: EPA, targets (WR1 + WR2)
   - QB personality: home/away split, consistency

2. **Game Context (2):**
   - spread_close, total_close

3. **Opponent Features (varies by experiment):**
   - **v2:** None (baseline)
   - **v3.1:** Continuous (opp_def_epa_exp, opp_pass_def_rank, matchup_difficulty)
   - **Exp 2 (this):** Categorical (def_tier_1, def_tier_2, def_tier_3, def_tier_4)
   - **Exp 3 (future):** Temporal (season-specific opponent metrics)
   - **Exp 4 (future):** Hybrid (tiers + continuous)

4. **Team Features (4):**
   - Offense EPA (3-game), pass rate, pace, success

5. **Target:**
   - passing_yards (actual QB passing yards)

### Data Splits

**Standard:**
- Training: 2018-2024 (7,083 samples)
- Test: 2025 (180 samples)

**Why?**
- 2018-2024: Large historical sample, robust training
- 2025: Current season, out-of-sample validation
- No data leakage (test not used during clustering/training)

### Hyperparameter Consistency

All experiments use identical XGBoost config for fair comparison:

```python
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': 42
}
```

**Why?**
- max_depth=6: Deep enough to capture patterns, shallow enough to avoid overfitting
- learning_rate=0.05: Slow learning prevents overfitting
- subsample=0.8: 80% of samples per iteration (stochastic)
- colsample_bytree=0.8: 80% of features per split (stochastic)

---

## Future Experiments

### Exp 3: Temporal Opponent Clustering
**Hypothesis:** Opponent strength varies season-to-season. Season-specific clusters capture time-varying defense effects.

**Approach:**
- Cluster separately for each season (2018, 2019, ..., 2024)
- Create tier dummies with season interaction
- Expected: Better capture of team-year-specific defensive trends

**Expected Runtime:** ~20 minutes

### Exp 4: L1 Regularization
**Hypothesis:** L1 penalty on opponent features creates sparse solution, selecting only most important defense metrics.

**Approach:**
- Keep continuous opponent features (opp_def_epa_exp, opp_pass_def_rank, matchup_difficulty)
- Add L1 regularization: alpha=0.1, 0.01, 0.001
- Compare: Which opponent features survive regularization?

**Expected Runtime:** ~30 minutes

### Exp 5: Sample Weighting
**Hypothesis:** Recent games are more predictive. Weight recent samples higher.

**Approach:**
- Assign weights: w(t) = exp(-lambda * (current_year - year(t)))
- Test: lambda=0.1, 0.2, 0.5
- Expected: Better generalization to 2025 data

**Expected Runtime:** ~20 minutes

### Exp 6: Hybrid Approach
**Hypothesis:** Combine tiers + continuous for robustness.

**Approach:**
- Include both: def_tier_1-4 AND opp_def_epa_exp, opp_pass_def_rank
- Let XGBoost learn which are useful
- Expected: Best of both worlds

**Expected Runtime:** ~20 minutes

---

## Interpreting Feature Importance

### XGBoost Gain vs Cover

XGBoost reports multiple importance metrics:

**Gain:** Average improvement in loss when feature is used
- Higher = Feature more useful for predictions
- Comparable across models (same metric)

**Cover:** Average number of observations affected by feature
- Higher = Feature used in more splits
- Indicates breadth of impact

**Frequency:** Number of splits using feature
- Higher = Feature consistently useful

**Why it matters:**
- Feature in top 5 by Gain: Critical for predictions
- Feature in top 5 by Frequency: Used in many decision boundaries
- Feature in top 5 by Cover: Affects many observations

---

## Production Integration

### If Exp 2 Succeeds (MAE < 40.20)

1. **Save Model:**
   - Model: `models/xgboost/xgb_exp2_clustering.json`
   - Metadata: `models/xgboost/xgb_exp2_clustering_summary.json`

2. **Register in Model Registry:**
   ```bash
   python scripts/model_registry.py register \
     --model_file models/xgboost/xgb_exp2_clustering.json \
     --metadata models/xgboost/xgb_exp2_clustering_summary.json \
     --name "xgb_exp2_defensive_clustering" \
     --version 1.0
   ```

3. **Integrate into Ensemble:**
   - Weight: 20% in v4 ensemble (allocate from other components)
   - Requires 2 new features: def_tier_1, def_tier_2, def_tier_3, def_tier_4
   - Update: `py/ensemble/enhanced_ensemble_v6.py`

4. **Backtest Full Ensemble:**
   - Test 4-way ensemble on 2022-2024 data
   - Compare to v5 (current baseline)
   - Expected improvement: +0.5-1.0% ROI

### If Exp 2 Fails (MAE ≥ 40.20)

1. **Keep v3.1:**
   - Continue using continuous opponent features
   - No model saved (won't beat baseline)

2. **Document Finding:**
   - Add entry to `docs/POSTMORTEMS.md`
   - Note: Continuous features sufficient, discretization loses signal

3. **Try Alternative:**
   - Exp 3: Temporal clustering
   - Exp 4: L1 regularization
   - Exp 5: Sample weighting

---

## Code Structure

### Main Script
**File:** `scripts/xgb_exp2_clustering_fixed.py` (505 lines)

**Functions:**
1. `load_data()` - Load from PostgreSQL
2. `create_defensive_clusters()` - K-means clustering
3. `assign_clusters()` - Create tier dummies
4. `prepare_features()` - Feature matrix preparation
5. `train_and_evaluate()` - XGBoost training + metrics
6. `main()` - Orchestration

### Data Flow

```
PostgreSQL (ml.model_features)
    ↓
load_data() → DataFrame (7,263 rows)
    ↓
Split: Train (2018-2024) / Test (2025)
    ↓
create_defensive_clusters() → KMeans(k=4)
    ↓
assign_clusters() → def_tier_1, def_tier_2, def_tier_3, def_tier_4
    ↓
prepare_features() → X (20 features), y (target)
    ↓
train_and_evaluate() → XGBRegressor
    ↓
Results: MAE, RMSE, R², Feature Importance
    ↓
Save: model.json + summary.json + importance.csv
```

---

## Troubleshooting

### PostgreSQL Connection Error
```
Error: "FATAL: Ident authentication failed"
Solution: Check DB_URI credentials in script
  DB_URI = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
```

### Missing ml.model_features Table
```
Error: "relation "ml.model_features" does not exist"
Solution: Run data ingestion pipeline
  python scripts/ingest_pbp.py
  python scripts/generate_features.py
```

### OOM / Slow Training
```
Issue: Training takes >30 minutes or crashes
Solution: Reduce n_estimators (500 → 300)
         Or reduce sample size (temporary testing only)
```

### Model Didn't Save
```
Likely: MAE >= 40.20 (didn't beat baseline)
Check: Review xgb_exp2_clustering_summary.json for actual MAE
Decision: Keep v3.1, try different experiment
```

---

## References

### Related Files
- **v2 Baseline:** `models/xgboost/passing_xgboost_v2_canonical.pkl`
- **v3.1 Reference:** `models/xgboost/passing_xgboost_v3_with_opponent_strength.pkl`
- **Data Schema:** `docs/database/DATABASE_QUICK_REFERENCE.md`
- **Feature Engineering:** `docs/FEATURE_ENGINEERING_GUIDE.md`

### Documentation
- **Comprehensive Guide:** `docs/XGB_EXP2_CLUSTERING_GUIDE.md`
- **Quick Start:** `scripts/XGB_EXP2_QUICK_START.txt`
- **Architecture:** `docs/ARCHITECTURE_REFERENCE.md`

### Project Context
- **Project Status:** `PROJECT_STATUS.md`
- **Recent Work:** `docs/milestones/RECENT_WORK_HISTORY.md`
- **CLAUDE.md:** `CLAUDE.md` (this file - AI assistant guide)

---

## Glossary

**MAE:** Mean Absolute Error - Average prediction error in yards (lower is better)
**RMSE:** Root Mean Squared Error - Penalizes large errors more (lower is better)
**R²:** Coefficient of determination - % of variance explained (0-1, higher is better)
**EPA:** Expected Points Added - Advanced metric for individual performance
**CPOE:** Completion Percentage Over Expected
**K-means:** Unsupervised clustering algorithm (partitions into k groups)
**Tier Dummy:** Binary feature (1 if in tier, 0 otherwise)
**Early Stopping:** Stop training if test error stops improving (prevents overfitting)

---

## Contact & Support

**For issues:**
- PostgreSQL: See `docs/database/` directory
- Feature engineering: See code comments in `scripts/xgb_exp2_clustering_fixed.py`
- Results interpretation: See `docs/XGB_EXP2_CLUSTERING_GUIDE.md`

**To extend:**
- Copy `xgb_exp2_clustering_fixed.py` → `xgb_exp3_temporal.py`
- Modify feature engineering section
- Update hypotheses and documentation

---

**Created:** November 14, 2025
**Status:** ✅ COMPLETE - Ready for execution
**Expected Runtime:** 20 minutes
**Target:** Beat v2 baseline (MAE < 40.20 yards)
