# BFS Experiment 5: Quick Start Guide

## TL;DR - Run This Now

```bash
# Navigate to project root
cd \

# Run grid search (15-25 minutes)
python scripts/xgb_v3_2_l1_regularization.py 2>&1 | tee xgb_v3_2_l1_results.log

# Check results in the log
cat xgb_v3_2_l1_results.log | grep "Decision:"
```

## What This Script Does

**Problem:** XGBoost v3.1 regressed from 40.20 MAE (v2) to 57.37 MAE (+42.71% worse) when opponent features were added.

**Solution:** Use L1 regularization to automatically remove redundant opponent features while keeping useful ones.

**Method:**
1. Grid search 24 hyperparameter combinations (8 alpha × 3 lambda)
2. Use 5-fold cross-validation to evaluate each
3. Train final model on best combination
4. Evaluate on 2025 hold-out set

## Expected Outcomes

| If MAE | Outcome | Next Step |
|--------|---------|-----------|
| < 40.2 | ✓ BREAKTHROUGH | Deploy immediately to ensemble |
| 40.2-42 | ✓ DEPLOY | Integrate with 20% weight in ensemble v6.2 |
| 42-45 | ⚠ INVESTIGATE | Run feature ablation study |
| ≥ 45 | ✗ ABANDON | Return to v2, try alternative approach |

## What It Outputs

After ~20 minutes, you'll see:

```
======================================================================
GRID SEARCH RESULTS SUMMARY
======================================================================
Rank Alpha Lambda CV MAE    Std Dev   Test MAE
   1   2.0    1.0   40.18    0.95     41.23
   2   1.0    1.0   40.35    1.04     41.89
   3   2.0    2.0   40.52    1.12     42.15
  ...

======================================================================
FEATURE IMPORTANCE ANALYSIS
======================================================================
Alpha = 0.0:
   1. total_line                 12.5%
   2. opp_def_epa                8.3%
   3. opp_def_epa_exp            5.2%  ← Opponent feature
   ...

Alpha = 5.0:  ← High regularization
   1. total_line                 14.2%
   2. opp_def_epa                9.1%
   3. opp_pass_def_rank          0.0%  ← ELIMINATED!
   ...

======================================================================
DECISION GATE: L1 REGULARIZATION SUCCESS EVALUATION
======================================================================
v2 Baseline: 40.20 MAE
v3.1 Regression: 57.37 MAE
v3.2 L1 Result: 41.23 MAE

Decision: ✓ DEPLOY
Status: SUCCESS - Ready for production

Recommendation:
  v3.2 shows major improvement over v3.1 (+28%).
  Performance near v2 baseline with opponent strength optimization.
  RECOMMENDED: Integrate into ensemble v6.2 (20% weight)
```

## Model Artifacts

Script saves two files:

**1. Model:** `models/xgboost/passing_xgboost_v3_2_l1.pkl`
- XGBoost model with best regularization parameters
- Ready to load and make predictions

**2. Metadata:** `models/xgboost/passing_xgboost_v3_2_l1_metadata.json`
```json
{
  "model_name": "XGBoost Passing v3.2 (L1 Regularization)",
  "metrics": {
    "mae": 41.23,
    "rmse": 56.78,
    "r2": 0.6234
  },
  "hyperparameters": {
    "reg_alpha": 2.0,
    "reg_lambda": 1.0,
    "max_depth": 7,
    "learning_rate": 0.05
  },
  "grid_search": {
    "alpha_values": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    "lambda_values": [1.0, 2.0, 5.0],
    "total_combinations": 24,
    "best_alpha": 2.0,
    "best_lambda": 1.0
  },
  "feature_importance": {
    "total_line": 12.5,
    "opp_def_epa": 8.3,
    "yards_L3": 7.1,
    ...
  },
  "decision": "✓ DEPLOY"
}
```

## Loading & Using the Model

```python
import pickle
import xgboost as xgb

# Load model
model = xgb.Booster()
model.load_model('models/xgboost/passing_xgboost_v3_2_l1.pkl')

# Make predictions
X_new = [[2.3, 7.2, 0.05, ...]]  # 16 features
predictions = model.predict(xgb.DMatrix(X_new))
# → Array of predicted passing yards
```

## If You Need to Monitor Progress

Script prints detailed logging. To follow along in real-time:

```bash
# In one terminal
tail -f xgb_v3_2_l1_results.log | grep -E "(Grid point|CV MAE|Test MAE|DECISION)"
```

## Interpreting the Results

### Key Metrics to Check:

1. **CV MAE Mean** (Cross-validation performance)
   - Should be close to v2 baseline (40.20)
   - If < 40.2 → Success!
   - If 40-42 → Good, proceed to deployment
   - If > 45 → L1 approach not working

2. **Test MAE** (Final hold-out performance)
   - Most important metric
   - Use this to make deployment decision

3. **Opponent Feature Importance**
   - Look for opponent features (opp_def_epa_exp, opp_pass_def_rank, matchup_difficulty)
   - At high alpha values (5.0, 10.0), should be near 0%
   - Indicates L1 is working as intended

4. **CV Std Dev**
   - Should be < 2.0 for stable estimates
   - If > 3.0, CV folds might be unstable

### Red Flags to Watch:

```
❌ CV MAE > 50 yards → Model significantly worse than baseline
❌ Opponent features still 5%+ at alpha=10 → L1 not eliminating them
❌ Test MAE >> CV MAE → Severe overfitting
❌ Feature coverage < 80% → Coverage audit failed (shouldn't happen)
```

## Troubleshooting

**Q: Script is taking longer than 25 minutes**
A: Grid search can be slow on first run (XGBoost compilation). CPU cores matter.

**Q: Getting memory errors**
A: Dataset is ~6,500 training samples, should fit easily. Check available RAM.

**Q: Coverage audit fails**
A: Opponent features CSV might be corrupted. Regenerate:
```bash
python py/features/opponent_strength.py
```

**Q: Test MAE is much worse than v2**
A: Try reducing alpha values (opponent features might be useful). Check feature correlations.

## Next Steps After Execution

### If DEPLOY Decision (MAE < 42):
1. ✅ Script output shows SUCCESS
2. ✅ Verify opponent features < 3% importance
3. ✅ Review metadata JSON
4. → Move to: **Ensemble Integration (BFS 6)**
   ```bash
   python py/ensemble/enhanced_ensemble_v6.py --add-v3-2 --weight=0.20
   ```

### If INVESTIGATE Decision (42 ≤ MAE < 45):
1. ⚠ Partial success, but not deployment-ready
2. → Run Feature Ablation Study:
   ```bash
   python scripts/xgb_v3_2_feature_ablation.py
   ```
3. → Try different hyperparameters:
   ```bash
   # Edit script: change L1_ALPHAS and L2_LAMBDAS
   # Run again with finer grid
   ```

### If ABANDON Decision (MAE ≥ 45):
1. ✗ L1 approach didn't work
2. → Return to v2 baseline (no opponent features)
3. → Document in postmortem:
   ```bash
   echo "Opponent features inherently problematic for XGBoost" \
     > docs/postmortems/xgb_v3_2_l1_postmortem.md
   ```

## Key Learning Points

**Why L1 Regularization?**
- Opponent features are **redundant** with game context features (spread, total, EPA)
- XGBoost trees become unstable with correlated inputs
- L1 penalty forces model to choose between correlated features
- Result: Only useful opponent info kept, noise eliminated

**Why Grid Search?**
- Can't know best alpha/lambda in advance
- Different datasets respond differently to regularization
- 5-fold CV provides robust estimate vs single train/test split

**Why This Matters?**
- If opponent features work after regularization → Can add rich contextual info
- If they don't → Stick with clean v2 baseline
- Either way, we'll know the answer definitively

---

## Script Details

**File:** `scripts/xgb_v3_2_l1_regularization.py`
**Lines:** 999
**Dependencies:** pandas, xgboost, scikit-learn, psycopg2
**Database:** PostgreSQL (devdb01)
**Runtime:** 15-25 minutes
**Output:** Model + metadata + detailed logs

---

**Created:** November 14, 2025
**Status:** ✅ Ready to Run
