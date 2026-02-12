# BFS Experiment 5 Execution Checklist

## Pre-Execution (5 minutes)

- [ ] Verify database is running: `pg_isready -h localhost -p 5544`
- [ ] Check opponent strength features exist: `ls -lh data/processed/features/opponent_strength_features.csv`
- [ ] Verify Python environment: `python -c "import xgboost; print(xgboost.__version__)"`
- [ ] Check available disk space: `df -h` (need ~500MB)
- [ ] Clear previous logs: `rm -f xgb_v3_2_l1_results.log` (optional)

## Execution (20-30 minutes)

```bash
cd \

# Run with output capture for later review
python scripts/xgb_v3_2_l1_regularization.py 2>&1 | tee xgb_v3_2_l1_results.log
```

## Monitor Progress (Real-time)

```bash
# In a second terminal, watch progress
tail -f xgb_v3_2_l1_results.log | grep -E "(Grid point|CV MAE|Test MAE|Decision)"
```

## Post-Execution Analysis (5 minutes)

### 1. Check Decision Gate

```bash
grep -A 10 "DECISION GATE" xgb_v3_2_l1_results.log
```

**Expected output:**
```
Decision: ✓ DEPLOY | ⚠ INVESTIGATE | ✗ ABANDON
```

### 2. Review Final Metrics

```bash
grep -A 6 "Performance Metrics:" xgb_v3_2_l1_results.log
```

**Expected MAE:**
- < 40.2: ✓ BREAKTHROUGH (beats baseline)
- 40.2-42: ✓ DEPLOY (fixes regression)
- 42-45: ⚠ INVESTIGATE (partial improvement)
- ≥ 45: ✗ ABANDON (approach not working)

### 3. Verify Grid Search Results

```bash
grep -A 20 "GRID SEARCH RESULTS SUMMARY" xgb_v3_2_l1_results.log | head -15
```

**Check:** Top result should have CV MAE < 41

### 4. Check Feature Importance

```bash
grep -A 30 "OPPONENT FEATURE IMPORTANCE PROGRESSION" xgb_v3_2_l1_results.log
```

**Expected:** Opponent features drop to 0% at high alpha values

### 5. Verify Model Saved

```bash
ls -lh models/xgboost/passing_xgboost_v3_2_l1*
```

**Expected:** 2 files (model + metadata)

## Decision Matrix

| MAE Result | Decision | Next Action |
|-----------|----------|------------|
| **< 40.2** | ✓ BREAKTHROUGH | Deploy immediately to ensemble |
| **40.2-42.0** | ✓ DEPLOY | Integrate into ensemble v6.2 (20% weight) |
| **42.0-45.0** | ⚠ INVESTIGATE | Run feature ablation study |
| **≥ 45.0** | ✗ ABANDON | Document, try alternative approach |

## If DEPLOY (MAE < 42):

```bash
# 1. Review metadata
cat models/xgboost/passing_xgboost_v3_2_l1_metadata.json | jq '.metrics'

# 2. Update CLAUDE.md
# Add to "Current Status" section:
# - v3.2 L1 Regularization: MAE=41.23, +28% vs v3.1 regression

# 3. Schedule ensemble integration
# - Next: enhanced_ensemble_v6.py with v3.2 weight=0.20

# 4. Create git commit
git add models/xgboost/passing_xgboost_v3_2_l1*
git commit -m "feat(xgboost): v3.2 L1 regularization (MAE 41.23, +28% vs v3.1)"
```

## If INVESTIGATE (42 < MAE < 45):

```bash
# 1. Run feature ablation study
# (Create script to remove each opponent feature one by one)

# 2. Try broader grid search
# Edit L1_ALPHAS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
# Re-run with finer granularity

# 3. Check baseline stability
# Verify v2 baseline MAE = 40.20 ✓
```

## If ABANDON (MAE ≥ 45):

```bash
# 1. Document findings in postmortem
# 2. Update CLAUDE.md with findings
# 3. Return to v2 baseline (no opponent features)
# 4. Create git commit
git add docs/postmortems/
git commit -m "docs: xgb v3.2 l1 regularization postmortem - approach did not work"
```

## Troubleshooting

### Error: "Database connection failed"
```bash
# Check if database is running
docker-compose ps

# Restart database if needed
docker-compose down && docker-compose up -d pg
```

### Error: "Opponent strength features not found"
```bash
# Regenerate features
python py/features/opponent_strength.py
```

### Error: "Feature coverage audit failed"
```bash
# Check data quality
python << 'PYEOF'
import pandas as pd
df = pd.read_csv('data/processed/features/opponent_strength_features.csv')
print(df.info())
print(df.isnull().sum())
PYEOF
```

### Script takes > 30 minutes
- Normal on slower machines (depends on CPU cores)
- XGBoost compilation can be slow first run
- If > 45 min, something might be wrong

### Grid search doesn't converge (CV MAE >> 50)
- Check feature engineering (might have leakage)
- Verify data is from correct seasons
- Look for NaN/inf values in features

## Files to Review After Completion

1. **Script Output:** `xgb_v3_2_l1_results.log`
2. **Model Artifacts:**
   - `models/xgboost/passing_xgboost_v3_2_l1.pkl`
   - `models/xgboost/passing_xgboost_v3_2_l1_metadata.json`
3. **Documentation:**
   - `docs/BFS_EXPERIMENT_5_L1_REGULARIZATION.md` (comprehensive)
   - `docs/BFS_EXPERIMENT_5_QUICK_START.md` (quick reference)
   - `docs/BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md` (interpretation guide)

## Time Estimates

| Task | Time |
|------|------|
| Pre-execution checks | 2 min |
| Script execution | 18-22 min |
| Post-execution analysis | 5 min |
| Decision & documentation | 5-10 min |
| **Total** | **30-40 min** |

## Key Success Indicators

- Script completes without errors
- Grid search shows convergence (top 3 results similar)
- Feature importance shows opponent features eliminated at high alpha
- Final MAE in reasonable range (38-50 yards)
- Model saved successfully
- Metadata JSON valid

## Support

If anything fails:
1. Check logs: `tail -100 xgb_v3_2_l1_results.log`
2. Review error messages carefully
3. Check database connectivity
4. Verify opponent features CSV exists
5. Try running with smaller dataset if needed

---

**Created:** November 14, 2025
**Purpose:** Quick checklist for BFS Experiment 5 execution
**Script:** `scripts/xgb_v3_2_l1_regularization.py`
