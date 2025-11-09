# 4-Way Ensemble Testing Report

**Date:** November 9, 2025
**Tested By:** Claude Code (Sonnet 4.5)
**Ensemble Version:** enhanced_ensemble_v5.py
**Test Duration:** 90 minutes

---

## Executive Summary

**GO/NO-GO DECISION: 🛑 NO-GO FOR PRODUCTION**

The 4-way ensemble (State-Space v2 + Hierarchical v2 + XGBoost v2 + Informative Priors v2) has **multiple critical blockers** that prevent production deployment:

1. ❌ **State-Space v2 produces unrealistic predictions** (639 yards vs actual 266 yards)
2. ❌ **XGBoost v2 has feature mismatch** (trained on 39 features, ensemble query returns 13)
3. ❌ **Hierarchical v2 failed convergence** (Rhat=1.0337 > 1.01 threshold)
4. ❌ **Informative Priors v2 training incomplete** (ESS calculation error)
5. ❌ **Inference latency way above target** (31.9s vs 500ms target, 64x too slow)

**Recommendation:** Retrain all 4 models with consistent features and proper validation before attempting production deployment.

---

## Test Environment

- **Database:** postgresql://localhost:5544/devdb01
- **Python Environment:** .venv (uv managed)
- **R Version:** 4.5.1
- **Key Dependencies:** rpy2 3.6.4, xgboost 2.1.3, brms 2.23.0, Stan 2.37.0

---

## Model Availability Check

| Model | Status | File Size | File Path | Notes |
|-------|--------|-----------|-----------|-------|
| State-Space v2 | ✅ Exists | 1,495.6 MB | models/bayesian/simplified_random_walk_v2.rds | TRAINED Nov 8, APPROVED Nov 8 |
| State-Space Skills | ✅ Exists | 0.2 MB | models/bayesian/simplified_random_walk_skills_v2.csv | 4,321 player-game skill estimates |
| Hierarchical v2 | ✅ Exists | 41.1 MB | models/bayesian/passing_yards_hierarchical_v2.rds | **CONVERGENCE FAILURE** (Rhat=1.0337) |
| Informative Priors v2 | ✅ Exists | 40.1 MB | models/bayesian/passing_informative_priors_v2.rds | **TRAINING INCOMPLETE** (ESS error) |
| XGBoost v2 | ✅ Exists | 0.6 MB | models/xgboost/passing_xgboost_v2_canonical.pkl | Feature mismatch issue |

**Result:** ✅ All 5 model files exist (5/5), but only 1 is production-ready (State-Space v2 APPROVED).

---

## Test Case: Patrick Mahomes, Week 10, 2024

### Ground Truth
- **Player:** Patrick Mahomes (00-0033873)
- **Season:** 2024, Week 10
- **Opponent:** DEN @ KC (home game)
- **Actual Result:** 266 yards on 42 attempts (6.33 YPA)

### Individual Model Predictions

| Model | Prediction (yards) | Std Dev (yards) | Error (yards) | % Error | Status |
|-------|-------------------|-----------------|---------------|---------|--------|
| State-Space v2 | 639.5 | 230.9 | **+373.5** | **+140%** | ❌ **CRITICAL BUG** |
| Hierarchical v2 | 295.3 | 56.9 | +29.3 | +11% | ⚠️ Reasonable but convergence failed |
| Informative Priors v2 | 295.9 | 58.5 | +29.9 | +11% | ⚠️ Reasonable but training incomplete |
| XGBoost v2 | N/A | N/A | N/A | N/A | ❌ **FEATURE MISMATCH** |

**Key Findings:**

1. **State-Space v2:** Predicts 639.5 yards (2.4x actual result). This is a **critical bug** in the prediction logic. The model was APPROVED based on MAE=47.77 on test data, but the ensemble integration is broken.

2. **Hierarchical v2 & Informative Priors v2:** Both predict ~295 yards (~11% error), which is reasonable. However, both have training issues (convergence failure, incomplete training).

3. **XGBoost v2:** Failed to generate prediction due to feature mismatch (trained on 39 features, ensemble query returns 13).

### Ensemble Prediction (3-Way: State-Space + Hierarchical + Informative Priors)

| Metric | Value | Notes |
|--------|-------|-------|
| Ensemble Mean | 433.1 yards | **+167 yards error (+63%)** |
| Ensemble Std | 227.4 yards | Very wide (high uncertainty) |
| 90% CI | [59.1, 807.1] yards | Absurdly wide interval |
| Active Weights | State-Space: 40%, Hierarchical: 40%, Informative: 20% | XGBoost excluded (feature mismatch) |
| Inference Time | 31,865 ms (31.9s) | **64x above target (500ms)** |

**Analysis:** The ensemble prediction is dominated by the broken State-Space v2 model, resulting in a **63% overprediction**. This is unacceptable for production betting.

---

## Convergence Analysis

### Hierarchical v2 (November 8, 2025 training)
```
Max Rhat: 1.0337 (threshold: < 1.01)
Min ESS: 5,316 (threshold: > 400)
Divergent transitions: 107/4000 (3.0%)
Training time: 6.3 minutes (360 seconds)
```

**Diagnosis:**
- Rhat > 1.01 indicates **poor chain convergence**
- Divergent transitions suggest the model struggled with posterior geometry
- ESS is adequate, but Rhat failure is a hard blocker

**Action Required:** Retrain with:
- Increased iterations (4000 → 8000)
- Stronger priors to regularize
- Reparameterization (e.g., non-centered parameterization)

### Informative Priors v2 (November 9, 2025 training)
```
Training: 2015-2024 (4,738 games)
Prior estimation: N(5.62, 0.36) for intercept
Status: INCOMPLETE (ESS calculation error)
```

**Diagnosis:**
- Training log shows successful MCMC sampling started
- Abrupt termination during post-processing (ESS calculation)
- Model file exists (40.1 MB) but may be corrupted

**Action Required:** Retrain from scratch with proper error handling.

---

## Feature Mismatch Analysis

### XGBoost v2 Expected Features (39 features)
```python
['avg_time_to_throw', 'avg_air_yards_differential', 'cpoe', 'spread_close',
 'total_close', 'is_home', 'temp', 'wind', 'is_division_game', 'is_thursday_night',
 'is_short_week', 'away_rest', 'home_rest', 'rest_days', 'opp_def_epa',
 'completion_pct', 'yards_per_attempt', 'td_rate', 'log_attempts', 'is_dome',
 'is_cold', 'is_windy', 'is_bad_weather', 'spread_abs', 'is_favorite',
 'spread_magnitude', 'total_line_high', 'is_short_rest', 'week_in_season',
 'is_early_season', 'is_late_season', 'yards_L3', 'yards_per_attempt_L3',
 'td_rate_L3', 'completion_pct_L3', 'yards_L5', 'yards_per_attempt_L5',
 'yards_L3_vs_avg', 'opp_def_epa_good', 'is_turf']
```

### Ensemble Query Actual Features (13 features)
```python
['log_attempts', 'yards_per_attempt', 'td_rate', 'is_home', 'spread_abs',
 'total_line', 'is_bad_weather', 'yards_L3', 'yards_per_attempt_L3',
 'td_rate_L3', 'week_in_season', 'temp', 'wind']
```

### Missing Features (26 features)
```
avg_time_to_throw, avg_air_yards_differential, cpoe, spread_close, total_close,
is_division_game, is_thursday_night, is_short_week, away_rest, home_rest,
rest_days, opp_def_epa, completion_pct, is_dome, is_cold, is_windy,
is_favorite, spread_magnitude, total_line_high, is_short_rest, is_early_season,
is_late_season, completion_pct_L3, yards_L5, yards_per_attempt_L5,
yards_L3_vs_avg, opp_def_epa_good, is_turf
```

**Root Cause:** The ensemble uses a simplified query (enhanced_ensemble_v5.py:222-265) that does NOT match the rich feature set used during XGBoost training (train_xgboost_canonical.py).

**Action Required:**
1. Update ensemble query to include all 39 features
2. OR retrain XGBoost v2 with only the 13 features available in the ensemble query
3. Document canonical feature set for all models

---

## Variance Pooling Validation

**Test:** Simulate 2-model ensemble with known means and stds

```
Model 1: 250 yards ± 50 yards (weight: 0.6)
Model 2: 270 yards ± 60 yards (weight: 0.4)

Ensemble mean: 258.0 yards ✅
Within-model variance: 2,940 ✅
Between-model variance: 96 ✅
Total variance: 3,036 ✅
Ensemble std: 55.1 yards ✅
```

**Result:** ✅ Variance pooling math is correct (within-model + between-model variance).

---

## Performance Benchmarking

### Single Prediction Latency

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Model Loading (one-time) | 20 | 0.1% |
| State-Space Prediction | 201 | 0.6% |
| Hierarchical v2 Prediction | 25,046 | 78.6% |
| Informative Priors v2 Prediction | 6,618 | 20.8% |
| **TOTAL** | **31,865** | **100%** |

**Findings:**
- **Hierarchical v2 is the bottleneck** (25 seconds per prediction)
- Total inference time is **64x above target** (500ms)
- State-Space v2 is reasonably fast (201ms)
- XGBoost v2 would be fast (<50ms) if feature mismatch were fixed

**Action Required:**
1. Profile brms prediction code to identify bottleneck
2. Consider pre-computing posterior samples for common players
3. Implement caching layer for recent predictions

### Batch Prediction Throughput

**Not tested** - single prediction took 31.9 seconds, making batch testing impractical.

**Projected throughput:** ~2 predictions/minute (unacceptable for real-time betting)

---

## State-Space v2 Bug Investigation

### Expected Behavior
State-Space v2 was APPROVED November 8, 2025 with:
- MAE: 47.77 yards
- RMSE: 62.34 yards
- 90% Calibration: 88.5%
- Statistical superiority over v1 (p=0.0069)

### Actual Behavior in Ensemble
- Prediction: 639.5 yards (2.4x actual result)
- Error: +373.5 yards (140% overprediction)

### Hypothesis: Log-Scale Transform Issue

The state-space model was trained on `log(yards)` but the ensemble prediction logic may have an **exponentiation bug**:

```python
# From enhanced_ensemble_v5.py:359
log_yards_mean = latest_skill + np.log(attempts) + home_effect
log_yards_std = sigma

# Convert to yards scale
yards_mean = np.exp(log_yards_mean + 0.5 * sigma**2)  # Lognormal correction
yards_std = yards_mean * np.sqrt(np.exp(sigma**2) - 1)
```

**Potential Issue:**
- `latest_skill` may already be on the exponentiated scale (not log-scale)
- OR the lognormal correction (`0.5 * sigma**2`) is being applied incorrectly
- OR the `sigma` parameter (0.35) is incorrect for this data

**Action Required:**
1. Verify the scale of `simplified_random_walk_skills_v2.csv` (log-yards or yards?)
2. Re-run state-space model evaluation script to confirm MAE=47.77
3. Compare standalone state-space predictions to ensemble predictions
4. Fix the scaling bug in `_get_state_space_prediction()`

---

## Fallback Ensemble Feasibility

Given that 3/4 models failed, we attempted a **2-way fallback ensemble** (State-Space v2 + XGBoost v2).

### 2-Way Ensemble Status

| Component | Status | Blocker |
|-----------|--------|---------|
| State-Space v2 | ❌ | Critical prediction bug (639 yards vs 266 actual) |
| XGBoost v2 | ❌ | Feature mismatch (39 expected vs 13 provided) |

**Result:** 🛑 **2-way fallback is NOT viable** - both models have blocking issues.

### Alternative: Single-Model Baseline

If ensemble is blocked, we could deploy a **single-model baseline**:

| Model | Viability | Performance | Notes |
|-------|-----------|-------------|-------|
| State-Space v2 | ❌ | MAE: 47.77 (test), 373.5 (ensemble) | Approved but ensemble integration broken |
| Hierarchical v2 | ⚠️ | MAE: ~30 (estimated) | Reasonable predictions but convergence failed |
| Informative Priors v2 | ⚠️ | MAE: ~30 (estimated) | Reasonable predictions but training incomplete |
| XGBoost v2 | ❌ | Unknown | Feature mismatch prevents testing |

**Recommendation:** If production deployment is urgent, use **Hierarchical v2** as a single-model baseline (despite convergence issues), with strict monitoring and conservative bet sizing (max 1% bankroll).

---

## Root Cause Analysis

### Why Did Testing Reveal These Issues So Late?

1. **Insufficient Integration Testing:** Models were trained and evaluated in isolation, not as part of the ensemble pipeline.

2. **Feature Engineering Drift:** XGBoost was trained with a rich feature set (39 features), but the ensemble query was simplified (13 features) without updating the model.

3. **Scale Transform Confusion:** State-space model trained on log-scale, but ensemble prediction logic has exponentiation bug.

4. **Convergence Checks Ignored:** Hierarchical v2 training log shows Rhat=1.0337 failure, but the model was saved anyway (script bug).

5. **No End-to-End Testing:** Each model was tested separately, but the full ensemble pipeline was never validated until now.

### Lessons Learned

1. ✅ **Always test models in the production pipeline**, not just in isolation
2. ✅ **Enforce feature schema contracts** between training and inference
3. ✅ **Hard-fail on convergence issues** (don't save models with Rhat > 1.01)
4. ✅ **Sanity-check predictions** against actual data before deployment
5. ✅ **Profile inference latency early** (31.9s is unacceptable)

---

## Corrective Action Plan

### Phase 1: Critical Fixes (1-2 days)

1. **Fix State-Space v2 prediction bug**
   - Verify scale of skill parameters (log vs linear)
   - Test standalone state-space predictions vs ensemble
   - Update `_get_state_space_prediction()` logic
   - Re-test on 10+ games to validate MAE < 50 yards

2. **Fix XGBoost v2 feature mismatch**
   - Option A: Retrain XGBoost with 13-feature set (faster)
   - Option B: Update ensemble query to include all 39 features (better)
   - Validate predictions on 10+ games

3. **Add convergence checks to training scripts**
   - Refuse to save models with Rhat > 1.01
   - Refuse to save models with ESS < 400
   - Log warnings prominently

### Phase 2: Model Retraining (3-5 days)

4. **Retrain Hierarchical v2**
   - Increase iterations: 2000 → 8000
   - Add non-centered parameterization
   - Verify Rhat < 1.01 on all parameters
   - Validate MAE < 55 yards

5. **Retrain Informative Priors v2**
   - Fix ESS calculation error
   - Complete full training + post-processing
   - Verify Rhat < 1.01 on all parameters
   - Validate MAE < 55 yards

### Phase 3: Performance Optimization (2-3 days)

6. **Optimize brms prediction latency**
   - Profile posterior_predict() bottleneck
   - Reduce ndraws: 1000 → 500 (if acceptable)
   - Implement prediction caching
   - Target: < 500ms per prediction

7. **End-to-end integration testing**
   - Test 50+ games from 2024 Week 10
   - Validate MAE < 50 yards
   - Validate inference < 500ms per prediction
   - Validate 90% calibration

### Phase 4: Production Readiness (1-2 days)

8. **Full backtest (2022-2024)**
   - Run ensemble on 562 test games
   - Compute MAE, RMSE, calibration, Sharpe ratio
   - Validate ROI > +5%

9. **Deployment with monitoring**
   - Real-time prediction logging
   - Alerts for unrealistic predictions (>500 yards, <100 yards)
   - Daily performance reports
   - Kill switch for automatic model disabling

---

## GO/NO-GO Decision Matrix

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Model Convergence | Rhat < 1.01 | Hierarchical: 1.0337, Informative: Unknown | ❌ |
| Prediction Accuracy | MAE < 55 yards | State-Space: 373 yards error | ❌ |
| Feature Consistency | All models use same features | XGBoost mismatch | ❌ |
| Inference Latency | < 500ms | 31,865ms (64x too slow) | ❌ |
| 90% Calibration | > 85% | Not tested (unrealistic predictions) | ❌ |
| Ensemble Diversity | 4 independent models | Only 2 usable (State-Space broken) | ❌ |

**Overall:** **0/6 criteria met** → **NO-GO FOR PRODUCTION**

---

## Recommendations

### Short-Term (1 week)

1. **DO NOT DEPLOY** the current 4-way ensemble to production
2. **Fix critical bugs** (State-Space exponentiation, XGBoost features)
3. **Retrain brms models** with better convergence
4. **Re-run full testing suite** after fixes

### Medium-Term (2-3 weeks)

5. **Complete Phase 1-4** corrective action plan
6. **Run full backtest** (2022-2024, 562 games)
7. **Deploy single-model baseline** if ensemble still blocked
8. **Monitor performance** with conservative bet sizing (1% max)

### Long-Term (1-2 months)

9. **Establish feature schema contracts** for all models
10. **Implement automated integration testing** (CI/CD)
11. **Build prediction caching layer** for performance
12. **Develop ensemble weight optimization** based on live data

---

## Test Artifacts

### Test Script
- **File:** `/Users/dro/rice/nfl-analytics/test_4way_ensemble.py`
- **Lines:** 172
- **Runtime:** 31.9 seconds

### Test Output
```
Patrick Mahomes (00-0033873), Week 10, 2024
- Actual: 266 yards on 42 attempts
- Ensemble: 433.1 ± 227.4 yards (3/4 models)
- Error: +167 yards (+63%)
- Individual:
  - State-Space v2: 639.5 ± 230.9 yards (❌ +373 yards error)
  - Hierarchical v2: 295.3 ± 56.9 yards (⚠️ +29 yards error)
  - Informative Priors v2: 295.9 ± 58.5 yards (⚠️ +30 yards error)
  - XGBoost v2: FAILED (feature mismatch)
```

### Related Documentation
- **Training Logs:** `logs/hierarchical_v2_training_FIXED.log`, `logs/informative_priors_v2_training_4000iter.log`
- **Model Registry:** `models/MODEL_REGISTRY.yaml`
- **State-Space Approval:** `docs/milestones/STATE_SPACE_FINAL_DECISION.md`
- **Canonical Window:** `docs/milestones/TRAINING_WINDOW_ABLATION_RESULTS.md`

---

## Conclusion

The 4-way ensemble is **NOT production-ready**. Critical bugs in State-Space v2 prediction logic, XGBoost feature mismatch, and brms convergence failures make the ensemble unreliable for live betting.

**Immediate action:** Pause production deployment and execute Phase 1 corrective action plan (1-2 days). Re-test after fixes.

**Fallback plan:** Deploy Hierarchical v2 as a single-model baseline (despite convergence issues) with strict monitoring and conservative bet sizing, IF production deployment cannot wait for full fixes.

**Long-term vision:** A robust, well-tested, fast 4-way ensemble is achievable, but requires disciplined feature engineering, proper convergence validation, and end-to-end integration testing.

---

**Report Author:** Claude Code (Sonnet 4.5)
**Test Date:** November 9, 2025
**Next Review:** After Phase 1 fixes (November 11, 2025)
