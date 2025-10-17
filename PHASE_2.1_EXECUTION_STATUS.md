# Phase 2.1: Deeper BNN Architecture - Execution Status

**Status:** MCMC Training In Progress
**Started:** October 16, 2025, 17:53 PST
**Expected Completion:** ~2-4 hours (sequential sampling)

---

## 🎯 Objective

Fix Phase 1 calibration issue: 26% → target 75%+ (90% CI coverage)

## 🏗️ Architecture Changes

### Phase 1 (Baseline)
- **Layers:** 2 (single hidden layer)
- **Hidden size:** 16 units
- **Noise:** Global σ with HalfNormal prior
- **Activation:** ReLU
- **Result:** 26% coverage (severely under-calibrated)

### Phase 2.1 (Current)
- **Layers:** 4 (three hidden layers)
- **Hidden sizes:** 32 → 16 → 8 units
- **Skip connection:** Concatenate h1 + h3 → output (40 units total)
- **Noise:** **Learned per-sample** σ(x) = exp(α₀ + α₁ · Σfeatures)
  - Clipped to prevent numerical issues: σ ∈ [0.14, 7.4]
- **Activation:** Tanh
- **Sampling:** Sequential (cores=1) to avoid multiprocessing issues
- **Target accept:** 0.90

## 📊 Training Configuration

```python
{
    'n_features': 4,  # carries, avg_rushing_l3, season_avg, week
    'hidden_sizes': [32, 16, 8],
    'n_samples': 2000,
    'n_chains': 4,
    'n_tune': 1000,
    'target_accept': 0.90,
    'init': 'adapt_diag'
}
```

**Data:**
- Training: 2,663 samples (2020-2024 weeks 1-6)
- Test: 374 samples (2024 weeks 7+)
- 183 unique players

## 🔧 Technical Fixes Applied

### Issue 1: EOFError in Multiprocessing
- **Cause:** Complex BNN architecture causing multiprocessing crashes
- **Fix:** Changed `cores=4` → `cores=1` (sequential sampling)
- **Trade-off:** 4× slower but stable

### Issue 2: Numerical Instability in Learned Noise
- **Original:** σ(x) = exp(α₀ + α₁ · mean(X))
- **Fixed:**
  - Use sum instead of mean (better scaling with standardized features)
  - Clip log(σ) to [-2, 2] range
  - More conservative priors (α₀ ~ N(0, 0.3), α₁ ~ N(0, 0.1))

### Issue 3: Model Complexity
- **Original:** [64, 32, 16] hidden units (112 total)
- **Fixed:** [32, 16, 8] hidden units (56 total)
- **Rationale:** Smaller model trains faster, less prone to divergences

## 📈 Success Criteria

**Target:** 90% CI coverage ≥ 75%

| Metric | Phase 1 | Target | Phase 2.1 Goal |
|--------|---------|--------|----------------|
| 90% Coverage | 26% | 90% | **≥75%** |
| 68% Coverage | 19% | 68% | ≥60% |
| MAE | 18.7 | N/A | ≤20 |

**Decision Tree:**
- ✅ **If ≥75%:** Proceed to Phase 2.3 (database expansion)
- ⚠️  **If 50-75%:** Try Phase 2.2 (mixture-of-experts)
- ❌ **If <50%:** Implement hybrid calibration (Phase 2.3 fallback)

## 🔍 Monitoring

**Log file:** `/Users/dro/rice/nfl-analytics/logs/bnn_deeper_v2_fixed_20251016_175729.log`

**Check progress:**
```bash
tail -f /Users/dro/rice/nfl-analytics/logs/bnn_deeper_v2_fixed_*.log
```

**Expected output checkpoints:**
1. ✅ Data loaded (3,037 samples, 183 players)
2. ✅ Train/test split (2,663 / 374)
3. ✅ MCMC initialized (adapt_diag)
4. ⏳ Chain 1 tuning (1,000 draws, ~30-40 min)
5. ⏳ Chain 1 sampling (2,000 draws, ~60-80 min)
6. ⏳ Chain 2-4 (repeat, ~4-6 hours total sequential)
7. ⏳ Evaluation & comparison to Phase 1

## 📁 Output Files

**Models:**
- `/Users/dro/rice/nfl-analytics/models/bayesian/bnn_deeper_v2.pkl`

**Results:**
- `/Users/dro/rice/nfl-analytics/experiments/calibration/deeper_bnn_v2_results.json`

**Comparison format:**
```json
{
  "mae": float,
  "rmse": float,
  "coverage_90": float,  // ← KEY METRIC
  "coverage_68": float,
  "target_90": 90.0,
  "target_68": 68.0
}
```

## 🚨 Known Issues

1. **Sequential sampling is slow:** ~4-6 hours vs 1-2 hours parallel
   - Necessary to avoid multiprocessing crashes
   - Trade-off accepted for stability

2. **Postgres warning:** "pandas only supports SQLAlchemy"
   - Non-blocking, code works fine
   - Can be fixed later with SQLAlchemy engine

3. **Model complexity:** Still testing if [32, 16, 8] is sufficient
   - May need to try [48, 24, 12] if results underwhelm
   - Or switch to mixture-of-experts (Phase 2.2)

## 📚 Related Documentation

- Phase Plan: `/Users/dro/rice/nfl-analytics/PHASE_PLAN.md`
- Autonomous Status: `/Users/dro/rice/nfl-analytics/AUTONOMOUS_EXECUTION_STATUS.md`
- Dissertation Cleanup: `/Users/dro/rice/nfl-analytics/analysis/dissertation/DISSERTATION_CLEANUP_SUMMARY.md`

---

**Last Updated:** October 16, 2025, 17:58 PST
**Next Update:** When MCMC completes (~22:00 PST)
