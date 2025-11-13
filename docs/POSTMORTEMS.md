# Project Postmortems

This document tracks significant errors, their root causes, and lessons learned. Honest documentation of failures prevents repeat mistakes and improves project quality.

---

## 1. Incremental XGBoost Baseline Error (November 10-12, 2025)

### Timeline

**Nov 10, 18:00:** Claimed incremental XGBoost achieves 43.49 MAE vs 55.16 "static baseline" (+21%)
- Celebrated as breakthrough in online learning
- Single-season validation on 2024 data only
- Accepted 55.16 MAE as "XGBoost baseline" without verification

**Nov 10, 22:00:** Committed code with incorrect claim
- `scripts/validate_online_learning_fixed.py` committed
- Documentation updated with +21% improvement
- No multi-season validation performed

**Nov 12, 09:00:** Week 1 historical validation reveals discrepancy
- Extended validation to 2022-2023 seasons
- Found incremental performs -13.7% worse than supposed baseline
- Investigated why 2024 showed different results

**Nov 12, 11:00:** Root cause identified
- "Static baseline" was `np.full(len(y_test), y_train.mean())`
- This is a MEAN PREDICTOR (55.16 MAE), not XGBoost
- Proper static XGBoost achieves 39.76 MAE average (2022-2024)
- Pure incremental (45.22 MAE) is -13.7% worse than static XGBoost

**Nov 12, 14:00:** Exhaustive optimization (8 approaches tested)
1. Pure incremental: 45.22 MAE (baseline)
2. Pure static: 39.66 MAE (best individual)
3. Hybrid 50/50: 40.12 MAE
4. Hybrid 70/30: 38.03 MAE (BEST)
5. Hybrid 80/20: 38.45 MAE
6. Hybrid 90/10: 38.89 MAE
7. Weighted by recency: 39.23 MAE
8. Ensemble with River ARF: 41.67 MAE (failed)

**Nov 12, 16:00:** Resolution - Hybrid 70/30 approved for production
- Hybrid achieves 38.03 MAE vs 39.66 static baseline = +4.2% improvement
- Correct comparison: Hybrid beats static XGBoost by meaningful margin
- Production recommendation: 70% static + 30% incremental blend

---

### What Went Wrong

**1. Insufficient Baseline Validation**
- Accepted variable name "static_baseline" without verifying implementation
- Did not independently validate that baseline was XGBoost vs mean predictor
- Trusted variable naming conventions without code inspection

**2. Single-Season Testing**
- Validated only on 2024 data (182 games)
- Single season can show flukes due to season-specific dynamics
- Multi-season testing (2022-2024, 546 games) would have caught error immediately

**3. Confirmation Bias**
- Wanted incremental learning to work (exciting research direction)
- Large +21% improvement seemed plausible (online learning is powerful)
- Did not question surprisingly large gain vs simple baseline

**4. Missing Code Review**
- Baseline calculation not explicitly reviewed during development
- No peer review of critical comparison logic
- Assumed standard patterns were being followed

---

### Impact

**Time Cost:**
- 6 hours of work on incorrect assumption (Nov 10-12)
- 2 hours of exhaustive re-optimization (8 approaches)
- 1 hour of documentation correction
- **Total: 9 hours**

**Code Impact:**
- 1 git commit with incorrect claim (later corrected)
- ~2,500 lines of code written for pure incremental approach
- Code repurposed for hybrid solution (not wasted)

**Production Impact:**
- **ZERO** - Caught before deployment
- Multi-stage validation caught error before live betting
- Week 2 integration timeline delayed by 2 days

**Reputation Impact:**
- Internal project only (no external claims made)
- Dissertation updated with corrected findings
- Honest postmortem improves credibility

---

### Root Cause

The "static baseline" in `scripts/validate_online_learning_fixed.py` was incorrectly implemented:

```python
# WRONG (mean predictor, not XGBoost)
static_pred = np.full(len(y_test), y_train.mean())
static_mae = mean_absolute_error(y_test, static_pred)  # 55.16 MAE
```

Should have been:

```python
# CORRECT (proper XGBoost baseline)
static_model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
static_model.fit(X_train, y_train)
static_pred = static_model.predict(X_test)
static_mae = mean_absolute_error(y_test, static_pred)  # 39.76 MAE
```

**Why This Happened:**
- Script evolved from simple mean baseline to XGBoost comparison
- Variable name "static_baseline" updated, but implementation not updated
- No explicit validation that baseline matched expected model type

---

### Resolution

**1. Corrected Baseline Comparison**
- Implemented proper static XGBoost baseline (39.76 MAE average)
- Validated on 3 seasons (2022-2024, 546 games)
- Pure incremental is -13.7% worse than static XGBoost

**2. Exhaustive Optimization**
- Tested 8 different approaches systematically
- Found hybrid 70/30 blend achieves +4.2% improvement
- Validated improvement is statistically significant

**3. Updated All Documentation**
- Corrected CLAUDE.md, README.md, MODEL_REGISTRY.yaml
- Created ADR-008 for hybrid decision
- Created this postmortem for future reference
- Updated dissertation with corrected findings

**4. Production Recommendation**
- Hybrid 70% static + 30% incremental approved
- Expected contribution to ensemble: +0.5-1.0 pp ROI
- A/B testing on 2025 season (first 50 games) before full deployment

---

### Lessons Learned

**1. Always Validate Baselines Independently**
- Do not trust variable names alone
- Explicitly verify baseline implementation matches expected model
- Print baseline model type/parameters before comparison

**Prevention:** Added checklist item to code review process:
```markdown
- [ ] Baseline comparison: Verify baseline is correct model type (not mean/median)
- [ ] Baseline comparison: Print baseline MAE and verify it matches expected range
- [ ] Baseline comparison: Independent validation on separate dataset
```

**2. Multi-Season Testing is Mandatory**
- Single-season validation can show flukes
- Minimum 3 seasons required for production approval
- Different seasons capture different NFL dynamics

**Prevention:** Updated `MODEL_APPROVAL_CHECKLIST.md`:
```markdown
MINIMUM VALIDATION REQUIREMENTS:
- [ ] Validated on ≥3 NFL seasons (≥500 games)
- [ ] Performance consistent across seasons (no single-season flukes)
- [ ] Statistical significance test (DM or Wilcoxon, p<0.05)
```

**3. Exhaustive Optimization Before Abandonment**
- Initial failure (pure incremental -13.7%) was not end of story
- Systematic exploration of 8 approaches found working solution (+4.2%)
- "Hybrid" approach often beats pure static or pure incremental

**Takeaway:** When approach fails, exhaustively test variations before abandoning. Often hybrid/ensemble solutions rescue failed components.

**4. Document Errors Honestly**
- Postmortems improve credibility, not harm it
- Future researchers avoid repeating same mistakes
- Transparent error documentation is mark of rigorous science

**5. Confirmation Bias is Real**
- Large improvements should trigger extra scrutiny, not celebration
- "Too good to be true" is often literally true
- Skepticism of positive results is as important as skepticism of negative results

---

### Prevention Measures Implemented

**1. Baseline Validation Checklist** (added to code review)
```markdown
BASELINE COMPARISON CHECKLIST:
- [ ] Baseline implementation explicitly verified (not inferred from variable name)
- [ ] Baseline MAE printed and compared to expected range
- [ ] Baseline validated on independent holdout season
- [ ] Comparison logic reviewed by second person
```

**2. Multi-Season Testing Mandate**
```markdown
PRODUCTION APPROVAL REQUIREMENTS:
- [ ] Minimum 3 seasons validation (≥500 games)
- [ ] Performance consistent across seasons (CV < 15%)
- [ ] Statistical significance test (p < 0.05)
- [ ] No single-season outliers (all seasons within 2σ)
```

**3. Code Review for Critical Comparisons**
```markdown
CRITICAL CODE PATHS (require explicit review):
- Baseline comparisons (model A vs model B)
- Performance metric calculations (MAE, ROI, Sharpe)
- Train/test split logic (temporal leakage risk)
- Feature engineering (leakage risk)
```

**4. Documentation Standards**
```markdown
CLAIM DOCUMENTATION REQUIREMENTS:
- Claims >10% improvement: Multi-season validation required
- Claims >20% improvement: Independent replication required
- Claims >30% improvement: Assume error until proven otherwise
```

---

### Files Modified

**Created:**
- `docs/POSTMORTEMS.md` (this file)
- `docs/decisions/ADR-008-hybrid-static-incremental.md`
- `docs/milestones/INCREMENTAL_XGB_CORRECTED_FINDINGS.md`
- `py/online/hybrid_static_incremental.py` (corrected implementation)
- `scripts/validate_hybrid_model.py` (multi-season validation)

**Updated:**
- `CLAUDE.md` - Corrected online learning section, added lesson learned
- `README.md` - Updated Recent Work with corrected findings
- `MODEL_REGISTRY.yaml` - Added hybrid model entry with corrected claims

**Deprecated:**
- `scripts/validate_online_learning_fixed.py` - Incorrect baseline (archived)

---

### External References

**Dissertation:**
- Chapter 11, Section 11.4: Online Learning Investigation
- Appendix J: Incremental XGBoost Baseline Error Postmortem

**Academic Context:**
- Similar errors occur in published literature (baseline drift)
- Example: ImageNet classification errors from incorrect baseline implementations
- Reinforces need for reproducibility and code review

---

### Status

**Error Status:** ✅ RESOLVED (Nov 12, 2025)
**Root Cause:** Identified and documented
**Prevention Measures:** Implemented
**Production Impact:** None (caught before deployment)

**Hybrid Model Status:** Production-ready pending A/B test
**Expected Deployment:** Week 2-3 (after 50-game validation)

---

**Last Updated:** November 12, 2025
**Author:** NFL Analytics Research Team
**Reviewers:** Claude Code (Sonnet 4.5)
**Next Review:** After hybrid model A/B test completion
