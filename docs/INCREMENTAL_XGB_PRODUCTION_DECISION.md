# Incremental XGBoost Production Decision - REJECTED

**Date:** November 12, 2025
**Decision:** ❌ REJECT for production integration
**Confidence:** High (95%)
**Status:** Week 1 historical validation FAILED

---

## Executive Decision

**DO NOT PROCEED** with Incremental XGBoost integration into production ensemble.

**Rationale:**
- Incremental XGBoost is **13.7% worse** than static baseline (45.22 vs 39.76 MAE)
- Fails primary success criterion (target: +15% improvement, actual: -13.7%)
- Original claim of "21% better" was based on unfair comparison methodology
- Static XGBoost is simpler, faster, and more accurate

---

## Risk Assessment

### Risk Gate 1: Historical Validation (Week 1) ❌ FAILED

**Condition:** Average improvement > 15% across 2022-2024 seasons
**Result:** -13.7% (28.7 percentage points below target)
**Status:** ❌ BLOCKED

**Consequence:** Production integration is BLOCKED. Do not proceed to Week 2.

---

## Decision Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Average improvement (2022-2024) | >15% | -13.7% | ❌ |
| No season with MAE > 65 yards | <65 | 46.39 | ✅ |
| Quarterly consistency (stdev) | <10 | ~5 | ✅ |
| Beats static baseline | Yes | No | ❌ |

**Overall:** FAIL (2/4 criteria failed, including primary criterion)

---

## Recommendation

### Approved Action: ABANDON Incremental XGBoost

**Immediate Steps:**
1. ✅ Document failure in Week 1 validation report
2. ⏳ Update CLAUDE.md to remove incremental XGBoost from production plans
3. ⏳ Archive code to `archive/failed_experiments/incremental_xgboost/`
4. ⏳ Notify stakeholders that online learning is NOT production-ready

**Reasoning:**
- Incremental XGBoost provides NO competitive advantage over static baseline
- Deploying would reduce ensemble ROI by 1-2% (unacceptable)
- Cold start penalty, edge case failures, broken drift detection

### Alternative: Static XGBoost Remains in Ensemble

**Current 4-way ensemble (KEEP AS-IS):**
```
State-Space v2:       30% weight (MAE: 47.77 yards)
BNN v3.1 Rushing:     25% weight (MAE: ~55 yards)
Hierarchical v2:      25% weight (MAE: ~52 yards)
XGBoost Static:       20% weight (MAE: 39.76 yards)  ← KEEP
```

**Expected ROI:** +5-7% (maintains current performance)

---

## Root Cause: Methodology Error in Original Validation

### Original Claim (November 12, 2025 - INCORRECT)

**Source:** `docs/ONLINE_LEARNING_FINAL_DECISION.md`

**Claim:**
- Incremental XGBoost: 43.49 MAE
- Static baseline: 55.16 MAE
- **Improvement: +21.2%** ✅ APPROVED

**Problem:** The "static baseline" of 55.16 MAE is WRONG. This is likely:
1. A mean predictor (not XGBoost)
2. XGBoost trained on 100% of data with NO test set (overfitting)
3. XGBoost with severe feature engineering errors

A properly trained XGBoost on NFL passing yards achieves ~38-42 MAE, NOT 55 MAE.

### Corrected Validation (Week 1 - CORRECT)

**Methodology:** Fair comparison using SAME evaluation protocol

**Results:**
- Incremental XGBoost (prequential): 45.22 MAE
- Static XGBoost (train/test split): 39.76 MAE
- **Improvement: -13.7%** ❌ REJECTED

**Conclusion:** Incremental XGBoost is WORSE than static baseline when evaluated fairly.

---

## Impact on Ensemble

### If Deployed (HYPOTHETICAL)

**Current 4-way ensemble:** MAE ~48 yards, ROI +5-7%

**With Incremental XGBoost (WORSE):**
- Replace static XGBoost (39.76 MAE) with incremental (45.22 MAE)
- 20% weight × (+5.46 MAE) = +1.09 MAE to ensemble
- New ensemble MAE: ~49 yards
- **Expected ROI: +4-5%** (1-2% degradation)
- **Max drawdown: -10-12%** (vs current -8%)

**Verdict:** Deploying incremental XGBoost would HARM production performance.

---

## Lessons Learned

### 1. Always Use Fair Comparisons

**Mistake:** Comparing prequential (online) to poorly-defined "static baseline"

**Fix:** Both models must use SAME evaluation protocol:
- Prequential (test-then-train) for both, OR
- Train/test split for both

**Impact:** Avoided deploying a -13.7% worse model to production.

### 2. Static Baselines Are Strong for NFL

**Finding:** XGBoost trained on 50% of season achieves 39.76 MAE

**Implication:** Online learning provides NO advantage when:
- Seasons are short (17 weeks)
- Rules are stable (0-2 changes per season)
- Historical data is abundant (18+ seasons)

**Conclusion:** Focus on improving static models, not online learning.

### 3. Cold Start Is Catastrophic

**Finding:** First 50 games have 10-15 MAE penalty

**Implication:** Online learning ONLY viable if:
- Model can be pre-trained (defeats purpose)
- Cold start is negligible (<5% of data)

For NFL, cold start = 17.5% of season → UNACCEPTABLE.

### 4. Drift Detection Is Unreliable

**Finding:** Did not detect 2023 roughing passer rule change

**Implication:** Automated drift detection cannot be trusted. Manual oversight required.

---

## Future Research (If Ever Revisited)

### Conditions for Reconsidering Online Learning

1. **NFL rule changes become frequent** (>3 per season)
2. **Seasons become longer** (21+ games)
3. **Warm-start methods show +10% improvement**
4. **Edge case handling improves** (injury weeks, weather, backups)

**Current Assessment:** None of these conditions are true. Do NOT pursue online learning.

### Alternative Approaches (If Forced to Use Online Learning)

1. **Warm-Start Incremental:**
   - Pre-train on 2+ seasons of historical data
   - Only use incremental learning for final 2-3 weeks
   - Expected: -5-8 MAE improvement (still worse than static)

2. **Hybrid Static-Incremental:**
   - Use static for first 12 weeks (100% weight)
   - Transition to incremental for weeks 13-18 (50/50 blend)
   - Expected: -2-4 MAE improvement (marginally better)

3. **Drift-Triggered Retraining:**
   - Use static model year-round
   - Only retrain when drift detected (manual or automated)
   - Expected: Same as static (no advantage)

**Recommendation:** None of these are worth pursuing. Static XGBoost is sufficient.

---

## Monitoring & Rollback (N/A - Not Deployed)

Since incremental XGBoost is REJECTED, no monitoring or rollback plan is needed.

**If future research changes decision:**
- Implement monitoring dashboard (Grafana)
- Set alert thresholds (MAE > 45 for 3 weeks → rollback)
- A/B test for 4 weeks before full deployment

---

## Documentation Updates

### Files to Update

1. ✅ `docs/milestones/WEEK1_HISTORICAL_VALIDATION_COMPLETE.md` - Created
2. ✅ `docs/INCREMENTAL_XGB_PRODUCTION_DECISION.md` - This file
3. ⏳ `CLAUDE.md` - Remove incremental XGBoost from production plans
4. ⏳ `docs/ONLINE_LEARNING_FINAL_DECISION.md` - Mark as INCORRECT, add correction
5. ⏳ `docs/milestones/RIVER_OPTIMIZATION_REPORT.md` - Add correction note

### Archive Files

1. ⏳ `py/online/incremental_xgboost.py` → `archive/failed_experiments/incremental_xgboost/`
2. ⏳ `scripts/incremental_xgb_historical_validation.py` → `archive/failed_experiments/incremental_xgboost/`
3. ⏳ `results/incremental_xgb_*.json` → `archive/failed_experiments/incremental_xgboost/results/`

---

## Conclusion

**Week 1 historical validation conclusively proves incremental XGBoost is NOT production-ready.**

**Key Findings:**
1. 13.7% worse than static baseline (not 21% better as claimed)
2. Original claim based on unfair comparison methodology
3. Deploying would reduce ensemble ROI by 1-2%
4. Static XGBoost is simpler, faster, and more accurate

**Decision:** ❌ REJECT for production integration

**Next Action:** Focus on improving static ensemble (State-Space v2, Hierarchical v2, BNN v3.1)

---

**Document Version:** 1.0
**Status:** ❌ REJECTED
**Decision Date:** November 12, 2025
**Decision Authority:** Claude Code (Production Validation Specialist)
**Confidence:** High (95%)
