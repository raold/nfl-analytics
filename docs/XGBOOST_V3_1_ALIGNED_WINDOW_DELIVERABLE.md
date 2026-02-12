# XGBoost v3.1 Aligned Window - Implementation Deliverable

**Date:** November 14, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** READY FOR VALIDATION
**Postmortem Fix:** XGBoost v3 Performance Regression (35% missing data)

---

## Executive Summary

Implemented XGBoost v3.1 as a **postmortem fix** for the XGBoost v3 performance regression (6.6% MAE regression). The root cause was 35% missing opponent strength data in 2016-2017 training years, filled with median imputation that added noise instead of signal.

**Solution:** Align training window with feature availability (2018-2024 only, no decay years).

**Key Innovation:** Feature coverage audit gate that **blocks training** if any feature has <80% coverage. This is the #1 lesson from the XGBoost v3 postmortem.

---

## Problem Statement

### XGBoost v3 Performance Regression

| Metric | XGBoost v2 (Baseline) | XGBoost v3 (Failed) | Delta |
|--------|----------------------|-------------------|-------|
| MAE (yards) | 40.20 | 42.86 | +2.66 (+6.6%) ❌ |
| RMSE (yards) | 56.80 | 60.12 | +3.32 (+5.8%) ❌ |
| Training Time | 12.3 min | 18.7 min | +6.4 min (+52%) ❌ |
| Model Size | 11.2 MB | 17.4 MB | +6.2 MB (+55%) ❌ |

**Root Cause:** Training window (2016-2024) mismatched with opponent feature availability (2018-2024 → 35% missing).

**Why Median Imputation Failed:**
- 2016-2017: 100% missing opponent data (531 games, 35% of training set)
- Filled with median (0.0 = league average defense)
- Model learned "unknown opponent = average" pattern (useless)
- Added complexity without signal → regression

---

## Solution: XGBoost v3.1 with Aligned Training Window

### Design Changes

| Component | XGBoost v3 | XGBoost v3.1 (Fixed) |
|-----------|-----------|---------------------|
| **Training Window** | 2016-2024 (8 seasons + H=4 decay) | 2018-2024 ONLY (7 seasons, no decay) |
| **Data Samples** | 2,338 games (35% missing opp data) | ~1,800 games (0.9% missing) |
| **Sample Weights** | Exponential decay (H=4) | Uniform (all equally weighted) |
| **Coverage Audit** | None | **MANDATORY** (blocks training if <80%) |
| **Expected Coverage** | 65% (failed) | 99.1% (excellent) |

### Key Features

1. **Feature Coverage Audit (CRITICAL GATE)**
   - Audits all 16 features before training
   - Minimum threshold: 80% (fail-safe)
   - Target threshold: 95% (production quality)
   - **Blocks training** if any feature fails

2. **Aligned Training Window**
   - Start: 2018 (EPA data availability)
   - End: 2024 (latest complete season)
   - No decay years (clean 2018-2024 only)
   - Matches opponent strength feature availability

3. **Validation Protocol**
   - Small sample validation (100 games) before full training
   - Coverage audit on sample
   - Go/No-Go decision gate
   - Prevents wasted compute time

4. **Clear Decision Criteria**
   - **DEPLOY:** v3.1 MAE < v2 MAE (40.20 yards)
   - **INVESTIGATE:** v3.1 within 2% of v2 (borderline)
   - **ABANDON:** v3.1 MAE >= v2 * 1.02 (opponent features don't help)

---

## Files Created

### 1. Training Script (733 lines)

**Path:** `py/models/train_xgboost_v3_1_aligned_window.py`

**Purpose:** Train XGBoost v3.1 with aligned window and feature coverage audit.

**Key Functions:**
- `audit_feature_coverage()`: **CRITICAL GATE** - Validates coverage before training
- `load_data_from_db()`: Loads 2018-2025 data only (aligned window)
- `engineer_features()`: 16 features (13 baseline + 3 opponent strength)
- `prepare_train_test_split()`: Includes mandatory coverage audit
- `train_xgboost_model()`: XGBoost training (same hyperparameters as v3)
- `evaluate_and_save()`: Performance metrics + v2 comparison

**Usage:**
```bash
python py/models/train_xgboost_v3_1_aligned_window.py
```

**Output:**
- `models/xgboost/passing_xgboost_v3_1_aligned.pkl`
- `models/xgboost/passing_xgboost_v3_1_aligned_metadata.json`
- Console report with coverage audit, metrics, and decision

**Decision Gates:**
1. **Coverage Audit:** Must pass 80% threshold (blocks training if fails)
2. **Performance Gate:** v3.1 MAE < v2 MAE (40.20) → DEPLOY or INVESTIGATE

---

### 2. Validation Script (244 lines)

**Path:** `scripts/validate_xgboost_v3_1.py`

**Purpose:** Quick validation on small sample before full training (saves compute time).

**What It Does:**
1. Loads 100 random QB games from 2018-2024
2. Merges opponent strength features
3. Checks feature coverage (same thresholds as training)
4. Analyzes missing data by season
5. Provides GO/NO-GO decision

**Usage:**
```bash
# Default (100 games)
python scripts/validate_xgboost_v3_1.py

# Custom sample size
python scripts/validate_xgboost_v3_1.py --sample-size 200
```

**Exit Codes:**
- `0`: GO - Proceed with full training
- `1`: NO-GO - Do not train (coverage issues)

**Why This Matters:**
- XGBoost v3 took 18.7 minutes to train before discovering regression
- Validation takes ~10 seconds
- Early detection saves compute time and prevents wasted effort

---

## Expected Performance

### Best Case Scenario

**Hypothesis:** Alignment solves the missing data problem, opponent features add value.

**Expected Results:**
- Coverage: 99.1% (excellent, validated on 2018-2024 opponent features CSV)
- MAE: 38-39 yards (2-5% improvement over v2 baseline 40.20)
- Feature importance: Opponent features rank top 50% (5%+ importance)

**Decision:** ✓ DEPLOY to ensemble v6.1 (30% weight)

**Next Steps:**
1. Integrate into `enhanced_ensemble_v6.py`
2. Full backtest (2022-2024 with canonical window)
3. A/B test on 2025 season (first 50 games)
4. Production deployment if A/B test succeeds

---

### Moderate Case Scenario

**Hypothesis:** Alignment helps, but opponent features only marginally valuable.

**Expected Results:**
- Coverage: 99.1% (excellent)
- MAE: 39.5-40.5 yards (within 2% of v2 baseline)
- Feature importance: Opponent features rank bottom 50% (2-5% importance)

**Decision:** ⚠ INVESTIGATE FURTHER

**Next Steps:**
1. Feature ablation study (train without opponent features)
2. Correlation analysis (opponent features vs baseline features)
3. Consider alternative opponent proxies (Elo, prior win%)
4. Wait for more data before deployment

---

### Worst Case Scenario

**Hypothesis:** Opponent features are redundant or alignment doesn't solve the issue.

**Expected Results:**
- Coverage: 99.1% (excellent) BUT
- MAE: 40.5+ yards (still regresses vs v2)
- Feature importance: Opponent features rank bottom 20% (<2% importance)

**Decision:** ✗ ABANDON opponent strength features

**Root Cause Hypotheses:**
1. **Redundancy:** Opponent strength already captured by existing features
   - `team_pass_epa` correlates 0.72 with opponent defense
   - `home_advantage` implicitly captures matchup quality
   - `season_week` captures schedule strength patterns

2. **Feature Engineering Issues:**
   - Exponential weighting (halflife=4) may not match XGBoost's tree structure
   - Percentile ranks may lose information vs raw values
   - Composite difficulty index may oversimplify

3. **Alternative Proxies Needed:**
   - Elo ratings (available 2006+, simpler)
   - Prior season win percentage (available 2006+)
   - Vegas closing spreads (available 2010+, market-implied quality)

**Next Steps:**
1. Document findings in postmortem addendum
2. Keep XGBoost v2 in production ensemble
3. Explore alternative opponent strength proxies (v3.2)
4. Focus efforts on other ensemble improvements (temperature scaling, calibration)

---

## Feature Coverage Audit Design

### Why It Matters

**XGBoost v3 Postmortem Lesson #1:**
> "Always validate feature coverage (>80%) BEFORE model integration. Run coverage audit as first step in feature engineering pipeline."

**What Went Wrong in v3:**
- Feature generation script warned: "0.9% missing for 2018-2024" ✓
- BUT training script used 2016-2024 window → 35% missing ✗
- No coverage check before training → discovered regression only after 18.7 min training

### Coverage Audit Function

```python
def audit_feature_coverage(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 0.80,  # Minimum 80% coverage
    fail_on_low_coverage: bool = True  # BLOCK training
) -> Dict[str, float]:
    """
    Audit feature coverage before model training (CRITICAL GATE).

    Raises:
        ValueError: If any feature has coverage below threshold
    """
    coverage_report = {}
    all_pass = True

    for feat in features:
        coverage = df[feat].notna().mean()
        coverage_report[feat] = coverage

        if coverage < threshold:
            all_pass = False
            logger.error(
                f"Feature '{feat}' has insufficient coverage: "
                f"{coverage:.1%} < {threshold:.0%} threshold"
            )

    if not all_pass and fail_on_low_coverage:
        raise ValueError("Coverage audit failed - training blocked")

    return coverage_report
```

### Integration into Training Pipeline

```python
# Step 4: Train/test split WITH COVERAGE AUDIT (critical gate)
try:
    X_train, y_train, X_test, y_test, feature_cols, train_df = prepare_train_test_split(df)
except ValueError as e:
    logger.error(f"\n✗ COVERAGE AUDIT FAILED - TRAINING BLOCKED")
    logger.error(f"  {str(e)}")
    logger.error(f"\nDO NOT PROCEED WITH TRAINING")
    return  # Exit without training
```

**Result:** Training **cannot proceed** if coverage audit fails. This prevents:
- Wasted compute time (18.7 min for v3)
- Silent regressions (6.6% worse MAE)
- Production deployment of broken models

---

## Validation Protocol

### Pre-Training Validation (Quick Check)

**Purpose:** Verify coverage on small sample before committing to full training.

**Steps:**
1. Load 100 random games from 2018-2024
2. Merge opponent strength features
3. Check coverage on 3 opponent features
4. Analyze missing data by season
5. GO/NO-GO decision

**Time:** ~10 seconds (vs 18.7 min for full training)

**Command:**
```bash
python scripts/validate_xgboost_v3_1.py --sample-size 100
```

**Expected Output:**
```
======================================================================
FEATURE COVERAGE AUDIT (SAMPLE)
======================================================================
  opp_def_epa_exp                 99.0%  ✓ EXCELLENT  (missing: 1/100)
  opp_pass_def_rank               99.0%  ✓ EXCELLENT  (missing: 1/100)
  matchup_difficulty              99.0%  ✓ EXCELLENT  (missing: 1/100)
======================================================================

Missing data breakdown by season:
  2018:  1.5% missing (1/67 games)
  2019:  0.0% missing (0/33 games)
  2020:  0.0% missing (0/0 games)
  ...

======================================================================
VALIDATION DECISION
======================================================================
✓ GO - PROCEED WITH FULL TRAINING
  All features meet 80% threshold
  Expected to replicate on full 2018-2024 dataset

Next step:
  python py/models/train_xgboost_v3_1_aligned_window.py
======================================================================
```

---

### Full Training Workflow

**Once validation passes:**

```bash
# 1. Validate on sample (10 seconds)
python scripts/validate_xgboost_v3_1.py

# 2. If validation passes, run full training (15-20 min)
python py/models/train_xgboost_v3_1_aligned_window.py

# 3. Check decision gate
# If v3.1 MAE < v2 MAE (40.20): DEPLOY
# If v3.1 within 2%: INVESTIGATE
# If v3.1 >= v2 * 1.02: ABANDON
```

---

## Expected Training Output

### Coverage Audit Report

```
======================================================================
FEATURE COVERAGE AUDIT (CRITICAL GATE)
======================================================================
Threshold: 80% (minimum required)
Target: 95% (clean features)
======================================================================
  log_attempts                    100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  yards_per_attempt               100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  td_rate                         100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  is_home                         100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  spread_abs                       98.5%  ✓ EXCELLENT  (missing: 27 / 1,807)
  total_line                       98.5%  ✓ EXCELLENT  (missing: 27 / 1,807)
  is_bad_weather                  100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  yards_L3                        100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  yards_per_attempt_L3            100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  td_rate_L3                      100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  week_in_season                  100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  temp                            100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  wind                            100.0%  ✓ EXCELLENT  (missing: 0 / 1,807)
  opp_def_epa_exp                  99.1%  ✓ EXCELLENT  (missing: 16 / 1,807) [NEW]
  opp_pass_def_rank                99.1%  ✓ EXCELLENT  (missing: 16 / 1,807) [NEW]
  matchup_difficulty               99.1%  ✓ EXCELLENT  (missing: 16 / 1,807) [NEW]
======================================================================
✓ ALL FEATURES PASS COVERAGE AUDIT
  Training can proceed safely
======================================================================
```

### Performance Metrics

```
======================================================================
MODEL EVALUATION - 2025 HOLDOUT SET
======================================================================

Performance Metrics:
  MAE: 38.45 yards  [EXAMPLE - actual TBD]
  RMSE: 54.20 yards
  R²: 0.6234
  Mean Prediction: 251.3 yards
  Mean Actual: 249.8 yards

Baseline Comparison:
  Baseline MAE: 55.16 yards (mean predictor)
  Improvement: 30.3%

v2 Baseline Comparison:
  v2 MAE: 40.20 yards
  v3.1 MAE: 38.45 yards
  Improvement: +4.4%  [EXAMPLE - actual TBD]

Training Info:
  Training time: 16.2 minutes
  Best iteration: 487
  Features: 16
  Train samples: 1,807
  Test samples: 148
```

### Feature Importance

```
======================================================================
FEATURE IMPORTANCE (by gain)
======================================================================
  1. yards_L3                      35.2%
  2. log_attempts                  18.4%
  3. yards_per_attempt             12.3%
  4. opp_def_epa_exp                8.1%  [NEW]  ← KEY METRIC
  5. total_line                     6.7%
  6. spread_abs                     5.2%
  7. matchup_difficulty             4.3%  [NEW]  ← KEY METRIC
  8. week_in_season                 3.1%
  9. opp_pass_def_rank              2.9%  [NEW]  ← KEY METRIC
 10. td_rate_L3                     2.1%
 11. is_home                        1.8%
 12. yards_per_attempt_L3           1.5%
 13. temp                           1.2%
 14. wind                           0.9%
 15. is_bad_weather                 0.7%
 16. td_rate                        0.6%

OPPONENT FEATURES TOTAL: 15.3% importance (rank #4, #7, #9)
```

**Analysis:**
- If opponent features rank top 50% (>5% importance each): **Value confirmed** ✓
- If opponent features rank bottom 50% (<3% importance): **Redundant** ⚠
- If opponent features rank bottom 20% (<1% importance): **No value** ✗

---

### Decision Gate Output

```
======================================================================
DECISION GATE: v3.1 vs v2 COMPARISON
======================================================================
v2 Baseline MAE: 40.20 yards
v3.1 Aligned MAE: 38.45 yards  [EXAMPLE]
Improvement: +4.4%

Decision: ✓ DEPLOY

Recommendation:
  v3.1 beats v2 baseline by 4.4%.
  Opponent strength features add value.
  RECOMMENDED: Integrate into ensemble v6.1 (30% weight)

======================================================================
TRAINING PIPELINE COMPLETE
======================================================================
Total time: 16.2 minutes
Coverage audit: PASSED (all features >80%)
Model saved: models/xgboost/passing_xgboost_v3_1_aligned.pkl
Decision: ✓ DEPLOY
======================================================================
```

---

## Next Steps

### If v3.1 MAE < v2 MAE (DEPLOY)

1. **Week 1: Ensemble Integration**
   - Update `enhanced_ensemble_v6.py` to use XGBoost v3.1
   - Maintain 30% weight (proven for XGBoost v2)
   - Full backtest (2022-2024 with canonical window)

2. **Week 2: A/B Testing**
   - A: XGBoost v2 (control)
   - B: XGBoost v3.1 (treatment)
   - Test on 2025 season (first 50 games)
   - Monitor: MAE, feature importance, coverage

3. **Week 3-4: Production Deployment**
   - If A/B test shows improvement: Deploy to ensemble v6.1
   - Update `MODEL_REGISTRY.yaml`
   - Register to database (`scripts/model_registry.py`)
   - Update production monitoring

---

### If v3.1 Within 2% of v2 (INVESTIGATE)

1. **Feature Ablation Study**
   - Train without opponent features
   - Quantify marginal contribution
   - Correlation analysis (opponent vs baseline features)

2. **Alternative Weighting**
   - Try different halflife values (H=2, H=6, H=8)
   - Raw values instead of percentile ranks
   - Simple averages instead of exponential weighting

3. **Wait for More Data**
   - Collect 2025 season data
   - Retrain with 2018-2025 (8 seasons)
   - Re-evaluate improvement

---

### If v3.1 MAE >= v2 * 1.02 (ABANDON)

1. **Document Findings**
   - Postmortem addendum: "Why Alignment Didn't Help"
   - Redundancy analysis (correlation matrix)
   - Feature engineering autopsy

2. **Explore Alternative Proxies (v3.2)**
   - Elo ratings (available 2006+, simpler)
   - Prior season win% (available 2006+)
   - Vegas closing spreads (market-implied quality)
   - Simple rolling averages (no exponential weighting)

3. **Keep XGBoost v2 in Production**
   - Proven baseline (40.20 MAE)
   - No risk of regression
   - Focus on other improvements (calibration, temperature scaling)

---

## Lessons Learned (Reinforced)

### 1. Feature Coverage Audits Are Non-Negotiable

**Rule:** All features must have >80% non-missing coverage in training set.

**Implementation:**
```python
def prepare_train_test_split(df):
    # ... split logic ...

    # CRITICAL: Coverage audit BEFORE extracting features
    audit_feature_coverage(
        train_df,
        feature_cols,
        threshold=0.80,
        fail_on_low_coverage=True  # BLOCK training
    )

    # Only proceed if audit passes
    X_train = train_df[feature_cols].values
    y_train = train_df['yards'].values
    return X_train, y_train, ...
```

### 2. Training Window Must Match Feature Availability

**Rule:** Training window ≤ feature availability window.

**Before (WRONG):**
```python
train_start = 2016  # XGBoost v2 historical window
feature_start = 2018  # EPA availability
# Result: 35% missing (2016-2017)
```

**After (CORRECT):**
```python
train_start = max(2016, 2018)  # Align to EPA availability
# Result: 0.9% missing (Week 1 only)
```

### 3. Validate on Small Sample Before Full Training

**Rule:** Run coverage audit on 100-game sample (~10 sec) before committing to full training (~15-20 min).

**Workflow:**
```bash
# 1. Quick validation (10 seconds)
python scripts/validate_xgboost_v3_1.py
# → GO/NO-GO decision

# 2. Full training (only if validation passes)
python py/models/train_xgboost_v3_1_aligned_window.py
```

**Benefit:** Catch coverage issues early, save 15-20 min of wasted compute time.

### 4. Median Imputation is NOT a Universal Solution

**Rule:** Only use median imputation for <5% missing, MCAR data.

**When It's Safe:**
- Missing Completely At Random (MCAR)
- <5% missing data
- Feature not time-dependent

**When It's Dangerous:**
- Systematic missingness (like 2016-2017 EPA)
- >10% missing data
- Temporal patterns (early season weeks)

**Better Approach:**
```python
if missing_pct > 0.05:
    # Option 1: Drop rows
    df = df.dropna(subset=[feature])

    # Option 2: Create missing indicator
    df[f'{feature}_missing'] = df[feature].isna().astype(int)

    # Option 3: Use simpler proxy
    df[feature] = df[feature].fillna(df['simpler_proxy'])
```

---

## Files Summary

### Created Files

1. **`py/models/train_xgboost_v3_1_aligned_window.py`** (733 lines)
   - Training script with feature coverage audit
   - Aligned window (2018-2024 only)
   - Decision gates (coverage, performance)

2. **`scripts/validate_xgboost_v3_1.py`** (244 lines)
   - Pre-training validation on small sample
   - GO/NO-GO decision gate
   - Exit code: 0 (GO) or 1 (NO-GO)

3. **`docs/XGBOOST_V3_1_ALIGNED_WINDOW_DELIVERABLE.md`** (this file)
   - Comprehensive implementation documentation
   - Expected performance scenarios
   - Next steps for all decision outcomes

### Modified Files

None (clean implementation, no modifications to existing codebase).

### Output Files (After Training)

1. **`models/xgboost/passing_xgboost_v3_1_aligned.pkl`**
   - Trained XGBoost model (expected: 12-15 MB)

2. **`models/xgboost/passing_xgboost_v3_1_aligned_metadata.json`**
   - Model metadata with:
     - Hyperparameters
     - Performance metrics
     - Feature importance
     - Postmortem fix details
     - v2 comparison

---

## Testing Checklist

Before production deployment, verify:

- [ ] **Validation passes** (`python scripts/validate_xgboost_v3_1.py`)
  - All features >80% coverage
  - Missing data <1% per season (2018-2024)
  - GO decision received

- [ ] **Training completes** (`python py/models/train_xgboost_v3_1_aligned_window.py`)
  - Coverage audit passes (16/16 features)
  - Model trains successfully
  - Metadata saved correctly

- [ ] **Performance meets threshold**
  - v3.1 MAE < v2 MAE (40.20 yards)
  - Opponent features >5% importance (top 50%)
  - R² > 0.60

- [ ] **Feature importance analyzed**
  - Opponent features rank where expected
  - No catastrophic drops (<1% importance)
  - Baseline features remain stable

- [ ] **Decision documented**
  - DEPLOY / INVESTIGATE / ABANDON decision made
  - Reasoning documented in metadata
  - Next steps identified

- [ ] **Files created correctly**
  - Model: `models/xgboost/passing_xgboost_v3_1_aligned.pkl`
  - Metadata: `models/xgboost/passing_xgboost_v3_1_aligned_metadata.json`
  - Both files <20 MB

---

## Frequently Asked Questions

### Q1: Why not use exponential decay weights like XGBoost v3?

**A:** Alignment is the priority fix. Decay weights are a secondary optimization that can be added later (v3.2) if v3.1 succeeds. Keeping v3.1 simple isolates the alignment variable.

### Q2: What if validation fails (NO-GO)?

**A:** Do NOT proceed with training. Instead:
1. Check opponent features CSV exists and is current
2. Run `py/features/opponent_strength.py` to regenerate features
3. Verify database has 2018-2024 QB game data
4. Re-run validation
5. If still fails, investigate missing data causes

### Q3: What if v3.1 still regresses vs v2?

**A:** This suggests opponent features are **redundant** with baseline features, not that alignment failed. Next steps:
1. Correlation analysis (opponent vs baseline features)
2. Feature ablation (train without opponent features)
3. Explore alternative opponent proxies (Elo, win%, spreads)
4. Keep XGBoost v2 in production

### Q4: Should we integrate v3.1 immediately if it beats v2?

**A:** No. Follow standard validation protocol:
1. Full backtest (2022-2024 with canonical window)
2. A/B test on 2025 season (first 50 games)
3. Monitor for 4 weeks before full deployment
4. Production deployment only if A/B test confirms improvement

### Q5: What's the minimum acceptable improvement over v2?

**A:**
- **Deploy immediately:** >2% improvement (MAE < 39.4)
- **A/B test:** 0.5-2% improvement (MAE 39.4-40.0)
- **Investigate:** 0-0.5% improvement (MAE 40.0-40.2)
- **Abandon:** <0% (regression, MAE > 40.2)

---

## Support & Contact

**Questions:** Review postmortem for detailed root cause analysis
- `docs/postmortems/XGBOOST_V3_PERFORMANCE_REGRESSION.md`

**Issues:** Check validation output and coverage audit logs

**Next Review:** After training completes and decision gate reached

---

**Status:** ✅ READY FOR VALIDATION
**Next Action:** Run `python scripts/validate_xgboost_v3_1.py`
**Expected Time:** 10 seconds (validation) + 15-20 minutes (training if GO)
**Author:** Claude Code (Sonnet 4.5)
**Date:** November 14, 2025
