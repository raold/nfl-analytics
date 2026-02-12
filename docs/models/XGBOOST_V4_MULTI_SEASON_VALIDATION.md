# XGBoost v4 Multi-Season Validation

**Status:** Ready for execution
**Created:** November 14, 2025
**Purpose:** Validate XGBoost v4 (Ridge Residuals) robustness across multiple seasons

---

## Overview

This document describes the comprehensive multi-season validation protocol for XGBoost v4 (Ridge Residuals), which showed 87% improvement on initial testing (2025 season: 5.10 MAE vs 40.20 baseline).

**Key Question:** Does the 87% improvement replicate consistently across independent seasons (2022, 2023, 2024)?

---

## Validation Protocol

### 1. Independent Season Testing

For each test year (2022, 2023, 2024):
- **Training data:** 2018 to (year - 1)
- **Test data:** Full season (year)
- **Comparison:** v2 baseline (40.20 MAE)
- **Statistical test:** Diebold-Mariano (one-sided, α=0.05)

### 2. Success Criteria

**Per-Season Gates:**
1. MAE < 8 yards (87% improvement threshold)
2. Improvement > 80% vs baseline
3. Diebold-Mariano p-value < 0.05

**Consistency Gates:**
1. All 3 seasons pass per-season gates
2. MAE standard deviation < 2 yards
3. Mean improvement > 85%

### 3. Production Recommendation

**PASS:** All 6 gates passed → Integrate into ensemble v6 (20% weight)
**FAIL:** Any gate failed → Further investigation required

---

## Files

### Validation Script

**Location:** `scripts/xgb_v4_multi_season_validation.py`

**Features:**
- Multi-season training and evaluation (2022, 2023, 2024)
- Diebold-Mariano statistical significance testing
- Consistency validation across seasons
- Comprehensive result reporting

**Runtime:** ~30-45 minutes (3 seasons × 10-15 min each)

**Usage:**
```bash
cd \
python scripts/xgb_v4_multi_season_validation.py
```

### Comprehensive Tests

**Location:** `tests/models/test_xgboost_v4_validation.py`

**Test Coverage:**
1. **Ridge Coefficient Stability** (3 tests)
   - Stability across seasons
   - Sign consistency (opponent strength should be negative)
   - Magnitude reasonableness

2. **Residual Distribution Consistency** (3 tests)
   - Normality (Shapiro-Wilk test)
   - Mean near zero
   - Homoskedasticity (constant variance)

3. **Feature Importance Stability** (2 tests)
   - Consistency across seasons
   - Residual feature prominence

4. **Prediction Consistency** (2 tests)
   - Deterministic behavior (same input → same output)
   - Full pipeline determinism

5. **Error Handling** (4 tests)
   - Missing values
   - Extreme residuals
   - Zero variance features
   - Perfect collinearity

6. **Property-Based Tests** (3 tests, 60 examples)
   - Residuals sum to zero
   - Residuals orthogonal to predictions
   - Variance reduction from residual feature

7. **Integration Tests** (1 test)
   - Full two-stage pipeline end-to-end

8. **Regression Tests** (1 test)
   - v4 beats known v2 baseline

**Total:** 19 tests (60+ property-based examples)

**Usage:**
```bash
cd \
pytest tests/models/test_xgboost_v4_validation.py -v --tb=short
```

---

## Expected Results

Based on initial validation (2025 season: 5.10 MAE, 2024 backtest: 4.27 MAE):

| Season | Expected MAE | Expected Improvement | Expected p-value |
|--------|--------------|---------------------|------------------|
| 2022   | 5-7 yards    | 83-88%              | < 0.001          |
| 2023   | 5-7 yards    | 83-88%              | < 0.001          |
| 2024   | 4.27 yards   | 89.4%               | < 0.001          |

**Consistency Metrics:**
- Mean MAE: ~5.5 yards
- MAE std dev: ~1.0 yards (well below 2-yard threshold)
- Mean improvement: ~87%

---

## Output Files

### 1. Summary JSON

**File:** `results/xgb_v4_multi_season_validation_summary_{timestamp}.json`

**Contents:**
```json
{
  "validation_timestamp": "20251114_120000",
  "baseline_mae": 40.20,
  "test_seasons": [2022, 2023, 2024],
  "per_season_results": {
    "2022": {
      "mae": 5.8,
      "improvement_pct": 85.6,
      "dm_p_value_one_sided": 0.0001,
      ...
    },
    ...
  },
  "consistency_metrics": {
    "mae_mean": 5.5,
    "mae_std": 1.1,
    "all_gates_pass": true
  },
  "production_recommendation": "PASS"
}
```

### 2. Statistical Significance CSV

**File:** `results/xgb_v4_statistical_significance_{timestamp}.csv`

**Columns:**
- season
- mae
- baseline_mae
- improvement_yards
- improvement_pct
- dm_statistic
- dm_p_value_one_sided
- dm_p_value_two_sided
- significant_at_0.05

### 3. Per-Season Full Results

**Files:** `results/xgb_v4_season_{year}_full_{timestamp}.json`

**Contents:** Complete results including predictions, actuals, feature importance

---

## Architecture: Two-Stage Pipeline

### Stage 1: Ridge Regression (Opponent Features)

**Purpose:** Isolate opponent strength effects

**Features:**
- `opp_def_epa_exp` - Opponent defensive EPA
- `opp_points_allowed_avg` - Opponent points allowed average
- `matchup_difficulty` - Categorical (0=easy, 1=medium, 2=hard)

**Output:** Residuals (opponent-adjusted QB performance)

**Why Ridge?** Handles multicollinearity in opponent features, provides stable coefficients

### Stage 2: XGBoost (Baseline + Residuals)

**Purpose:** Predict passing yards using player features + opponent-adjusted residuals

**Features:**
- `pass_yards_l3` - Player 3-game average (most predictive)
- `pass_attempts_l3` - 3-game average attempts
- `completions_l3` - 3-game average completions
- `completions_pct_l3` - Completion percentage
- `home_away_encoded` - Home/away indicator
- **`residual_feature`** - Opponent-adjusted performance from Ridge

**Output:** Final passing yards prediction

**Why XGBoost?** Non-linear interactions, handles residual feature naturally

---

## Key Insights

### 1. Why Multi-Season Validation?

**Problem:** Single-season validation can be misleading
- 2024 might be anomalous year (rule changes, injuries, etc.)
- Overfitting to specific season characteristics
- Need proof of generalization

**Solution:** Independent validation on 3 consecutive seasons
- 2022: Pre-rule-change era
- 2023: Transition year
- 2024: Current era

### 2. Why Diebold-Mariano Test?

**Problem:** Simple MAE comparison can be misleading
- Ignores prediction error correlation
- Doesn't account for sample size
- No formal hypothesis test

**Solution:** Diebold-Mariano test
- Accounts for autocorrelation in forecast errors (h=1 lag)
- Provides p-value for formal significance testing
- Asymptotically valid under weak conditions

### 3. Why Consistency Checks?

**Problem:** High variance across seasons indicates instability
- Model might work well on some years, poorly on others
- Not suitable for production (unpredictable performance)

**Solution:** Cross-season variance gates
- MAE std dev < 2 yards ensures stable performance
- Mean improvement > 85% ensures consistent gains
- All seasons significant ensures robust improvement

---

## Comparison to Historical Validation

### Incremental XGBoost Investigation (Nov 10-12, 2025)

**Lesson Learned:** Always validate baselines independently

**Initial Claim:** Incremental XGBoost +21% vs "static baseline" (55.16 MAE)
**Error Discovered:** 55.16 was mean predictor, not XGBoost (39.76 MAE)
**Corrected Result:** Pure incremental -13.7% vs proper static baseline

**Why XGBoost v4 Avoids This:**
1. Explicit baseline defined (v2 XGBoost: 40.20 MAE)
2. Multi-season testing (not single year)
3. Statistical significance tests (not just MAE comparison)
4. Independent validation seasons (2022, 2023, 2024)

See: `docs/POSTMORTEMS.md` for full incremental XGBoost postmortem

---

## Next Steps

### If Validation PASSES (Expected)

1. **Week 2:** Integrate into ensemble v6 (20% weight)
   - Replace XGBoost v2 baseline
   - Update `py/ensemble/enhanced_ensemble_v6.py`
   - Retrain ensemble on canonical window (2018-2024 + H=4)

2. **Week 3:** A/B test on 2025 season
   - First 50 games (Sep-Oct 2025)
   - Compare ensemble v6 (with v4) vs ensemble v5 (without v4)
   - Success: ROI improvement > 1%

3. **Week 4-5:** Production deployment
   - Full season deployment if A/B test succeeds
   - Real-time prediction API integration
   - Monitoring and alerting

### If Validation FAILS (Unlikely)

1. **Investigate failed seasons**
   - Which gates failed?
   - Seasonal characteristics (rule changes, injuries, etc.)
   - Feature engineering issues?

2. **Hybrid approach**
   - Season-specific models?
   - Ensemble weights by season?
   - Feature selection by season?

3. **Re-validation**
   - Fix identified issues
   - Re-run multi-season validation
   - Re-evaluate production readiness

---

## References

### Statistical Methods

1. **Diebold-Mariano Test:**
   - Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.

2. **Ridge Regression:**
   - Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.

3. **XGBoost:**
   - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD*.

### Project Documentation

- `docs/POSTMORTEMS.md` - Incremental XGBoost baseline error
- `docs/ARCHITECTURE_REFERENCE.md` - System architecture
- `CLAUDE.md` - Project overview and status

---

## Contact

**Questions or issues?**
- Check `CLAUDE.md` for project context
- Review `docs/POSTMORTEMS.md` for lessons learned
- See `tests/models/test_xgboost_v4_validation.py` for test examples

**Validation timestamp:** Run script to generate timestamp
**Next review:** After validation completion
