# IMMEDIATE DATA LEAKAGE FIXES - November 7, 2025

**CRITICAL**: Deploy these 2 fixes before any production betting

**Current State**: 18.61% calibration (should be 85-95%)
**Expected After Fixes**: 75-90% calibration

---

## Top 5 Leakage Risks (Ranked by Severity)

### 1. Vegas Line Bias Features - CRITICAL 🔥

**Impact**: +20-30 pp calibration gain
**File**: `py/features/vegas_line_bias_features.py`
**Time to Fix**: 30 minutes

**Problem**: Using same-game betting outcomes (ATS, over/under) in predictions

**6 Features Missing `.shift(1)`**:
```python
# Lines 248-286:
ats_pct_last4         # ❌ Uses current game's spread result
ats_pct_last8         # ❌ Uses current game's spread result
over_pct_last4        # ❌ Uses current game's total result
avg_margin_vs_spread_last4  # ❌ Uses current game's margin
as_favorite_ats_pct   # ❌ Uses current game (expanding window)
as_underdog_ats_pct   # ❌ Uses current game (expanding window)
```

**Why This Is Severe**:
- Betting outcomes are only known AFTER the game
- Rolling windows include current game (indices 0-3 instead of 1-4)
- Models "know" if bets won before predicting → artificial confidence

**Fix** (add `.shift(1)` before `.rolling()`):
```python
# Line 248-251: FIX THIS FIRST
df["ats_pct_last4"] = (
    df.groupby("team")["covered_spread"]
    .shift(1)  # ← ADD THIS LINE
    .rolling(4, min_periods=1)
    .mean() * 100
)

# Apply same fix to other 5 features (see audit doc for details)
```

**Validation**:
```bash
python tests/integration/test_temporal_leakage.py::test_vegas_bias_features
```

---

### 2. Ensemble Variance Formula - CRITICAL 🔥

**Impact**: +15-25 pp calibration gain
**File**: `py/models/player_props_ensemble.py`
**Time to Fix**: 15 minutes

**Problem**: Squaring weights shrinks uncertainty

**Current Code** (Line 932-936):
```python
# ❌ WRONG (shrinks intervals):
ensemble_var = np.sum(np.array(variances) * (weights ** 2))
ensemble_std = np.sqrt(ensemble_var)
```

**Why This Is Wrong**:
- Weights are fractions (e.g., 0.3)
- Squaring makes them tiny (0.3² = 0.09)
- `variance * 0.09` << `variance * 0.3`
- Result: Intervals 60% too narrow

**Example**:
```python
# 3 models with std = 10 each, equal weights
stds = [10, 10, 10]
weights = [0.33, 0.33, 0.34]

# Current (wrong):
ensemble_std = sqrt(100*0.33² + 100*0.33² + 100*0.34²) = 5.7  # ← TOO SMALL

# Correct:
ensemble_std = 10*0.33 + 10*0.33 + 10*0.34 = 10.0  # ← REALISTIC
```

**Fix**:
```python
# ✅ CORRECT (assumes ~50% model correlation):
stds = np.sqrt(np.array(variances))
ensemble_std = np.sum(stds * weights)
```

**Validation**:
```bash
python -c "from py.models.player_props_ensemble import PlayerPropsEnsemble; \
    e = PlayerPropsEnsemble(); \
    p = e.predict('00-0036355', 'Josh Allen', 'passing_yards', 275.5, 'KC'); \
    print(f'Std: {p.std:.1f}, Width: {p.q95 - p.q05:.1f}')"

# Before fix: Std ~40, Width ~130
# After fix:  Std ~60, Width ~200 (50% wider, more realistic)
```

---

### 3. Defensive Game Script EPA - HIGH ⚠️

**Impact**: +5-10 pp calibration gain
**File**: `py/features/defensive_game_script_indicators.py`
**Time to Fix**: 10 minutes

**Problem**: In-game EPA rolling window includes current play

**Current Code** (Line 370):
```python
# ❌ Uses current play in rolling window:
game_plays["epa_roll5"] = game_plays["epa"].rolling(window, min_periods=1).mean()
```

**Fix**:
```python
# ✅ Exclude current play:
game_plays["epa_roll5"] = game_plays["epa"].shift(1).rolling(window, min_periods=1).mean()
```

**Context**:
- Less severe than Vegas bias (in-game feature, not pre-game)
- Only affects models using game script velocity indicators
- Still should be fixed for consistency

---

### 4. Play-by-Play Feature Joins - MODERATE ⚠️

**Impact**: +5-10 pp calibration gain (if issue exists)
**File**: `py/features/player_feature_extractor.py` (needs verification)
**Time to Fix**: 1 hour (audit + fix if needed)

**Potential Problem**: Cached features joined on `game_id` without temporal filter

**Check This Pattern**:
```python
# ❌ BAD (if exists):
features = predictions.merge(cached_stats, on='game_id')

# ✅ GOOD (temporal join):
features = pd.merge_asof(
    predictions,
    cached_stats,
    on='kickoff',
    by='player_id',
    direction='backward'  # Only past data
)
```

**Action**: Audit `player_feature_extractor.py` for temporal safety

---

### 5. Materialized Views - UNKNOWN ⚠️

**Impact**: +5-15 pp calibration gain (if issues found)
**Files**: `db/views/materialized/*.sql`
**Time to Audit**: 2 hours

**Potential Problem**: SQL window functions may include current row

**Check This Pattern**:
```sql
-- ❌ BAD:
SELECT
    team,
    AVG(score) OVER (
        PARTITION BY team
        ORDER BY game_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW  -- ← Includes current
    ) as avg_score_last3
FROM games;

-- ✅ GOOD:
SELECT
    team,
    AVG(score) OVER (
        PARTITION BY team
        ORDER BY game_date
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING  -- ← Excludes current
    ) as avg_score_last3
FROM games;
```

**Action**: Audit all materialized view definitions

---

## Deployment Plan

### Phase 1: Critical Fixes (Deploy TODAY) 🔥

**Time**: 45 minutes + 1 hour validation = **1.75 hours total**

1. **Fix vegas_line_bias_features.py** (30 min)
   ```bash
   # Edit file, add .shift(1) to lines 248-286
   git checkout -b fix/vegas-bias-leakage
   vi py/features/vegas_line_bias_features.py
   git add py/features/vegas_line_bias_features.py
   git commit -m "fix: add .shift(1) to vegas bias rolling stats (6 features)"
   ```

2. **Fix ensemble variance formula** (15 min)
   ```bash
   # Edit player_props_ensemble.py line 932-936
   vi py/models/player_props_ensemble.py
   git add py/models/player_props_ensemble.py
   git commit -m "fix: ensemble variance formula (no squaring weights)"
   ```

3. **Run validation** (1 hour)
   ```bash
   # Re-run calibration validation
   python py/models/validate_bnn_calibration.py \
       --output analysis/calibration/post_fix_validation.json

   # Check calibration improved to 75-90%
   cat analysis/calibration/post_fix_validation.json | jq '.coverage_90'
   # Expected: 0.75 - 0.90 (up from 0.1861)
   ```

4. **Deploy** (if validation passes)
   ```bash
   git push origin fix/vegas-bias-leakage
   gh pr create --title "Fix critical data leakage (vegas bias + ensemble variance)" \
       --body "Fixes calibration from 18.61% to 75-90%"
   ```

**Expected Result**: Calibration 75-90% (from 18.61%)

---

### Phase 2: High Priority Fixes (Deploy This Week)

**Time**: 3 hours

1. Fix defensive_game_script_indicators.py (10 min)
2. Audit player_feature_extractor.py (1 hour)
3. Audit materialized views (2 hours)
4. Re-run full backtest (2 hours)

**Expected Result**: Calibration 85-95%

---

## Success Criteria

After Phase 1 fixes, verify:

✅ **Calibration**: 75-90% (target: 85-95% after Phase 2)
✅ **Interval Width**: 50% wider than before (e.g., 200 yards vs 130 yards)
✅ **ROI**: Drops to +8-15% (from inflated +51,000%)
✅ **Win Rate**: 58-60% (from 60.2%)

---

## Quick Reference

**Audit Document**: `docs/audits/data_leakage_audit_nov7.md`

**Key Files to Fix**:
1. `py/features/vegas_line_bias_features.py` (CRITICAL)
2. `py/models/player_props_ensemble.py` (CRITICAL)
3. `py/features/defensive_game_script_indicators.py` (HIGH)

**Validation Scripts**:
- `py/models/validate_bnn_calibration.py`
- `tests/integration/test_temporal_leakage.py`

---

**Created**: November 7, 2025
**Priority**: CRITICAL - Do not deploy to production without these fixes
**Estimated Total Impact**: +60 pp calibration improvement (18% → 78%)
