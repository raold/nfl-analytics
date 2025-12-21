# Data Leakage Audit Report - UPDATED

**Date:** 2025-10-30
**Analyst:** Claude Code
**Status:** 🔴 **CRITICAL ISSUES FOUND**

---

## Executive Summary

The model showing 77.4% win rate (even after removing vegas features) contains **SEVERE LOOK-AHEAD BIAS** in baseline team statistics. The exceptional results are due to using **future game data** when making predictions.

---

## Investigation Timeline

### Phase 1: Vegas Feature Leakage (COMPLETED)
**Finding:** Top 2 features (`home_implied_prob`, `away_implied_prob`) are derived from closing lines
**Action Taken:** Created `RealisticGameOutcomeEnsemble` without these features
**Result:** Win rate decreased from 81% to 77.4% - **still unrealistic**

### Phase 2: Remaining Phase 5 Features (COMPLETED)
**Investigated:** 5 remaining features in realistic model
- `extreme_cold` - Weather (external) ✅ Clean
- `high_wind` - Weather (external) ✅ Clean
- `dome_team_outdoor_wind` - Weather + venue (external) ✅ Clean
- `is_conference_game` - Schedule (public) ✅ Clean
- `away_short_rest` - Rest days (public) ✅ Clean

**Finding:** NO indirect vegas leakage in Phase 5 features

### Phase 3: Baseline Features - **SMOKING GUN FOUND** 🚨

**Critical Issue:** `get_team_season_stats()` uses **FULL SEASON** statistics

```python
# py/ensemble/game_outcome_ensemble.py:47-101
def get_team_season_stats(season: int) -> pd.DataFrame:
    """Fetch team aggregated stats for a season."""
    sql = """
    WITH game_stats AS (
        SELECT season, week, home_team AS team, ...
        FROM games
        WHERE season = %s  -- ❌ Gets ALL games from season!
          AND home_score IS NOT NULL
    )
    SELECT team, COUNT(*) AS games_played, SUM(win) AS wins, ...
    FROM game_stats
    GROUP BY team  -- ❌ Aggregates ENTIRE season
    """
```

**The Problem:**

When predicting a game in Week 10 of 2023:
- Function loads `get_team_season_stats(2023)`
- This computes stats using **ALL 18 weeks** of 2023
- Model knows final season records (e.g., KC 14-3) **before** Week 10 happens
- This is **severe look-ahead bias** - using future information

**Example:**
```
Predicting: KC vs BUF, Week 10, 2023

Current (broken) behavior:
  → Uses KC's full 2023 season stats (14-3 record, 24.7 PPG)
  → Knows the future!

Correct behavior should be:
  → Uses only KC's weeks 1-9 stats (6-2 record)
  → OR uses only 2022 season stats (no 2023 data)
```

---

## Root Cause Analysis

### How the Leakage Occurs

1. **Training Phase:**
   - Model trains on 2019-2021 seasons
   - For each game, `get_team_season_stats(season)` loads **full season** stats
   - Model learns: "Teams with 14-3 records beat teams with 9-8 records"
   - This is trivially true - better teams win more!

2. **Prediction Phase:**
   - Backtest on 2022-2024 seasons
   - For a Week 10, 2023 game, loads `get_team_season_stats(2023)`
   - Provides model with **full 2023 season** stats (including future weeks 11-18)
   - Model makes "prediction" using knowledge of the future

3. **Result:**
   - 77.4% win rate (vs realistic 52-56%)
   - +469% annual ROI (vs realistic 3-5%)
   - Model appears to have "edge" but it's just cheating

### Why Removing Vegas Features Didn't Help

The optimized model (81% win rate) had **TWO sources** of leakage:
1. Vegas features (`home_implied_prob`, `away_implied_prob`) - 4% contribution
2. Look-ahead bias in baseline features - **primary source** (77% win rate)

Removing only vegas features left the **worse** leakage intact.

---

## Evidence

### Reality Check Output

Even without vegas features, realistic model shows:

```
⚠️  Win rate: 77.4% (UNREALISTIC - professionals achieve 54-56%)
⚠️  Annual ROI: 469.3% (UNREALISTIC - professionals achieve 3-5%)
⚠️  Sharpe ratio: 2.45 (UNREALISTIC - professionals achieve ~1.0-1.5)

VERDICT: WARNINGS - POSSIBLE DATA LEAKAGE OR OVERFITTING
```

### Comparison of Results

| Model | Vegas Features? | Look-Ahead Bias? | Win Rate | Annual ROI |
|-------|----------------|------------------|----------|------------|
| Optimized | ✅ Yes | ✅ Yes | 81.0% | 6,930% |
| Realistic | ❌ No | ✅ Yes | 77.4% | 469% |
| Expected (no leakage) | ❌ No | ❌ No | 52-56% | 2-7% |

---

## Recommendations

### IMMEDIATE ACTIONS (Critical Priority)

#### 1. Fix `get_team_season_stats()` to Use Only Past Data

**Option A: Use Previous Season Stats (Simplest)**
```python
def get_team_season_stats(season: int) -> pd.DataFrame:
    """
    Fetch team stats from PREVIOUS season only.

    For 2023 games, uses 2022 season stats (no look-ahead bias).
    """
    prior_season = season - 1
    sql = """
    ...
    WHERE season = %s  -- Use prior_season
    """
    return df
```

**Option B: Use Only Past Games from Current Season (More Complex)**
```python
def get_team_week_stats(season: int, week: int) -> pd.DataFrame:
    """
    Fetch team stats through week N-1 of current season.

    For Week 10 2023 game, uses only weeks 1-9 of 2023.
    """
    sql = """
    ...
    WHERE season = %s AND week < %s  -- Only past weeks
    """
    return df
```

**Recommendation:** Start with **Option A** (previous season) for simplicity and no leakage risk.

#### 2. Add Time-Based Validation Checks

```python
def validate_no_lookahead(game_date: date, feature_data_cutoff: date):
    """Ensure features only use data from before game_date."""
    assert feature_data_cutoff < game_date, \
        f"Leakage detected: Using data from {feature_data_cutoff} for game on {game_date}"
```

#### 3. Rerun All Backtests with Fixed Stats

After fixing `get_team_season_stats()`:
- Retrain all ensembles (baseline, optimized, realistic)
- Expect results to drop to realistic levels (52-56% win rate)
- This will validate the fix worked

---

## Expected Realistic Results (After Fix)

After fixing look-ahead bias and removing vegas features:

| Metric | Before (Leaking) | After (Realistic) |
|--------|------------------|-------------------|
| Win Rate | 77.4% | **52-56%** |
| Annual ROI | 469% | **2-7%** |
| Sharpe Ratio | 2.45 | **0.5-1.5** |
| Max Drawdown | -178% | **-10% to -20%** |

**Note:** Even 54% win rate with proper bankroll management is **profitable and excellent**!

---

## Lessons Learned

### What Went Wrong

1. **Insufficient time-based validation:**
   - No checks that features use only past data
   - Assumed SQL queries were correct without verification

2. **Over-trusting extraordinary results:**
   - 81% win rate should have triggered immediate investigation
   - "Too good to be true" is usually true in betting markets

3. **Sequential debugging:**
   - Fixed vegas features first (minor issue)
   - Should have audited baseline features first (major issue)

### What Went Right

1. **Reality check system worked:**
   - Automated warnings caught unrealistic results
   - Flagged leakage even after removing obvious sources

2. **Systematic investigation:**
   - Methodically checked all feature sources
   - Found root cause through code inspection

3. **Good infrastructure:**
   - Easy to test different feature sets
   - Reproducible experiments make debugging possible

---

## Next Steps

1. **Fix `get_team_season_stats()` to use previous season stats**
   - Modify SQL to use `season - 1`
   - Add validation that no current-season data is used

2. **Retrain all models with fixed baseline features**
   - Baseline ensemble (no Phase 5)
   - Realistic ensemble (no vegas, clean Phase 5)
   - Optimized ensemble (all features, no leakage)

3. **Add time-based validation framework**
   - Check all features use only past data
   - Prevent future leakage bugs

4. **Document results honestly in dissertation**
   - Explain the leakage discovery
   - Show before/after results
   - Emphasize importance of validation

---

## Conclusion

The model's exceptional performance (77-81% win rate) was due to **TWO** leakage sources:

1. **Vegas features** (4% impact): Direct market predictions
2. **Look-ahead bias** (23% impact): Using future game results in team stats

After fixing both issues, expect realistic performance:
- ✅ Win rate: 52-56% (professionals achieve 54-56%)
- ✅ Annual ROI: 2-7% (professionals achieve 3-5%)
- ✅ Sharpe: 0.5-1.5 (professionals achieve ~1.0)

**This is still a successful project!** A 54% win rate with 3% annual ROI is **excellent** in sports betting.

---

## Files Affected

**Created:**
- `analysis/DATA_LEAKAGE_AUDIT_REPORT_UPDATED.md` (this file)
- `py/audit/data_leakage_audit.py`
- `py/ensemble/game_outcome_ensemble_realistic.py`
- `py/backtests/realistic_backtest.py`

**Requires Fix:**
- `py/ensemble/game_outcome_ensemble.py:47-101` - `get_team_season_stats()`
- `py/ensemble/game_outcome_ensemble_optimized.py` - Inherits same bug
- `py/ensemble/game_outcome_ensemble_realistic.py` - Inherits same bug

---

**Status:** Ready for fix implementation and retraining
