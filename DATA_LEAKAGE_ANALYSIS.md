# Data Leakage Analysis - November 2, 2025

## Executive Summary

**CRITICAL FINDING**: Both the original backtest (`full_historical_backtest.py`) and cleaned version still show unrealistic ROI due to data leakage, though improved from 4,000,000% to 51,000%.

## Original Backtest Issues (SEVERE LEAKAGE)

### File: `py/backtest/full_historical_backtest.py`

**Problem** (lines 54-72):
```sql
play_stats AS (
    SELECT
        game_id,
        posteam,
        AVG(epa) as avg_epa,          -- ❌ USES ACTUAL GAME PLAYS
        AVG(yards_gained) as avg_yards -- ❌ USES ACTUAL GAME YARDS
    FROM plays
    WHERE posteam IS NOT NULL
    GROUP BY game_id, posteam
)
```

**Impact**:
- Features: `home_epa`, `away_epa`, `epa_diff`
- Calculated FROM actual game plays (post-game data)
- Model can "see the future"
- Result: **+4,469,906% ROI** (State-Space), 61.2% win rate

## Clean Backtest Attempt (STILL HAS LEAKAGE)

### File: `py/backtest/clean_historical_backtest.py`

**Improvements Made**:
- Removed post-game EPA queries
- Used `asof_features.py` rolling features
- Features use `.shift(1)` to exclude current game
- Only pre-game betting lines (`spread_close`, `total_close`)

**Features Used**:
```python
clean_feature_cols = [
    'spread_close', 'total_close',  # Pre-game lines
    'prior_epa_mean_diff',           # Rolling EPA (excludes current)
    'epa_pp_last3_diff',             # Last 3 games EPA
    'margin_last3_diff',             # Last 3 games margin
    'prior_win_pct_diff',            # Win % before current game
    'rest_diff', 'qb_change_diff', 'coach_change_diff'
]
```

**Results** (Test Period 2020-2024):
- Logistic Regression: **+51,106% ROI**, 62.4% win rate
- XGBoost: -99.7% ROI (went broke)

**Analysis**:
- 62% win rate is STILL unrealistic for NFL spreads
- Market efficiency dictates max ~54-55% sustainable win rate
- Suggests **HIDDEN LEAKAGE** somewhere in feature pipeline

## Potential Remaining Leakage Sources

### 1. Same-Season Training/Testing
**Issue**: Training on 1999-2019, testing on 2020-2024
- BUT: If any features use season-wide aggregations, they might leak

### 2. Week Ordering Issues
**Issue**: Games within same week might not be properly ordered
- If Week 1 games processed in wrong order, later games could leak to earlier

### 3. QB/Coach Change Detection
**Issue**: How is "change" detected?
- If comparing to current game's starter, that's leakage
- Should compare to PREVIOUS game's starter only

### 4. Market Line Quality
**Issue**: `spread_close` is closing line (pre-game)
- BUT: Closing lines are informationally efficient
- A 62% win rate against closing lines is nearly impossible

## Recommendations

### Immediate Actions Required

1. **Audit asof_features.py Line-by-Line**
   - Verify ALL aggregations use `.shift(1)`
   - Check week ordering logic
   - Validate QB/coach change detection

2. **Use Spread Only as Baseline**
   - Test model with ONLY `spread_close` feature
   - Should achieve ~52-53% win rate (random)
   - If >55%, confirms leakage

3. **Add Stricter Validation**
   ```python
   # In clean_historical_backtest.py
   def validate_no_leakage(df, test_season):
       """Verify test set has no future data."""
       test_df = df[df['season'] == test_season]

       # Check: All prior_* features use only past data
       assert test_df['home_prior_games'].min() > 0, "Found Week 1 games in test"

       # Check: No future season data
       train_df = df[df['season'] < test_season]
       assert len(train_df) > 0, "No training data"

       # Check: Reasonable win rates
       baseline_acc = 0.5 + abs(test_df['spread_close'].mean()) / 14
       assert baseline_acc < 0.55, f"Baseline {baseline_acc:.1%} too high"
   ```

4. **Simplify Feature Set**
   - Start with ONLY spread + total
   - Add ONE feature at a time
   - Measure ROI impact of each feature
   - Any single feature causing >2% ROI boost is suspicious

### Longer-Term Fixes

1. **Rewrite Feature Pipeline**
   - Use pandas `.expanding()` with explicit `min_periods`
   - Never use `.cumsum()` without explicit exclusion
   - Add unit tests for every rolling calculation

2. **Cross-Validation by Season**
   - Use `TimeSeriesSplit` from sklearn
   - Never train and test on same season
   - Leave 1-season gaps between train/test

3. **Expected Performance Benchmarks**

| Scenario | Expected Win Rate | Expected ROI |
|----------|------------------|--------------|
| Random guessing | 50% | -4.5% (vig) |
| Spread only | 52-53% | -1% to +1% |
| Good model | 53-54% | +1% to +3% |
| Excellent model | 54-55% | +3% to +5% |
| **Impossible** | >56% | >+5% |

## Files Modified

1. **Created**: `py/backtest/clean_historical_backtest.py`
   - Uses asof_features.py rolling statistics
   - Removed post-game EPA queries
   - Still shows leakage (needs further investigation)

2. **Generated**: `data/processed/features/asof_clean_backtest.csv`
   - 6,988 games with 100+ rolling features
   - Properly time-shifted (in theory)
   - Validation needed

3. **Original** (DO NOT USE): `py/backtest/full_historical_backtest.py`
   - Severe data leakage
   - Uses post-game EPA from plays table
   - Results are completely invalid

## Conclusion

While we eliminated the OBVIOUS leakage (post-game EPA), the 62% win rate indicates SUBTLE leakage remains. The feature engineering pipeline needs a complete audit to ensure temporal integrity.

**Bottom Line**: Current backtest results are NOT trustworthy for evaluating model performance. Further investigation required before any ROI claims can be validated.

---

**Date**: November 2, 2025
**Author**: Claude Code (Sonnet 4.5)
**Status**: Investigation ongoing - do not use current ROI metrics for production decisions
