# Data Leakage Audit Report

**Date:** 2025-10-30
**Analyst:** Claude Code
**Status:** 🔴 **CRITICAL ISSUES FOUND**

---

## Executive Summary

The model showing 81% win rate and +13,859% ROI contains **severe data leakage** from vegas-derived features. The results are **statistically impossible** and cannot be replicated in live betting.

---

## Key Findings

### 1. VEGAS FEATURE LEAKAGE 🔴

The top 2 "predictive" features are:

```json
{
    "top_10_phase5_features": [
        "home_implied_prob",    // ← MARKET'S PREDICTION
        "away_implied_prob",    // ← MARKET'S PREDICTION
        "spread_magnitude",
        "extreme_cold",
        ...
    ]
}
```

**Why This Is Leakage:**
- `home_implied_prob` and `away_implied_prob` are derived from **closing lines**
- They encode the market's collective wisdom: bookmakers' best estimate of win probability
- Using these is circular: "When market thinks team has 65% chance, they win 65% of the time"
- **We're not predicting - we're just learning the market is efficient!**

**Analogy:**
```
Task: Predict tomorrow's weather
Leaking Feature: Tomorrow's actual weather report
```

---

### 2. UNREALISTIC RESULTS

#### Our "Results":
| Metric | Value | Reality Check |
|--------|-------|---------------|
| Win Rate | 81.0% | Impossible in efficient markets |
| ROI (2 years) | +13,859% | $10k → $1.4M (absurd) |
| Sharpe Ratio | 2.94 | Higher than Renaissance Technologies |

#### Professional Benchmarks:
| Level | Win Rate | Annual ROI | Sharpe |
|-------|----------|------------|--------|
| Elite Professional | 56% | 3-5% | 1.5 |
| Top Sportsbook | 54.5% | 3% | 1.0 |
| Break Even | 52.4% | 0% | 0.0 |
| **Our Model** | **81%** | **~6,930%/yr** | **2.94** |

**Our model is 1.45x better than elite professionals** - statistically impossible without leakage.

---

### 3. KELLY SIZING WITHOUT CONSTRAINTS

#### Current Implementation:
```python
# From game_outcome_backtest.py
kelly_fraction = edge / (1 - implied_prob)
bet_fraction = kelly_multiplier * kelly_fraction

# NO CAPS! If model predicts 0.95, Kelly suggests betting 40%+ of bankroll
```

#### Problems:
- **No bet size cap**: Can bet 50%+ of bankroll on "sure things"
- **No absolute cap**: No $5,000 max bet limit
- **No probability clipping**: Overconfident 0.95+ predictions not capped
- **Exponential growth**: Lucky streak + overconfidence = fantasy returns

#### Reality:
Professional bettors **NEVER** bet more than 5% of bankroll per game, regardless of edge.

---

### 4. ROOT CAUSE ANALYSIS

#### How Did This Happen?

1. **Phase 5 Feature Engineering** (py/features/feature_loader.py):
   - Added vegas-derived features: `home_implied_prob`, `away_implied_prob`
   - These come from `spread_close` column in database
   - Closing line = market's final prediction = essentially the answer

2. **Feature Selection** (analysis/phase5_feature_importance/):
   - XGBoost correctly identified these as "most important"
   - Because they have highest correlation with target (they ARE the target!)
   - Model learned: `if home_implied_prob > 0.6: predict home_win = True`

3. **No Reality Checks**:
   - No warning when win rate > 60%
   - No sanity checks on ROI
   - No probability clipping

---

## Recommendations

### IMMEDIATE ACTIONS (Critical)

#### 1. Remove Vegas Leakage Features
```python
# Remove these from top_phase5_features:
BANNED_FEATURES = [
    "home_implied_prob",   # ← LEAKAGE
    "away_implied_prob",   # ← LEAKAGE
    "spread_magnitude",    # Derived from answer
    "is_large_spread",     # Derived from answer
    "home_is_favorite",    # Derived from answer
]

# Keep only legitimate features:
ALLOWED_FEATURES = [
    "extreme_cold",              # Weather (external)
    "dome_team_outdoor_wind",    # Weather (external)
    "is_conference_game",        # Schedule (known)
    "away_short_rest",           # Rest (known)
    "high_wind",                 # Weather (external)
]
```

#### 2. Add Betting Constraints
```python
# In GameOutcomeBacktest:
MAX_BET_FRACTION = 0.05  # Never bet more than 5% of bankroll
MAX_BET_AMOUNT = 5000    # Absolute cap: $5,000 per game
MIN_EDGE = 0.03          # Only bet if edge ≥ 3%
PROB_CLIP_MIN = 0.10     # Clip probabilities to [0.10, 0.90]
PROB_CLIP_MAX = 0.90     # Prevent overconfidence
```

#### 3. Add Reality Checks
```python
# Warn if results are unrealistic:
if win_rate > 0.60:
    print("⚠️  WARNING: Win rate > 60% indicates data leakage!")

if roi_per_year > 0.10:
    print("⚠️  WARNING: ROI > 10%/year is unrealistic!")

if sharpe_ratio > 2.0:
    print("⚠️  WARNING: Sharpe > 2.0 suggests overfitting!")
```

---

### MEDIUM-TERM ACTIONS (Important)

#### 4. Strict Time-Based Validation
```python
# Ensure no look-ahead bias:
# - Features for 2023-09-10 game can ONLY use data from before 2023-09-10
# - No "season stats" that include future games
# - Verify spread_close is from BEFORE game, not AFTER
```

#### 5. Feature Provenance Tracking
```python
# Document each feature's source:
FEATURE_PROVENANCE = {
    "extreme_cold": "weather API (external, no leakage)",
    "home_implied_prob": "closing line (LEAKAGE - remove!)",
    "home_win_pct": "team stats through previous week (safe)",
}
```

---

## Expected Realistic Results

After fixes, expect results similar to:

| Metric | Before (Leaking) | After (Realistic) |
|--------|------------------|-------------------|
| Win Rate | 81% | 52-56% |
| 2-Year ROI | +13,859% | +5-15% |
| Sharpe Ratio | 2.94 | 0.5-1.5 |
| Annual ROI | ~6,930% | +2-7% |

**This is still excellent!** Professional bettors with 54% win rate make millions.

---

## Implementation Plan

### Phase 1: Quick Fix (2 hours)
1. ✅ Audit completed - leakage identified
2. Create `RealisticGameOutcomeEnsemble` without vegas features
3. Add betting constraints to backtest
4. Add reality check warnings
5. Rerun backtest

### Phase 2: Validation (1 day)
1. Verify no look-ahead bias in remaining features
2. Add feature provenance documentation
3. Cross-validate on multiple time periods
4. Walk-forward validation (2016→2024)

### Phase 3: Production (1 week)
1. Live paper trading with realistic model
2. Monitor actual vs predicted performance
3. Compare to market lines
4. Validate edge exists

---

## Lessons Learned

### What Went Wrong:
1. **Trusted feature importance blindly**: XGBoost said vegas features were "important" - because they leak!
2. **No sanity checks**: 81% win rate should have triggered alarms
3. **No domain knowledge validation**: Real bettors know 81% is impossible

### What Went Right:
1. **Caught it before deployment**: Would have lost money in live trading
2. **Good infrastructure**: Easy to swap out features and retest
3. **Reproducible research**: Can trace exactly what caused the issue

---

## Conclusion

The model's exceptional performance was due to **data leakage from vegas-derived features**. After removing these features and adding realistic constraints, we expect:

- ✅ Win rate: 52-56% (realistic, profitable)
- ✅ Annual ROI: 2-7% (excellent for betting)
- ✅ Sharpe: 0.5-1.5 (good risk-adjusted returns)

**This is still a successful project!** Even 54% win rate with proper bankroll management can generate significant returns.

---

## Next Steps

1. Create `RealisticGameOutcomeEnsemble` (remove vegas features)
2. Update `GameOutcomeBacktest` with constraints
3. Rerun Phase 5 comparison with realistic settings
4. Document results in dissertation with honesty about the leakage discovery
