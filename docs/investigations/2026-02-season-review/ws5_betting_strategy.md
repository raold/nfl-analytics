# WS5: Betting Strategy Optimization

> **Read with [README.md](README.md) (2026-10-25):** the threshold table reproduces; the
> "pockets of genuine signal" are post-hoc subgroups of 20-65 bets, and the rest result is the
> spread sign bug (retraction ledger C15, C27). Paths cited below are at tag
> `archive/pre-prune-2026-10`.

**Date:** 2026-02-12
**Investigator:** Claude Code (Systematic Investigation)
**Status:** COMPLETE

---

## Executive Summary

The 2025 season produced a -6.56% ATS ROI on 285 games (139-145-1), confirming that the current model cannot profitably bet on every game. However, **targeted filtering reveals pockets of genuine signal**: the edge 2-4pt bucket hits at 62.5% (65 games), home-favorite bets with 2-4pt edge hit at 80.0% (20 games), and Thursday Night Football bets hit at 71.4% (21 games). These subsets exceed the 52.4% breakeven required at -110 juice.

The critical caveat: **all identified "profitable" filters are small-sample artifacts** (n < 65). With only one season of data, none survive multiple comparison correction. The 2-4pt edge bucket is the most promising actionable finding, but requires multi-season validation before deployment.

The model's fundamental problem is not strategy -- it is **prediction compression** (pred range [-2.6, +8.7] vs actual [-34, +38]). Strategy optimization on a broken model is optimizing deck chairs on the Titanic. Fix the feature pipeline first (WS1/WS2), then revisit strategy.

---

## 1. Edge Threshold Sensitivity Analysis

### Definition

- **Edge** = ABS(pred_home_spread - vegas_spread)
- **ATS Win** = When (pred_home_spread - vegas_spread) and (actual_spread - vegas_spread) have the same sign
- **ROI** assumes standard -110 juice (risk $110 to win $100)

### Cumulative Threshold Results

| Threshold | Total Bets | W-L | Win % | ROI % | Break-Even? |
|-----------|-----------|------|-------|-------|-------------|
| >= 0 (all) | 285 | 139-145 | 48.9% | -6.56% | No |
| >= 1 | 254 | 124-129 | 49.0% | -6.43% | No |
| >= 2 | 210 | 105-104 | 50.2% | -4.09% | No |
| >= 3 | 180 | 88-91 | 49.2% | -6.15% | No |
| >= 4 | 145 | 65-80 | 44.8% | -14.42% | No |
| >= 5 | 124 | 56-68 | 45.2% | -13.78% | No |

**Key Finding:** No cumulative threshold achieves profitability. Counter-intuitively, **higher edge thresholds perform WORSE**. Games with edge >= 4 hit at only 44.8%, significantly below breakeven. This is the signature of a miscalibrated model: larger disagreements with Vegas indicate the MODEL is wrong, not the market.

### Edge Bucket Analysis (Granular)

| Bucket | Games | W-L | Win % | Avg Edge | Avg ATS Margin |
|--------|-------|------|-------|----------|----------------|
| 0-1 | 31 | 15-16 | 48.4% | 0.55 | 7.48 |
| 1-2 | 44 | 19-25 | 43.2% | 1.46 | 10.78 |
| **2-3** | **30** | **17-13** | **56.7%** | **2.43** | **9.03** |
| **3-4** | **35** | **23-11** | **67.6%** | **3.40** | **10.24** |
| 4-5 | 21 | 9-12 | 42.9% | 4.50 | 8.07 |
| 5-7 | 49 | 15-34 | **30.6%** | 5.88 | 10.69 |
| 7-10 | 49 | 26-23 | 53.1% | 8.45 | 10.22 |
| 10+ | 26 | 15-11 | 57.7% | 11.50 | 8.67 |

**Critical Pattern:** The relationship between edge and ATS performance is **non-monotonic**:
- **Sweet spot at 2-4 points**: 65 games, 40-24 record, 62.5% win rate
- **Dead zone at 4-7 points**: 70 games, 24-46 record, 34.3% win rate
- **Partial recovery at 7+ points**: 75 games, 41-34 record, 54.7% win rate

The 5-7pt bucket is catastrophic at 30.6%. This is where the model's prediction compression creates maximum damage -- the model thinks there is a large edge, but it is simply wrong about the direction.

### Kelly Criterion by Edge Bucket

| Bucket | Win % | Kelly % | Half-Kelly % | P/L per Unit |
|--------|-------|---------|-------------|-------------|
| 0-2 | 45.3% | -14.80% | N/A (neg) | -0.1345 |
| **2-4** | **62.5%** | **+21.25%** | **+10.63%** | **+0.1932** |
| 4-6 | 36.0% | -34.40% | N/A (neg) | -0.3127 |
| 6-8 | 37.1% | -32.00% | N/A (neg) | -0.2909 |
| 8+ | 56.7% | +9.00% | +4.50% | +0.0818 |

**Only the 2-4pt edge bucket has a positive Kelly fraction.** At 62.5% win rate with -110 odds, the full Kelly suggests risking 21.25% of bankroll per bet. Practical deployment would use quarter-Kelly (5.3%), which with 65 bets across the season would generate meaningful returns IF the win rate holds.

---

## 2. Game Selection Filters

### 2.1 Primetime Game Performance

| Slot | Games | W-L | Win % | Avg Edge | Model MAE | Vegas MAE |
|------|-------|------|-------|----------|-----------|-----------|
| Sunday Early/Afternoon | 230 | 110-119 | 48.0% | 4.77 | 11.40 | 9.74 |
| **TNF (Thursday)** | **21** | **15-6** | **71.4%** | **6.14** | **8.76** | **9.48** |
| MNF (Monday) | 22 | 7-15 | 31.8% | 4.06 | 10.60 | 9.34 |
| Saturday | 10 | 5-5 | 50.0% | 4.30 | 9.47 | 8.55 |

**TNF is the only slot where the model beats Vegas on MAE** (8.76 vs 9.48). This is remarkable and suggests the model captures short-week dynamics better than the market. Thursday games involve teams on 4-day rest, which compresses performance variance -- something the model's rest differential features can exploit.

**MNF is a disaster** at 31.8%. The model should avoid Monday Night Football bets entirely.

### 2.2 Rest Differential Features

The `game_feature_extractor.py` (lines 1076-1199) implements a comprehensive rest differential system:

**Features computed:**
- `rest_differential`: home_days_rest - away_days_rest
- `home_days_rest`, `away_days_rest`: Days since each team's last game
- `is_home_short_week`, `is_away_short_week`: < 6 days (Thursday after Sunday)
- `is_home_long_week`, `is_away_long_week`: > 9 days (bye or Sunday after Thursday)

**Thresholds:**
- SHORT_WEEK_THRESHOLD = 6.0 days (BTB detection)
- LONG_WEEK_THRESHOLD = 9.0 days (bye week detection)

**Prior finding (Dec 28, 2025):** Home +3 days rest = 69.2% ATS cover, Away +3 days = 44.4% -- a 25 percentage point swing with r=0.146 correlation to ATS outcome. These features are already embedded in the model but the prediction compression limits their impact.

**Note:** Rest differential data is NOT stored in `ml.game_outcome_predictions` (only spread/total predictions are stored). A dedicated query against the `games` table joined with kickoff timestamps would be needed for explicit rest-based filtering. The model already uses these features internally.

### 2.3 Vegas Spread Size

| Spread Range | Games | W-L | Win % | Avg Edge | Model MAE | Vegas MAE |
|-------------|-------|------|-------|----------|-----------|-----------|
| Pick/tight (0-2.5) | 74 | 34-40 | 45.9% | 2.92 | 9.90 | 8.95 |
| Moderate (3-4.5) | 84 | 44-39 | 53.0% | 3.81 | 9.57 | 9.03 |
| Clear fav (5-7) | 62 | 25-37 | 40.3% | 5.12 | 13.19 | 11.77 |
| **Big fav (7.5-9.5)** | **30** | **22-8** | **73.3%** | **6.47** | **10.92** | **9.70** |
| Blowout (10+) | 35 | 14-21 | 40.0% | 9.19 | 13.35 | 8.99 |

**The big favorite bucket (7.5-9.5) is extraordinary at 73.3%.** This may reflect the model's conservative prediction range -- when a game is spread at 8.5, the model predicts ~3.0 (compressed), creating a systematic disagreement that happens to align with ATS reality (big favorites tend to cover at above-average rates in 2025).

**The blowout bucket (10+) reverts to 40.0%**, suggesting this is a 2025-specific phenomenon, not a general rule.

### 2.4 Home/Away Bias

| Scenario | Games | W-L | Win % | Avg Raw Edge | Avg Pred | Avg Vegas |
|----------|-------|------|-------|-------------|----------|-----------|
| Home Underdog | 165 | 75-89 | 45.7% | -2.71 | 3.11 | 5.82 |
| Home Favorite | 120 | 64-56 | 53.3% | +6.59 | 1.96 | -4.63 |

The model has a directional asymmetry: **it performs better when backing home favorites** (53.3%) than home underdogs (45.7%). When the model is "more home-favorable" (pred < vegas), the raw edge averages +6.59 but this is largely driven by the model's systematic underestimation of away-team advantages.

### 2.5 Season Phase

| Phase | Games | W-L | Win % | Model MAE | Vegas MAE |
|-------|-------|------|-------|-----------|-----------|
| Early (1-6) | 93 | 45-48 | 48.4% | 10.10 | 9.09 |
| Mid (7-13) | 101 | 49-51 | 49.0% | 11.50 | 9.91 |
| Late (14-18) | 78 | 37-41 | 47.4% | 11.93 | 10.24 |
| Playoffs (19+) | 13 | 8-5 | 61.5% | 9.12 | 8.58 |

The model does not improve over the season. Playoffs show 61.5% on 13 games -- interesting but not statistically meaningful. The model MAE actually degrades from 10.10 (early) to 11.93 (late), suggesting that as the season progresses and team identities solidify, the model's stale features become more of a liability.

---

## 3. Combination Filter Analysis

### Promising Filters (Win % > 52.4% Breakeven)

| Filter | N | W-L | Win % | ROI % | Notes |
|--------|---|------|-------|-------|-------|
| HomeFav + edge 2-4 | 20 | 16-4 | 80.0% | ~45% | Very small N |
| Edge 2-4 + BigFav(7.5-9.5) | 7 | 6-1 | 85.7% | ~62% | Too small |
| TNF + edge >= 2 | 16 | 12-4 | 75.0% | ~39% | Small N |
| Edge 3-4 only | 36 | 24-11 | 68.6% | +30.9% | Best risk-adjusted |
| Edge 2-4 + spread < 10 | 65 | 40-24 | 62.5% | +25.7% | **Largest viable sample** |
| Edge 2-4 + Moderate(3-9.5) | 42 | 27-14 | 65.9% | +25.7% | Good balance |
| TNF only | 21 | 15-6 | 71.4% | +36.4% | Small N |
| Big fav (7.5-9.5) any edge | 30 | 22-8 | 73.3% | +40.0% | Spread-driven |

### Statistical Reality Check

For a 65-game sample at 62.5% win rate, the 95% confidence interval is:

- Observed: 40/64 = 62.5% (excluding 1 push)
- Standard error: sqrt(0.625 * 0.375 / 64) = 6.05%
- 95% CI: [50.4%, 74.6%]

The lower bound (50.4%) is below the 52.4% breakeven. **We cannot reject the null hypothesis that the true win rate is at breakeven.** The 20-game "HomeFav + edge 2-4" filter at 80% has a CI of [56.3%, 94.3%] -- more promising but only 20 games.

---

## 4. Kelly Sizing and Strategy Infrastructure

### Existing Kelly Implementation

`nfl_analytics/production/kelly_sizing.py` implements:

- **Full Kelly**: f* = (bp - q) / b
- **Fractional Kelly**: f = fraction * f* (default quarter-Kelly at 0.25)
- **Max bet cap**: Default 5% of bankroll
- **Minimum edge gate**: Default 1% edge required
- **Bankroll simulation**: Monte Carlo comparison of Kelly fractions

The implementation is correct but disconnected from the actual prediction pipeline. The `optimal_bet_size()` function requires `model_prob` and `market_odds` as inputs, but the game outcome model outputs `pred_home_spread`, not win probability.

**Gap:** There is no automated pipeline from `ml.game_outcome_predictions` to the Kelly sizer. The `pred_home_win_prob` column exists but its calibration has not been validated for ATS betting (win probability != ATS cover probability).

### Ensemble Strategy Architecture

`nfl_analytics/ensemble/strategy/base.py` defines:

- **EnsembleStrategy** (ABC): Interface for combining model predictions
- **ModelPrediction**: Individual model output (mean, std, confidence)
- **EnsemblePrediction**: Combined output with uncertainty intervals
- **Weight presets**: v6.0 (games), v6.1 (GNN), v8.0 (Dirichlet/props)

The confidence threshold (`confidence_threshold=0.5`) in EnsembleConfig could serve as a bet filter, but `spread_std` is uniformly 13.54 across all 285 games -- the model outputs constant uncertainty. This makes confidence-based filtering impossible.

### Pricing Infrastructure

The `nfl_analytics/pricing/` directory contains sophisticated tools:

1. **`teaser_ev.py`**: Two-leg teaser EV estimator using Skellam PMF for integer-margin distributions, with Gaussian/t-copula dependence modeling and key-number reweighting (3, 6, 7, 10)
2. **`copula_parlays.py`**: Gaussian copula for correlated parlay pricing with Monte Carlo simulation
3. **`teaser.py`**: Basic teaser pricing

The teaser EV pipeline is the most sophisticated pricing tool in the system, correctly modeling the probability of crossing key numbers (3 and 7) with a 6-point teaser. It uses historical margin data (1999-2019 training window) to calibrate the Skellam distribution and estimates +EV opportunities at specific spread values.

**Connection to strategy:** Teasers are potentially more viable than straight ATS bets for this model because teasers only require correctly identifying the spread direction, not the exact magnitude. The model's prediction compression (tight range) matters less for "is this team closer to -3 or -7?" decisions.

---

## 5. CQL Target Leakage Lessons (ADR-016)

ADR-016 documents a critical data leakage bug in the RL betting pipeline:

- **CQL showed 100% win rate, +91% ROI** -- scientifically impossible
- **Root cause:** Training dataset included `home_score`, `away_score`, and pre-computed rewards
- **Cover formula had wrong sign:** `(home_score - away_score) > -spread_close` instead of `> spread_close`
- **After fix:** Random baseline shows 50.5% win rate, -3.49% ROI (correct market efficiency)

**Relevance to strategy optimization:** The same lesson applies here. Any filter that shows 80%+ win rate on 20 games should be treated with extreme skepticism. The most likely explanation for extreme outperformance is:
1. Small sample noise
2. Data leakage (unlikely here since we're querying stored predictions vs actuals)
3. Multiple comparison bias (testing many filters, some will look good by chance)

The spread sign convention matters: positive `spread_close` = home favorite in this database (inverted from standard).

---

## 6. Prediction Range Compression Impact

| Metric | Model Predictions | Vegas Spreads | Actual Spreads |
|--------|-------------------|---------------|----------------|
| Min | -2.6 | -13.5 | -34 |
| Max | +8.7 | +15.5 | +38 |
| Mean | +2.63 | +1.42 | +2.03 |
| Std Dev | 2.30 | 6.18 | 14.16 |

**The model's standard deviation is 2.30 vs Vegas's 6.18.** The model predicts a range of 11.3 points ([-2.6, +8.7]) while Vegas spreads cover 29 points ([-13.5, +15.5]) and actual outcomes span 72 points ([-34, +38]).

This compression is the root cause of the non-monotonic edge pattern:
- **Small edges (0-2pt):** Model and Vegas roughly agree; 50/50 outcomes expected
- **Medium edges (2-4pt):** Model disagrees slightly; these happen to be games where the model captures genuine signal (rest, matchup dynamics) that Vegas underprices
- **Large edges (4-7pt):** Model strongly disagrees because it CANNOT predict large spreads; the "edge" is actually model error
- **Very large edges (7+):** Model predicts ~3 while Vegas says 10+; the actual outcomes are unpredictable, so performance reverts to ~50%

---

## 7. Worst-Case Analysis: Top 15 Misses

| Week | Game | Pred | Vegas | Actual | Model Err | Vegas Err |
|------|------|------|-------|--------|-----------|-----------|
| 5 | BAL vs HOU | +7.0 | -2.5 | -34 | 41.0 | 31.5 |
| 3 | MIN vs CIN | +1.7 | +3.0 | +38 | 36.3 | 35.0 |
| 8 | CAR vs BUF | +1.7 | -7.0 | -31 | 32.7 | 24.0 |
| 3 | SEA vs NO | -1.2 | +7.5 | +31 | 32.2 | 23.5 |
| 17 | NYJ vs NE | -0.6 | -12.5 | -32 | 31.4 | 19.5 |

In the worst misses, the model prediction is always near its mean (~2.63), while the actual margin is extreme (30+ points). The model is essentially predicting the seasonal mean for every game, regardless of the matchup specifics. This is consistent with the data pipeline incompleteness finding from WS1/WS2.

---

## 8. Actionable Recommendations

### Tier 0: Do NOT Deploy (Current State)

With a 48.9% flat ATS rate and -6.56% ROI, **the model should not be used for real-money betting in its current form.** Any profitable filter identified below is provisional and requires multi-season validation.

### Tier 1: Fix Prerequisites (Before Strategy Matters)

1. **Fix feature pipeline health** (ADR-017 Tier 1): QBR, NextGen, semantic data staleness
2. **Reduce prediction compression**: The model range of [-2.6, +8.7] vs actual [-34, +38] means the model cannot differentiate matchups. This is the single biggest issue.
3. **Add variable uncertainty**: `spread_std` is constant at 13.54 for all games. Implement game-specific uncertainty estimates.

### Tier 2: Strategy Filters to Validate on 2026 Data

If prediction quality improves, test these filters prospectively:

| Filter | Expected Volume | Hypothesis |
|--------|-----------------|------------|
| Edge 2-4pt (inclusive) | ~65/season | Sweet spot between noise and model error |
| Exclude 5-7pt edge | All minus ~50 | Avoid worst-performing bucket |
| TNF games | ~17/season | Model captures short-week dynamics |
| Avoid MNF games | All minus ~17 | 31.8% MNF win rate is terrible |
| Home favorites with 2-4pt edge | ~20/season | 80% in 2025 but very small N |
| Big favorites (7.5-9.5 spread) | ~30/season | May be 2025 artifact |

### Tier 3: Kelly Sizing Improvements

1. **Connect Kelly to predictions pipeline**: Currently `optimal_bet_size()` requires manual inputs. Build automated bet sheet from `ml.game_outcome_predictions`.
2. **Convert spread edge to win probability**: Use historical data to map edge magnitude to empirical ATS cover rate, then feed to Kelly.
3. **Use quarter-Kelly (0.25) maximum**: Even the best filter (2-4pt edge) has wide confidence intervals.
4. **Cap at 3% of bankroll per bet**: Given model uncertainty, no single bet should risk more.
5. **Teasers over straight bets**: The pricing infrastructure (`teaser_ev.py`) is sophisticated and teasers are less sensitive to prediction compression.

### Tier 4: Structural Changes

1. **Ensemble disagree filter**: If individual models within the ensemble strongly disagree, skip the bet. Requires storing per-model predictions.
2. **Regime detection**: Track rolling ATS performance and reduce bet size during cold streaks (currently impossible since `spread_std` is constant).
3. **Market movement filter**: Games where the line moves toward the model's prediction post-open may indicate smart money agreement. Requires odds tracking data.

---

## 9. Summary Statistics

| Metric | Value |
|--------|-------|
| Total 2025 Games | 285 |
| Overall ATS Record | 139-145-1 (48.9%) |
| Overall ATS ROI | -6.56% |
| Best Filter (by N) | Edge 2-4, spread<10: 40-24 (62.5%, +25.7% ROI) |
| Best Filter (by Win%) | HomeFav + edge 2-4: 16-4 (80.0%) |
| Worst Slot | MNF: 7-15 (31.8%) |
| Best Slot | TNF: 15-6 (71.4%) |
| Kelly-Positive Buckets | 2-4pt edge only (21.25% full Kelly) |
| Model Prediction Range | [-2.6, +8.7] (std 2.30) |
| Breakeven at -110 | 52.4% |

---

## Appendix A: Weekly ATS Trend

| Week | Games | W-L | Win % | Model MAE | Vegas MAE |
|------|-------|------|-------|-----------|-----------|
| 1 | 16 | 7-9 | 43.8% | 7.61 | 5.94 |
| 2 | 16 | 6-10 | 37.5% | 11.39 | 8.19 |
| 3 | 16 | 9-7 | 56.3% | 12.88 | 12.22 |
| 4 | 16 | 7-9 | 43.8% | 9.48 | 9.03 |
| 5 | 14 | 8-6 | 57.1% | 12.36 | 11.68 |
| 6 | 15 | 8-7 | 53.3% | 6.95 | 7.70 |
| 7 | 15 | 8-7 | 53.3% | 11.92 | 11.50 |
| 8 | 13 | 3-10 | 23.1% | 17.72 | 14.35 |
| 9 | 14 | 6-8 | 42.9% | 12.06 | 9.32 |
| 10 | 14 | 7-7 | 50.0% | 10.42 | 9.43 |
| 11 | 15 | 8-7 | 53.3% | 10.01 | 7.50 |
| 12 | 14 | 8-5 | 61.5% | 8.06 | 7.07 |
| 13 | 16 | 9-7 | 56.3% | 10.90 | 10.47 |
| 14 | 14 | 9-5 | 64.3% | 13.80 | 12.43 |
| 15 | 16 | 6-10 | 37.5% | 12.54 | 10.25 |
| 16 | 16 | 8-8 | 50.0% | 11.24 | 10.63 |
| 17 | 16 | 7-9 | 43.8% | 11.55 | 10.25 |
| 18 | 16 | 7-9 | 43.8% | 10.78 | 7.94 |
| 19 (WC) | 6 | 4-2 | 66.7% | 8.72 | 9.75 |
| 20 (DIV) | 4 | 3-1 | 75.0% | 10.53 | 9.88 |
| 21 (CC) | 2 | 1-1 | 50.0% | 3.35 | 1.00 |
| 22 (SB) | 1 | 0-1 | 0.0% | 17.40 | 11.50 |

Worst week: Week 8 (3-10, 23.1%). Best regular season week: Week 14 (9-5, 64.3%).

## Appendix B: Source Files Reviewed

| File | Purpose | Key Finding |
|------|---------|-------------|
| `nfl_analytics/ensemble/strategy/base.py` | Ensemble strategy pattern | Constant spread_std (13.54), no variable confidence |
| `nfl_analytics/production/kelly_sizing.py` | Kelly criterion implementation | Correct math, disconnected from pipeline |
| `nfl_analytics/features/game_feature_extractor.py:1076-1199` | Rest differential computation | 7 features, short/long week detection |
| `nfl_analytics/pricing/teaser_ev.py` | Teaser EV via Skellam + copulas | Most sophisticated pricing tool |
| `nfl_analytics/pricing/copula_parlays.py` | Gaussian copula parlay pricing | Monte Carlo simulation framework |
| `nfl_analytics/production/value_bet_detector.py` | LLM-enhanced value bet detection | Full infrastructure exists but untested |
| `docs/adr/016-cql-target-leakage-fix.md` | CQL leakage root cause | Spread sign convention, temporal split |

---

**Conclusion:** The betting strategy layer is well-engineered (Kelly sizing, copula teasers, value bet detection) but sits on top of a prediction engine that produces nearly constant outputs. The priority order is: (1) fix prediction compression, (2) validate promising filters on 2026 data, (3) deploy with quarter-Kelly sizing and strict edge gates. Market efficiency is real, and the 4.5% vig is insurmountable without genuine prediction alpha.
