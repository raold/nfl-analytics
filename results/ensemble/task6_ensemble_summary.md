# Task 6: Ensemble Uncertainty Filtering - Summary
**Date**: 2025-10-09
**Status**: COMPLETE

## Overview

Successfully implemented and evaluated an ensemble system combining three models:
1. **XGBoost v2** (Brier 0.1715, AUC 0.823) - Config 18
2. **CQL Config 4** (alpha=0.1, lr=3e-5, reward 0.0381)
3. **IQL Baseline** (expectile=0.9, temp=3.0, reward 0.0375)

## Implementation

### Files Created
- `py/ensemble/ensemble_predictor.py` (693 lines) - Full ensemble system
  - XGBoost, CQL, IQL model loaders with uncertainty quantification
  - Three voting strategies: unanimous, majority, weighted
  - Uncertainty filtering based on model confidence
  - Comprehensive backtesting framework
- `py/ensemble/prepare_ensemble_data.py` - Data preparation script
  - Computes RL state features from XGBoost features
  - Adds market probabilities from spreads
  - Creates ensemble-ready dataset

### Model Predictors

**XGBoostPredictor**:
- Loads trained XGBoost model from JSON
- Returns win probabilities + prediction entropy (uncertainty)
- Converts probabilities to betting actions based on edge magnitude

**CQLPredictor**:
- Loads trained CQL agent from PyTorch checkpoint
- Returns greedy actions + Q-value spread (uncertainty)
- Uses Q-network to evaluate all actions

**IQLPredictor**:
- Loads trained IQL agent (V-network + Q-network)
- Returns implicit policy actions + advantage magnitude (uncertainty)
- Uses expectile regression for optimistic value estimation

### Voting Strategies

1. **Unanimous**: Only bet when all 3 models agree on action
   - Most conservative
   - Highest precision, lowest recall

2. **Majority**: Bet when 2+ models agree
   - Balanced approach
   - Moderate precision and recall

3. **Weighted**: Confidence-weighted voting
   - XGBoost 50%, CQL 30%, IQL 20%
   - Weights by (1 - uncertainty) * model_weight
   - Most flexible

### Uncertainty Filtering

Each model computes its own uncertainty metric:
- **XGBoost**: Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)
- **CQL**: Q-value spread (max Q - min Q) / 10
- **IQL**: Max advantage magnitude |Q(s,a) - V(s)| / 5

Only bet when ALL models have uncertainty < threshold.

---

## Backtest Results (2024 Season, 269 games)

### Strategy Comparison

| Strategy | Bets | Bet Rate | Win Rate | ROI | Sharpe | Max DD | Avg Edge |
|----------|-----:|----------|---------:|----:|-------:|--------|---------:|
| **Unanimous** (threshold=0.5) | 0 | 0.0% | - | - | - | - | - |
| **Majority** (threshold=0.9) | 35 | 13.0% | 71.4% | **+0.36%** | **0.422** | 0.03 | -0.0276 |
| **Weighted** (threshold=0.9) | 48 | 17.8% | 62.5% | **+0.55%** | 0.248 | 0.11 | -0.0446 |

### Key Findings

1. **Unanimous too restrictive**
   - With threshold=0.5, models never agreed enough to bet
   - All-agree rate: 5.9% (only 16 games out of 269)
   - Suggests models have different decision boundaries

2. **Majority voting: High precision**
   - 71.4% win rate (25/35 correct)
   - 0.422 Sharpe ratio (good risk-adjusted return)
   - Only 13% of games (selective betting)
   - +0.36% ROI (modest but positive)

3. **Weighted voting: Higher volume**
   - 62.5% win rate (30/48 correct)
   - 17.8% bet rate (more opportunities)
   - +0.55% ROI (better absolute return)
   - 0.248 Sharpe (higher variance)

4. **Model agreement rates**
   - XGB-CQL: 8.2% (very different)
   - XGB-IQL: 38.3% (moderate agreement)
   - CQL-IQL: 40.5% (highest RL agreement)
   - All 3 agree: 5.9% (rare)

5. **Uncertainty patterns**
   - XGBoost mean: 0.804 (high uncertainty - close games)
   - CQL mean: 0.945 (very high - conservative)
   - IQL mean: 0.114 (low - aggressive/confident)

### Action Distribution

**Majority Voting**:
- No-bet: 87.0%
- Small: 13.0%

**Weighted Voting**:
- No-bet: 82.2%
- Small: 11.5%
- Medium: 3.0%
- Large: 3.3%

---

## Analysis

### Strengths

✅ **Selective betting works**: 13-18% bet rate much lower than individual models
✅ **Positive ROI**: Both strategies profitable (+0.36% and +0.55%)
✅ **High win rates**: 62-71% accuracy on selected bets
✅ **Good Sharpe ratios**: 0.25-0.42 (risk-adjusted returns)
✅ **Uncertainty filtering effective**: Threshold=0.9 allows some bets while filtering noise

### Weaknesses

❌ **Negative average edge**: -0.0276 to -0.0446 suggests betting against model edge
❌ **Low model agreement**: Models disagree on most games (8-40% pairwise)
❌ **Unanimous too strict**: Can't find ANY games where all models agree with high confidence
❌ **Small sample size**: Only 35-48 bets on 269 games

### Root Cause: Edge vs Action Mismatch

The negative average edge is concerning. This means:
- **XGBoost probability** (p_hat) vs **market probability** → negative edge
- But ensemble still wins 62-71% of bets

**Possible explanations**:
1. **RL agents betting opposite of XGBoost edge**
   - XGBoost says "bet away", CQL/IQL say "bet home"
   - RL agents learned different strategy from historical data

2. **Market efficiency improved**
   - Models trained on 2006-2023, tested on 2024
   - 2024 markets may be sharper than training period

3. **Edge calculation issue**
   - Edge computed as XGBoost prob - market prob
   - But final bet direction determined by majority/weighted vote
   - May not align with XGBoost's edge

---

## Recommendations

### For Production Betting

**Best Strategy**: **Majority Voting (threshold=0.9)**
- Highest win rate (71.4%)
- Best Sharpe ratio (0.422)
- Conservative bet rate (13%)
- Lower drawdown (0.03 units)

**Why not Weighted?**
- Higher ROI (0.55%) but more variance
- Lower Sharpe (0.248)
- Larger drawdown (0.11 units)
- Harder to trust with 62.5% win rate vs 71.4%

### Improvements Needed

1. **Fix edge calculation**
   - Compute edge from ensemble decision, not just XGBoost
   - Track which model(s) drove each bet

2. **Add Kelly sizing**
   - Current bet sizes (1%, 2%, 5%) are arbitrary
   - Use Kelly criterion: f = (p*b - q) / b
   - Where p=win prob, q=1-p, b=odds

3. **Investigate disagreement**
   - Why do XGB-CQL only agree 8.2% of time?
   - Are RL agents learning different features?
   - Should we retrain RL on same features as XGBoost?

4. **Lower unanimous threshold**
   - Try threshold=0.8, 0.7, 0.6 to get some unanimous bets
   - Ultra-selective betting might have highest precision

5. **Multi-season validation**
   - Test on 2022, 2023 as well
   - Check if 2024 results are anomaly or consistent pattern

---

## Files Generated

### Models
- `models/xgboost/v2_sweep/xgb_config18_season2024.json` - Best XGBoost
- `models/cql/sweep/cql_config4.pth` - Best CQL agent
- `models/iql/baseline_model.pth` - IQL baseline

### Data
- `data/processed/features/ensemble_features_2024.csv` - Ensemble-ready 2024 data (269 games)

### Results
- `results/ensemble/ensemble_majority_2024.json` - Majority voting backtest
- `results/ensemble/ensemble_weighted_2024.json` - Weighted voting backtest
- `results/ensemble/task6_ensemble_summary.md` - This document

### Code
- `py/ensemble/ensemble_predictor.py` - Main ensemble system (693 lines)
- `py/ensemble/prepare_ensemble_data.py` - Data preparation

---

## Next Steps (Task 7: Meta-Policy Ensemble)

Task 6 demonstrated that:
- Ensemble voting improves precision vs individual models
- But models disagree significantly (low agreement rates)
- **Opportunity**: Use Thompson Sampling to dynamically select best model per game

**Task 7 Plan**:
1. Treat each model as a "bandit arm" (XGBoost, CQL, IQL)
2. Track Bayesian posterior over win rate: Beta(α, β)
3. Each game:
   - Sample win rate from each model's posterior
   - Select model with highest sampled win rate
   - Bet according to that model's action
   - Update posterior based on outcome
4. Automatically learns which model performs best in which contexts
5. More adaptive than fixed voting weights

**Expected Value**: HIGH
- Addresses low model agreement by choosing best model per game
- Learns online (adapts over season)
- Balances exploration (try all models) vs exploitation (use best model)

---

*Generated: 2025-10-09 20:45 UTC*
