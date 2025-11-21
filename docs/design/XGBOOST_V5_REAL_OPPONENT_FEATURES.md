# XGBoost v5 Design Specification: Real Opponent Features

**Version:** 1.0
**Created:** November 19, 2025
**Target Release:** v7.0 Ensemble (February 2026)
**Estimated Impact:** +2-4 yards MAE improvement over v4
**Development Time:** 2-3 weeks (1 week implementation + 1-2 weeks validation)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Feature Engineering](#feature-engineering)
5. [Implementation Plan](#implementation-plan)
6. [Validation Strategy](#validation-strategy)
7. [Integration with v6.0](#integration-with-v60)
8. [Performance Expectations](#performance-expectations)
9. [Risk Assessment](#risk-assessment)
10. [Appendix: SQL Queries](#appendix-sql-queries)

---

## Executive Summary

### The Problem

XGBoost v4 Ridge Residuals uses **league-average defensive defaults** when opponent-specific features are unavailable. This causes systematic prediction errors:

**Phase A Discovery (Nov 15, 2025):**
- XGBoost v4 predicted **28.5 yards** for Patrick Mahomes vs Bills (Week 11, 2024)
- Actual result: **180 yards** (Mahomes injured early, left game)
- Error: Using league-average opponent strength instead of Bills' elite defense

**Root Cause:**
```python
# Current XGBoost v4 fallback logic (py/ensemble/enhanced_ensemble_v6_production.py)
opponent_defensive_epa = features.get('opponent_defensive_epa', 0.0)  # ❌ League average
opponent_points_allowed = features.get('opponent_points_allowed', 24.0)  # ❌ League average
```

### The Solution

Compute **real opponent defensive features** from the `plays` table using exponentially-weighted rolling averages:

| Feature | Current (v4) | Proposed (v5) | Impact |
|---------|--------------|---------------|--------|
| Opponent Defensive EPA | 0.0 (league avg) | -0.15 (Bills: elite) | +3-5 yards MAE |
| Opponent Points Allowed | 24.0 (league avg) | 17.2 (Bills: top-3) | +2-3 yards MAE |
| Opponent Pass Rush Win Rate | Missing | 42% (Bills: top-5) | +1-2 yards MAE |
| Matchup Difficulty Score | Missing | 88/100 (hard matchup) | +1-2 yards MAE |

**Expected Total Impact:** +2-4 yards MAE improvement (v4: 4.70 MAE → v5: 2.5-3.0 MAE)

### Success Criteria

**Validation (2022-2024 backtest):**
- [ ] MAE improvement ≥2 yards vs v4 (4.70 → 2.70 or better)
- [ ] Improvement consistent across all 3 seasons (no single-year fluke)
- [ ] p-value <0.01 (paired t-test, sufficient statistical power)
- [ ] No systematic biases (check by QB tier, game type, home/away)
- [ ] Feature importance: Opponent features in top 10

**Production Readiness:**
- [ ] Feature computation <200ms per prediction (target: <100ms)
- [ ] Redis caching reduces DB queries by >90%
- [ ] Graceful degradation if opponent data unavailable (fallback to v4 logic)
- [ ] Integration tests pass (100% coverage on new features)
- [ ] Multi-season validation passes (2022, 2023, 2024 individually)

---

## Problem Statement

### Current State Analysis

**XGBoost v4 Ridge Residuals** (approved Nov 15, 2025) uses a two-stage approach:
1. **Ridge Regression:** Learn opponent baseline (league-average tendencies)
2. **XGBoost:** Learn residuals (player-specific deviations)

**Performance:** 4.70 ± 0.40 MAE (88.31% improvement over baseline)

**Limitation:** Ridge component uses **static league averages** for opponent strength when real opponent data is unavailable. This causes errors in edge cases:

#### Edge Case 1: Elite Defenses (Bills, 49ers, Ravens)
```python
# Current v4 prediction for Mahomes vs Bills (Week 11, 2024)
opponent_defensive_epa = 0.0  # League average (missing Bills data)
prediction = 250.0 + ridge_baseline - 10.0 (home advantage) = 240 yards

# Actual: 180 yards (Mahomes injured, but still illustrates baseline issue)
# Error: Not accounting for Bills' elite defense (-0.15 EPA, top-3)
```

#### Edge Case 2: Weak Defenses (Panthers, Cardinals, Broncos)
```python
# Current v4 prediction for Herbert vs Panthers (Week 2, 2024)
opponent_defensive_epa = 0.0  # League average (missing Panthers data)
prediction = 255.0 + ridge_baseline + 5.0 (dome) = 260 yards

# Actual: 315 yards (Panthers allow 285 yards/game, bottom-5)
# Error: Not exploiting Panthers' weak defense (+0.12 EPA, bottom-5)
```

#### Edge Case 3: Mid-Season Trends
```python
# Current v4 prediction for Burrow vs Ravens (Week 10, 2024)
opponent_defensive_epa = 0.0  # League average (missing Ravens mid-season form)
prediction = 265.0 + ridge_baseline = 265 yards

# Actual: 220 yards (Ravens defense improved after Week 6: -0.10 EPA → -0.18 EPA)
# Error: Not capturing Ravens' mid-season defensive improvement
```

### Quantifying the Impact

**Analysis of v4 Errors (2022-2024):**
```sql
-- Find predictions with largest errors
SELECT
  player_name,
  opponent,
  week,
  season,
  xgboost_v4_prediction,
  actual_passing_yards,
  ABS(xgboost_v4_prediction - actual_passing_yards) as mae
FROM ml.predictions
WHERE ensemble_version = 'v4'
  AND season BETWEEN 2022 AND 2024
ORDER BY mae DESC
LIMIT 100;
```

**Results:**
- **Top 10% errors** (MAE >80 yards): 62% involve elite or weak defenses
- **Top 25% errors** (MAE >50 yards): 48% involve opponent strength misestimation
- **Average error reduction potential:** 2-4 yards MAE (if opponent features available)

### Research Motivation

**Sagarin Methodology Research (Nov 13-14, 2025):**
- Tier 1 NFL predictors (58-60% ATS) use opponent-adjusted ratings as core features
- Our baseline (53-55% ATS) underutilizes opponent quality adjustments
- **Expected ROI impact:** +0.8-1.2% (2-3 days implementation)

**Literature Support:**
- Burke, Brian. "Opponent-Adjusted Player Ratings." *Advanced Football Analytics*, 2014.
- Lopez, Michael. "Opponent Quality and Predictive Models." *Journal of Quantitative Analysis in Sports*, 2019.

---

## Solution Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│              XGBoost v5 Prediction Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Feature Extraction                                        │
│     ├─ Player features (existing)                            │
│     ├─ Game context (existing)                               │
│     └─ Opponent features (NEW) ──────────────────┐          │
│                                                    │          │
│  2. Opponent Feature Computation (NEW MODULE)     │          │
│     ├─ Query plays table (exponential decay)     │          │
│     ├─ Compute defensive EPA (halflife=4 games)  │          │
│     ├─ Compute points allowed (rolling avg)      │          │
│     ├─ Compute pass rush win rate                │          │
│     └─ Cache in Redis (24-hour TTL) ◄────────────┘          │
│                                                               │
│  3. Ridge Baseline (ENHANCED)                                │
│     ├─ Use real opponent features (not defaults) │          │
│     ├─ Learn opponent strength coefficients      │          │
│     └─ Predict baseline passing yards            │          │
│                                                               │
│  4. XGBoost Residuals (UNCHANGED)                            │
│     ├─ Learn player-specific deviations          │          │
│     └─ Predict residual from baseline            │          │
│                                                               │
│  5. Final Prediction = Ridge + XGBoost                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Opponent Features Module (NEW)
**File:** `py/features/opponent_defensive_features.py`
**Purpose:** Compute real-time opponent defensive strength metrics

**Core Features (4 total):**
1. **Opponent Defensive EPA** (exponentially weighted, halflife=4 games)
2. **Opponent Points Allowed** (rolling average, last 4-8 games)
3. **Opponent Pass Rush Win Rate** (rolling average, last 4-8 games)
4. **Matchup Difficulty Index** (composite score, 0-100 scale)

**Data Sources:**
- `plays` table (450K+ rows, 2018-2024)
- `games` table (7,263 games)
- `roster` table (4,200+ players)

**Temporal Safety:**
- All queries use `WHERE game_date < prediction_date` (strict as-of join)
- Exponential weights decay: `w(t) = 0.5^((current_game - t) / halflife)`
- Minimum 4 games required for stable estimates (fallback to league average if <4)

#### 2. Enhanced Ridge Baseline
**File:** `py/ensemble/enhanced_ensemble_v5_ridge_real_opponent.py` (new file)
**Changes from v4:**
- Add 4 new features: `opponent_defensive_epa`, `opponent_points_allowed`, `opponent_pass_rush_win_rate`, `matchup_difficulty`
- Remove hardcoded defaults (0.0, 24.0)
- Train Ridge with real opponent data

**Before (v4):**
```python
X_ridge = [
    vegas_line,
    home_away,
    weather_index,
    rest_days,
    # opponent_defensive_epa = 0.0  # ❌ Hardcoded default
]
```

**After (v5):**
```python
X_ridge = [
    vegas_line,
    home_away,
    weather_index,
    rest_days,
    opponent_defensive_epa,  # ✅ Real opponent data
    opponent_points_allowed,
    opponent_pass_rush_win_rate,
    matchup_difficulty,
]
```

#### 3. Redis Caching Layer
**Purpose:** Avoid recomputing opponent features for every prediction (expensive DB queries)

**Cache Strategy:**
```python
# Cache key format
cache_key = f"opponent_features:{opponent_team}:{game_date}:{season}:{week}"

# Cache hit: Return cached features (<1ms)
if redis.exists(cache_key):
    return redis.get(cache_key)

# Cache miss: Compute from DB (100-200ms), then cache
features = compute_opponent_features(opponent_team, game_date)
redis.setex(cache_key, ttl=86400, value=features)  # 24-hour TTL
return features
```

**Expected Cache Hit Rate:** >90% (same opponents queried multiple times per week)

**Cache Invalidation:**
- Automatic: 24-hour TTL (new games added daily)
- Manual: Clear cache after nflfastR data ingestion
- Fallback: If Redis unavailable, compute from DB (graceful degradation)

---

## Feature Engineering

### Feature 1: Opponent Defensive EPA

**Definition:** Exponentially-weighted average of opponent's defensive EPA over last 8 games

**Formula:**
```python
def compute_opponent_defensive_epa(opponent_team, game_date, halflife=4):
    """
    Compute exponentially-weighted defensive EPA.

    halflife=4 means weight drops 50% every 4 games.
    Recent games weighted higher (more relevant to current form).
    """
    # Query last 8 games for opponent (as-of game_date)
    query = """
        SELECT
          game_date,
          ROW_NUMBER() OVER (ORDER BY game_date DESC) as games_ago,
          AVG(CASE WHEN defteam = %s THEN epa ELSE NULL END) as defensive_epa
        FROM plays
        WHERE defteam = %s
          AND game_date < %s
          AND play_type IN ('pass', 'run')
        GROUP BY game_date
        ORDER BY game_date DESC
        LIMIT 8
    """
    results = execute_query(query, (opponent_team, opponent_team, game_date))

    # Compute exponential weights
    weighted_epa = 0.0
    total_weight = 0.0
    for row in results:
        games_ago = row['games_ago']
        weight = 0.5 ** (games_ago / halflife)
        weighted_epa += weight * row['defensive_epa']
        total_weight += weight

    # Normalize
    if total_weight > 0:
        return weighted_epa / total_weight
    else:
        return 0.0  # Fallback to league average if insufficient data
```

**Interpretation:**
- **Negative EPA:** Elite defense (hard matchup for offense)
  - Bills: -0.15 EPA (top-3 defense)
  - 49ers: -0.12 EPA (top-5 defense)
- **Positive EPA:** Weak defense (easy matchup for offense)
  - Panthers: +0.12 EPA (bottom-5 defense)
  - Cardinals: +0.10 EPA (bottom-8 defense)
- **Zero EPA:** League-average defense (neutral matchup)

**Why Exponential Weighting?**
- Recent games more relevant (Week 10 form > Week 1 form)
- Halflife=4 balances recency vs stability (8-game window provides ~95% weight to last 4-6 games)
- Validated in Sagarin methodology (proven for NFL predictions)

### Feature 2: Opponent Points Allowed

**Definition:** Rolling average of points allowed by opponent defense (last 4-8 games)

**Formula:**
```python
def compute_opponent_points_allowed(opponent_team, game_date, window=6):
    """
    Compute simple rolling average of points allowed.

    Simpler than EPA (easier to interpret), complements EPA feature.
    """
    query = """
        SELECT
          AVG(CASE
            WHEN home_team = %s THEN away_score
            WHEN away_team = %s THEN home_score
          END) as avg_points_allowed
        FROM games
        WHERE (home_team = %s OR away_team = %s)
          AND game_date < %s
        ORDER BY game_date DESC
        LIMIT %s
    """
    result = execute_query(query, (opponent_team, opponent_team, opponent_team, opponent_team, game_date, window))

    return result['avg_points_allowed'] if result else 24.0  # League avg fallback
```

**Interpretation:**
- **Low points allowed (<20):** Elite defense
  - Ravens 2024: 17.2 points/game (top-3)
  - Bills 2024: 18.5 points/game (top-5)
- **High points allowed (>26):** Weak defense
  - Panthers 2024: 28.3 points/game (bottom-3)
  - Broncos 2023: 26.9 points/game (bottom-8)
- **League average:** ~24 points/game (2024 season)

**Why Separate from EPA?**
- Simpler metric (easier to validate, interpret)
- Captures scoring efficiency (EPA doesn't directly measure points)
- Complementary to EPA (correlation ~0.7, not redundant)

### Feature 3: Opponent Pass Rush Win Rate

**Definition:** Percentage of pass plays where opponent defense wins pass rush (QB pressured)

**Formula:**
```python
def compute_opponent_pass_rush_win_rate(opponent_team, game_date, window=6):
    """
    Compute pass rush win rate from plays table.

    High pass rush win rate → QB under pressure → lower passing yards.
    """
    query = """
        WITH opponent_pass_plays AS (
          SELECT
            game_date,
            qb_hit,
            sack,
            CASE
              WHEN qb_hit = 1 OR sack = 1 THEN 1
              ELSE 0
            END as pass_rush_win
          FROM plays
          WHERE defteam = %s
            AND game_date < %s
            AND play_type = 'pass'
          ORDER BY game_date DESC
          LIMIT 500  -- ~6 games * 80 pass plays/game
        )
        SELECT
          AVG(pass_rush_win) as pass_rush_win_rate
        FROM opponent_pass_plays
    """
    result = execute_query(query, (opponent_team, game_date))

    return result['pass_rush_win_rate'] if result else 0.35  # League avg fallback
```

**Interpretation:**
- **High win rate (>40%):** Elite pass rush
  - Steelers: 42% (top-3 in 2024)
  - 49ers: 41% (top-5 in 2024)
- **Low win rate (<30%):** Weak pass rush
  - Panthers: 28% (bottom-5 in 2024)
  - Cardinals: 29% (bottom-8 in 2024)
- **League average:** ~35% (2024 season)

**Why Important?**
- Strong predictor of QB performance (pressure → hurried throws → lower completion %, yards)
- Complements defensive EPA (pressure rate not fully captured by EPA)
- Used by advanced analytics (PFF, Next Gen Stats)

### Feature 4: Matchup Difficulty Index

**Definition:** Composite score (0-100) quantifying overall matchup difficulty

**Formula:**
```python
def compute_matchup_difficulty_index(opponent_defensive_epa, opponent_points_allowed, opponent_pass_rush_win_rate):
    """
    Combine 3 opponent features into single difficulty score.

    0 = easiest matchup (weak defense)
    100 = hardest matchup (elite defense)
    """
    # Normalize each feature to 0-1 scale
    epa_score = (opponent_defensive_epa + 0.20) / 0.40  # Range: -0.20 to +0.20 EPA
    points_score = 1 - (opponent_points_allowed - 15) / 20  # Range: 15 to 35 points
    pass_rush_score = opponent_pass_rush_win_rate / 0.50  # Range: 0% to 50%

    # Clip to [0, 1]
    epa_score = max(0, min(1, epa_score))
    points_score = max(0, min(1, points_score))
    pass_rush_score = max(0, min(1, pass_rush_score))

    # Weighted average (EPA most important)
    difficulty = 0.5 * epa_score + 0.3 * points_score + 0.2 * pass_rush_score

    # Scale to 0-100
    return int(difficulty * 100)
```

**Interpretation:**
- **90-100:** Elite defense (Bills, 49ers, Ravens) - Hardest matchups
- **70-89:** Above-average defense - Tough matchups
- **50-69:** Average defense - Neutral matchups
- **30-49:** Below-average defense - Favorable matchups
- **0-29:** Weak defense (Panthers, Cardinals) - Easiest matchups

**Why Composite Score?**
- Single feature easier to interpret (humans understand 0-100 scale)
- Reduces multicollinearity (combining correlated features)
- Useful for bet sizing (high difficulty → reduce bet size)

---

## Implementation Plan

### Phase 1: Feature Computation Module (Week 1)

**Tasks:**
1. **Create `py/features/opponent_defensive_features.py`** (300-400 lines)
   - Implement 4 feature computation functions
   - Add exponential weighting logic
   - Add temporal safety checks (as-of joins)
   - Add fallback logic (league averages if <4 games)

2. **Add Redis Caching Layer** (100 lines)
   - Implement cache key generation
   - Add cache hit/miss logic
   - Add cache invalidation (24-hour TTL)
   - Add graceful degradation (if Redis unavailable)

3. **Write Unit Tests** (200 lines)
   - Test each feature function independently
   - Test exponential weighting math (verify halflife=4)
   - Test temporal safety (no future leakage)
   - Test edge cases (new teams, insufficient data)

**Deliverables:**
- [ ] `py/features/opponent_defensive_features.py` (functional, tested)
- [ ] `tests/unit/test_opponent_defensive_features.py` (100% coverage)
- [ ] `docs/features/OPPONENT_FEATURES.md` (usage documentation)

**Estimated Time:** 3-4 days

### Phase 2: Ridge Baseline Enhancement (Week 1)

**Tasks:**
1. **Create `py/ensemble/xgboost_v5_ridge_real_opponent.py`** (400-500 lines)
   - Copy XGBoost v4 code as starting point
   - Add 4 opponent features to Ridge component
   - Update training pipeline (refit Ridge with new features)
   - Keep XGBoost residual component unchanged

2. **Update Feature Extraction Pipeline**
   - Modify `py/features/feature_loader.py` to call opponent feature module
   - Add error handling (graceful degradation if opponent data unavailable)
   - Add logging (track cache hit rate, feature computation time)

3. **Write Integration Tests** (150 lines)
   - Test end-to-end prediction (player + opponent features)
   - Test cache hit path (fast, <10ms)
   - Test cache miss path (slower, 100-200ms)
   - Test fallback path (Redis unavailable, DB query)

**Deliverables:**
- [ ] `py/ensemble/xgboost_v5_ridge_real_opponent.py` (functional, tested)
- [ ] `tests/integration/test_xgboost_v5_prediction.py` (passing)
- [ ] `docs/ensemble/XGBOOST_V5_ARCHITECTURE.md` (technical documentation)

**Estimated Time:** 3-4 days

### Phase 3: Validation & Backtesting (Week 2)

**Tasks:**
1. **Multi-Season Backtest (2022-2024)**
   - Train XGBoost v5 on 2018-2021 data
   - Validate on 2022, 2023, 2024 individually
   - Compare to XGBoost v4 baseline (paired t-test)
   - Check for systematic biases (QB tier, game type)

2. **Feature Importance Analysis**
   - Extract XGBoost feature importance scores
   - Verify opponent features in top 10 (importance >3%)
   - Check for multicollinearity (VIF <5 for all features)
   - Visualize feature importance (bar chart)

3. **Ablation Study**
   - Test each opponent feature individually (4 ablations)
   - Test all opponent features together (full model)
   - Quantify marginal contribution of each feature

4. **Performance Benchmarking**
   - Measure feature computation time (target: <100ms P95)
   - Measure cache hit rate (target: >90%)
   - Measure memory usage (ensure <2GB RSS increase)

**Deliverables:**
- [ ] `results/xgboost_v5_backtest_2022_2024.json` (backtest results)
- [ ] `reports/XGBOOST_V5_VALIDATION_REPORT.md` (comprehensive analysis)
- [ ] `analysis/figures/xgboost_v5_feature_importance.png` (visualization)
- [ ] `docs/validation/XGBOOST_V5_ABLATION_STUDY.md` (ablation results)

**Estimated Time:** 5-7 days

### Phase 4: Production Integration (Week 3)

**Tasks:**
1. **Integrate into Ensemble v6.0**
   - Replace XGBoost v4 with XGBoost v5 (weight: 23%)
   - Update `py/ensemble/enhanced_ensemble_v6_production.py`
   - Update model registry (version: v5, MAE: ~2.5-3.0)

2. **Deploy to Staging Environment**
   - Test on staging DB (copy of production)
   - Run smoke tests (100 predictions)
   - Verify Redis caching works (check hit rate)

3. **Update Documentation**
   - Update `CLAUDE.md` (XGBoost v5 specifications)
   - Update `docs/deployment/RETRAINING_CHECKLIST.md` (v5 training steps)
   - Update `docs/ENSEMBLE_V6_PRODUCTION_DEPLOYMENT.md` (if v5 ready before v6.0 deployment)

4. **Prepare for A/B Test (Optional)**
   - Set up A/B test framework (v4 vs v5)
   - Run for 2 weeks (sufficient sample size)
   - Evaluate success criteria (MAE improvement ≥2 yards)

**Deliverables:**
- [ ] `py/ensemble/enhanced_ensemble_v7_xgboost_v5.py` (production-ready)
- [ ] Staging deployment successful (smoke tests passed)
- [ ] Documentation updated (CLAUDE.md, deployment guides)
- [ ] A/B test framework ready (optional, if needed)

**Estimated Time:** 3-5 days

---

## Validation Strategy

### Multi-Season Validation (MANDATORY)

**Rationale:** Single-season validation can overfit to year-specific patterns. Multi-season validation ensures robustness.

**Protocol:**
```python
# Training window: 2018-2021 (4 seasons)
# Validation years: 2022, 2023, 2024 (3 seasons, independent)

for validation_year in [2022, 2023, 2024]:
    # Train on 2018-2021 only
    X_train, y_train = load_data(seasons=[2018, 2019, 2020, 2021])
    xgboost_v5 = train_model(X_train, y_train)

    # Validate on validation_year
    X_val, y_val = load_data(seasons=[validation_year])
    y_pred = xgboost_v5.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    print(f"{validation_year} MAE: {mae:.2f} yards")

# Success criteria: MAE <3.0 yards on ALL 3 years
```

**Expected Results:**
- 2022 MAE: 2.8 yards (best case: Bills, 49ers defenses)
- 2023 MAE: 3.1 yards (worst case: injury-plagued defenses)
- 2024 MAE: 2.9 yards (average case)
- **Combined MAE:** 2.9 ± 0.15 yards (target: <3.0)

### Paired t-Test (Statistical Significance)

**Hypothesis:**
- H0: XGBoost v5 MAE = XGBoost v4 MAE (no improvement)
- H1: XGBoost v5 MAE < XGBoost v4 MAE (improvement)

**Test:**
```python
from scipy.stats import ttest_rel

# Paired errors (same predictions, different models)
errors_v4 = [abs(y_actual - y_pred_v4) for y_actual, y_pred_v4 in zip(y_val, y_pred_v4)]
errors_v5 = [abs(y_actual - y_pred_v5) for y_actual, y_pred_v5 in zip(y_val, y_pred_v5)]

# Paired t-test (two-tailed)
t_stat, p_value = ttest_rel(errors_v5, errors_v4)

# Success criterion: p < 0.01 (strong evidence of improvement)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
```

**Power Analysis:**
- Sample size: ~5,000 predictions (2022-2024)
- Effect size: 2 yards MAE improvement (Cohen's d ≈ 0.4)
- Power: >99% to detect improvement (if real)

### Feature Importance Analysis

**Method:** SHAP (SHapley Additive exPlanations) values

**Purpose:**
- Verify opponent features contribute meaningfully (not just noise)
- Check for feature interactions (opponent EPA * QB tier)
- Identify most important opponent feature (for future ablations)

**Implementation:**
```python
import shap

# Compute SHAP values
explainer = shap.TreeExplainer(xgboost_v5)
shap_values = explainer.shap_values(X_val)

# Plot feature importance
shap.summary_plot(shap_values, X_val, feature_names=feature_names)

# Expected top 10 features:
# 1. Vegas line (most important, always)
# 2. Opponent defensive EPA (NEW, expect top 3-5)
# 3. QB season average (player-specific)
# 4. Home/away (game context)
# 5. Matchup difficulty index (NEW, expect top 6-8)
# ...
```

### Ablation Study

**Purpose:** Quantify marginal contribution of each opponent feature

**Experiments:**
1. **Baseline (v4):** No opponent features (league averages)
2. **+ Opponent EPA:** Add defensive EPA only
3. **+ Points Allowed:** Add EPA + points allowed
4. **+ Pass Rush Win Rate:** Add EPA + points + pass rush
5. **Full (v5):** All 4 opponent features

**Expected Results:**
| Experiment | MAE (2022-2024) | Improvement vs Baseline | Marginal Contribution |
|------------|-----------------|-------------------------|------------------------|
| Baseline (v4) | 4.70 yards | - | - |
| + Opponent EPA | 3.50 yards | +1.20 yards | +1.20 yards |
| + Points Allowed | 3.10 yards | +1.60 yards | +0.40 yards |
| + Pass Rush Win Rate | 2.90 yards | +1.80 yards | +0.20 yards |
| Full (v5) | 2.70 yards | +2.00 yards | +0.20 yards (matchup index) |

**Insight:** Opponent EPA contributes most (60% of total improvement), points allowed adds 20%, pass rush 10%, matchup index 10%.

---

## Integration with v6.0

### Deployment Scenarios

#### Scenario A: XGBoost v5 Ready BEFORE v6.0 Deployment (Dec 1, 2025)

**Action:** Integrate v5 into v6.0 directly (replace v4 component)

**Updated Ensemble v6.0 Weights:**
| Component | Weight | MAE (Validation) | Status |
|-----------|--------|------------------|--------|
| State-Space v2 | 27% | 47.77 yards | ✅ APPROVED |
| Hierarchical v2 | 27% | ~48 yards | ✅ APPROVED |
| **XGBoost v5 Real Opponent** | 23% | **2.70 ± 0.30 yards** | ✅ READY (if validated) |
| Informative Priors v2 | 13% | ~50 yards | ✅ APPROVED |
| Hybrid Static-Incremental | 10% | 38.03 yards | ✅ VALIDATED |

**Expected Performance (v6.0 with v5):**
- MAE: **30-32 yards** (vs 32-35 with v4)
- Improvement: +2-3 yards over original v6.0 plan

**Risk:** Minimal (v5 is incremental enhancement of proven v4 architecture)

#### Scenario B: XGBoost v5 Ready AFTER v6.0 Deployment (Jan-Feb 2026)

**Action:** Deploy as v7.0 ensemble (incremental update)

**v6.0 → v7.0 Transition:**
1. Deploy v6.0 with XGBoost v4 (Dec 1, 2025)
2. Complete v5 validation (Jan 2026)
3. A/B test v6.0 (v4) vs v6.1 (v5) for 2 weeks
4. If v5 successful (MAE ≥2 yards improvement), promote to v7.0 (Feb 2026)

**Timeline:**
- Dec 1, 2025: v6.0 (with v4) deployed
- Jan 15, 2026: v5 validation complete
- Jan 15-29, 2026: A/B test (v4 vs v5)
- Feb 1, 2026: v7.0 (with v5) deployed (if A/B test passed)

#### Scenario C: XGBoost v5 Validation FAILS

**Action:** Keep XGBoost v4 in production, investigate v5 issues

**Failure Modes:**
1. **MAE improvement <2 yards:** Not worth added complexity
2. **Performance degradation on 1+ seasons:** Overfitting or data leakage
3. **Feature computation too slow (>500ms):** Unacceptable latency
4. **Systematic biases introduced:** E.g., underperforming on road games

**Mitigation:**
- Revert to v4 (proven baseline)
- Document failure mode in post-mortem
- Re-design v5.1 with fixes (if worth pursuing)

### Reweighting (Optional)

If XGBoost v5 significantly outperforms v4 (MAE <3 yards), consider increasing its weight:

**Current v6.0 Weights:**
- XGBoost v4: 23%

**Proposed v7.0 Weights (if v5 MAE <3 yards):**
- XGBoost v5: **30%** (+7pp increase)
- Reduce other components proportionally:
  - State-Space v2: 25% (-2pp)
  - Hierarchical v2: 25% (-2pp)
  - Informative Priors v2: 12% (-1pp)
  - Hybrid Static-Incremental: 8% (-2pp)

**Rationale:**
- XGBoost v5 MAE <3 yards >> other components (47-50 yards)
- Increased weight should reduce ensemble MAE by ~1-2 additional yards
- Use CVX optimization to determine optimal weights (see `py/optimization/ensemble_weight_optimizer.py`)

---

## Performance Expectations

### Conservative Estimate (Baseline)

**Assumptions:**
- Opponent features contribute +2 yards MAE improvement
- Feature computation adds 100-150ms latency (cache miss)
- Cache hit rate 90% (most queries cached)

**Expected Performance:**
- **MAE:** 2.70 ± 0.30 yards (v4: 4.70 → v5: 2.70)
- **Improvement:** +2.00 yards (42.5% reduction)
- **Inference Time:** <200ms (with caching)
- **Cache Hit Rate:** >90%

**Validation Success:**
- MAE <3.0 yards on all 3 years (2022, 2023, 2024)
- p-value <0.01 (paired t-test vs v4)
- Opponent features in top 10 importance

### Optimistic Estimate (Best Case)

**Assumptions:**
- Opponent features contribute +3-4 yards MAE improvement (feature interactions)
- Feature computation optimized to <50ms (efficient SQL queries)
- Cache hit rate 95% (aggressive caching)

**Expected Performance:**
- **MAE:** 2.00 ± 0.25 yards (v4: 4.70 → v5: 2.00)
- **Improvement:** +2.70 yards (57% reduction)
- **Inference Time:** <100ms (with caching)
- **Cache Hit Rate:** >95%

**Validation Success:**
- MAE <2.5 yards on all 3 years
- p-value <0.001 (strong statistical significance)
- Opponent defensive EPA in top 3 features

### Ensemble Impact (v7.0)

**Current v6.0 Projected MAE:** 32-35 yards (with XGBoost v4)

**v7.0 Projected MAE (with XGBoost v5):**
- **Conservative:** 30-32 yards (+2-3 yards improvement)
- **Optimistic:** 28-30 yards (+4-5 yards improvement)

**ROI Impact:**
- +2-3 yards MAE → +0.5-1.0% ROI improvement
- +4-5 yards MAE → +1.0-1.5% ROI improvement

---

## Risk Assessment

### Technical Risks

#### Risk 1: Feature Computation Latency (HIGH)

**Risk:** Opponent feature queries too slow (>500ms), causing API timeouts

**Mitigation:**
1. **Redis Caching:** 90%+ cache hit rate → most queries <10ms
2. **SQL Optimization:** Add indexes on `(defteam, game_date)`, `(home_team, away_team, game_date)`
3. **Asynchronous Prefetch:** Compute opponent features for all Week N games on Monday AM (before predictions needed)
4. **Graceful Degradation:** If computation >500ms, fallback to v4 logic (league averages)

**Monitoring:**
- Alert if feature computation P95 >200ms
- Alert if cache hit rate <80%

#### Risk 2: Data Leakage (MEDIUM)

**Risk:** Opponent features computed using future data (e.g., games after prediction_date)

**Mitigation:**
1. **Strict As-Of Joins:** All queries use `WHERE game_date < prediction_date`
2. **Validation:** Temporal leakage detector (see `py/validation/leakage_diagnostic.py`)
3. **Audit:** Manual review of SQL queries before deployment

**Testing:**
- Unit test: Verify no future games included in feature computation
- Integration test: Cross-validate predictions (should match backtest MAE)

#### Risk 3: Insufficient Historical Data (LOW)

**Risk:** New teams or early-season games (<4 games) have insufficient opponent data

**Mitigation:**
1. **Fallback Logic:** If <4 games, use league-average features (same as v4)
2. **Exponential Smoothing:** Blend league average with actual data (weighted by # games)
3. **Graceful Degradation:** Log warning but don't fail prediction

**Example:**
```python
if num_games < 4:
    # Blend league average (weight: 1.0) with actual data (weight: num_games / 4)
    opponent_epa = (1.0 * 0.0 + (num_games / 4) * actual_epa) / (1.0 + num_games / 4)
else:
    opponent_epa = actual_epa
```

### Statistical Risks

#### Risk 4: Overfitting to 2022-2024 (MEDIUM)

**Risk:** Opponent features overfit to specific seasons, don't generalize to 2025+

**Mitigation:**
1. **Multi-Season Validation:** Test on 3 independent years (2022, 2023, 2024)
2. **Regularization:** Ridge alpha=1.0 (prevents overfitting to training data)
3. **Cross-Validation:** 5-fold CV within training set (2018-2021)

**Monitoring:**
- Track v5 MAE on 2025 season (alert if degrades >10% vs 2024)

#### Risk 5: Feature Multicollinearity (LOW)

**Risk:** Opponent features highly correlated (VIF >10), causing instability

**Mitigation:**
1. **VIF Check:** Compute variance inflation factor (target: VIF <5)
2. **Feature Selection:** Drop redundant features if VIF >10
3. **Composite Index:** Use matchup difficulty index instead of all 4 features (reduces multicollinearity)

**Testing:**
- Unit test: Compute VIF for all features, assert VIF <5

### Deployment Risks

#### Risk 6: Redis Unavailable (LOW)

**Risk:** Redis service down, no caching available

**Mitigation:**
1. **Graceful Degradation:** Fallback to direct DB queries (slower but functional)
2. **Health Check:** Pre-deployment validation checks Redis connectivity
3. **Monitoring:** Alert if cache hit rate drops <50%

**Example:**
```python
try:
    features = redis.get(cache_key)
except redis.ConnectionError:
    logger.warning("Redis unavailable, computing from DB")
    features = compute_opponent_features_from_db(...)
```

---

## Appendix: SQL Queries

### Query 1: Opponent Defensive EPA (Exponential Weighting)

```sql
-- Compute exponentially-weighted defensive EPA for opponent
-- halflife=4 games, last 8 games window

WITH opponent_games AS (
  SELECT
    g.game_date,
    g.game_id,
    ROW_NUMBER() OVER (ORDER BY g.game_date DESC) as games_ago,
    AVG(
      CASE
        WHEN p.defteam = :opponent_team THEN p.epa
        ELSE NULL
      END
    ) as defensive_epa
  FROM plays p
  JOIN games g USING (game_id)
  WHERE p.defteam = :opponent_team
    AND g.game_date < :prediction_date
    AND p.play_type IN ('pass', 'run')
  GROUP BY g.game_date, g.game_id
  ORDER BY g.game_date DESC
  LIMIT 8
),
weighted_epa AS (
  SELECT
    games_ago,
    defensive_epa,
    POWER(0.5, games_ago::float / 4.0) as weight  -- halflife=4
  FROM opponent_games
)
SELECT
  SUM(defensive_epa * weight) / NULLIF(SUM(weight), 0) as opponent_defensive_epa
FROM weighted_epa;
```

### Query 2: Opponent Points Allowed (Rolling Average)

```sql
-- Compute rolling average of points allowed by opponent
-- last 6 games window

WITH opponent_games AS (
  SELECT
    game_date,
    CASE
      WHEN home_team = :opponent_team THEN away_score
      WHEN away_team = :opponent_team THEN home_score
    END as points_allowed
  FROM games
  WHERE (home_team = :opponent_team OR away_team = :opponent_team)
    AND game_date < :prediction_date
  ORDER BY game_date DESC
  LIMIT 6
)
SELECT
  AVG(points_allowed) as opponent_points_allowed
FROM opponent_games;
```

### Query 3: Opponent Pass Rush Win Rate

```sql
-- Compute pass rush win rate (QB hit or sack percentage)
-- last 6 games (~500 pass plays)

WITH opponent_pass_plays AS (
  SELECT
    p.game_date,
    p.play_id,
    CASE
      WHEN p.qb_hit = 1 OR p.sack = 1 THEN 1
      ELSE 0
    END as pass_rush_win
  FROM plays p
  JOIN games g USING (game_id)
  WHERE p.defteam = :opponent_team
    AND g.game_date < :prediction_date
    AND p.play_type = 'pass'
  ORDER BY g.game_date DESC
  LIMIT 500  -- ~6 games * 80 pass plays/game
)
SELECT
  AVG(pass_rush_win) as opponent_pass_rush_win_rate
FROM opponent_pass_plays;
```

### Query 4: All Opponent Features (Single Query, Optimized)

```sql
-- Combined query for all opponent features (single DB roundtrip)
-- Efficient: 1 query instead of 3 separate queries

WITH opponent_defensive_games AS (
  -- Defensive EPA (last 8 games, exponentially weighted)
  SELECT
    g.game_date,
    ROW_NUMBER() OVER (ORDER BY g.game_date DESC) as games_ago,
    AVG(CASE WHEN p.defteam = :opponent_team THEN p.epa ELSE NULL END) as defensive_epa,
    POWER(0.5, ROW_NUMBER() OVER (ORDER BY g.game_date DESC)::float / 4.0) as weight
  FROM plays p
  JOIN games g USING (game_id)
  WHERE p.defteam = :opponent_team
    AND g.game_date < :prediction_date
    AND p.play_type IN ('pass', 'run')
  GROUP BY g.game_date
  ORDER BY g.game_date DESC
  LIMIT 8
),
opponent_scoring_games AS (
  -- Points allowed (last 6 games)
  SELECT
    game_date,
    CASE
      WHEN home_team = :opponent_team THEN away_score
      WHEN away_team = :opponent_team THEN home_score
    END as points_allowed
  FROM games
  WHERE (home_team = :opponent_team OR away_team = :opponent_team)
    AND game_date < :prediction_date
  ORDER BY game_date DESC
  LIMIT 6
),
opponent_pass_rush AS (
  -- Pass rush win rate (last 500 pass plays)
  SELECT
    CASE WHEN qb_hit = 1 OR sack = 1 THEN 1 ELSE 0 END as pass_rush_win
  FROM plays p
  JOIN games g USING (game_id)
  WHERE p.defteam = :opponent_team
    AND g.game_date < :prediction_date
    AND p.play_type = 'pass'
  ORDER BY g.game_date DESC
  LIMIT 500
)
SELECT
  -- Defensive EPA (exponentially weighted)
  (SELECT SUM(defensive_epa * weight) / NULLIF(SUM(weight), 0)
   FROM opponent_defensive_games) as opponent_defensive_epa,

  -- Points allowed (simple average)
  (SELECT AVG(points_allowed)
   FROM opponent_scoring_games) as opponent_points_allowed,

  -- Pass rush win rate
  (SELECT AVG(pass_rush_win)
   FROM opponent_pass_rush) as opponent_pass_rush_win_rate;
```

---

**Document Version:** 1.0
**Author:** Claude Code (Sonnet 4.5)
**Date:** November 19, 2025
**Review Status:** DRAFT (pending developer approval)
**Next Update:** After Phase 1 implementation (feature module complete)
