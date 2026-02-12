# Adaptive Learning System - Complete Guide

**Created**: November 22, 2025
**Status**: ✅ Infrastructure Complete, Ready for Data Accumulation
**Author**: Claude Code (Sonnet 4.5)

---

## Executive Summary

The Adaptive Learning System enables the NFL Analytics ensemble to **learn and improve week-by-week** by automatically re-optimizing component weights based on recent performance. As predictions accumulate and games are played, the system:

1. **Learns** which models perform best in recent weeks
2. **Adapts** ensemble weights using exponentially-weighted least squares (WLS)
3. **Tracks** weight evolution and feature importance over time
4. **Identifies** which features help, hurt, or don't matter

**Current State (Nov 22, 2025)**:
- ✅ Database tables created (`ml.ensemble_weight_history`, `ml.feature_importance_history`)
- ✅ Weekly re-optimization pipeline built (`scripts/weekly_ensemble_reoptimization.py`)
- ✅ Baseline v6.0 weights initialized in database
- ⏳ Waiting for historical data (need 4-6 weeks of predictions + actuals)

**Expected Impact**:
- **+2-5% MAE improvement** as system learns optimal weight combinations
- **Faster adaptation** to model drift and seasonal changes
- **Feature drift detection** to identify changing predictive patterns

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Database Schema](#database-schema)
3. [Weekly Re-Optimization Pipeline](#weekly-re-optimization-pipeline)
4. [Usage Guide](#usage-guide)
5. [Monitoring & Visualization](#monitoring--visualization)
6. [Roadmap & Future Enhancements](#roadmap--future-enhancements)
7. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE LEARNING SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  Weekly Re-Optimization Pipeline         │
        │  (scripts/weekly_ensemble_reoptimization.py) │
        └──────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ PostgreSQL   │    │ WLS Optimizer    │    │ Performance  │
│ Predictions  │───▶│ (Grid Search)    │───▶│ Tracking     │
│ ml.predictions│    │ py/ensemble/     │    │ ml.ensemble_ │
└──────────────┘    │ adaptive_wls_v2  │    │ weight_history│
                    └──────────────────┘    └──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Production       │
                    │ Ensemble         │
                    │ (Auto-reload)    │
                    └──────────────────┘
```

### Data Flow

1. **Prediction Generation**: Each week, ensemble v6.0 generates predictions and stores component predictions in `ml.predictions.model_components`
2. **Actual Outcomes**: After games are played, actual values are filled in `ml.predictions.actual_value`
3. **Weekly Re-Optimization** (every Monday 6am):
   - Fetch last 6 weeks of predictions + actuals
   - Run WLS optimization with grid search over halflife candidates (2-8 weeks)
   - Save optimized weights to `ml.ensemble_weight_history`
   - Mark previous weights as `is_active = FALSE`
4. **Production Reload**: Ensemble automatically reloads weights on next prediction call

---

## Database Schema

### Table 1: `ml.ensemble_weight_history`

Tracks weekly ensemble weight re-optimization results.

**Key Columns**:
- `ensemble_version` - Ensemble version (e.g., 'v6.0')
- `effective_season` / `effective_week` - Week these weights were deployed
- `state_space_weight`, `hierarchical_weight`, `xgboost_weight`, `informative_priors_weight`, `hybrid_weight` - Component weights
- `component_weights` - Full weight vector as JSONB
- `optimization_method` - Method used ('wls_grid_search', 'equal_weight', 'meta_learning')
- `halflife` - WLS recency halflife in weeks (2-8 typical)
- `validation_mae` / `validation_rmse` - Performance on validation set
- `weight_change_magnitude` - L1 norm of weight changes from previous week
- `is_active` - Currently in production?

**Constraints**:
- Weights must sum to 1.0 (within 0.001 tolerance)
- All weights must be non-negative
- Unique (ensemble_version, effective_season, effective_week)

**Sample Query**:
```sql
-- Get weight evolution for v6.0 over last 8 weeks
SELECT
    effective_week,
    state_space_weight,
    hierarchical_weight,
    xgboost_weight,
    validation_mae,
    weight_change_magnitude
FROM ml.ensemble_weight_history
WHERE ensemble_version = 'v6.0'
  AND effective_season = 2024
ORDER BY effective_week;
```

### Table 2: `ml.feature_importance_history`

Tracks feature importance evolution over time for each model.

**Key Columns**:
- `model_name` - Model name (e.g., 'xgboost_v4_ridge')
- `computed_season` / `computed_week` - Week importance was computed
- `feature_name` - Name of feature
- `composite_score` - Normalized 0-1 composite importance
- `importance_rank` - Rank among all features (1 = most important)
- `rank_change` - Change from previous week
- `importance_trend` - 'increasing', 'decreasing', 'stable'

**Sample Query**:
```sql
-- Get top 10 features with biggest rank changes
SELECT
    feature_name,
    importance_rank,
    rank_change,
    importance_trend
FROM ml.feature_importance_history
WHERE model_name = 'xgboost_v4_ridge'
  AND computed_season = 2024
  AND computed_week = 13
ORDER BY ABS(rank_change) DESC
LIMIT 10;
```

### Views

**`ml.latest_ensemble_weights`**:
- Current production weights for each ensemble version
- Automatically filters to `is_active = TRUE`

**`ml.ensemble_weight_evolution`**:
- Time series of weight changes for Grafana dashboards
- Includes season_week formatted column (e.g., 'S2024W13')

**`ml.top_features_current`**:
- Top 10 most important features for each model
- Latest computation only

### Functions

**`ml.get_weight_change(ensemble_version, season, week_start, week_end)`**:
- Compare weights between two weeks
- Returns: component, weight_start, weight_end, absolute_change, percent_change
- Example: `SELECT * FROM ml.get_weight_change('v6.0', 2024, 10, 13);`

---

## Weekly Re-Optimization Pipeline

### Script: `scripts/weekly_ensemble_reoptimization.py`

**Purpose**: Automated weekly weight re-optimization using recent performance data

**Algorithm**:
1. **Fetch Data**: Retrieve predictions + actuals from last N weeks (default: 6)
2. **Grid Search**: Try multiple halflife candidates (2, 3, 4, 5, 6, 8 weeks)
3. **WLS Optimization**: For each halflife:
   - Compute exponential recency weights: `w_i = 2^{-(n - i) / halflife}`
   - Solve weighted least squares: `min_weights ||W^0.5 (Xw - y)||_2^2`
   - Constrain: weights ≥ 0, sum(weights) = 1
4. **Validation**: Use 80/20 train-validation split to select best halflife
5. **Save**: Store optimized weights to database with metadata

**Parameters**:
- `--season` - NFL season (e.g., 2024)
- `--week` - Current week (weights will be deployed for this week)
- `--window` - Training window size in weeks (default: 6)
- `--halflife-candidates` - Comma-separated values (default: 2,3,4,5,6,8)
- `--ensemble-version` - Ensemble version (default: v6.0)
- `--dry-run` - Run optimization but don't save to database

**Example Usage**:
```bash
# Manual run (dry run for testing)
uv run python scripts/weekly_ensemble_reoptimization.py \
    --season 2024 \
    --week 13 \
    --window 6 \
    --dry-run

# Production run (saves to database)
uv run python scripts/weekly_ensemble_reoptimization.py \
    --season 2024 \
    --week 13

# Custom halflife candidates
uv run python scripts/weekly_ensemble_reoptimization.py \
    --season 2024 \
    --week 13 \
    --halflife-candidates 3,4,5
```

### Cron Job Setup

**Schedule**: Every Monday 6am (after all Week N games completed)

```bash
# Add to crontab
0 6 * * 1 cd \ && \
    uv run python scripts/weekly_ensemble_reoptimization.py \
    --season 2024 \
    --week AUTO \
    >> /tmp/weekly_reoptimization.log 2>&1
```

**Note**: `--week AUTO` would require implementing auto-detection of current NFL week (TODO)

---

## Usage Guide

### Scenario 1: Start Using Adaptive Weights (After 6+ Weeks)

**Prerequisites**:
- At least 4-6 weeks of v6.0 predictions with actuals in `ml.predictions`
- Component predictions stored in `model_components` JSONB column

**Steps**:

1. **Verify Data**:
```sql
-- Check prediction coverage
SELECT
    season,
    week,
    COUNT(*) as predictions,
    COUNT(CASE WHEN actual_value IS NOT NULL THEN 1 END) as with_actuals
FROM ml.predictions
WHERE ensemble_version = 'v6.0'
GROUP BY season, week
ORDER BY season, week;

-- Should have 4-6 weeks with actuals filled in
```

2. **Run Re-Optimization** (dry run first):
```bash
uv run python scripts/weekly_ensemble_reoptimization.py \
    --season 2024 \
    --week 13 \
    --window 6 \
    --dry-run
```

3. **Review Results**:
```
✅ Optimization complete:
   Optimal halflife: 4 weeks
   Validation MAE: 45.32
   Optimal weights:
     state_space_v2: 0.2854
     hierarchical_v2: 0.2691
     xgboost_v4_ridge: 0.2377
     informative_priors_v2: 0.1278
     hybrid_static_incremental: 0.0800
```

4. **Deploy to Production**:
```bash
# Remove --dry-run flag
uv run python scripts/weekly_ensemble_reoptimization.py \
    --season 2024 \
    --week 13
```

5. **Verify Deployment**:
```sql
SELECT
    ensemble_version,
    effective_week,
    state_space_weight,
    hierarchical_weight,
    xgboost_weight,
    validation_mae,
    is_active
FROM ml.ensemble_weight_history
WHERE ensemble_version = 'v6.0'
  AND effective_season = 2024
ORDER BY effective_week DESC
LIMIT 3;
```

### Scenario 2: Monitor Weight Evolution

```sql
-- Visualize weight changes over time
SELECT
    CONCAT('W', effective_week) as week,
    state_space_weight,
    hierarchical_weight,
    xgboost_weight,
    informative_priors_weight,
    hybrid_weight,
    validation_mae,
    weight_change_magnitude
FROM ml.ensemble_weight_evolution
WHERE ensemble_version = 'v6.0'
  AND effective_season = 2024
ORDER BY effective_week;
```

### Scenario 3: Detect Sudden Weight Shifts

```sql
-- Find weeks with large weight changes (>10% total shift)
SELECT
    effective_week,
    weight_change_magnitude,
    max_weight_change,
    changed_component,
    validation_mae
FROM ml.ensemble_weight_history
WHERE ensemble_version = 'v6.0'
  AND effective_season = 2024
  AND weight_change_magnitude > 0.10  -- 10% total shift
ORDER BY weight_change_magnitude DESC;
```

**Investigation**: If large weight shift detected:
1. Check validation MAE - did performance improve?
2. Review component predictions for that week - any failures?
3. Check feature importance - did key features shift?

---

## Monitoring & Visualization

### Grafana Dashboard (TODO - Phase 2)

**Recommended Panels**:

1. **Weight Evolution Time Series**:
   - Data Source: PostgreSQL (`ml.ensemble_weight_evolution`)
   - Visualization: Time series (stacked area chart)
   - Query:
   ```sql
   SELECT
       deployed_at as time,
       state_space_weight,
       hierarchical_weight,
       xgboost_weight,
       informative_priors_weight,
       hybrid_weight
   FROM ml.ensemble_weight_evolution
   WHERE ensemble_version = 'v6.0'
     AND $__timeFilter(deployed_at)
   ORDER BY deployed_at
   ```

2. **Validation MAE Trend**:
   - Track if adaptive weights improve validation MAE over time
   - Compare to static baseline (v6.0 initial weights)

3. **Weight Stability Gauge**:
   - Alert if `weight_change_magnitude > 0.15` (15% total shift)
   - Indicates potential model instability or data drift

4. **Component Contribution Heatmap**:
   - Visualize which components are gaining/losing weight over time

### Alert Rules

```yaml
# Grafana alert: Large weight shift
- alert: LargeEnsembleWeightShift
  expr: weight_change_magnitude > 0.15
  for: 1h
  annotations:
    summary: "Ensemble weights shifted by {{ $value | humanizePercentage }}"
    description: "Component {{ $labels.changed_component }} changed significantly"

# Grafana alert: Validation MAE degradation
- alert: AdaptiveWeightsDegraded
  expr: validation_mae > (baseline_mae * 1.10)
  for: 2h
  annotations:
    summary: "Adaptive weights performing {{ $value | humanize }} worse than baseline"
```

---

## Roadmap & Future Enhancements

### Phase 1: Infrastructure (✅ COMPLETE - Nov 22, 2025)
- [x] Create database tables (`ml.ensemble_weight_history`, `ml.feature_importance_history`)
- [x] Build weekly re-optimization pipeline
- [x] Initialize baseline v6.0 weights

### Phase 2: Data Accumulation (In Progress)
- [ ] Generate predictions for Weeks 13-18 (2024 season)
- [ ] Fill in actual values as games complete
- [ ] Accumulate 6 weeks of data (by Week 18)

### Phase 3: First Adaptive Optimization (ETA: Dec 23, 2024)
- [ ] Run first adaptive weight optimization (Week 18 or Playoffs)
- [ ] Compare adaptive weights to static baseline
- [ ] Validate MAE improvement (+2-5% expected)

### Phase 4: Automation (ETA: Jan 2025)
- [ ] Set up cron job for weekly Monday 6am runs
- [ ] Implement AUTO week detection (parse NFL schedule)
- [ ] Add Slack/email notifications on completion

### Phase 5: Feature Importance Tracking (ETA: Feb 2025)
- [ ] Implement weekly feature importance computation
- [ ] Populate `ml.feature_importance_history`
- [ ] Build feature drift detection alerts

### Phase 6: Advanced Adaptations (ETA: Mar 2025)
- [ ] Experiment with alternative optimization methods (meta-learning, Thompson sampling)
- [ ] Implement per-position adaptive weights (QB vs RB vs WR)
- [ ] Add contextual weight adaptation (home/away, division games, weather)

---

## Troubleshooting

### Issue 1: "No predictions found in database"

**Symptoms**:
```
❌ No predictions found in database for specified window
   Make sure predictions have been generated and actual values filled in
```

**Cause**: Not enough v6.0 predictions with actuals in `ml.predictions`

**Solution**:
1. Check prediction coverage:
```sql
SELECT season, week, COUNT(*)
FROM ml.predictions
WHERE ensemble_version = 'v6.0'
  AND actual_value IS NOT NULL
GROUP BY season, week;
```

2. If no data: Generate predictions for historical weeks:
```bash
# Generate v6.0 predictions for Week N
uv run python scripts/generate_predictions_v6.py --season 2024 --week N
```

3. Wait for games to complete and actual values to be filled

### Issue 2: Weights sum to ≠ 1.0

**Symptoms**:
```
ERROR: new row for relation "ensemble_weight_history" violates check constraint "check_weights_sum_to_one"
```

**Cause**: CVXPY optimizer returned weights that don't sum exactly to 1.0 due to numerical precision

**Solution**: The constraint allows 0.001 tolerance. If still failing, check WLS optimizer:
```python
# In adaptive_wls_weights_v2.py
# Add renormalization after optimization
weights = optimal_weights / sum(optimal_weights.values())
```

### Issue 3: Validation MAE worse than baseline

**Symptoms**: `validation_mae > 50` when baseline is ~40

**Possible Causes**:
1. **Too little data**: Need at least 4 weeks for stable WLS estimates
2. **Halflife too short**: Try longer halflife (6-8 weeks) for more stability
3. **Model failures**: Check if any component had systematic failures in training window

**Solution**:
```bash
# Try longer halflife
uv run python scripts/weekly_ensemble_reoptimization.py \
    --halflife-candidates 6,8,10

# Or increase training window
uv run python scripts/weekly_ensemble_reoptimization.py \
    --window 8
```

### Issue 4: Component weight goes to 0

**Symptoms**: One component gets 0% weight in optimization

**Possible Causes**:
1. Component predictions systematically worse than others
2. Component failed for many games in training window
3. High correlation with another component (redundant)

**Investigation**:
```sql
-- Check component MAE in training window
SELECT
    AVG(ABS(actual_value - (model_components->>'state_space_v2_mean')::NUMERIC)) as state_space_mae,
    AVG(ABS(actual_value - (model_components->>'hierarchical_v2_mean')::NUMERIC)) as hierarchical_mae,
    AVG(ABS(actual_value - (model_components->>'xgboost_v4_ridge_mean')::NUMERIC)) as xgboost_mae
FROM ml.predictions
WHERE ensemble_version = 'v6.0'
  AND season = 2024
  AND week BETWEEN 7 AND 12
  AND actual_value IS NOT NULL;
```

**Solution**: If one component consistently underperforms, consider:
- Debugging that component's predictions
- Adding minimum weight constraint (e.g., each component ≥ 5%)
- Removing component from ensemble if permanently degraded

---

## Key Takeaways

### What Works Right Now ✅
- Database tables created and initialized
- Weekly re-optimization pipeline functional
- WLS with grid search over halflife candidates
- Weight change tracking and history
- Baseline v6.0 weights saved

### What Needs Data ⏳
- Historical predictions + actuals (4-6 weeks minimum)
- First adaptive optimization requires ~300-500 predictions with actuals
- Expected availability: Late December 2024 (after Week 16-17)

### Expected Benefits 📈
- **+2-5% MAE improvement** from adaptive weighting
- **Faster drift detection** when models degrade
- **Automated optimization** no manual weight tuning
- **Historical tracking** of weight evolution and stability

### Next Steps 🚀
1. **Continue generating v6.0 predictions** each week
2. **Fill in actuals** after games complete
3. **Monitor data accumulation** in `ml.predictions`
4. **First adaptive run**: Week 18 or Playoffs (Dec 2024)

---

**Questions or Issues**: See [docs/POSTMORTEMS.md](POSTMORTEMS.md) or [docs/troubleshooting/](troubleshooting/)

**Created**: November 22, 2025
**Last Updated**: November 22, 2025
**Status**: ✅ Infrastructure Complete, Ready for Data Accumulation
