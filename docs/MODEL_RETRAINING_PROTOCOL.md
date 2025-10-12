# Model Retraining Protocol: Bayesian + XGBoost Integration

**Status**: Production Ready
**Last Updated**: October 11, 2025
**Owner**: NFL Analytics Team

---

## Overview

This document specifies the weekly model retraining protocol for the NFL betting system, integrating Bayesian hierarchical team ratings with XGBoost gradient boosting models. The protocol ensures models remain calibrated to current NFL dynamics while maintaining stable, reproducible predictions.

## Retraining Schedule

### Weekly Cycle (Every Tuesday)

**Why Tuesday?**
- All week's games completed by Monday night
- Injury reports finalized
- Sufficient time to validate before Thursday Night Football
- Market lines not yet available (avoid data leakage)

### Timeline

```
Monday Night    → Games complete, data ingestion begins
Tuesday 6am     → Data quality checks complete
Tuesday 8am     → Bayesian model retraining starts
Tuesday 8:20am  → Bayesian model complete (18 sec runtime)
Tuesday 8:30am  → Feature generation with Bayesian ratings
Tuesday 9:00am  → XGBoost retraining starts (if needed)
Tuesday 10:00am → XGBoost complete, validation begins
Tuesday 11:00am → Predictions generated for upcoming week
Tuesday 12:00pm → Retrospective analysis on previous week
Tuesday 2:00pm  → Learning patterns extracted, model adjustments identified
Wednesday       → Predictions reviewed, betting recommendations finalized
```

---

## Component 1: Bayesian Hierarchical Model Retraining

### Training Data Window

**Standard Configuration:**
- **Lookback**: Most recent 5 seasons (e.g., 2020-2024 for 2025 predictions)
- **Regular season only**: Exclude preseason and playoffs
- **Min games per season**: 250 (quality check)

**Rationale:**
- 5 seasons = ~1,350 games provides stable posteriors
- Captures recent NFL evolution (rule changes, offensive trends)
- Avoids stale data from pre-2015 era (different game dynamics)

### Model Specification

**Production Model**: Model 2 (Time-Varying Effects)
```r
brm(
  margin ~ home + (1 + t | home_team) + (1 + t | away_team),
  data = games,
  family = gaussian(),
  prior = c(
    prior(normal(2.4, 1), class = Intercept),  # home advantage prior
    prior(exponential(0.3), class = sd),        # team effect SD
    prior(exponential(0.1), class = sigma)      # residual SD
  ),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  backend = "cmdstanr",
  seed = 42
)
```

**Key Parameters:**
- `t`: Normalized season progress [0, 1]
- `home`: Fixed home-field advantage (~2.4 points)
- Random effects: Team-specific intercepts + slopes (momentum)

### Execution

```bash
# Run weekly Bayesian retraining
Rscript R/models/bayesian_hierarchical_weekly.R \
  --current-season 2025 \
  --lookback-seasons 5 \
  --output models/bayesian/weekly_ratings_$(date +%Y%m%d).rds
```

### Convergence Diagnostics

**Required Checks:**
1. **$\hat{R}$ (Gelman-Rubin)**: All parameters < 1.01
2. **Effective Sample Size (ESS)**: All > 1000
3. **Divergent transitions**: < 10 per chain
4. **Tree depth saturation**: < 5% of iterations

**Automated Validation:**
```r
# In R/models/bayesian_hierarchical_weekly.R
diagnostics <- rstan::monitor(fit$fit, print = FALSE)
if (any(diagnostics$Rhat > 1.01)) {
  stop("❌ Convergence failure: Rhat > 1.01")
}
if (any(diagnostics$n_eff < 1000)) {
  warning("⚠ Low ESS detected, consider more iterations")
}
```

### Export Bayesian Ratings

After fitting, export posterior means and SDs:
```r
# Extract team ratings
ratings <- posterior_summary(fit, variable = "r_team") %>%
  as_tibble() %>%
  mutate(
    team = str_extract(variable, "[A-Z]{2,3}"),
    parameter = str_extract(variable, "Intercept|t"),
    rating_mean = Estimate,
    rating_sd = Est.Error,
    rating_q05 = Q5,
    rating_q95 = Q95
  )

# Save to database
DBI::dbWriteTable(conn, "mart.bayesian_team_ratings", ratings, overwrite = TRUE)
```

---

## Component 2: Feature Engineering with Bayesian Integration

### Pipeline

```bash
# Generate Bayesian-enhanced features
python py/features/bayesian_features.py \
  --input data/processed/features/asof_team_features_v3.csv \
  --output data/processed/features/asof_team_features_v3_bayesian.csv \
  --home-advantage 2.4 \
  --add-predictions
```

### New Features Added (13 total)

| Feature | Type | Description | Usage |
|---------|------|-------------|-------|
| `home_bayesian_rating` | Float | Home team posterior mean strength | Direct input to XGBoost |
| `away_bayesian_rating` | Float | Away team posterior mean strength | Direct input to XGBoost |
| `bayesian_rating_diff` | Float | home - away rating | Primary signal |
| `home_bayesian_sd` | Float | Home team posterior uncertainty | Kelly sizing weight |
| `away_bayesian_sd` | Float | Away team posterior uncertainty | Kelly sizing weight |
| `bayesian_combined_sd` | Float | $\sqrt{\sigma_h^2 + \sigma_a^2}$ | Game-level uncertainty |
| `bayesian_confidence` | Float | $1/(1 + \text{combined\_sd})$ | Inverse uncertainty |
| `bayesian_pred_margin` | Float | Predicted margin (Bayesian only) | Ensemble component |
| `bayesian_prob_home` | Float | $\Phi(\text{margin}/\sigma)$ | Ensemble probability |
| `home_bayesian_q05` | Float | 5th percentile (90% CI lower) | Risk assessment |
| `home_bayesian_q95` | Float | 95th percentile (90% CI upper) | Risk assessment |
| `away_bayesian_q05` | Float | Away 5th percentile | Risk assessment |
| `away_bayesian_q95` | Float | Away 95th percentile | Risk assessment |

---

## Component 3: XGBoost Model Retraining

### When to Retrain XGBoost

**Full Retraining Triggers:**
1. **New Bayesian features added** (first time integration)
2. **Quarterly schedule** (January, April, July, October)
3. **Major data pipeline changes**
4. **Performance degradation** (Brier > 0.260 for 3 consecutive weeks)

**Feature Update Only:**
- Weekly Bayesian rating refresh **does not** require XGBoost retraining
- XGBoost learned feature interactions remain valid
- Only update input features, use existing model weights

### Full Retraining Protocol

```bash
# Retrain XGBoost v3 with Bayesian features
uv run python py/models/xgboost_gpu_v3.py \
  --features-csv data/processed/features/asof_team_features_v3_bayesian.csv \
  --start-season 2006 \
  --end-season 2023 \
  --test-seasons 2024 \
  --sweep \
  --output-dir models/xgboost/v3_bayesian_$(date +%Y%m%d) \
  --device cuda  # or cpu
```

**Expected Duration:**
- Hyperparameter sweep (100 trials): 4-6 hours on GPU
- Single best config: 15-20 minutes on GPU
- Validation: 10 minutes

### Hyperparameter Search Space

```python
{
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [1.0, 1.5, 2.0],
}
```

**Best Config (as of Oct 2025):**
```json
{
  "max_depth": 6,
  "learning_rate": 0.03,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "min_child_weight": 3,
  "reg_alpha": 0.1,
  "reg_lambda": 1.5,
  "n_estimators": 500
}
```

### Validation Metrics

**Acceptance Criteria (2024 test set):**
- Brier Score: < 0.260
- Log Loss: < 0.65
- AUC: > 0.58
- Calibration ECE: < 0.05
- ATS Accuracy: > 51.5%

**Regression Testing:**
```python
# Compare new model vs current production
old_metrics = load_metrics("models/xgboost/v3_best.json")
new_metrics = evaluate_model("models/xgboost/v3_bayesian_20251011.json")

assert new_metrics["brier"] <= old_metrics["brier"] * 1.02  # No more than 2% degradation
assert new_metrics["ats_accuracy"] >= 0.515  # Absolute minimum
```

---

## Component 4: Ensemble Prediction Generation

### Ensemble Configuration

**Weights:**
- Bayesian: 25% (0.25)
- XGBoost: 75% (0.75)

**Agreement Threshold:**
- Max probability difference: 10% (0.10)
- Only bet if $|p_{\text{Bayes}} - p_{\text{XGB}}| < 0.10$

**Edge Threshold:**
- Minimum edge after vig: 2% (0.02)

### Weekly Prediction Generation

```bash
# Generate predictions for upcoming week
python py/predictions/generate_predictions.py \
  --season 2025 \
  --week 7 \
  --version "5_days_out" \
  --bayesian-weight 0.25
```

**Output:** Predictions stored in `predictions.game_predictions` table with:
- Ensemble probability
- Component breakdowns (Bayesian vs XGBoost)
- Betting recommendations
- Kelly fractions
- Edge estimates

### Prediction Versioning

Track prediction evolution as information updates:
```bash
# 5 days before games
python py/predictions/generate_predictions.py --week 7 --version "5_days_out"

# Day before games (with updated injury reports)
python py/predictions/generate_predictions.py --week 7 --version "day_before"

# Day of games (final predictions)
python py/predictions/generate_predictions.py --week 7 --version "day_of"
```

---

## Component 5: Retrospective Analysis & Learning

### Post-Week Analysis

**Every Tuesday after new data loads:**
```bash
# Analyze completed games
python py/predictions/retrospective_analyzer.py --auto
```

**Automated Classification:**
- `correct_high_conf`: Predicted correctly with >70% confidence
- `wrong_upset`: Missed major upset (>7 pt favorite lost)
- `wrong_blowout`: Massive error (>14 pts off)

### Monthly Pattern Extraction

**First Tuesday of each month:**
```bash
# Extract learning patterns
python py/predictions/extract_learning_patterns.py \
  --min-sample-size 10 \
  --verbose
```

**Output:** Patterns stored in `predictions.learning_loop` with:
- Statistical significance (p-values, effect sizes)
- Feature recommendations
- Expected ROI improvement estimates
- Implementation priority

### Example: Thursday Night Effect

If pattern detected:
```
pattern_name: thursday_night_home_advantage
sample_size: 18
avg_prediction_error: +2.1 points
p_value: 0.024 (significant!)
feature_to_add: thursday_night_home
expected_improvement: +0.51% ROI
priority: high
```

**Action:**
1. Add `thursday_night` binary feature to feature engineering
2. Retrain XGBoost with new feature
3. Validate improvement on 2022-2023 holdout
4. Mark pattern as `implemented` in database
5. Track actual vs expected ROI gain

---

## Model Performance Monitoring

### Key Metrics Dashboard

**Track Weekly:**
- Win rate (target: 54-55%)
- Expected ROI (target: +2.0-3.0%)
- Closing Line Value (CLV) in basis points
- Brier score on completed games
- Calibration error (ECE)

### Alert Thresholds

**Trigger retraining if:**
- Win rate < 52% for 3 consecutive weeks
- CLV < -5 bps for 2 consecutive weeks
- Brier score > 0.270 for any week
- Bayesian convergence failures ($\hat{R} > 1.01$)
- XGBoost prediction latency > 100ms

### Automated Email Alerts

```python
# In monitoring script
if win_rate < 0.52 and weeks_below_threshold >= 3:
    send_alert(
        subject="⚠ Model Performance Degradation",
        body=f"Win rate {win_rate:.1%} for {weeks_below_threshold} weeks. Consider retraining."
    )
```

---

## Dependency Management

### Required Software Versions

```
Python: 3.11+
R: 4.3+
brms: 2.20+
cmdstanr: 0.7+
xgboost: 2.0+
pandas: 2.1+
numpy: 1.26+
scipy: 1.11+
psycopg: 3.1+
```

### Environment Setup

```bash
# Python environment
cd nfl-analytics
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# R packages
Rscript -e "install.packages(c('brms', 'cmdstanr', 'tidyverse', 'arrow'))"
```

---

## Rollback Procedure

If new model performs poorly:

1. **Immediate**: Revert to previous week's model
   ```bash
   cp models/xgboost/v3_best_backup.json models/xgboost/v3_best.json
   cp models/bayesian/ratings_backup.rds models/bayesian/ratings_current.rds
   ```

2. **Investigation**: Check diagnostics
   - Review convergence warnings
   - Examine feature distributions for anomalies
   - Compare predictions to market lines

3. **Fix & Retrain**: Address root cause
   - Bad data? Re-run ETL
   - Convergence issues? Increase iterations or adjust priors
   - Feature corruption? Regenerate features

4. **Validation**: Backtest on 2023-2024 before re-deploying

---

## Reproducibility

### Seeds and Versioning

**Always set seeds:**
```python
# Python
np.random.seed(42)
random.seed(42)

# R
set.seed(42)

# XGBoost
params = {"seed": 42, ...}
```

**Version Control:**
- Tag each model release: `git tag v3.1_bayesian_integration`
- Store model artifacts with timestamps
- Log hyperparameters in MLflow or similar

### Model Registry

```
models/
├── xgboost/
│   ├── v3_best.json                 # Current production
│   ├── v3_bayesian_20251011.json    # This week's retrained model
│   └── archive/
│       └── v3_20251004.json         # Last week's backup
├── bayesian/
│   ├── ratings_current.rds          # Current production
│   ├── ratings_20251011.rds         # This week's retrained
│   └── archive/
│       └── ratings_20251004.rds     # Last week's backup
└── metadata/
    ├── training_log_20251011.json   # Training metrics, runtime, etc.
    └── hyperparams_v3_bayesian.json # Best hyperparameters
```

---

## Summary: Weekly Checklist

### Tuesday Morning (Data & Bayesian)
- [ ] Verify all Monday games loaded (check game count)
- [ ] Run Bayesian model retraining (18 sec)
- [ ] Check convergence diagnostics ($\hat{R}$, ESS)
- [ ] Export ratings to `mart.bayesian_team_ratings`
- [ ] Regenerate features with Bayesian integration

### Tuesday Late Morning (XGBoost & Predictions)
- [ ] Update XGBoost input features (weekly)
- [ ] Retrain XGBoost if triggered (quarterly or on-demand)
- [ ] Generate predictions for upcoming week
- [ ] Store predictions with version labels

### Tuesday Afternoon (Retrospective & Learning)
- [ ] Run retrospective analysis on last week's games
- [ ] Review high-priority learnings
- [ ] Extract patterns if month-end
- [ ] Update master_todos with implementation tasks

### Wednesday (Review & Deploy)
- [ ] Review betting recommendations
- [ ] Verify edge estimates vs opening lines
- [ ] Deploy predictions to production API
- [ ] Monitor Closing Line Value through week

---

**Author**: NFL Analytics Team
**Review**: Approved for Production
**Next Review**: Monthly or after major performance changes
