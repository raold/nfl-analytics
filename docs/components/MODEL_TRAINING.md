# Model Training Pipeline - Comprehensive Component Documentation

**Last Updated:** November 15, 2025
**Version:** 5.1 (Production)
**Status:** Operational
**Author:** Claude Code (Sonnet 4.5)

## Table of Contents

1. [Component Overview](#component-overview)
2. [Training Architecture](#training-architecture)
3. [R Model Training (Bayesian)](#r-model-training-bayesian)
4. [Python Model Training (XGBoost, BNN)](#python-model-training-xgboost-bnn)
5. [Training Orchestration](#training-orchestration)
6. [MCMC Diagnostics & Validation](#mcmc-diagnostics--validation)
7. [Model Registry Integration](#model-registry-integration)
8. [Canonical Training Window](#canonical-training-window)
9. [Parallel Training](#parallel-training)
10. [Model Versioning & Tracking](#model-versioning--tracking)
11. [Convergence Validation](#convergence-validation)
12. [Performance Optimization](#performance-optimization)
13. [Usage Examples](#usage-examples)
14. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
15. [Production Deployment Checklist](#production-deployment-checklist)

---

## Component Overview

The **Model Training Pipeline** manages the end-to-end workflow for training, validating, and registering production NFL props prediction models. It coordinates:

- **4 Bayesian models** (R/Stan/brms): State-Space v2, Hierarchical v2, Informative Priors v2
- **1 ML model** (Python/XGBoost): Player Props v2
- **MCMC diagnostics** (Rhat, ESS, divergences)
- **Model registry** (database sync, versioning)
- **Convergence validation** (85%+ calibration target)

### Key Capabilities

| Capability | Implementation | Status |
|------------|----------------|--------|
| R model training | brms + cmdstanr | ✅ Production |
| XGBoost training | Python + scikit-learn | ✅ Production |
| Parallel training | GNU parallel + shell | ✅ Operational |
| MCMC diagnostics | posterior + bayesplot | ✅ Automated |
| Model registry | PostgreSQL + YAML | ✅ Synced |
| Convergence validation | Rhat <1.01, ESS >400 | ✅ Enforced |

---

## Training Architecture

### High-Level Workflow

```
Training Request
    ↓
┌─────────────────────────────────────────────────────┐
│ 1. Data Preparation                                  │
│    - Load from mart.player_game_stats               │
│    - Apply training window (2015-2024 or 2018+decay)│
│    - Feature engineering                            │
│    - Temporal validation (no leakage)               │
└──────────────┬──────────────────────────────────────┘
               ↓
      ┌────────┴──────────┬─────────────┐
      │                   │             │
      ▼                   ▼             ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────┐
│ R Training   │  │ XGBoost       │  │ State-Space  │
│ (brms)       │  │ Training      │  │ Training     │
│              │  │               │  │              │
│ - Hierarchical│  │ - DMatrix    │  │ - Stan       │
│ - Informative │  │ - Optuna     │  │ - Kalman     │
│   Priors      │  │ - HPO        │  │   filter     │
│              │  │               │  │              │
│ 3-6 hours    │  │ 30-60 min    │  │ 1-2 hours    │
└──────┬───────┘  └───────┬───────┘  └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  ▼
   .rds files      .pkl files          .rds files
       │                  │                  │
       └──────────────────┴──────────────────┘
                          ↓
           ┌──────────────────────────────┐
           │ 2. MCMC Diagnostics          │
           │    - Rhat < 1.01             │
           │    - ESS > 400               │
           │    - Divergences < 10        │
           │    - Energy diagnostic       │
           └──────────┬───────────────────┘
                      ↓
           ┌──────────────────────────────┐
           │ 3. Calibration Validation    │
           │    - 90% CI coverage > 85%   │
           │    - Prediction vs actual    │
           │    - MAE < 50 yards          │
           └──────────┬───────────────────┘
                      ↓
           ┌──────────────────────────────┐
           │ 4. Model Registry            │
           │    - Database insert         │
           │    - YAML metadata           │
           │    - Git LFS tracking        │
           └──────────┬───────────────────┘
                      ↓
            ✅ APPROVED or ❌ REJECTED
```

### File Organization

```
models/
├── bayesian/
│   ├── simplified_random_walk_v2.rds         # State-Space v2 (47.77 MAE)
│   ├── passing_yards_hierarchical_v2.rds     # Hierarchical v2
│   ├── passing_informative_priors_v2.rds     # Informative Priors v2
│   └── *_metadata.json                       # Model metadata
│
├── xgboost/
│   ├── passing_xgboost_v2_canonical.pkl      # XGBoost v2
│   └── *_metadata.json
│
├── calibration/
│   └── temperature_scaler_4way.pkl           # Temperature scaler
│
└── MODEL_REGISTRY.yaml                       # Production manifest
```

---

## R Model Training (Bayesian)

### Hierarchical Bayesian Model v2

**Purpose:** Player-level hierarchical model with team/position random effects

**Training Script:** `R/train_hierarchical_v2_canonical.R`

**Formula:**
```r
yards ~ log_attempts + is_home + is_favored + spread_abs +
        years_exp + is_bad_weather + is_dome + total_close +
        (1 | player_id) + (1 | team) + (1 | opponent)
```

**Priors:**
```r
priors <- c(
  prior(normal(5.5, 0.5), class = Intercept),      # Log-scale mean
  prior(normal(0, 0.3), class = b),                # Regression coefficients
  prior(exponential(5), class = sd),               # Random effect std
  prior(exponential(1), class = sigma)             # Residual std
)
```

**Training Configuration:**
```r
fit <- brm(
  formula = formula,
  data = train_data,
  family = gaussian(),
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  thin = 2,
  cores = 4,
  backend = "cmdstanr",
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  seed = 42
)
```

**Output Files:**
- Model: `models/bayesian/passing_yards_hierarchical_v2.rds`
- Metadata: `models/bayesian/passing_yards_hierarchical_v2_metadata.json`
- Diagnostics: Printed to console (Rhat, ESS, divergences)

**Training Time:** 3-6 hours (4 chains × 4000 iterations)

**Expected Performance:**
- MAE: ~45 yards
- Calibration: 85-90%
- Rhat: <1.01
- ESS (bulk): >400

### State-Space Model v2

**Purpose:** Random walk model with player-specific skill trajectories

**Training Script:** `R/state_space/train_state_space_v2_canonical.R`

**Model:** Two-level random walk
```
Observation: y_it ~ Normal(μ_it, σ_obs)
State: μ_it ~ Normal(μ_i,t-1, σ_state)
Initial: μ_i0 ~ Normal(α, σ_init)
```

**Stan Code:**
```stan
data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1,upper=P> player[N];
  vector[N] y;
  vector[N] attempts;
}
parameters {
  real alpha;
  real<lower=0> sigma_obs;
  real<lower=0> sigma_state;
  real<lower=0> sigma_init;
  vector[P] skill_0;
  matrix[P, T] skill;
}
model {
  // Priors
  alpha ~ normal(250, 50);
  sigma_obs ~ exponential(0.01);
  sigma_state ~ exponential(0.1);

  // Initial states
  skill_0 ~ normal(alpha, sigma_init);

  // State evolution
  for (p in 1:P) {
    skill[p, 1] ~ normal(skill_0[p], sigma_state);
    for (t in 2:T) {
      skill[p, t] ~ normal(skill[p, t-1], sigma_state);
    }
  }

  // Observations
  y ~ normal(skill[player], sigma_obs);
}
```

**Training Window:**
- Primary: 2018-2024 (full weight)
- Historical: 2015-2017 (H=4 exponential decay)
- Weights: w(s; 2025, 4) = 0.5^((2025-s)/4)

**Output Files:**
- Model: `models/bayesian/simplified_random_walk_v2.rds`
- Skills CSV: `models/bayesian/state_space_skills_v2.csv`
- Metadata: `models/bayesian/simplified_random_walk_v2_metadata.json`

**Training Time:** 1-2 hours (Stan compilation + sampling)

**Expected Performance:**
- MAE: 47.77 yards (validated Nov 8, 2025)
- Calibration: 88-92%
- Convergence: Rhat <1.01 for all parameters

### Informative Priors Model v2

**Purpose:** Incorporate domain knowledge via informative priors

**Training Script:** `R/train_informative_priors_v2_canonical.R`

**Formula:** Same as Hierarchical v2, but with stronger priors

**Informative Priors:**
```r
priors <- c(
  # Intercept: Typical QB throws 250 yards/game
  prior(normal(5.52, 0.2), class = Intercept),  # log(250) ≈ 5.52

  # Home advantage: ~10 yards
  prior(normal(0.04, 0.02), coef = is_home),   # log(260/250) ≈ 0.04

  # Favored teams: ~15 yards
  prior(normal(0.06, 0.02), coef = is_favored),

  # Bad weather: -20 yards
  prior(normal(-0.08, 0.03), coef = is_bad_weather),

  # Generic coefficients
  prior(normal(0, 0.1), class = b)
)
```

**Rationale:** Priors encode NFL domain knowledge (home advantage, weather effects) while allowing data to override if evidence is strong.

**Output Files:**
- Model: `models/bayesian/passing_informative_priors_v2.rds`
- Metadata: `models/bayesian/passing_informative_priors_v2_metadata.json`

**Training Time:** 3-5 hours

**Expected Performance:**
- MAE: ~46 yards
- Calibration: 86-89%
- Convergence: Excellent (informative priors aid sampling)

---

## Python Model Training (XGBoost, BNN)

### XGBoost Player Props v2

**Purpose:** Gradient boosted trees for player performance prediction

**Training Script:** `py/models/train_xgboost_canonical.py`

**Features:**
- Player stats: attempts, yards, TDs (last 3 games, season avg)
- Team stats: offensive/defensive efficiency
- Opponent: defensive ranks, matchup difficulty
- Vegas: spread, total, implied score
- Weather: temperature, wind, roof type
- Contextual: home/away, division rival, primetime

**Hyperparameter Optimization (Optuna):**
```python
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    model = xgb.XGBRegressor(**params, objective='reg:squarederror', seed=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    return mae  # Minimize MAE
```

**Best Hyperparameters (from 100 trials):**
```python
{
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0
}
```

**Training:**
```python
model = xgb.XGBRegressor(**best_params, seed=42)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=10
)
```

**Output Files:**
- Model: `models/xgboost/passing_xgboost_v2_canonical.pkl`
- Feature importance: `models/xgboost/feature_importance_v2.csv`
- Metadata: `models/xgboost/passing_xgboost_v2_metadata.json`

**Training Time:** 30-60 minutes (HPO) + 5 minutes (final model)

**Expected Performance:**
- MAE: ~39 yards
- Calibration: N/A (point predictions, not probabilistic)
- Feature importance: Top 3 are attempts, recent yards, opponent rank

### Bayesian Neural Network (BNN) - Research Line

**Status:** ❌ ABANDONED November 9, 2025

**Reason:** After 6 failed approaches and 84 person-hours, BNN passing model research terminated. Only BNN v3.1 Rushing (89.5% calibrated) remains in production.

**Failed Approaches:**
1. Multi-Output Passing v1 (data leakage, deleted)
2. Multi-Output Passing v2 (NaN losses)
3. Ensemble approach (convergence issues)
4. Informative priors (overfitting)
5. Simplified architecture (underfitting)
6. Hybrid BNN+XGBoost (marginal gains)

**Documentation:** See `analysis/dissertation/main/appendix_bnn_investigation.tex` (780 lines)

**Key Learnings:**
- Feature engineering critical (lag/rolling caused leakage)
- BNNs require careful architecture tuning
- Bayesian methods not always superior to XGBoost
- Know when to abandon research line

---

## Training Orchestration

### Sequential Training (Recommended)

**Script:** Manual execution of each training script sequentially

**Workflow:**
```bash
# 1. Train State-Space v2 (1-2 hours)
Rscript R/state_space/train_state_space_v2_canonical.R

# 2. Train Hierarchical v2 (3-6 hours)
Rscript R/train_hierarchical_v2_canonical.R

# 3. Train Informative Priors v2 (3-5 hours)
Rscript R/train_informative_priors_v2_canonical.R

# 4. Train XGBoost v2 (30-60 min)
python py/models/train_xgboost_canonical.py

# Total: 8-14 hours
```

**Advantages:**
- Simple to monitor
- Easy to debug failures
- Predictable resource usage

**Disadvantages:**
- Slow (sequential)
- Manual intervention required

### Parallel Training (Experimental)

**Script:** GNU parallel for concurrent training

**Workflow:**
```bash
# Create training manifest
cat > training_manifest.txt <<EOF
Rscript R/state_space/train_state_space_v2_canonical.R
Rscript R/train_hierarchical_v2_canonical.R
Rscript R/train_informative_priors_v2_canonical.R
python py/models/train_xgboost_canonical.py
EOF

# Run in parallel (4 jobs)
parallel -j 4 --eta < training_manifest.txt

# Total: 3-6 hours (limited by slowest)
```

**Advantages:**
- 2-3x faster (parallel execution)
- Automated

**Disadvantages:**
- Higher memory usage (4 processes)
- Harder to debug (interleaved logs)
- Risk of OOM if all models use large datasets

**Recommendations:**
- Use sequential for production
- Use parallel for experiments/backtests

---

## MCMC Diagnostics & Validation

### Convergence Criteria

All Bayesian models must pass these diagnostic gates:

| Diagnostic | Threshold | Purpose |
|------------|-----------|---------|
| **Rhat** | <1.01 | Chain convergence |
| **ESS (bulk)** | >400 | Effective sample size |
| **ESS (tail)** | >400 | Tail behavior |
| **Divergences** | <10 | Sampling issues |
| **Max treedepth** | <10 | Numerical stability |
| **Energy** | Pass | Geometry issues |

### R-hat (Gelman-Rubin)

**Formula:**
```
Rhat = sqrt(Var_pooled / W)

where:
  Var_pooled = between-chain variance
  W = within-chain variance
```

**Interpretation:**
- Rhat = 1.00: Perfect convergence
- Rhat < 1.01: Acceptable
- Rhat > 1.01: ⛔ FAILED - retrain with more iterations

**Check in R:**
```r
library(posterior)
fit_summary <- summarise_draws(fit)
max_rhat <- max(fit_summary$rhat, na.rm = TRUE)

if (max_rhat > 1.01) {
  stop("❌ Convergence failed: max Rhat = ", max_rhat)
}
```

### Effective Sample Size (ESS)

**Formula:**
```
ESS = (Total draws) / (1 + 2 * sum(autocorrelations))
```

**Interpretation:**
- ESS > 1000: Excellent
- ESS > 400: Acceptable
- ESS < 400: ⛔ FAILED - retrain with more iterations or thinning

**Check in R:**
```r
min_ess_bulk <- min(fit_summary$ess_bulk, na.rm = TRUE)
min_ess_tail <- min(fit_summary$ess_tail, na.rm = TRUE)

if (min_ess_bulk < 400 || min_ess_tail < 400) {
  stop("❌ ESS too low: bulk=", min_ess_bulk, ", tail=", min_ess_tail)
}
```

### Divergences

**What they are:** Numerical integration failures in HMC sampler

**Causes:**
- Posterior geometry issues (fat tails, multimodality)
- Step size too large
- Max treedepth too small
- Poorly specified priors

**Fixes:**
```r
# Increase adapt_delta (0.8 → 0.95)
control = list(adapt_delta = 0.95)

# Increase max_treedepth (10 → 12)
control = list(max_treedepth = 12)

# Reparameterize model (non-centered parameterization)
# Before: theta ~ normal(mu, sigma)
# After: theta_raw ~ normal(0, 1); theta = mu + theta_raw * sigma
```

### Diagnostic Workflow

```r
# 1. Fit model
fit <- brm(formula, data, ...)

# 2. Check diagnostics
print(fit)  # Prints Rhat, ESS, divergences

# 3. Visual diagnostics
library(bayesplot)
mcmc_trace(fit)       # Trace plots (should look like "hairy caterpillars")
mcmc_pairs(fit)       # Pairwise correlations
mcmc_nuts_divergence(fit)  # Divergence locations

# 4. Posterior predictive check
pp_check(fit, ndraws = 100)

# 5. Save if passed
if (all diagnostic gates pass) {
  saveRDS(fit, "models/bayesian/model_v2.rds")
  register_to_database(fit, metadata)
}
```

---

## Model Registry Integration

### Database Schema

**Table:** `ml.model_metadata`

```sql
CREATE TABLE ml.model_metadata (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,  -- 'bayesian_hierarchical', 'xgboost', etc.
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    training_window_start DATE,
    training_window_end DATE,
    performance_metrics JSONB,  -- MAE, calibration, etc.
    convergence_diagnostics JSONB,  -- Rhat, ESS, divergences
    feature_list TEXT[],
    registered_at TIMESTAMP DEFAULT NOW(),
    trained_by TEXT DEFAULT 'system',
    production_ready BOOLEAN DEFAULT FALSE,
    notes TEXT
);
```

### Registration Workflow

**Script:** `scripts/register_model.py`

```python
from py.utils.model_registry import ModelRegistry

registry = ModelRegistry()

# Register model
registry.register_model(
    model_name="passing_hierarchical_v2",
    model_type="bayesian_hierarchical",
    version="2.0",
    file_path="models/bayesian/passing_yards_hierarchical_v2.rds",
    training_window_start="2015-01-01",
    training_window_end="2024-12-31",
    performance_metrics={
        "mae": 45.2,
        "calibration_coverage": 0.887,
        "samples": 12456
    },
    convergence_diagnostics={
        "max_rhat": 1.003,
        "min_ess_bulk": 1842,
        "min_ess_tail": 1654,
        "divergences": 3
    },
    feature_list=[
        "log_attempts", "is_home", "is_favored", "spread_abs",
        "years_exp", "is_bad_weather", "is_dome", "total_close"
    ],
    production_ready=True,
    notes="Canonical training window (2015-2024 uniform)"
)
```

### YAML Manifest

**File:** `models/MODEL_REGISTRY.yaml`

```yaml
models:
  - name: passing_hierarchical_v2
    type: bayesian_hierarchical
    version: "2.0"
    file: models/bayesian/passing_yards_hierarchical_v2.rds
    training_window:
      start: "2015-01-01"
      end: "2024-12-31"
    performance:
      mae: 45.2
      calibration: 0.887
    convergence:
      max_rhat: 1.003
      min_ess: 1654
      divergences: 3
    production_ready: true
    registered_at: "2025-11-08T14:23:00Z"
```

### Sync Status

**Current (Nov 15, 2025):**
- Models in sync: 8
- Pending registrations: 227
- Corrupted models: 10

**Command:**
```bash
# Check sync status
python scripts/model_registry.py status

# Auto-register pending models
python scripts/model_registry.py register --auto
```

---

## Canonical Training Window

### Established: November 8, 2025

**Rationale:** Ablation study (148 games, 2022-2024 holdout) showed:
- Scenario B (2018-2024 + H=4 decay on 2015-2017): MAE 47.77 yards
- Scenario D (2015-2024 uniform): MAE 47.78 yards (equivalent)

**Decision:** Use simpler approach (uniform 2015-2024) for brms models where weight implementation is complex.

### Training Windows by Model

| Model | Training Window | Weights | Rationale |
|-------|-----------------|---------|-----------|
| **State-Space v2** | 2018-2024 + 2015-2017 decay | H=4 exponential | Requires explicit weights for state evolution |
| **Hierarchical v2** | 2015-2024 uniform | Equal | Equivalent to decay, simpler in brms |
| **Informative Priors v2** | 2015-2024 uniform | Equal | Priors encode domain knowledge |
| **XGBoost v2** | 2018-2024 | Equal | Gradient boosting less sensitive to old data |

### Exponential Decay Formula

```
w(s; t, H) = 0.5 ^ ((t - s) / H)

where:
  s = season year
  t = evaluation year (2025)
  H = half-life parameter (4 years)
```

**Example (H=4, t=2025):**
- 2024: w = 0.5^((2025-2024)/4) = 0.5^0.25 = 0.841
- 2023: w = 0.5^((2025-2023)/4) = 0.5^0.50 = 0.707
- 2018: w = 0.5^((2025-2018)/4) = 0.5^1.75 = 0.297
- 2017: w = 0.5^((2025-2017)/4) = 0.5^2.00 = 0.250
- 2015: w = 0.5^((2025-2015)/4) = 0.5^2.50 = 0.177

---

## Parallel Training

### GNU Parallel Implementation

**Script:** `scripts/train_all_models_parallel.sh`

```bash
#!/bin/bash
# Train all production models in parallel using GNU parallel

set -euo pipefail

# Create training manifest
cat > /tmp/training_jobs.txt <<EOF
Rscript R/state_space/train_state_space_v2_canonical.R > logs/state_space_v2.log 2>&1
Rscript R/train_hierarchical_v2_canonical.R > logs/hierarchical_v2.log 2>&1
Rscript R/train_informative_priors_v2_canonical.R > logs/informative_priors_v2.log 2>&1
python py/models/train_xgboost_canonical.py > logs/xgboost_v2.log 2>&1
EOF

# Run in parallel (4 jobs, show progress)
echo "Starting parallel training (4 jobs)..."
parallel --jobs 4 --eta --joblog logs/parallel_training.log < /tmp/training_jobs.txt

# Check for failures
if [ $? -ne 0 ]; then
    echo "❌ Training failed. Check logs in logs/"
    exit 1
fi

echo "✅ All models trained successfully"

# Validate convergence
python scripts/validate_all_models.py
```

### Resource Management

**Considerations:**
- **Memory:** Each R process uses ~4GB RAM → 4 jobs = 16GB total
- **CPU:** Each job uses 4 cores → 4 jobs = 16 cores total
- **Disk:** Simultaneous writes to `models/` directory

**Recommendations:**
- Mac M4: Run 2-3 jobs in parallel (8-12 cores)
- Windows RTX 4090: Run 4 jobs in parallel (16+ cores)
- Monitor memory usage with `htop` or Task Manager

---

## Model Versioning & Tracking

### Version Numbering Scheme

**Format:** `{stat_type}_{method}_v{major}.{minor}`

**Examples:**
- `passing_hierarchical_v2.0`: Major update (architecture change)
- `passing_hierarchical_v2.1`: Minor update (hyperparameter tuning)

**Major Version (v1 → v2):**
- Architecture change (formula, priors, features)
- Training window change
- Breaking API change

**Minor Version (v2.0 → v2.1):**
- Hyperparameter tuning
- Bug fixes
- Performance improvements (no architecture change)

### Git LFS Tracking

**Setup:**
```bash
# Initialize Git LFS
git lfs install

# Track model files
git lfs track "models/**/*.rds"
git lfs track "models/**/*.pkl"
git lfs track "models/**/*.pth"

# Commit .gitattributes
git add .gitattributes
git commit -m "chore: track model files with Git LFS"
```

**Workflow:**
```bash
# Train model
Rscript R/train_hierarchical_v2_canonical.R

# Add model to Git
git add models/bayesian/passing_hierarchical_v2.rds
git add models/bayesian/passing_hierarchical_v2_metadata.json

# Commit with descriptive message
git commit -m "feat: train hierarchical v2 model (MAE 45.2, calibration 88.7%)"

# Push to remote (Git LFS uploads large files)
git push origin main
```

**Cross-Machine Sync:**
```bash
# On Mac M4 (training machine)
git push origin main

# On Windows RTX 4090 (deployment machine)
git pull origin main
git lfs pull  # Download large model files
```

---

## Convergence Validation

### Automated Validation Script

**Script:** `scripts/validate_convergence.R`

```r
library(tidyverse)
library(brms)
library(posterior)

validate_brms_convergence <- function(model_path, max_rhat = 1.01, min_ess = 400) {
  cat("Validating:", model_path, "\n")

  # Load model
  fit <- readRDS(model_path)

  # Extract diagnostics
  summary <- summarise_draws(fit)

  # Check Rhat
  max_rhat_actual <- max(summary$rhat, na.rm = TRUE)
  rhat_pass <- max_rhat_actual < max_rhat

  # Check ESS
  min_ess_bulk <- min(summary$ess_bulk, na.rm = TRUE)
  min_ess_tail <- min(summary$ess_tail, na.rm = TRUE)
  ess_pass <- (min_ess_bulk >= min_ess && min_ess_tail >= min_ess)

  # Check divergences
  div <- nuts_params(fit, pars = "divergent__")
  n_div <- sum(subset(div, Value == 1)$Value)
  div_pass <- n_div < 10

  # Print results
  cat(sprintf("  Rhat: %.4f %s\n", max_rhat_actual, ifelse(rhat_pass, "✅", "❌")))
  cat(sprintf("  ESS bulk: %d %s\n", min_ess_bulk, ifelse(ess_pass, "✅", "❌")))
  cat(sprintf("  ESS tail: %d %s\n", min_ess_tail, ifelse(ess_pass, "✅", "❌")))
  cat(sprintf("  Divergences: %d %s\n", n_div, ifelse(div_pass, "✅", "❌")))

  # Overall result
  passed <- rhat_pass && ess_pass && div_pass

  list(
    passed = passed,
    max_rhat = max_rhat_actual,
    min_ess_bulk = min_ess_bulk,
    min_ess_tail = min_ess_tail,
    n_divergences = n_div
  )
}

# Validate all production models
models <- c(
  "models/bayesian/simplified_random_walk_v2.rds",
  "models/bayesian/passing_yards_hierarchical_v2.rds",
  "models/bayesian/passing_informative_priors_v2.rds"
)

results <- map(models, validate_brms_convergence)

# Fail if any model failed
if (!all(map_lgl(results, "passed"))) {
  stop("❌ Convergence validation failed")
}

cat("\n✅ All models passed convergence validation\n")
```

### Calibration Validation

**Script:** `py/models/validate_calibration.py`

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def validate_calibration(predictions, actuals, ci_lower, ci_upper, coverage_target=0.90):
    """Validate model calibration and accuracy."""

    # Calculate MAE
    mae = mean_absolute_error(actuals, predictions)

    # Calculate coverage (% of actuals within CI)
    coverage = np.mean((actuals >= ci_lower) & (actuals <= ci_upper))

    # Calculate interval width
    interval_width = np.mean(ci_upper - ci_lower)

    # Pass/fail criteria
    mae_pass = mae < 50  # Target: <50 yards
    coverage_pass = coverage >= 0.85  # Target: ≥85%

    print(f"MAE: {mae:.2f} yards {'✅' if mae_pass else '❌'}")
    print(f"Coverage: {coverage:.1%} {'✅' if coverage_pass else '❌'}")
    print(f"Interval width: {interval_width:.1f} yards")

    passed = mae_pass and coverage_pass

    return {
        'passed': passed,
        'mae': mae,
        'coverage': coverage,
        'interval_width': interval_width
    }

# Example usage
predictions = pd.read_csv("predictions.csv")
actuals = pd.read_csv("actuals.csv")

result = validate_calibration(
    predictions['mean'],
    actuals['yards'],
    predictions['ci_lower'],
    predictions['ci_upper']
)

if not result['passed']:
    raise ValueError("❌ Calibration validation failed")
```

---

## Performance Optimization

### Training Speed Optimizations

**1. Use cmdstanr backend (not rstan)**
```r
# cmdstanr is 2-3x faster than rstan
fit <- brm(..., backend = "cmdstanr")
```

**2. Parallelize chains**
```r
# Use all available cores
fit <- brm(..., chains = 4, cores = 4)
```

**3. Reduce iterations if convergence is good**
```r
# Default: 4000 iterations, 2000 warmup
# If Rhat <1.01 with 2000 iterations, use that
fit <- brm(..., iter = 2000, warmup = 1000)
```

**4. Use thinning to reduce memory**
```r
# Thin by 2 (keep every 2nd sample)
fit <- brm(..., thin = 2)  # 4000 iter → 2000 samples
```

**5. Use prior_sample_only for prior debugging**
```r
# Skip MCMC sampling (just prior simulation)
fit <- brm(..., prior_sample_only = TRUE)  # <1 minute
```

### Memory Optimizations

**1. Free intermediate objects**
```r
# Free memory after model training
rm(data, X_train, X_val)
gc()
```

**2. Use sparse matrices for large datasets**
```r
library(Matrix)
X_sparse <- sparse.model.matrix(formula, data)
```

**3. Stream data from database (don't load all)**
```r
# Use dbFetch() with batch_size instead of dbGetQuery()
res <- dbSendQuery(conn, query)
while (TRUE) {
  batch <- dbFetch(res, n = 10000)
  if (nrow(batch) == 0) break
  process_batch(batch)
}
dbClearResult(res)
```

---

## Usage Examples

### Example 1: Train Hierarchical v2 Model

```bash
# Set database password
export DB_PASSWORD="sicillionbillions"  # pragma: allowlist secret

# Train model (3-6 hours)
Rscript R/train_hierarchical_v2_canonical.R

# Output:
# ================================================================================
# HIERARCHICAL BAYESIAN MODEL v2 - CANONICAL TRAINING WINDOW
# ================================================================================
# Training: 2015-2024 (uniform, equivalent to 2018-2024 + decay)
# ================================================================================
#
# Step 1: Connecting to database...
# Step 2: Loading training data (2015-2024)...
# Loaded 12456 passing records
# Seasons: 2015-2024
# Unique QBs: 347
#
# Step 3: Preparing features...
# Step 4: Fitting brms model (4 chains, 4000 iterations)...
# [Chain 1] Iteration: 1 / 4000 [  0%]
# ...
# [Chain 4] Iteration: 4000 / 4000 [100%]
#
# Step 5: Checking convergence...
#   Max Rhat: 1.003 ✅
#   Min ESS bulk: 1842 ✅
#   Min ESS tail: 1654 ✅
#   Divergences: 3 ✅
#
# Step 6: Saving model...
# Model saved to: models/bayesian/passing_yards_hierarchical_v2.rds
# Metadata saved to: models/bayesian/passing_yards_hierarchical_v2_metadata.json
#
# ✅ Training complete
```

### Example 2: Train XGBoost v2 with HPO

```bash
# Train with hyperparameter optimization (60 min)
python py/models/train_xgboost_canonical.py \
    --training-window 2018-2024 \
    --n-trials 100 \
    --output models/xgboost/passing_xgboost_v2_canonical.pkl

# Output:
# Starting Optuna hyperparameter optimization (100 trials)...
# Trial 1: MAE = 42.3
# Trial 2: MAE = 39.8
# ...
# Trial 100: MAE = 38.9
#
# Best parameters:
#   max_depth: 6
#   learning_rate: 0.05
#   n_estimators: 500
#   ...
#
# Training final model with best parameters...
# Final MAE: 39.2 yards
#
# Saving model to models/xgboost/passing_xgboost_v2_canonical.pkl
# ✅ Training complete
```

### Example 3: Validate All Models

```bash
# Run convergence validation
Rscript scripts/validate_convergence.R

# Output:
# Validating: models/bayesian/simplified_random_walk_v2.rds
#   Rhat: 1.0028 ✅
#   ESS bulk: 1456 ✅
#   ESS tail: 1342 ✅
#   Divergences: 2 ✅
#
# Validating: models/bayesian/passing_yards_hierarchical_v2.rds
#   Rhat: 1.0034 ✅
#   ESS bulk: 1842 ✅
#   ESS tail: 1654 ✅
#   Divergences: 3 ✅
#
# Validating: models/bayesian/passing_informative_priors_v2.rds
#   Rhat: 1.0019 ✅
#   ESS bulk: 2103 ✅
#   ESS tail: 1987 ✅
#   Divergences: 0 ✅
#
# ✅ All models passed convergence validation
```

---

## Common Issues & Troubleshooting

### Issue 1: "Rhat > 1.01" (Convergence Failure)

**Symptom:**
```
❌ Convergence failed: max Rhat = 1.045
```

**Causes:**
- Insufficient iterations
- Poor initialization
- Multimodal posterior
- Misspecified model

**Solutions:**
```r
# 1. Increase iterations
fit <- brm(..., iter = 8000, warmup = 4000)  # Double from 4000/2000

# 2. Increase chains
fit <- brm(..., chains = 8)  # More chains for better diagnostics

# 3. Adjust adapt_delta
fit <- brm(..., control = list(adapt_delta = 0.99))  # Increase from 0.95

# 4. Check for collinearity
cor(model.matrix(formula, data))  # High correlations? Remove features

# 5. Reparameterize (non-centered)
# Before: (1 | player_id)
# After: player_effect[player_id] ~ normal(0, sigma_player)
```

### Issue 2: "ESS < 400" (Low Effective Sample Size)

**Symptom:**
```
❌ ESS too low: bulk=285, tail=192
```

**Causes:**
- High autocorrelation
- Slow mixing chains
- Complex posterior geometry

**Solutions:**
```r
# 1. Increase thinning
fit <- brm(..., thin = 5)  # Keep every 5th sample

# 2. Increase total iterations
fit <- brm(..., iter = 10000)  # More samples → higher ESS

# 3. Simplify model
# Remove complex interactions or random effects

# 4. Use informative priors
# Priors reduce posterior variance → better mixing
```

### Issue 3: "Divergences > 10"

**Symptom:**
```
❌ Divergences: 47
```

**Causes:**
- Stiff posterior geometry
- Step size too large
- Poor parameterization

**Solutions:**
```r
# 1. Increase adapt_delta
fit <- brm(..., control = list(adapt_delta = 0.99))  # Default 0.8

# 2. Increase max_treedepth
fit <- brm(..., control = list(max_treedepth = 15))  # Default 10

# 3. Reparameterize (non-centered)
# This often solves divergence issues

# 4. Check prior-data conflict
prior_summary(fit)  # Are priors too strong?
```

### Issue 4: "OOM (Out of Memory)"

**Symptom:**
```
Error: cannot allocate vector of size 8.2 GB
```

**Causes:**
- Too many iterations
- Too many parameters
- Too much data loaded

**Solutions:**
```r
# 1. Reduce iterations/chains
fit <- brm(..., chains = 2, iter = 2000)  # Instead of 4 chains × 4000

# 2. Use thinning
fit <- brm(..., thin = 4)  # Keep every 4th sample

# 3. Free memory
rm(list = setdiff(ls(), c("fit", "data")))
gc()

# 4. Stream data from database
# Don't load entire dataset into RAM
```

### Issue 5: "Training takes >12 hours"

**Symptom:** Hierarchical model still running after 12 hours

**Causes:**
- Too many iterations
- Too many random effects
- Complex formula

**Solutions:**
```r
# 1. Check if converged early
# Stop training, load partial model, check Rhat
fit_partial <- readRDS(".../model.rds")  # brms saves periodically
summary(fit_partial)  # If Rhat <1.01, done early

# 2. Simplify random effects
# Before: (1 + is_home | player_id)
# After: (1 | player_id)

# 3. Use prior_sample_only first
# Verify model runs in <5 minutes
fit_prior <- brm(..., prior_sample_only = TRUE)

# 4. Reduce data size (for testing)
data_sample <- data %>% sample_frac(0.1)
fit_test <- brm(formula, data_sample, ...)
```

---

## Production Deployment Checklist

```
✅ Pre-Training Validation:
  [ ] Database schema verified (correct column names)
  [ ] Training window defined (2015-2024 or 2018+decay)
  [ ] Features validated (no data leakage)
  [ ] Priors specified (informative or weakly informative)
  [ ] MCMC configuration set (chains, iter, warmup)

✅ Training Execution:
  [ ] Database password set (DB_PASSWORD env var)
  [ ] Sufficient RAM available (16GB+ for brms)
  [ ] Sufficient disk space (5GB+ for models)
  [ ] Logs directory created (logs/)
  [ ] Training started with monitoring

✅ Post-Training Validation:
  [ ] Convergence diagnostics pass (Rhat <1.01, ESS >400)
  [ ] Divergences acceptable (<10)
  [ ] Calibration validated (coverage >85%)
  [ ] MAE meets target (<50 yards)
  [ ] Model file saved successfully (.rds or .pkl)
  [ ] Metadata generated (.json)

✅ Model Registry:
  [ ] Model registered to database (ml.model_metadata)
  [ ] YAML manifest updated (MODEL_REGISTRY.yaml)
  [ ] Git LFS tracking enabled
  [ ] Model committed to Git
  [ ] Production_ready flag set TRUE

✅ Integration Testing:
  [ ] Model loads successfully in ensemble
  [ ] Predictions generated without errors
  [ ] Prediction time <500ms (single)
  [ ] Batch prediction works (50+ games)
  [ ] R model integration (rpy2) operational

✅ Production Deployment:
  [ ] Ensemble health check passes
  [ ] A/B test configuration created
  [ ] Monitoring alerts configured
  [ ] Rollback plan documented
  [ ] Team notified of deployment
```

---

## Summary

The **Model Training Pipeline** provides:

✅ **Multi-Language Support**: R (Bayesian) + Python (XGBoost)
✅ **Rigorous Validation**: MCMC diagnostics, calibration checks
✅ **Production-Ready**: Model registry, versioning, Git LFS
✅ **Performance**: 8-14 hours to train all models sequentially
✅ **Reliability**: Convergence gates, automated validation

**Key Files:**
- R training scripts: `R/train_*.R`
- Python training scripts: `py/models/train_*.py`
- Model registry: `scripts/model_registry.py`
- Validation scripts: `scripts/validate_*.py`

---

**Version:** 1.0
**Last Updated:** November 15, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
