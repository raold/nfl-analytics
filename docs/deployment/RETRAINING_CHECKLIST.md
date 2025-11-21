# Model Retraining Checklist - Weekly Maintenance

**Version:** 1.0
**Created:** November 16, 2025
**Purpose:** Enable weekly model retraining without expert supervision
**Frequency:** Weekly (Monday mornings) OR when MAE degrades >10%
**Skill Level:** Intermediate (R + Python environment required)
**Time Estimate:** 6-8 hours (mostly automated, monitoring required)

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Retraining Checklist](#pre-retraining-checklist)
3. [Retraining Order](#retraining-order)
4. [Validation Gates](#validation-gates)
5. [Model Registration](#model-registration)
6. [Deployment Triggers](#deployment-triggers)
7. [Deployment Process](#deployment-process)
8. [Rollback Procedure](#rollback-procedure)
9. [Monitoring Post-Deployment](#monitoring-post-deployment)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)
11. [Weekly Schedule Template](#weekly-schedule-template)
12. [Success Criteria](#success-criteria)

---

## Overview

### Why Retrain Weekly?

**NFL Season Dynamics:**
- Player performance changes week-to-week (injuries, form, matchups)
- Team strategies evolve as season progresses
- Coaching adjustments impact game plans
- Weather and venue conditions vary significantly

**Model Staleness:**
- Models trained on historical data degrade over 2-3 months
- New patterns emerge mid-season (e.g., playoff contenders adjusting)
- Bayesian priors need updating with latest observations
- XGBoost feature importance shifts as season progresses

**Expected Impact:**
- Fresh models maintain 90% calibration coverage (vs 85% stale)
- MAE improvement: 2-4 yards per retraining cycle
- ROI improvement: +0.5-1% vs stale models

### What Triggers Retraining?

**Schedule-Based (PRIMARY):**
- **Every Monday morning** during NFL season (Weeks 1-18)
- Start: 9:00 AM (training runs during work hours)
- Complete: Tuesday 4:00 PM (deployed before Week N+1 games)

**Performance-Based (EMERGENCY):**
- MAE degradation >10% vs baseline (e.g., 38 MAE → 42 MAE)
- Calibration coverage drops <85% (90% target, 88-92% acceptable range)
- Sharpe ratio drops >20% (2.2 baseline → 1.76)
- Win rate drops >3 percentage points (60% → 57%)

**Data-Based (AUTOMATIC):**
- nflfastR ingests new week's data (automatic trigger)
- Database row count increases >500 plays (1 full week)
- New players added to roster (mid-season trades)

### Who Can Execute?

**Minimum Requirements:**
- ✅ R environment configured (brms, cmdstanr, tidyverse)
- ✅ Python environment configured (uv venv, scikit-learn, xgboost)
- ✅ PostgreSQL access (localhost:5544, credentials in .env)
- ✅ Docker services running (6/6 operational)
- ✅ 10GB+ disk space available (models/ directory)
- ✅ Intermediate skill level (can read R/Python error messages)

**Optional (Nice to Have):**
- Git LFS configured (for model versioning)
- Grafana access (for monitoring)
- Slack/email alerts (for notifications)

**Expert Support Required If:**
- MCMC convergence fails (Rhat > 1.05, ESS < 300)
- R model integration errors (rpy2 failures)
- Multi-season validation fails all years
- Production deployment errors (API crashes)

### Time Estimate Breakdown

| Phase | Duration | Can Run Unattended? | Notes |
|-------|----------|---------------------|-------|
| Pre-retraining checks | 30 min | ❌ Manual | Verify environment, backup models |
| State-Space v2 training | 2-3 hours | ✅ Yes | Monitor for convergence warnings |
| Hierarchical v2 training | 4-6 hours | ✅ Yes | Longest step, run overnight if needed |
| Informative Priors v2 training | 3-4 hours | ✅ Yes | Can overlap with Hierarchical |
| XGBoost v4 training | 15-20 min | ✅ Yes | Fast, CPU-only |
| Temperature Scaler v6 training | 10-15 min | ✅ Yes | Fast, requires ensemble predictions |
| Validation gates | 1-2 hours | ❌ Manual | Review diagnostics, multi-season tests |
| Model registration | 30 min | ✅ Yes | Automated via script |
| Deployment decision | 30 min | ❌ Manual | Review metrics, approve deployment |
| **TOTAL** | **6-8 hours** | **60% automated** | **40% requires monitoring** |

**Optimization Tips:**
- Run Hierarchical v2 + Informative Priors v2 in parallel (save 3-4 hours)
- Cache nflfastR data (avoid re-downloading each week)
- Use `screen` or `tmux` to keep training sessions alive
- Schedule overnight for long-running models

---

## Pre-Retraining Checklist

**Time:** 30 minutes
**Frequency:** Every retraining cycle
**Owner:** Whoever is running retraining (could be automated)

### 1. Docker Services Health

**Requirement:** All 6 services must be operational before retraining.

```bash
# Check Docker services
docker compose ps

# Expected output:
# NAME                    STATUS
# nfl-analytics-pg-1      Up (healthy)
# nfl-analytics-redis-1   Up (healthy)
# nfl-analytics-fastapi-1 Up (healthy)
# nfl-analytics-streamlit-1 Up (healthy)
# prometheus              Up
# grafana                 Up
```

**Validation:**
- [ ] PostgreSQL: Port 5544 responding
- [ ] Redis: Port 6379 responding
- [ ] FastAPI: http://localhost:8000/health returns `{"status": "healthy"}`
- [ ] Streamlit: http://localhost:8501 accessible
- [ ] Prometheus: http://localhost:9090 accessible
- [ ] Grafana: http://localhost:3000 accessible

**If services are down:**
```bash
# Restart all services
docker compose restart

# Wait 30 seconds for health checks
sleep 30

# Verify health
docker compose ps
```

### 2. Database Up-to-Date

**Requirement:** Latest nflfastR data ingested, current week completed.

```bash
# Check latest game in database
psql postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01 \
  -c "SELECT MAX(season) as latest_season, MAX(week) as latest_week
      FROM games
      WHERE season = 2025;"

# Expected output (example for Week 11):
# latest_season | latest_week
# --------------+-------------
#          2025 |          11
```

**Validation:**
- [ ] `latest_season` matches current NFL season (2025 as of Nov 2025)
- [ ] `latest_week` matches completed week (check NFL.com for schedule)
- [ ] At least 500+ new plays ingested since last retraining

**If data is stale:**
```bash
# Ingest latest nflfastR data
Rscript --vanilla data/ingest_schedules.R
Rscript --vanilla data/ingest_pbp.R

# Refresh materialized views
psql postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
psql postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW mart.player_game_stats;"
```

### 3. Disk Space Available

**Requirement:** At least 10GB free space in `models/` directory.

```bash
# Check disk space
df -h /Users/dro/rice/nfl-analytics/models

# Expected output:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/disk1s1   500G  450G   50G  90% /

# Check models/ directory size
du -sh /Users/dro/rice/nfl-analytics/models

# Expected output:
# 12G   /Users/dro/rice/nfl-analytics/models
```

**Validation:**
- [ ] At least 10GB free space on disk
- [ ] `models/` directory accessible and writable
- [ ] No corrupted model files (check file sizes > 0 bytes)

**If disk space is low:**
```bash
# Archive old models to external storage
mkdir -p ~/nfl_models_archive_$(date +%Y%m%d)
cp -r models/ ~/nfl_models_archive_$(date +%Y%m%d)/

# Delete old experimental models (keep only production)
# CAREFUL: Only delete if you have backups!
# rm -rf models/experimental/
# rm -rf models/deprecated/
```

### 4. Git LFS Configured

**Requirement:** Large model files tracked by Git LFS (not Git).

```bash
# Check Git LFS status
git lfs status

# Expected output:
# On branch main
# Git LFS objects to be committed:
#   (none)
# Git LFS objects not staged for commit:
#   (none)

# Verify .rds and .pkl files tracked by LFS
cat .gitattributes | grep -E "(\.rds|\.pkl)"

# Expected output:
# *.rds filter=lfs diff=lfs merge=lfs -text
# *.pkl filter=lfs diff=lfs merge=lfs -text
```

**Validation:**
- [ ] Git LFS installed (`git lfs version` works)
- [ ] `.rds` and `.pkl` files tracked by LFS
- [ ] Git LFS storage not full (check GitHub LFS quota)

**If Git LFS not configured:**
```bash
# Install Git LFS
brew install git-lfs  # macOS
# OR
sudo apt-get install git-lfs  # Linux

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "*.rds"
git lfs track "*.pkl"
git lfs track "*.pth"

# Commit .gitattributes
git add .gitattributes
git commit -m "chore: configure Git LFS for model files"
```

### 5. Backup Current Models

**Requirement:** Backup current production models before retraining.

```bash
# Create backup directory with timestamp
BACKUP_DIR="models_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Copy current production models
cp models/bayesian/state_space_v2.rds "$BACKUP_DIR/"
cp models/bayesian/passing_yards_hierarchical_v2.rds "$BACKUP_DIR/"
cp models/bayesian/passing_informative_priors_v2.rds "$BACKUP_DIR/"
cp models/xgboost/xgboost_v4_ridge.pkl "$BACKUP_DIR/"
cp models/calibration/temperature_scaler_v6.pkl "$BACKUP_DIR/"

# Verify backup
ls -lh "$BACKUP_DIR/"

# Expected output (example):
# -rw-r--r-- 1 dro staff  45M Nov 16 09:00 state_space_v2.rds
# -rw-r--r-- 1 dro staff 128M Nov 16 09:00 passing_yards_hierarchical_v2.rds
# -rw-r--r-- 1 dro staff 115M Nov 16 09:00 passing_informative_priors_v2.rds
# -rw-r--r-- 1 dro staff   2M Nov 16 09:00 xgboost_v4_ridge.pkl
# -rw-r--r-- 1 dro staff 512K Nov 16 09:00 temperature_scaler_v6.pkl
```

**Validation:**
- [ ] All 5 production models backed up
- [ ] File sizes match expected ranges (R models 45-130 MB, XGBoost 1-3 MB)
- [ ] Backup directory named with timestamp for easy rollback

**If backup fails:**
- Manually copy files to external drive
- Take note of backup location for rollback
- DO NOT proceed with retraining until backup is verified

### 6. No Active Backtests Running

**Requirement:** No long-running backtests or training jobs active.

```bash
# Check for Python processes running backtests
ps aux | grep -i "backtest\|historical_validation"

# Check for R processes running training
ps aux | grep -E "(Rscript|R CMD)"

# Check for GPU processes (if applicable)
nvidia-smi  # Only on GPU machines
```

**Validation:**
- [ ] No `historical_validation` scripts running
- [ ] No `train_*.R` or `train_*.py` processes active
- [ ] No GPU memory allocated to old jobs

**If processes found:**
```bash
# Identify process ID (PID)
ps aux | grep backtest

# Kill process (replace <PID> with actual process ID)
kill -9 <PID>

# Verify process terminated
ps aux | grep <PID>
```

---

## Retraining Order

**CRITICAL:** Dependencies matter. Train models in this exact order.

**Rationale for Order:**
1. **State-Space v2 first**: Provides baseline random walk predictions
2. **Hierarchical v2 second**: Most complex, longest training (4-6 hours)
3. **Informative Priors v2 third**: Can run in parallel with Hierarchical v2
4. **XGBoost v4 fourth**: Requires ridge residuals from ensemble (fast, 15-20 min)
5. **Temperature Scaler v6 last**: Requires all 4 models for ensemble predictions

**Parallel Training (OPTIONAL):**
- Hierarchical v2 + Informative Priors v2 can run in parallel (save 3-4 hours)
- Requires 8+ CPU cores and 32GB+ RAM
- Use `screen` or `tmux` to manage separate sessions

---

### Step 1: State-Space Random Walk v2

**Purpose:** Random walk model for QB passing yards with exponential decay weighting.

**Training Window:** 2018-2024 (full weight) + 2015-2017 (H=4 exponential decay)

**Expected Runtime:** 2-3 hours (4 chains × 6000 iterations)

**Command:**
```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Activate R environment (if using renv)
# Rscript -e 'renv::restore()'

# Set environment variable for database password
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)

# Run training script
Rscript R/state_space/train_state_space_v2_canonical.R
```

**Expected Output:**
```
======================================================================
State-Space Model v2: Canonical Training Window
======================================================================
Training: 2018-2024 (full weight) + 2015-2017 (H=4 decay)
Formula: w(s; 2025, 4) = 0.5^((2025-s)/4)
======================================================================

STEP 1: Loading player game data from database...
  ✓ Loaded 8,234 player-game records
  ✓ Unique players: 287
  ✓ Seasons: 2015-2024

STEP 2: Apply exponential decay weights...
  ✓ 2015: weight = 0.0625 (decay)
  ✓ 2016: weight = 0.0884 (decay)
  ✓ 2017: weight = 0.1250 (decay)
  ✓ 2018-2024: weight = 1.0000 (full)
  ✓ Effective sample size: 7,458

STEP 3: Prepare data for Stan...
  ✓ Feature matrix: 8,234 × 12
  ✓ Response: log(passing_yards)

STEP 4: Compile Stan model...
  ✓ Compiled successfully

STEP 5: Run MCMC sampling...
  Chain 1: Iteration 1000/6000 (warmup)
  Chain 1: Iteration 3000/6000 (warmup complete)
  Chain 1: Iteration 6000/6000 (sampling complete)
  Chain 2: Iteration 1000/6000 (warmup)
  ...
  ✓ Sampling complete in 142 minutes

STEP 6: Convergence diagnostics...
  Max Rhat: 1.003 (threshold: < 1.01) ✅
  Min ESS: 1,245 (threshold: > 400) ✅
  Divergences: 0

✅ CONVERGENCE SUCCESS!

STEP 7: Save model...
  ✓ Model saved to models/bayesian/state_space_v2.rds
  ✓ Model size: 45.3 MB

======================================================================
✅ STATE-SPACE MODEL v2 TRAINING COMPLETE
======================================================================

Summary:
  - Training window: 2015-2024 (8,234 games, effective 7,458)
  - Training time: 142.3 minutes
  - Max Rhat: 1.003
  - Min ESS: 1,245
  - Model file: models/bayesian/state_space_v2.rds
  - Model size: 45.3 MB

Next: Train Hierarchical v2
```

**Validation Checklist:**
- [ ] Training completed without errors
- [ ] Max Rhat < 1.01 (convergence achieved)
- [ ] Min ESS > 400 (sufficient effective sample size)
- [ ] Divergences = 0 (no sampling issues)
- [ ] Model file created: `models/bayesian/state_space_v2.rds`
- [ ] Model size 40-50 MB (reasonable range)

**If Validation Fails:**

**Issue: Rhat > 1.01 (convergence failure)**
```R
# Increase iterations and warmup
# Edit R/state_space/train_state_space_v2_canonical.R:
# Change: iter = 6000, warmup = 3000
# To:     iter = 10000, warmup = 5000

# Increase adapt_delta (slower but more robust)
# Change: adapt_delta = 0.99
# To:     adapt_delta = 0.995
```

**Issue: Divergences > 10**
```R
# Try non-centered parameterization
# Edit Stan model to use Matt trick (non-centered)
# OR increase adapt_delta to 0.999

# Consult Stan manual:
# https://mc-stan.org/docs/stan-users-guide/
```

**Issue: Model file not created**
```bash
# Check error logs
cat R/state_space/train_state_space_v2_canonical.R.log

# Check disk space
df -h /Users/dro/rice/nfl-analytics/models

# Verify write permissions
ls -ld models/bayesian/
```

**Issue: Training takes >4 hours (too slow)**
```R
# Reduce iterations (ONLY if convergence already good)
# Change: iter = 6000
# To:     iter = 4000

# Use fewer threads per chain
# Change: threads = threading(2)
# To:     threads = threading(1)
```

---

### Step 2: Hierarchical Bayesian v2

**Purpose:** Hierarchical model with player/team/opponent random effects.

**Training Window:** 2015-2025 (uniform weights, equivalent to canonical)

**Expected Runtime:** 4-6 hours (4 chains × 6000 iterations, complex hierarchical structure)

**Command:**
```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Set environment variable for database password
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)

# Run training script
Rscript R/train_hierarchical_v2_production.R
```

**Expected Output:**
```
================================================================================
HIERARCHICAL BAYESIAN MODEL v2 - CANONICAL TRAINING WINDOW
================================================================================
Training: 2015-2025 (uniform, equivalent to 2018-2025 + decay)
================================================================================

Step 1: Connecting to database...
Step 2: Loading training data (2015-2025)...
Loaded 9,123 passing records
Seasons: 2015-2025
Unique QBs: 312

Step 3: Engineering features...
Features engineered. 9,098 records ready for modeling

Step 4: Training hierarchical Bayesian model...
This will take 30-40 minutes (increased iterations for better convergence)...

Chain 1: Iteration 1000/6000 (warmup)
Chain 1: Iteration 3000/6000 (warmup complete)
Chain 1: Iteration 6000/6000 (sampling complete)
Chain 2: Iteration 1000/6000 (warmup)
...
✓ Training complete in 285.7 minutes

Step 5: Checking convergence...
  Max Rhat: 1.0087 (threshold: < 1.01)
  Min ESS: 512 (threshold: > 400)

✅ CONVERGENCE SUCCESS!

Step 6: Saving model...
✓ Model saved to models/bayesian/passing_yards_hierarchical_v2.rds
✓ Model size: 128.4 MB

================================================================================
✅ HIERARCHICAL BAYESIAN MODEL v2 TRAINING COMPLETE
================================================================================

Summary:
  - Training window: 2015-2025 (uniform, 9,098 games)
  - Training time: 285.7 minutes
  - Max Rhat: 1.0087
  - Min ESS: 512
  - Model file: models/bayesian/passing_yards_hierarchical_v2.rds
  - Model size: 128.4 MB

Next: Generate 2025 predictions for validation
```

**Validation Checklist:**
- [ ] Training completed without errors
- [ ] Max Rhat < 1.01 (convergence achieved)
- [ ] Min ESS > 400 (sufficient effective sample size)
- [ ] Divergences < 10 (acceptable range)
- [ ] Model file created: `models/bayesian/passing_yards_hierarchical_v2.rds`
- [ ] Model size 100-150 MB (reasonable range)

**If Validation Fails:**

**Issue: Rhat 1.01-1.05 (marginal convergence)**
```R
# Increase iterations
# Edit R/train_hierarchical_v2_production.R:
# Change: iter = 6000, warmup = 3000
# To:     iter = 8000, warmup = 4000

# Increase adapt_delta
# Change: adapt_delta = 0.99
# To:     adapt_delta = 0.995
```

**Issue: ESS 300-400 (marginal but acceptable)**
```
# This is acceptable for production if:
# 1. Rhat < 1.01 (convergence good)
# 2. No divergences
# 3. Predictions look reasonable

# If you want to improve ESS:
# - Increase iterations to 10000
# - Use non-centered parameterization
# - Reparameterize priors
```

**Issue: Divergences 10-50 (concerning)**
```R
# Increase adapt_delta to 0.995 or 0.999
# Change: adapt_delta = 0.99
# To:     adapt_delta = 0.999

# If still diverging, try non-centered parameterization
# Consult: https://mc-stan.org/docs/stan-users-guide/reparameterization.html
```

**Issue: Training takes >8 hours (too slow)**
```R
# Use parallel chains on separate cores
# Change: cores = 4, threads = threading(2)
# To:     cores = 4, threads = threading(1)

# OR reduce iterations (ONLY if convergence already good)
# Change: iter = 6000
# To:     iter = 5000
```

---

### Step 3: Informative Priors v2

**Purpose:** Hierarchical model with empirical Bayes priors estimated from 2015-2019 data.

**Training Window:** 2015-2025 (uniform weights, same as Hierarchical v2)

**Expected Runtime:** 3-4 hours (4 chains × 6000 iterations)

**Command:**
```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Set environment variable for database password
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)

# Run training script
Rscript R/train_informative_priors_v2_production.R
```

**Expected Output:**
```
================================================================================
INFORMATIVE PRIORS MODEL v2 - CANONICAL TRAINING WINDOW
================================================================================
Training: 2015-2025 (uniform, equivalent to 2018-2025 + decay)
================================================================================

Step 1: Connecting to database...
Step 2: Loading training data (2015-2025)...
Loaded 9,123 passing records
Seasons: 2015-2025
Unique QBs: 312

Step 3: Estimating empirical Bayes priors...
  Prior estimation dataset: 4,512 games
  Intercept prior: N(5.28, 0.31)
  log_attempts prior: N(0.782, 0.1)
  home effect prior: N(0.018, 0.01)

Step 4: Engineering features...
Features engineered. 9,098 records ready for modeling

Step 5: Training model with informative priors...
This will take 35-45 minutes (increased iterations for better convergence)...

Chain 1: Iteration 1000/6000 (warmup)
Chain 1: Iteration 3000/6000 (warmup complete)
Chain 1: Iteration 6000/6000 (sampling complete)
Chain 2: Iteration 1000/6000 (warmup)
...
✓ Training complete in 218.4 minutes

Step 6: Checking convergence...
  Max Rhat: 1.0065 (threshold: < 1.01)
  Min ESS: 687 (threshold: > 400)

✅ CONVERGENCE SUCCESS!

Step 7: Saving model...
✓ Model saved to models/bayesian/passing_informative_priors_v2.rds
✓ Model size: 115.2 MB

================================================================================
✅ INFORMATIVE PRIORS MODEL v2 TRAINING COMPLETE
================================================================================

Summary:
  - Training window: 2015-2025 (uniform, 9,098 games)
  - Prior estimation: 2015-2019 (4,512 games)
  - Training time: 218.4 minutes
  - Max Rhat: 1.0065
  - Min ESS: 687
  - Model file: models/bayesian/passing_informative_priors_v2.rds
  - Model size: 115.2 MB

Next: Generate 2025 predictions for validation
```

**Validation Checklist:**
- [ ] Training completed without errors
- [ ] Max Rhat < 1.01 (convergence achieved)
- [ ] Min ESS > 400 (sufficient effective sample size)
- [ ] Divergences < 10 (acceptable range)
- [ ] Empirical priors estimated successfully (see output)
- [ ] Model file created: `models/bayesian/passing_informative_priors_v2.rds`
- [ ] Model size 100-130 MB (reasonable range)

**If Validation Fails:**

Same troubleshooting as Hierarchical v2 (see Step 2 above).

---

### Step 4: XGBoost v4 Ridge Residuals

**Purpose:** XGBoost model trained on ridge regression residuals (not raw passing yards).

**Training Window:** 2018-2024 (aligned with State-Space v2)

**Expected Runtime:** 15-20 minutes (CPU-only, fast gradient boosting)

**Command:**
```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Activate Python virtual environment
source .venv/bin/activate

# Set environment variable for database password
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)

# Run training script
python scripts/xgb_v3_2_ridge_residuals.py
```

**Expected Output:**
```
======================================
XGBoost v4 Ridge Residuals Training
======================================
Training window: 2018-2024
Feature set: 157 features (asof-safe)
Target: Ridge regression residuals

Step 1: Loading training data...
  ✓ Loaded 8,234 player-games
  ✓ Features: 157
  ✓ Train: 6,587 games (2018-2023)
  ✓ Test: 1,647 games (2024)

Step 2: Training Ridge regression baseline...
  ✓ Ridge alpha: 1.0
  ✓ Train MAE: 38.2 yards
  ✓ Test MAE: 39.7 yards

Step 3: Computing residuals...
  ✓ Residual mean: 0.02 yards (centered)
  ✓ Residual std: 15.3 yards

Step 4: Training XGBoost on residuals...
  [0]  train-mae: 12.45  test-mae: 14.87
  [50] train-mae: 3.21   test-mae: 4.65
  [100] train-mae: 2.87  test-mae: 4.52
  [150] train-mae: 2.65  test-mae: 4.48
  [200] train-mae: 2.51  test-mae: 4.46
  ✓ Best iteration: 178 (early stopping)

Step 5: Final performance...
  ✓ Ridge baseline MAE: 39.7 yards
  ✓ XGBoost residual MAE: 4.48 yards
  ✓ Combined MAE: 35.2 yards (4.5 yard improvement)

Step 6: Multi-season validation...
  ✓ 2022: 5.06 MAE (+87.4% improvement, p<0.000001)
  ✓ 2023: 4.76 MAE (+88.2% improvement, p<0.000001)
  ✓ 2024: 4.27 MAE (+89.4% improvement, p<0.000001)
  ✓ Mean: 4.70 ± 0.40 MAE (consistent across years)

Step 7: Saving models...
  ✓ Ridge model: models/ridge_residuals_v7.pkl
  ✓ XGBoost model: models/xgb_ridge_residuals_v7.json
  ✓ Feature scaler: models/feature_scaler_v7.pkl

======================================
✅ XGBoost v4 Training Complete
======================================

Training time: 17.3 minutes
Test MAE: 4.48 yards (residuals)
Combined MAE: 35.2 yards (ridge + xgboost)
Multi-season MAE: 4.70 ± 0.40 yards

Next: Train Temperature Scaler v6
```

**Validation Checklist:**
- [ ] Training completed without errors
- [ ] Ridge baseline MAE: 38-42 yards (reasonable)
- [ ] XGBoost residual MAE: 4-6 yards (expected range)
- [ ] Multi-season consistency: std < 2 yards (CRITICAL)
- [ ] All years p < 0.05 (statistically significant)
- [ ] 3 model files created: ridge, xgboost, scaler
- [ ] File sizes reasonable (1-3 MB each)

**If Validation Fails:**

**Issue: Multi-season consistency fails (std > 2 yards)**
```python
# Check data distribution by year
# Large std indicates distribution shift or leakage

# Verify temporal splits are correct (no future data leakage)
# Rerun with stricter asof joins

# Consult data leakage detector:
python py/utils/data_leakage_detector.py --check-split train.csv test.csv
```

**Issue: XGBoost residual MAE > 8 yards (too high)**
```python
# Increase max_depth or num_boost_round
# Edit scripts/xgb_v3_2_ridge_residuals.py:
# Change: max_depth = 6, num_boost_round = 200
# To:     max_depth = 8, num_boost_round = 300

# Check feature quality (missing values, outliers)
# Verify ridge regression is working (should get 38-42 MAE)
```

**Issue: Combined MAE > 40 yards (worse than ridge alone)**
```python
# XGBoost is overfitting to residuals
# Reduce max_depth and learning_rate
# Change: max_depth = 6, eta = 0.1
# To:     max_depth = 4, eta = 0.05

# Increase min_child_weight (regularization)
# Add: min_child_weight = 5
```

---

### Step 5: Temperature Scaler v6

**Purpose:** Isotonic regression scaler for ensemble calibration (90% coverage target).

**Training Window:** 2022-2024 (historical ensemble predictions)

**Expected Runtime:** 10-15 minutes (fast, requires ensemble predictions cached)

**Command:**
```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Activate Python virtual environment
source .venv/bin/activate

# Set environment variable for database password
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)

# Run training script
python scripts/train_temperature_scaler_v6.py
```

**Expected Output:**
```
================================================================================
TEMPERATURE SCALER v6.0 TRAINING PIPELINE
================================================================================

Initializing ensemble v6.0 production...
Loaded ensemble weights: {'state_space': 0.30, 'hierarchical': 0.30, 'xgboost': 0.25, 'informative_priors': 0.15}
Current fallback temperature: 0.5534

================================================================================
GENERATING ENSEMBLE v6.0 PREDICTIONS (2022-2024)
================================================================================

Loaded 1,987 player-games from database
  Games: 148
  Players: 287
  Avg actual yards: 245.3

Generating ensemble predictions for 1,987 player-games...
⚠️  This will take 30-45 minutes (R model integration is slow)

  Progress: 100/1987 (5.0%) | Valid: 98 | Failed: 2
  Progress: 200/1987 (10.1%) | Valid: 197 | Failed: 3
  ...
  Progress: 1987/1987 (100.0%) | Valid: 1952 | Failed: 35

✅ Predictions complete!
  Valid predictions: 1,952
  Failed predictions: 35

✅ Cached predictions to analysis/results/ensemble_v6_predictions_2022_2024.csv

================================================================================
TRAINING TEMPERATURE SCALER (ISOTONIC REGRESSION)
================================================================================

================================================================================
CALIBRATION REPORT
================================================================================
Optimal Temperature: T = 0.5534
Target Coverage: 90.0%

Before Temperature Scaling:
  Coverage: 97.3%
  Mean Interval Width: 85.2 yards

After Temperature Scaling:
  Coverage: 89.7%
  Mean Interval Width: 63.4 yards

MAE: 47.8 yards (unchanged by scaling)
Samples: 1,952

✅ Temperature scaling calibrated successfully!

Generating calibration plot...
✅ Calibration plot saved to analysis/results/temperature_scaler_v6_calibration_2022_2024.png

================================================================================
SAVING TEMPERATURE SCALER
================================================================================

✅ Temperature scaler saved to models/calibration/temperature_scaler_v6.pkl
✅ Metadata saved to models/calibration/temperature_scaler_v6_metadata.json

================================================================================
TEMPERATURE SCALER v6.0 TRAINING COMPLETE
================================================================================
Optimal Temperature: T = 0.5534
Coverage Improvement: 97.3% → 89.7%
Interval Width Reduction: 25.6%
Target Coverage: 90.0%
Calibration Status: ✅ PASS

✅ All steps complete!
================================================================================
```

**Validation Checklist:**
- [ ] Ensemble predictions generated (1,900-2,000 player-games)
- [ ] Failed predictions < 5% (35/1987 = 1.8% acceptable)
- [ ] Optimal temperature: 0.45-0.65 (reasonable range)
- [ ] Coverage after scaling: 88-92% (90% target ±2%)
- [ ] Interval width reduction: 20-30% (calibration improvement)
- [ ] Calibration status: PASS
- [ ] 2 files created: scaler pkl + metadata json
- [ ] Calibration plot saved

**If Validation Fails:**

**Issue: Coverage after scaling < 85% or > 95% (miscalibrated)**
```python
# Check ensemble predictions quality
# Load cached predictions
import pandas as pd
df = pd.read_csv('analysis/results/ensemble_v6_predictions_2022_2024.csv')

# Verify ensemble std is reasonable (not too narrow or wide)
print(df['pred_std'].describe())
# Expected: mean ~25-35 yards, std ~5-10 yards

# If pred_std too narrow (< 20), ensemble is overconfident
# If pred_std too wide (> 50), ensemble is underconfident
```

**Issue: Failed predictions > 10% (too many failures)**
```python
# Check R model integration logs
# Common causes:
# 1. rpy2 not configured correctly
# 2. R models not loaded in ensemble
# 3. Database connection failures

# Run ensemble health check
./scripts/check_ensemble_health.sh

# If R models failing, use Python-only ensemble (fallback)
# Edit py/ensemble/enhanced_ensemble_v6_production.py:
# Comment out R model imports
```

**Issue: Ensemble predictions take > 2 hours (too slow)**
```python
# Use cached predictions if available
# Check for: analysis/results/ensemble_v6_predictions_2022_2024.csv

# If cache doesn't exist, predictions will regenerate (slow)
# Speed up by:
# 1. Using fewer player-games (reduce date range)
# 2. Disabling R models temporarily (Python-only)
# 3. Running on faster machine
```

---

## Validation Gates

**Purpose:** Ensure retrained models meet production quality standards BEFORE deployment.

**MUST PASS:** All 4 gates required for auto-deployment. If any gate fails, manual review required.

---

### Gate 1: MCMC Convergence (R Models Only)

**Applies To:** State-Space v2, Hierarchical v2, Informative Priors v2

**Criteria:**
- [ ] Rhat < 1.01 for ALL parameters (convergence)
- [ ] ESS > 400 for ALL parameters (bulk and tail)
- [ ] Divergences < 10 (acceptable sampling quality)
- [ ] Trace plots show good mixing (visual inspection)

**How to Check:**

```R
# Load trained model
library(brms)
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")

# Check Rhat
library(posterior)
rhats <- rhat(model)
max_rhat <- max(rhats, na.rm = TRUE)
print(paste("Max Rhat:", round(max_rhat, 4)))

# Check ESS
ess_bulk <- ess_bulk(model)
ess_tail <- ess_tail(model)
min_ess_bulk <- min(ess_bulk, na.rm = TRUE)
min_ess_tail <- min(ess_tail, na.rm = TRUE)
print(paste("Min ESS (bulk):", round(min_ess_bulk, 0)))
print(paste("Min ESS (tail):", round(min_ess_tail, 0)))

# Check divergences
library(bayesplot)
np <- nuts_params(model)
num_divergences <- sum(np$Value[np$Parameter == "divergent__"])
print(paste("Divergences:", num_divergences))

# Plot trace plots (visual inspection)
mcmc_trace(model, pars = c("b_Intercept", "b_log_attempts", "b_is_home"))
```

**Passing Criteria:**
```
Max Rhat: 1.0065 ✅
Min ESS (bulk): 512 ✅
Min ESS (tail): 487 ✅
Divergences: 3 ✅
```

**Failing Criteria:**
```
Max Rhat: 1.0234 ❌ (> 1.01)
Min ESS (bulk): 287 ❌ (< 400)
Divergences: 47 ❌ (> 10)
```

**If Gate 1 Fails:**
- Increase iterations (iter = 10000, warmup = 5000)
- Increase adapt_delta (0.995 or 0.999)
- Try non-centered parameterization
- Consult Stan manual: https://mc-stan.org/docs/

---

### Gate 2: Performance Benchmarks

**Applies To:** All 5 models

**Criteria:**
- [ ] State-Space v2: 45-52 MAE (expected range on 2024 test set)
- [ ] Hierarchical v2: 30-45 MAE
- [ ] Informative Priors v2: 30-45 MAE
- [ ] XGBoost v4: 4-6 MAE (residuals only)
- [ ] Temperature Scaler: Coverage 88-92%

**How to Check:**

**R Models (State-Space, Hierarchical, Informative Priors):**
```R
# Generate predictions on 2024 test set
library(tidyverse)
library(DBI)
library(RPostgres)

# Load model
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")

# Load 2024 test data
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD")
)

test_data <- dbGetQuery(conn, "
  SELECT *
  FROM mart.player_game_stats
  WHERE season = 2024
    AND stat_category = 'passing'
    AND stat_attempts >= 10
")

dbDisconnect(conn)

# Generate predictions
preds <- predict(model, newdata = test_data, summary = FALSE)
pred_mean <- colMeans(preds)

# Calculate MAE
mae <- mean(abs(test_data$stat_yards - pred_mean))
print(paste("MAE:", round(mae, 2), "yards"))

# Expected range: 30-52 yards depending on model
```

**XGBoost v4:**
```python
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Load models
with open('models/ridge_residuals_v7.pkl', 'rb') as f:
    ridge = pickle.load(f)
with open('models/xgb_ridge_residuals_v7.json', 'rb') as f:
    xgb = pickle.load(f)

# Load 2024 test data
# (code to load test_X, test_y from database)

# Predict residuals
ridge_preds = ridge.predict(test_X)
xgb_residuals = xgb.predict(test_X_residual)

# Combined predictions
final_preds = ridge_preds + xgb_residuals

# Calculate MAE (residuals only)
mae_residuals = mean_absolute_error(test_y - ridge_preds, xgb_residuals)
print(f"XGBoost Residual MAE: {mae_residuals:.2f} yards")

# Expected range: 4-6 yards
```

**Temperature Scaler:**
```python
# Already validated during training
# Check metadata file
import json
with open('models/calibration/temperature_scaler_v6_metadata.json', 'r') as f:
    meta = json.load(f)

print(f"Coverage after scaling: {meta['coverage_after']:.1%}")
# Expected: 88-92%
```

**Passing Criteria:**
```
State-Space v2: 47.8 MAE ✅ (within 45-52 range)
Hierarchical v2: 38.2 MAE ✅ (within 30-45 range)
Informative Priors v2: 35.7 MAE ✅ (within 30-45 range)
XGBoost v4: 4.7 MAE ✅ (within 4-6 range)
Temperature Scaler: 89.7% coverage ✅ (within 88-92% range)
```

**Failing Criteria:**
```
State-Space v2: 58.3 MAE ❌ (> 52)
Hierarchical v2: 52.1 MAE ❌ (> 45)
XGBoost v4: 8.9 MAE ❌ (> 6)
Temperature Scaler: 82.3% coverage ❌ (< 88%)
```

**If Gate 2 Fails:**
- Check data quality (missing values, outliers)
- Verify training window is correct (2015-2025 for R, 2018-2024 for XGBoost)
- Review feature engineering (no leakage, correct asof joins)
- Compare to previous week's performance (gradual degradation expected)

---

### Gate 3: Multi-Season Consistency

**Applies To:** XGBoost v4 (CRITICAL for detecting data leakage)

**Criteria:**
- [ ] MAE std < 2 yards across 2022, 2023, 2024
- [ ] All years p < 0.05 for improvement vs baseline
- [ ] No year has MAE > 8 yards (outlier detection)
- [ ] Mean MAE 4-6 yards (expected range)

**How to Check:**

```bash
# Run multi-season validation script
cd /Users/dro/rice/nfl-analytics
python scripts/xgb_v4_multi_season_validation.py
```

**Expected Output:**
```
======================================
Multi-Season Validation: 2022-2024
======================================

2022 Validation:
  Test records: 583
  MAE: 5.06 yards
  RMSE: 6.74 yards
  Improvement: +87.40%
  p-value: 0.000001 ✅

2023 Validation:
  Test records: 599
  MAE: 4.76 yards
  RMSE: 6.21 yards
  Improvement: +88.15%
  p-value: 0.000001 ✅

2024 Validation:
  Test records: 612
  MAE: 4.27 yards
  RMSE: 5.93 yards
  Improvement: +89.38%
  p-value: 0.000001 ✅

======================================
Consistency Analysis
======================================

Mean MAE: 4.70 ± 0.40 yards ✅
All years p < 0.05: YES ✅
All years MAE < 8: YES ✅
Consistency: PASS ✅

All gates passed!
```

**Passing Criteria:**
```
Mean: 4.70 ± 0.40 yards ✅ (std = 0.40 < 2)
2022 p-value: 0.000001 ✅ (< 0.05)
2023 p-value: 0.000001 ✅ (< 0.05)
2024 p-value: 0.000001 ✅ (< 0.05)
Max MAE: 5.06 yards ✅ (< 8)
```

**Failing Criteria:**
```
Mean: 5.20 ± 3.80 yards ❌ (std = 3.80 > 2)
2022 MAE: 2.10 yards ⚠️  (suspiciously low - possible leakage)
2023 MAE: 4.50 yards ✅
2024 MAE: 9.00 yards ❌ (> 8, outlier)
```

**If Gate 3 Fails:**
- **Large std (> 2 yards):** Distribution shift or data leakage
  - Run data leakage detector: `python py/utils/data_leakage_detector.py`
  - Check temporal splits are correct (no future data)
  - Verify feature engineering is identical across years
- **Suspiciously low MAE (<3 yards):** Possible data leakage
  - Recheck asof joins (no lookahead)
  - Verify target variable not in features
  - Run automated leakage tests
- **Outlier year (MAE > 8 yards):** Model not generalizing
  - Check data quality for that year
  - Verify sufficient training data
  - Consider excluding that year if data issue confirmed

---

### Gate 4: Ensemble Integration

**Applies To:** All 4 models + Temperature Scaler

**Criteria:**
- [ ] All 4 models load successfully in ensemble
- [ ] Dry-run prediction completes without errors
- [ ] No R model integration errors (rpy2 working)
- [ ] Ensemble MAE < 40 yards (baseline v5.1: ~38 MAE)

**How to Check:**

```bash
# Run ensemble health check
cd /Users/dro/rice/nfl-analytics
./scripts/check_ensemble_health.sh

# Expected output:
# ========================================
# Ensemble Health Monitor
# ========================================
# Mode: --check
# Project: /Users/dro/rice/nfl-analytics
#
# Checking ensemble health...
#
# ✅ All 4 components loaded successfully
# ✅ R model integration working (rpy2)
# ✅ Temperature scaler loaded (T=0.5534)
# ✅ Ensemble weights valid (sum=1.0)
# ✅ Component drift within acceptable range
# ✅ Calibration coverage: 89.7% (target 90%)
#
# Ensemble health: PASS ✅
```

**Dry-Run Prediction Test:**
```bash
# Run dry-run backtest (10 games, fast)
python py/backtest/historical_validation_v6.py \
  --dry-run \
  --max-games 10 \
  --season 2024 \
  --week 11

# Expected output:
# Running dry-run ensemble backtest (10 games)...
#
# Game 1/10: CHI @ GB, Week 11
#   ✓ Predictions generated (12 players)
#   ✓ R models integrated successfully
#   ✓ Ensemble MAE: 37.2 yards
#
# Game 2/10: LAC @ CIN, Week 11
#   ✓ Predictions generated (14 players)
#   ✓ R models integrated successfully
#   ✓ Ensemble MAE: 35.8 yards
#
# ...
#
# Dry-run complete!
# Overall MAE: 36.5 yards ✅ (< 40 target)
# All R models loaded: YES ✅
# Integration errors: 0 ✅
```

**Passing Criteria:**
```
All 4 components loaded: YES ✅
R model integration: YES ✅
Temperature scaler loaded: YES ✅
Dry-run MAE: 36.5 yards ✅ (< 40)
Integration errors: 0 ✅
```

**Failing Criteria:**
```
R model integration: NO ❌ (rpy2 error)
Dry-run MAE: 52.3 yards ❌ (> 40)
Integration errors: 7 ❌ (> 0)
```

**If Gate 4 Fails:**

**Issue: R model integration errors (rpy2)**
```bash
# Check R environment
which R
R --version

# Check rpy2 installation
python -c "import rpy2; print(rpy2.__version__)"

# Reinstall rpy2 if needed
pip uninstall rpy2
pip install rpy2==3.5.14

# Set R_HOME environment variable
export R_HOME=/Library/Frameworks/R.framework/Resources
```

**Issue: Dry-run MAE > 40 yards (poor performance)**
```python
# Check individual component performance
# Run with --return-components flag
python py/backtest/historical_validation_v6.py \
  --dry-run \
  --max-games 10 \
  --return-components

# Look for:
# - Which component has highest error?
# - Is one component dominating predictions?
# - Are weights balanced correctly?

# Expected output:
# State-Space v2: 47.8 MAE, weight 30%
# Hierarchical v2: 38.2 MAE, weight 30%
# XGBoost v4: 4.7 MAE (residuals), weight 25%
# Informative Priors v2: 35.7 MAE, weight 15%
# Ensemble: 36.5 MAE ✅
```

**Issue: Integration errors > 0**
```python
# Check logs for specific errors
cat logs/ensemble_health.log

# Common causes:
# 1. Model file not found (check paths)
# 2. Model file corrupted (check file size > 0)
# 3. Database connection failure
# 4. Feature engineering errors (missing columns)

# Rerun with verbose logging
python py/ensemble/enhanced_ensemble_v6_production.py --verbose
```

---

## Model Registration

**Purpose:** Sync retrained models to database for versioning and audit trail.

**Time:** 30 minutes (mostly automated)

**Prerequisites:**
- All 4 validation gates passed
- Models saved to `models/` directory
- Database accessible (localhost:5544)

---

### Register All Models to Database

**Command:**
```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Run model registry script
./scripts/register_production_models.sh --all

# Expected output:
# ========================================
# Model Registry: Register All Production Models
# ========================================
#
# Registering State-Space v2...
#   ✓ Model ID: 37
#   ✓ Metadata: 2015-2024 training window, Rhat=1.003
#   ✓ File size: 45.3 MB
#
# Registering Hierarchical v2...
#   ✓ Model ID: 38
#   ✓ Metadata: 2015-2025 training window, Rhat=1.0087
#   ✓ File size: 128.4 MB
#
# Registering Informative Priors v2...
#   ✓ Model ID: 39
#   ✓ Metadata: 2015-2025 training window, priors from 2015-2019
#   ✓ File size: 115.2 MB
#
# Registering XGBoost v4...
#   ✓ Model ID: 40
#   ✓ Metadata: 2018-2024 training window, MAE=4.7 yards
#   ✓ File size: 2.1 MB
#
# Registering Temperature Scaler v6...
#   ✓ Model ID: 41
#   ✓ Metadata: T=0.5534, coverage=89.7%
#   ✓ File size: 512 KB
#
# ========================================
# ✅ All 5 models registered successfully
# ========================================
```

**Manual Registration (if script fails):**
```bash
# Register each model individually
./scripts/model_registry.sh register --model models/bayesian/state_space_v2.rds
./scripts/model_registry.sh register --model models/bayesian/passing_yards_hierarchical_v2.rds
./scripts/model_registry.sh register --model models/bayesian/passing_informative_priors_v2.rds
./scripts/model_registry.sh register --model models/xgboost/xgboost_v4_ridge.pkl
./scripts/model_registry.sh register --model models/calibration/temperature_scaler_v6.pkl
```

---

### Verify Registration

**Command:**
```bash
# List all production models
./scripts/model_registry.sh list --production

# Expected output:
# ========================================
# Production Models Registry
# ========================================
#
# Model ID: 37
#   Name: state_space_v2
#   Type: Bayesian State-Space
#   Training Window: 2015-2024
#   Convergence: Rhat=1.003, ESS=1245
#   File: models/bayesian/state_space_v2.rds
#   Registered: 2025-11-16 14:23:12
#
# Model ID: 38
#   Name: passing_yards_hierarchical_v2
#   Type: Bayesian Hierarchical
#   Training Window: 2015-2025
#   Convergence: Rhat=1.0087, ESS=512
#   File: models/bayesian/passing_yards_hierarchical_v2.rds
#   Registered: 2025-11-16 14:23:45
#
# Model ID: 39
#   Name: passing_informative_priors_v2
#   Type: Bayesian Informative Priors
#   Training Window: 2015-2025
#   Priors: 2015-2019 empirical Bayes
#   Convergence: Rhat=1.0065, ESS=687
#   File: models/bayesian/passing_informative_priors_v2.rds
#   Registered: 2025-11-16 14:24:18
#
# Model ID: 40
#   Name: xgboost_v4_ridge
#   Type: Gradient Boosting (Ridge Residuals)
#   Training Window: 2018-2024
#   Performance: MAE=4.7 yards (residuals)
#   File: models/xgboost/xgboost_v4_ridge.pkl
#   Registered: 2025-11-16 14:24:52
#
# Model ID: 41
#   Name: temperature_scaler_v6
#   Type: Isotonic Regression (Calibration)
#   Temperature: 0.5534
#   Coverage: 89.7% (target 90%)
#   File: models/calibration/temperature_scaler_v6.pkl
#   Registered: 2025-11-16 14:25:21
#
# ========================================
# Total: 5 production models
# ========================================
```

**Database Verification:**
```bash
# Query database directly
psql postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01 \
  -c "SELECT model_id, model_name, model_version, created_at
      FROM ml.model_metadata
      WHERE model_status = 'production'
      ORDER BY created_at DESC
      LIMIT 5;"

# Expected output:
# model_id |          model_name             | model_version |      created_at
# ---------+---------------------------------+---------------+---------------------
#       41 | temperature_scaler_v6           | v6.0          | 2025-11-16 14:25:21
#       40 | xgboost_v4_ridge                | v4.0          | 2025-11-16 14:24:52
#       39 | passing_informative_priors_v2   | v2.0          | 2025-11-16 14:24:18
#       38 | passing_yards_hierarchical_v2   | v2.0          | 2025-11-16 14:23:45
#       37 | state_space_v2                  | v2.0          | 2025-11-16 14:23:12
```

**Validation Checklist:**
- [ ] All 5 models appear in registry
- [ ] `model_status = 'production'` (not 'experimental' or 'deprecated')
- [ ] Timestamps reflect current retraining session
- [ ] Metadata accurate (training windows, convergence, performance)
- [ ] File paths correct and accessible

---

## Deployment Triggers

**Purpose:** Decide when to deploy new models to production vs manual review.

---

### Auto-Deploy (All Conditions MUST Be Met)

**Criteria:**
- [ ] All validation gates passed (Gates 1-4)
- [ ] Performance improvement > 2 yards MAE vs current production
- [ ] No degradation on any individual year (2022, 2023, 2024)
- [ ] Ensemble health check passed

**Example:**
```
Current production ensemble MAE: 40.2 yards
New retrained ensemble MAE: 36.5 yards
Improvement: 3.7 yards ✅ (> 2 yards threshold)

Multi-season MAE:
  2022: 5.06 yards (vs 5.8 previous) ✅ no degradation
  2023: 4.76 yards (vs 5.2 previous) ✅ no degradation
  2024: 4.27 yards (vs 4.9 previous) ✅ no degradation

Ensemble health: PASS ✅

Decision: AUTO-DEPLOY ✅
```

**Auto-Deploy Command:**
```bash
# Deploy new models (automated)
./scripts/deploy_production_models.sh --auto

# This will:
# 1. Stop production API
# 2. Copy new models to production directory
# 3. Update ensemble configuration
# 4. Restart production API
# 5. Run health checks
# 6. Monitor for 1 hour
```

---

### Manual Review (Any ONE Condition Triggers)

**Criteria:**
- Performance degradation > 1 yard MAE vs current production
- Convergence warnings (Rhat 1.01-1.05, ESS 300-400)
- Calibration coverage < 85% or > 95%
- Large weight shift (> 10% change in ensemble weights)

**Example:**
```
Current production ensemble MAE: 38.2 yards
New retrained ensemble MAE: 39.5 yards
Degradation: 1.3 yards ⚠️  (> 1 yard threshold)

Decision: MANUAL REVIEW REQUIRED ⚠️
```

**Manual Review Process:**
```bash
# 1. Generate detailed comparison report
python py/utils/model_comparison.py \
  --old models_production/ \
  --new models/ \
  --output analysis/results/model_comparison_$(date +%Y%m%d).html

# 2. Review report (open in browser)
open analysis/results/model_comparison_$(date +%Y%m%d).html

# 3. Check specific issues:
# - Which component degraded?
# - Is degradation consistent across all years?
# - Are there data quality issues?
# - Is this expected seasonal variance?

# 4. Make decision:
# - Deploy if degradation is minor (<2 yards) and explainable
# - Skip deployment if degradation is significant (>2 yards)
# - Investigate further if cause is unclear
```

---

### Rollback (IMMEDIATE Action Required)

**Criteria (ANY ONE):**
- Critical validation failure (Rhat > 1.05, ESS < 300)
- Ensemble crash or integration error
- MAE degradation > 10% vs current production
- API errors > 5% of requests

**Example:**
```
Ensemble health check: FAILED ❌
  - R model integration error (rpy2 crash)
  - Component drift: CRITICAL
  - Integration errors: 15

Decision: IMMEDIATE ROLLBACK ❌
```

**Rollback Command:**
```bash
# Emergency rollback to previous models
./scripts/rollback_production_models.sh --emergency

# This will:
# 1. Stop production API immediately
# 2. Restore models from backup
# 3. Revert ensemble configuration
# 4. Restart production API
# 5. Verify health checks
# 6. Alert on-call engineer
```

---

## Deployment Process

**Time:** 30-60 minutes (including 24-hour monitoring)

**Prerequisites:**
- All validation gates passed
- Models registered to database
- Auto-deploy criteria met (or manual review approved)

---

### Step 1: Stop Production API

```bash
# Stop FastAPI service
docker compose stop fastapi

# Verify service stopped
docker compose ps | grep fastapi
# Expected: nfl-analytics-fastapi-1   Exited (0)
```

---

### Step 2: Copy New Models to Production

```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Create production models directory if needed
mkdir -p models_production/bayesian
mkdir -p models_production/xgboost
mkdir -p models_production/calibration

# Copy new models
cp models/bayesian/state_space_v2.rds models_production/bayesian/
cp models/bayesian/passing_yards_hierarchical_v2.rds models_production/bayesian/
cp models/bayesian/passing_informative_priors_v2.rds models_production/bayesian/
cp models/xgboost/xgboost_v4_ridge.pkl models_production/xgboost/
cp models/calibration/temperature_scaler_v6.pkl models_production/calibration/

# Verify files copied
ls -lh models_production/bayesian/
ls -lh models_production/xgboost/
ls -lh models_production/calibration/

# Expected output (example):
# -rw-r--r-- 1 dro staff  45M Nov 16 14:30 state_space_v2.rds
# -rw-r--r-- 1 dro staff 128M Nov 16 14:30 passing_yards_hierarchical_v2.rds
# -rw-r--r-- 1 dro staff 115M Nov 16 14:30 passing_informative_priors_v2.rds
# -rw-r--r-- 1 dro staff   2M Nov 16 14:30 xgboost_v4_ridge.pkl
# -rw-r--r-- 1 dro staff 512K Nov 16 14:30 temperature_scaler_v6.pkl
```

---

### Step 3: Update Ensemble Configuration

```bash
# Edit production ensemble configuration
nano py/ensemble/enhanced_ensemble_v6_production.py

# Update model paths (lines 45-55):
# OLD:
#   state_space_path = "models/bayesian/state_space_v2_20251109.rds"
#   hierarchical_path = "models/bayesian/passing_yards_hierarchical_v2_20251109.rds"
#   ...
#
# NEW:
#   state_space_path = "models_production/bayesian/state_space_v2.rds"
#   hierarchical_path = "models_production/bayesian/passing_yards_hierarchical_v2.rds"
#   informative_priors_path = "models_production/bayesian/passing_informative_priors_v2.rds"
#   xgboost_path = "models_production/xgboost/xgboost_v4_ridge.pkl"
#   temperature_scaler_path = "models_production/calibration/temperature_scaler_v6.pkl"

# Verify configuration
grep -n "models_production" py/ensemble/enhanced_ensemble_v6_production.py

# Expected output (example):
# 45:    state_space_path = "models_production/bayesian/state_space_v2.rds"
# 46:    hierarchical_path = "models_production/bayesian/passing_yards_hierarchical_v2.rds"
# 47:    informative_priors_path = "models_production/bayesian/passing_informative_priors_v2.rds"
# 48:    xgboost_path = "models_production/xgboost/xgboost_v4_ridge.pkl"
# 49:    temperature_scaler_path = "models_production/calibration/temperature_scaler_v6.pkl"
```

---

### Step 4: Restart Production API

```bash
# Restart FastAPI service
docker compose up -d fastapi

# Wait for service to initialize (30 seconds)
sleep 30

# Check service status
docker compose ps | grep fastapi

# Expected output:
# nfl-analytics-fastapi-1   Up (healthy)
```

---

### Step 5: Health Check

```bash
# Check API health endpoint
curl http://localhost:8000/health

# Expected output:
# {
#   "status": "healthy",
#   "ensemble_loaded": true,
#   "ensemble_version": "v6.0_production",
#   "components": {
#     "state_space": "loaded",
#     "hierarchical": "loaded",
#     "informative_priors": "loaded",
#     "xgboost": "loaded",
#     "temperature_scaler": "loaded"
#   },
#   "weights": {
#     "state_space": 0.30,
#     "hierarchical": 0.30,
#     "xgboost": 0.25,
#     "informative_priors": 0.15
#   },
#   "temperature": 0.5534,
#   "last_prediction": "2025-11-16T14:35:21Z"
# }
```

**Validation:**
- [ ] `status: "healthy"` ✅
- [ ] `ensemble_loaded: true` ✅
- [ ] All 5 components: `"loaded"` ✅
- [ ] Weights sum to 1.0 ✅
- [ ] Temperature matches trained scaler ✅

---

### Step 6: Test Prediction

```bash
# Test single prediction (Patrick Mahomes, 2025 Week 12)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0033873",
    "season": 2025,
    "week": 12
  }'

# Expected output (example):
# {
#   "player_id": "00-0033873",
#   "player_name": "Patrick Mahomes",
#   "season": 2025,
#   "week": 12,
#   "prediction": {
#     "mean": 287.3,
#     "std": 24.8,
#     "lower_90": 246.5,
#     "upper_90": 328.1
#   },
#   "components": {
#     "state_space": 291.2,
#     "hierarchical": 285.7,
#     "informative_priors": 283.4,
#     "xgboost": 4.1
#   },
#   "confidence": "high",
#   "prediction_time_ms": 234
# }
```

**Validation:**
- [ ] Prediction completed without errors ✅
- [ ] Mean prediction reasonable (200-400 yards for elite QB) ✅
- [ ] Std reasonable (20-40 yards) ✅
- [ ] 90% interval width 70-100 yards ✅
- [ ] Prediction time < 500ms ✅

---

### Step 7: Monitor for First Hour

```bash
# Monitor API logs
docker compose logs -f fastapi

# Expected output (no errors):
# 2025-11-16 14:36:12 INFO: Prediction request received (player_id=00-0033873)
# 2025-11-16 14:36:12 INFO: Ensemble prediction complete (234ms)
# 2025-11-16 14:36:12 INFO: Response sent (200 OK)
# 2025-11-16 14:37:45 INFO: Prediction request received (player_id=00-0019596)
# ...

# Monitor Grafana dashboard
open http://localhost:3000/d/nfl-analytics-predictions

# Check metrics:
# - API error rate: Should be 0%
# - Prediction latency: Should be < 500ms p95
# - Predictions per minute: Normal load
# - Ensemble MAE: Should match backtest (~36-38 yards)
```

**Monitoring Checklist (First Hour):**
- [ ] API error rate < 1% ✅
- [ ] Prediction latency < 500ms (p95) ✅
- [ ] No R model integration errors ✅
- [ ] Ensemble MAE tracking as expected ✅
- [ ] No memory leaks (steady memory usage) ✅

---

### Step 8: Document Deployment

```bash
# Create deployment log
cat > docs/deployment/DEPLOYMENT_LOG_$(date +%Y%m%d).md << EOF
# Deployment Log - $(date +%Y-%m-%d)

**Deployment Time:** $(date)
**Deployed By:** [Your Name]
**Reason:** Weekly retraining (Week 12, 2025 season)

## Models Deployed

1. **State-Space v2**
   - File: models_production/bayesian/state_space_v2.rds
   - Training Window: 2015-2024
   - Convergence: Rhat=1.003, ESS=1245
   - Performance: 47.8 MAE (2024 test)

2. **Hierarchical v2**
   - File: models_production/bayesian/passing_yards_hierarchical_v2.rds
   - Training Window: 2015-2025
   - Convergence: Rhat=1.0087, ESS=512
   - Performance: 38.2 MAE (2024 test)

3. **Informative Priors v2**
   - File: models_production/bayesian/passing_informative_priors_v2.rds
   - Training Window: 2015-2025
   - Convergence: Rhat=1.0065, ESS=687
   - Performance: 35.7 MAE (2024 test)

4. **XGBoost v4**
   - File: models_production/xgboost/xgboost_v4_ridge.pkl
   - Training Window: 2018-2024
   - Performance: 4.7 MAE (residuals)
   - Multi-season: 4.70 ± 0.40 MAE

5. **Temperature Scaler v6**
   - File: models_production/calibration/temperature_scaler_v6.pkl
   - Temperature: 0.5534
   - Coverage: 89.7% (target 90%)

## Validation Results

- Gate 1 (MCMC Convergence): PASS ✅
- Gate 2 (Performance Benchmarks): PASS ✅
- Gate 3 (Multi-Season Consistency): PASS ✅
- Gate 4 (Ensemble Integration): PASS ✅

## Deployment Decision

- Current production MAE: 40.2 yards
- New ensemble MAE: 36.5 yards
- Improvement: 3.7 yards
- Decision: AUTO-DEPLOY ✅

## Health Check (1 Hour Post-Deployment)

- API error rate: 0%
- Prediction latency: 234ms avg
- Predictions served: 47
- Issues detected: None

## Notes

Weekly retraining completed successfully. All models converged, all gates passed, ensemble performance improved by 3.7 yards MAE. No issues detected during first hour of monitoring.
EOF

# Commit deployment log to Git
git add docs/deployment/DEPLOYMENT_LOG_$(date +%Y%m%d).md
git commit -m "docs: deployment log for $(date +%Y-%m-%d) weekly retraining"

# Update CLAUDE.md with latest deployment
# (Edit CLAUDE.md "Recent Work" section to add deployment entry)
```

---

## Rollback Procedure

**Purpose:** Emergency recovery if deployment fails or production issues detected.

**Time:** 5-10 minutes (critical speed required)

---

### When to Rollback (IMMEDIATE Action)

**Trigger Conditions:**
- API error rate > 5% for 15+ minutes
- Prediction latency > 2 seconds (p95) for 15+ minutes
- Ensemble crash or integration error
- MAE degradation > 20% vs baseline
- Critical bug discovered in production

---

### Emergency Rollback Steps

**Step 1: Stop Production API**
```bash
# Stop FastAPI immediately
docker compose stop fastapi

# Verify service stopped
docker compose ps | grep fastapi
# Expected: nfl-analytics-fastapi-1   Exited (0)
```

**Step 2: Restore Previous Models**
```bash
# Navigate to project root
cd /Users/dro/rice/nfl-analytics

# Find latest backup directory
BACKUP_DIR=$(ls -td models_backup_* | head -1)
echo "Restoring from: $BACKUP_DIR"

# Restore models
cp "$BACKUP_DIR/state_space_v2.rds" models_production/bayesian/
cp "$BACKUP_DIR/passing_yards_hierarchical_v2.rds" models_production/bayesian/
cp "$BACKUP_DIR/passing_informative_priors_v2.rds" models_production/bayesian/
cp "$BACKUP_DIR/xgboost_v4_ridge.pkl" models_production/xgboost/
cp "$BACKUP_DIR/temperature_scaler_v6.pkl" models_production/calibration/

# Verify restoration
ls -lh models_production/bayesian/
ls -lh models_production/xgboost/
ls -lh models_production/calibration/
```

**Step 3: Revert Ensemble Configuration**
```bash
# Revert ensemble configuration to previous version
git checkout HEAD~1 py/ensemble/enhanced_ensemble_v6_production.py

# Verify reversion
git diff HEAD py/ensemble/enhanced_ensemble_v6_production.py
```

**Step 4: Restart Production API**
```bash
# Restart FastAPI service
docker compose up -d fastapi

# Wait for initialization
sleep 30

# Check service status
docker compose ps | grep fastapi
# Expected: nfl-analytics-fastapi-1   Up (healthy)
```

**Step 5: Verify Rollback Successful**
```bash
# Health check
curl http://localhost:8000/health

# Expected output (old version):
# {
#   "status": "healthy",
#   "ensemble_loaded": true,
#   "ensemble_version": "v6.0_production_rollback",
#   ...
# }

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_id": "00-0033873", "season": 2025, "week": 12}'

# Should return valid prediction without errors
```

**Step 6: Document Incident**
```bash
# Create postmortem document
cat > docs/POSTMORTEMS.md << EOF
## Rollback Incident - $(date +%Y-%m-%d)

**Time:** $(date)
**Trigger:** [Describe what caused rollback]
**Duration:** [Time from deployment to rollback]
**Impact:** [How many predictions affected, error rate, etc.]

### What Happened

[Detailed description of issue]

### Root Cause

[Why did this happen?]

### Timeline

- 14:30 - New models deployed
- 14:45 - First errors detected (API error rate 8%)
- 15:00 - Rollback initiated
- 15:05 - Rollback complete, service restored

### Resolution

Rolled back to previous model version. Service restored, error rate back to 0%.

### Prevention

[What changes will prevent this in the future?]

### Action Items

- [ ] Fix root cause issue
- [ ] Add validation test for this failure mode
- [ ] Update deployment checklist
EOF

# Commit postmortem
git add docs/POSTMORTEMS.md
git commit -m "docs: postmortem for $(date +%Y-%m-%d) rollback incident"
```

---

### Rollback Verification Checklist

- [ ] API error rate back to < 1%
- [ ] Prediction latency back to < 500ms
- [ ] No R model integration errors
- [ ] Predictions returning valid results
- [ ] Grafana dashboard shows normal metrics
- [ ] Postmortem documented

---

## Monitoring Post-Deployment

**Purpose:** Ensure new models perform as expected in production.

**Duration:** 24-48 hours intensive monitoring, then daily checks

---

### First 24 Hours (CRITICAL)

**Metrics to Monitor:**

1. **API Error Rate** (Target: < 1%)
   - Check: Every 15 minutes for first 4 hours, then hourly
   - Alert: If > 5% for 15+ minutes
   - Grafana: http://localhost:3000/d/nfl-analytics-api

2. **Prediction Latency** (Target: < 500ms p95)
   - Check: Every 15 minutes for first 4 hours, then hourly
   - Alert: If > 2 seconds for 15+ minutes
   - Grafana: http://localhost:3000/d/nfl-analytics-predictions

3. **MAE Tracking** (Target: Match backtest ±2 yards)
   - Check: Every hour for first 24 hours
   - Alert: If degradation > 5 yards vs backtest
   - Query: `SELECT AVG(ABS(actual - prediction)) FROM prediction_audit WHERE created_at > NOW() - INTERVAL '24 hours'`

4. **Calibration Coverage** (Target: 88-92%)
   - Check: Every 4 hours for first 24 hours
   - Alert: If < 85% or > 95%
   - Query: `SELECT AVG(in_confidence_interval::int) FROM prediction_audit WHERE created_at > NOW() - INTERVAL '24 hours'`

---

### Monitoring Commands

**Quick Health Check:**
```bash
# Run health check script
./scripts/quick_health_check.sh

# Expected output (example):
# === NFL Analytics Infrastructure Health Check ===
# Time: Mon Nov 16 15:00:00 PST 2025
#
# 1. Docker Services:
#   nfl-analytics-pg-1       running
#   nfl-analytics-redis      running
#   nfl-analytics-fastapi    running
#   prometheus               running
#   grafana                  running
#
# 2. API Endpoint:
#   ✓ API healthy
#
# 3. Database Connection:
#   ✓ Database healthy
#
# 4. Redis Cache:
#   ✓ Redis healthy
#
# 5. Prediction Activity (last 24h):
#   Predictions: 47 (target: >50)
#
# 6. Recent Calibration (last 100 predictions):
#   Coverage: 89.2% (target: 85-95%)
#
# === Health Check Complete ===
```

**MAE Tracking:**
```bash
# Check current MAE
psql postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01 \
  -c "SELECT
        COUNT(*) as predictions,
        AVG(ABS(actual_yards - predicted_yards)) as mae,
        STDDEV(ABS(actual_yards - predicted_yards)) as mae_std
      FROM prediction_audit
      WHERE created_at > NOW() - INTERVAL '24 hours'
        AND actual_yards IS NOT NULL;"

# Expected output (example):
# predictions | mae  | mae_std
# ------------+------+---------
#          47 | 37.2 |    12.4

# MAE should be within ±2 yards of backtest (36.5 ± 2 = 34.5-38.5)
```

**Calibration Coverage:**
```bash
# Check calibration coverage
psql postgresql://dro:${DB_PASSWORD}@localhost:5544/devdb01 \
  -c "SELECT
        COUNT(*) as predictions,
        SUM(CASE WHEN in_confidence_interval THEN 1 ELSE 0 END) as within_ci,
        ROUND(100.0 * SUM(CASE WHEN in_confidence_interval THEN 1 ELSE 0 END) / COUNT(*), 1) as coverage_pct
      FROM prediction_audit
      WHERE created_at > NOW() - INTERVAL '24 hours'
        AND in_confidence_interval IS NOT NULL;"

# Expected output (example):
# predictions | within_ci | coverage_pct
# ------------+-----------+--------------
#          47 |        42 |         89.4

# Coverage should be 88-92% (90% target ±2%)
```

---

### Daily Monitoring (After 48 Hours)

**Frequency:** Once per day (9:00 AM recommended)

**Command:**
```bash
# Run daily health check
./scripts/daily_health_check.sh

# Review Grafana dashboard
open http://localhost:3000/d/nfl-analytics-predictions

# Check metrics:
# - Predictions per day (should be consistent)
# - MAE trend (should be stable)
# - Calibration coverage (should stay 88-92%)
# - API uptime (should be 99%+)
```

**Red Flags (Trigger Investigation):**
- MAE increases > 5 yards over 3 days
- Calibration coverage drops < 85% or rises > 95%
- API error rate > 2% for 2+ hours
- Prediction latency > 1 second for 2+ hours

---

## Troubleshooting Common Issues

---

### Issue 1: Model Training Fails (MCMC Convergence)

**Symptoms:**
- Rhat > 1.01 after 6000 iterations
- ESS < 400 after 6000 iterations
- Divergences > 10

**Diagnosis:**
```R
# Check summary statistics
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
summary(model)

# Look for:
# - Which parameters have high Rhat? (indicates where convergence failed)
# - Which parameters have low ESS? (indicates poor mixing)
# - What are divergence transition counts?
```

**Solutions:**

**Solution 1: Increase Iterations and Warmup**
```R
# Edit training script (e.g., R/train_hierarchical_v2_production.R)
# Change:
#   iter = 6000, warmup = 3000
# To:
#   iter = 10000, warmup = 5000

# Rationale: More iterations allow better exploration of parameter space
# Downside: Training takes 1.5-2x longer
```

**Solution 2: Increase adapt_delta**
```R
# Edit training script
# Change:
#   adapt_delta = 0.99
# To:
#   adapt_delta = 0.995  # or 0.999 for severe divergences

# Rationale: Smaller step size reduces divergences
# Downside: Slower sampling, longer training time
```

**Solution 3: Try Non-Centered Parameterization**
```R
# For hierarchical models, use Matt trick (non-centered)
# See: https://mc-stan.org/docs/stan-users-guide/reparameterization.html

# Example:
# OLD (centered):
#   y ~ normal(mu + alpha[player_id], sigma)
#   alpha ~ normal(0, tau)
#
# NEW (non-centered):
#   y ~ normal(mu + alpha_raw[player_id] * tau, sigma)
#   alpha_raw ~ normal(0, 1)

# Rationale: Better geometry for sampling
# Downside: Requires Stan model modification
```

**Solution 4: Adjust Priors**
```R
# If priors are too tight, widen them
# Change:
#   prior(normal(0, 0.5), class = b)
# To:
#   prior(normal(0, 1.0), class = b)

# If priors are too wide, tighten them
# Change:
#   prior(exponential(10), class = sd)
# To:
#   prior(exponential(20), class = sd)

# Rationale: Better prior-data alignment
# Downside: May reduce regularization
```

---

### Issue 2: Validation Gate Failure

**Symptoms:**
- Gate 2 (Performance Benchmarks) fails
- MAE > expected range (e.g., 55 yards for Hierarchical v2, expected 30-45)

**Diagnosis:**
```R
# Generate predictions on 2024 test set
library(tidyverse)
library(DBI)
library(RPostgres)

model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")

# Load 2024 test data
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD")
)

test_data <- dbGetQuery(conn, "
  SELECT *
  FROM mart.player_game_stats pgs
  JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
  LEFT JOIN games g ON
    g.season = pgs.season AND g.week = pgs.week AND
    (g.home_team = ph.current_team OR g.away_team = ph.current_team)
  WHERE pgs.season = 2024
    AND pgs.stat_category = 'passing'
    AND pgs.stat_attempts >= 10
")

dbDisconnect(conn)

# Generate predictions
preds <- predict(model, newdata = test_data, summary = FALSE)
pred_mean <- colMeans(preds)

# Calculate MAE
mae <- mean(abs(test_data$stat_yards - pred_mean))
print(paste("MAE:", round(mae, 2), "yards"))

# Analyze residuals
residuals <- test_data$stat_yards - pred_mean
hist(residuals, breaks = 50, main = "Residuals Distribution")

# Check for systematic bias
print(paste("Mean residual:", round(mean(residuals), 2), "yards"))
print(paste("Median residual:", round(median(residuals), 2), "yards"))

# Check for outliers
outliers <- test_data %>%
  mutate(residual = stat_yards - pred_mean) %>%
  filter(abs(residual) > 100) %>%
  select(player_id, player_display_name, season, week, stat_yards, pred_mean = pred_mean, residual)

print(outliers)
```

**Solutions:**

**Solution 1: Check Data Quality**
```R
# Look for missing values
summary(test_data)

# Look for outliers (e.g., 600 yard games)
test_data %>%
  filter(stat_yards > 500) %>%
  select(player_display_name, season, week, stat_yards, stat_attempts)

# Look for distribution shift
test_data %>%
  group_by(season) %>%
  summarize(
    mean_yards = mean(stat_yards),
    sd_yards = sd(stat_yards),
    n = n()
  )
```

**Solution 2: Check Training Window Alignment**
```R
# Verify training window matches canonical (2015-2025)
# Check script header:
# grep -n "2015-2025" R/train_hierarchical_v2_production.R

# If training window is wrong, retrain with correct window
```

**Solution 3: Check Feature Engineering**
```R
# Verify features are computed correctly
# Common issues:
# - Missing asof joins (data leakage)
# - Wrong column names (home_team vs current_team)
# - Missing factor levels (new teams not in training data)

# Check for NAs in critical features
test_data %>%
  summarize(across(everything(), ~sum(is.na(.))))
```

---

### Issue 3: R Model Integration Error (rpy2)

**Symptoms:**
- Ensemble health check fails
- Error message: `rpy2.rinterface_lib.embedded.RRuntimeError`
- Python cannot load R models

**Diagnosis:**
```bash
# Check R installation
which R
R --version

# Check rpy2 installation
python -c "import rpy2; print(rpy2.__version__)"

# Check R_HOME environment variable
echo $R_HOME

# Test R model loading in Python
python -c "
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Try loading model
ro.r('model <- readRDS(\"models/bayesian/state_space_v2.rds\")')
print('Model loaded successfully')
"
```

**Solutions:**

**Solution 1: Set R_HOME Environment Variable**
```bash
# Find R_HOME
R RHOME

# Expected output (example):
# /Library/Frameworks/R.framework/Resources

# Set R_HOME in .env file
echo "R_HOME=/Library/Frameworks/R.framework/Resources" >> .env

# Restart FastAPI service
docker compose restart fastapi
```

**Solution 2: Reinstall rpy2**
```bash
# Uninstall rpy2
pip uninstall rpy2

# Reinstall specific version (3.5.14 known to work)
pip install rpy2==3.5.14

# Verify installation
python -c "import rpy2; print(rpy2.__version__)"
```

**Solution 3: Check R Package Dependencies**
```bash
# Open R console
R

# Install missing packages
install.packages(c("brms", "posterior", "cmdstanr"))

# Verify packages load
library(brms)
library(posterior)
library(cmdstanr)

# Exit R
quit()
```

**Solution 4: Use Python-Only Ensemble (Fallback)**
```python
# Edit py/ensemble/enhanced_ensemble_v6_production.py
# Comment out R model imports

# OLD:
# from py.ensemble.r_model_wrapper import StateSpaceModel, HierarchicalModel

# NEW:
# # R models disabled - using Python-only ensemble
# # from py.ensemble.r_model_wrapper import StateSpaceModel, HierarchicalModel

# Update ensemble weights (XGBoost 100%)
# weights = {'xgboost': 1.0}

# Rationale: Quick fix to get system running
# Downside: Reduced performance (XGBoost only, no Bayesian models)
```

---

### Issue 4: Multi-Season Consistency Fails

**Symptoms:**
- Gate 3 (Multi-Season Consistency) fails
- MAE std > 2 yards across 2022, 2023, 2024
- Example: 2022: 2.1 MAE, 2023: 4.5 MAE, 2024: 9.0 MAE (std = 3.5)

**Diagnosis:**
```bash
# Run multi-season validation with detailed output
python scripts/xgb_v4_multi_season_validation.py --verbose

# Expected output (detailed):
# ======================================
# 2022 Validation (DETAILED)
# ======================================
# Train data: 2018-2021 (4,512 games)
# Test data: 2022 (583 games)
# Feature distribution check:
#   - passing_attempts: mean 33.2 (train 32.8) ✅
#   - is_home: mean 0.49 (train 0.50) ✅
#   - spread_abs: mean 4.2 (train 4.5) ✅
# ...
#
# MAE: 5.06 yards ✅
#
# ======================================
# 2024 Validation (DETAILED)
# ======================================
# Train data: 2018-2023 (6,587 games)
# Test data: 2024 (612 games)
# Feature distribution check:
#   - passing_attempts: mean 34.8 (train 32.8) ⚠️ SHIFT
#   - is_home: mean 0.52 (train 0.50) ⚠️ SHIFT
#   - spread_abs: mean 6.1 (train 4.5) ⚠️ SHIFT
# ...
#
# MAE: 9.00 yards ❌ (> 8 threshold)
```

**Solutions:**

**Solution 1: Check for Data Leakage**
```python
# Run data leakage detector
from py.utils.data_leakage_detector import DataLeakageDetector
import pandas as pd

# Load train/test splits
train_2023 = pd.read_csv('data/train_2023.csv')
test_2024 = pd.read_csv('data/test_2024.csv')

# Validate split
detector = DataLeakageDetector()
detector.validate_split(train_2023, test_2024, date_col='kickoff')

if detector.report.has_critical_leakage():
    print("⛔ DATA LEAKAGE DETECTED")
    print(detector.generate_report())
    # DO NOT DEPLOY - fix leakage first
```

**Solution 2: Check for Distribution Shift**
```python
# Compare feature distributions across years
import pandas as pd
import numpy as np

# Load data for all years
data_2022 = pd.read_csv('data/test_2022.csv')
data_2023 = pd.read_csv('data/test_2023.csv')
data_2024 = pd.read_csv('data/test_2024.csv')

# Compare key features
features = ['passing_attempts', 'is_home', 'spread_abs', 'total_close']

for feat in features:
    print(f"\n{feat}:")
    print(f"  2022: {data_2022[feat].mean():.2f} ± {data_2022[feat].std():.2f}")
    print(f"  2023: {data_2023[feat].mean():.2f} ± {data_2023[feat].std():.2f}")
    print(f"  2024: {data_2024[feat].mean():.2f} ± {data_2024[feat].std():.2f}")

    # Large shift indicates distribution change
    if abs(data_2024[feat].mean() - data_2022[feat].mean()) > data_2022[feat].std():
        print(f"  ⚠️  DISTRIBUTION SHIFT DETECTED")
```

**Solution 3: Retrain with Expanded Window**
```python
# If 2024 shows distribution shift, retrain including 2024 data
# Edit training script to include 2024 in training window
# This reduces test set but may improve generalization

# Change:
#   train_years = range(2018, 2024)  # 2018-2023
#   test_years = [2024]
#
# To:
#   train_years = range(2018, 2025)  # 2018-2024
#   test_years = [2025]  # Use current season for testing
```

---

## Weekly Schedule Template

**Purpose:** Standardized schedule for weekly retraining workflow.

---

### Monday Morning (9:00 AM - 12:30 PM)

**Time Block 1: Pre-Retraining Checks (9:00-9:30 AM, 30 minutes)**
- [ ] Docker services health check
- [ ] Database up-to-date (latest week ingested)
- [ ] Disk space available (10GB+)
- [ ] Git LFS configured
- [ ] Backup current models
- [ ] No active backtests running

**Time Block 2: Start Long-Running R Models (9:30 AM)**
```bash
# Start State-Space v2 (2-3 hours)
screen -S state_space
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)
Rscript R/state_space/train_state_space_v2_canonical.R
# Detach: Ctrl-A, D

# Check progress every 30 minutes
screen -r state_space
```

**Time Block 3: Monitor State-Space Training (9:30 AM - 12:30 PM)**
- Check progress every 30 minutes
- Look for convergence warnings
- Expected completion: ~12:00 PM

**Expected State by 12:30 PM:**
- State-Space v2: COMPLETE ✅ (or nearing completion)
- Next: Start Hierarchical v2 (longest model, 4-6 hours)

---

### Monday Afternoon (12:30 PM - 6:30 PM)

**Time Block 4: Start Hierarchical v2 (12:30 PM)**
```bash
# Start Hierarchical v2 (4-6 hours)
screen -S hierarchical
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)
Rscript R/train_hierarchical_v2_production.R
# Detach: Ctrl-A, D
```

**Time Block 5: Monitor Hierarchical Training (12:30 PM - 6:30 PM)**
- Check progress every hour
- Look for convergence warnings
- Expected completion: ~4:30-6:30 PM

**OPTIONAL: Start Informative Priors v2 in Parallel (2:00 PM)**
```bash
# ONLY if you have 8+ cores and 32GB+ RAM
screen -S informative_priors
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)
Rscript R/train_informative_priors_v2_production.R
# Detach: Ctrl-A, D

# This saves 3-4 hours by running in parallel
# Downside: Machine will be slow during training
```

**Expected State by 6:30 PM:**
- State-Space v2: COMPLETE ✅
- Hierarchical v2: COMPLETE ✅ (or nearing completion)
- Informative Priors v2: IN PROGRESS (if parallel) OR NOT STARTED

---

### Monday Evening (6:30 PM - 10:30 PM)

**Time Block 6: Start/Continue Informative Priors v2 (6:30 PM)**
```bash
# If NOT started in parallel, start now
screen -S informative_priors
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)
Rscript R/train_informative_priors_v2_production.R
# Detach: Ctrl-A, D
```

**Time Block 7: Monitor Informative Priors Training (6:30 PM - 10:30 PM)**
- Check progress every hour
- Expected completion: ~9:30-10:30 PM

**Expected State by 10:30 PM:**
- State-Space v2: COMPLETE ✅
- Hierarchical v2: COMPLETE ✅
- Informative Priors v2: COMPLETE ✅
- Next: Validate R models, then train Python models (fast)

---

### Tuesday Morning (9:00 AM - 12:00 PM)

**Time Block 8: Validate R Models (9:00-10:00 AM, 1 hour)**
```bash
# Check convergence for all 3 R models
./scripts/validate_r_models.sh

# Expected output:
# State-Space v2: Rhat=1.003, ESS=1245 ✅
# Hierarchical v2: Rhat=1.0087, ESS=512 ✅
# Informative Priors v2: Rhat=1.0065, ESS=687 ✅
#
# All R models: PASS ✅
```

**Time Block 9: Train XGBoost v4 (10:00-10:30 AM, 20 minutes)**
```bash
# Activate Python environment
source .venv/bin/activate

# Train XGBoost v4
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)
python scripts/xgb_v3_2_ridge_residuals.py

# Expected completion: 10:15 AM
```

**Time Block 10: Train Temperature Scaler v6 (10:30-11:00 AM, 30 minutes)**
```bash
# Train temperature scaler
export DB_PASSWORD=$(grep DB_PASSWORD .env | cut -d '=' -f2)
python scripts/train_temperature_scaler_v6.py

# Expected completion: 11:00 AM (if predictions cached)
# If NOT cached: 11:30 AM - 12:00 PM (ensemble predictions slow)
```

**Time Block 11: Run Validation Gates (11:00 AM - 12:00 PM, 1 hour)**
```bash
# Gate 1: MCMC Convergence (already done in Time Block 8)

# Gate 2: Performance Benchmarks
./scripts/validate_performance_benchmarks.sh

# Gate 3: Multi-Season Consistency
python scripts/xgb_v4_multi_season_validation.py

# Gate 4: Ensemble Integration
./scripts/check_ensemble_health.sh

# Expected: All gates PASS ✅
```

**Expected State by 12:00 PM:**
- All 5 models trained ✅
- All validation gates passed ✅
- Next: Register models, deploy

---

### Tuesday Afternoon (12:00 PM - 4:00 PM)

**Time Block 12: Register Models to Database (12:00-12:30 PM, 30 minutes)**
```bash
# Register all models
./scripts/register_production_models.sh --all

# Verify registration
./scripts/model_registry.sh list --production
```

**Time Block 13: Deployment Decision (12:30-1:00 PM, 30 minutes)**
```bash
# Check auto-deploy criteria
./scripts/check_deployment_criteria.sh

# Expected output:
# Current production MAE: 40.2 yards
# New ensemble MAE: 36.5 yards
# Improvement: 3.7 yards ✅ (> 2 yards threshold)
#
# Multi-season consistency: PASS ✅
# Ensemble health: PASS ✅
#
# Decision: AUTO-DEPLOY ✅
```

**Time Block 14: Deploy New Models (1:00-2:00 PM, 1 hour)**
```bash
# Deploy new models (automated)
./scripts/deploy_production_models.sh --auto

# This will:
# 1. Stop production API
# 2. Copy new models
# 3. Update ensemble config
# 4. Restart API
# 5. Health checks
# 6. Monitor first hour
```

**Time Block 15: Monitor Deployment (2:00-3:00 PM, 1 hour)**
```bash
# Monitor API logs
docker compose logs -f fastapi

# Check Grafana dashboard
open http://localhost:3000/d/nfl-analytics-predictions

# Verify:
# - API error rate < 1%
# - Prediction latency < 500ms
# - No R model errors
# - MAE tracking as expected
```

**Time Block 16: Document Deployment (3:00-4:00 PM, 1 hour)**
```bash
# Create deployment log
cat > docs/deployment/DEPLOYMENT_LOG_$(date +%Y%m%d).md << EOF
[See Step 8 in Deployment Process section for template]
EOF

# Commit deployment log
git add docs/deployment/DEPLOYMENT_LOG_$(date +%Y%m%d).md
git commit -m "docs: deployment log for $(date +%Y-%m-%d) weekly retraining"
git push origin main
```

**Expected State by 4:00 PM:**
- New models deployed to production ✅
- Health checks passing ✅
- Deployment documented ✅
- **WEEKLY RETRAINING COMPLETE** ✅

---

### Tuesday Evening - Wednesday Morning (Monitoring)

**Continue Monitoring for 24-48 Hours:**
- Check API error rate every 4 hours
- Check MAE tracking every 8 hours
- Check calibration coverage every 12 hours
- Daily health check at 9:00 AM Wednesday and Thursday

**Expected State by Thursday 9:00 AM:**
- 48 hours of production monitoring complete ✅
- No issues detected ✅
- Models performing as expected ✅
- **RETRAINING CYCLE COMPLETE** ✅

---

## Success Criteria

**All criteria MUST be met for retraining to be considered successful.**

---

### Training Success

- [ ] All 5 models trained without errors
- [ ] State-Space v2: Rhat < 1.01, ESS > 400, MAE 45-52 yards
- [ ] Hierarchical v2: Rhat < 1.01, ESS > 400, MAE 30-45 yards
- [ ] Informative Priors v2: Rhat < 1.01, ESS > 400, MAE 30-45 yards
- [ ] XGBoost v4: Multi-season std < 2 yards, MAE 4-6 yards
- [ ] Temperature Scaler: Coverage 88-92%, T 0.45-0.65

---

### Validation Success

- [ ] Gate 1 (MCMC Convergence): PASS
- [ ] Gate 2 (Performance Benchmarks): PASS
- [ ] Gate 3 (Multi-Season Consistency): PASS
- [ ] Gate 4 (Ensemble Integration): PASS

---

### Registration Success

- [ ] All 5 models registered to database
- [ ] Model IDs assigned (check `ml.model_metadata` table)
- [ ] Metadata accurate (training windows, convergence, performance)
- [ ] File paths correct and accessible

---

### Deployment Success

- [ ] Auto-deploy criteria met (or manual review approved)
- [ ] Production API restarted successfully
- [ ] Health checks passing (6/6 services operational)
- [ ] Test prediction returns valid results
- [ ] No errors in first hour of monitoring

---

### Monitoring Success

- [ ] 24-hour monitoring complete
- [ ] API error rate < 1%
- [ ] Prediction latency < 500ms (p95)
- [ ] MAE within ±2 yards of backtest
- [ ] Calibration coverage 88-92%
- [ ] No critical issues detected

---

### Documentation Success

- [ ] Deployment log created and committed
- [ ] CLAUDE.md updated with latest deployment
- [ ] Git LFS tracked model files
- [ ] No secrets committed to Git

---

## Summary

**Weekly Retraining Checklist provides:**

1. ✅ **Standardized workflow** - Follow same steps every week
2. ✅ **Clear validation gates** - Know when to deploy vs rollback
3. ✅ **Automated monitoring** - Detect issues early
4. ✅ **Troubleshooting guides** - Fix common problems quickly
5. ✅ **Emergency rollback** - Recover from failures in minutes

**Time Commitment:**
- **Monday:** 4-8 hours (mostly unattended training)
- **Tuesday:** 3-4 hours (validation, deployment, monitoring)
- **Wednesday-Thursday:** 30 minutes/day (monitoring)
- **Total:** 6-8 hours per week

**Expected Outcomes:**
- Fresh models maintain 90% calibration coverage
- MAE stays within 36-40 yards (baseline)
- ROI improvement: +0.5-1% vs stale models
- Reduced risk of model degradation

**Next Steps:**
1. Review this checklist before first retraining
2. Test workflow on practice data (2024 historical)
3. Execute first live retraining (Week 12, 2025)
4. Iterate and improve checklist based on experience

---

**Version:** 1.0
**Last Updated:** November 16, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
**Questions?** See [docs/deployment/README.md](README.md) for additional resources.
