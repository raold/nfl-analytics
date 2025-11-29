# NFL Analytics - Onboarding Guide

**Last Updated:** 2025-11-28
**Purpose:** Day-1 guide for developers joining the NFL Analytics project
**Time to Complete:** 2-4 hours

---

## Welcome

This guide walks you through everything you need to be productive on Day 1. By the end, you'll have:

1. A working development environment
2. Understanding of the project architecture
3. Ability to run the full prediction pipeline
4. Knowledge of where to find documentation

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (30 minutes)](#quick-setup-30-minutes)
3. [Project Overview](#project-overview)
4. [Key Concepts](#key-concepts)
5. [First Tasks](#first-tasks)
6. [Documentation Map](#documentation-map)
7. [Common Workflows](#common-workflows)
8. [Getting Help](#getting-help)

---

## Prerequisites

### Required Skills

| Skill | Level | Purpose |
|-------|-------|---------|
| Python | Intermediate | ML models, API, pipeline |
| SQL | Basic | Database queries |
| Git | Basic | Version control |
| R (optional) | Basic | Bayesian models (brms/Stan) |
| Docker | Basic | Service orchestration |

### Required Software

Before starting, install:

- **Python 3.10+** (pyenv recommended)
- **Docker Desktop** (for PostgreSQL, Redis)
- **Git** with Git LFS
- **R 4.0+** (optional, for Bayesian models)

---

## Quick Setup (30 minutes)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd nfl-analytics
```

### Step 2: Set Up Python

```bash
# Create virtual environment
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Start Services

```bash
# Start Docker services (PostgreSQL, Redis)
docker compose up -d

# Verify services
docker compose ps
# Expected: pg-1 (5544), redis (6379) running
```

### Step 4: Verify Database

```bash
# Connect to PostgreSQL
PGPASSWORD='sicillionbillions' psql -h localhost -p 5544 -U dro -d devdb01 -c "SELECT count(*) FROM games;"
# Expected: ~7,263 rows
```

### Step 5: Run Tests

```bash
# Quick validation
pytest tests/unit -m unit -x
```

### Step 6: Start API (Optional)

```bash
# Start FastAPI server
uvicorn py.api.main:app --host 0.0.0.0 --port 8000 --reload

# Test health endpoint
curl http://localhost:8000/health
```

**Setup complete.** You're ready to work on the project.

---

## Project Overview

### What This Project Does

NFL Analytics is a **production betting prediction system** that:

1. **Ingests** NFL play-by-play data from nflverse
2. **Trains** Bayesian + ML models on historical data
3. **Generates** game predictions with uncertainty quantification
4. **Optimizes** betting decisions via Kelly criterion & risk management
5. **Monitors** model performance in production

### Architecture

```
                 ┌─────────────┐
                 │   nflverse  │
                 │   (data)    │
                 └──────┬──────┘
                        ▼
┌─────────────────────────────────────────────────┐
│                  Data Layer                      │
│  PostgreSQL (5544) │ 257 columns │ 1.25M plays  │
└─────────────────────────┬───────────────────────┘
                          ▼
┌─────────────────────────────────────────────────┐
│               Model Layer (v6.1)                 │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐             │
│  │ State-Space  │  │ Hierarchical │  R (brms)   │
│  │   v2 (27%)   │  │   v2 (27%)   │             │
│  └──────────────┘  └──────────────┘             │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐             │
│  │   XGBoost    │  │ Informative  │  Python     │
│  │ Ridge (23%)  │  │ Priors (13%) │             │
│  └──────────────┘  └──────────────┘             │
│                                                  │
│  ┌──────────────┐                               │
│  │   Hybrid     │  ELO + Incremental            │
│  │   (10%)      │                               │
│  └──────────────┘                               │
└─────────────────────────┬───────────────────────┘
                          ▼
┌─────────────────────────────────────────────────┐
│              Ensemble (5-way)                    │
│  Temperature Scaling (T=0.7083) │ 90% calibrated │
└─────────────────────────┬───────────────────────┘
                          ▼
┌─────────────────────────────────────────────────┐
│            API Layer (FastAPI)                   │
│  /predict/game │ /predict/games/batch │ /health │
└─────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| ROI | +6.45% | Profit after costs |
| Win Rate | 60.8% | Games predicted correctly |
| Sharpe Ratio | 2.51 | Risk-adjusted returns |
| MAE | ~34 yards | Prediction accuracy |

---

## Key Concepts

### 1. Ensemble Architecture

We use a **5-way ensemble** combining diverse model types:

| Component | Type | Weight | Why? |
|-----------|------|--------|------|
| State-Space v2 | Bayesian (R) | 27% | Time-varying player skill |
| Hierarchical v2 | Bayesian (R) | 27% | Partial pooling across groups |
| XGBoost Ridge | Gradient Boosting (Python) | 23% | Non-linear feature interactions |
| Informative Priors v2 | Bayesian (R) | 13% | Domain knowledge encoded |
| Hybrid | ELO + Incremental | 10% | Recency weighting |

**Key insight:** Diversification beats sophistication. Adding uncorrelated weak models reduces variance more than improving strong models.

### 2. Ridge Residuals (Feature Orthogonalization)

The single biggest accuracy improvement (+88%) came from **orthogonalizing features**:

1. Train Ridge regression predicting yards from opponent strength
2. Extract residuals (opponent-independent player skill)
3. Use residuals as features in XGBoost

This removes opponent-player confounding that biases naive models.

### 3. PostgreSQL R Model Lookup

R models are slow (2-4s per prediction via rpy2). We pre-compute predictions:

```
R Models → PostgreSQL (ml.r_model_predictions) → Python Ensemble
                    <1ms lookup vs 2-4s inference
```

See: `docs/adr/006-postgresql-r-model-lookup.md`

### 4. Canonical Training Window

All production models use:
- **Training:** 2015-2024 (uniform weights)
- **Holdout:** Current season (2025)
- **Window:** H=4 half-life decay for time-weighted features

---

## First Tasks

### Day 1: Explore the Data

```bash
# Connect to database
PGPASSWORD='sicillionbillions' psql -h localhost -p 5544 -U dro -d devdb01

# Key tables
\dt games                    -- Game-level data
\dt mart.player_game_stats   -- Player stats per game
\dt ml.r_model_predictions   -- Pre-computed R model predictions
```

### Day 1: Generate a Prediction

```python
# In Python
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsembleV6

ensemble = EnhancedEnsembleV6()
prediction = ensemble.predict(
    home_team="KC",
    away_team="BUF",
    season=2024,
    week=12
)
print(prediction)
```

### Day 2: Run Backtest

```bash
# Run historical validation
python scripts/backtest_game_predictions.py --seasons 2024
```

### Day 3: Review Architecture Decisions

Read these ADRs to understand why the system is designed this way:

1. `docs/adr/004-bnn-research-abandonment.md` - Why no BNN
2. `docs/adr/005-temperature-scaling-resolution.md` - Calibration
3. `docs/adr/006-postgresql-r-model-lookup.md` - R model caching
4. `docs/adr/009-abandon-ensemble-v7-meta-learning.md` - Why static weights

---

## Documentation Map

### Must-Read Documents

| Document | Purpose | Location |
|----------|---------|----------|
| **CLAUDE.md** | AI assistant context | `/CLAUDE.md` |
| **SETUP.md** | Detailed setup | `/SETUP.md` |
| **DATABASE_QUICK_REFERENCE.md** | Schema guide | `docs/database/` |
| **SQL_PATTERNS.md** | Query patterns | `docs/database/` |

### By Topic

| Topic | Location |
|-------|----------|
| Architecture | `docs/ARCHITECTURE_REFERENCE.md` |
| ADRs | `docs/adr/*.md` |
| R Modeling | `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md` |
| Troubleshooting | `docs/guides/TROUBLESHOOTING_MASTER.md` |
| API Endpoints | `docs/api/ENDPOINT_INVENTORY.md` |
| Dissertation | `analysis/dissertation/` |

### Code Entry Points

| Component | Entry Point | Purpose |
|-----------|-------------|---------|
| Ensemble | `py/ensemble/enhanced_ensemble_v6_production.py` | Main prediction engine |
| API | `py/api/main.py` | FastAPI server |
| R Models | `R/train_hierarchical_v2_canonical.R` | Bayesian training |
| Dashboard | `dashboard/app.py` | Streamlit UI |

---

## Common Workflows

### Generate Weekly Predictions

```bash
# Run weekly prediction script
python scripts/generate_weekly_predictions.py --week 13 --season 2024
```

### Train R Model

```bash
# Train hierarchical model
Rscript R/train_hierarchical_v2_canonical.R

# Generate predictions for PostgreSQL lookup
Rscript R/batch_generate_r_predictions_optimized.R
```

### Run Full Pipeline

```bash
# 1. Update data
python py/pipeline/refresh_data.py

# 2. Generate predictions
python scripts/generate_game_predictions_v6.py

# 3. Run backtest
python scripts/backtest_game_predictions.py
```

### Debug an Issue

1. Check logs: `docker logs nfl-api-1`
2. Check database: `psql -c "SELECT * FROM ml.ensemble_predictions LIMIT 5"`
3. Check troubleshooting: `docs/guides/TROUBLESHOOTING_MASTER.md`

---

## Getting Help

### Documentation

- **CLAUDE.md** - Comprehensive project context
- **docs/guides/TROUBLESHOOTING_MASTER.md** - Common issues
- **docs/database/SQL_PATTERNS.md** - Query patterns

### Debugging

```bash
# Check service health
docker compose ps
curl http://localhost:8000/health

# Check database
PGPASSWORD='sicillionbillions' psql -h localhost -p 5544 -U dro -d devdb01

# Check logs
docker logs nfl-api-1 --tail 100
```

### Key Files When Lost

| Situation | File |
|-----------|------|
| "How does the ensemble work?" | `py/ensemble/enhanced_ensemble_v6_production.py` |
| "What columns exist?" | `docs/database/DATABASE_QUICK_REFERENCE.md` |
| "Why is this designed this way?" | `docs/adr/*.md` |
| "How to train R models?" | `docs/guides/R_BAYESIAN_MODELING_COOKBOOK.md` |
| "What went wrong?" | `docs/guides/TROUBLESHOOTING_MASTER.md` |

---

## Checklist

Before your first PR, verify:

- [ ] All tests pass: `pytest tests/unit -m unit`
- [ ] Linting passes: `pre-commit run --all-files`
- [ ] No secrets committed (check `.env` ignored)
- [ ] Documentation updated if needed

---

**Welcome to the team.** The codebase is large but well-documented. Start by exploring the database and running predictions. Ask questions early.

---

**Related Documents:**
- `SETUP.md` - Detailed environment setup
- `CONTRIBUTING.md` - Contribution guidelines
- `docs/guides/TROUBLESHOOTING_MASTER.md` - When things go wrong
