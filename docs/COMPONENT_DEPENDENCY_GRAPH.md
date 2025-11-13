# NFL Analytics - Component Dependency Graph

**Generated:** November 13, 2025
**Purpose:** Visual map of system dependencies for architecture analysis
**Related:** [ARCHITECTURE_ANALYSIS_NOV13_2025.md](ARCHITECTURE_ANALYSIS_NOV13_2025.md)

---

## High-Level Component Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐         ┌──────────────────┐                        │
│  │   Streamlit UI   │────────▶│   FastAPI        │                        │
│  │   (port 8501)    │   HTTP  │   (port 8000)    │                        │
│  └────────┬─────────┘         └────────┬─────────┘                        │
│           │                             │                                   │
│           │ SQL                         │ SQL                               │
│           │                             │                                   │
│           ▼                             ▼                                   │
│  ┌─────────────────────────────────────────────────────┐                  │
│  │                  PostgreSQL                          │◀─────┐           │
│  │         (public, mart, ml, metadata schemas)         │      │           │
│  └─────────────────────────────────────────────────────┘      │           │
│           ▲                             ▲                       │           │
│           │                             │                       │           │
└───────────┼─────────────────────────────┼───────────────────────┼───────────┘
            │                             │                       │
            │                             │                       │
┌───────────┼─────────────────────────────┼───────────────────────┼───────────┐
│           │         APPLICATION LAYER   │                       │           │
├───────────┼─────────────────────────────┼───────────────────────┼───────────┤
│           │                             │                       │           │
│  ┌────────┴────────────────────────────▼─────────┐             │           │
│  │   EnhancedEnsembleV5 (Python)                 │             │           │
│  │   ├─ weights: {state_space: 0.30,             │             │           │
│  │   │            hierarchical: 0.30,             │             │           │
│  │   │            xgboost: 0.25,                  │             │           │
│  │   │            informative_priors: 0.15}       │             │           │
│  │   └─ predict(player_id, season, week)         │             │           │
│  └────────┬─────────┬─────────┬─────────┬────────┘             │           │
│           │         │         │         │                       │           │
│           │         │         │         │                       │           │
│           ▼         ▼         ▼         ▼                       │           │
│  ┌────────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────────┐  │           │
│  │ State-Space│ │Hierarch.│ │XGBoost  │ │Informative Priors│  │           │
│  │  v2 (R)    │ │v2 (R)   │ │v2 (Py)  │ │   v2 (R)         │  │           │
│  │  .rds      │ │ .rds    │ │ .pkl    │ │   .rds           │  │           │
│  │  + .csv    │ │         │ │         │ │                  │  │           │
│  │  skills    │ │         │ │         │ │                  │  │           │
│  └────────┬───┘ └────┬────┘ └────┬────┘ └─────────┬────────┘  │           │
│           │          │           │                 │            │           │
│           │ rpy2     │ rpy2      │ xgboost.load   │ rpy2       │           │
│           │          │           │                 │            │           │
│           ▼          ▼           ▼                 ▼            │           │
│  ┌─────────────────────────────────────────────────────────┐   │           │
│  │           Model Artifacts (Filesystem)                   │   │           │
│  │   models/bayesian/*.rds (R models)                       │   │           │
│  │   models/bayesian/*.csv (state-space skills)             │   │           │
│  │   models/xgboost/*.pkl (Python models)                   │   │           │
│  │   models/cql/*.pth (PyTorch RL models)                   │   │           │
│  └─────────────────────────────────────────────────────────┘   │           │
│           ▲                                                     │           │
│           │                                                     │           │
│           │ Training Pipeline (writes models)                  │           │
│           │                                                     │           │
└───────────┼─────────────────────────────────────────────────────┼───────────┘
            │                                                     │
            │                                                     │
┌───────────┼─────────────────────────────────────────────────────┼───────────┐
│           │      MODEL TRAINING LAYER                           │           │
├───────────┼─────────────────────────────────────────────────────┼───────────┤
│           │                                                     │           │
│  ┌────────┴──────────────────┐    ┌────────────────────────────┴─────┐    │
│  │  R Training Scripts       │    │  Python Training Scripts         │    │
│  ├───────────────────────────┤    ├──────────────────────────────────┤    │
│  │ • train_hierarchical_     │    │ • train_xgboost_canonical.py     │    │
│  │   v2_production.R         │    │ • train_bnn_multioutput.py       │    │
│  │ • train_informative_      │    │ • hybrid_static_incremental.py   │    │
│  │   priors_v2_production.R  │    │ • train_cql_agent.py             │    │
│  │ • train_simplified_       │    │                                  │    │
│  │   random_walk.R           │    │                                  │    │
│  └────────┬──────────────────┘    └────────┬─────────────────────────┘    │
│           │ DBI::dbConnect()               │ psycopg2.connect()            │
│           │                                │                                │
│           ▼                                ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                PostgreSQL (Raw Data)                                 │  │
│  │  ├─ public.games (7,263 games, 1999-2024)                           │  │
│  │  ├─ public.plays (1,254,173 plays, 257 columns)                     │  │
│  │  ├─ mart.player_game_stats (24,950 records)                         │  │
│  │  ├─ mart.player_hierarchy (15,213 players)                          │  │
│  │  └─ ml.model_registry (8 models synced, 227 pending)                │  │
│  └────────▲────────────────────────────────────────────────────────────┘  │
│           │                                                                │
│           │                                                                │
└───────────┼────────────────────────────────────────────────────────────────┘
            │
            │
┌───────────┼────────────────────────────────────────────────────────────────┐
│           │      DATA INGESTION LAYER                                      │
├───────────┼────────────────────────────────────────────────────────────────┤
│           │                                                                │
│  ┌────────┴──────────────┐         ┌─────────────────────────┐            │
│  │  R Ingestion Scripts  │         │ Python Ingestion        │            │
│  ├───────────────────────┤         ├─────────────────────────┤            │
│  │ • ingest_pbp.R        │         │ • ingest_odds_history.py│            │
│  │   (nflfastR)          │         │   (Odds API)            │            │
│  │ • ingest_schedules.R  │         │                         │            │
│  │   (nflreadr)          │         │                         │            │
│  └────────┬──────────────┘         └────────┬────────────────┘            │
│           │                                 │                              │
│           ▼                                 ▼                              │
│  ┌────────────────┐              ┌──────────────────┐                     │
│  │  nflfastR API  │              │   Odds API       │                     │
│  │  (nflverse)    │              │  (The Odds API)  │                     │
│  └────────────────┘              └──────────────────┘                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Dependencies

### 1. Presentation Layer Dependencies

#### Streamlit Dashboard (`dashboard/app.py`)

**Depends On:**
- PostgreSQL (direct SQL queries)
  - `public.games`
  - `mart.player_game_stats`
  - `ml.model_registry`
- FastAPI (`http://localhost:8000/predict`)
- Redis (caching layer)

**Dependency Type:** Runtime

**Coupling Level:** HIGH (direct SQL + HTTP calls)

#### FastAPI Service (`api/main.py`)

**Depends On:**
- EnsemblePredictor (`py/ensemble/enhanced_ensemble_v5.py`)
- PostgreSQL (health check)
- Redis (caching)
- Prometheus (metrics)

**Dependency Type:** Runtime

**Coupling Level:** MEDIUM (abstracted through classes)

---

### 2. Application Layer Dependencies

#### EnhancedEnsembleV5 (`py/ensemble/enhanced_ensemble_v5.py`)

**Depends On (Runtime):**
- R Models (via rpy2):
  - `models/bayesian/simplified_random_walk_v2.rds`
  - `models/bayesian/passing_yards_hierarchical_v2.rds`
  - `models/bayesian/passing_informative_priors_v2.rds`
- Python Models:
  - `models/xgboost/passing_xgboost_v2_canonical.pkl`
  - `models/bayesian/simplified_random_walk_skills_v2.csv`
- PostgreSQL (feature queries):
  - `mart.player_game_stats`
  - `mart.player_hierarchy`
  - `public.games`
- Libraries:
  - rpy2 (R interface)
  - xgboost (model loading)
  - psycopg2 (database)

**Dependency Graph:**
```
EnhancedEnsembleV5
├── rpy2.robjects (R interface)
│   ├── requires: R runtime + brms library
│   └── loads: .rds model files
├── xgboost.Booster (XGBoost loader)
│   └── loads: .pkl model files
├── pandas (CSV skills loader)
│   └── loads: .csv state-space skills
└── psycopg2 (database client)
    └── queries: player stats, games, features
```

**Critical Runtime Dependencies:**
1. R runtime installed in container
2. brms R package available
3. Model files present in `/app/models/`
4. PostgreSQL accessible

**Failure Modes:**
- Missing R library → Import error, ensemble crashes
- Corrupted .rds file → Silent prediction failure
- Database connection failure → No features, no predictions
- Missing .csv skills → State-space predictions fail (ensemble continues with 3 models)

---

### 3. Model Training Layer Dependencies

#### R Training Scripts

**Example:** `R/train_hierarchical_v2_production.R`

**Depends On:**
- R Libraries:
  - `brms` (Bayesian regression)
  - `cmdstanr` (Stan backend)
  - `DBI` + `RPostgres` (database)
  - `tidyverse` (data wrangling)
- PostgreSQL:
  - `mart.player_game_stats`
  - `public.games`
  - `mart.player_hierarchy`
- Filesystem:
  - Writes to `models/bayesian/*.rds`
  - Writes to `ml.model_registry` (if registration enabled)

**Dependency Graph:**
```
train_hierarchical_v2_production.R
├── brms::brm() (model training)
│   ├── requires: cmdstanr backend
│   └── writes: .rds file
├── DBI::dbConnect() (database client)
│   └── queries: player stats, games, features
└── saveRDS() (model serialization)
    └── writes: models/bayesian/passing_yards_hierarchical_v2.rds
```

**Training Time:** 30-60 minutes (MCMC sampling)

#### Python Training Scripts

**Example:** `py/models/train_xgboost_canonical.py`

**Depends On:**
- Python Libraries:
  - `xgboost` (gradient boosting)
  - `psycopg2` (database)
  - `pandas` + `numpy` (data processing)
- PostgreSQL:
  - `mart.player_game_stats`
  - `public.games`
- Filesystem:
  - Writes to `models/xgboost/*.pkl`

**Training Time:** 5-10 minutes (CPU) or 1-2 minutes (GPU)

---

### 4. Data Layer Dependencies

#### PostgreSQL Database

**Depends On:**
- TimescaleDB extension (for `odds_history`)
- Disk storage (`pgdata/` volume)
- Network (port 5544 exposed)

**Consumers:**
- Streamlit UI (direct SQL)
- FastAPI service (health check)
- EnhancedEnsembleV5 (feature queries)
- R training scripts (data loading)
- Python training scripts (data loading)

**Critical Tables:**

```
public.plays (1.25M rows)
  ├── Used by: Feature engineering, model training
  ├── Columns: 257 (Phase 5 expansion)
  └── Dependencies: None

public.games (7.2K rows)
  ├── Used by: Ensemble predictions, training
  ├── Columns: spread_close, total_close, kickoff, etc.
  └── Dependencies: None

mart.player_game_stats (24.9K rows)
  ├── Used by: Ensemble predictions, training
  ├── Dependencies: Materialized view → requires refresh
  └── Refresh: Manual (REFRESH MATERIALIZED VIEW)

mart.player_hierarchy (15.2K rows)
  ├── Used by: Ensemble predictions, training
  ├── Dependencies: Derived from public.players
  └── Update: Automatic on player data changes

ml.model_registry (8 rows, 227 pending)
  ├── Used by: Model management, Grafana dashboards
  ├── Dependencies: None
  └── Sync: Manual via model_registry.py
```

#### Redis Cache

**Depends On:**
- Memory (2GB limit configured)
- Network (port 6379 exposed)

**Consumers:**
- FastAPI (prediction caching)
- Streamlit (session state)

**Cache Keys:**
```
predictions:{player_id}:{season}:{week} → JSON prediction
  TTL: 5 minutes

model_status:{model_name} → JSON metadata
  TTL: 1 hour
```

---

## Circular Dependency Detection

### Detected Circular Dependencies

#### Circular Dependency 1: Ensemble ↔ Database

```
EnhancedEnsembleV5 ──queries──▶ PostgreSQL
        ▲                           │
        │                           │
        └────────reads───────────────┘
             (model training)
```

**Issue:** Ensemble queries database for features, but models trained from same database.

**Risk Level:** LOW (expected pattern, not a true circular dependency)

#### Circular Dependency 2: Training Scripts ↔ Model Registry

```
train_hierarchical_v2_production.R ──writes──▶ models/*.rds
        ▲                                           │
        │                                           │
        └────────registers────────────ml.model_registry
                                  (via model_registry.py)
```

**Issue:** Training script writes model, then model_registry.py reads model to register in DB.

**Risk Level:** LOW (sequential dependency, not circular)

### No Critical Circular Dependencies Detected

The system has **proper unidirectional data flow**:
1. Data ingestion → PostgreSQL
2. PostgreSQL → Model training
3. Model training → Filesystem (.rds/.pkl files)
4. Filesystem → Ensemble loading
5. Ensemble → API predictions

---

## Integration Points & Failure Modes

### Integration Point 1: R ↔ Python (rpy2)

**Components:**
- `EnhancedEnsembleV5._get_r_prediction()` (Python)
- R brms models (loaded via rpy2)

**Failure Modes:**
1. R runtime not found → ImportError
2. brms library not installed → R error, Python exception
3. .rds file corrupted → Silent failure (returns None)
4. Version mismatch (brms 2.18 vs 2.20) → Incorrect predictions

**Mitigation:**
- ✅ Docker container pins R version
- ⚠️ brms version not pinned (risk)
- ❌ No checksum validation of .rds files
- ❌ No fallback if R prediction fails

**Recommendation:** Add health check validating R model loading on API startup.

### Integration Point 2: Python ↔ PostgreSQL

**Components:**
- All layers (Streamlit, FastAPI, Ensemble, Training)
- PostgreSQL database

**Failure Modes:**
1. Database connection timeout → psycopg2.OperationalError
2. Table not found → psycopg2.ProgrammingError
3. Materialized view stale → Incorrect predictions (silent failure)
4. Connection pool exhausted → Deadlock

**Mitigation:**
- ✅ Health checks for database connectivity
- ⚠️ No retry logic for transient failures
- ❌ No materialized view freshness monitoring
- ❌ No connection pooling (creates new connection per query)

**Recommendation:** Add connection pooling (SQLAlchemy) and materialized view monitoring.

### Integration Point 3: FastAPI ↔ Ensemble

**Components:**
- `api/main.py` (FastAPI)
- `api/prediction.py` (EnsemblePredictor)
- `EnhancedEnsembleV5` (ensemble logic)

**Failure Modes:**
1. Ensemble initialization fails → API starts but predictions fail
2. Model loading timeout (R models) → API latency spike
3. NaN/Inf in predictions → JSON serialization error
4. Ensemble returns None → 500 Internal Server Error

**Mitigation:**
- ✅ Health check validates ensemble initialization
- ⚠️ No timeout on model loading (can hang)
- ❌ No validation of prediction output (NaN/Inf not caught)
- ❌ No fallback to simpler model if ensemble fails

**Recommendation:** Add prediction validation and fallback logic.

---

## Dependency Version Matrix

| Component | Language | Primary Dependencies | Version Pinning |
|-----------|----------|---------------------|-----------------|
| Streamlit Dashboard | Python 3.11 | streamlit, psycopg2, redis | ✅ requirements.txt |
| FastAPI Service | Python 3.11 | fastapi, uvicorn, psycopg | ✅ api/requirements.txt |
| EnhancedEnsembleV5 | Python 3.11 | rpy2, xgboost, psycopg2 | ✅ requirements.txt |
| State-Space Model | R 4.3 | brms, cmdstanr, tidyverse | ⚠️ renv.lock (not enforced) |
| Hierarchical Model | R 4.3 | brms, DBI, RPostgres | ⚠️ renv.lock (not enforced) |
| XGBoost Training | Python 3.11 | xgboost, pandas, numpy | ✅ requirements.txt |
| PostgreSQL | - | timescaledb-pg16 | ✅ Docker image tag |
| Redis | - | redis:7-alpine | ✅ Docker image tag |

**Risk:** R package versions not enforced in Docker build, leading to potential version drift.

**Recommendation:** Add `renv::restore()` to Docker container build step.

---

## External API Dependencies

### 1. nflfastR API (Data Ingestion)

**Component:** `R/ingest_pbp.R`

**Dependency:**
```R
library(nflfastR)
data <- nflfastR::load_pbp(seasons = 1999:2024)
```

**Failure Modes:**
- API rate limiting → Partial data ingestion
- Schema change → Column mismatch, ingestion failure
- API downtime → No new data

**Mitigation:**
- ⚠️ No retry logic for API failures
- ❌ No schema validation before ingestion
- ✅ Historical data cached in PostgreSQL

### 2. Odds API (Betting Lines)

**Component:** `py/data/ingest_odds_history.py`

**Dependency:**
```python
response = requests.get(f"https://api.the-odds-api.com/v4/sports/...",
                        params={"apiKey": ODDS_API_KEY})
```

**Failure Modes:**
- API key quota exhausted → 429 Too Many Requests
- API downtime → No odds data
- Delayed odds updates → Stale predictions

**Mitigation:**
- ✅ API key stored in .env (not hardcoded)
- ⚠️ No rate limit monitoring
- ❌ No fallback to alternative odds source

---

## Summary of Critical Dependencies

### High-Risk Dependencies (Single Point of Failure)

1. **rpy2 (R Interface)**: Entire ensemble fails if R models cannot load
2. **PostgreSQL**: All components require database access
3. **Model Artifacts (Filesystem)**: No redundancy or checksums
4. **R Runtime in Docker**: Must be present for ensemble to function

### Mitigation Strategies

1. **Add fallback logic**: If R models fail, use XGBoost-only predictions
2. **Implement model checksums**: Validate .rds/.pkl files before loading
3. **Add connection pooling**: Prevent database connection exhaustion
4. **Deploy read replicas**: Reduce load on primary PostgreSQL instance
5. **Implement health checks**: Validate all dependencies on API startup

---

**Document Version:** 1.0
**Last Updated:** November 13, 2025
**Related:** [ARCHITECTURE_ANALYSIS_NOV13_2025.md](ARCHITECTURE_ANALYSIS_NOV13_2025.md)
