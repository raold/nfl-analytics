# NFL Analytics - Quick Start Guide

**Goal**: Get from zero to first prediction in under 30 minutes

**Status**: Production-ready (November 15, 2025)
**Version**: 1.0
**Target Audience**: New developers joining the project

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Database Initialization](#database-initialization)
4. [Data Ingestion](#data-ingestion)
5. [Run Your First Prediction](#run-your-first-prediction)
6. [Validate Installation](#validate-installation)
7. [Next Steps](#next-steps)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

Before starting, ensure you have these installed:

```bash
# Check versions
docker --version        # Docker 20.10+
docker compose version  # Docker Compose 2.0+
git --version          # Git 2.30+
python3 --version      # Python 3.9+
R --version           # R 4.2+

# Optional but recommended
psql --version        # PostgreSQL client 14+
git lfs version       # Git LFS 3.0+
```

### Hardware Requirements

**Minimum:**
- 8 GB RAM
- 20 GB free disk space
- 4 CPU cores

**Recommended:**
- 16 GB RAM (for full ensemble)
- 50 GB free disk space (includes models and data)
- 8+ CPU cores
- GPU (optional): NVIDIA RTX 4090 or Apple M4 for training

### Time Budget

- **Minimal setup** (no data): 5 minutes
- **Full setup** (with data): 20-30 minutes
- **First prediction**: <1 minute after setup

---

## Environment Setup

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics

# Install Git LFS for model files
git lfs install
git lfs pull  # Download model binaries (~2 GB)
```

### Step 2: Create Environment File

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your preferred editor
nano .env  # or vim, code, etc.
```

**Critical settings to review:**

```bash
# Database credentials (CHANGE IN PRODUCTION!)
DB_PASSWORD=sicillionbillions  # pragma: allowlist secret

# Redis password (CHANGE IN PRODUCTION!)
REDIS_PASSWORD=changeme_redis_password

# Optional: API keys for external data
ODDS_API_KEY=your_key_here  # Only needed for real-time odds
```

**Security Note**: The default passwords are for local development only. NEVER use these in production.

### Step 3: Setup Python Environment

**Option A: Using uv (recommended - fastest)**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

uv pip install -r requirements.txt
```

**Option B: Using standard pip**

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Verify Python installation:**

```bash
python -c "import pandas, numpy, sklearn; print('✅ Core packages installed')"
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"
```

### Step 4: Setup R Environment

```bash
# Install R packages using renv (recommended)
Rscript -e 'renv::restore()'

# Alternative: Use setup script
Rscript setup_packages.R
```

**Critical R packages:**
- `nflreadr` - NFL data ingestion
- `brms` / `cmdstanr` - Bayesian modeling
- `tidyverse` - Data manipulation
- `DBI` / `RPostgres` - Database connectivity

**Verify R installation:**

```bash
Rscript -e "library(nflreadr); library(brms); print('✅ R packages installed')"
```

---

## Database Initialization

### Step 1: Create Docker Network

```bash
# Create external network (required for Docker Compose)
docker network create nfl-analytics 2>/dev/null || echo "Network already exists"
```

### Step 2: Start Database Container

```bash
# Initialize database and apply schema
bash scripts/dev/init_dev.sh
```

**What this does:**
1. Starts TimescaleDB (PostgreSQL 16 + TimescaleDB extension)
2. Waits for database to be ready (up to 2 minutes)
3. Applies all database migrations from `db/migrations/`
4. Creates 5 schemas: `public`, `mart`, `ml`, `staging`, `metadata`
5. Creates 56 tables across schemas

**Expected output:**

```
[init] Bringing up TimescaleDB container (service: pg)
[init] Waiting for database readiness on localhost:5544...
[init] Database is ready
[init] Applying schema migrations
[init] Applying 001_init.sql
[init] Applying 002_timescale.sql
...
[init] Done. Next steps:
  - Load schedules: Rscript --vanilla R/ingestion/ingest_schedules.R
  - Ingest odds: python py/ingest_odds_history.py
```

### Step 3: Verify Database

```bash
# Connect to database
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01  # pragma: allowlist secret

# Run basic queries
\dt public.*     # List tables in public schema
\dt mart.*       # List tables in mart schema
SELECT COUNT(*) FROM public.games;  # Should be 0 (no data yet)
\q
```

---

## Data Ingestion

### Option 1: Quick Setup (2024 season only, ~2 minutes)

```bash
# Ingest 2024 schedule only
Rscript -e "
  library(nflreadr)
  library(DBI)
  con <- dbConnect(RPostgres::Postgres(),
    host = 'localhost', port = 5544,
    dbname = 'devdb01', user = 'dro', password = 'sicillionbillions')
  schedules <- nflreadr::load_schedules(2024)
  dbWriteTable(con, c('public', 'games'), schedules, append = TRUE)
  dbDisconnect(con)
  cat('✅ 2024 schedule loaded\n')
"

# Ingest 2024 play-by-play (~30 seconds)
Rscript -e "
  library(nflreadr)
  library(DBI)
  con <- dbConnect(RPostgres::Postgres(),
    host = 'localhost', port = 5544,
    dbname = 'devdb01', user = 'dro', password = 'sicillionbillions')
  pbp <- nflreadr::load_pbp(2024)
  dbWriteTable(con, c('public', 'play_by_play'), pbp, append = TRUE)
  dbDisconnect(con)
  cat('✅ 2024 play-by-play loaded\n')
"
```

### Option 2: Full Setup (1999-2024, ~5 minutes)

```bash
# Ingest full schedules (1999-2024)
Rscript --vanilla R/ingestion/ingest_schedules.R

# Ingest full play-by-play (1999-2024)
Rscript --vanilla R/ingestion/ingest_pbp.R
```

**Expected data volumes:**

| Dataset | Seasons | Rows | Time |
|---------|---------|------|------|
| Schedules | 1999-2024 | 7,263 games | 30 sec |
| Play-by-Play | 1999-2024 | 1.25M plays | 3-5 min |
| Player Stats | Derived | 450K rows | 1 min |

### Step 3: Refresh Materialized Views

```bash
# Refresh game summary view
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \  # pragma: allowlist secret
  -c "REFRESH MATERIALIZED VIEW mart.game_summary;"

# Refresh player stats view
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \  # pragma: allowlist secret
  -c "REFRESH MATERIALIZED VIEW mart.player_game_stats;"
```

**Verify data ingestion:**

```bash
# Check row counts
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "  # pragma: allowlist secret
  SELECT
    (SELECT COUNT(*) FROM public.games) as games,
    (SELECT COUNT(*) FROM public.play_by_play) as plays,
    (SELECT COUNT(*) FROM mart.player_game_stats) as player_stats;
"
```

**Expected output (full dataset):**

```
  games  |  plays  | player_stats
---------+---------+--------------
   7263  | 1254173 |   450000
```

---

## Run Your First Prediction

### Option 1: Quick API Test (Recommended)

```bash
# Start FastAPI server in background
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for models to load (10-15 seconds)
sleep 15

# Run test script
python scripts/quick_api_test.py
```

**Expected output:**

```
============================================================
TEST 1: Health Check
============================================================
Status: 200
✅ API is healthy
Models loaded: 1
Ensemble version: v5.1

============================================================
TEST 2: Player Prediction
============================================================
Status: 200
✅ Prediction: 268.5 ± 32.4 yards
✅ 90% CI: [215.1, 321.9]
✅ Model Components:
   state_space_v2: 265.2 yards
   hierarchical_v2: 272.8 yards
   hybrid_xgb_v1: 270.1 yards
   informative_priors_v2: 264.7 yards
```

### Option 2: Python Script

```python
# Create test_prediction.py
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble

# Initialize ensemble
ensemble = EnhancedEnsemble()
ensemble.load_models()

# Make prediction for Jalen Hurts, 2024 Week 10
result = ensemble.predict(
    player_id="HurtJa00",
    season=2024,
    week=10
)

print(f"Prediction: {result['mean']:.1f} ± {result['std']:.1f} yards")
print(f"90% CI: [{result['ci_lower']:.1f}, {result['ci_upper']:.1f}]")
print(f"Components: {result['components']}")
```

```bash
# Run prediction
python test_prediction.py
```

### Option 3: Interactive R Session

```r
# Load required libraries
library(DBI)
library(brms)

# Connect to database
con <- dbConnect(RPostgres::Postgres(),
  host = "localhost", port = 5544,
  dbname = "devdb01", user = "dro", password = "sicillionbillions")

# Load state-space model
model <- readRDS("models/bayesian/state_space_v2_canonical.rds")

# Get recent player data
player_data <- dbGetQuery(con, "
  SELECT * FROM mart.player_game_stats
  WHERE player_id = 'HurtJa00'
    AND season = 2024
  ORDER BY week DESC
  LIMIT 5
")

# Generate predictions
predictions <- posterior_predict(model, newdata = player_data[1,])
mean_pred <- mean(predictions)
ci <- quantile(predictions, c(0.05, 0.95))

cat(sprintf("Prediction: %.1f yards\n", mean_pred))
cat(sprintf("90%% CI: [%.1f, %.1f]\n", ci[1], ci[2]))

dbDisconnect(con)
```

---

## Validate Installation

### Comprehensive Health Check

```bash
# Run full validation
python scripts/validate_installation.py
```

**Creates this script if it doesn't exist:**

```python
# scripts/validate_installation.py
import sys
import subprocess
from pathlib import Path

def check_database():
    """Check database connectivity."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM public.games")
        count = cur.fetchone()[0]
        conn.close()
        print(f"✅ Database: {count} games loaded")
        return True
    except Exception as e:
        print(f"❌ Database: {e}")
        return False

def check_models():
    """Check model files exist."""
    models_dir = Path("models")
    required = [
        "bayesian/state_space_v2_canonical.rds",
        "bayesian/hierarchical_v2_canonical.rds",
        "xgboost/hybrid_static_incremental_v1/static_model.pkl"
    ]

    all_exist = True
    for model in required:
        path = models_dir / model
        if path.exists():
            print(f"✅ Model: {model}")
        else:
            print(f"❌ Model: {model} (missing)")
            all_exist = False
    return all_exist

def check_python_packages():
    """Check critical Python packages."""
    packages = ["pandas", "numpy", "sklearn", "torch", "fastapi"]
    all_installed = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ Python: {pkg} installed")
        except ImportError:
            print(f"❌ Python: {pkg} missing")
            all_installed = False
    return all_installed

def check_r_packages():
    """Check critical R packages."""
    cmd = """Rscript -e "
    packages <- c('brms', 'nflreadr', 'tidyverse', 'RPostgres')
    for (pkg in packages) {
      if (require(pkg, character.only=TRUE, quietly=TRUE)) {
        cat(sprintf('✅ R: %s installed\\n', pkg))
      } else {
        cat(sprintf('❌ R: %s missing\\n', pkg))
      }
    }
    " """
    subprocess.run(cmd, shell=True)
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("NFL Analytics - Installation Validation")
    print("=" * 60)

    checks = [
        ("Database", check_database),
        ("Models", check_models),
        ("Python Packages", check_python_packages),
        ("R Packages", check_r_packages),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n{name} Check:")
        print("-" * 60)
        results.append(check_fn())

    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! Installation complete.")
        sys.exit(0)
    else:
        print("❌ Some checks failed. See errors above.")
        sys.exit(1)
```

---

## Next Steps

### Week 1: Explore the System

1. **Review documentation**: Start with [CLAUDE.md](../../CLAUDE.md) for project overview
2. **Explore database schema**: See [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md)
3. **Run example backtest**: Try `python py/backtest/baseline_glm.py` (2-3 minutes)
4. **Start dashboard**: Run `streamlit run dashboard/app.py` and explore at http://localhost:8501

### Week 2: Make Your First Contribution

1. **Read developer workflow**: [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md)
2. **Pick a good first issue**: Check GitHub issues labeled "good-first-issue"
3. **Create feature branch**: `git checkout -b feature/your-feature`
4. **Write tests**: Add to `tests/unit/` or `tests/integration/`
5. **Submit PR**: Follow contribution guidelines in [CONTRIBUTING.md](../../CONTRIBUTING.md)

### Week 3: Deep Dive

1. **Train a model**: Follow [COMMON_TASKS_REFERENCE.md](COMMON_TASKS_REFERENCE.md)
2. **Run full backtest**: Validate ensemble on 2022-2024 seasons
3. **Explore causal inference**: See causal framework in `py/causal/`
4. **Contribute to dissertation**: Add to `analysis/dissertation/main/`

---

## Troubleshooting

### Database Won't Start

**Symptom**: `init_dev.sh` times out waiting for database

**Solutions:**

```bash
# Check if port 5544 is already in use
lsof -i :5544

# Kill existing PostgreSQL process
docker compose down
docker volume rm nfl-analytics_pgdata  # WARNING: deletes all data
bash scripts/dev/init_dev.sh
```

### Models Not Loading

**Symptom**: "FileNotFoundError: models/bayesian/..." errors

**Solutions:**

```bash
# Verify Git LFS is installed
git lfs install

# Pull model files
git lfs pull

# Check model files exist
ls -lh models/bayesian/
ls -lh models/xgboost/
```

### Python Package Conflicts

**Symptom**: ImportError or version conflicts

**Solutions:**

```bash
# Clean reinstall
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### R Packages Won't Install

**Symptom**: brms or cmdstanr installation fails

**Solutions:**

```bash
# Install Stan manually first
Rscript -e "install.packages('cmdstanr', repos = c('https://mc-stan.org/r-packages/', getOption('repos')))"
Rscript -e "cmdstanr::install_cmdstan()"

# Then retry renv restore
Rscript -e 'renv::restore()'
```

### Data Ingestion Fails

**Symptom**: "Connection refused" or timeout errors

**Solutions:**

```bash
# Verify database is running
docker compose ps

# Test database connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"  # pragma: allowlist secret

# Check database logs
docker compose logs pg-1 | tail -50
```

### API Won't Start

**Symptom**: "Address already in use" or uvicorn errors

**Solutions:**

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Verify Python environment
which python  # Should be .venv/bin/python
python -c "import fastapi; print('FastAPI installed')"

# Start with debug logging
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

### Out of Memory

**Symptom**: Killed process or "Cannot allocate memory"

**Solutions:**

```bash
# Reduce ensemble size (edit py/ensemble/enhanced_ensemble_v5.py)
# Use only 2 models instead of 4

# Increase Docker memory (Docker Desktop settings)
# macOS: Docker > Preferences > Resources > Memory: 8 GB

# Use lighter models
# XGBoost only (no Bayesian models): much lower memory
```

---

## Common Error Messages

### "Cannot connect to database"

```
DETAIL: Connection refused. Is the server running on host "localhost" and accepting TCP/IP connections on port 5544?
```

**Fix**: Start database with `docker compose up -d pg-1`

### "No module named 'rpy2'"

```
ModuleNotFoundError: No module named 'rpy2'
```

**Fix**: Install R interface: `pip install rpy2`

### "Factor 'LAR' not in training data"

```
Error: New factor levels are not allowed.
```

**Fix**: Use `allow_new_levels=TRUE` in brms predictions. See [R_MODEL_INTERFACE_GUIDE.md](R_MODEL_INTERFACE_GUIDE.md)

### "Git LFS quota exceeded"

```
Error: Your LFS quota has been exceeded.
```

**Fix**:
1. Use HTTPS clone (not SSH): `git clone https://github.com/raold/nfl-analytics.git`
2. Or download models manually from releases page

---

## Success Checklist

Before moving on, verify:

- [ ] Database running and accessible (`docker compose ps`)
- [ ] At least 2024 season data loaded (7,263 games for full dataset)
- [ ] Materialized views refreshed (mart.game_summary, mart.player_game_stats)
- [ ] Python environment activated (`.venv/bin/python`)
- [ ] R packages installed (brms, nflreadr, tidyverse)
- [ ] Model files present (check `models/bayesian/` and `models/xgboost/`)
- [ ] First prediction successful (API test or manual script)
- [ ] No critical errors in validation script

---

## Getting Help

### Documentation

- **Project overview**: [CLAUDE.md](../../CLAUDE.md)
- **Database schema**: [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md)
- **Common tasks**: [COMMON_TASKS_REFERENCE.md](COMMON_TASKS_REFERENCE.md)
- **Production deployment**: [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)
- **Developer workflow**: [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md)

### Support Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation Index**: [DOCUMENTATION_INDEX.md](../../DOCUMENTATION_INDEX.md) - All 397 docs

### Debugging Resources

- **Troubleshooting Master**: [TROUBLESHOOTING_MASTER.md](TROUBLESHOOTING_MASTER.md)
- **Postmortems**: [POSTMORTEMS.md](../POSTMORTEMS.md) - Learn from past mistakes
- **SQL Error Prevention**: [SQL_ERROR_PREVENTION_SUMMARY.md](../database/SQL_ERROR_PREVENTION_SUMMARY.md)

---

**Status**: Guide complete. You should now have a working NFL Analytics installation!

**Time to first prediction**: 20-30 minutes (with full data) or 5-10 minutes (2024 only)

**Next**: Read [DEVELOPER_WORKFLOW.md](DEVELOPER_WORKFLOW.md) to start contributing.
