# NFL Analytics – Local Dev Quickstart

This repo provides R + Python pipelines and a TimescaleDB schema for NFL data (games, plays, weather, and odds history). See **AGENTS.md** and **CLAUDE.md** for detailed guidance. Below is a minimal local bootstrap.

## Prerequisites
- Docker and docker compose
- psql (optional; script falls back to container psql)
- R (4.x) and Python (3.10+) if you plan to run ingestors
- Git for version control

## Quick Start

### 1. Initialize Database
Start the database and apply schema:
```bash
bash scripts/dev/init_dev.sh
```

### 2. Setup Python Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing
```

### 3. Setup R Environment
```bash
# Install R packages
Rscript -e 'renv::restore()'
# OR
Rscript setup_packages.R
```

### 4. Ingest Data

**Load schedules** (idempotent, 1999–2024):
```bash
Rscript --vanilla data/ingest_schedules.R
```

**Ingest play-by-play** (1999-2024, ~3-5 minutes):
```bash
Rscript --vanilla data/ingest_pbp.R
```

**Ingest historical odds** (requires `ODDS_API_KEY` in `.env`):
```bash
export ODDS_API_KEY="your_key_here"
python py/ingest_odds_history.py --start-date 2023-09-01 --end-date 2023-09-10
```

**Refresh materialized views**:
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
# Optional: refresh enhanced features view (if used)
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT mart.refresh_game_features();"
```

### 5. Build Features & Run Models

**Build as-of features** (leakage-safe, game-level):
```bash
python py/features/asof_features.py \
  --output analysis/features/asof_team_features.csv \
  --season-start 1999 \
  --season-end 2024 \
  --validate
```

**Run baseline GLM ATS backtest**:
```bash
python py/backtest/baseline_glm.py \
  --start-season 2003 \
  --end-season 2024 \
  --output-csv analysis/results/glm_baseline_metrics.csv \
  --tex analysis/dissertation/figures/out/glm_baseline_table.tex
```

Optional: apply probability calibration (Platt or isotonic) and change decision thresholds:
```bash
python py/backtest/baseline_glm.py \
  --start-season 2003 --end-season 2024 \
  --calibration platt --cv-folds 5 \
  --decision-threshold 0.50 \
  --cal-plot analysis/dissertation/figures/out/glm_calibration_platt.png \
  --cal-csv analysis/results/glm_calibration_platt.csv \
  --output-csv analysis/results/glm_baseline_metrics_cal_platt.csv \
  --tex analysis/dissertation/figures/out/glm_baseline_table_cal_platt.tex
```

Sweep thresholds and compare configs (harness):
```bash
python py/backtest/harness.py \
  --features-csv analysis/features/asof_team_features.csv \
  --start-season 2003 --end-season 2024 \
  --thresholds 0.45,0.50,0.55 \
  --calibrations none,platt,isotonic --cv-folds 5 \
  --cal-bins 10 --cal-out-dir analysis/results/calibration \
  --output-csv analysis/results/glm_harness_metrics.csv \
  --tex analysis/dissertation/figures/out/glm_harness_table.tex \
  --tex-overall analysis/dissertation/figures/out/glm_harness_overall.tex
```

This writes per‑season and overall reliability CSVs/plots under `analysis/results/calibration/` and emits an overall comparison table with ECE/MCE alongside Brier/LogLoss.

## Testing

This project includes comprehensive unit tests, integration tests, and CI/CD workflows.

### Quick Test Commands
```bash
# Setup testing environment (one-time)
bash scripts/setup_testing.sh

# Run unit tests (fast, no DB required)
pytest tests/unit -m unit

# Run integration tests (requires Docker + Postgres)
docker compose up -d pg
pytest tests/integration -m integration

# Run all tests with coverage
pytest --cov=py --cov-report=html
open htmlcov/index.html
```

### Pre-commit Hooks
```bash
# Install hooks (automatic code quality checks)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### CI/CD (GitHub Actions)
Three automated workflows run on push/PR:
- **Test Suite**: Unit tests, integration tests, coverage reporting
- **Pre-commit**: Code quality and formatting checks
- **Nightly Data Quality**: Schema validation and data integrity checks

See **tests/README.md** and **tests/TESTING.md** for detailed testing documentation.

## Containerized Workflow (local laptop)

**Build and start services**:
```bash
docker compose up -d --build pg app
```

**Run tasks inside container**:
```bash
# Setup
docker compose exec app bash -lc "bash scripts/dev_setup.sh"

# Data ingestion
docker compose exec app bash -lc "Rscript --vanilla data/ingest_schedules.R"

# Render notebooks
docker compose exec app bash -lc "quarto render notebooks/04_score_validation.qmd"

# RL pipeline
docker compose exec app bash -lc "python py/rl/dataset.py --output data/rl_logged.csv --season-start 2020 --season-end 2024"
docker compose exec app bash -lc "python py/rl/ope_gate.py --dataset data/rl_logged.csv --output analysis/reports/ope_gate.json"
```

**Stop services**:
```bash
docker compose down  # Data persists in pgdata/
```

## Local Python via uv (no container)

Install uv: https://docs.astral.sh/uv/
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Project Structure

```
nfl-analytics/
├── py/                     # Python modules (features, models, pricing)
├── R/                      # R utilities
├── data/                   # Data ingestion scripts
├── db/                     # SQL schema and migrations
├── notebooks/              # Quarto analysis notebooks
├── tests/                  # Test suite (unit, integration, e2e)
├── scripts/                # Automation scripts
├── analysis/               # Outputs, reports, dissertation
├── docker/                 # Docker configuration
├── .github/workflows/      # CI/CD workflows
└── pgdata/                 # PostgreSQL data volume (do not edit)
```

## Key Files

- **CLAUDE.md**: Comprehensive project documentation for AI assistants
- **AGENTS.md**: Repository guidelines and patterns
- **requirements.txt**: Python dependencies
- **requirements-dev.txt**: Testing and development tools
- **renv.lock**: R package versions
- **pytest.ini**: Test configuration
- **.pre-commit-config.yaml**: Pre-commit hook configuration

## Database

- **Host**: `localhost:5544`
- **Database**: `devdb01`
- **User**: `dro`
- **Schema**: See `db/001_init.sql` and `db/002_timescale.sql` (overview below)

### Schema Overview
- public
  - `games` (game_id PK) – core game metadata and lines
  - `plays` ((game_id, play_id) PK) – play-by-play with EPA
  - `weather` (game_id PK) – temp_c, wind_kph, humidity, pressure, precip_mm
  - `injuries` – per-game injury status records
  - `odds_history` (Timescale hypertable) – bookmaker/market snapshot history
- mart
  - `mart.game_summary` (materialized view) – enriched game-level summary
  - `mart.game_weather` (materialized view) – derived weather features
  - `mart.team_epa` (table) – per-game EPA summaries by team
  - `mart.team_4th_down_features` (table) – 4th-down decision metrics
  - `mart.team_playoff_context` (table) – playoff probabilities/status
  - `mart.team_injury_load` (table) – injury load metrics by team-week
  - `mart.game_features_enhanced` (materialized view) – composite modeling features

Full documentation and lineage: `docs/database/schema.md`.

**Current Data**:
- Games: 6,991 rows (1999-2024)
- Plays: 1,230,857 rows (1999-2024)
- Odds: 820,080 rows (Sept 2023 - Feb 2025, 16 bookmakers, 3 markets)

## Notes

- Database runs on `localhost:5544` (see `docker-compose.yaml`)
- Data volume is mounted at `pgdata/` — do not edit manually
- Keep secrets in `.env`; do not commit real keys
- GLM baseline table is auto-included in Chapter 4 if present: `analysis/dissertation/figures/out/glm_baseline_table.tex`
- Test coverage target: 60%+ overall, 80%+ for critical paths

## Getting Help

- **Testing issues**: See `tests/README.md`
- **Project context**: See `CLAUDE.md`
- **Repository patterns**: See `AGENTS.md`
- **CI/CD failures**: Check `.github/workflows/` logs
