NFL Analytics – Local Dev Quickstart

This repo provides R + Python pipelines and a TimescaleDB schema for NFL data (games, plays, weather, and odds history). See AGENTS.md for detailed guidance. Below is a minimal local bootstrap.

Prereqs
- Docker and docker compose
- psql (optional; script falls back to container psql)
- R (4.x) and Python (3.10+) if you plan to run ingestors

Initialize Database
- Start DB and apply schema:
  - `bash scripts/init_dev.sh`

Suggested Next Steps
- Install Python deps: `pip install -r requirements.txt`
- Install R deps: `Rscript -e 'renv::restore()'` or `Rscript setup_packages.R`
- Load schedules (idempotent, 1999–2024):
  - `Rscript --vanilla data/ingest_schedules.R`
- Ingest historical odds (requires `ODDS_API_KEY` in `.env`):
  - `python py/ingest_odds_history.py --start-date 2023-09-01 --end-date 2023-09-03`
- Refresh analytic marts:
  - `psql postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB -c "REFRESH MATERIALIZED VIEW mart.game_summary;"`

Containerized Workflow (local laptop)
- Build and start services:
  - `docker compose up -d --build pg app`
- Inside the `app` container, finish setup and run tasks:
  - `docker compose exec app bash -lc "bash scripts/dev_setup.sh"`
  - `docker compose exec app bash -lc "Rscript --vanilla data/ingest_schedules.R"`
  - `docker compose exec app bash -lc "quarto render notebooks/04_score_validation.qmd"`
  - `docker compose exec app bash -lc "python py/rl/dataset.py --output data/rl_logged.csv --season-start 2020 --season-end 2024"`
  - `docker compose exec app bash -lc "python py/rl/ope_gate.py --dataset data/rl_logged.csv --output analysis/reports/ope_gate.json"`
- Stop services: `docker compose down` (data persists in `pgdata/`).

Local Python via uv (no container)
- Install uv (https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create venv and install: `uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`

Notes
- Database runs on `localhost:5544` (see `docker-compose.yaml`).
- Data volume is mounted at `pgdata/` — do not edit manually.
- Keep secrets in `.env`; do not commit real keys.
