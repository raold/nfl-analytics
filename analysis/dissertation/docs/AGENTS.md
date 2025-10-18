# Repository Guidelines

## Project Structure & Module Organization
- `data/`: R ingestion pipelines (e.g., `ingest_schedules.R`).
- `py/`: Python feature engineering modules; promote reusable notebook code here.
- `db/`: SQL migrations (e.g., `001_init.sql`); extend with numbered files. Odds history persists in `odds_history`.
- `notebooks/`: Quarto workflows (`01_ingest_schedules.qmd`, `02_pbp_features.qmd`, `03_odds_weather_join.qmd`, `04_score_validation.qmd`, `05_copula_gof.qmd`, `10_model_spread_xgb.qmd`, `11_monte_carlo_skellam.qmd`, `12_risk_sizing.qmd`, `80_rl_ablation.qmd`, `90_simulator_acceptance.qmd`). Upstream finalized logic into `py/` or `R/`.
- `mart/`: Auto-created schema/tables for analytics (e.g., `mart.team_epa`, `mart.game_summary` MV).
- `pgdata/`: TimescaleDB volume. Do not edit or commit snapshots.
- Key assets: `py/ingest_odds_history.py`; `data/raw/nflverse_schedules_1999_2024.rds`.

## Build, Test, and Development Commands
- Start DB: `docker compose up -d pg` (TimescaleDB on port 5544).
- Restore R env: `Rscript -e 'renv::restore()'`.
- Python deps: `uv pip install -r requirements.txt` (or `uv sync` if `pyproject.toml`).
- Load schedules/lines: `Rscript --vanilla data/ingest_schedules.R` (idempotent; refreshes 1999–2024).
- Ingest odds history: `python py/ingest_odds_history.py --start-date 2023-09-01 --end-date 2023-09-03` (set `ODDS_API_KEY`).
- RL OPE (scaffold): build `data/rl_logged.csv` via `python py/rl/dataset.py --output data/rl_logged.csv --season-start 2020 --season-end 2024`; evaluate `python py/rl/ope_gate.py --dataset data/rl_logged.csv --output analysis/reports/ope_gate.json`.
- Risk sizing (scaffold): generate scenarios `python py/risk/generate_scenarios.py --bets data/bets.csv --output data/scenarios.csv --sims 20000`; size `python py/risk/cvar_lp.py --scenarios data/scenarios.csv --alpha 0.95 --output analysis/reports/cvar_stakes.json`.
- Quarto renders for dissertation figures/tables: `quarto render notebooks/04_score_validation.qmd`, `quarto render notebooks/05_copula_gof.qmd`, `quarto render notebooks/12_risk_sizing.qmd`, `quarto render notebooks/80_rl_ablation.qmd`.
- Refresh marts: `psql postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB -c "REFRESH MATERIALIZED VIEW mart.game_summary;"`.
- Coverage report: `Rscript R/report_odds_coverage.R`.
- Stop services: `docker compose down` (data persists in `pgdata/`).

## Coding Style & Naming Conventions
- R: tidyverse style; snake_case objects; pipes; explicit `mutate`/`transmute`.
- Python: PEP 8, 4-space indents; modules `snake_case`, classes `PascalCase`.
- SQL: lower-case keywords; snake_case identifiers (match `001_init.sql`).
- Formatting: `styler::style_file()` (R); `uvx -p 3.11 black .` and `uvx -p 3.11 ruff check . --fix` (Python) before PRs.

## Testing Guidelines
- R tests: `R/tests/` with `testthat`; mock DB via temporary schemas.
- Python tests: `py/tests/` with `pytest`; fixtures target the `pg` service.
- Smoke checks: confirm core tables (`games`, `plays`, `odds_history`) after migrations/ingests.
- Integration: `uvx -p 3.11 pytest tests/integration -k ingestion` for idempotency and index/EXPLAIN checks.

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits (`feat:`, `fix:`, `docs:`); each commit should run/compile; avoid mixing unrelated refactors with schema/ingestion changes.
- PRs: include summary, testing notes, schema impacts, re-ingest needs; link issues and attach screenshots or table diffs when analytics outputs change.

## Security & Configuration Tips
- Keep secrets in `.env` or `.Renviron`; never commit real keys. Provide `ODDS_API_KEY` via `.env`.
- Never edit or commit `pgdata/`. Respect rate limits for The Odds API.
