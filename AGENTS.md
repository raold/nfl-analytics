# Repository Guidelines

## Project Structure & Module Organization
- `data/` contains R ingestion pipelines such as `ingest_schedules.R`.
- `py/` is reserved for Python feature engineering modules; promote reusable notebook code here.
- `db/` holds SQL migrations (e.g., `001_init.sql`); extend with numbered files. Odds history persists in `odds_history`.
- `notebooks/` holds Quarto workflows (`01_ingest_schedules.qmd`, `02_pbp_features.qmd`, `03_odds_weather_join.qmd`, `04_score_validation.qmd`, `05_copula_gof.qmd`, `10_model_spread_xgb.qmd`, `11_monte_carlo_skellam.qmd`, `12_risk_sizing.qmd`, `80_rl_ablation.qmd`, `90_simulator_acceptance.qmd`); upstream finalized logic into `py/` or `R/`.
- `mart/` schema is created automatically (`mart.team_epa` table + `mart.game_summary` materialized view) for analytic queries.
- `pgdata/` is the mounted TimescaleDB volume; never edit manually or commit snapshots.
- `py/ingest_odds_history.py` fetches The Odds API snapshots and writes to `odds_history`.
- `data/raw/nflverse_schedules_1999_2024.rds` caches the nflverse schedule/lines pull used by the R ingestor.

## Build, Test, and Development Commands
- `docker compose up -d pg` boots the local TimescaleDB instance on port 5544.
- `Rscript -e 'renv::restore()'` restores the locked R environment from `renv.lock`.
- `pip install -r requirements.txt` provisions the Python toolchain listed in `requirements.txt`.
- `Rscript --vanilla data/ingest_schedules.R` runs the idempotent schedule loader against the local database; rerun after schema updates to refresh moneylines, spread odds, and totals (1999-2024).
- `python py/ingest_odds_history.py --start-date 2023-09-01 --end-date 2023-09-03` ingests historical odds (set `ODDS_API_KEY`, defaults to spreads/totals markets, and respect rate limits).
- RL OPE (scaffold):
  - Build logged dataset: `python py/rl/dataset.py --output data/rl_logged.csv --season-start 2020 --season-end 2024`
  - Evaluate grid: `python py/rl/ope_gate.py --dataset data/rl_logged.csv --output analysis/reports/ope_gate.json`
- Risk sizing (scaffold):
  - Generate scenarios: `python py/risk/generate_scenarios.py --bets data/bets.csv --output data/scenarios.csv --sims 20000`
  - CVaR sizing: `python py/risk/cvar_lp.py --scenarios data/scenarios.csv --alpha 0.95 --output analysis/reports/cvar_stakes.json`
- `docker compose down` stops services while preserving data in `pgdata/`.
- `Rscript R/report_odds_coverage.R` regenerates `data/raw/odds_coverage_by_season.csv`; GitHub Actions (`.github/workflows/nightly-etl.yml`) runs this nightly.
- Quarto notebooks (locally): run `quarto render` on `04_score_validation.qmd`, `05_copula_gof.qmd`, `12_risk_sizing.qmd`, and `80_rl_ablation.qmd` as needed for dissertation figures/tables.
- `psql postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_PORT/$POSTGRES_DB -c "REFRESH MATERIALIZED VIEW mart.game_summary;"` refreshes analytic marts after play ingests.

## Coding Style & Naming Conventions
- Prefer tidyverse style in R: snake_case objects, pipes for transformations, explicit `transmute`/`mutate`.
- Python code should follow PEP 8 with 4-space indents; module names in snake_case, classes in PascalCase.
- SQL migrations use lower-case keywords and snake_case identifiers to match `001_init.sql`.
- Check in formatted code; use `styler::style_file()` for R and `black` or `ruff format` for Python before opening a PR.

## Testing Guidelines
- Add R tests under `R/tests/` with `testthat`; mock database calls using temporary schemas when possible.
- Python tests belong in `py/tests/` using `pytest`; isolate DB dependencies with fixtures that target the `pg` service.
- Include smoke checks confirming tables (`games`, `plays`, etc.) exist after migrations and ingestion runs.
- Document dataset expectations in the PR so reviewers can reproduce results.
- Run `pytest tests/integration -k ingestion` to validate idempotent schedule loads and index usage inside a disposable Docker Postgres instance.

## Commit & Pull Request Guidelines
- No history exists yet; adopt Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) for clarity.
- Each commit should compile and run; avoid bundling unrelated refactors with ingestion or schema changes.
- PRs need a summary, testing notes, and call out schema impacts or required re-ingests.
- Link related issues or notebooks, and attach screenshots or table diffs when changing analytics outputs.

## Data & Security Notes
- Store credentials in `.env` or local `.Renviron`; never hard-code secrets in scripts or checked-in configs.
- Provide `ODDS_API_KEY` via `.env` when running odds ingestion; never commit real keys.
- Scrub `pgdata/` before sharing branches; large binary diffs slow reviews and CI.

## Workflow Roadmap
- Automate nightly ETL (schedules + odds) and publish coverage checks (`data/raw/odds_coverage_by_season.csv`).
- Design analytic marts (e.g., `mart_team_epa`, `mart_game_summary`) joining games, odds_history, and play-level features.
- Productionize modeling notebooks by exporting model artifacts and serving dashboards (Quarto/Plotly).
- Add integration tests that spin up Postgres via Docker, run ingestors, and assert idempotency with EXPLAIN-plan checks on new indexes.
