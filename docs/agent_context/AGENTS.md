# Agent Handbook – Enterprise Edition

Welcome to the production-ready NFL analytics platform. This document orients AI agents and contributors to the reorganized repository, the enterprise ETL framework, and the modelling / monitoring toolchain.

---

## 1. Top-Level Map

```
nfl-analytics/
├── docs/                # Central documentation hub
├── etl/                 # Enterprise ETL framework (config + code)
├── py/                  # Python package (features, models, pricing, risk)
├── R/                   # R ingestion + specialist analytics
├── db/                  # SQL migrations, views, functions, seeds
├── data/                # Raw/processed/staging data caches
├── scripts/             # dev, deploy, maintenance, analysis workflows
├── infrastructure/      # Docker + future IaC placeholders
├── analysis/            # Dissertation + analytic outputs
├── models/              # Model artefacts (trained weights, configs)
├── logs/                # Structured pipeline and ETL logs
└── tests/               # Unit, integration, e2e suites
```

Key documentation lives under `docs/README.md` (setup), `docs/architecture/` (data pipeline diagrams), and `docs/operations/` (deployment, monitoring, troubleshooting). This file is the quick-start guide for agents.

---

## 2. Environment & Tooling Checklist

1. **Services** – `docker compose up -d pg` (database only) or `docker compose up -d --build pg app` (full stack).
2. **Python** – `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt`.
3. **R** – `Rscript -e 'renv::restore()'` (or `Rscript setup_packages.R`).
4. **Credentials** – populate `.env` with `POSTGRES_*`, `ODDS_API_KEY`, etc. Never commit secrets.
5. **Reset** – `bash scripts/dev/reset_db.sh` (drops / recreates schema, runs migrations).

---

## 3. Data Ingestion & Backfill Workflows

### 3.1 R Ingestion / Backfills (still canonical for nflverse extracts)
```
Rscript --vanilla R/ingestion/ingest_schedules.R      # schedules + lines (1999‑present)
Rscript --vanilla R/ingestion/ingest_pbp.R            # play-by-play with EPA
Rscript --vanilla R/backfill_rosters.R                # player & roster history
Rscript --vanilla R/backfill_game_metadata.R          # stadium, QB, coach metadata
Rscript --vanilla R/backfill_pbp_advanced.R           # turnover/penalty enrichments
```
After each backfill run: `psql ... -c "REFRESH MATERIALIZED VIEW mart.game_summary;"`.

### 3.2 Python ETL Framework (enterprise pipelines)
- **Configs** – `etl/config/sources.yaml`, `.../schemas.yaml`, `.../validation_rules.yaml` define sources, schema expectations, and quality rules.
- **Base extractor** – `etl/extract/base.py` implements retry, rate limiting, caching, and logging. Extend it for new sources.
- **Transforms / loads / validators** – extend under `etl/transform/`, `etl/load/`, `etl/validate/` (stubs provided).
- **Monitoring** – `etl/monitoring/{metrics,alerts,logging}.py` centralizes metrics & alerting hooks.
- **Pipelines** – scaffold end-to-end flows under `etl/pipelines/` (daily, weekly, backfill). Current repo includes scaffolding; implementers should wire configs + stage-specific logic when operationalizing.

### 3.3 Odds & Weather APIs
```
python py/ingest_odds_history.py --start-date 2025-09-01 --end-date 2025-09-07
python py/weather_meteostat.py --games-csv data/processed/features/games.csv --output data/raw/weather/hourly.csv
```
Respect API rate limits defined in `etl/config/sources.yaml`.

---

## 4. Feature Engineering

### 4.1 As-of Feature Generation
```
python py/features/asof_features.py \
  --output analysis/features/asof_team_features.csv \
  --write-table mart.asof_team_features \
  --season-start 2003 --season-end 2025 --validate
```
- Enhanced metadata (QB/coach tenure, surface/roof deltas) lives in `analysis/features/asof_team_features.csv`.
- Alternate generator with experimental stats: `python py/features/asof_features_enhanced.py --output analysis/features/asof_team_features.csv`.

### 4.2 R Feature Enrichment
```
Rscript --vanilla R/features/features_epa.R          # EPA summaries
Rscript --vanilla R/features/features_4th_down.R     # 4th-down aggressiveness
Rscript --vanilla R/features/features_injury_load.R  # Injury impact metrics
```

---

## 5. Modelling & Evaluation Workflows

### 5.1 Baseline GLM & Reliability
```
python py/backtest/baseline_glm.py \
  --features-csv analysis/features/asof_team_features.csv \
  --start-season 2003 --end-season 2025 \
  --output-csv analysis/results/glm_baseline_metrics.csv \
  --preds-csv analysis/results/glm_baseline_preds.csv \
  --tex analysis/dissertation/figures/out/glm_baseline_table.tex

python py/backtest/harness.py \
  --features-csv analysis/features/asof_team_features.csv \
  --start-season 2003 --end-season 2025 \
  --thresholds 0.45,0.50,0.55 \
  --calibrations none,platt,isotonic --cv-folds 5 \
  --cal-bins 10 --cal-out-dir analysis/reports/calibration \
  --output-csv analysis/results/glm_harness_metrics.csv \
  --tex analysis/dissertation/figures/out/glm_harness_table.tex \
  --tex-overall analysis/dissertation/figures/out/glm_harness_overall.tex
```
Warnings about pandas `infer_datetime_format` / `GroupBy.apply` have been resolved; expect clean logs.

### 5.2 Multi-Model & Ensemble Harness (new)
```
python py/backtest/harness_multimodel.py \
  --features-csv analysis/features/asof_team_features.csv \
  --seasons 2003-2025 \
  --threshold 0.55 \
  --output-csv analysis/results/multimodel_comparison.csv \
  --per-season-csv analysis/results/multimodel_per_season.csv \
  --predictions-csv analysis/results/multimodel_predictions.csv \
  --diagnostics-dir analysis/results/multimodel_diagnostics \
  --bootstrap-samples 200
```
Produces:
- Overall + per-season metrics (Brier, LogLoss, Accuracy, ROI)
- Ensemble variants (equal weight + stacked logistic for every subset)
- Diagnostics (`analysis/results/multimodel_diagnostics/`):
  - `glm_importance.csv`, `xgb_importance.csv` (feature importances)
  - `metric_cis.json` (bootstrap 95% CIs)
  - `residual_correlation.csv` (prediction residual covariance matrix)

### 5.3 RL / OPE / Simulation
```
python py/rl/dataset.py --output data/rl_logged.csv --season-start 2020 --season-end 2025
python py/rl/ope_gate.py --dataset data/rl_logged.csv --policy policy.json \
  --output analysis/reports/ope_gate.json --grid-clips 5,10,20 --grid-shrinks 0.0,0.1,0.2 --alpha 0.05 \
  --tex analysis/dissertation/figures/out/ope_grid_table.tex
python py/sim/acceptance.py --hist analysis/reports/sim_hist.json --sim analysis/reports/sim_run.json \
  --output analysis/reports/sim_accept.json --tex analysis/dissertation/figures/out/sim_acceptance_table.tex
```

### 5.4 Pricing & Risk
```
python py/pricing/teaser_ev.py --start 2020 --end 2024 --teaser 6 --price -120 \
  --dep gaussian --rho 0.10 --sensitivity \
  --tex analysis/dissertation/figures/out/teaser_ev_oos_table.tex \
  --png analysis/dissertation/figures/out/teaser_pricing_copula_delta.png

python py/risk/generate_scenarios.py --bets data/bets.csv --output data/scenarios.csv --sims 20000
python py/risk/cvar_lp.py --scenarios data/scenarios.csv --alpha 0.95 --output analysis/reports/cvar_a95.json
python py/risk/cvar_report.py --json analysis/reports/cvar_a95.json --json analysis/reports/cvar_a90.json \
  --tex analysis/dissertation/figures/out/cvar_benchmark_table.tex
```

### 5.5 One-Button Pipeline
- `bash scripts/analysis/run_reports.sh`
  - Regenerates features, GLM tables, OPE, CVaR, teaser EV, simulator acceptance, and rebuilds the dissertation PDF.

---

## 6. Validation, Monitoring, and QA

- **Schema validation** – `etl/config/schemas.yaml` defines expected columns/types; extend `etl/validate/schemas.py` (scaffold) to enforce.
- **Data quality rules** – captured in `etl/config/validation_rules.yaml` (null thresholds, ranges, dedupe keys).
- **Monitoring hooks** – use `etl/monitoring/metrics.py` for pipeline metrics, `alerts.py` for notification wiring (Slack/email), and `logging.py` for structured JSON logs (currently logs to `logs/pipeline.log`).
- **Diagnostics outputs** – `analysis/results/multimodel_diagnostics/` now holds bootstrap CIs, covariance matrices, and feature importances for downstream risk analysis (stacked error, covariance-aware portfolios).
- **Automated validation** – run `bash scripts/dev/run_tests.sh` (wraps `pytest`) and `bash scripts/dev/setup_env.sh` to ensure configs & env consistent. Nightly GitHub Action `nightly-data-quality` executes schema + data quality checks.

---

## 7. Testing & CI

```
bash scripts/dev/setup_testing.sh       # one-time test DB prep
pytest tests/unit                       # unit tests
pytest tests/integration -m integration # DB integration tests
pytest --cov=py --cov-report=html       # coverage (reports in htmlcov/)
```

GitHub Actions pipelines:
- `test.yml` – runs linting + unit/integration suites
- `pre-commit.yml` – enforces `black`, `ruff`, `styler`, YAML checks
- `nightly-data-quality.yml` – ingests sample data, runs ETL validation + monitoring smoke tests

---

## 8. Documentation Pointers

- `docs/setup/development.md` – full local setup (Docker, env vars, DB migrations)
- `docs/architecture/data_pipeline.md` – ETL diagrams, flow descriptions, and contract expectations
- `docs/operations/monitoring.md` – metrics, alert routing, dashboard placeholders
- `docs/reports/` – historical backfill results, audits, production certification reports

When updating agent guidance, ensure both this file and companion docs (`CLAUDE.md`, `GEMINI.md`) stay aligned.

---

## 9. Data Governance & Security

- Secrets live in `.env` (local) or environment variables in deployment; never commit keys.
- `data/raw/` holds cached source pulls – scrub before publishing branches.
- ETL logs and monitoring outputs in `logs/` may contain sensitive metadata (API responses, pipeline state). Rotate and sanitize for shared environments.
- `pgdata/` is a mounted volume – **never** commit or modify directly.

---

## 10. Next Steps / Open Hooks

- Flesh out ETL pipeline implementations (`etl/pipelines/`) – connect configs to extract/transform/load modules.
- Extend validation modules (`etl/validate/`) to enforce schemas & quality rules programmatically.
- Wire monitoring outputs to real alert channels (Slack/Email/Webhooks).
- Expand ensemble harness (e.g., Bayesian model averaging, covariance-aware Kelly sizing) using diagnostics already emitted.
- Keep documentation synchronized: update `docs/architecture/*.md` when changing the data flow.

Happy shipping!
