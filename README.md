<p align="center">
  <img src="docs/images/logo.png" alt="NFL Analytics" width="520" />
</p>

<h1 align="center">NFL Analytics</h1>

Research platform for NFL game and player-prop modeling: R (brms/Stan) and Python (XGBoost,
PyMC) models over a PostgreSQL warehouse of nflverse data, plus the code and LaTeX source for a
~720-page dissertation. The trading system that grew out of it now lives in a separate
repository (Sibyl).

## Results, honestly

The full 2025 season was evaluated out of sample (ADR-017,
[docs/investigations/2026-02-season-review/](docs/investigations/2026-02-season-review/)):

| Metric (2025, 285 games) | Model | Closing line |
|---|---|---|
| Spread MAE (points) | 11.05 | 9.67 |
| ATS ROI on 208 qualified bets | −3.2% (95% CI ≈ −16% to +10%) | — |
| Win-probability accuracy | 56.5% | — |

The model does not beat the closing line. Simple baselines graded with
[`nfl_analytics.evaluation.market_benchmark`](nfl_analytics/evaluation/market_benchmark/) on
2019–2024 tell the same story:

| Baseline (2019–2024, edge ≥ 1 pt) | ATS | ROI (95% CI) |
|---|---|---|
| Walk-forward power ratings | 609-608-31, 50.0% | −4.5% (−8.9% to 0.0%) |
| Elo | 581-596-31, 49.4% | −5.8% (−11.5% to −0.4%) |

**Retracted:** a December 2025 claim of "74.9% ATS / +43% ROI" for a power-ratings model came
from a spread sign error, not from the model. See [docs/POSTMORTEMS.md](docs/POSTMORTEMS.md) §4.
`games.spread_close` is the nflverse line (positive = home favored); always grade covers through
[`nfl_analytics/utils/spread_convention.py`](nfl_analytics/utils/spread_convention.py).

## Setup

Requirements: Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/), Docker, R 4.x, Git LFS.

```bash
git lfs install && git lfs pull          # model artifacts (.rds, .pkl) and DB dumps
uv sync --extra dev                      # core + test tooling, from uv.lock
cp .env.example .env                     # then set DB_PASSWORD and API keys
docker compose up -d pg-1 redis          # PostgreSQL (localhost:5544) and Redis
bash scripts/dev/init_dev.sh             # apply the schema
Rscript R/setup_packages.R               # R packages
```

Optional extras, installed as needed: `api`, `trading`, `online`, `torch`, `rl`, `bayes`,
`causal`, `semantic`, `r`, `dashboard`, `compute`, `research`
(for example `uv sync --extra dev --extra torch --extra bayes`). See [SETUP.md](SETUP.md) for
data ingestion, GPU notes and troubleshooting.

## Common commands

```bash
uv run pytest -m "not integration"       # unit tests, no database needed
uv run pytest -m integration             # needs the Docker database

# Benchmark against the closing market (baselines need no database)
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline power_ratings \
    --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024

# Load data
Rscript --vanilla R/ingestion/ingest_schedules.R
Rscript --vanilla R/ingestion/ingest_pbp.R
Rscript --vanilla R/ingestion/ingest_current_season.R
```

## Layout

| Path | Contents |
|---|---|
| `nfl_analytics/` | Python package: features, ensemble, models, evaluation, production API, utils |
| `R/` | brms/Stan models, nflverse ingestion, dissertation figures |
| `db/` | SQL migrations and materialized views |
| `analysis/dissertation/` | LaTeX source (`main/main.tex`) |
| `tests/` | `unit/` runs offline; `integration/`, `infrastructure/`, `e2e/` need services |
| `docs/` | Reference docs, ADRs, postmortems ([index](docs/README.md), [history](docs/HISTORY.md)) |

## Documentation

- [docs/README.md](docs/README.md) — documentation index
- [docs/adr/README.md](docs/adr/README.md) — design decisions, with evidence quality
- [docs/database/DATABASE_QUICK_REFERENCE.md](docs/database/DATABASE_QUICK_REFERENCE.md) — schema and SQL conventions
- [CLAUDE.md](CLAUDE.md) — working notes for AI-assisted development
- [CONTRIBUTING.md](CONTRIBUTING.md)
