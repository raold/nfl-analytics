<p align="center">
  <img src="docs/images/logo.png" alt="NFL Analytics" width="520" />
</p>

<h1 align="center">NFL Analytics</h1>

A research archive on predicting NFL games against the betting market: a PostgreSQL warehouse of
nflverse data, the models built on it in 2025-2026, the tool that graded them against the closing
line, and the LaTeX source of a ~720-page dissertation. Work stopped in October 2026 with a
negative result. The trading system that grew out of it lives in a separate repository (Sibyl).

## Results

No model in this project beats the NFL closing line. The full grade, with the scripts that
produce every number below, is in
[docs/investigations/2026-10-final-evaluation/](docs/investigations/2026-10-final-evaluation/README.md).
Spread convention throughout: `spread_close` is the expected home margin (positive = home
favored), the home team covers when margin - line > 0, and a model's edge is prediction - line.

**Game Outcome v2 on the 2025 season** (285 games, one prediction run made after the season, so
retroactive; [model README](models/production/game_outcome_v2/README.md)):

| Measure | Game Outcome v2 | Closing market |
|---|---|---|
| Spread MAE, all 285 games | 11.05 | 9.67 (Diebold-Mariano p = 2e-6) |
| Winners picked (model: P(home win) > 0.5; the one tie left out) | 162/284 = 57.0% | favorite 187/284 = 65.8% |
| Win-probability log-loss (market: moneyline, vig removed proportionally) | 0.6871 | 0.6070 (coin flip 0.6931) |
| ATS, a bet on every game at -110 | 139-145-1, 48.9%, ROI -6.6% (95% CI -15.2% to +1.6%) | |
| ATS, only \|edge\| > 2 points (the old "-3.2% on 208 bets" headline) | 105-102-1, 50.7%, ROI -3.2% | |
| ATS, regular season, \|edge\| ≥ 1 at listed prices (`market_benchmark` default) | 117-124-1, 48.5%, ROI -7.1% (-19.2% to +3.4%) | |
| Coefficient on prediction - line when the margin is regressed on both | -0.49 (95% CI -1.36 to +0.37) | |

Break-even at -110 is 52.38%. The win-rate and ROI intervals are wide enough to hide a small
edge; MAE and log-loss are not, and both favor the market.

**Walk-forward baselines** (no database needed; these rows reproduce exactly):

| Baseline, 2019-2024, \|edge\| ≥ 1, at -110 | ATS | ROI (95% CI) |
|---|---|---|
| Power ratings (point margins to date, shrunk toward last season; home field 2.5) | 609-608-31, 50.0% | -4.5% (-8.9% to 0.0%) |
| Elo (FiveThirtyEight's NFL update with 48 points of home field) | 581-596-31, 49.4% | -5.8% (-11.5% to -0.4%) |

Over 2019-2025 on devdb01 at listed prices, Elo and the power ratings lost 5.3% and 4.3%, a Kalman
state-space model (its three parameters tuned on 2024) 4.9%, and the October 2025 GLM walk-forward
over 2004-2024 won 51.0% of 3,826 bets (ROI -2.6% at -110). On the 94 completed 2026 games Elo is
35-30-4, too few to mean anything.

**Retracted.** The December 2025 "74.9% ATS / +43% ROI" power-ratings result was a spread sign
error; every other edge claim traced to the sign bug, leakage, in-sample tuning, or simulated or
hard-coded numbers. The [retraction ledger](docs/investigations/2026-10-final-evaluation/retraction_ledger.md)
lists all 44 headline numbers and their corrected values, and
[docs/POSTMORTEMS.md](docs/POSTMORTEMS.md) explains the failures. The dissertation text was not
changed: [analysis/dissertation/ERRATA.md](analysis/dissertation/ERRATA.md) records its errors,
including an audit that found the numbers in 176 of its 290 table units and 30 of its 51 figures
made up.

## Setup

Requirements: Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/). The database is optional:
the benchmark and the v2 grade run from CSV files. For the database: Docker, and Git LFS only if
you want the December 2025 dumps in `backups/`.

```bash
uv sync --extra dev                      # core + test tooling, from uv.lock
cp .env.example .env                     # fill in DB_PASSWORD (the dev password is in docker-compose.yml)
set -a; . ./.env; set +a                 # most code reads the environment, not .env
bash scripts/dev/init_dev.sh             # start PostgreSQL 17 + TimescaleDB (127.0.0.1:5544), apply the schema to an empty DB
```

The schema comes without data. Loading data, restoring dumps and what the public dumps in
`backups/` contain are covered in [db/README.md](db/README.md) and [SETUP.md](SETUP.md).

## Common commands

```bash
uv run pytest -m "not integration"       # offline tests, what CI runs
uv run pytest -m integration             # needs the database and DB_PASSWORD (see tests/README.md)

# Baselines against the closing market from the bundled CSV
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline power_ratings \
    --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024

# The current season straight from nflverse
curl -sSfL -o games.csv https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline elo \
    --nflverse-csv games.csv --seasons 2026

# Prospective predictions for a week not yet played, then grade them after the games
uv run python -m nfl_analytics.evaluation.market_benchmark predict --baseline elo \
    --nflverse-csv games.csv --season 2026 --week 9 --out preds_2026_w09_elo.csv
uv run python -m nfl_analytics.evaluation.market_benchmark \
    --predictions-csv preds_2026_w09_elo.csv --nflverse-csv games.csv --seasons 2026

# Game Outcome v2's 2025 grade, without the database
uv run python scripts/final_evaluation/grade_game_outcome_v2_2025.py --out-dir /tmp/v2_grade
```

Any model can be graded: a CSV with `game_id` and the predicted home margin
(`--predictions-csv preds.csv --pred-col <column>`), against `--db`, `--nflverse-csv` or
`--market-csv`. Options and guardrails:
[docs/investigations/2026-10-market-benchmark.md](docs/investigations/2026-10-market-benchmark.md).

## Layout

| Path | Contents |
|---|---|
| `nfl_analytics/evaluation/market_benchmark/` | the grading tool |
| `nfl_analytics/utils/spread_convention.py` | the spread sign convention, pinned by tests on real games |
| `nfl_analytics/` (rest) | Game Outcome v2's model and feature extractors, state-space and power-rating models, the 2004-2024 walk-forward harness, DB utilities |
| `models/production/game_outcome_v2/` | frozen v2 boosters (DO NOT DEPLOY) |
| `scripts/final_evaluation/` | the scripts behind the final evaluation and the retraction ledger |
| `scripts/` (rest) | odds panel, market facts, data-provenance scrapers, dissertation tooling |
| `data/` | small tracked inputs: the benchmark CSV (`historical_weather_2018_2024.csv`), v2's 2025 predictions, stadium, travel, coaching and international-game tables |
| `analysis/dissertation/` | LaTeX source (`main/main.tex`) and `ERRATA.md`; `analysis/results/` holds the 2004-2024 walk-forward predictions |
| `db/` | schema snapshot, migration history, materialized views |
| `tests/` | `unit/` runs offline; `integration/` and `infrastructure/` need the database |
| `docs/` | [index](docs/README.md): investigations, ADRs, postmortems, history |

The October 2026 prune removed the trading stack, API, dashboards, R pipeline, notebooks and
most docs; they are recoverable from the local tag `archive/pre-prune-2026-10`
([docs/HISTORY.md](docs/HISTORY.md)).

## Documentation

- [docs/README.md](docs/README.md): documentation index
- [PROJECT_STATUS.md](PROJECT_STATUS.md): what was done in the salvage and what is left for the owner
- [docs/adr/README.md](docs/adr/README.md): design decisions and how their evidence held up
- [docs/database/DATABASE_QUICK_REFERENCE.md](docs/database/DATABASE_QUICK_REFERENCE.md): schema facts and SQL traps
- [CLAUDE.md](CLAUDE.md): working notes and rules for AI-assisted work
- [CONTRIBUTING.md](CONTRIBUTING.md)
