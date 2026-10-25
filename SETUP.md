# Setup

How to get a working checkout of the archive. Most of it needs no database: the tests CI runs,
the market benchmark and the Game Outcome v2 grade all work from files in the repo or from the
public nflverse schedule file.

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 (CI uses 3.11) | managed by [uv](https://docs.astral.sh/uv/); dependencies locked in `uv.lock` |
| Docker | any recent | only for the database |
| PostgreSQL + TimescaleDB | 17 + 2.24 | image `timescale/timescaledb:latest-pg17` (devdb01 runs 17.7 with TimescaleDB 2.24.0) |
| Git LFS | any | only for the two December 2025 database dumps in `backups/` |
| TeX Live with `latexmk` | any recent | only for the dissertation |

R is no longer needed. Three R scripts are left, all dissertation figure tooling:
`R/dissertation_figures.R`, `analysis/dissertation/figures/R/gnn_optimization_figures.R` and
`scripts/generate_mcmc_traces_figure.R` (they need ggplot2 and friends). The first two type their
numbers in or simulate them (see the known issues in [PROJECT_STATUS.md](PROJECT_STATUS.md));
do not regenerate figures from them.

## Python environment

```bash
uv sync --extra dev                  # core + pytest, black, ruff, pre-commit
uv run pytest -m "not integration"   # the offline suite CI runs
```

Run anything with `uv run ...`, or activate `.venv`. The optional extras in `pyproject.toml`
(`torch`, `bayes`, `causal`, ...) were for subsystems the October 2026 prune removed; the only
offline test that needs one is `tests/unit/test_evidential_regression.py` (`--extra torch`; it
skips without it).

## Database

```bash
cp .env.example .env                 # fill in DB_PASSWORD; the dev password is in docker-compose.yml
set -a; . ./.env; set +a             # export it: only two scripts read .env themselves
make db                              # docker compose up -d pg-1: nfl-analytics-pg-1 on 127.0.0.1:5544, data in ./pgdata
bash scripts/dev/init_dev.sh         # waits for devdb01 and applies db/schema/ if it has no tables yet
```

Code reads the connection from `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` and `DB_PASSWORD`
(`.env.example` lists the other names a few modules use). For psql inside the container:
`docker exec -it nfl-analytics-pg-1 psql -U dro -d devdb01`.

A fresh `pgdata/` gives an empty `devdb01`, and `init_dev.sh` gives it the schema but no rows.
[db/README.md](db/README.md) has the details:

- **Schema only** from the snapshot `db/schema/devdb01_schema_2026-10-24.sql`. The files in
  `db/migrations/` are history: applied in order they do not rebuild today's schema.
- **Schema and data** from a dump, with `scripts/maintenance/restore.sh` (`make restore`). The full
  dumps of the repaired database are private (`backups/local/`; they contain licensed odds data).
  The two public dumps in `backups/` (Git LFS) are from 2025-12-07: they predate the 2026-10-24
  repair (old kickoffs, mislabelled playoff games, the wrong-sign cover views), lack several later
  tables, including `ml.game_outcome_predictions`, and hold `odds_history` empty.

Rules for any write: back up first (`make backup`), never `DELETE FROM games` (plays and
participation cascade), and derive local times with `kickoff AT TIME ZONE 'America/New_York'`
([CLAUDE.md](CLAUDE.md#critical-rules)).

There is no ingestion pipeline any more. For new games, use nflverse's schedule file directly:

```bash
T=$(mktemp -d)                       # keep the download out of the repo
curl -sSfL -o $T/games.csv https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline elo --nflverse-csv $T/games.csv --seasons 2026
```

## Pre-commit

```bash
uvx pre-commit@4.7.0 install         # optional; CI runs the same lint subset, black and the sign guard
uvx pre-commit@4.7.0 run --all-files
make lint                            # the CI lint steps, the guard's self-test and the guard
```

## Dissertation

```bash
cd analysis/dissertation/main && latexmk -pdf -interaction=nonstopmode main.tex
```

Read [analysis/dissertation/ERRATA.md](analysis/dissertation/ERRATA.md) before using any number in
it.

## Troubleshooting

- `password authentication failed`: `DB_PASSWORD` in your shell or `.env` does not match the
  container's `POSTGRES_PASSWORD` (set when `pgdata/` was first initialised).
- `connection refused` on 5544: `docker compose ps`; `docker logs nfl-analytics-pg-1 --tail 50`.
- `ModuleNotFoundError: torch`: install the extra (`uv sync --extra dev --extra torch`) or ignore;
  the test skips.
- The integration tests (`uv run pytest -m integration`) need the database with data and
  `DB_PASSWORD` set; several are known to fail on the current schema
  ([tests/README.md](tests/README.md#integration-status)).
