# db/

Schema snapshot, migration history and DBA scripts for `devdb01`, the local PostgreSQL 17 +
TimescaleDB database (container `nfl-analytics-pg-1` from `docker-compose.yml`, 127.0.0.1:5544).

## Bootstrap an empty database

```bash
make db                          # docker compose up -d pg-1; data directory ./pgdata
bash scripts/dev/init_dev.sh     # waits for devdb01, applies db/schema/ unless public.games exists
```

[`schema/devdb01_schema_2026-10-24.sql`](schema/devdb01_schema_2026-10-24.sql) is a
`pg_dump --schema-only` of devdb01 as the 2026-10-24 salvage left it (migration 104 applied,
`games` repaired); its header has the exact command. It leaves out the `salvage` schema (a staging
copy of the nflverse schedule), Sibyl's 13 `mlb_*`/`nhl_*` tables and the TimescaleDB chunks. It
ends with the two `create_hypertable()` calls that a schema-only dump cannot carry:
`public.odds_history` on `captured_at` (30-day chunks) and `ml.model_metrics` on `timestamp`
(7 days).

Checked on 2026-10-25 in throwaway `timescale/timescaledb:latest-pg17` containers, both with
`psql -f` and through `init_dev.sh`: it applies without errors and gives the same 1,667 schema
objects as devdb01 in `public`, `ml`, `mart` and `metadata` (relations, columns, constraints,
indexes, view, function and trigger definitions, aggregates, comments, hypertables). All four
materialized views refresh, and a test insert into `odds_history` creates a chunk.

A bootstrapped database has no rows: `schema_migrations` is empty and the materialized views are
unpopulated. After loading data, refresh `mart.player_hierarchy` before `mart.player_game_stats`
(the only view that reads another one), then `mart.game_summary` and
`public.mv_betting_features`. `make refresh` (`scripts/refresh_marts.py`) does the three `mart`
views; refresh `public.mv_betting_features` with psql.

## Restore a full dump

The full dumps of the repaired database are private: they contain licensed The Odds API data
(`public.odds_history`) and live only in `backups/local/` on the owner's machine (git-ignored).
`backups/local/README.md` lists them with their checks. To restore one into devdb01:

```bash
TIMESCALEDB_VERSION=2.24.0 bash scripts/maintenance/restore.sh backups/local/devdb01_post_repair_20261024.dump
```

`restore.sh` asks before dropping devdb01, first saves the current database to `BACKUP_DIR`
(default `~/nfl-analytics-backups`), then follows the TimescaleDB procedure: a new database from
`template0`, the extension at the dump's version, `timescaledb_pre_restore()`, `pg_restore`,
`timescaledb_post_restore()`. The 2026-10-24 dumps were taken with TimescaleDB 2.24.0, while the
image now defaults to 2.25.0; with the wrong version, `timescaledb_post_restore()` fails with
"catalog version mismatch". Tested on 2026-10-25 against a throwaway container: the
post-repair dump restored in about 20 s with 7,276 games, 1,279,628 plays, 117,266 odds rows,
66 `schema_migrations` rows and `avg(home_covered)` 0.4905 in `mart.game_summary`.

`make backup` (`scripts/maintenance/backup.sh`) writes `nfl_analytics_<time>.backup` plus a
`.meta` file that records the TimescaleDB version, so `make restore` picks the version itself.

The three dumps directly in `backups/` (Git LFS, public on GitHub) are from December 2025 and
predate the 2026-10-24 repair.

## db/migrations is history

The 72 SQL files in `migrations/` (added between 2025-10-04 and 2026-10-24) record how the
schema grew. They cannot rebuild it, and the runner that listed them (`scripts/run_migrations.py`,
which did not know 104) was removed on 2026-10-25.

- devdb01's `schema_migrations` has 66 rows, and some recorded migrations left nothing behind:
  there is no `auth` schema (045) and no `ml.game_predictions` (048a-e).
- Do not re-apply old files. 005, 006 and 100 define mart views with the sportsbook cover rule
  (migration 104 replaced them on 2026-10-24). 021 and 022 rewrite 2025 week-6 kickoffs: against
  the repaired values, 021 would move 6 games to the wrong day (7 of its 13 game ids do not
  exist), and 022 would move 2025_06_DEN_NYJ (London, 13:30 UTC) to 17:00 UTC.
- The `spread_cover` trigger in 048 and its ADDENDUM added the line to the margin; the sign was
  fixed in both files on 2026-10-25. The table they serve is not in devdb01.

To change the schema now: take a backup (`make backup`), write the next numbered migration that
records itself in `schema_migrations` (as 104 does), apply it with
`psql -v ON_ERROR_STOP=1 -f`, and regenerate the snapshot in `schema/` with the command in its
header.

## Conventions the SQL relies on

- `games.spread_close` is nflverse `spread_line`: the expected home margin, positive when home is
  favored. The home team covers when `(home_score - away_score) - spread_close > 0`; pushes have
  no result (`nfl_analytics/utils/spread_convention.py`). `odds_history.home_spread` is in
  sportsbook sign (negative = home favored).
- `games.kickoff` is the true kickoff instant (timestamptz) for 2000-2025 and NULL for 1999.
  Derive dates, weekdays and hours with `kickoff AT TIME ZONE 'America/New_York'`.
- `public.avg(boolean)` and `public.sum(boolean)` are custom aggregates; the schema snapshot
  creates them.

## Other files

- `views/materialized/`: view definitions; `05_mv_betting_features.sql` uses the same cover rule
  as migration 104.
- `optimize_indexes.sql`, `performance_indexes*.sql`, `setup_pg_stat_statements.sql`,
  `views/db_performance_metrics.sql`: November 2025 DBA scripts that create indexes, an extension
  or a view. `pg_stat_statements` is not loaded in the container.
- `health_check.sql`, `verify_database.sql`, `monitoring_queries*.sql`,
  `performance_benchmark.sql`, `queries/`, `audit/`: read-only checks; `audit/`, the `*.txt`
  files and the Markdown reports here are November 2025 snapshots.
