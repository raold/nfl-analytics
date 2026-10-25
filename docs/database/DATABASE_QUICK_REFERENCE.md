# Database Quick Reference

**Updated:** 2026-10-25. Every count below was taken in a read-only session against devdb01 on
that day. Schema bootstrap, dumps and restores: [db/README.md](../../db/README.md). Working SQL:
[SQL_PATTERNS.md](SQL_PATTERNS.md).

## Connection

PostgreSQL 17.7 with TimescaleDB 2.24.0, container `nfl-analytics-pg-1` (`docker compose up -d
pg-1`), listening on 127.0.0.1:5544, database `devdb01`, user `dro`. Code reads `DB_HOST`,
`DB_PORT`, `DB_NAME`, `DB_USER` and `DB_PASSWORD` (see `.env.example`; dev defaults in
`nfl_analytics/utils/db_config.py`). From the host:

```bash
psql "postgresql://dro@127.0.0.1:5544/devdb01"            # password from PGPASSWORD or ~/.pgpass
docker exec -it nfl-analytics-pg-1 psql -U dro -d devdb01  # inside the container
```

The `dro` role is a superuser and its password is public: keep the database on loopback.

## Semantics that have caused wrong results

| Column | Meaning | Use |
|---|---|---|
| `games.spread_close` | nflverse `spread_line`: expected HOME margin, positive = home favored (`2018_01_BUF_BAL` = 7.5) | home covers iff `(home_score - away_score) - spread_close > 0`; edge = `pred - spread_close` (`nfl_analytics/utils/spread_convention.py`) |
| `games.kickoff` | `timestamptz`, the true kickoff instant for 2000-2025 (repaired 2026-10-24); NULL for 1999 | local hour, weekday or date: `kickoff AT TIME ZONE 'America/New_York'` |
| `odds_history.home_spread` | sportsbook sign: negative = home favored (corr -0.957 with `spread_close`) | negate before comparing with `spread_close` |
| `odds_history.market_type` | the name of the backfill request: 'closing' was 4.5-5.5 h before the day's first kickoff, 'opening' 14:00 UTC; neither is the open or the close | use `games.spread_close` as the close; timing in [the line-movement write-up](../investigations/2026-10-line-movement/README.md) |
| `ml.game_outcome_predictions.kickoff` | the game date at midnight Eastern written as UTC, not the kickoff | join `games` for the kickoff |
| `games.home_team`, `away_team` | nflverse codes: Rams `LA` (`STL` before 2016), Chargers `LAC` (`SD` before 2017), Raiders `LV` (`OAK` before 2020), Washington `WAS`; there is no `LAR` in `games` | `nfl_analytics.utils.team_mapping.normalize_team()` converts these; `draft_picks` and `espn_qbr` use other code sets ([SQL_PATTERNS.md](SQL_PATTERNS.md) §3) |

Other conventions: `kickoff` not `game_date`, `spread_close` not `spread_line`, `total_close` not
`total_line`, `quarter` not `qtr`, `posteam`/`defteam` not `team`/`opponent`. `plays` has both
`pass` and `pass_attempt` (different nflverse definitions).

**Never `DELETE FROM games`:** `plays` references `games` and `participation` references `plays`,
both `ON DELETE CASCADE` (`international_games` blocks deletes of its 18 games). Change games with
`UPDATE` or `INSERT ... ON CONFLICT (game_id) DO UPDATE`.

## Schemas

| Schema | Contents |
|---|---|
| `public` | 46 tables and 1 view (`rosters`, over `rosters_weekly`) plus the matview `mv_betting_features`; 13 `mlb_*`/`nhl_*` tables belong to the Sibyl project |
| `mart` | 6 tables (`team_epa`, `team_4th_down_features`, `team_injury_load`, `team_playoff_context`, `position_groups`, `bayesian_player_ratings`) and 3 matviews (`game_summary`, `player_game_stats`, `player_hierarchy`) |
| `ml` | 12 tables (`game_outcome_predictions`, `model_registry`, `model_metrics`, `model_evaluations`, Dirichlet and Bayesian-update tables, and five empty training/deployment tables) and 8 views |
| `metadata` | `feature_freshness` |
| `salvage` | `nflverse_games`: the nflverse schedule download (2026-10-24 16:54 EDT, 7,548 rows) used to repair `games` |

TimescaleDB adds its internal schemas; `odds_history` and `ml.model_metrics` are hypertables.

## Tables

| Table | Rows | Coverage | Notes |
|---|---|---|---|
| `games` | 7,276 | 1999-2025 | 41 columns; all games completed; `game_type`, scores, lines and kickoffs equal nflverse for every game |
| `plays` | 1,279,628 | 1999-2025 | 115 columns, 44 boolean; `quarter`, `game_seconds_remaining` and `half_seconds_remaining` are NULL for all 1,230,857 plays of 1999-2024 |
| `participation` | 428,409 | 2016-2024 (2,476 games) | one row per play; key `(nflverse_game_id, play_id)` |
| `odds_history` | 117,266 | 2020-2024 | 'closing' 59,920 and 'opening' 57,314 rows for 1,061 games, 25 books; 32 'game_lines' rows; one backfill (captured 2025-12-07 23:50 to 2025-12-08 21:40 UTC); no Rams games, and 10 of the 264 other games that start on the next UTC date; **licensed** |
| `injuries` | 84,682 | 2009-2024 | |
| `weather` | 2,061 | 2018-2025 | wind speeds from the Meteostat ingest are in the wrong unit (ERRATA E5); prefer nflverse `wind` |
| `nextgen_passing` / `_rushing` / `_receiving` | 5,696 / 5,794 / 14,187 | 2016-2025 | include week-0 season totals (403 / 499 / 1,228 rows) |
| `espn_qbr` | 10,709 | | |
| `rosters_weekly` | 39,591 | 2012-2025 | partial: about 2,000-3,200 rows a season |
| `players` | 25,847 | | |
| `coaching_staff` | 544 | 2009-2025 | |
| `semantic_news` | 66 | | February 2026 ESPN items stamped season 2024, week 1; an exhibit of look-ahead, not a feature source |
| `mart.game_summary` (matview) | 7,276 | 1999-2025 | covers graded correctly since migration 104: pooled home cover 0.4905 over 7,081 decided games |
| `public.mv_betting_features` (matview) | 7,276 | 1999-2025 | rebuilt by migration 104 |
| `mart.player_game_stats` (matview) | 25,677 | 2016-2025 | passing 5,696, rushing 5,794, receiving 14,187; no `game_id`; 2,130 week-0 rows |
| `mart.player_hierarchy` (matview) | 25,847 | | one row per player; `current_team` is the latest team only |
| `mart.team_epa` | 14,366 | | key `(game_id, posteam)` |
| `mart.team_4th_down_features` | 14,362 | | key `(game_id, team)` |
| `mart.team_injury_load` | 2,798 | | key `(season, week, team)` |
| `mart.team_playoff_context` | 192 | | |
| `ml.game_outcome_predictions` | 285 | 2025 | Game Outcome v2's one retroactive run (2026-02-12); exported to `data/game_outcome_v2_predictions_2025.csv` |
| `ml.model_registry` | 17 | | v2 is model_id 17, still marked `production`; ignore the flag (the model is DO NOT DEPLOY) |
| `ml.model_metrics` | 10 | | v2 metrics entered on 2025-12-23 and 2026-02-12: `ats_win_pct` 52.1 is straight-up side accuracy, not ATS (ledger C26); `ats_roi_pct` -3.2 holds only at \|edge\| > 2 (C02); the "2024 holdout" rows are the early-stopping season (C17) |

Tables the older docs describe that do **not** exist: the feedback-loop tables (`ml.feedback_runs`
and ten others), `ml.game_predictions`, `ml.feature_store`, `ml.r_model_predictions`,
`mart.team_offensive_stats`, `mart.team_defensive_stats`, the `semantic.*` and `reference.*`
schemas.

## Objects created outside the migrations

Created by hand in February 2026; no file in `db/migrations/` creates them, and the schema
snapshot in `db/schema/` includes them:

- Custom aggregates `avg(boolean)` and `sum(boolean)` (with `bool_avg_sfunc`, `bool_avg_ffunc`,
  `bool_sum_sfunc`) and the helpers `bool_to_int`, `int_to_bool`, `bool_eq_int`, `int_eq_bool`, added
  when 26 `plays` columns were converted from integer to boolean during the 2025 ingest. Prefer
  explicit casts (`AVG(col::int)`).
- `ml.game_outcome_predictions`, created directly for the v2 prediction runs (migration 051 alters
  a different version of it and was not applied here).

## Maintenance

```bash
uv run python scripts/refresh_marts.py --status     # matview status (read-only)
uv run python scripts/refresh_marts.py --all        # REFRESH every matview (writes)
```

`mart.refresh_all_views()` and `mart.refresh_game_features()` do the same in SQL. Back up before
any write (`make backup`). A fresh database built from `db/migrations/` would miss the objects
above and differ elsewhere too; build from the schema snapshot instead.

## Checking a query

```bash
uv run python -m nfl_analytics.utils.sql_validator "SELECT pgs.game_id FROM mart.player_game_stats pgs"
# ERROR: Column 'game_id' does not exist in 'mart.player_game_stats'. Did you mean '(season, week)'?
```

The validator knows the usual traps (`game_id` in `player_game_stats`, `game_date`). It does not
know the semantics above: a query can be valid and still grade covers with the wrong sign or
join a player to the wrong team (SQL_PATTERNS.md, mistake 4).
