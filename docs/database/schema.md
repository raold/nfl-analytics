# Database Schema Overview

This document summarizes the operational database used by the pipelines in this repository. It covers base (public) tables, marts, materialized views, indexing, and lineage from ingestion to feature-ready views.

## ER Diagram

See the ERD for high-level relationships: `docs/database/erd.md` (PNG: `docs/database/erd.png`).

## Schemas

- public: Raw and lightly normalized entities used as sources for marts.
- mart: Derived tables and materialized views for analytics and modeling.

## Public Tables

- games
  - Primary key: game_id
  - Core columns: season, week, home_team, away_team, kickoff, spread_close, total_close, home_moneyline, away_moneyline, home_spread_odds, away_spread_odds, over_odds, under_odds, stadium, surface
  - Notes: Weather fields temp and wind have been removed from games; use weather table instead. See migration 006.
  - DDL reference: db/migrations/001_init.sql:1

- plays
  - Primary key: (game_id, play_id)
  - Core columns: posteam, defteam, quarter, time_seconds, down, ydstogo, epa, pass, rush
  - DDL reference: db/migrations/001_init.sql:13

- weather
  - Primary key: game_id
  - Core columns: station, temp_c, rh (humidity), wind_kph, pressure_hpa, precip_mm
  - DDL reference: db/migrations/001_init.sql:20

- injuries
  - Key columns: game_id, team, player_id, status
  - DDL reference: db/migrations/001_init.sql:24

- odds_history (TimescaleDB hypertable)
  - Primary key: (event_id, bookmaker_key, market_key, outcome_name, snapshot_at)
  - Indexes: (bookmaker_key, market_key, snapshot_at), (event_id, snapshot_at)
  - Timescale: Hypertable on snapshot_at, compression enabled; retention/compression policies defined
  - DDL references: db/migrations/001_init.sql:28, db/migrations/002_timescale.sql:1

## Mart Tables and Views

- mart.team_epa (table)
  - Aggregated per-game EPA summaries per team
  - Primary key: (game_id, posteam)
  - DDL reference: db/migrations/001_init.sql:50

- mart.game_summary (materialized view)
  - Enriched game-level summary: scores, lines, EPA, venue metadata, weather (joined from weather), QB/coach (if populated), differentials
  - Latest definition: db/migrations/006_remove_weather_duplication.sql:12
  - Indexes: (season, week), (home_team, season)

- mart.game_weather (materialized view)
  - Weather features derived from public.weather joined to games
  - DDL reference: db/migrations/003_mart_game_weather.sql:5

- mart.team_4th_down_features (table)
  - 4th-down aggressiveness and decision quality metrics
  - Primary key: (game_id, team)
  - DDL reference: db/migrations/004_advanced_features.sql:16

- mart.team_playoff_context (table)
  - Playoff probabilities and status flags per team-week
  - Primary key: (team, season, week)
  - DDL reference: db/migrations/004_advanced_features.sql:48

- mart.team_injury_load (table)
  - Aggregated injury metrics per team-week
  - Primary key: (season, week, team)
  - DDL reference: db/migrations/004_advanced_features.sql:86

- mart.game_features_enhanced (materialized view)
  - Composite modeling view joining EPA, weather, 4th-down, playoff context, and injury load
  - Refresh helper: SELECT mart.refresh_game_features();
  - DDL reference: db/migrations/004_advanced_features.sql:120

## Migration Ordering

Recommended order for a fresh database:

1) 001_init.sql – base tables and initial mart.team_epa and mart.game_summary
2) 002_timescale.sql – enable TimescaleDB and convert odds_history to hypertable
3) 003_mart_game_weather.sql – derived weather view
4) 004_advanced_features.sql – feature tables and enhanced features view + refresh function
5) 005_enhance_mart_views.sql – enrich mart.game_summary with expanded metadata (superseded by 006 for weather columns)
6) 006_remove_weather_duplication.sql – ensure weather is joined from weather table (drops games.temp/wind)

See scripts/dev/init_dev.sh for an example of applying the initial schema steps.

## Lineage and Ingestion

- R ingestors/backfills populate public tables:
  - data/ingest_pbp.R → plays (with EPA)
  - R/backfill_game_metadata.R → games (stadium, QB, coach, penalties/turnovers, etc.)
  - R/backfill_rosters.R → roster/person tables if present (and enrich downstream joins)
  - R/backfill_pbp_advanced.R → advanced play-level enrichments
  - data/ingest_injuries.R → injuries
- Python ingestion pipelines:
  - py/ingest_odds_history.py → odds_history (time-series snapshots)
  - py/weather_meteostat.py → weather (per game)
- After loads, refresh marts:
  - psql ... -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
  - SELECT mart.refresh_game_features();

## Indexing and Performance

- games: season, week
- mart.game_summary: (season, week), (home_team, season)
- mart.game_weather: (game_id), (season, week), partial indexes on wind_kph/temp_c
- odds_history: segment-by compression on (bookmaker_key, market_key); retention and compression policies configured

## Validation

- tests/sql/test_schema.sql provides schema assertions for existence of core tables, primary keys, hypertables, and mart presence.
- Nightly data quality job validates schema and integrity: .github/workflows/nightly-data-quality.yml

## Notes on Evolving Columns

- A set of enriched columns (QB names, coaches, referee, rest, turnovers, penalties, etc.) are populated by backfill scripts and may not appear in 001_init.sql. Downstream marts reference these when available.
- Weather duplication has been removed from games; always source meteorology from public.weather for analytics and reporting.
