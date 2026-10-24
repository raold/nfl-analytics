# Database Quick Reference

**Last Updated:** February 12, 2026
**Purpose:** Quick lookup reference for database schema, connections, and critical conventions
**Full Documentation:** See [DATABASE_SCHEMA_AUDIT.md](DATABASE_SCHEMA_AUDIT.md)
**Health Check:** `python scripts/db_health_check.py`

---

## ⚠️ CRITICAL: Common Schema Mistakes

**This error has occurred DOZENS of times. Read this section carefully.**

### ❌ MISTAKE: Assuming `game_id` exists in `mart.player_game_stats`

**WRONG CODE (will fail):**
```sql
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id  -- ❌ Column does not exist
```

**ERROR:** `column pgs.game_id does not exist`

### ✅ CORRECT: Join via (season, week) composite key

**CORRECT CODE:**
```sql
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.player_id = '00-0019596';
```

**KEY POINTS:**
1. `mart.player_game_stats` does NOT have `game_id` column
2. Join via `(season, week)` composite key
3. Filter to player's team: `(home_team = current_team OR away_team = current_team)`
4. Use `LEFT JOIN` (some games may not have betting lines)

**Full Pattern Library:** See [SQL_PATTERNS.md](SQL_PATTERNS.md) for working examples

**Automated Validation:**
```python
from nfl_analytics.utils.sql_validator import validate_query

errors = validate_query("SELECT pgs.game_id FROM mart.player_game_stats pgs")
# Returns: ["ERROR: Column 'game_id' does not exist in 'mart.player_game_stats'..."]
```

---

## Connection Information

**PostgreSQL 17 with TimescaleDB Extension**

```
Host:     localhost:5544
Database: devdb01
User:     dro
Password: sicillionbillions (dev only)
```

**Connection String:**
```bash
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

---

## Critical Conventions

⚠️ **ALWAYS USE THESE COLUMN NAMES:**

| Use This | NOT This | Context |
|----------|----------|---------|
| `g.kickoff` | `g.game_date`, `g.gameday` | Game timestamp (stored as DATE) |
| `g.spread_close` | `g.spread_line` | Closing betting spread |
| `g.total_close` | `g.total_line` | Closing over/under total |
| `LAR` or `LA` | - | Rams: DB uses `LA`, canonical `LAR` - both valid |

**Why?** These are the actual column names in `public.games`. Using wrong names will cause JOIN failures and silent NULL values.

---

## Schema Overview

**50 tables across 4 user schemas** (+ TimescaleDB internal schemas):
- `public` (31) - Raw data (games, plays, weather, injuries, odds_history, nextgen_*, players, rosters_weekly)
- `mart` (6) - Derived tables and materialized views (player_hierarchy, player_game_stats, team_epa, team_injury_load)
- `ml` (12) - Model registry, predictions, feedback loop (model_registry, game_outcome_predictions, feedback_runs, etc.)
- `metadata` (1) - Column lineage tracking

**Note:** `plays` table has 26 boolean columns (converted from INTEGER Feb 2026). Custom `AVG(boolean)` and `SUM(boolean)` aggregates exist to support legacy SQL. See [CROSS_PLATFORM_SETUP.md](../CROSS_PLATFORM_SETUP.md) for details.

**Note:** TimescaleDB adds 8 internal schemas (_timescaledb_*) for hypertable management.

---

## Key Tables

### public schema (Raw Data)

| Table | Rows | Period | Purpose |
|-------|------|--------|---------|
| `games` | 7,263 | 1999-2025 | Game metadata, betting lines, outcomes |
| `plays` | 1.2M | 1999-2024 | Play-by-play data with EPA values |
| `weather` | ~7K | Varies | Weather conditions (game_id PK) |
| `injuries` | ~50K | Varies | Injury status records |
| `odds_history` | 820K | Varies | Bookmaker odds snapshots (TimescaleDB) |

### mart schema (Derived/Materialized)

**Materialized Views** (refresh with `SELECT mart.refresh_all_views();`):
- `mart.game_summary` - 7,263 games with betting analysis, weather
- `mart.player_hierarchy` - 24,341 players with position groups, team mappings
- `mart.player_game_stats` - 25,677 player-game records from NextGen stats

**Tables:**
- `mart.team_epa` - 14,366 team-game EPA summaries
- `mart.team_4th_down_features` - 14,362 4th-down decision metrics
- `mart.team_playoff_context` - 3,360 playoff probabilities (2020+)
- `mart.team_injury_load` - 2,798 injury load metrics
- `mart.position_groups` - 29 position→position_group mappings
- `mart.bayesian_player_ratings` - Bayesian model output (populated by models)

### ml schema (Machine Learning)

| Table | Rows | Purpose |
|-------|------|---------|
| `ml.game_outcome_predictions` | 285 | Game-level predictions from v2 model (spread, total, win prob) |
| `ml.model_registry` | 16 | Registered models (9 production, 7 staging) |
| `ml.model_metrics` | 0 | Model performance metrics (TODO: populate — ADR-017 Tier 1.2) |
| `ml.production_deployments` | 0 | Active deployment records (TODO: populate — ADR-017 Tier 1.2) |
| `ml.feedback_runs` | varies | Weekly feedback orchestration summaries |
| `ml.semantic_attribution` | varies | Per-game feature SHAP attribution scores |
| `ml.feature_performance` | varies | Rolling feature contribution tracking |
| `ml.feature_selection_state` | varies | Feature status (active/disabled/probation) |

**Note:** Game Outcome v2 is the production model but is NOT yet registered in `ml.model_registry` (ADR-017 Tier 1.2).

---

## Refresh Materialized Views

```bash
# Refresh game_summary
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW mart.game_summary;"

# Refresh all game features
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT mart.refresh_game_features();"
```

---

## Common Queries

**Check connection:**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT COUNT(*) FROM games;"
```

**List all tables in mart schema:**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "\dt mart.*"
```

**Check materialized view freshness:**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT schemaname, matviewname FROM pg_matviews WHERE schemaname='mart';"
```

---

## Key Integration Patterns

**Feature Engineering Setup:**
1. Join `games` on `game_id` for line data
2. Use `kickoff` for time-based filtering (not `game_date`)
3. LEFT JOIN `weather` on `game_id` for weather features
4. Join `mart.team_epa` on (`game_id`, `team_abbr`) for EPA data
5. Use `spread_close` and `total_close` for Vegas features

**Player Stats Lookup:**
1. Query `mart.player_hierarchy` for player metadata (15,213 players)
2. Join `mart.player_game_stats` on `player_id` (NOT `game_id` - see warning above)
3. Join `games` via `(season, week)` composite key + team matching
4. Match positions and teams via hierarchy table

**Example:**
```sql
SELECT pgs.*, ph.player_name, g.spread_close
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team);
```

---

## Full Documentation

See **[DATABASE_SCHEMA_AUDIT.md](DATABASE_SCHEMA_AUDIT.md)** for:
- Complete table inventory with all columns
- Data quality findings and recommendations
- Team abbreviation standards (complete list)
- Integration patterns with code examples
- Historical data coverage by table
- Performance optimization notes

---

**Version:** 1.0
**Maintainer:** Claude Code
**Related:** CLAUDE.md § Database Schema, DATABASE_SCHEMA_AUDIT.md
