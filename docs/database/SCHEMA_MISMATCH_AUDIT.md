# NFL Analytics Database Schema Mismatch Audit

**Generated:** December 7, 2025
**Updated:** December 7, 2025 (POST-FIX)
**Database:** nfl-analytics-pg-1 (devdb01)
**Status:** ✅ RESOLVED - All critical issues fixed

## Resolution Summary

All critical schema mismatches have been resolved by running:
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/migrations/100_schema_reconciliation.sql
Rscript R/ingestion/ingest_injuries.R
```

**Health Check:** `python scripts/db_health_check.py` - All 34 checks PASS

---

## Original Audit (For Reference)

---

## Executive Summary

The database has **significant schema mismatches** between:
1. Migration SQL files (`db/migrations/*.sql`)
2. R ingestion scripts (`R/ingestion/*.R`)
3. Python ingestion scripts (`py/ingest_*.py`)
4. Documentation expectations (`docs/database/DATABASE_QUICK_REFERENCE.md`)

**Root Cause:** Different scripts create tables with different schemas, and migrations were not run in the correct order or were overwritten by ingestion scripts.

---

## Data Population Status

### Working Tables (With Data)
| Table | Rows | Status | Notes |
|-------|------|--------|-------|
| `games` | 7,263 | ✅ Working | 1999-2025 |
| `plays` | 1,264,341 | ✅ Working | Full PBP with 67 columns |
| `players` | 24,341 | ✅ Working | All historical |
| `rosters_weekly` | 39,580 | ✅ Working | 38 columns |
| `nextgen_passing` | 5,696 | ✅ Working | 2016-2025 |
| `nextgen_receiving` | 14,187 | ✅ Working | 2016-2025 |
| `nextgen_rushing` | 5,794 | ✅ Working | 2016-2025 |
| `participation` | 433,805 | ✅ Working | 2016-2025 |
| `officials` | 21,290 | ✅ Working | 2015-2025 |
| `mart.team_4th_down_features` | 14,362 | ✅ Working | |
| `mart.team_injury_load` | 2,798 | ✅ Working | |

### Broken Tables (Empty - Schema Issues)
| Table | Rows | Status | Blocking Issue |
|-------|------|--------|----------------|
| `injuries` | 0 | ❌ Schema Mismatch | PK constraint conflict |
| `weather` | 0 | ❌ No data source | Schema OK, no ingestion |
| `rosters` | 0 | ❌ Deprecated | Use rosters_weekly instead |
| `odds_history` | 0 | ❌ No API integration | TimescaleDB OK |
| `mart.team_epa` | 0 | ❌ Not populated | Needs aggregation |
| `mart.team_playoff_context` | 0 | ❌ Not populated | Needs calculation |

### Missing Tables/Views (Expected by Documentation)
| Expected | Status | Where Defined |
|----------|--------|---------------|
| `mart.game_summary` | ❌ Missing | 000_complete_schema.sql (MView) |
| `mart.game_weather` | ❌ Missing | 003_mart_game_weather.sql |
| `mart.game_features_enhanced` | ❌ Missing | 005_enhance_mart_views.sql |
| `mart.position_groups` | ❌ Missing | 023_player_hierarchy_schema.sql |
| `mart.player_hierarchy` | ❌ Missing | 023_player_hierarchy_schema.sql (MView) |
| `mart.player_game_stats` | ❌ Missing | 023_player_hierarchy_schema.sql (MView) |
| `mart.bayesian_player_ratings` | ❌ Missing | 023_player_hierarchy_schema.sql |

---

## Critical Schema Mismatches

### 1. `injuries` Table - BLOCKING

**Actual Schema (6 columns):**
```sql
CREATE TABLE injuries (
  season INTEGER NOT NULL,
  week INTEGER NOT NULL,
  team TEXT NOT NULL,
  player_name TEXT NOT NULL,  -- OLD: Uses player_name
  position TEXT,
  injury_status TEXT,
  PRIMARY KEY (season, week, team, player_name)
);
```

**R Ingestion Script Expects (16 columns):**
```sql
CREATE TABLE injuries (
  season INTEGER NOT NULL,
  game_type VARCHAR(10) NOT NULL,  -- NEW: Added
  team VARCHAR(3) NOT NULL,
  week INTEGER NOT NULL,
  gsis_id VARCHAR(20) NOT NULL,    -- NEW: Uses gsis_id instead of player_name
  position VARCHAR(10),
  full_name VARCHAR(100),           -- NEW
  first_name VARCHAR(50),           -- NEW
  last_name VARCHAR(50),            -- NEW
  report_primary_injury TEXT,       -- NEW
  report_secondary_injury TEXT,     -- NEW
  report_status VARCHAR(50),        -- NEW
  practice_primary_injury TEXT,     -- NEW
  practice_secondary_injury TEXT,   -- NEW
  practice_status VARCHAR(50),      -- NEW
  date_modified TIMESTAMPTZ,        -- NEW
  PRIMARY KEY (season, game_type, team, week, gsis_id)
);
```

**Source:** `R/ingestion/ingest_injuries.R:64-83`

**Impact:** R ingestion fails with column mismatch errors.

**Fix Required:**
```sql
DROP TABLE IF EXISTS injuries CASCADE;
-- Then run R ingestion script to create correct schema
```

---

### 2. `rosters` vs `rosters_weekly` - CONFUSION

**Problem:** Two competing tables for roster data.

| Aspect | `rosters` | `rosters_weekly` |
|--------|-----------|------------------|
| Rows | 0 | 39,580 |
| Columns | 10 | 38 |
| PK | `(season, week, team, player_id)` | `(season, week, team, gsis_id)` |
| Status | Deprecated | Active |

**Migration Expected (`000_complete_schema.sql`):**
- `rosters` with `gsis_id` as identifier

**Actual Usage:**
- `rosters_weekly` is populated by tmp ingestion scripts
- `rosters` remains empty

**Fix Required:**
- Drop `rosters` table
- Rename `rosters_weekly` to `rosters` OR
- Update all code to use `rosters_weekly`

---

### 3. `games` Table - COLUMN MISMATCH

**Actual Schema (37 columns):** Correct, from migration

**R Ingestion Script (`ingest_schedules.R`) Creates (16 columns):**
```sql
CREATE TABLE IF NOT EXISTS games (
  game_id text PRIMARY KEY,
  season int, week int,
  home_team text, away_team text,
  kickoff timestamptz,
  spread_close real, total_close real,
  home_score int, away_score int,
  home_moneyline real, away_moneyline real,
  home_spread_odds real, away_spread_odds real,
  over_odds real, under_odds real
);
```

**Missing Columns Not Populated:**
- `stadium`, `stadium_id`, `roof`, `surface` (venue info)
- `away_rest`, `home_rest` (rest days)
- `away_qb_id`, `home_qb_id`, `away_qb_name`, `home_qb_name`
- `away_coach`, `home_coach`, `referee`
- `game_type`, `overtime`
- `home_turnovers`, `away_turnovers`, `home_penalties`, `away_penalties`
- `home_penalty_yards`, `away_penalty_yards`, `updated_at`

**Impact:** 21 columns exist in schema but are NULL (not populated).

---

### 4. Missing `mart` Materialized Views

**Expected by `docs/database/DATABASE_QUICK_REFERENCE.md`:**
- `mart.game_summary` - Game-level enriched data
- `mart.game_weather` - Weather features
- `mart.game_features_enhanced` - 157-feature composite

**Migration `023_player_hierarchy_schema.sql` expects:**
- `mart.position_groups` (table)
- `mart.player_hierarchy` (materialized view)
- `mart.player_game_stats` (materialized view)
- `mart.bayesian_player_ratings` (table)

**Current mart schema contains only:**
- `mart.team_4th_down_features` (14,362 rows)
- `mart.team_epa` (0 rows)
- `mart.team_injury_load` (2,798 rows)
- `mart.team_playoff_context` (0 rows)

---

## Root Cause Analysis

### Issue 1: Migration Order Not Enforced
Migrations are not numbered consistently (000, 001, 002, 003, 004, 005, 007, 008, 009, 010...). Multiple migrations have same prefix (e.g., 018_*, 026_*, 048_*).

### Issue 2: CREATE TABLE IF NOT EXISTS in Ingestion Scripts
R ingestion scripts include `CREATE TABLE IF NOT EXISTS` with different schemas than migrations. When run, they create tables with wrong schema.

### Issue 3: No Schema Version Tracking
No mechanism to track which migrations have been applied. Running migrations after ingestion creates conflicts.

### Issue 4: R and Python Use Different Conventions
| Aspect | R Scripts | Python Scripts |
|--------|-----------|----------------|
| Player ID | `gsis_id` | `player_id` |
| Team Code | `team` | `team`, `club_code` |
| Date Fields | `gameday` → `kickoff` | `captured_at` |

---

## Recommended Fixes

### Phase 1: Fix Critical Blocking Issues

**1.1 Fix `injuries` table:**
```sql
-- Backup if any data exists
CREATE TABLE injuries_backup AS SELECT * FROM injuries;

-- Drop and recreate with correct schema
DROP TABLE injuries CASCADE;

-- Run R ingestion to create correct schema
-- Rscript R/ingestion/ingest_injuries.R
```

**1.2 Create missing mart views:**
```sql
-- Run migration 023 for player hierarchy
\i db/migrations/023_player_hierarchy_schema.sql

-- Create game_summary materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS mart.game_summary AS
SELECT
  g.game_id, g.season, g.week,
  g.home_team, g.away_team, g.kickoff,
  g.stadium, g.roof, g.surface,
  g.spread_close, g.total_close,
  g.home_score, g.away_score,
  g.home_score - g.away_score AS margin,
  g.home_score + g.away_score AS total,
  CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_win,
  CASE WHEN g.home_score IS NOT NULL THEN 1 ELSE 0 END AS is_final,
  w.temp, w.wind, w.conditions
FROM games g
LEFT JOIN weather w ON g.game_id = w.game_id;
```

**1.3 Populate `mart.team_epa`:**
```sql
INSERT INTO mart.team_epa (game_id, posteam, plays, epa_sum, epa_mean, explosive_pass, explosive_rush)
SELECT
  game_id,
  posteam,
  COUNT(*) as plays,
  SUM(epa) as epa_sum,
  AVG(epa) as epa_mean,
  SUM(CASE WHEN pass = 1 AND yards_gained >= 20 THEN 1 ELSE 0 END)::float / NULLIF(SUM(pass), 0) as explosive_pass,
  SUM(CASE WHEN rush = 1 AND yards_gained >= 10 THEN 1 ELSE 0 END)::float / NULLIF(SUM(rush), 0) as explosive_rush
FROM plays
WHERE posteam IS NOT NULL
GROUP BY game_id, posteam;
```

### Phase 2: Standardize Ingestion Scripts

**2.1 Remove CREATE TABLE from ingestion scripts**
Ingestion scripts should ONLY insert data, never create tables. Schema creation belongs in migrations.

**2.2 Create canonical column mapping**
Document the authoritative column names and ensure all scripts use them.

### Phase 3: Add Schema Validation

**3.1 Pre-ingestion validation:**
```python
def validate_schema(table_name, expected_columns):
    """Validate table schema before ingestion."""
    actual_columns = get_table_columns(table_name)
    missing = set(expected_columns) - set(actual_columns)
    if missing:
        raise SchemaError(f"Missing columns in {table_name}: {missing}")
```

**3.2 Create migration tracking table:**
```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
  version VARCHAR(100) PRIMARY KEY,
  applied_at TIMESTAMP DEFAULT NOW()
);
```

---

## Files to Update

| File | Issue | Action |
|------|-------|--------|
| `R/ingestion/ingest_injuries.R` | Creates wrong schema | Remove CREATE TABLE |
| `R/ingestion/ingest_schedules.R` | Creates partial schema | Remove CREATE TABLE |
| `db/migrations/000_complete_schema.sql` | Missing materialized views | Add mart.game_summary |
| `db/migrations/023_player_hierarchy_schema.sql` | Not applied | Run migration |
| `docs/database/DATABASE_QUICK_REFERENCE.md` | Outdated | Update to match reality |

---

## Verification Queries

```sql
-- Check table row counts
SELECT
  'games' as tbl, COUNT(*) FROM games
UNION ALL SELECT 'plays', COUNT(*) FROM plays
UNION ALL SELECT 'injuries', COUNT(*) FROM injuries
UNION ALL SELECT 'weather', COUNT(*) FROM weather
UNION ALL SELECT 'rosters_weekly', COUNT(*) FROM rosters_weekly;

-- Check mart schema objects
SELECT table_type, table_name
FROM information_schema.tables
WHERE table_schema = 'mart'
UNION ALL
SELECT 'MATVIEW', matviewname
FROM pg_matviews
WHERE schemaname = 'mart';

-- Check for missing columns in games
SELECT column_name FROM information_schema.columns
WHERE table_name = 'games' AND table_schema = 'public'
AND column_name NOT IN ('game_id','season','week','home_team','away_team',
  'kickoff','spread_close','total_close','home_score','away_score')
ORDER BY column_name;
```

---

## Priority Matrix

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| Missing mart.player_hierarchy | HIGH (blocks modeling) | MEDIUM | 1 |
| injuries schema mismatch | HIGH (no injury data) | LOW | 2 |
| Empty mart.team_epa | MEDIUM (feature gap) | LOW | 3 |
| Missing mart.game_summary | MEDIUM (convenience) | LOW | 4 |
| rosters vs rosters_weekly | LOW (workaround exists) | MEDIUM | 5 |
| games columns NULL | LOW (not critical) | HIGH | 6 |

---

**Next Steps:**
1. Run `db/migrations/023_player_hierarchy_schema.sql` to create player modeling tables
2. Fix injuries table schema and re-run ingestion
3. Populate mart.team_epa from plays data
4. Update documentation to match actual schema

---

*Generated by Claude Code comprehensive schema audit*
