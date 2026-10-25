# Canonical SQL Patterns for NFL Analytics

**Last Updated:** November 14, 2025
**Purpose:** Document correct SQL patterns to prevent recurring schema errors
**Background:** Schema errors have occurred "DOZENS of times" - this document prevents future mistakes

---

## Table of Contents

1. [Critical Schema Facts](#critical-schema-facts)
2. [Common Mistakes](#common-mistakes)
3. [Pattern Library](#pattern-library)
4. [Index Usage Guide](#index-usage-guide)
5. [Performance Notes](#performance-notes)

---

## Critical Schema Facts

### mart.player_game_stats Schema

**⚠️ DOES NOT HAVE `game_id` COLUMN**

```sql
-- Key columns in mart.player_game_stats:
player_id      TEXT          -- NFL player ID (e.g., '00-0019596')
season         INTEGER       -- NFL season year (e.g., 2024)
week           INTEGER       -- Week number (1-18 for regular season)
stat_category  TEXT          -- 'passing', 'rushing', 'receiving'
stat_yards     INTEGER       -- Total yards for stat_category
stat_attempts  INTEGER       -- Total attempts (passes, rushes, targets)
-- ... (15+ other columns, see \d mart.player_game_stats for full list)
```

**Primary Key:** `(player_id, season, week, stat_category)`
**IMPORTANT:** Composite key `(season, week)` is used to join to `games` table

### games Schema

**✅ HAS `game_id` COLUMN (Primary Key)**

```sql
-- Key columns in public.games:
game_id        TEXT PRIMARY KEY  -- Unique game identifier (e.g., '2024_01_LAC_MIA')
season         INTEGER           -- NFL season year
week           INTEGER           -- Week number
home_team      TEXT              -- 3-letter team abbreviation
away_team      TEXT              -- 3-letter team abbreviation
kickoff        TIMESTAMPTZ       -- Game start time (NOT game_date)
spread_close   REAL              -- Closing point spread (NOT spread_line)
total_close    REAL              -- Closing total (NOT total_line)
-- ... (50+ other columns)
```

**Composite Index:** `(season, week)` for joining to player stats

### mart.player_hierarchy Schema

**⚠️ DOES NOT HAVE `game_id` COLUMN**

```sql
-- Key columns in mart.player_hierarchy:
player_id               TEXT UNIQUE   -- NFL player ID (primary key)
player_name             TEXT          -- Display name
position                TEXT          -- Detailed position (QB, RB, WR, etc.)
position_group          VARCHAR(10)   -- Coarse grouping (QB, RB, WR, TE, etc.)
current_team            TEXT          -- Most recent team (3-letter abbrev)
years_exp               INTEGER       -- Years of experience
hierarchy_position      TEXT          -- Composite: position_team
hierarchy_team_position TEXT          -- Composite: team_position
-- ... (14 total columns)
```

---

## Common Mistakes

### ❌ MISTAKE #1: Assuming game_id in mart.player_game_stats

**WRONG CODE:**
```sql
-- This will fail with "column pgs.game_id does not exist"
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id  -- ❌ pgs.game_id does not exist
WHERE pgs.player_id = '00-0019596';
```

**ERROR MESSAGE:**
```
ERROR:  column pgs.game_id does not exist
LINE 3: JOIN games g ON g.game_id = pgs.game_id
```

**WHY IT'S WRONG:**
- `mart.player_game_stats` is a materialized view aggregating player stats
- Multiple players can play in the same game → no single `game_id` column
- Must join via composite key `(season, week)` + team matching

---

### ✅ CORRECT PATTERN: Join via (season, week) composite key

**CORRECT CODE:**
```sql
-- Join via (season, week) composite key
SELECT
    pgs.player_id,
    pgs.season,
    pgs.week,
    pgs.stat_yards,
    pgs.stat_attempts,
    g.game_id,
    g.spread_close,
    g.total_close,
    g.kickoff
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.player_id = '00-0019596'
    AND pgs.stat_category = 'passing';
```

**KEY POINTS:**
1. Join `games` via `(season, week)` composite key
2. Filter game to player's team: `(home_team = current_team OR away_team = current_team)`
3. Use `player_hierarchy` to get `current_team` for filtering
4. Use `LEFT JOIN` (some games may not have betting lines)

---

### ❌ MISTAKE #2: Using wrong column names

**WRONG CODE:**
```sql
SELECT
    g.game_date,        -- ❌ Column doesn't exist (use g.kickoff)
    g.spread_line,      -- ❌ Column doesn't exist (use g.spread_close)
    g.total_line        -- ❌ Column doesn't exist (use g.total_close)
FROM games g;
```

**✅ CORRECT CODE:**
```sql
SELECT
    g.kickoff,          -- ✅ TIMESTAMPTZ game start time
    g.spread_close,     -- ✅ Closing point spread
    g.total_close       -- ✅ Closing total
FROM games g;
```

**Reference:** See [DATABASE_QUICK_REFERENCE.md](DATABASE_QUICK_REFERENCE.md#critical-conventions)

---

### ❌ MISTAKE #3: Team abbreviation errors

**WRONG CODE:**
```sql
-- ❌ LA is ambiguous (Rams or Chargers?)
SELECT * FROM games WHERE home_team = 'LA';
```

**✅ CORRECT CODE:**
```sql
-- ✅ Use LAR (Rams) or LAC (Chargers)
SELECT * FROM games WHERE home_team IN ('LAR', 'LAC');
```

---

## Pattern Library

### Pattern 1: Player Stats → Games Join (CANONICAL)

**Use Case:** Get player stats with game context (spread, total, weather)

**Production Example:** From `R/train_hierarchical_v2_production.R` (lines 47-94)

```sql
SELECT
    pgs.player_id,
    pgs.season,
    pgs.week,
    pgs.stat_yards as yards,
    pgs.stat_attempts as attempts,

    -- Game context
    g.home_team,
    g.away_team,
    ph.current_team as team,
    CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

    -- Betting lines
    CASE
        WHEN ph.current_team = g.home_team
        THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
        ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
    END as is_favored,
    ABS(g.spread_close) as spread_abs,
    g.total_close,

    -- Weather
    COALESCE(g.roof, 'unknown') as roof,
    CASE
        WHEN g.roof IN ('outdoors', 'open')
        AND CAST(g.temp AS NUMERIC) < 40
        THEN 1 ELSE 0
    END as is_bad_weather,
    CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,

    -- Player info
    ph.years_exp,
    CASE
        WHEN ph.current_team = g.home_team THEN g.away_team
        ELSE g.home_team
    END as opponent

FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)

WHERE pgs.stat_category = 'passing'
    AND pgs.stat_yards IS NOT NULL
    AND pgs.stat_attempts IS NOT NULL
    AND pgs.stat_attempts >= 10  -- Minimum attempts
    AND pgs.season BETWEEN 2015 AND 2025
    AND ph.position_group = 'QB'

ORDER BY pgs.season, pgs.week, pgs.player_id;
```

**Performance:**
- Uses index `idx_player_game_stats_lookup (player_id, season, week)`
- Uses index `games_season_week_idx (season, week)`
- Typical execution: 50-200ms for single player, 5-20s for all QBs

---

### Pattern 2: Single Player-Game Lookup

**Use Case:** Get stats for a specific player in a specific game (common in prediction pipeline)

**Production Example:** From `docs/ENSEMBLE_BACKTEST_PERFORMANCE_ESTIMATE.md` (lines 369-384)

```sql
SELECT
    pgs.stat_yards,
    pgs.stat_attempts,
    pgs.stat_touchdowns,
    ph.current_team,
    g.spread_close,
    g.total_close,
    g.game_id

FROM mart.player_game_stats pgs
LEFT JOIN mart.player_hierarchy ph USING(player_id)
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)

WHERE pgs.player_id = %s           -- Parameterized query
    AND pgs.season = %s
    AND pgs.week = %s
    AND pgs.stat_category = 'passing'

LIMIT 1;
```

**Performance:**
- Uses unique index `idx_player_game_stats_unique (player_id, season, week, stat_category)`
- Execution: 10-50ms (index seek)

**Usage in Python:**
```python
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host="localhost", port=5544, database="devdb01",
    user="dro", password="sicillionbillions"
)

query = """
SELECT pgs.stat_yards, pgs.stat_attempts, g.spread_close
FROM mart.player_game_stats pgs
LEFT JOIN mart.player_hierarchy ph USING(player_id)
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.player_id = %s
    AND pgs.season = %s
    AND pgs.week = %s
    AND pgs.stat_category = %s
LIMIT 1;
"""

df = pd.read_sql_query(
    query,
    conn,
    params=('00-0019596', 2024, 1, 'passing')
)
```

**Usage in R:**
```r
library(DBI)
library(RPostgres)

conn <- dbConnect(
    Postgres(),
    host = "localhost", port = 5544, dbname = "devdb01",
    user = "dro", password = "sicillionbillions"
)

query <- "
SELECT pgs.stat_yards, pgs.stat_attempts, g.spread_close
FROM mart.player_game_stats pgs
LEFT JOIN mart.player_hierarchy ph USING(player_id)
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.player_id = $1
    AND pgs.season = $2
    AND pgs.week = $3
    AND pgs.stat_category = $4
LIMIT 1;
"

df <- dbGetQuery(conn, query, params = list('00-0019596', 2024, 1, 'passing'))
```

---

### Pattern 3: Player Historical Stats (Rolling Window)

**Use Case:** Get player's last N games for rolling features

```sql
SELECT
    pgs.season,
    pgs.week,
    pgs.stat_yards,
    pgs.stat_attempts,
    pgs.stat_touchdowns,
    g.kickoff,
    g.game_id

FROM mart.player_game_stats pgs
LEFT JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)

WHERE pgs.player_id = '00-0019596'
    AND pgs.stat_category = 'passing'
    AND pgs.season >= 2020  -- Filter to recent seasons

ORDER BY g.kickoff DESC  -- Most recent first

LIMIT 8;  -- Last 8 games
```

**Performance:** Uses `idx_player_game_stats_lookup` + `games_season_week_kickoff`

---

### Pattern 4: All Active QBs (Current Season)

**Use Case:** Get all active QBs for current week predictions

```sql
SELECT DISTINCT
    ph.player_id,
    ph.player_name,
    ph.current_team,
    ph.years_exp

FROM mart.player_hierarchy ph
INNER JOIN mart.player_game_stats pgs ON ph.player_id = pgs.player_id

WHERE ph.position_group = 'QB'
    AND pgs.season = 2024  -- Current season
    AND pgs.stat_category = 'passing'
    AND pgs.stat_attempts >= 10  -- Minimum attempts

ORDER BY ph.player_name;
```

---

### Pattern 5: Team Schedule Lookup

**Use Case:** Find upcoming games for a team

```sql
SELECT
    g.game_id,
    g.season,
    g.week,
    g.kickoff,
    g.home_team,
    g.away_team,
    g.spread_close,
    g.total_close

FROM games g

WHERE (g.home_team = 'KC' OR g.away_team = 'KC')  -- Team filter
    AND g.season = 2024
    AND g.game_status = 'scheduled'  -- Upcoming games only

ORDER BY g.kickoff;
```

---

## Index Usage Guide

### mart.player_game_stats Indexes

```sql
-- 1. Unique constraint (enforces data integrity)
idx_player_game_stats_unique: (player_id, season, week, stat_category)
-- Use: Primary key, prevents duplicates

-- 2. Player lookup
idx_player_game_stats_player_id: (player_id)
-- Use: Get all stats for a player across all games

-- 3. Season/week lookup
idx_player_game_stats_season_week: (season, week)
-- Use: Get all players for a specific game week

-- 4. Stat category filter
idx_player_game_stats_category: (stat_category)
-- Use: Filter by passing/rushing/receiving

-- 5. Position group filter
idx_player_game_stats_position_group: (position_group)
-- Use: Filter by QB/RB/WR/TE

-- 6. Composite lookup (MOST EFFICIENT)
idx_player_game_stats_lookup: (player_id, season, week)
    WHERE stat_category IN ('passing', 'rushing', 'receiving')
-- Use: Single player-game lookups (PREFERRED for predictions)
```

### games Indexes

```sql
-- 1. Primary key
games_pkey: (game_id)

-- 2. Team lookups
games_home_team_idx: (home_team)
games_away_team_idx: (away_team)

-- 3. Season/week lookup (CRITICAL for joins)
games_season_week_idx: (season, week)
-- Use: Joining to player_game_stats

-- 4. Betting context (composite)
idx_games_betting_context: (game_id, season, week, spread_close, total_close,
                             home_team, away_team)
    WHERE spread_close IS NOT NULL AND total_close IS NOT NULL
-- Use: Feature engineering with betting lines
```

### mart.player_hierarchy Indexes

```sql
-- 1. Primary key
idx_player_hierarchy_player_id: (player_id) UNIQUE

-- 2. Position group filter
idx_player_hierarchy_position_group: (position_group)

-- 3. Position + team composite
idx_player_hierarchy_position_team: (position_group, current_team, player_id)

-- 4. Team lookup
idx_player_hierarchy_team: (current_team)
```

---

## Performance Notes

### Query Optimization Checklist

**✅ DO:**
1. Always use parameterized queries (`%s`, `$1`, etc.) to prevent SQL injection
2. Use `LEFT JOIN` for games (some stats may not have betting lines)
3. Filter on indexed columns (`player_id`, `season`, `week`, `stat_category`)
4. Use `LIMIT` for single-row lookups
5. Use `EXPLAIN ANALYZE` to verify index usage

**❌ DON'T:**
1. Use `SELECT *` in production (specify columns explicitly)
2. Join without `(season, week)` composite key
3. Assume `game_id` exists in `player_game_stats`
4. Use `game_date`, `spread_line`, `total_line` (wrong column names)
5. Use bare `LA` for team abbreviation

### Connection Pooling (CRITICAL for backtests)

**Problem:** Creating new connections is expensive (100-150ms overhead)

**Solution:** Use connection pooling

```python
from psycopg2 import pool

# Initialize once at startup
db_pool = pool.SimpleConnectionPool(
    minconn=2,
    maxconn=8,
    host="localhost", port=5544, database="devdb01",
    user="dro", password="sicillionbillions"
)

# In process_game():
conn = db_pool.getconn()
try:
    df = pd.read_sql_query(query, conn, params=(player_id, season, week, 'passing'))
finally:
    db_pool.putconn(conn)
```

**Savings:** 100-150ms per query = 18-24 minutes for 148-game backtest

---

## Verification

### Quick Sanity Checks

```sql
-- 1. Verify player_game_stats does NOT have game_id
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'mart'
    AND table_name = 'player_game_stats'
    AND column_name = 'game_id';
-- Expected: 0 rows

-- 2. Verify games HAS game_id
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public'
    AND table_name = 'games'
    AND column_name = 'game_id';
-- Expected: 1 row

-- 3. Test canonical join pattern
SELECT COUNT(*)
FROM mart.player_game_stats pgs
LEFT JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.stat_category = 'passing'
    AND pgs.season = 2024;
-- Expected: ~500+ rows (all 2024 passing stats with game context)
```

---

## Troubleshooting

### Error: "column pgs.game_id does not exist"

**Cause:** Trying to join `player_game_stats` via `game_id`

**Fix:** Use Pattern 1 or Pattern 2 (join via `season, week`)

### Error: "column g.game_date does not exist"

**Cause:** Using wrong column name

**Fix:** Use `g.kickoff` instead

### Error: "more than one row returned by a subquery"

**Cause:** Not filtering to specific player's team (multi-team players)

**Fix:** Add team filter in games join: `(home_team = current_team OR away_team = current_team)`

### Slow Query (>1 second)

**Diagnosis:** Run `EXPLAIN ANALYZE`

```sql
EXPLAIN ANALYZE
SELECT ...
FROM mart.player_game_stats pgs
...
```

**Common Issues:**
- Missing `WHERE stat_category = 'passing'` (uses partial index)
- Not using `(player_id, season, week)` filter (forces seq scan)
- Joining games without `season, week` filter

---

## References

- **Full Schema Audit:** `docs/database/DATABASE_SCHEMA_AUDIT.md` (removed in the 2026-10 prune; tag `archive/pre-prune-2026-10`)
- **Quick Reference:** [DATABASE_QUICK_REFERENCE.md](DATABASE_QUICK_REFERENCE.md)
- **Production Code:** `R/train_hierarchical_v2_production.R` (canonical example)
- **Performance Analysis:** `docs/ENSEMBLE_BACKTEST_PERFORMANCE_ESTIMATE.md`

---

**Last Updated:** November 14, 2025
**Version:** 1.0
**Maintainer:** Claude Code (Sonnet 4.5)
