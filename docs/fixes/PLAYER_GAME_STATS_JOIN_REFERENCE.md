# Player Game Stats Join Reference Card

**Quick Reference:** How to correctly join `mart.player_game_stats` with `games` table

## The Problem

❌ **WRONG** - This will fail:
```sql
FROM mart.player_game_stats pgs
JOIN games g ON pgs.game_id = g.game_id  -- ERROR: column pgs.game_id does not exist
```

## The Solution

✅ **CORRECT** - Use composite key `(season, week, team)`:
```sql
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
```

## Why?

### Schema Reality

`mart.player_game_stats` has these columns for joining:
- `player_id` (text) - Player identifier
- `season` (integer) - NFL season year
- `week` (integer) - Week number (1-18)
- `current_team` (text) - Team abbreviation (from player_hierarchy)

**Note:** The view has `game_id_pattern` (computed string), but it's NOT for joins.

### Natural Key

Player game stats use `(player_id, season, week, stat_category)` as natural key because:
1. Players play one game per week (no ambiguity)
2. Data is player-centric, not game-centric
3. `season + week + team` uniquely identifies the game

## Copy-Paste Templates

### Python (passing stats)
```python
query = """
    SELECT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.stat_yards as passing_yards,
        pgs.stat_attempts,
        ph.position_group,
        ph.current_team,
        g.spread_close,
        g.total_close,
        CAST(g.temp AS NUMERIC) as temp,
        CAST(g.wind AS NUMERIC) as wind
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    LEFT JOIN games g ON
        g.season = pgs.season
        AND g.week = pgs.week
        AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
    WHERE pgs.stat_category = 'passing'
      AND pgs.season >= 2015
      AND pgs.stat_yards IS NOT NULL
      AND ph.position_group = 'QB'
    ORDER BY pgs.season, pgs.week, pgs.player_id
"""
```

### Python (rushing stats)
```python
query = """
    SELECT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.stat_yards as rushing_yards,
        pgs.stat_attempts as carries,
        ph.position_group,
        ph.current_team,
        g.spread_close,
        g.total_close
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    LEFT JOIN games g ON
        g.season = pgs.season
        AND g.week = pgs.week
        AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
    WHERE pgs.stat_category = 'rushing'
      AND pgs.season >= 2015
      AND pgs.stat_yards IS NOT NULL
      AND pgs.stat_yards > 0
    ORDER BY pgs.season, pgs.week, pgs.player_id
"""
```

### R (passing stats)
```r
query <- "
SELECT
  pgs.player_id,
  pgs.season,
  pgs.week,
  pgs.stat_yards as passing_yards,
  pgs.stat_attempts,
  pgs.stat_completions,
  ph.position_group,
  ph.current_team,
  ph.years_exp,
  g.spread_close,
  g.total_close,
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
  AND pgs.season BETWEEN 2015 AND 2025
  AND ph.position_group = 'QB'
ORDER BY pgs.season, pgs.week, pgs.player_id
"
```

## Key Points

### 1. Always join player_hierarchy first
```sql
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
```
**Why?** You need `ph.current_team` to match games.

### 2. Use LEFT JOIN for games (not INNER)
```sql
LEFT JOIN games g ON ...
```
**Why?** Some games might not have odds/weather data. LEFT JOIN prevents data loss.

### 3. Match home OR away team
```sql
AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
```
**Why?** Player's team could be home or away in the game.

### 4. Filter by stat_category
```sql
WHERE pgs.stat_category = 'passing'  -- or 'rushing', 'receiving'
```
**Why?** View combines all stat types. Always filter explicitly.

### 5. Filter by position_group
```sql
AND ph.position_group = 'QB'  -- or 'RB', 'WR', 'TE'
```
**Why?** Ensures position consistency (e.g., QBs for passing, RBs for rushing).

## Common Mistakes

### Mistake 1: Using pgs.game_id
```sql
-- ❌ WRONG
JOIN games g ON pgs.game_id = g.game_id
```
**Error:** `ERROR: column pgs.game_id does not exist`

### Mistake 2: Forgetting player_hierarchy
```sql
-- ❌ WRONG (missing current_team)
FROM mart.player_game_stats pgs
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = pgs.current_team OR ...)  -- pgs.current_team doesn't exist!
```
**Error:** `ERROR: column pgs.current_team does not exist`

**Fix:** Use `ph.current_team` after joining `player_hierarchy`

### Mistake 3: Using pgs.current_team directly
```sql
-- ⚠️ SOMETIMES WORKS (but not recommended)
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = pgs.current_team OR ...)  -- Only works if view was updated
```
**Why risky?** Depends on view definition. Always join `player_hierarchy` explicitly for clarity.

### Mistake 4: Missing stat_category filter
```sql
-- ❌ WRONG (returns passing + rushing + receiving for same player/game)
SELECT * FROM mart.player_game_stats pgs
WHERE pgs.season = 2024 AND pgs.week = 1
```
**Result:** 3 rows per player (if they have all stat types)

**Fix:** Always filter `WHERE pgs.stat_category = 'passing'`

## Testing Your Query

### 1. Check row count is reasonable
```sql
-- Should return ~1.2M rows (2015-2024)
SELECT COUNT(*) FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.season >= 2015;
```

### 2. Verify no NULL games (if INNER JOIN)
```sql
-- Should return 0 if all games matched
SELECT COUNT(*) FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.season >= 2015
  AND g.game_id IS NULL;  -- Check for unmatched games
```

### 3. Sample data looks correct
```sql
-- Sample 10 rows
SELECT
  pgs.player_id,
  pgs.season,
  pgs.week,
  pgs.stat_category,
  ph.current_team,
  g.game_id,
  g.home_team,
  g.away_team
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.season = 2024 AND pgs.week = 1
LIMIT 10;
```

## Working Examples in Codebase

**Verified correct implementations:**
1. `py/ensemble/enhanced_ensemble_v6.py` (lines 218-223)
2. `py/models/train_xgboost_canonical.py` (lines 150-155)
3. `R/train_hierarchical_v2_production.R` (lines 81-86)
4. `R/state_space/train_state_space_v2_canonical.R` (lines 73-78)

**When in doubt:** Copy from these files!

## Schema Reference

### mart.player_game_stats columns
```
player_id                    TEXT        -- Player identifier
season                       INTEGER     -- Season year
week                         INTEGER     -- Week number (1-18)
stat_category                TEXT        -- 'passing', 'rushing', 'receiving'
player_display_name          TEXT
player_position              TEXT        -- Raw position (QB, RB, WR, TE)
stat_attempts                NUMERIC     -- Attempts/carries/targets
stat_completions             NUMERIC     -- Completions/receptions (NULL for rushing)
stat_yards                   NUMERIC     -- Yards gained
stat_touchdowns              NUMERIC     -- TDs scored
stat_negative                NUMERIC     -- Interceptions (passing only)
position_group               TEXT        -- 'QB', 'RB', 'WR', 'TE' (from hierarchy)
current_team                 TEXT        -- Team abbr (from hierarchy)
hierarchy_position           INTEGER     -- Team depth chart position
hierarchy_team_position      INTEGER     -- Overall depth chart position
game_id_pattern              TEXT        -- For display only (NOT for joins!)
```

### games table join keys
```
game_id         TEXT PRIMARY KEY
season          INTEGER
week            INTEGER
home_team       TEXT
away_team       TEXT
spread_close    NUMERIC
total_close     NUMERIC
temp            TEXT (cast to NUMERIC)
wind            TEXT (cast to NUMERIC)
```

---

**Created:** November 14, 2025
**Audit Reference:** docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md
**Last Validated:** November 8, 2025 (State-Space v2 canonical training)
