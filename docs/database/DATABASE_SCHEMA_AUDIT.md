# NFL Analytics Database Schema Audit

**Date:** 2025-10-30
**Database:** `devdb01` (PostgreSQL + TimescaleDB)
**Total Size:** ~2.5 GB

## Executive Summary

Comprehensive audit of the NFL Analytics database schema identifying key tables, data quality considerations, and recommendations for data mart design.

### Key Findings

1. **Schema Organization**: Well-organized with 5 functional schemas (public, mart, predictions, reference, monitoring)
2. **Team Abbreviations**: Inconsistency between `LA` (games table) vs `LAC`/`LAR` (teams table) for Los Angeles teams
3. **Temporal Data**: Primary time column is `kickoff` (TIMESTAMP WITH TIME ZONE), not `game_date` or `gameday`
4. **Betting Lines**: Uses `spread_close` and `total_close`, not `spread_line`/`total_line`
5. **Weather Data**: Stored as TEXT, requires parsing in application layer

---

## Schema Overview

### 1. **public** Schema (Core NFL Data)
Primary transactional and source-of-truth tables.

| Table | Size | Records (est.) | Key Columns |
|-------|------|----------------|-------------|
| `plays` | 854 MB | ~800K | game_id, play_id, posteam, defteam, down, ydstogo, yardline_100 |
| `participation` | 335 MB | ~5M | game_id, play_id, player_id |
| `rosters_weekly` | 414 MB | ~50K | season, team, week, player_id, position |
| `depth_charts` | 207 MB | ~200K | season, club_code, week, player_id, depth_team |
| `snap_counts` | 102 MB | ~150K | game_id, player_id, offense_snaps, defense_snaps |
| `injuries` | 39 MB | ~50K | season, team, week, player_id, report_status |
| `ftn_charting` | 35 MB | ~300K | game_id, play_id, ngs_player_id (advanced charting) |
| `pfr_defense` | 22 MB | ~30K | player_id, season, week (Pro Football Reference defensive stats) |
| `games` | 13 MB | ~6K | game_id, home_team, away_team, kickoff, spread_close, total_close |

**Critical Tables for Modeling:**
- `games`: Core game-level data with betting lines
- `plays`: Play-by-play data foundation
- `rosters_weekly`: Player availability tracking

### 2. **mart** Schema (Analytical Data Mart)
Aggregated, time-variant features for modeling.

| Table | Size | Purpose |
|-------|------|---------|
| `asof_team_features` | 6.4 MB | Point-in-time team features |
| `rolling_features` | 2.9 MB | Rolling window statistics |
| `team_epa` | 2.3 MB | Team EPA aggregations |
| `team_4th_down_features` | 2.0 MB | 4th down decision analytics |
| `team_ratings` | 1.7 MB | Historical team strength ratings |
| `bayesian_player_ratings` | 232 KB | Bayesian player skill ratings |

**Design Pattern:** Star schema with time-series optimization via TimescaleDB.

### 3. **predictions** Schema (ML Predictions & Feedback Loop)

| Table | Size | Purpose |
|-------|------|---------|
| `retrospectives` | 1.2 MB | Post-game analysis and model feedback |
| `game_predictions` | 344 KB | Game-level probabilistic predictions |
| `prop_predictions` | 144 KB | Player prop predictions |
| `bayesian_prop_predictions` | 96 KB | Bayesian player prop predictions |
| `learning_loop` | 56 KB | Reinforcement learning feedback |
| `prediction_versions` | 48 KB | Model versioning |

**Key Feature:** Closed-loop learning system with retrospectives.

### 4. **reference** Schema (Lookup Tables)

| Table | Size | Purpose |
|-------|------|---------|
| `teams` | 96 KB | Team metadata (32 teams) |
| `team_abbreviations` | 64 KB | Historical abbreviation mappings |
| `column_mappings` | 48 KB | Schema metadata |
| `stadiums` | 24 KB | Stadium information |

### 5. **monitoring** Schema (Observability)

All tables: 16 KB each (minimal data, recently created)
- `model_metrics`: Model performance tracking
- `feature_drift`: Data drift detection
- `alerts`: Alerting system
- `predictions`: Prediction monitoring

---

## Data Quality Findings

### 1. Team Abbreviation Inconsistencies

**Issue:** LA Rams represented as `LA` in games table but `LAR` in teams table.

```sql
-- Games table uses: LA
SELECT DISTINCT home_team FROM games WHERE home_team LIKE 'LA%';
-- Returns: LA, LAC

-- Teams table uses: LAR, LAC
SELECT team_abbr FROM teams WHERE team_abbr LIKE 'LA%';
-- Returns: LA, LAC, LAR
```

**Impact:** JOINs between games and teams may fail for LA Rams games.

**Recommendation:**
- Standardize on `LAR` for Rams, `LAC` for Chargers
- Create migration script to UPDATE games table
- Add CHECK constraint to prevent `LA` abbreviation

### 2. Temporal Column Naming

**Finding:** Primary time column is `kickoff` (not `game_date` or `gameday`).

```sql
-- Correct usage:
SELECT game_id, kickoff, home_team, away_team
FROM games
WHERE kickoff >= '2024-01-01';
```

**Common Mistakes in Code:**
- ❌ `g.game_date` (doesn't exist)
- ❌ `g.gameday` (doesn't exist)
- ✅ `g.kickoff` (correct)

**Recommendation:** Update all feature engineering scripts to use `kickoff`.

### 3. Betting Line Column Names

**Finding:** Uses `spread_close` and `total_close`, not `spread_line`/`total_line`.

```sql
-- Correct columns:
SELECT
    game_id,
    spread_close,      -- NOT spread_line
    total_close,       -- NOT total_line
    home_moneyline,
    away_moneyline
FROM games;
```

**Recommendation:** Create VIEWs with alias columns for compatibility if needed.

### 4. Weather Data Storage

**Finding:** `temp` and `wind` stored as TEXT, not numeric types.

```sql
-- Current schema:
temp | text  -- e.g., "72", "45"
wind | text  -- e.g., "10 mph", "15"
```

**Impact:** Requires parsing in application layer:
```python
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["wind"] = pd.to_numeric(df["wind"].str.extract(r'(\d+)')[0], errors="coerce")
```

**Recommendation:**
- Add computed columns `temp_numeric REAL`, `wind_numeric REAL`
- Create trigger to populate on INSERT/UPDATE
- Maintain TEXT for source-of-truth

### 5. Missing Columns for Recent Features

**Recently Added:**
- `third_down_converted` (BOOLEAN) - Added 2025-10-30
- `fourth_down_converted` (BOOLEAN) - Added 2025-10-30
- `yardline_100` (INTEGER) - Verified present
- `game_seconds_remaining` (INTEGER) - Verified present
- `half_seconds_remaining` (INTEGER) - Verified present

**Status:** ✅ All feature engineering dependencies satisfied.

---

## Data Mart Recommendations

### Proposed Fact Tables

#### 1. `fact_game_outcomes`
Point-in-time game results with all relevant dimensions.

```sql
CREATE TABLE mart.fact_game_outcomes (
    game_id TEXT PRIMARY KEY,
    season INTEGER,
    week INTEGER,
    kickoff TIMESTAMP WITH TIME ZONE,
    home_team TEXT REFERENCES teams(team_abbr),
    away_team TEXT REFERENCES teams(team_abbr),
    home_score INTEGER,
    away_score INTEGER,
    spread_close REAL,
    total_close REAL,
    home_moneyline REAL,
    away_moneyline REAL,
    -- Derived metrics
    home_covered BOOLEAN GENERATED ALWAYS AS (
        (home_score - away_score) > spread_close
    ) STORED,
    total_over BOOLEAN GENERATED ALWAYS AS (
        (home_score + away_score) > total_close
    ) STORED
);
```

#### 2. `fact_team_game_features`
Team-level features as-of game time (denormalized for fast access).

```sql
CREATE TABLE mart.fact_team_game_features (
    game_id TEXT,
    team TEXT,
    location TEXT CHECK (location IN ('home', 'away')),
    -- Recent form
    last_4_epa REAL,
    last_4_win_pct REAL,
    -- Situational
    days_rest INTEGER,
    is_division_game BOOLEAN,
    is_thursday_night BOOLEAN,
    -- Environmental
    travel_distance INTEGER,
    temp_numeric REAL,
    wind_numeric REAL,
    -- Injuries
    key_players_out INTEGER,
    injury_load_score REAL,
    PRIMARY KEY (game_id, team)
);
```

#### 3. `dim_game_context`
Game-level context and environmental factors.

```sql
CREATE TABLE mart.dim_game_context (
    game_id TEXT PRIMARY KEY,
    stadium TEXT,
    roof TEXT,
    surface TEXT,
    temp_numeric REAL,
    wind_numeric REAL,
    weather_condition TEXT,
    is_prime_time BOOLEAN,
    is_playoff BOOLEAN,
    referee TEXT
);
```

### Proposed Views for Compatibility

```sql
-- Alias view for legacy code
CREATE VIEW games_v1 AS
SELECT
    game_id,
    season,
    week,
    kickoff AS game_date,           -- Alias for compatibility
    kickoff AS gameday,              -- Alias for compatibility
    EXTRACT(DOW FROM kickoff)::INTEGER AS day_of_week,
    EXTRACT(HOUR FROM kickoff)::INTEGER AS game_hour,
    home_team,
    away_team,
    spread_close AS spread_line,    -- Alias for compatibility
    total_close AS total_line,      -- Alias for compatibility
    home_moneyline,
    away_moneyline,
    home_score,
    away_score,
    -- Parsed numeric weather
    COALESCE(temp::REAL, NULL) AS temp_numeric,
    COALESCE(NULLIF(regexp_replace(wind, '[^0-9]', '', 'g'), '')::REAL, NULL) AS wind_numeric
FROM games;
```

---

## Integration Points

### Feature Engineering Modules
All modules should reference this audit for correct column names:

1. **nfl_analytics/features/defensive_situational_strength.py**
   - ✅ Uses `yardline_100`, `game_seconds_remaining`
   - ✅ Added `third_down_converted`, `fourth_down_converted`

2. **nfl_analytics/features/environmental_contextual_features.py**
   - ✅ Uses `kickoff` (line 113)
   - ✅ Parses `temp` and `wind` as TEXT

3. **nfl_analytics/features/feature_interactions.py**
   - ✅ Fixed to use `kickoff` instead of `game_date`
   - ✅ Fixed to use `spread_close` instead of `spread_line`
   - ✅ Parses `temp` and `wind` properly

### Known Code Pattern
```python
# CORRECT pattern for feature engineering:
query = """
SELECT
    g.game_id,
    g.season,
    g.week,
    g.kickoff,                      -- NOT game_date or gameday
    EXTRACT(DOW FROM g.kickoff) as day_of_week,
    g.home_team,
    g.away_team,
    g.spread_close,                 -- NOT spread_line
    g.total_close,                  -- NOT total_line
    g.temp,                         -- TEXT, needs parsing
    g.wind,                         -- TEXT, needs parsing
    t.team_abbr                     -- Use LAR for Rams, LAC for Chargers
FROM games g
LEFT JOIN teams t ON g.home_team = t.team_abbr
WHERE g.season >= 2024
ORDER BY g.kickoff;
"""

# Parse weather
df["temp_numeric"] = pd.to_numeric(df["temp"], errors="coerce")
df["wind_numeric"] = pd.to_numeric(
    df["wind"].astype(str).str.extract(r'(\d+)')[0],
    errors="coerce"
)
```

---

## Action Items

### High Priority
1. ✅ Document schema in this file
2. ⏳ Fix LA/LAR/LAC abbreviation inconsistency in games table
3. ⏳ Create compatibility VIEWs for legacy code
4. ⏳ Update CLAUDE.md with schema reference
5. ⏳ Update README.md with database architecture section

### Medium Priority
6. Add `temp_numeric` and `wind_numeric` computed columns to games table
7. Create data quality tests for team abbreviations
8. Implement CHECK constraints for valid team codes
9. Add foreign key from games.home_team/away_team to teams.team_abbr

### Low Priority
10. Migrate historical games to use LAR abbreviation
11. Create materialized view for commonly joined tables
12. Document all indexes and their purposes
13. Set up automated schema documentation generation

---

## Database Statistics

### Current State (as of 2025-10-30)

```
Total Database Size: ~2.5 GB
Total Tables: 56
  - public schema: 30 tables
  - mart schema: 10 tables
  - predictions schema: 6 tables
  - reference schema: 4 tables
  - monitoring schema: 4 tables

Largest Tables:
  1. plays (854 MB) - ~800K play-by-play records
  2. participation (335 MB) - ~5M player participation records
  3. rosters_weekly (414 MB) - ~50K weekly roster records
  4. depth_charts (207 MB) - ~200K depth chart records
  5. snap_counts (102 MB) - ~150K snap count records

Data Coverage:
  - Seasons: 1999-2025 (27 seasons)
  - Games: ~6,000 games
  - Plays: ~800,000 plays
  - Players: ~10,000 unique players
```

---

## Conclusion

The database is well-structured with clear separation of concerns across schemas. Key recommendations:

1. **Standardize team abbreviations** - Critical for JOIN accuracy
2. **Use correct temporal columns** - `kickoff` is the source of truth
3. **Parse weather data** - Always convert TEXT to numeric in application layer
4. **Reference this document** - All feature engineering should validate column names

The schema supports advanced analytics and ML workflows with proper data lineage and versioning.

---

**Audit Conducted By:** Claude Code (Anthropic)
**Last Updated:** 2025-10-30
**Next Review:** 2026-01-01 or upon major schema changes
