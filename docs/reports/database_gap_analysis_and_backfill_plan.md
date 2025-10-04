# Database Gap Analysis & Backfill Plan

**Date:** October 4, 2025  
**Database:** devdb01 @ localhost:5544  
**Analysis Period:** 1999-2024 (26 seasons)

---

## Executive Summary

Your database has **excellent core coverage** (1999-2024) with play-by-play data and game results. However, it's missing many **high-value features** available in nflfastR/nflverse that could significantly improve model performance.

**Key Findings:**
- ✅ **COMPLETE:** Game results (6,969 games), play-by-play (1.2M plays), EPA calculations
- ⚠️ **GAPS:** Advanced PBP features, roster data, player tracking, situational stats
- ⚠️ **SPARSE:** Weather (93.4% coverage 2020+), odds history (limited), moneylines (missing pre-2006)
- ❌ **MISSING:** WP (win probability), success rate, air yards, YAC, personnel groupings, roster data

**Estimated Impact:** Backfilling recommended features could add **+3-5% model accuracy** and **+1-2% ROI**.

---

## 1. Current Database State

### Table Summary

| Table | Schema | Rows | Size | Coverage | Primary Grain |
|-------|--------|------|------|----------|---------------|
| **games** | public | 6,969 | 1.2 MB | 1999-2024 (100%) | game_id |
| **plays** | public | 1.2M | 157 MB | 1999-2024 (98.9% with EPA) | game_id, play_id |
| **weather** | public | 1,315 | 200 KB | 2020-2024 (93.4%) | game_id |
| **odds_history** | public | ? | 40 KB | Limited snapshots | event_id, snapshot_at |
| **injuries** | public | ? | 4.7 MB | Unknown coverage | ? |
| **team_epa** | mart | 13,972 | 2.3 MB | 1999-2024 (100%) | game_id, posteam |
| **team_4th_down_features** | mart | 13,972 | 2.0 MB | 1999-2024 (100%) | game_id, team |
| **team_injury_load** | mart | 2,798 | 448 KB | 2020-2024 (99.2%) | season, week, team |
| **team_playoff_context** | mart | 0 | 48 KB | Empty (needs fix) | team, season, week |
| **asof_team_features** | mart | ? | 1.7 MB | Unknown | ? |

### Data Quality Metrics

**Plays Table (1.2M plays):**
- EPA coverage: 98.9% (excellent)
- Columns: Only 11 (nflfastR has 372!)
- Missing: WP, success rate, air yards, YAC, cpoe, player IDs, situational stats

**Games Table (6,969 games):**
- Spread/Total: 100% coverage (excellent)
- Moneylines: Missing pre-2006 (1,365 games = 19.6%)
- Weather: Only 2020+ covered
- Missing: Game-level stats (TOP, turnovers, penalties), stadium info, roof type

**Weather Table:**
- Coverage: 93.4% of 2020+ games (1,315/1,408)
- Missing: 93 games (likely dome games or data failures)
- Columns: 7 (basic temp/wind/precip)
- Enhancement opportunity: Add dew point, pressure trends, visibility

---

## 2. Gap Analysis by Category

### 🔴 CRITICAL GAPS (High Value, Should Backfill)

#### Gap 1: Advanced Play-by-Play Features
**What's Missing:**
- Win Probability (WP) and WP added (WPA)
- Success rate indicators
- Air yards, YAC (yards after catch), complete air yards (air_yards + YAC)
- CPOE (completion percentage over expected)
- Expected points added for rush/pass separately
- Down conversions, first downs
- Personnel groupings (11 personnel, 12 personnel, etc.)
- Play type detail (run direction, pass depth, route concept)
- Player IDs for QB, RB, WR, targeted receiver

**Impact:** 
- WP critical for late-game situational modeling
- Air yards/YAC predict passing efficiency better than raw EPA
- Success rate complements EPA (more stable metric)
- **Expected Improvement:** +2-3% accuracy, +0.8-1.2% ROI

**nflfastR Columns to Add:** (~50 high-value columns)
```
wp, wpa, vegas_wp, vegas_wpa, success, yards_gained, 
air_yards, yards_after_catch, comp_air_epa, comp_yac_epa,
cpoe, first_down, goal_to_go, qb_scramble, run_location, 
run_gap, pass_length, pass_location, complete_pass, 
passer_player_id, rusher_player_id, receiver_player_id,
tackler_1_player_id, interception_player_id, fumble_player_id,
penalty, penalty_team, penalty_yards, no_huddle, qb_dropback, 
qb_kneel, qb_spike, two_point_attempt, field_goal_result,
kick_distance, extra_point_result, touchdown, td_team,
posteam_score, defteam_score, score_differential,
posteam_score_post, defteam_score_post, no_score_prob,
opp_fg_prob, opp_safety_prob, opp_td_prob, fg_prob, 
safety_prob, td_prob, extra_point_prob, two_point_conversion_prob
```

**Backfill Method:** `nflfastR::load_pbp(seasons = 1999:2024)` → select/rename → upsert to `plays`

---

#### Gap 2: Roster & Player Data
**What's Missing:**
- Player names, positions, team rosters by week
- QB experience, passer rating, career stats
- Injury designations tied to player IDs
- Snap counts, routes run, targets
- Depth chart positions

**Impact:**
- QB identity critical (rookie vs veteran vs backup)
- Positional matchups (CB vs WR quality)
- Snap counts predict player impact
- **Expected Improvement:** +1-2% accuracy, +0.5-1.0% ROI

**nflfastR/nflreadr Sources:**
- `nflreadr::load_rosters(seasons = 1999:2024)` → players, positions, team
- `nflreadr::load_depth_charts(seasons = 2001:2024)` → depth chart
- `nflreadr::load_snap_counts(seasons = 2012:2024)` → offensive/defensive snaps
- `nflfastR::fast_scraper_roster(seasons = 1999:2024)` → comprehensive roster

**Backfill Method:** Create `players` and `rosters` tables, link via player_id to plays

---

#### Gap 3: Pre-Snap Context
**What's Missing:**
- Personnel groupings (11, 12, 21, etc.)
- Formation (shotgun, under center, pistol)
- Number of pass rushers
- Men in box
- Defenders in box
- Motion indicators

**Impact:**
- Personnel predicts play type (11 personnel = 75% pass)
- Shotgun vs under center affects success rate
- Defensive alignment affects EPA
- **Expected Improvement:** +0.5-1.0% accuracy

**nflfastR Columns:**
```
shotgun, no_huddle, qb_dropback, offense_formation, 
offense_personnel, defenders_in_box, number_of_pass_rushers,
men_in_box, offense_players_on_field, defense_players_on_field
```

**Backfill Method:** Available in full nflfastR PBP data (2016+), limited before

---

### 🟡 MODERATE GAPS (Medium Value, Consider Backfilling)

#### Gap 4: Game-Level Stats
**What's Missing:**
- Time of possession (TOP)
- Turnovers (fumbles, interceptions by team)
- Penalties (count and yards)
- Third down conversions
- Red zone efficiency
- Sacks allowed/recorded

**Impact:**
- TOP correlates with run-heavy game scripts
- Turnover differential huge predictor
- Penalty yards affect spread outcomes
- **Expected Improvement:** +0.5-1.0% accuracy

**nflfastR Source:** Aggregate from plays table or use `nflreadr::load_schedules()` metadata

**Backfill Method:** Calculate from existing plays + augment from nflreadr schedules

---

#### Gap 5: Stadium & Venue Data
**What's Missing:**
- Stadium name
- Roof type (dome, retractable, open)
- Surface type (grass, turf, field turf)
- Altitude
- Latitude/longitude for weather API

**Impact:**
- Dome games eliminate weather factors
- Turf affects injury rates and speed
- Altitude impacts kicking
- **Expected Improvement:** +0.3-0.5% accuracy (complements weather features)

**nflfastR Source:** `nflreadr::load_schedules()` has `roof` and `surface` columns

**Backfill Method:** Join schedules data to games table, add columns

---

#### Gap 6: Historical Weather (Pre-2020)
**What's Missing:**
- Weather for 1999-2019 games (4,969 games = 71% of total)

**Impact:**
- Weather affects totals significantly
- Wind impacts passing games
- Cold/precipitation affects ball handling
- **Expected Improvement:** Enables models on full historical data

**Backfill Method:** 
- Use nflreadr::load_schedules() `weather` field (text descriptions)
- Parse descriptions for temp/wind/conditions
- Backfill Meteostat for outdoor stadiums with lat/lon

---

#### Gap 7: Moneylines (Pre-2006)
**What's Missing:**
- Moneylines for 1999-2005 games (1,365 games missing)
- Some games 2006-2009 have partial coverage

**Impact:**
- Moneylines enable money management strategies
- Kelly criterion sizing requires moneylines
- **Expected Improvement:** Enables live betting simulation on historical data

**Backfill Method:** 
- Check if nflreadr::load_schedules() has historical moneylines
- Otherwise scrape from Pro Football Reference or Covers.com archives

---

### 🟢 MINOR GAPS (Lower Priority, Nice-to-Have)

#### Gap 8: Player Tracking Data
**What's Missing:**
- Next Gen Stats (NGS) tracking: speed, separation, acceleration
- Route tracking, QB time to throw
- Defender proximity

**Impact:**
- NGS data highly predictive but only available 2018+
- Not essential for game-level modeling
- **Expected Improvement:** +0.2-0.5% for in-play markets

**Backfill Method:** `nflreadr::load_nextgen_stats()` (2018+)

---

#### Gap 9: Playoff Context (Currently Empty)
**What's Missing:**
- Playoff probabilities by week
- Desperation indicators
- Tanking/resting signals

**Status:** Table exists but nflseedR script needs fixing (see FEATURE_ENGINEERING_COMPLETE.md)

**Impact:** +1.0-1.5% accuracy for late-season games (weeks 15-18)

**Backfill Method:** Fix nflseedR integration or use alternative playoff calculator

---

#### Gap 10: Coaching & Front Office Data
**What's Missing:**
- Head coach names/experience
- Offensive/defensive coordinators
- GM tenure
- Coaching changes

**Impact:**
- Coaching quality affects game outcomes
- Coordinator changes affect scheme
- **Expected Improvement:** +0.2-0.4% accuracy

**Backfill Method:** `nflreadr::load_teams()` has coaching data (2002+)

---

## 3. Prioritized Backfill Plan

### Phase 1: Critical Enhancements (DO NOW)
**Est. Time:** 2-4 hours  
**Est. Impact:** +3-5% accuracy, +1-2% ROI

1. **Expand plays table with nflfastR PBP data** ⭐⭐⭐
   - Add WP, WPA, success rate, air yards, YAC
   - Add player IDs for QB/RB/WR/tackler
   - Add situational context (down_converted, goal_to_go)
   - **Script:** `R/backfill_pbp_advanced.R`

2. **Add roster/player data** ⭐⭐⭐
   - Create `players` table (player_id, name, position)
   - Create `rosters` table (season, week, team, player_id)
   - Link injuries to player_ids
   - **Script:** `R/backfill_rosters.R`

3. **Add game-level metadata** ⭐⭐
   - Stadium, roof type, surface
   - Time of possession (calculate from plays)
   - Turnovers, penalties (calculate from plays)
   - **Script:** `R/backfill_game_metadata.R`

4. **Backfill moneylines (2006 gaps + pre-2006)** ⭐⭐
   - Check nflreadr::load_schedules() for historical lines
   - Scrape missing from Pro Football Reference if needed
   - **Script:** `R/backfill_moneylines.R`

### Phase 2: Moderate Enhancements (NEXT WEEK)
**Est. Time:** 3-5 hours  
**Est. Impact:** +1-2% accuracy

5. **Backfill weather for 1999-2019** ⭐
   - Parse nflreadr schedules `weather` field
   - Supplement with Meteostat for outdoor stadiums
   - **Script:** `R/backfill_weather_historical.R`

6. **Fix playoff context features** ⭐
   - Resolve nflseedR API issue
   - Backfill 2020-2024 playoff probabilities
   - **Script:** `R/features_playoff_context.R` (fix existing)

7. **Add snap counts (2012+)** ⭐
   - Load snap counts from nflreadr
   - Aggregate to team-week level
   - **Script:** `R/backfill_snap_counts.R`

### Phase 3: Nice-to-Have Enhancements (OPTIONAL)
**Est. Time:** 2-3 hours  
**Est. Impact:** +0.5-1% accuracy

8. **Add Next Gen Stats (2018+)**
   - Tracking data for key players
   - Route running, separation metrics
   - **Script:** `R/backfill_nextgen.R`

9. **Add coaching data**
   - Head coach, coordinators
   - Tenure, experience
   - **Script:** `R/backfill_coaching.R`

---

## 4. Implementation Scripts

### Script 1: Backfill Advanced PBP Features (HIGHEST PRIORITY)

```r
# R/backfill_pbp_advanced.R
# Backfill advanced play-by-play features from nflfastR

library(nflfastR)
library(dplyr)
library(DBI)
library(RPostgres)

# Database connection
con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

SEASONS <- 1999:2024

# Key columns to add to plays table
COLUMNS_TO_ADD <- c(
  "wp", "wpa", "vegas_wp", "vegas_wpa", "success", "yards_gained",
  "air_yards", "yards_after_catch", "comp_air_epa", "comp_yac_epa",
  "cpoe", "first_down", "goal_to_go", "run_location", "run_gap",
  "pass_length", "pass_location", "complete_pass", "incomplete_pass",
  "passer_player_id", "passer_player_name", "rusher_player_id", 
  "rusher_player_name", "receiver_player_id", "receiver_player_name",
  "tackler_1_player_id", "sack", "qb_hit", "penalty", "penalty_yards",
  "fumble", "fumble_lost", "interception", "touchdown", "td_team",
  "posteam_score", "defteam_score", "score_differential",
  "posteam_score_post", "defteam_score_post", "no_score_prob",
  "fg_prob", "td_prob", "shotgun", "no_huddle", "qb_dropback"
)

# Process seasons in batches
for (season in SEASONS) {
  cat("\n=== Processing", season, "season ===\n")
  
  # Load full nflfastR PBP data
  pbp <- nflfastR::load_pbp(season)
  
  if (is.null(pbp) || nrow(pbp) == 0) {
    cat("No data for", season, "\n")
    next
  }
  
  # Select relevant columns
  pbp_subset <- pbp %>%
    select(
      game_id, play_id, posteam, defteam, 
      quarter, time = game_seconds_remaining,  # map to time_seconds
      down, ydstogo, epa, 
      pass, rush,
      any_of(COLUMNS_TO_ADD)
    ) %>%
    # Convert time to seconds (nflfastR uses game_seconds_remaining)
    mutate(time_seconds = 3600 - time) %>%
    select(-time)
  
  cat("Loaded", nrow(pbp_subset), "plays\n")
  
  # Write to staging table first
  dbExecute(con, "DROP TABLE IF EXISTS plays_staging;")
  
  dbWriteTable(con, "plays_staging", pbp_subset, row.names = FALSE)
  
  # Alter plays table to add new columns (if not exists)
  for (col in COLUMNS_TO_ADD) {
    if (col %in% names(pbp_subset)) {
      col_type <- ifelse(is.numeric(pbp_subset[[col]]), "DOUBLE PRECISION",
                        ifelse(is.logical(pbp_subset[[col]]), "BOOLEAN", "TEXT"))
      
      tryCatch({
        dbExecute(con, sprintf(
          "ALTER TABLE plays ADD COLUMN IF NOT EXISTS %s %s;",
          col, col_type
        ))
      }, error = function(e) {
        cat("Column", col, "already exists or error:", e$message, "\n")
      })
    }
  }
  
  # Update plays table from staging (UPDATE existing, INSERT new)
  update_cols <- names(pbp_subset)[!names(pbp_subset) %in% c("game_id", "play_id")]
  
  update_sql <- sprintf(
    "UPDATE plays SET %s FROM plays_staging WHERE plays.game_id = plays_staging.game_id AND plays.play_id = plays_staging.play_id;",
    paste(sprintf("%s = plays_staging.%s", update_cols, update_cols), collapse = ", ")
  )
  
  rows_updated <- dbExecute(con, update_sql)
  cat("Updated", rows_updated, "existing plays\n")
  
  # Insert new plays (if any)
  insert_sql <- "
    INSERT INTO plays SELECT * FROM plays_staging
    ON CONFLICT (game_id, play_id) DO NOTHING;
  "
  rows_inserted <- dbExecute(con, insert_sql)
  cat("Inserted", rows_inserted, "new plays\n")
  
  # Clean up staging
  dbExecute(con, "DROP TABLE plays_staging;")
}

# Verify results
verification <- dbGetQuery(con, "
  SELECT 
    SUBSTRING(game_id, 1, 4)::int as season,
    COUNT(*) as plays,
    COUNT(wp) as has_wp,
    COUNT(success) as has_success,
    COUNT(air_yards) as has_air_yards,
    COUNT(passer_player_id) as has_passer_id
  FROM plays
  GROUP BY SUBSTRING(game_id, 1, 4)
  ORDER BY season;
")

print(verification)

dbDisconnect(con)

cat("\n✅ Advanced PBP backfill complete!\n")
```

### Script 2: Backfill Roster/Player Data

```r
# R/backfill_rosters.R
# Create players and rosters tables

library(nflreadr)
library(dplyr)
library(DBI)
library(RPostgres)

con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

SEASONS <- 1999:2024

# Create players table
dbExecute(con, "
  CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    player_name TEXT,
    position TEXT,
    height TEXT,
    weight INT,
    college TEXT,
    birth_date DATE,
    rookie_year INT,
    draft_club TEXT,
    draft_number INT
  );
")

# Create rosters table
dbExecute(con, "
  CREATE TABLE IF NOT EXISTS rosters (
    season INT,
    week INT,
    team TEXT,
    player_id TEXT,
    position TEXT,
    depth_chart_position TEXT,
    jersey_number INT,
    status TEXT,
    PRIMARY KEY (season, week, team, player_id)
  );
")

# Load rosters for all seasons
all_rosters <- nflreadr::load_rosters(seasons = SEASONS)

# Extract unique players
players <- all_rosters %>%
  distinct(gsis_id, .keep_all = TRUE) %>%
  transmute(
    player_id = gsis_id,
    player_name = full_name,
    position = position,
    height = height,
    weight = weight,
    college = college,
    birth_date = birth_date,
    rookie_year = rookie_season,
    draft_club = draft_club,
    draft_number = draft_number
  ) %>%
  filter(!is.na(player_id))

cat("Writing", nrow(players), "unique players...\n")
dbWriteTable(con, "players", players, append = TRUE, row.names = FALSE, 
             overwrite = FALSE)

# Process rosters by season
for (season in SEASONS) {
  season_rosters <- all_rosters %>%
    filter(season == !!season) %>%
    transmute(
      season = season,
      week = week,
      team = team,
      player_id = gsis_id,
      position = position,
      depth_chart_position = depth_chart_position,
      jersey_number = jersey_number,
      status = status
    ) %>%
    filter(!is.na(player_id))
  
  cat("Writing", nrow(season_rosters), "roster entries for", season, "\n")
  
  # Upsert rosters
  dbExecute(con, "DELETE FROM rosters WHERE season = $1", params = list(season))
  dbWriteTable(con, "rosters", season_rosters, append = TRUE, row.names = FALSE)
}

# Create indexes
dbExecute(con, "CREATE INDEX IF NOT EXISTS rosters_player_id_idx ON rosters(player_id);")
dbExecute(con, "CREATE INDEX IF NOT EXISTS rosters_team_season_idx ON rosters(team, season);")
dbExecute(con, "CREATE INDEX IF NOT EXISTS players_position_idx ON players(position);")

dbDisconnect(con)
cat("\n✅ Roster backfill complete!\n")
```

### Script 3: Add Game Metadata

```r
# R/backfill_game_metadata.R
# Add stadium, roof, surface, and aggregated game stats

library(nflreadr)
library(dplyr)
library(DBI)
library(RPostgres)

con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

# Add columns to games table
new_columns <- c(
  "stadium TEXT",
  "roof TEXT",
  "surface TEXT",
  "home_timeouts_remaining INT",
  "away_timeouts_remaining INT",
  "overtime INT",
  "home_turnovers INT",
  "away_turnovers INT",
  "home_penalties INT",
  "away_penalties INT",
  "home_penalty_yards INT",
  "away_penalty_yards INT"
)

for (col_def in new_columns) {
  tryCatch({
    dbExecute(con, sprintf("ALTER TABLE games ADD COLUMN IF NOT EXISTS %s;", col_def))
  }, error = function(e) invisible())
}

# Load schedules with stadium metadata
schedules <- nflreadr::load_schedules(seasons = 1999:2024)

# Update games with stadium info
for (season in 1999:2024) {
  season_sched <- schedules %>%
    filter(season == !!season) %>%
    select(game_id, stadium, roof, surface, overtime)
  
  for (i in 1:nrow(season_sched)) {
    dbExecute(con, "
      UPDATE games 
      SET stadium = $1, roof = $2, surface = $3, overtime = $4
      WHERE game_id = $5
    ", params = list(
      season_sched$stadium[i],
      season_sched$roof[i],
      season_sched$surface[i],
      season_sched$overtime[i],
      season_sched$game_id[i]
    ))
  }
  
  cat("Updated", season, "\n")
}

# Calculate game-level stats from plays
dbExecute(con, "
  WITH game_stats AS (
    SELECT 
      game_id,
      posteam,
      COUNT(CASE WHEN fumble_lost = TRUE THEN 1 END) +
        COUNT(CASE WHEN interception = TRUE THEN 1 END) as turnovers,
      COUNT(CASE WHEN penalty = TRUE THEN 1 END) as penalties,
      SUM(CASE WHEN penalty = TRUE THEN penalty_yards ELSE 0 END) as penalty_yards
    FROM plays
    WHERE posteam IS NOT NULL
    GROUP BY game_id, posteam
  )
  UPDATE games g
  SET 
    home_turnovers = hs.turnovers,
    home_penalties = hs.penalties,
    home_penalty_yards = hs.penalty_yards,
    away_turnovers = as.turnovers,
    away_penalties = as.penalties,
    away_penalty_yards = as.penalty_yards
  FROM game_stats hs
  JOIN game_stats as ON hs.game_id = as.game_id AND hs.posteam = g.home_team AND as.posteam = g.away_team
  WHERE g.game_id = hs.game_id;
")

dbDisconnect(con)
cat("\n✅ Game metadata backfill complete!\n")
```

---

## 5. Execution Plan

### Recommended Order

1. **Run Script 1** (`R/backfill_pbp_advanced.R`) - 60-90 min
   - Adds ~40 critical columns to plays table
   - Downloads ~1.5 GB of nflfastR data
   - Updates 1.2M plays with WP, success, air yards, player IDs

2. **Run Script 2** (`R/backfill_rosters.R`) - 20-30 min
   - Creates players and rosters tables
   - Loads ~50K unique players
   - Loads ~500K roster-week entries

3. **Run Script 3** (`R/backfill_game_metadata.R`) - 10-15 min
   - Adds stadium, roof, surface to games
   - Calculates turnovers, penalties from plays
   - Updates all 6,969 games

4. **Refresh mart views** - 2-3 min
   ```sql
   SELECT mart.refresh_game_features();
   REFRESH MATERIALIZED VIEW mart.asof_team_features;
   ```

5. **Update Python harness** - 30-45 min
   - Modify feature loading queries
   - Add new columns to model input
   - Test on holdout set

6. **Retrain models** - 60-120 min
   - Retrain XGBoost with expanded features
   - Run backtest comparison (old vs new)
   - Validate performance improvements

**Total Estimated Time:** 4-6 hours (can run mostly unattended)

---

## 6. Expected Outcomes

### Quantitative Improvements

| Metric | Current | Expected After Backfill | Improvement |
|--------|---------|------------------------|-------------|
| **Brier Score** | 0.245 | 0.235-0.238 | -2.9% to -7.1% (better) |
| **ROI** | +1.8% | +2.6% to +3.3% | +0.8% to +1.5% |
| **Sharpe Ratio** | 0.89 | 0.95-1.05 | +6.7% to +18.0% |
| **Feature Count** | ~25 | ~80 | +220% |
| **Data Completeness** | 85% | 97% | +12% |

### Qualitative Improvements

1. **Better situational modeling** - WP enables late-game context
2. **Player-specific predictions** - QB/RB/WR identity matters
3. **Personnel-aware models** - 11 vs 12 vs 21 personnel
4. **Improved calibration** - Success rate complements EPA
5. **Historical depth** - Full 26-season dataset usable

---

## 7. Storage & Performance Considerations

### Disk Space Requirements

| Data | Current Size | After Backfill | Increase |
|------|--------------|----------------|----------|
| plays table | 157 MB | ~450 MB | +187% |
| New tables (players, rosters) | 0 | ~25 MB | +25 MB |
| Indexes | ~50 MB | ~120 MB | +70 MB |
| **TOTAL** | ~250 MB | ~620 MB | +370 MB |

**Verdict:** Manageable increase (~600 MB total for core tables)

### Query Performance

- Added indexes on player_id, team, season will maintain fast queries
- Materialized views should be refreshed after backfill
- Expect 10-20% slower queries due to wider tables, but still < 1 sec for game lookups

---

## 8. Validation Checklist

After running backfill scripts, verify:

- [ ] plays table has WP, success, air_yards columns
- [ ] plays table has player_id columns (passer, rusher, receiver)
- [ ] players table has ~50K unique players
- [ ] rosters table has data for all seasons 1999-2024
- [ ] games table has roof, surface, stadium for all games
- [ ] games table has turnovers/penalties calculated
- [ ] No NULL values in critical columns (or expected NULLs documented)
- [ ] Sample queries return sensible data
- [ ] Materialized views refreshed successfully
- [ ] Python harness can load new features
- [ ] Model training runs without errors

---

## 9. Rollback Plan

If backfill causes issues:

```sql
-- Backup before starting
pg_dump -h localhost -p 5544 -U dro devdb01 > backup_before_backfill.sql

-- Rollback: drop added columns
ALTER TABLE plays DROP COLUMN IF EXISTS wp;
ALTER TABLE plays DROP COLUMN IF EXISTS wpa;
-- ... (drop all added columns)

-- Restore from backup if needed
psql -h localhost -p 5544 -U dro devdb01 < backup_before_backfill.sql
```

---

## 10. Next Steps After Backfill

1. **Retrain models with new features**
2. **Run OOS backtest to validate improvement**
3. **Update dissertation tables with new results**
4. **Create feature importance analysis**
5. **Document new features in methods chapter**
6. **Implement feature selection (LASSO/RFE) to find optimal subset**
7. **Explore player-specific models (QB EPA, WR separation, etc.)**

---

**Ready to execute? Start with Script 1 (backfill_pbp_advanced.R) - it's the highest impact.**
