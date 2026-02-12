# Database Entity-Relationship Diagram (ERD)

**Last Updated:** November 15, 2025
**Diagram Type:** Entity-Relationship Diagram
**System:** NFL Analytics Platform
**Database:** PostgreSQL 14+ with TimescaleDB
**Status:** Production

---

## Overview

The NFL Analytics database consists of **56 tables across 5 schemas** storing 1.25M plays from 7,263 games (1999-2024). This ERD documents the critical relationships between tables, focusing on production features and model training workflows.

**Key Statistics:**
- **Total Tables:** 56 (public: 15, mart: 25, ml: 8, staging: 5, metadata: 3)
- **Total Columns:** 257 across core tables
- **Data Volume:** 1.25M plays, 7,263 games, 15,213 players
- **Materialized Views:** 5 (refresh required after data ingestion)
- **Time Series:** odds_history (TimescaleDB hypertable, 820K rows)

**Critical Schema Convention:**
- ⚠️ **NEVER assume columns exist** - Always check schema first
- ⚠️ **mart.player_game_stats has NO game_id column** (use `(season, week)` composite key)
- ⚠️ Use `kickoff` (not `game_date`), `spread_close` (not `spread_line`)
- ⚠️ Team abbreviations: LAR/LAC (not "LA")

---

## High-Level Schema Organization

```mermaid
graph TB
    subgraph Public["public schema (Raw Data)"]
        GAMES[("games\n7,263 rows")]
        PLAYS[("plays\n1.25M rows")]
        WEATHER[("weather\n~7K rows")]
        INJURIES[("injuries\n~50K rows")]
        ODDS_HISTORY[("odds_history\n820K rows\nTimescaleDB")]
    end

    subgraph Mart["mart schema (Derived/Materialized)"]
        GAME_SUMMARY[("game_summary\nMV")]
        PLAYER_GAME_STATS[("player_game_stats\n24,950 rows")]
        PLAYER_HIERARCHY[("player_hierarchy\n15,213 players")]
        TEAM_EPA[("team_epa")]
        GAME_FEATURES[("game_features_enhanced\n157 features")]
    end

    subgraph ML["ml schema (Model Registry)"]
        MODEL_METADATA[("model_metadata\n8 models")]
        TRAINING_RUNS[("training_runs")]
        PREDICTIONS[("predictions")]
    end

    subgraph Staging["staging schema (ETL)"]
        STAGING_GAMES[("staging_games")]
        STAGING_PLAYS[("staging_plays")]
    end

    subgraph Metadata["metadata schema (Admin)"]
        SCHEMA_VERSION[("schema_version")]
        REFRESH_LOG[("refresh_log")]
    end

    %% Relationships
    GAMES -->|game_id| PLAYS
    GAMES -->|game_id| WEATHER
    GAMES -->|game_id| ODDS_HISTORY
    GAMES -->|game_id| TEAM_EPA
    GAMES -->|game_id| GAME_SUMMARY

    PLAYER_HIERARCHY -->|player_id| PLAYER_GAME_STATS
    GAMES -.->|"(season,week)\nNO game_id!"| PLAYER_GAME_STATS

    GAME_SUMMARY -->|game_id| GAME_FEATURES
    TEAM_EPA -->|game_id| GAME_FEATURES

    PLAYER_GAME_STATS -->|features| MODEL_METADATA
    GAME_FEATURES -->|features| MODEL_METADATA
    MODEL_METADATA -->|model_id| PREDICTIONS
    MODEL_METADATA -->|model_id| TRAINING_RUNS

    %% Styling
    classDef public fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef mart fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef ml fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef staging fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    classDef metadata fill:#f3e5f5,stroke:#4a148c,stroke-width:2px

    class GAMES,PLAYS,WEATHER,INJURIES,ODDS_HISTORY public
    class GAME_SUMMARY,PLAYER_GAME_STATS,PLAYER_HIERARCHY,TEAM_EPA,GAME_FEATURES mart
    class MODEL_METADATA,TRAINING_RUNS,PREDICTIONS ml
    class STAGING_GAMES,STAGING_PLAYS staging
    class SCHEMA_VERSION,REFRESH_LOG metadata
```

---

## Detailed ERD: Core Tables

### Public Schema (Raw Data)

```mermaid
erDiagram
    games ||--o{ plays : "game_id"
    games ||--o| weather : "game_id"
    games ||--o{ odds_history : "game_id"
    games ||--o{ team_epa : "game_id"
    games ||--o{ injuries : "game_id"

    games {
        text game_id PK
        int season
        int week
        text home_team
        text away_team
        timestamptz kickoff
        real spread_close
        real total_close
        int home_score
        int away_score
        real home_moneyline
        real away_moneyline
        real home_spread_odds
        real away_spread_odds
        real over_odds
        real under_odds
    }

    plays {
        text game_id FK
        int play_id PK
        text posteam
        text defteam
        int quarter
        int time_seconds
        int down
        int ydstogo
        real epa
        int pass
        int rush
    }

    weather {
        text game_id PK
        int temperature
        int wind_speed
        text weather_description
        text roof_type
    }

    injuries {
        text game_id FK
        text player_id FK
        text injury_status
        date report_date
    }

    odds_history {
        timestamptz time PK
        text game_id FK
        text bookmaker
        real home_spread
        real away_spread
        real total
    }
```

### Mart Schema (Derived Views)

```mermaid
erDiagram
    player_hierarchy ||--o{ player_game_stats : "player_id"
    games ||..o{ player_game_stats : "season_week_composite"
    game_summary ||--o{ game_features_enhanced : "game_id"
    team_epa ||--o{ game_features_enhanced : "game_id"

    player_hierarchy {
        text player_id PK
        text player_name
        text position
        text current_team
        int entry_year
        int jersey_number
        text status
    }

    player_game_stats {
        text player_id FK
        int season FK
        int week FK
        text current_team
        int passing_yards
        int rushing_yards
        int receiving_yards
        int passing_tds
        int rushing_tds
        int receiving_tds
        int attempts
        int carries
        int targets
    }

    game_summary {
        text game_id PK
        int season
        int week
        text home_team
        text away_team
        int home_score
        int away_score
        real home_epa
        real away_epa
        real home_success_rate
        real away_success_rate
    }

    team_epa {
        text game_id FK
        text team_abbr FK
        real offense_epa
        real defense_epa
        real special_teams_epa
        real total_epa
    }

    game_features_enhanced {
        text game_id PK
        real spread_close
        real total_close
        real home_rest_days
        real away_rest_days
        real home_win_streak
        real away_win_streak
        real home_epa_last3
        real away_epa_last3
    }
```

### ML Schema (Model Registry)

```mermaid
erDiagram
    model_metadata ||--o{ training_runs : "model_id"
    model_metadata ||--o{ predictions : "model_id"

    model_metadata {
        int model_id PK
        text model_name
        text model_type
        text version
        text file_path
        jsonb hyperparameters
        timestamptz created_at
        text created_by
        text status
    }

    training_runs {
        int run_id PK
        int model_id FK
        int start_season
        int end_season
        jsonb metrics
        real mae
        real rmse
        real calibration_coverage
        timestamptz started_at
        timestamptz completed_at
        text status
    }

    predictions {
        int prediction_id PK
        int model_id FK
        text player_id
        int season
        int week
        text prop_type
        real predicted_value
        real predicted_std
        real actual_value
        timestamptz predicted_at
    }
```

---

## Critical Relationships

### 1. Games → Plays (One-to-Many)

**Relationship:** Each game has many plays (avg 130-150 plays/game)

**Join Pattern:**
```sql
SELECT g.season, g.week, g.home_team, g.away_team,
       COUNT(*) as play_count
FROM games g
JOIN plays p ON p.game_id = g.game_id
GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team;
```

**Volume:** 7,263 games × ~140 plays = 1.25M plays

---

### 2. Games → Player Game Stats (Many-to-Many via Composite Key)

**⚠️ CRITICAL MISTAKE (documented DOZENS of times):**

❌ **WRONG (game_id does NOT exist in player_game_stats):**
```sql
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id;  -- ERROR: Column does not exist
```

✅ **CORRECT (use season, week composite key):**
```sql
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team);
```

**Why No game_id?**
- `player_game_stats` is player-centric (not game-centric)
- One player plays in one game per week, but multiple players play in each game
- Composite key `(player_id, season, week)` is sufficient for unique constraint

**Reference:** [SQL_PATTERNS.md](docs/database/SQL_PATTERNS.md)

---

### 3. Player Hierarchy → Player Game Stats (One-to-Many)

**Relationship:** Each player has many game stat records

**Join Pattern:**
```sql
SELECT ph.player_name, ph.position, ph.current_team,
       pgs.season, pgs.week,
       pgs.passing_yards, pgs.rushing_yards, pgs.receiving_yards
FROM mart.player_hierarchy ph
JOIN mart.player_game_stats pgs ON pgs.player_id = ph.player_id
WHERE ph.player_id = '00-0033873';  -- Patrick Mahomes
```

**Volume:** 15,213 players, 24,950 player-game records (avg 1.6 games/player)

---

### 4. Games → Odds History (One-to-Many, Time Series)

**Relationship:** Each game has many odds snapshots (line movement tracking)

**Join Pattern:**
```sql
SELECT g.game_id, g.kickoff,
       oh.time, oh.bookmaker,
       oh.home_spread, oh.total
FROM games g
JOIN odds_history oh ON oh.game_id = g.game_id
WHERE g.game_id = '2024_10_KC_BUF'
ORDER BY oh.time;
```

**TimescaleDB Optimization:**
- `odds_history` is a hypertable (time-series optimized)
- Partitioned by `time` column for efficient range queries
- Compression enabled for historical data (>30 days old)

**Volume:** 820K odds snapshots across all games

---

### 5. Model Metadata → Predictions (One-to-Many)

**Relationship:** Each model generates many predictions

**Join Pattern:**
```sql
SELECT mm.model_name, mm.version,
       p.player_id, p.season, p.week,
       p.predicted_value, p.actual_value,
       ABS(p.predicted_value - p.actual_value) as mae
FROM ml.model_metadata mm
JOIN ml.predictions p ON p.model_id = mm.model_id
WHERE mm.model_name = 'state_space_v2';
```

**Use Case:** Evaluate model performance over time

---

## Materialized Views

**Purpose:** Precompute expensive aggregations for fast feature extraction

### 1. mart.game_summary

**Source:** `games` + `plays` + `team_epa`

**Refresh Frequency:** After each data ingestion

**Refresh Command:**
```sql
REFRESH MATERIALIZED VIEW mart.game_summary;
```

**Usage:** Feature engineering for game-level predictions

---

### 2. mart.game_features_enhanced

**Source:** `game_summary` + `team_epa` + `weather` + rolling windows

**Columns:** 157 features (spread, total, EPA, rest days, streaks, etc.)

**Refresh Time:** 2-5 minutes

**Usage:** Primary feature table for ensemble models

---

### 3. mart.player_game_stats

**Source:** Aggregated from `plays` (player-level stats)

**Refresh Frequency:** After play-by-play ingestion

**Usage:** Player props feature extraction

---

## Data Quality & Integrity

### Foreign Key Constraints

**Enforced Constraints:**
- `plays.game_id` → `games.game_id`
- `weather.game_id` → `games.game_id`
- `odds_history.game_id` → `games.game_id`
- `player_game_stats.player_id` → `player_hierarchy.player_id`

**Not Enforced (By Design):**
- `player_game_stats` → `games` (composite key `(season, week)`, not foreign key)

### Indexes

**Primary Indexes:**
- `games(game_id)` - Primary key
- `plays(game_id, play_id)` - Composite primary key
- `player_hierarchy(player_id)` - Primary key
- `player_game_stats(player_id, season, week)` - Composite primary key

**Performance Indexes (November 13, 2025):**
- `games(season, week)` - Composite (for player_game_stats joins)
- `games(kickoff)` - Temporal queries
- `games(home_team, away_team)` - Team lookups
- `plays(posteam, defteam)` - Team EPA aggregations
- `odds_history(game_id, time)` - Time-series queries (TimescaleDB optimized)
- `player_game_stats(season, week, current_team)` - Composite covering index

**Index Impact:** 40% backtest speedup (Nov 13, 2025)

---

## Connection Information

**PostgreSQL 14+ with TimescaleDB Extension**

```bash
Host:     localhost:5544
Database: devdb01
User:     dro
Password: sicillionbillions  # Dev only, use .env in production
```

**Connection String:**
```bash
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

**Python Connection:**
```python
import psycopg

conn = psycopg.connect(
    "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
)
```

**R Connection:**
```r
library(DBI)
library(RPostgres)

con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = "devdb01",
  host     = "localhost",
  user     = "dro",
  password = "sicillionbillions",
  port     = 5544
)
```

---

## Common Query Patterns

### 1. Get Player Historical Stats

```sql
SELECT pgs.player_id, ph.player_name, ph.position,
       pgs.season, pgs.week,
       pgs.passing_yards, pgs.rushing_yards, pgs.receiving_yards
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
WHERE ph.player_name = 'Patrick Mahomes'
  AND pgs.season >= 2020
ORDER BY pgs.season, pgs.week;
```

### 2. Get Game Features with Betting Lines

```sql
SELECT g.game_id, g.season, g.week, g.kickoff,
       g.home_team, g.away_team,
       g.spread_close, g.total_close,
       gf.home_epa_last3, gf.away_epa_last3,
       gf.home_rest_days, gf.away_rest_days
FROM games g
JOIN mart.game_features_enhanced gf ON gf.game_id = g.game_id
WHERE g.season = 2024 AND g.week = 10;
```

### 3. Get Player Stats with Opponent Defense

```sql
SELECT pgs.player_id, ph.player_name,
       pgs.season, pgs.week,
       pgs.passing_yards,
       te.defense_epa as opponent_defense_epa
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
LEFT JOIN mart.team_epa te ON
    te.game_id = g.game_id
    AND te.team_abbr = CASE
        WHEN g.home_team = ph.current_team THEN g.away_team
        ELSE g.home_team
    END
WHERE pgs.player_id = '00-0033873';
```

### 4. Track Line Movement

```sql
SELECT g.game_id, g.kickoff,
       oh.time, oh.bookmaker,
       oh.home_spread,
       LAG(oh.home_spread) OVER (PARTITION BY g.game_id ORDER BY oh.time) as prev_spread,
       oh.home_spread - LAG(oh.home_spread) OVER (PARTITION BY g.game_id ORDER BY oh.time) as line_move
FROM games g
JOIN odds_history oh ON oh.game_id = g.game_id
WHERE g.game_id = '2024_10_KC_BUF'
ORDER BY oh.time;
```

---

## Schema Evolution

### Version History

**v1.0 (1999-2020):** Basic games + plays schema
**v2.0 (2021):** Added mart schema, materialized views
**v2.5 (2023):** Added weather, injuries tables
**v3.0 (November 2, 2025):** Added ml schema (model registry)
**v3.1 (November 13, 2025):** Performance indexes (40% backtest speedup)
**v3.2 (November 15, 2025):** Documentation update (ERD, SQL patterns)

### Migration Process

**Schema Changes:**
1. Create migration SQL in `db/migrations/`
2. Test on staging database
3. Run migration on production
4. Update documentation (DATABASE_QUICK_REFERENCE.md, this ERD)
5. Refresh materialized views

**Backward Compatibility:**
- All schema changes must be backward compatible
- Deprecate columns (don't drop immediately)
- Use views for renamed columns

---

## Related Documentation

**Database Documentation:**
- [DATABASE_QUICK_REFERENCE.md](docs/database/DATABASE_QUICK_REFERENCE.md) - Quick lookup guide
- [DATABASE_SCHEMA_AUDIT.md](docs/database/DATABASE_SCHEMA_AUDIT.md) - Complete audit
- [SQL_PATTERNS.md](docs/database/SQL_PATTERNS.md) - Query patterns library

**Architecture Diagrams:**
- [c4_component_data_ingestion.md](c4_component_data_ingestion.md) - Data ingestion pipeline
- [c4_component_backtest_flow.md](c4_component_backtest_flow.md) - Backtest validation
- [data_flow_diagram.md](data_flow_diagram.md) - End-to-end data flow

**Performance Optimization:**
- [BATCH_PREDICTION_OPTIMIZATION.md](docs/performance/BATCH_PREDICTION_OPTIMIZATION.md)

---

## Legend

**Color Coding:**
- **Blue (#e1f5fe)** - public schema (raw data)
- **Yellow (#fff9c4)** - mart schema (derived/materialized)
- **Green (#c8e6c9)** - ml schema (model registry)
- **Orange (#ffccbc)** - staging schema (ETL)
- **Purple (#f3e5f5)** - metadata schema (admin)

**Relationship Notation:**
- **Solid line (||--o{)** - Foreign key constraint enforced
- **Dashed line (||..o{)** - Logical relationship (not foreign key)
- **PK** - Primary key
- **FK** - Foreign key

---

**Diagram Created:** November 15, 2025
**Maintained By:** Data Engineering Team
**Contact:** See [CLAUDE.md](CLAUDE.md) for questions
