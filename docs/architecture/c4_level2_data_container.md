# C4 Level 2: Data Container Detail

**Last Updated:** November 15, 2025
**Diagram Type:** C4 Container (Level 2)
**Container:** PostgreSQL 16 + TimescaleDB
**Status:** Production

---

## Overview

The PostgreSQL container is the central data warehouse, storing all NFL play-by-play data, player statistics, betting lines, and model registry metadata. It uses TimescaleDB extension for efficient time-series storage of betting odds.

**Container Specifications:**
- **Name:** nfl-analytics-pg-1
- **Image:** timescale/timescaledb:latest-pg16
- **Port:** 5544 (host) → 5432 (container)
- **Storage:** ~5GB (pgdata volume)
- **Version:** PostgreSQL 16.0 + TimescaleDB 2.15+

**Key Responsibilities:**
- Store 1.25M plays, 7,263 games, 15,213 players (2018-2024)
- Maintain 5 database schemas (public, mart, ml, staging, metadata)
- Provide materialized views for fast feature engineering
- Track betting odds history (TimescaleDB hypertable)
- Store model registry metadata (training runs, predictions)

---

## Data Container Diagram

```mermaid
graph TB
    subgraph External_Clients["External Clients"]
        R_INGEST["R Ingestion Scripts\n(data/ingest_pbp.R)\nLoad nflfastR data"]
        API["FastAPI\n(Feature extraction)"]
        STREAMLIT["Streamlit\n(Data queries)"]
        GRAFANA["Grafana\n(Direct queries)"]
        PG_EXPORTER["Postgres Exporter\n(Metrics)"]
    end

    subgraph PostgreSQL_Container["PostgreSQL 16 + TimescaleDB Container"]

        subgraph Public_Schema["public schema (Raw Data)"]
            GAMES["games\n(26 columns)\n7,263 rows\n\nPK: game_id\nIndexes: kickoff, season, week"]
            PLAYS["plays\n(67 columns)\n1.25M rows\n\nPK: play_id\nIndexes: game_id, posteam"]
            WEATHER["weather\n(10 columns)\nWeather conditions"]
            INJURIES["injuries\n(8 columns)\nInjury reports"]
            ODDS_HISTORY["odds_history\n(15 columns)\nTimescaleDB hypertable\n\nPartitioned by time\n60s granularity"]
        end

        subgraph Mart_Schema["mart schema (Derived Data)"]
            GAME_SUMMARY["game_summary\n(Materialized View)\n157 columns\n\nRolling EPA, scoring"]
            PLAYER_GAME_STATS["player_game_stats\n(Materialized View)\n45 columns\n\nKey: (season, week, player_id)\nNO game_id column!"]
            PLAYER_HIERARCHY["player_hierarchy\n(Table)\n12 columns\n\nPlayer metadata"]
            TEAM_EPA["team_epa\n(Materialized View)\nDefensive strength"]
            GAME_FEATURES["game_features_enhanced\n(Materialized View)\n87 columns\n\nML-ready features"]
        end

        subgraph ML_Schema["ml schema (Model Registry)"]
            MODEL_METADATA["model_metadata\n(Table)\n18 columns\n\nModel versions, hyperparameters"]
            TRAINING_RUNS["training_runs\n(Table)\n22 columns\n\nTraining history"]
            PREDICTIONS["predictions\n(Table)\n12 columns\n\nPrediction log"]
        end

        subgraph Staging_Schema["staging schema (ETL)"]
            STAGING_GAMES["staging_games\n(Temp table)\nData validation"]
            STAGING_PLAYS["staging_plays\n(Temp table)\nBatch inserts"]
        end

        subgraph Metadata_Schema["metadata schema (Admin)"]
            SCHEMA_VERSION["schema_version\n(Table)\nMigration tracking"]
            REFRESH_LOG["refresh_log\n(Table)\nMaterialized view refresh log"]
        end

        subgraph Indexes["Performance Indexes"]
            IDX_GAMES_KICKOFF["games_kickoff_idx\n(B-tree)"]
            IDX_PLAYS_GAME["plays_game_id_idx\n(B-tree)"]
            IDX_PGS_COMPOSITE["player_game_stats_composite_idx\n(season, week, player_id)"]
            IDX_ODDS_TIME["odds_history_time_idx\n(B-tree on TimescaleDB)"]
        end

        subgraph Extensions["PostgreSQL Extensions"]
            TIMESCALE["TimescaleDB\n(Time-series)"]
            PG_STAT_STATEMENTS["pg_stat_statements\n(Query tracking)"]
        end

    end

    subgraph File_System["Host File System"]
        PGDATA["pgdata/ Volume\n(Persistent storage)\n~5GB"]
    end

    %% External Clients to Schemas
    R_INGEST -->|"COPY, INSERT\nBatch transactions"| STAGING_GAMES
    R_INGEST -->|"COPY, INSERT\nBatch transactions"| STAGING_PLAYS
    STAGING_GAMES -->|"INSERT INTO\nAfter validation"| GAMES
    STAGING_PLAYS -->|"INSERT INTO\nAfter validation"| PLAYS
    R_INGEST -->|"REFRESH MATERIALIZED VIEW"| GAME_SUMMARY
    R_INGEST -->|"REFRESH MATERIALIZED VIEW"| PLAYER_GAME_STATS

    API -->|"SELECT\nFeature queries"| PLAYER_GAME_STATS
    API -->|"SELECT\nPlayer metadata"| PLAYER_HIERARCHY
    API -->|"SELECT\nGame details"| GAMES
    API -->|"INSERT\nLog predictions"| PREDICTIONS

    STREAMLIT -->|"SELECT\nData exploration"| GAMES
    STREAMLIT -->|"SELECT\nPlayer stats"| PLAYER_GAME_STATS
    STREAMLIT -->|"SELECT\nModel registry"| MODEL_METADATA

    GRAFANA -->|"SELECT\nDirect SQL queries"| GAMES
    GRAFANA -->|"SELECT\nSystem tables"| PG_STAT_STATEMENTS

    PG_EXPORTER -->|"SELECT\nMetrics queries"| PG_STAT_STATEMENTS

    %% Schema Relationships
    GAMES -.->|"1:N\ngame_id FK"| PLAYS
    GAMES -.->|"1:N\ngame_id FK"| WEATHER
    GAMES -.->|"1:N\nTimescaleDB"| ODDS_HISTORY
    PLAYER_HIERARCHY -.->|"1:N\nplayer_id FK"| PLAYER_GAME_STATS
    GAMES -.->|"(season, week)\nComposite join"| PLAYER_GAME_STATS
    MODEL_METADATA -.->|"1:N\nmodel_id FK"| TRAINING_RUNS
    MODEL_METADATA -.->|"1:N\nmodel_id FK"| PREDICTIONS

    %% Indexes
    GAMES -.-> IDX_GAMES_KICKOFF
    PLAYS -.-> IDX_PLAYS_GAME
    PLAYER_GAME_STATS -.-> IDX_PGS_COMPOSITE
    ODDS_HISTORY -.-> IDX_ODDS_TIME

    %% Extensions
    ODDS_HISTORY -.->|"Hypertable"| TIMESCALE
    PG_EXPORTER -.->|"Query stats"| PG_STAT_STATEMENTS

    %% Persistence
    GAMES -.->|"Data files"| PGDATA
    PLAYS -.->|"Data files"| PGDATA
    GAME_SUMMARY -.->|"Materialized"| PGDATA

    %% Styling
    classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef publicSchema fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef martSchema fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef mlSchema fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef stagingSchema fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    classDef metadataSchema fill:#d1c4e9,stroke:#311b92,stroke-width:2px
    classDef indexes fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    classDef extensions fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef volume fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px

    class R_INGEST,API,STREAMLIT,GRAFANA,PG_EXPORTER external
    class GAMES,PLAYS,WEATHER,INJURIES,ODDS_HISTORY publicSchema
    class GAME_SUMMARY,PLAYER_GAME_STATS,PLAYER_HIERARCHY,TEAM_EPA,GAME_FEATURES martSchema
    class MODEL_METADATA,TRAINING_RUNS,PREDICTIONS mlSchema
    class STAGING_GAMES,STAGING_PLAYS stagingSchema
    class SCHEMA_VERSION,REFRESH_LOG metadataSchema
    class IDX_GAMES_KICKOFF,IDX_PLAYS_GAME,IDX_PGS_COMPOSITE,IDX_ODDS_TIME indexes
    class TIMESCALE,PG_STAT_STATEMENTS extensions
    class PGDATA volume
```

---

## Schema Organization

### public schema (Raw Data)

**Purpose:** Store raw NFL data from nflfastR and betting lines from The Odds API

**Key Tables:**

#### 1. games (7,263 rows)

**Columns (26 total):**
```sql
CREATE TABLE games (
    game_id VARCHAR PRIMARY KEY,           -- "2024_10_KC_LV"
    season INTEGER NOT NULL,               -- 2024
    week INTEGER NOT NULL,                 -- 10
    kickoff DATE NOT NULL,                 -- "2024-11-10" (NOT game_date!)
    home_team VARCHAR(3) NOT NULL,         -- "KC"
    away_team VARCHAR(3) NOT NULL,         -- "LV"
    home_score INTEGER,                    -- 31
    away_score INTEGER,                    -- 17
    spread_close NUMERIC(4, 1),            -- -9.5 (NOT spread_line!)
    total_close NUMERIC(4, 1),             -- 42.5 (NOT total_line!)
    -- ... 16 more columns
);

CREATE INDEX games_kickoff_idx ON games (kickoff);
CREATE INDEX games_season_week_idx ON games (season, week);
```

**Critical Columns:**
- **kickoff** (NOT game_date) - Date of game
- **spread_close** (NOT spread_line) - Closing spread
- **total_close** (NOT total_line) - Closing total
- **LAR / LAC** (NOT "LA") - Los Angeles teams

**Common Mistake:**
```sql
-- ❌ WRONG
SELECT * FROM games WHERE game_date = '2024-11-10';  -- Column doesn't exist

-- ✅ CORRECT
SELECT * FROM games WHERE kickoff = '2024-11-10';
```

---

#### 2. plays (1.25M rows)

**Columns (67 total):**
```sql
CREATE TABLE plays (
    play_id VARCHAR PRIMARY KEY,           -- "2024_10_KC_LV_1234"
    game_id VARCHAR NOT NULL REFERENCES games(game_id),
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    posteam VARCHAR(3),                    -- Team in possession
    defteam VARCHAR(3),                    -- Defensive team
    down INTEGER,                          -- 1, 2, 3, 4
    ydstogo INTEGER,                       -- Yards to first down
    yards_gained NUMERIC(4, 1),            -- Play result
    epa NUMERIC(6, 4),                     -- Expected Points Added
    -- ... 57 more columns
);

CREATE INDEX plays_game_id_idx ON plays (game_id);
CREATE INDEX plays_posteam_idx ON plays (posteam);
```

**Data Volume:**
- **Total Rows:** 1,254,173 (as of Nov 2025)
- **Seasons:** 2018-2024 (7 seasons)
- **Avg Plays per Game:** ~173 plays/game

---

#### 3. odds_history (TimescaleDB Hypertable)

**Columns (15 total):**
```sql
CREATE TABLE odds_history (
    id SERIAL,
    game_id VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,        -- Poll time (60s intervals)
    sportsbook VARCHAR(50),                -- "Pinnacle", "DraftKings", "FanDuel"
    spread NUMERIC(4, 1),                  -- Current spread
    total NUMERIC(4, 1),                   -- Current total
    moneyline_home INTEGER,                -- Home moneyline odds
    moneyline_away INTEGER,                -- Away moneyline odds
    -- ... 7 more columns
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('odds_history', 'timestamp');

CREATE INDEX odds_history_game_id_idx ON odds_history (game_id, timestamp DESC);
```

**Purpose:** Track line movements for value betting
**Polling Frequency:** 60-second intervals during game week
**Data Volume:** ~1,000 polls/week × 52 weeks = ~50,000 rows/year

---

### mart schema (Derived Data)

**Purpose:** Pre-computed aggregations and materialized views for fast feature engineering

**Key Tables:**

#### 1. game_summary (Materialized View)

**Purpose:** Rolling EPA, scoring trends, team performance
**Columns:** 157 (comprehensive game-level statistics)
**Refresh Frequency:** After each data ingestion (weekly)

**Sample Columns:**
```sql
CREATE MATERIALIZED VIEW mart.game_summary AS
SELECT
    game_id,
    season,
    week,
    home_team,
    away_team,
    home_score,
    away_score,
    -- Rolling EPA (last 3, 5 games)
    AVG(home_total_epa) OVER (PARTITION BY home_team ORDER BY kickoff ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS home_epa_l3,
    AVG(away_total_epa) OVER (PARTITION BY away_team ORDER BY kickoff ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS away_epa_l5,
    -- ... 150+ more columns
FROM public.games
JOIN public.plays USING (game_id);

CREATE UNIQUE INDEX game_summary_game_id_idx ON mart.game_summary (game_id);
```

**Refresh Command:**
```sql
REFRESH MATERIALIZED VIEW mart.game_summary;
```

---

#### 2. player_game_stats (Materialized View)

**Purpose:** Player-level statistics per game (for feature engineering)
**Columns:** 45 (player performance metrics)
**Key:** (season, week, player_id) - **NO game_id column!**

**⚠️ CRITICAL SCHEMA PATTERN:**
```sql
CREATE MATERIALIZED VIEW mart.player_game_stats AS
SELECT
    pgs.season,
    pgs.week,
    pgs.player_id,
    ph.current_team,                      -- Team affiliation (NOT in games table)
    ph.position,                          -- QB, RB, WR, TE
    SUM(CASE WHEN p.pass_attempt = 1 THEN p.yards_gained ELSE 0 END) AS passing_yards,
    SUM(CASE WHEN p.rush_attempt = 1 THEN p.yards_gained ELSE 0 END) AS rushing_yards,
    -- ... 40 more columns
FROM public.plays p
JOIN mart.player_hierarchy ph ON p.passer_player_id = ph.player_id
GROUP BY pgs.season, pgs.week, pgs.player_id, ph.current_team, ph.position;

CREATE UNIQUE INDEX player_game_stats_composite_idx ON mart.player_game_stats (season, week, player_id);
```

**CRITICAL JOIN PATTERN:**
```sql
-- ❌ WRONG - game_id doesn't exist in player_game_stats
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id;  -- ERROR!

-- ✅ CORRECT - Join via (season, week) composite key
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.player_id = '00-0019596';
```

**See:** [docs/database/SQL_PATTERNS.md](../database/SQL_PATTERNS.md) for complete pattern library

---

#### 3. player_hierarchy (Table)

**Purpose:** Player metadata (team affiliation, position, status)
**Columns:** 12

```sql
CREATE TABLE mart.player_hierarchy (
    player_id VARCHAR PRIMARY KEY,         -- "00-0033873"
    player_name VARCHAR(100),              -- "Patrick Mahomes"
    current_team VARCHAR(3),               -- "KC"
    position VARCHAR(3),                   -- "QB"
    status VARCHAR(20),                    -- "ACT" (active)
    -- ... 7 more columns
);
```

**Usage:** JOIN to player_game_stats to get team affiliation

---

### ml schema (Model Registry)

**Purpose:** Track model versions, training runs, and predictions

**Key Tables:**

#### 1. model_metadata (Table)

**Purpose:** Model registry (versions, hyperparameters, performance)
**Columns:** 18

```sql
CREATE TABLE ml.model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,      -- "state_space_v2"
    model_type VARCHAR(50),                -- "bayesian", "xgboost", "ensemble"
    version VARCHAR(20),                   -- "v2"
    file_path VARCHAR(255),                -- "models/state_space_v2.rds"
    training_window_start DATE,            -- "2018-01-01"
    training_window_end DATE,              -- "2024-12-31"
    hyperparameters JSONB,                 -- {"chains": 4, "iter": 2000}
    performance_metrics JSONB,             -- {"mae": 47.77, "r2": 0.65}
    created_at TIMESTAMP DEFAULT NOW(),
    -- ... 8 more columns
);
```

**Example Row:**
```json
{
  "id": 11,
  "model_name": "state_space_v2",
  "model_type": "bayesian",
  "version": "v2",
  "file_path": "models/state_space_v2.rds",
  "hyperparameters": {
    "chains": 4,
    "iter": 2000,
    "warmup": 1000,
    "adapt_delta": 0.95
  },
  "performance_metrics": {
    "mae": 47.77,
    "r2": 0.65,
    "calibration_coverage": 0.90
  }
}
```

---

#### 2. training_runs (Table)

**Purpose:** Training history (logs, convergence, errors)
**Columns:** 22

```sql
CREATE TABLE ml.training_runs (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml.model_metadata(id),
    run_timestamp TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20),                    -- "success", "failed", "running"
    convergence_metrics JSONB,             -- {"rhat": 1.01, "ess_bulk": 800}
    training_time_seconds NUMERIC(10, 2),  -- 1800.5 (30 minutes)
    error_message TEXT,                    -- NULL if success
    -- ... 15 more columns
);
```

---

#### 3. predictions (Table)

**Purpose:** Log all predictions (for audit and backtesting)
**Columns:** 12

```sql
CREATE TABLE ml.predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml.model_metadata(id),
    game_id VARCHAR NOT NULL,
    player_id VARCHAR NOT NULL,
    stat_type VARCHAR(50),                 -- "passing_yards"
    prediction NUMERIC(6, 2),              -- 265.3
    uncertainty NUMERIC(6, 2),             -- 52.1
    actual NUMERIC(6, 2),                  -- 278.0 (NULL if future game)
    created_at TIMESTAMP DEFAULT NOW(),
    -- ... 3 more columns
);
```

---

## Performance Indexes

**Purpose:** Speed up common queries (40% backtest speedup)

**Index Strategy:**

| Index Name | Table | Columns | Type | Purpose |
|------------|-------|---------|------|---------|
| **games_kickoff_idx** | games | kickoff | B-tree | Filter by date |
| **games_season_week_idx** | games | (season, week) | B-tree | Join to player_game_stats |
| **plays_game_id_idx** | plays | game_id | B-tree | Filter by game |
| **player_game_stats_composite_idx** | player_game_stats | (season, week, player_id) | B-tree | Composite key joins |
| **odds_history_time_idx** | odds_history | (timestamp DESC) | B-tree | Time-series queries |
| **model_metadata_name_idx** | model_metadata | model_name | B-tree | Model lookup |

**Creating Indexes (Non-Blocking):**
```sql
-- Create index without locking table (PostgreSQL 11+)
CREATE INDEX CONCURRENTLY games_kickoff_idx ON games (kickoff);
```

**Covering Indexes (Include Columns):**
```sql
-- Include frequently accessed columns (PostgreSQL 11+)
CREATE INDEX player_game_stats_covering_idx ON mart.player_game_stats (player_id)
INCLUDE (passing_yards, rushing_yards, receiving_yards);
```

---

## PostgreSQL Extensions

### 1. TimescaleDB (Time-Series Data)

**Purpose:** Efficient storage and querying of betting odds history

**Installation:**
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

**Hypertable Creation:**
```sql
-- Convert odds_history to hypertable (partitioned by time)
SELECT create_hypertable('odds_history', 'timestamp', chunk_time_interval => INTERVAL '1 day');
```

**Benefits:**
- **Compression:** 10-20x size reduction (compressed chunks)
- **Fast Queries:** Time-based partitioning (query only relevant chunks)
- **Retention Policies:** Auto-delete old data (e.g., keep 90 days only)

**Example Query:**
```sql
-- Fetch line movements for specific game (fast, only queries relevant chunks)
SELECT timestamp, sportsbook, spread, total
FROM odds_history
WHERE game_id = '2024_10_KC_LV'
  AND timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;
```

---

### 2. pg_stat_statements (Query Tracking)

**Purpose:** Track query performance (identify slow queries)

**Installation:**
```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

**Configuration (postgresql.conf):**
```
shared_preload_libraries = 'timescaledb,pg_stat_statements'
pg_stat_statements.track = all
```

**Querying Stats:**
```sql
-- Find slowest queries (by total time)
SELECT
    query,
    calls,
    total_exec_time / 1000 AS total_seconds,
    mean_exec_time / 1000 AS mean_seconds,
    stddev_exec_time / 1000 AS stddev_seconds
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

**Use Cases:**
- Identify missing indexes (slow sequential scans)
- Detect N+1 query problems
- Monitor query performance over time (feed to Prometheus)

---

## Data Ingestion Pipeline

**Workflow:**

```
1. nflfastR API
   ↓ (HTTP GET, JSON response)
2. R Ingestion Scripts (data/ingest_pbp.R)
   ↓ (Validate schema, deduplicate)
3. staging_plays / staging_games (Temporary tables)
   ↓ (Batch INSERT, transaction safety)
4. public.plays / public.games (Raw data tables)
   ↓ (REFRESH MATERIALIZED VIEW)
5. mart.* (Materialized views for features)
```

**R Ingestion Script Example:**
```r
# Load data from nflfastR
pbp <- nflfastR::load_pbp(seasons = 2024)

# Validate schema
stopifnot(all(c("game_id", "season", "week") %in% colnames(pbp)))

# Insert into staging table (batch transaction)
DBI::dbWriteTable(conn, "staging_plays", pbp, overwrite = TRUE)

# Deduplicate and insert into production table
DBI::dbExecute(conn, "
  INSERT INTO plays
  SELECT DISTINCT ON (play_id) *
  FROM staging_plays
  ON CONFLICT (play_id) DO NOTHING
")

# Refresh materialized views
DBI::dbExecute(conn, "REFRESH MATERIALIZED VIEW mart.game_summary")
DBI::dbExecute(conn, "REFRESH MATERIALIZED VIEW mart.player_game_stats")
```

**Performance:**
- **Ingestion Time:** 13.2 minutes (full data load, 1.25M plays)
- **Throughput:** 1,691 plays/second
- **Error Rate:** <0.1%

---

## Query Patterns

### Common Queries (Optimized)

**1. Get Player Rolling Averages:**
```sql
-- Fetch last 5 games for player
WITH player_stats AS (
    SELECT
        season,
        week,
        passing_yards,
        ROW_NUMBER() OVER (ORDER BY season DESC, week DESC) AS rn
    FROM mart.player_game_stats
    WHERE player_id = '00-0033873'  -- Patrick Mahomes
)
SELECT
    AVG(passing_yards) AS avg_passing_yards_l5
FROM player_stats
WHERE rn <= 5;
```

**Query Plan:** Index scan on player_game_stats_composite_idx → Fast (<5ms)

---

**2. Get Opponent Defensive Strength:**
```sql
-- Fetch opponent EPA allowed (last 3 games)
SELECT
    g.away_team AS opponent,
    AVG(te.epa_allowed) OVER (
        PARTITION BY te.team
        ORDER BY g.kickoff
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS opponent_epa_allowed_l3
FROM games g
JOIN mart.team_epa te ON te.team = g.away_team
WHERE g.home_team = 'KC'
  AND g.season = 2024
  AND g.week = 10;
```

---

**3. Join Player Stats to Game Details (CORRECT PATTERN):**
```sql
-- ✅ CORRECT - Join via (season, week) composite key + team filter
SELECT
    pgs.player_id,
    pgs.passing_yards,
    g.spread_close,
    g.total_close,
    g.home_team,
    g.away_team
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.player_id = '00-0033873'
  AND pgs.season = 2024
  AND pgs.week = 10;
```

**Why This Pattern?**
- `player_game_stats` does NOT have `game_id` column
- Must join via `(season, week)` composite key
- Must filter to player's team using `player_hierarchy.current_team`

**See:** [docs/database/SQL_PATTERNS.md](../database/SQL_PATTERNS.md) for 20+ more patterns

---

## Backup & Recovery

### Backup Strategy

**pg_dump (Daily):**
```bash
# Full database backup (schema + data)
pg_dump postgresql://dro:sicillionbillions@localhost:5544/devdb01 \  # pragma: allowlist secret
  --format=custom \
  --compress=9 \
  --file=backup_$(date +%Y%m%d).dump

# Compress to ~500MB (from ~5GB)
```

**Backup Schedule:**
- **Frequency:** Daily at 3 AM (cron job)
- **Retention:** 30 days
- **Storage:** Git LFS (~500MB per backup)

**Restore Procedure:**
```bash
# Drop existing database (WARNING: destructive)
dropdb devdb01

# Create new database
createdb devdb01

# Restore from backup
pg_restore --dbname=devdb01 \
  --verbose \
  --no-owner \
  --jobs=4 \
  backup_20251115.dump

# Refresh materialized views
psql devdb01 -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
psql devdb01 -c "REFRESH MATERIALIZED VIEW mart.player_game_stats;"
```

**Recovery Time Objective (RTO):** 1-2 hours (restore 5GB database)
**Recovery Point Objective (RPO):** 24 hours (daily backups)

---

### Continuous Archiving (Future)

**WAL Archiving (PostgreSQL):**
```
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /mnt/backup/wal_archive/%f'
```

**Benefits:**
- Point-in-time recovery (restore to any timestamp)
- RPO: <1 minute (near-zero data loss)
- Continuous backup (no scheduled downtime)

**Status:** Not yet implemented (planned for production)

---

## Performance Tuning

### Configuration (postgresql.conf)

**Memory Settings:**
```
shared_buffers = 2GB              # 25% of RAM (8GB total)
effective_cache_size = 6GB        # 75% of RAM
work_mem = 64MB                   # Per-query memory
maintenance_work_mem = 512MB      # For VACUUM, CREATE INDEX
```

**Connection Settings:**
```
max_connections = 100             # Max concurrent connections
shared_preload_libraries = 'timescaledb,pg_stat_statements'
```

**Query Tuning:**
```
random_page_cost = 1.1            # SSD (lower than default 4.0)
effective_io_concurrency = 200    # SSD concurrency
```

---

### Vacuum & Analyze

**Auto-Vacuum (Enabled):**
```
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
```

**Manual Vacuum (Weekly):**
```sql
-- Full vacuum (reclaim disk space)
VACUUM FULL ANALYZE;

-- Analyze only (update statistics, no disk reclaim)
ANALYZE;
```

---

### Query Optimization

**EXPLAIN ANALYZE:**
```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT
    pgs.player_id,
    pgs.passing_yards
FROM mart.player_game_stats pgs
WHERE pgs.season = 2024 AND pgs.week = 10;
```

**Output:**
```
Index Scan using player_game_stats_composite_idx on player_game_stats pgs
  (cost=0.42..8.45 rows=1 width=12) (actual time=0.015..0.016 rows=1 loops=1)
  Index Cond: ((season = 2024) AND (week = 10))
  Buffers: shared hit=4
Planning Time: 0.125 ms
Execution Time: 0.042 ms
```

**Optimization Goals:**
- Execution Time < 10ms (simple queries)
- Execution Time < 100ms (complex joins)
- Buffers: Prefer shared hit (cache) over shared read (disk)

---

## Monitoring

### Prometheus Metrics (via Postgres Exporter)

**Metrics Exported:**

| Metric | Type | Description |
|--------|------|-------------|
| **pg_up** | Gauge | Database reachable (1=up, 0=down) |
| **pg_stat_database_numbackends** | Gauge | Active connections |
| **pg_stat_database_xact_commit** | Counter | Committed transactions |
| **pg_stat_database_xact_rollback** | Counter | Rolled back transactions |
| **pg_stat_database_blks_hit** | Counter | Cache hits |
| **pg_stat_database_blks_read** | Counter | Disk reads |
| **pg_stat_database_tup_returned** | Counter | Rows returned |
| **pg_stat_statements_total_time_seconds** | Counter | Query execution time |

**Cache Hit Ratio:**
```
cache_hit_ratio = blks_hit / (blks_hit + blks_read)
```

**Target:** >95% (currently: 97-99% with proper indexes)

---

### Grafana Dashboards

**Database Metrics Panel:**
- **Connections:** Current, max, idle
- **Queries per Second:** SELECT, INSERT, UPDATE, DELETE
- **Transaction Rate:** Commits, rollbacks
- **Cache Hit Rate:** Shared buffers, table cache
- **Table Sizes:** games, plays, player_game_stats
- **Slow Queries:** Queries taking >100ms (from pg_stat_statements)

**Alerts:**
- Connections >80% max capacity → Warning
- Cache hit rate <90% → Warning
- Disk usage >80% → Critical
- Slow query >1s → Warning

---

## Related Documentation

**Schema Documentation:**
- [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md) - Quick lookup reference
- [DATABASE_SCHEMA_AUDIT.md](../database/DATABASE_SCHEMA_AUDIT.md) - Full schema audit
- [SQL_PATTERNS.md](../database/SQL_PATTERNS.md) - Query pattern library

**Container Documentation:**
- [c4_level2_container_overview.md](c4_level2_container_overview.md) - Overall container architecture
- [c4_level2_api_container.md](c4_level2_api_container.md) - FastAPI internal structure
- [c4_level2_monitoring_container.md](c4_level2_monitoring_container.md) - Prometheus/Grafana stack

**Component Documentation:**
- [c4_component_data_ingestion.md](c4_component_data_ingestion.md) - Data ingestion pipeline
- [database_erd.md](database_erd.md) - Entity-relationship diagram

---

## Legend

**Color Coding:**
- **Blue (#e1f5fe)** - External clients
- **Yellow (#fff9c4)** - public schema (raw data)
- **Green (#c8e6c9)** - mart schema (derived data)
- **Purple (#f3e5f5)** - ml schema (model registry)
- **Orange (#ffccbc)** - staging schema (ETL)
- **Light Purple (#d1c4e9)** - metadata schema (admin)
- **Cyan (#b3e5fc)** - Performance indexes
- **Red (#ffebee)** - PostgreSQL extensions
- **Dark Green (#c8e6c9)** - Persistent volume

**Edge Styles:**
- **Solid arrow (→)** - Data flow (INSERT, SELECT)
- **Dashed arrow (-.→)** - Relationships (FK, composite join)

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-15 | 1.0 | Initial data container diagram created |

---

**Diagram Created:** November 15, 2025
**Maintained By:** Architecture Team
**Contact:** See [CLAUDE.md](../../CLAUDE.md) for questions
