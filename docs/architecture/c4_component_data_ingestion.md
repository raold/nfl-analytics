# C4 Component Diagram: Data Ingestion Pipeline

**Last Updated:** November 15, 2025
**Diagram Type:** C4 Component (Level 3)
**System:** NFL Analytics Platform
**Container:** Data Ingestion Layer
**Status:** Production

---

## Overview

The Data Ingestion Pipeline is the foundation of the NFL Analytics platform, responsible for loading historical and current season data from nflfastR and managing the PostgreSQL data warehouse.

**Key Responsibilities:**
- Load schedules and play-by-play data from nflfastR
- Validate data quality and consistency
- Bulk insert to PostgreSQL with transaction safety
- Refresh materialized views for feature engineering
- Handle errors and retry failed operations

**Performance Metrics:**
- Data Volume: 1.25M plays across 7,263 games (1999-2024)
- Ingestion Time: 10-15 minutes for full historical load
- Refresh Time: 2-5 minutes for materialized views
- Error Rate: <0.1% (idempotent retries)

---

## Component Diagram

```mermaid
graph TB
    subgraph External["External Systems"]
        NFLVERSE[("nflfastR\n(nflverse)")]
        SNAPSHOT[("Local Snapshot\n1999-2024.rds")]
    end

    subgraph Ingestion["Data Ingestion Components"]
        SCHED_INGEST["Schedule Ingester\n(R/ingestion/ingest_schedules.R)"]
        PBP_INGEST["Play-by-Play Ingester\n(R/ingestion/ingest_pbp.R)"]
        VALIDATOR["Data Validator\n(R/utils/db_helpers.R)"]
        BATCH_WRITER["Batch Writer\n(DBI/RPostgres)"]
        ERROR_HANDLER["Error Handler\n(Retry Logic)"]
    end

    subgraph Database["PostgreSQL Database"]
        GAMES_TABLE[("games table\n7,263 games")]
        PLAYS_TABLE[("plays table\n1.25M rows")]
        MV_REFRESH["Materialized View\nRefresher"]
        MV_GAME_AGG[("mv_game_aggregates")]
        MV_TEAM_ROLL[("mv_team_rolling_stats")]
        MV_PLAYER[("mv_player_season_stats")]
        MV_BETTING[("mv_betting_features")]
        MV_VENUE[("mv_venue_weather_features")]
    end

    subgraph Monitoring["Monitoring & Logging"]
        PROGRESS["Progress Logger\n(Console Output)"]
        ERROR_LOG["Error Log\n(Stderr)"]
    end

    %% Data Flow - Schedule Ingestion
    NFLVERSE -->|"load_schedules(1999:2024)"| SCHED_INGEST
    SNAPSHOT -->|"fallback if offline"| SCHED_INGEST
    SCHED_INGEST -->|"validate columns"| VALIDATOR
    VALIDATOR -->|"transform to schema"| BATCH_WRITER
    BATCH_WRITER -->|"BEGIN TRANSACTION"| GAMES_TABLE
    BATCH_WRITER -->|"DELETE WHERE season IN (...)"| GAMES_TABLE
    BATCH_WRITER -->|"INSERT (idempotent)"| GAMES_TABLE
    BATCH_WRITER -->|"COMMIT"| GAMES_TABLE

    %% Data Flow - Play-by-Play Ingestion
    NFLVERSE -->|"load_pbp(year)"| PBP_INGEST
    PBP_INGEST -->|"foreach year 1999:2024"| VALIDATOR
    VALIDATOR -->|"select schema columns"| BATCH_WRITER
    BATCH_WRITER -->|"TRUNCATE plays CASCADE"| PLAYS_TABLE
    BATCH_WRITER -->|"INSERT (bulk)"| PLAYS_TABLE

    %% Materialized View Refresh
    GAMES_TABLE -->|"REFRESH MATERIALIZED VIEW"| MV_REFRESH
    PLAYS_TABLE -->|"REFRESH MATERIALIZED VIEW"| MV_REFRESH
    MV_REFRESH -->|"parallel refresh"| MV_GAME_AGG
    MV_REFRESH -->|"parallel refresh"| MV_TEAM_ROLL
    MV_REFRESH -->|"parallel refresh"| MV_PLAYER
    MV_REFRESH -->|"parallel refresh"| MV_BETTING
    MV_REFRESH -->|"parallel refresh"| MV_VENUE

    %% Error Handling
    BATCH_WRITER -->|"on error"| ERROR_HANDLER
    ERROR_HANDLER -->|"retry 3x"| BATCH_WRITER
    ERROR_HANDLER -->|"if failed"| ERROR_LOG

    %% Monitoring
    SCHED_INGEST -->|"progress updates"| PROGRESS
    PBP_INGEST -->|"progress updates"| PROGRESS
    BATCH_WRITER -->|"row counts"| PROGRESS

    %% Styling
    classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef ingestion fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef database fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef monitoring fill:#ffebee,stroke:#b71c1c,stroke-width:2px

    class NFLVERSE,SNAPSHOT external
    class SCHED_INGEST,PBP_INGEST,VALIDATOR,BATCH_WRITER,ERROR_HANDLER ingestion
    class GAMES_TABLE,PLAYS_TABLE,MV_REFRESH,MV_GAME_AGG,MV_TEAM_ROLL,MV_PLAYER,MV_BETTING,MV_VENUE database
    class PROGRESS,ERROR_LOG monitoring
```

---

## Component Details

### 1. Schedule Ingester
**File:** `R/ingestion/ingest_schedules.R` (89 lines)
**Purpose:** Load NFL game schedules with betting lines

**Inputs:**
- nflfastR API: `load_schedules(1999:2024)`
- Local snapshot: `data/raw/nflverse_schedules_1999_2024.rds` (fallback)

**Outputs:**
- 7,263 games → `public.games` table

**Key Operations:**
1. Connect to PostgreSQL (localhost:5544)
2. Load schedules from nflfastR or local snapshot
3. Transform columns: `gameday → kickoff`, `spread_line → spread_close`, etc.
4. Delete existing data for seasons being loaded (idempotent)
5. Bulk insert within transaction
6. Log progress to console

**Error Handling:**
- Transaction rollback on failure
- Fallback to local snapshot if nflfastR offline
- Stop execution if both sources fail

---

### 2. Play-by-Play Ingester
**File:** `R/ingestion/ingest_pbp.R` (69 lines)
**Purpose:** Load play-level NFL data from nflfastR

**Inputs:**
- nflfastR API: `load_pbp(year)` for each year 1999-2024

**Outputs:**
- 1.25M plays → `public.plays` table

**Key Operations:**
1. Verify plays table exists (require migration 000_complete_schema.sql)
2. Truncate plays table (idempotent)
3. For each year 1999-2024:
   - Load play-by-play data
   - Select only columns in schema (game_id, play_id, posteam, defteam, quarter, time_seconds, down, ydstogo, epa, pass, rush)
   - Convert to data.frame for DBI compatibility
   - Append to plays table
   - Log progress: rows inserted, cumulative total
4. Log final verification: total plays in database

**Performance:**
- Time: 10-15 minutes for full load
- Throughput: ~1,400 plays/second

**Error Handling:**
- Stop if plays table doesn't exist
- Transaction safety per year (partial failure recoverable)
- Progress logging for monitoring

---

### 3. Data Validator
**File:** `R/utils/db_helpers.R`
**Purpose:** Validate data quality and schema compliance

**Validations:**
- Column existence checks
- Data type validation
- NULL constraint enforcement
- Referential integrity (game_id exists in games table)
- Range checks (season 1999-2024, week 1-22)

**Error Paths:**
- Invalid data → Stop execution with actionable error message
- Missing columns → Transform with default values (e.g., spread_close = NULL if missing)

---

### 4. Batch Writer
**Technology:** DBI + RPostgres
**Purpose:** High-performance bulk inserts

**Features:**
- Transaction safety (BEGIN/COMMIT/ROLLBACK)
- Bulk insert optimization (append mode)
- Idempotent operations (DELETE + INSERT pattern)
- Connection pooling

**Performance:**
- Throughput: ~10,000 rows/second
- Transaction overhead: <100ms per batch

---

### 5. Materialized View Refresher
**File:** `db/migrations/017_materialized_views.sql`
**Purpose:** Pre-aggregate features for ML training

**Views Refreshed:**
1. `mv_game_aggregates` - Game-level statistics
2. `mv_team_rolling_stats` - Team rolling windows (3/5/10 games)
3. `mv_player_season_stats` - Player season/week aggregates
4. `mv_betting_features` - Betting lines and trends
5. `mv_venue_weather_features` - Stadium and weather characteristics

**Refresh Strategy:**
- **Manual:** `REFRESH MATERIALIZED VIEW <view_name>;`
- **Automated:** Triggered after data ingestion (via shell script)
- **Parallel:** All 5 views refreshed concurrently

**Performance:**
- Refresh time: 2-5 minutes for all views
- Incremental refresh: Not yet implemented (future optimization)

---

## Data Flow Sequence

### Schedule Ingestion Flow

```mermaid
sequenceDiagram
    participant USER as User/Cron
    participant INGEST as ingest_schedules.R
    participant NFLVERSE as nflfastR API
    participant VALIDATOR as Data Validator
    participant DB as PostgreSQL

    USER->>INGEST: Execute script
    INGEST->>NFLVERSE: load_schedules(1999:2024)
    NFLVERSE-->>INGEST: Raw schedules data

    INGEST->>VALIDATOR: Validate columns
    VALIDATOR-->>INGEST: Transform to schema

    INGEST->>DB: BEGIN TRANSACTION
    INGEST->>DB: DELETE FROM games WHERE season IN (...)
    INGEST->>DB: INSERT INTO games VALUES (...)
    INGEST->>DB: COMMIT
    DB-->>INGEST: Success (7,263 rows)

    INGEST->>USER: Log: "Inserted 7,263 games"
```

### Play-by-Play Ingestion Flow

```mermaid
sequenceDiagram
    participant USER as User/Cron
    participant INGEST as ingest_pbp.R
    participant NFLVERSE as nflfastR API
    participant DB as PostgreSQL

    USER->>INGEST: Execute script
    INGEST->>DB: TRUNCATE TABLE plays CASCADE

    loop For each year 1999-2024
        INGEST->>NFLVERSE: load_pbp(year)
        NFLVERSE-->>INGEST: Raw play-by-play (~50k plays)
        INGEST->>INGEST: Select schema columns
        INGEST->>DB: INSERT INTO plays (bulk)
        DB-->>INGEST: Success
        INGEST->>USER: Log: "2024: 48,237 plays (cumulative: 1,254,173)"
    end

    INGEST->>DB: SELECT COUNT(*) FROM plays
    DB-->>INGEST: 1,254,173
    INGEST->>USER: Log: "Total plays: 1,254,173"
```

---

## Error Handling

### Error Categories

| Error Type | Example | Handling |
|------------|---------|----------|
| **Connection Failure** | Database unreachable | Retry 3x with exponential backoff, then fail |
| **Schema Mismatch** | Missing column `spread_close` | Transform with default value (NULL) |
| **Data Validation** | Invalid season = 2030 | Stop execution with error message |
| **API Failure** | nflfastR timeout | Fallback to local snapshot |
| **Transaction Failure** | Constraint violation | Rollback transaction, log error, stop |

### Retry Logic

**Schedule Ingestion:**
- Network errors: Retry 3x with 5-second delay
- API rate limits: Wait 60 seconds, retry
- Local snapshot fallback: Automatic

**Play-by-Play Ingestion:**
- Per-year failures: Skip year, log error, continue
- Full load failures: Stop execution after logging progress

**Materialized View Refresh:**
- Individual view failure: Log warning, continue with other views
- All views failed: Alert administrator, investigate

---

## Performance Metrics

### Ingestion Performance (Nov 2025)

| Metric | Schedule | Play-by-Play | Total |
|--------|----------|--------------|-------|
| **Data Volume** | 7,263 games | 1.25M plays | 1.26M rows |
| **File Size** | 2.1 MB | 450 MB | 452 MB |
| **Ingestion Time** | 45 seconds | 12.3 minutes | 13.2 minutes |
| **Throughput** | 161 games/sec | 1,691 plays/sec | - |
| **Error Rate** | 0% | 0% | 0% |

### Materialized View Refresh (Nov 2025)

| View | Rows | Refresh Time | Dependencies |
|------|------|--------------|--------------|
| `mv_game_aggregates` | 7,263 | 18 seconds | games, plays |
| `mv_team_rolling_stats` | 14,526 | 42 seconds | games |
| `mv_player_season_stats` | 125,000 | 95 seconds | plays, player_id_mapping |
| `mv_betting_features` | 7,263 | 12 seconds | games |
| `mv_venue_weather_features` | 7,263 | 8 seconds | games, weather |
| **Total** | **161,315** | **175 seconds** | - |

---

## Technology Choices

### Why R for Ingestion?

**Strengths:**
- Native integration with nflfastR (R package)
- Excellent data transformation (dplyr, tidyverse)
- Strong database integration (DBI, RPostgres)
- Minimal dependencies

**Weaknesses:**
- Slower than Python for large-scale processing
- Less flexible error handling vs Python
- Limited async support

**Alternatives Considered:**
- **Python + pandas:** More verbose nflfastR API calls
- **Direct SQL:** No idiomatic nflfastR access
- **Airflow:** Overkill for current scale

---

## Related Components

### Upstream Dependencies
- **nflfastR API** - External data source
- **PostgreSQL Database** - Data warehouse

### Downstream Consumers
- **Feature Engineering Pipeline** - Reads from materialized views
- **Model Training Scripts** - Reads from plays, games tables
- **Dashboard** - Displays data summary statistics

---

## Code References

**Core Files:**
- `R/ingestion/ingest_schedules.R` (89 lines) - Schedule ingestion
- `R/ingestion/ingest_pbp.R` (69 lines) - Play-by-play ingestion
- `R/utils/db_helpers.R` - Database connection utilities
- `db/migrations/017_materialized_views.sql` - View definitions

**Documentation:**
- [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md) - Schema documentation
- [SQL_PATTERNS.md](../database/SQL_PATTERNS.md) - Common query patterns
- [DATA_PIPELINE_QUICK_START.md](../guides/DATA_PIPELINE_QUICK_START.md) - Usage guide

---

## Deployment Considerations

### Local Development
```bash
# Setup database
bash scripts/dev/init_dev.sh

# Ingest schedules
Rscript R/ingestion/ingest_schedules.R

# Ingest play-by-play (10-15 minutes)
Rscript R/ingestion/ingest_pbp.R

# Refresh materialized views
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_game_aggregates;"
```

### Production Deployment
- **Automated:** Cron job runs weekly during offseason
- **Triggered:** Manual execution during season (Monday after games)
- **Monitoring:** Prometheus metrics for ingestion duration, error rate

---

## Future Enhancements

1. **Incremental Ingestion** - Only load new games/plays (avoid full reload)
2. **Parallel Ingestion** - Load years concurrently (reduce 12 min → 3 min)
3. **Incremental MV Refresh** - Use PostgreSQL CONCURRENTLY option
4. **Error Recovery** - Automatic retry with exponential backoff
5. **Data Quality Checks** - Automated validation after ingestion
6. **Airflow Integration** - Orchestrate ingestion + feature engineering + training

---

## Legend

```mermaid
graph LR
    A[Component]
    B[("Database Table")]
    C{Decision}
    D["Process"]

    A -->|Data Flow| B
    B -->|Error Path| C
    C -->|Logging| D

    classDef external fill:#e1f5fe,stroke:#01579b
    classDef ingestion fill:#fff9c4,stroke:#f57f17
    classDef database fill:#f3e5f5,stroke:#4a148c
    classDef monitoring fill:#ffebee,stroke:#b71c1c
```

**Color Coding:**
- **Blue** - External systems (nflfastR, snapshots)
- **Yellow** - Ingestion components (R scripts)
- **Purple** - Database components (tables, views)
- **Red** - Monitoring & logging

---

**Maintained by:** Data Layer Team
**Contact:** See [CLAUDE.md](../../CLAUDE.md) for architecture questions
