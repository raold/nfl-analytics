# Enterprise Project Restructuring Plan

**Date**: October 4, 2025  
**Purpose**: Reorganize project for production-grade data pipelines with enterprise best practices

---

## Current Issues

1. **Scripts scattered** across `/data`, `/scripts`, `/R`, root directory
2. **Documentation mixed** with root-level files (many MD files)
3. **Data files mixed** with ingestion scripts in `/data`
4. **No clear ETL structure** for data validation, deduplication, monitoring
5. **Manual ingestion** without scheduling, error handling, or rollback
6. **No data contracts** or schema validation between sources and database

---

## Proposed New Structure

```
nfl-analytics/
├── README.md                           # Main project documentation
├── pyproject.toml / setup.py           # Python package config
├── requirements.txt / requirements-dev.txt
├── pytest.ini, .gitignore, etc.        # Config files (root)
│
├── docs/                               # All documentation
│   ├── README.md                       # Documentation index
│   ├── setup/                          # Setup guides
│   │   ├── database.md
│   │   ├── development.md
│   │   └── production.md
│   ├── architecture/                   # System design
│   │   ├── data_pipeline.md
│   │   ├── database_schema.md
│   │   └── feature_engineering.md
│   ├── operations/                     # Ops guides
│   │   ├── deployment.md
│   │   ├── monitoring.md
│   │   └── troubleshooting.md
│   ├── reports/                        # Analysis reports
│   │   ├── 2025_data_ingestion.md
│   │   ├── database_audit.md
│   │   ├── backfill_results.md
│   │   └── feature_engineering.md
│   └── agent_context/                  # AI agent guidelines
│       ├── AGENTS.md
│       ├── CLAUDE.md
│       └── GEMINI.md
│
├── etl/                                # Data ingestion & ETL pipelines
│   ├── __init__.py
│   ├── config/                         # ETL configuration
│   │   ├── sources.yaml                # Data source configs
│   │   ├── schemas.yaml                # Expected schemas
│   │   └── validation_rules.yaml       # Data quality rules
│   ├── extract/                        # Data extraction
│   │   ├── __init__.py
│   │   ├── nflverse.py                 # nflreadr extraction
│   │   ├── odds_api.py                 # The Odds API
│   │   ├── weather.py                  # Meteostat
│   │   └── base.py                     # Base extractor class
│   ├── transform/                      # Data transformation
│   │   ├── __init__.py
│   │   ├── schedules.py
│   │   ├── plays.py
│   │   ├── rosters.py
│   │   ├── odds.py
│   │   └── weather.py
│   ├── load/                           # Database loading
│   │   ├── __init__.py
│   │   ├── games.py
│   │   ├── plays.py
│   │   ├── players.py
│   │   └── base.py                     # Base loader with upsert logic
│   ├── validate/                       # Data validation
│   │   ├── __init__.py
│   │   ├── schemas.py                  # Schema validation
│   │   ├── quality.py                  # Data quality checks
│   │   ├── referential.py              # FK integrity
│   │   └── deduplication.py            # Duplicate detection
│   ├── pipelines/                      # End-to-end pipelines
│   │   ├── __init__.py
│   │   ├── daily.py                    # Daily refresh
│   │   ├── weekly.py                   # Weekly full refresh
│   │   ├── backfill.py                 # Historical backfill
│   │   └── realtime.py                 # Real-time updates
│   └── monitoring/                     # Pipeline monitoring
│       ├── __init__.py
│       ├── metrics.py                  # Pipeline metrics
│       ├── alerts.py                   # Error alerting
│       └── logging.py                  # Structured logging
│
├── py/                                 # Core Python package
│   ├── __init__.py
│   ├── features/                       # Feature engineering
│   │   ├── __init__.py
│   │   ├── asof_features_enhanced.py
│   │   ├── base.py
│   │   └── generators/
│   ├── models/                         # ML models
│   │   ├── __init__.py
│   │   ├── baseline/
│   │   ├── ensemble/
│   │   └── rl/
│   ├── backtest/                       # Backtesting
│   ├── risk/                           # Risk management
│   ├── pricing/                        # Bet pricing
│   ├── sim/                            # Simulation
│   └── utils/                          # Utilities
│       ├── db.py                       # Database utilities
│       ├── config.py                   # Config management
│       └── validation.py               # Validation helpers
│
├── R/                                  # R scripts (legacy/specialized)
│   ├── ingestion/                      # R-specific ingestors
│   │   ├── nflverse_extract.R
│   │   └── pbp_transform.R
│   ├── features/                       # R feature engineering
│   │   └── epa_features.R
│   ├── analysis/                       # R analysis scripts
│   └── reports/                        # R reports (Quarto)
│
├── db/                                 # Database migrations & schema
│   ├── migrations/                     # Numbered migrations
│   │   ├── 001_init.sql
│   │   ├── 002_timescale.sql
│   │   ├── 003_mart_game_weather.sql
│   │   ├── 004_advanced_features.sql
│   │   ├── 005_enhance_mart_views.sql
│   │   ├── 006_remove_weather_duplication.sql
│   │   └── 007_data_quality_log.sql
│   ├── views/                          # View definitions
│   │   ├── mart_game_summary.sql
│   │   └── mart_team_stats.sql
│   ├── functions/                      # Database functions
│   │   ├── calculate_turnovers.sql
│   │   └── upsert_helpers.sql
│   └── seeds/                          # Reference data
│       ├── teams.csv
│       └── stadiums.csv
│
├── infrastructure/                     # Infrastructure as code
│   ├── docker/
│   │   ├── Dockerfile.etl              # ETL container
│   │   ├── Dockerfile.api              # API container
│   │   └── docker-compose.yaml
│   ├── k8s/                            # Kubernetes manifests (future)
│   └── terraform/                      # Cloud infrastructure (future)
│
├── scripts/                            # Operational scripts
│   ├── dev/                            # Development utilities
│   │   ├── setup_env.sh
│   │   ├── reset_db.sh
│   │   └── run_tests.sh
│   ├── deploy/                         # Deployment scripts
│   │   ├── deploy_etl.sh
│   │   └── migrate_db.sh
│   ├── maintenance/                    # Maintenance scripts
│   │   ├── vacuum_db.sh
│   │   └── backup_data.sh
│   └── analysis/                       # One-off analysis scripts
│       ├── check_2025_data.R
│       └── run_reports.sh
│
├── data/                               # Data storage (gitignored except samples)
│   ├── raw/                            # Raw data cache
│   │   ├── nflverse/                   # Cached nflverse data
│   │   ├── odds/                       # Odds history
│   │   └── weather/                    # Weather cache
│   ├── processed/                      # Processed/transformed data
│   │   ├── features/                   # Generated features
│   │   └── predictions/                # Model outputs
│   ├── staging/                        # Staging area for validation
│   └── archive/                        # Historical archives
│
├── analysis/                           # Analysis & research
│   ├── notebooks/                      # Jupyter/Quarto notebooks
│   │   ├── exploratory/
│   │   └── production/
│   ├── dissertation/                   # Academic work
│   │   ├── main/
│   │   └── figures/
│   └── reports/                        # Generated reports
│
├── tests/                              # Test suite
│   ├── unit/                           # Unit tests
│   │   ├── etl/
│   │   ├── features/
│   │   └── models/
│   ├── integration/                    # Integration tests
│   │   ├── test_full_pipeline.py
│   │   └── test_database.py
│   ├── fixtures/                       # Test fixtures
│   │   ├── sample_schedules.csv
│   │   └── mock_api_responses/
│   └── conftest.py
│
├── logs/                               # Application logs
│   ├── etl/                            # ETL pipeline logs
│   ├── models/                         # Model training logs
│   └── api/                            # API logs (future)
│
├── models/                             # Trained model artifacts
│   ├── baseline/
│   ├── ensemble/
│   └── metadata/                       # Model metadata/versioning
│
├── .github/                            # GitHub Actions
│   └── workflows/
│       ├── ci.yml                      # CI/CD pipeline
│       ├── nightly-etl.yml             # Scheduled ETL
│       └── weekly-backfill.yml         # Weekly full refresh
│
└── pgdata/                             # PostgreSQL data (Docker volume)
    └── (gitignored)
```

---

## Migration Steps

### Phase 1: Documentation Organization (Low Risk)
1. Create `docs/` structure
2. Move all MD files to appropriate subdirectories
3. Create index and navigation
4. Update README with new structure

### Phase 2: ETL Pipeline Creation (Medium Risk)
1. Create `etl/` package structure
2. Port ingestion scripts to Python ETL framework
3. Add validation, deduplication, error handling
4. Implement data quality monitoring
5. Create configuration files

### Phase 3: Code Organization (Medium Risk)
1. Reorganize `py/` package
2. Move R scripts to proper structure
3. Consolidate `scripts/` into categorized subdirectories
4. Update imports and references

### Phase 4: Data Organization (Low Risk)
1. Reorganize `data/` directory
2. Move raw data to proper cache locations
3. Create staging/processed directories
4. Update .gitignore

### Phase 5: Infrastructure (Medium Risk)
1. Create `infrastructure/` directory
2. Move Docker configs
3. Add deployment scripts
4. Create CI/CD workflows

### Phase 6: Testing (High Value)
1. Reorganize `tests/` directory
2. Add fixtures for ETL testing
3. Create integration test suite
4. Add data quality tests

---

## ETL Framework Design

### Key Principles
1. **Idempotent**: Can re-run safely without duplicates
2. **Transactional**: Rollback on failure
3. **Monitored**: Log all operations, track metrics
4. **Validated**: Schema + data quality checks before load
5. **Audited**: Track lineage and changes

### Data Quality Checks
```python
# Schema validation
- Column names match expected schema
- Data types correct
- Required fields not null
- Value ranges within bounds

# Business logic validation  
- Game dates within valid range
- Scores non-negative
- Teams exist in reference table
- No future games with scores

# Referential integrity
- game_id exists before inserting plays
- player_id exists before inserting rosters
- Foreign keys validated

# Deduplication
- Check existing records by natural key
- UPSERT with conflict resolution
- Log duplicate detection
```

### Error Handling Strategy
```python
# Extract failures
- API rate limiting → exponential backoff
- Network errors → retry with circuit breaker
- Missing data → log warning, continue

# Transform failures  
- Schema mismatch → quarantine + alert
- Invalid data → quarantine + alert
- Parsing errors → log + skip row

# Load failures
- FK violation → rollback + alert
- Duplicate key → upsert logic
- Database down → retry with backoff
```

### Monitoring Metrics
```python
# Pipeline metrics
- Records extracted/transformed/loaded
- Success/failure rates
- Processing time per stage
- Data quality score

# Data metrics
- Freshness (time since last update)
- Completeness (% of expected records)
- Accuracy (validation pass rate)
- Consistency (cross-table checks)

# Alerts
- Pipeline failure → immediate
- Data quality degradation → warning
- Unusual patterns → investigation
```

---

## Configuration Management

### Environment Variables
```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5544
POSTGRES_DB=devdb01
POSTGRES_USER=dro
POSTGRES_PASSWORD=***

# APIs
ODDS_API_KEY=***
ODDS_API_BASE_URL=https://api.the-odds-api.com

# ETL Config
ETL_LOG_LEVEL=INFO
ETL_BATCH_SIZE=1000
ETL_RETRY_ATTEMPTS=3
ETL_RETRY_DELAY=5

# Feature Flags
ENABLE_REALTIME=false
ENABLE_BACKFILL=true
VALIDATE_BEFORE_LOAD=true
```

### Data Source Configuration (YAML)
```yaml
# etl/config/sources.yaml
nflverse:
  type: r_package
  package: nflreadr
  endpoints:
    schedules:
      function: load_schedules
      params:
        seasons: [2025]
    pbp:
      function: load_pbp
      params:
        seasons: [2025]
    rosters:
      function: load_rosters
      params:
        seasons: [2025]
  rate_limit: null
  retry_config:
    max_attempts: 3
    backoff: exponential

odds_api:
  type: rest_api
  base_url: https://api.the-odds-api.com
  endpoints:
    spreads:
      path: /v4/sports/americanfootball_nfl/odds
      params:
        markets: spreads,totals
        regions: us
        oddsFormat: american
  rate_limit:
    requests_per_minute: 500
  retry_config:
    max_attempts: 3
    backoff: exponential
```

### Schema Definitions (YAML)
```yaml
# etl/config/schemas.yaml
schedules:
  required_columns:
    - game_id
    - season
    - week
    - home_team
    - away_team
    - kickoff
  optional_columns:
    - spread_close
    - total_close
    - home_score
    - away_score
  data_types:
    game_id: str
    season: int
    week: int
    home_score: Optional[int]
  constraints:
    season:
      min: 1999
      max: 2030
    week:
      min: 1
      max: 22
    home_score:
      min: 0
      max: 100

plays:
  required_columns:
    - play_id
    - game_id
    - posteam
    - quarter
  column_aliases:
    quarter: [qtr, quarter]  # Handle name variations
    time_seconds: [game_seconds_remaining]
  # ... more schema definitions
```

---

## Automation Strategy

### Daily Pipeline (6am EST)
```bash
# Extract new/updated data
- Check for completed games since last run
- Fetch play-by-play for completed games
- Fetch latest odds snapshots
- Fetch weather for game locations

# Validate & Transform
- Schema validation
- Data quality checks
- Deduplicate against existing data
- Transform to database schema

# Load
- Transactional load to staging
- Run data quality tests
- Promote to production tables
- Refresh materialized views

# Post-process
- Generate features for new games
- Update predictions
- Calculate performance metrics
- Send summary report
```

### Weekly Pipeline (Monday 10am EST)
```bash
# Full refresh
- Re-fetch current season data
- Validate against database
- Identify and fix discrepancies
- Backfill any missing data

# Maintenance
- Update rosters (weekly changes)
- Refresh all materialized views
- Vacuum and analyze tables
- Generate weekly performance report
```

### Continuous Monitoring
```bash
# Health checks (every 5 min)
- Database connectivity
- API availability
- Data freshness
- Queue depth

# Alerts
- Pipeline failure → PagerDuty
- Data quality drop → Slack
- API rate limit → Email
```

---

## Next Actions

1. **Review this plan** - Confirm architecture decisions
2. **Start Phase 1** - Organize documentation (safe, low risk)
3. **Build ETL framework** - Create base classes and configuration
4. **Port one pipeline** - Start with schedules as proof of concept
5. **Add monitoring** - Implement logging and metrics
6. **Iterate** - Apply learnings to remaining pipelines

---

## Benefits of This Structure

✅ **Clear separation of concerns** - ETL, features, models, analysis  
✅ **Enterprise data quality** - Validation, monitoring, deduplication  
✅ **Production ready** - Error handling, rollback, observability  
✅ **Scalable** - Add new data sources easily  
✅ **Maintainable** - Documented, tested, organized  
✅ **Automated** - Scheduled pipelines with monitoring  
✅ **Professional** - Matches industry best practices  

