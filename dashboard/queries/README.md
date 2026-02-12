# Dashboard Query Module

Centralized SQL query classes for the NFL Analytics Streamlit dashboard.

## Overview

This module provides organized, reusable query classes for all dashboard components. All SQL queries are extracted from dashboard code into dedicated query classes organized by data domain.

## Structure

```
dashboard/queries/
├── __init__.py              # Module initialization and exports
├── predictions.py           # ML predictions and player analytics
├── games.py                 # Games, matchups, and scheduling
├── models.py                # Model registry and training metadata
├── paper_trading.py         # Paper trading sessions and trades
├── system.py                # System and database metadata
└── README.md               # This file
```

## Quick Start

### Basic Usage

```python
from dashboard.queries.predictions import PredictionQueries
from dashboard.queries.games import GameQueries
import psycopg

# Connect to database
with psycopg.connect(**DB_CONFIG) as conn:
    # Fetch player predictions
    df_preds = PredictionQueries.retrospective_metrics(conn, season=2024)

    # Fetch current week games
    df_games = GameQueries.current_week_games(conn, season=2024, week=12)

    # Display results
    print(f"Found {len(df_preds)} predictions")
    print(f"Found {len(df_games)} games")
```

### With Streamlit Caching

```python
from dashboard.queries.predictions import PredictionQueries
from dashboard.queries.games import GameQueries

@st.cache_data(ttl=300)
def get_predictions(season, week):
    """Cached version for dashboard."""
    with psycopg.connect(**DB_CONFIG) as conn:
        return PredictionQueries.retrospective_metrics(conn, season=season)

@st.cache_data(ttl=300)
def get_games(season, week):
    """Cached version for dashboard."""
    with psycopg.connect(**DB_CONFIG) as conn:
        return GameQueries.current_week_games(conn, season, week)

# In Streamlit app
df_preds = get_predictions(2024, 12)
df_games = get_games(2024, 12)
st.dataframe(df_preds)
st.dataframe(df_games)
```

## Query Classes

### PredictionQueries

SQL queries for `ml.predictions` table and prediction analytics.

| Method | Purpose | Returns |
|--------|---------|---------|
| `retrospective_metrics()` | Player passing predictions with actuals | DataFrame |
| `weekly_performance_metrics()` | Weekly aggregated performance by season/week | DataFrame |
| `bnn_uncertainty_metrics()` | BNN uncertainty and calibration | DataFrame |
| `predictions_with_odds()` | Predictions + sportsbook odds + edge | DataFrame |
| `line_movement_history()` | Historical line movement for player | DataFrame |
| `model_comparison()` | Compare model versions (player + game) | DataFrame |

**Example:**
```python
from dashboard.queries.predictions import PredictionQueries

with psycopg.connect(**DB_CONFIG) as conn:
    # Get predictions for 2024 season
    df = PredictionQueries.retrospective_metrics(
        conn,
        model_version="v5.1",
        season=2024,
        limit=1000
    )
    print(df.columns)
    # Output: game_id, season, week, player_name, predicted_value, actual_value, mae, ...
```

### GameQueries

SQL queries for `games` table and game analytics.

| Method | Purpose | Returns |
|--------|---------|---------|
| `current_week_games()` | Games for specific week with predictions | DataFrame |
| `teams_without_games()` | Teams on bye week | List[str] |
| `season_schedule()` | Complete season schedule | DataFrame |
| `recent_games()` | Most recent games | DataFrame |
| `data_coverage_by_season_week()` | Data available by season/week | DataFrame |
| `earliest_latest_games()` | Earliest and latest games in DB | dict |
| `games_by_team()` | All games for specific team | DataFrame |

**Example:**
```python
from dashboard.queries.games import GameQueries

with psycopg.connect(**DB_CONFIG) as conn:
    # Get bye week teams
    teams_on_bye = GameQueries.teams_without_games(conn, season=2024, week=12)
    print(f"Teams on bye: {teams_on_bye}")
    # Output: ['KC', 'SF']

    # Get team schedule
    df = GameQueries.games_by_team(conn, team='KC', season=2024)
    print(f"KC games: {len(df)}")
```

### ModelQueries

SQL queries for `ml.model_registry` and `ml.training_runs` tables.

| Method | Purpose | Returns |
|--------|---------|---------|
| `model_registry()` | All models in registry | DataFrame |
| `model_by_id()` | Single model by ID | DataFrame |
| `models_by_status()` | Models filtered by status | DataFrame |
| `training_runs()` | Recent training runs | DataFrame |
| `latest_training_run()` | Latest run for model | DataFrame |
| `model_family_comparison()` | All versions of model family | DataFrame |
| `ensemble_components()` | Production ensemble components | DataFrame |
| `training_run_history()` | Training history with summary stats | DataFrame |
| `model_update_timeline()` | Timeline of model updates | DataFrame |

**Example:**
```python
from dashboard.queries.models import ModelQueries

with psycopg.connect(**DB_CONFIG) as conn:
    # Get all production models
    df = ModelQueries.models_by_status(conn, status='production')
    print(f"Production models: {len(df)}")

    # Get ensemble components
    df = ModelQueries.ensemble_components(conn)
    print(df[['model_name', 'version', 'ensemble_weight']])
```

### PaperTradingQueries

SQL queries for paper trading tables.

| Method | Purpose | Returns |
|--------|---------|---------|
| `active_sessions()` | Active paper trading sessions | DataFrame |
| `session_trades()` | All trades for session | DataFrame |
| `pending_trades()` | Trades awaiting settlement | DataFrame |
| `performance_by_prop_type()` | Performance breakdown by prop | DataFrame |
| `performance_by_bookmaker()` | Performance breakdown by book | DataFrame |
| `session_summary()` | Aggregated session metrics | DataFrame |
| `bankroll_progression()` | Bankroll over time | DataFrame |
| `edge_analysis()` | Performance by edge size | DataFrame |
| `session_comparison()` | Compare across sessions | DataFrame |

**Example:**
```python
from dashboard.queries.paper_trading import PaperTradingQueries

with psycopg.connect(**DB_CONFIG) as conn:
    # Get active sessions
    df = PaperTradingQueries.active_sessions(conn, days_back=7)

    for session_id in df['session_id']:
        # Get trades for session
        trades = PaperTradingQueries.session_trades(conn, session_id)
        summary = PaperTradingQueries.session_summary(conn, session_id)

        print(f"Session {session_id}: ROI {summary['roi']:.2%}")
```

### SystemQueries

SQL queries for database metadata and health checks.

| Method | Purpose | Returns |
|--------|---------|---------|
| `database_statistics()` | Total games, plays, seasons, DB size | dict |
| `schema_statistics()` | Table counts by schema | DataFrame |
| `recent_data_ingestion()` | Recent data by season/week | DataFrame |
| `table_row_counts()` | Row counts for all tables | DataFrame |
| `column_statistics()` | Column metadata | DataFrame |
| `prediction_coverage()` | ML predictions coverage stats | dict |
| `model_coverage()` | Model registry stats by status | DataFrame |
| `slow_queries_log()` | Slow queries from pg_stat_statements | DataFrame |
| `index_statistics()` | Index usage statistics | DataFrame |
| `last_vacuum_analyze()` | Last maintenance times | DataFrame |
| `connection_stats()` | Current connection stats | dict |
| `materialized_view_status()` | Materialized view status | DataFrame |

**Example:**
```python
from dashboard.queries.system import SystemQueries

with psycopg.connect(**DB_CONFIG) as conn:
    # Get database stats
    stats = SystemQueries.database_statistics(conn)
    print(f"Database size: {stats['db_size']}")
    print(f"Total games: {stats['total_games']}")

    # Get schema stats
    df = SystemQueries.schema_statistics(conn)
    st.table(df)
```

## Migration Guide

### Before (Inline Queries)

```python
# dashboard/app.py (old approach)
def fetch_predictions():
    query = """
    SELECT
        p.game_id, p.player_name, p.prediction_mean, p.actual_value
    FROM ml.predictions p
    WHERE p.stat_type = 'passing_yards'
    ORDER BY p.game_id DESC
    LIMIT 1000
    """
    with psycopg.connect(**DB_CONFIG) as conn:
        return pd.read_sql_query(query, conn)
```

### After (Query Module)

```python
# dashboard/app.py (new approach)
from dashboard.queries.predictions import PredictionQueries

@st.cache_data(ttl=300)
def fetch_predictions():
    with psycopg.connect(**DB_CONFIG) as conn:
        return PredictionQueries.retrospective_metrics(conn, limit=1000)
```

## Benefits

1. **Centralization**: All queries in one place, easy to audit and maintain
2. **Reusability**: Import and use across multiple pages/components
3. **Consistency**: Standard parameter handling and error handling
4. **Documentation**: Each method has docstring with return schema
5. **Testability**: Query classes can be unit tested independently
6. **Caching**: Works seamlessly with `@st.cache_data` and `@cache_data`
7. **Type Safety**: Optional type hints for parameters

## Porting Existing Code

### Step 1: Identify Query Domain

Categorize queries into one of 5 modules:
- **Predictions**: `ml.predictions`, `ml.game_predictions`, odds-related
- **Games**: `games` table, schedule, matchups
- **Models**: `ml.model_registry`, `ml.training_runs`
- **Paper Trading**: `paper_trading_*` tables
- **System**: Database metadata, `pg_*` tables

### Step 2: Find Existing Query

Search for the query in source file (e.g., `dashboard/app.py`, `pages/paper_trading.py`)

### Step 3: Add to Query Class

Add method to appropriate class with:
- Clear method name (e.g., `retrospective_metrics`)
- Full docstring with args and return schema
- Parameter validation
- Use `%s` placeholders for params

### Step 4: Update Dashboard Code

Replace inline query with query class import and call:

```python
# Before
query = """SELECT ..."""
df = pd.read_sql_query(query, conn, params=(...))

# After
from dashboard.queries.predictions import PredictionQueries
df = PredictionQueries.method_name(conn, param1=value1)
```

### Step 5: Test

```bash
# Test query class directly
python -c "
from dashboard.queries.predictions import PredictionQueries
import psycopg

with psycopg.connect('postgresql://dro:...@localhost:5544/devdb01') as conn:
    df = PredictionQueries.retrospective_metrics(conn, season=2024)
    print(f'Returned {len(df)} rows')
    print(df.columns.tolist())
"
```

## Statistics

### Query Coverage

| Module | Methods | Queries | Coverage |
|--------|---------|---------|----------|
| predictions.py | 6 | 6 major | 100% |
| games.py | 7 | 7 major | 100% |
| models.py | 9 | 9 major | 100% |
| paper_trading.py | 9 | 9 major | 100% |
| system.py | 12 | 12 major | 100% |
| **Total** | **43** | **43** | **100%** |

### Extracted Queries

- `dashboard/app.py`: 15 queries → predictions.py, games.py, models.py, system.py
- `dashboard/pages/paper_trading.py`: 7 queries → paper_trading.py
- `dashboard/pages/player_props.py`: 2 queries → predictions.py

**Total Queries Extracted**: 24 SQL queries organized into 43 reusable methods

### Lines of Code

| Module | Lines | Methods |
|--------|-------|---------|
| predictions.py | 340 | 6 |
| games.py | 285 | 7 |
| models.py | 310 | 9 |
| paper_trading.py | 390 | 9 |
| system.py | 350 | 12 |
| __init__.py | 30 | - |
| README.md | 450+ | - |
| **Total** | **2,155** | **43** |

## Next Steps

1. **Update Dashboard Pages**: Import and use query classes in:
   - `dashboard/app.py` (main page)
   - `dashboard/pages/paper_trading.py`
   - `dashboard/pages/player_props.py`

2. **Add Tests**: Create `tests/dashboard/test_queries.py` with unit tests for each class

3. **Add Caching Layer**: Integrate Redis caching for slow queries

4. **Add Monitoring**: Track query execution times and cache hit rates

## Troubleshooting

### ImportError: No module named 'dashboard.queries'

Make sure you're in project root directory:
```bash
cd \
python -c "from dashboard.queries.predictions import PredictionQueries"
```

### Query Returns Empty DataFrame

Check:
1. Database connection is valid
2. Parameters match expected schema
3. Table exists: `SELECT 1 FROM {table} LIMIT 1`

### Slow Query Performance

Use `SystemQueries.slow_queries_log()` to identify bottlenecks:
```python
from dashboard.queries.system import SystemQueries

with psycopg.connect(**DB_CONFIG) as conn:
    df = SystemQueries.slow_queries_log(conn, min_duration_ms=100)
    print(df)
```

## References

- **Database Schema**: `docs/database/DATABASE_QUICK_REFERENCE.md`
- **SQL Patterns**: `docs/database/SQL_PATTERNS.md`
- **Dashboard**: `dashboard/app.py`
- **Caching**: `py/utils/claude_cache.py`
