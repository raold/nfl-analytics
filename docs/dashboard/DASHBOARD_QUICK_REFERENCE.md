# Dashboard Quick Reference Guide

**Last Updated:** November 22, 2025
**Dashboard Version:** 2.0 (Refactored)

## Quick Start

### Running the Dashboard

```bash
# Activate virtual environment
source .venv/bin/activate  # or: uv venv && source .venv/bin/activate

# Run refactored dashboard (NEW)
streamlit run dashboard/app_refactored.py

# Run original dashboard (LEGACY - for comparison)
streamlit run dashboard/app.py

# Run on custom port
streamlit run dashboard/app_refactored.py --server.port 8502
```

### Accessing the Dashboard

- **Local URL:** http://localhost:8501
- **Default Pages:**
  - 📊 Predictions - Main analytics page
  - 🎯 Game Predictions - Weekly game forecasts
  - 🤖 Model Status - Model comparison and registry
  - ⚙️ System Status - Service health monitoring
  - 📊 Data Summary - Database coverage statistics

## Module Reference

### 1. Query Modules (`dashboard/queries/`)

All queries return pandas DataFrames. Always use with connection pooling:

```python
from dashboard.queries import PredictionQueries
from dashboard.utils.db import get_db_connection

with get_db_connection() as conn:
    df = PredictionQueries.retrospective_metrics(conn)
```

#### PredictionQueries

| Method | Purpose | Returns |
|--------|---------|---------|
| `retrospective_metrics(conn, model_version, season)` | Historical predictions with outcomes | DataFrame |
| `weekly_performance_metrics(conn)` | Weekly aggregated metrics | DataFrame |
| `model_comparison(conn)` | Compare model performance | DataFrame |
| `bnn_uncertainty_metrics(conn)` | BNN epistemic/aleatoric uncertainty | DataFrame |

#### GameQueries

| Method | Purpose | Returns |
|--------|---------|---------|
| `current_week_games(conn, season, week)` | Games for specific week | DataFrame |
| `teams_without_games(conn, season, week)` | Bye week teams | DataFrame |
| `data_coverage_by_season_week(conn)` | Data availability by period | DataFrame |

#### ModelQueries

| Method | Purpose | Returns |
|--------|---------|---------|
| `model_registry(conn)` | All registered models | DataFrame |
| `models_by_status(conn, status)` | Filter by status (production/staging) | DataFrame |
| `training_runs(conn)` | Model training history | DataFrame |

#### SystemQueries

| Method | Purpose | Returns |
|--------|---------|---------|
| `database_statistics(conn)` | Database stats (games, plays, seasons) | Dict |
| `schema_statistics(conn)` | Table counts by schema | DataFrame |
| `table_row_counts(conn)` | Row counts for all tables | DataFrame |

### 2. Chart Components (`dashboard/components/charts.py`)

All chart functions return Plotly `go.Figure` objects:

```python
from dashboard.components import plot_win_rate_over_time
import streamlit as st

fig = plot_win_rate_over_time(df_performance)
st.plotly_chart(fig, use_container_width=True)
```

#### Available Charts

| Function | Required Columns | Description |
|----------|------------------|-------------|
| `plot_win_rate_over_time(df)` | season, week, win_rate | Win rate trend line |
| `plot_mae_over_time(df)` | season, week, mae | MAE trend line |
| `plot_calibration(df)` | prob_win, status | Calibration scatter plot |
| `plot_error_distribution(df)` | predicted_value, actual_value | Error histogram |
| `plot_pnl_over_time(df)` | season, week, weekly_pnl | Cumulative P&L |
| `plot_model_comparison_mae(df)` | model_name, version, mae | Model comparison bar chart |
| `plot_edge_distribution(df)` | edge | Edge histogram |

### 3. Metrics Components (`dashboard/components/metrics.py`)

Statistical calculations and display functions:

```python
from dashboard.components import display_summary_metrics, calculate_roi

# Display metrics automatically in Streamlit
display_summary_metrics(df_predictions)

# Calculate individual metrics
roi = calculate_roi(df_predictions, stake=110.0)
sharpe = calculate_sharpe_ratio(df_predictions)
```

#### Calculation Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `calculate_win_rate(df)` | Win rate (0.0-1.0) | float |
| `calculate_mae(df)` | Mean absolute error | float |
| `calculate_roi(df, stake)` | Return on investment % | float |
| `calculate_sharpe_ratio(df)` | Risk-adjusted returns | float |
| `calculate_max_drawdown(df)` | Max cumulative loss | float |
| `calculate_kelly_fraction(win_rate, odds)` | Optimal bet fraction | float |

#### Display Functions (Streamlit)

| Function | Purpose | Displays |
|----------|---------|----------|
| `display_summary_metrics(df)` | Top-line metrics | 4 columns: Total, Win Rate, MAE, P&L |
| `display_advanced_metrics(df)` | Advanced stats | 4 columns: ROI, Sharpe, Drawdown, Calibration |
| `display_edge_statistics(df)` | Edge distribution | 3 columns: Mean, Median, Std, etc. |

### 4. Table Components (`dashboard/components/tables.py`)

DataFrame formatting for display:

```python
from dashboard.components import format_predictions_table
import streamlit as st

df_formatted = format_predictions_table(df_raw)
st.dataframe(df_formatted, use_container_width=True)
```

#### Formatting Functions

| Function | Purpose | Output Columns |
|----------|---------|----------------|
| `format_predictions_table(df)` | Player predictions | Season, Week, Player, Team, Predicted, Actual, Error, P&L |
| `format_model_comparison_table(df)` | Model performance | Model, Version, MAE, Win Rate, Calibration, P&L |
| `format_game_predictions_table(df)` | Game matchups | Matchup, Time, Score, Spread, Total, Win Prob |

#### Utility Functions

| Function | Purpose |
|----------|---------|
| `highlight_best_model(df, metric)` | Highlight top performer (green) |
| `add_rank_column(df, metric)` | Add ranking column |
| `paginate_table(df, page_size, page)` | Pagination for large tables |
| `sort_table(df, column, ascending)` | Sort by column |

### 5. Card Components (`dashboard/components/cards.py`)

Pre-built Streamlit UI components:

```python
from dashboard.components import service_health_card, game_prediction_card

service_health_card({"name": "PostgreSQL", "host": "localhost", "port": 5544})
game_prediction_card(game_dict)
```

#### Available Cards

| Function | Purpose |
|----------|---------|
| `service_health_card(service)` | Service status indicator (green/red) |
| `game_prediction_card(game)` | Full game card with predictions |
| `metric_card(label, value, delta)` | Custom metric display |
| `section_banner(title, icon)` | Section header with icon |

## Common Patterns

### Pattern 1: Full Analytics Page

```python
import streamlit as st
from dashboard.queries import PredictionQueries
from dashboard.components import (
    display_summary_metrics,
    plot_win_rate_over_time,
    format_predictions_table
)
from dashboard.utils.db import get_db_connection

def render_analytics_page():
    st.header("Analytics")

    # Fetch data
    with get_db_connection() as conn:
        df = PredictionQueries.retrospective_metrics(conn)

    # Display metrics
    display_summary_metrics(df)

    # Show chart
    fig = plot_win_rate_over_time(df)
    st.plotly_chart(fig, use_container_width=True)

    # Show table
    df_formatted = format_predictions_table(df)
    st.dataframe(df_formatted, use_container_width=True)
```

### Pattern 2: Model Comparison

```python
from dashboard.queries import PredictionQueries
from dashboard.components import plot_model_comparison_mae, format_model_comparison_table

with get_db_connection() as conn:
    df_models = PredictionQueries.model_comparison(conn)

# Chart
fig = plot_model_comparison_mae(df_models)
st.plotly_chart(fig, use_container_width=True)

# Table
df_formatted = format_model_comparison_table(df_models)
st.dataframe(df_formatted, use_container_width=True)
```

### Pattern 3: Live Game Updates

```python
from dashboard.queries import GameQueries
from dashboard.components import game_prediction_card

with get_db_connection() as conn:
    df_games = GameQueries.current_week_games(conn, season=2025, week=None)

for _, game in df_games.iterrows():
    game_prediction_card(game.to_dict())
    st.markdown("---")
```

## File Locations

```
dashboard/
├── app_refactored.py          # Main application (USE THIS)
├── app.py                      # Legacy version (BACKUP)
│
├── queries/                    # SQL query modules
│   ├── predictions.py
│   ├── games.py
│   ├── models.py
│   └── system.py
│
├── components/                 # UI components
│   ├── charts.py              # Plotly charts
│   ├── metrics.py             # Statistical calculations
│   ├── tables.py              # DataFrame formatting
│   └── cards.py               # Pre-built UI cards
│
└── utils/                      # Utility modules
    ├── db.py                  # Connection pooling
    └── schema_validator.py    # Schema validation
```

## Debugging

### Import Errors

```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Test specific import
python3 -c "from dashboard.queries import PredictionQueries; print('OK')"
```

### Database Connection Issues

```bash
# Check environment variables
echo $DB_HOST $DB_PORT $DB_NAME $DB_USER

# Test connection
python3 -c "
import psycopg
conn = psycopg.connect(host='localhost', port=5544, dbname='devdb01', user='dro')
print('Connected:', conn.info.status)
conn.close()
"
```

### Page Not Loading

1. Check Streamlit logs in terminal
2. Verify database is running: `docker ps | grep postgres`
3. Test query module in isolation:
   ```python
   from dashboard.queries import PredictionQueries
   from dashboard.utils.db import get_db_connection

   with get_db_connection() as conn:
       df = PredictionQueries.retrospective_metrics(conn)
       print(f"Rows: {len(df)}")
   ```

### Chart Not Rendering

```python
# Test chart function with minimal data
import pandas as pd
from dashboard.components import plot_win_rate_over_time

df = pd.DataFrame({
    'season': [2024, 2024],
    'week': [1, 2],
    'win_rate': [0.55, 0.58]
})

fig = plot_win_rate_over_time(df)
print(f"Chart created: {fig is not None}")
```

## Performance Tips

### 1. Use Connection Pooling

```python
# ✅ GOOD - Reuses connections
from dashboard.utils.db import get_db_connection

with get_db_connection() as conn:
    df = PredictionQueries.retrospective_metrics(conn)

# ❌ BAD - Creates new connection each time
conn = psycopg.connect(**DB_CONFIG)
df = pd.read_sql_query(query, conn)
conn.close()
```

### 2. Cache Expensive Queries

```python
import streamlit as st

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_predictions():
    with get_db_connection() as conn:
        return PredictionQueries.retrospective_metrics(conn)

# Fast on subsequent loads
df = fetch_predictions()
```

### 3. Limit Query Results

```python
# Limit rows in query (faster than filtering in Python)
df = PredictionQueries.retrospective_metrics(
    conn,
    model_version="v6.1",
    season=2025  # Filter server-side
)
```

### 4. Use Column Selection

```python
# Only fetch needed columns
query = "SELECT game_id, season, week, mae FROM ml.predictions LIMIT 1000"
df = pd.read_sql_query(query, conn)  # Faster than SELECT *
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'dashboard'` | Python path incorrect | Run from project root: `cd /Users/dro/rice/nfl-analytics` |
| `OperationalError: could not connect to server` | Database not running | Start Docker: `docker-compose up -d postgres` |
| `KeyError: 'win_rate'` | DataFrame missing column | Check query returned expected columns |
| `AttributeError: module has no attribute 'plot_win_rate_over_time'` | Import error | Check `__init__.py` exports function |

## Cheat Sheet

### Adding a New Chart

1. Add function to `dashboard/components/charts.py`
2. Export in `dashboard/components/__init__.py`
3. Import in page: `from dashboard.components import my_new_chart`
4. Use: `fig = my_new_chart(df); st.plotly_chart(fig)`

### Adding a New Query

1. Add method to appropriate class in `dashboard/queries/[module].py`
2. Already exported via class (no __init__ change needed)
3. Import in page: `from dashboard.queries import PredictionQueries`
4. Use: `df = PredictionQueries.my_new_query(conn)`

### Adding a New Page

1. Create function in `dashboard/app_refactored.py`:
   ```python
   def render_my_new_page():
       st.header("My Page")
       # ...page content
   ```
2. Add to navigation in `main()`:
   ```python
   page = st.sidebar.radio("Select Page", [..., "My New Page"])
   if page == "My New Page":
       render_my_new_page()
   ```

## Resources

- **Full Documentation:** `docs/deployment/DASHBOARD_REFACTORING_COMPLETE.md`
- **Query Schemas:** `docs/database/DATABASE_QUICK_REFERENCE.md`
- **SQL Patterns:** `docs/database/SQL_PATTERNS.md`
- **Component Examples:** `dashboard/components/*.py` (docstrings)

## Support

- Check logs in terminal where Streamlit is running
- Review error messages in browser (Streamlit shows full tracebacks)
- Test queries independently before adding to dashboard
- Use `st.write(df.columns)` to debug DataFrame structure
