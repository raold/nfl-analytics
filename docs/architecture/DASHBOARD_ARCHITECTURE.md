# Dashboard Architecture Documentation

**Last Updated:** November 22, 2025
**Version:** 2.0 (Post-Refactoring)
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Architecture Layers](#architecture-layers)
4. [Component Catalog](#component-catalog)
5. [Data Flow](#data-flow)
6. [Configuration Management](#configuration-management)
7. [Query Architecture](#query-architecture)
8. [API Integration](#api-integration)
9. [Caching Strategy](#caching-strategy)
10. [Making Updates](#making-updates)
11. [Common Tasks](#common-tasks)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The NFL Analytics Dashboard is a **multi-page Streamlit application** that provides real-time monitoring of prediction performance, model metrics, and betting opportunities.

### Key Characteristics

- **Framework:** Streamlit 1.28+
- **Architecture:** Multi-page with shared components
- **Database:** PostgreSQL 14+ (psycopg3 connections)
- **API:** FastAPI backend (localhost:8000)
- **Caching:** Redis + Streamlit @cache decorators
- **Deployment:** Docker container (port 8501)

### Design Principles

1. **Separation of Concerns** - Queries, components, business logic separated
2. **Type Safety** - Comprehensive type hints (81% coverage)
3. **Performance** - Connection pooling, Redis caching, optimized queries
4. **Maintainability** - Centralized constants, reusable components
5. **Reliability** - Graceful degradation, comprehensive error handling

---

## Directory Structure

```
dashboard/
├── app.py                      # Main dashboard application (3,500+ lines)
├── constants.py                # 68 configuration constants
│
├── components/                 # Reusable UI components
│   ├── __init__.py
│   └── cards.py               # Game cards, service health cards
│
├── queries/                   # SQL query layer (43 methods)
│   ├── __init__.py
│   ├── predictions.py         # ML predictions queries (6 methods)
│   ├── games.py              # Games table queries (7 methods)
│   ├── models.py             # Model registry queries (9 methods)
│   ├── paper_trading.py      # Paper trading queries (9 methods)
│   └── system.py             # System metadata (12 methods)
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   └── api_client.py         # FastAPI client with validation (641 lines)
│
└── pages/                     # Additional dashboard pages
    ├── player_props.py        # Player props analysis
    └── paper_trading.py       # Paper trading simulation
```

### Supporting Files

```
py/utils/
└── psycopg_pool.py           # Connection pool manager (328 lines)

docs/
├── architecture/
│   └── DASHBOARD_ARCHITECTURE.md  # This file
├── guides/
│   ├── DASHBOARD_CONNECTION_POOLING.md
│   └── API_CLIENT_INTEGRATION_GUIDE.md
└── dashboard/
    ├── REPLACEMENT_GUIDE.md
    └── HTML_REPLACEMENT_COMPLETE.md
```

---

## Architecture Layers

### Layer 1: Presentation (Streamlit Components)

**Purpose:** User interface rendering

**Files:**
- `dashboard/app.py` - Main app, page navigation, layouts
- `dashboard/pages/*.py` - Additional pages
- `dashboard/components/cards.py` - Reusable UI components

**Key Functions:**
- `main()` - Entry point, page router
- `render_system_status_page()` - Service health monitoring
- `render_model_status_page()` - Model performance metrics
- `render_data_summary_page()` - Database statistics

**Responsibilities:**
- Page layout and navigation
- User input handling (filters, selectors)
- Data visualization (Plotly charts, tables)
- Component rendering

**Do NOT include in this layer:**
- SQL queries (use Query Layer)
- Business logic calculations
- API calls (use API Client)

---

### Layer 2: Business Logic

**Purpose:** Data transformation and calculations

**Files:**
- `dashboard/app.py` - Calculation functions
- `dashboard/utils/api_client.py` - Response validation

**Key Functions:**
- `calculate_edge(predicted, sportsbook)` - Edge calculation
- `normalize_team_display(team_abbr)` - Team name normalization
- `plot_win_rate_over_time(df)` - Chart generation

**Responsibilities:**
- Data transformations (e.g., edge calculation)
- Aggregations and summaries
- Chart/visualization logic
- Response validation and mapping

**Do NOT include in this layer:**
- Direct database queries (use Query Layer)
- Raw API calls (use API Client)
- UI rendering code

---

### Layer 3: Data Access (Query Layer)

**Purpose:** Database query abstraction

**Files:**
- `dashboard/queries/predictions.py` - ML predictions
- `dashboard/queries/games.py` - Games data
- `dashboard/queries/models.py` - Model registry
- `dashboard/queries/paper_trading.py` - Trading data
- `dashboard/queries/system.py` - System metadata

**Example:**
```python
from dashboard.queries.predictions import PredictionQueries

@cache_data(ttl=300)
def fetch_retrospective_metrics(model_version=None, season=None):
    pool = get_db_pool()
    with pool.get_connection() as conn:
        return PredictionQueries.retrospective_metrics(conn, model_version, season)
```

**Responsibilities:**
- SQL query definitions
- Parameter binding and validation
- Schema documentation
- Query optimization

**Guidelines:**
- ✅ One method per logical query
- ✅ Explicit column lists (no `SELECT *`)
- ✅ Type-hinted parameters
- ✅ Comprehensive docstrings
- ❌ No business logic in query methods
- ❌ No UI code

---

### Layer 4: Infrastructure

**Purpose:** Low-level services and configuration

**Files:**
- `py/utils/psycopg_pool.py` - Connection pooling
- `dashboard/constants.py` - Configuration constants
- `dashboard/utils/api_client.py` - API communication

**Key Components:**

#### a) Connection Pool
```python
from py.utils.psycopg_pool import get_connection_pool

pool = get_connection_pool(
    host="localhost",
    port=5544,
    dbname="devdb01",
    user="dro",
    password=os.getenv("DB_PASSWORD"),
    min_size=2,
    max_size=10
)

with pool.get_connection() as conn:
    # Use connection...
```

#### b) API Client
```python
from dashboard.utils.api_client import APIClient

api = APIClient("http://localhost:8000")
prediction = api.get_game_prediction(game_id, season, week)
```

#### c) Constants
```python
from dashboard.constants import (
    STRONG_SPREAD_EDGE_THRESHOLD,
    REDIS_CACHE_TTL_MEDIUM,
    COLOR_SUCCESS
)
```

**Responsibilities:**
- Connection management
- Configuration loading
- External service integration
- Error handling and retry logic

---

## Component Catalog

### 1. Service Health Card

**Location:** `dashboard/components/cards.py`

**Function:**
```python
def service_health_card(name: str, port: int, is_healthy: bool) -> None
```

**Usage:**
```python
from dashboard.components.cards import service_health_card

service_health_card("PostgreSQL", 5544, True)
```

**Renders:** Status badge with service name and port

---

### 2. Game Prediction Card

**Location:** `dashboard/components/cards.py`

**Function:**
```python
def game_prediction_card(
    game: dict,
    prediction: dict,
    spread_edge: float,
    total_edge: float,
    card_type: str = "scheduled"
) -> None
```

**Card Types:**
- `"scheduled"` - Upcoming game
- `"live"` - Game in progress
- `"completed"` - Finished game with results
- `"international"` - International game marker

**Usage:**
```python
game_prediction_card(
    game=game_row,
    prediction=api_prediction,
    spread_edge=-0.5,
    total_edge=1.5,
    card_type="live"
)
```

**Renders:**
- Team logos and names
- Betting lines comparison table
- Edge highlighting (green/yellow/gray)
- Prediction details (expandable)
- Accuracy tracking (completed games only)

---

### 3. Performance Charts

**Location:** `dashboard/app.py` (visualization functions)

**Functions:**
- `plot_win_rate_over_time(df: pd.DataFrame) -> go.Figure`
- `plot_mae_over_time(df: pd.DataFrame) -> go.Figure`
- `plot_ats_accuracy(df: pd.DataFrame) -> go.Figure`
- `plot_error_distribution(df: pd.DataFrame) -> go.Figure`
- `plot_calibration(df: pd.DataFrame) -> go.Figure`

**Usage:**
```python
df = fetch_performance_over_time()
fig = plot_win_rate_over_time(df)
st.plotly_chart(fig, use_container_width=True)
```

**Common Chart Properties:**
- Height: `CHART_HEIGHT_DEFAULT` (400px)
- Color scheme: Constants module colors
- Interactive: Hover tooltips, zoom, pan

---

## Data Flow

### 1. Page Load Flow

```
User navigates to page
    ↓
main() function
    ↓
Page router (st.sidebar.radio)
    ↓
render_*_page() function
    ↓
┌─────────────────────────────────────┐
│  Cache Check (@cache_data)          │
│  - Redis cache (30s-1hr TTL)        │
│  - Streamlit session cache          │
└─────────────────────────────────────┘
    ↓ (cache miss)
fetch_*() function
    ↓
Query Layer (PredictionQueries.*)
    ↓
Connection Pool
    ↓
PostgreSQL Database
    ↓
pd.DataFrame returned
    ↓
Business Logic (transformations)
    ↓
Visualization Components
    ↓
Rendered to user
```

### 2. API Prediction Flow

```
User requests game prediction
    ↓
get_game_predictions(game_id, season, week)
    ↓
APIClient.get_game_prediction()
    ↓
┌─────────────────────────────────────┐
│  Redis Cache Check (optional)       │
│  - 5min TTL for predictions          │
└─────────────────────────────────────┘
    ↓ (cache miss)
FastAPI POST /predict/game
    ↓
Ensemble v6.1 prediction
    ↓
Response validation (TypedDict)
    ↓
Dict[str, Any] returned
    ↓
game_prediction_card() component
    ↓
Rendered to user
```

### 3. Caching Strategy

**Three-Level Cache:**

1. **Redis Cache** (cross-session, shared)
   - TTL: 30s (live data) to 1hr (static data)
   - Keys: `streamlit:{function}:{args_hash}`
   - Hit rate: 70-85% typical

2. **Streamlit @cache_data** (per-session)
   - TTL: 300s (5 min) default
   - Cleared on: App restart, manual refresh
   - Stores: DataFrames, lists, dicts

3. **Streamlit @cache_resource** (singleton)
   - Persists: Until app restart
   - Used for: Connection pools, API clients

**Cache Keys:**
```python
# Redis cache key format
f"streamlit:{fetch_retrospective_metrics}:{hash(model_version, season)}"

# Example
"streamlit:fetch_retrospective_metrics:a3f2b1c8"
```

---

## Configuration Management

### Constants Module (`dashboard/constants.py`)

**68 constants organized by category:**

#### 1. Cache & Timeout (7 constants)
```python
REDIS_CACHE_TTL_LIVE = 30       # Live data (30 seconds)
REDIS_CACHE_TTL_MEDIUM = 300    # Queries (5 minutes)
REDIS_CACHE_TTL_LONG = 3600     # Static data (1 hour)
API_TIMEOUT_DEFAULT = 10        # Seconds
SOCKET_TIMEOUT_SECONDS = 5      # Health checks
```

#### 2. Prediction Thresholds (4 constants)
```python
STRONG_SPREAD_EDGE_THRESHOLD = 2.0      # Points
MODERATE_SPREAD_EDGE_THRESHOLD = 1.0    # Points
STRONG_TOTAL_EDGE_THRESHOLD = 3.0       # Points
MODERATE_TOTAL_EDGE_THRESHOLD = 2.0     # Points
```

#### 3. Win Probability (6 constants)
```python
SPREAD_TO_PROB_LOGISTIC_FACTOR = 7  # Logistic regression
QB_YARDS_TO_SCORE_INTERCEPT = 10    # Base team score
QB_YARDS_TO_SCORE_SLOPE = 20        # Yards per point
```

#### 4. Colors (14 constants)
```python
COLOR_SUCCESS = "#28a745"       # Green
COLOR_WARNING = "#ffc107"       # Yellow
COLOR_DANGER = "#dc3545"        # Red
COLOR_INFO = "#17a2b8"          # Blue
```

**Usage:**
```python
from dashboard.constants import STRONG_SPREAD_EDGE_THRESHOLD, COLOR_SUCCESS

if abs(edge) >= STRONG_SPREAD_EDGE_THRESHOLD:
    st.markdown(f"<span style='color: {COLOR_SUCCESS};'>Strong Edge!</span>")
```

**To Modify:**
1. Edit `dashboard/constants.py`
2. Change single constant value
3. All dashboard pages automatically use new value
4. No code changes needed elsewhere

---

## Query Architecture

### Query Class Structure

**5 Query Classes | 43 Methods Total**

#### 1. PredictionQueries (6 methods)
```python
from dashboard.queries.predictions import PredictionQueries

# Fetch retrospective metrics
df = PredictionQueries.retrospective_metrics(
    conn,
    model_version="2.0",  # Optional
    season=2024            # Optional
)

# Weekly performance
df = PredictionQueries.weekly_performance_metrics(conn)

# BNN uncertainty
df = PredictionQueries.bnn_uncertainty_metrics(conn)

# Model comparison
df = PredictionQueries.model_comparison(conn, season=2024)

# Predictions with odds
df = PredictionQueries.predictions_with_odds(conn, season=2024, week=12)

# Line movement
df = PredictionQueries.line_movement_history(conn, player_name="Patrick Mahomes")
```

#### 2. GameQueries (7 methods)
```python
from dashboard.queries.games import GameQueries

# Current week games
df = GameQueries.current_week_games(conn, season=2024, week=12)

# Teams without games (bye weeks)
teams = GameQueries.teams_without_games(conn, season=2024, week=12)

# Season schedule
df = GameQueries.season_schedule(conn, season=2024)

# Games by team
df = GameQueries.games_by_team(conn, team="KC", season=2024)
```

#### 3. ModelQueries (9 methods)
```python
from dashboard.queries.models import ModelQueries

# Model registry
df = ModelQueries.model_registry(conn)

# Training runs
df = ModelQueries.training_runs(conn, model_id=5)

# Ensemble components
df = ModelQueries.ensemble_components(conn, ensemble_version="v6.1")
```

#### 4. PaperTradingQueries (9 methods)
```python
from dashboard.queries.paper_trading import PaperTradingQueries

# Active sessions
df = PaperTradingQueries.active_sessions(conn)

# Session trades
df = PaperTradingQueries.session_trades(conn, session_id=1)

# Performance by prop type
df = PaperTradingQueries.performance_by_prop_type(conn)
```

#### 5. SystemQueries (12 methods)
```python
from dashboard.queries.system import SystemQueries

# Database statistics
stats = SystemQueries.database_statistics(conn)

# Table row counts
df = SystemQueries.table_row_counts(conn)

# Slow queries
df = SystemQueries.slow_queries_log(conn, min_duration_ms=100)
```

### Query Method Pattern

**Standard Template:**
```python
@staticmethod
def method_name(
    conn,                          # Connection (required)
    param1: Optional[Type] = None  # Optional filter
) -> pd.DataFrame:
    """
    Brief description.

    Args:
        conn: Database connection from pool
        param1: Description of parameter

    Returns:
        DataFrame with columns: col1, col2, col3
        Empty DataFrame if no results
    """
    query = """
    SELECT
        explicit_column_1,
        explicit_column_2
    FROM schema.table
    WHERE condition = %s
    """

    params = [param1] if param1 else []
    return pd.read_sql_query(query, conn, params=params if params else None)
```

---

## API Integration

### APIClient (`dashboard/utils/api_client.py`)

**Purpose:** Type-safe FastAPI communication with validation

#### Initialization
```python
from dashboard.utils.api_client import APIClient

api = APIClient(base_url="http://localhost:8000")
```

#### Methods

**1. Health Check**
```python
health = api.get_health()
if health and health['status'] == 'healthy':
    st.success("API healthy")
```

**2. List Models**
```python
models = api.get_models()
if models:
    st.write(f"{models['loaded_models']}/{models['total_models']} models loaded")
```

**3. Player Prediction**
```python
pred = api.get_player_prediction(
    game_id="2024_12_KC_BUF",
    player_id="00-0033873",
    stat_type="passing_yards"
)

if pred:
    st.metric("Prediction", f"{pred['prediction']:.1f} yards")
```

**4. Game Prediction** (Main Method)
```python
pred = api.get_game_prediction(
    game_id="2024_12_KC_BUF",
    season=2024,
    week=12
)

if pred:
    st.write(f"Spread: {pred['predicted_spread']:+.1f}")
    st.write(f"Total: {pred['predicted_total']:.1f}")
    st.write(f"Home Win Prob: {pred['predicted_home_win_prob']:.1%}")
```

#### Response Validation

All responses are validated via TypedDict:

```python
class GamePrediction(TypedDict):
    game_id: str
    home_team: str
    away_team: str
    home_qb_yards: float
    predicted_spread: float
    predicted_total: float
    predicted_home_win_prob: float
```

**Validation Checks:**
- ✅ Required fields exist
- ✅ Numeric types valid
- ✅ Ranges sensible (e.g., yards 0-600)
- ✅ Bounds ordered correctly

**Error Handling:**
```python
prediction = api.get_game_prediction(...)
if prediction is None:
    st.warning("Could not load prediction - API may be down")
    return  # Graceful degradation
```

---

## Making Updates

### Adding a New Page

**1. Create Page File**
```bash
touch dashboard/pages/new_feature.py
```

**2. Implement Page Structure**
```python
import streamlit as st
from dashboard.queries.predictions import PredictionQueries
from dashboard.constants import REDIS_CACHE_TTL_MEDIUM

st.set_page_config(page_title="New Feature", layout="wide")
st.title("🎯 New Feature")

@st.cache_data(ttl=REDIS_CACHE_TTL_MEDIUM)
def fetch_data():
    pool = get_db_pool()
    with pool.get_connection() as conn:
        return PredictionQueries.retrospective_metrics(conn)

df = fetch_data()
st.dataframe(df)
```

**3. Add to Navigation**

In `dashboard/app.py`:
```python
page = st.sidebar.radio(
    "Select Page",
    ["Predictions", "New Feature", "System Status", ...],  # Add here
    index=0
)

if page == "New Feature":
    from dashboard.pages import new_feature
    new_feature.main()
    return
```

---

### Adding a New Query

**1. Choose Query Class**

Determine which class fits your query:
- Predictions → `predictions.py`
- Games → `games.py`
- Models → `models.py`
- Paper Trading → `paper_trading.py`
- System Metadata → `system.py`

**2. Add Method to Class**

In `dashboard/queries/<file>.py`:
```python
@staticmethod
def new_query_method(
    conn,
    param1: Optional[int] = None
) -> pd.DataFrame:
    """
    Description of what this query returns.

    Args:
        conn: Database connection
        param1: Filter parameter (optional)

    Returns:
        DataFrame with columns: col1, col2, col3
    """
    query = """
    SELECT
        col1,
        col2,
        col3
    FROM schema.table
    WHERE condition = %s
    """

    params = [param1] if param1 else []
    return pd.read_sql_query(query, conn, params=params if params else None)
```

**3. Use in Dashboard**

In `dashboard/app.py`:
```python
from dashboard.queries.predictions import PredictionQueries

@cache_data(ttl=300)
def fetch_new_data(param1=None):
    pool = get_db_pool()
    with pool.get_connection() as conn:
        return PredictionQueries.new_query_method(conn, param1)

df = fetch_new_data(param1=123)
st.dataframe(df)
```

---

### Adding a New Component

**1. Add to Component Library**

In `dashboard/components/cards.py`:
```python
def new_component_card(
    data: dict,
    config: dict
) -> None:
    """
    Renders a new component card.

    Args:
        data: Data to display
        config: Configuration options
    """
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Metric 1", value=data['value1'])

        with col2:
            st.metric(label="Metric 2", value=data['value2'])
```

**2. Export from Module**

In `dashboard/components/__init__.py`:
```python
from .cards import (
    service_health_card,
    game_prediction_card,
    new_component_card  # Add here
)

__all__ = [
    'service_health_card',
    'game_prediction_card',
    'new_component_card'  # Add here
]
```

**3. Use in Dashboard**

```python
from dashboard.components import new_component_card

new_component_card(data=my_data, config=my_config)
```

---

### Adding a New Constant

**1. Add to Constants Module**

In `dashboard/constants.py`:
```python
# === New Feature Configuration ===
NEW_FEATURE_THRESHOLD = 10  # Description of what this controls
NEW_FEATURE_COLOR = "#ff6b6b"  # Red for alerts
```

**2. Use in Code**

```python
from dashboard.constants import NEW_FEATURE_THRESHOLD, NEW_FEATURE_COLOR

if value > NEW_FEATURE_THRESHOLD:
    st.markdown(f"<span style='color: {NEW_FEATURE_COLOR};'>Alert!</span>")
```

**Benefits:**
- ✅ Single source of truth
- ✅ Easy A/B testing (change one file)
- ✅ Consistent across all pages

---

### Modifying Database Schema

**When schema changes occur:**

**1. Update Query Layer**

If columns added/removed/renamed:
```python
# OLD
query = """SELECT old_column FROM table"""

# NEW
query = """SELECT new_column FROM table"""
```

**2. Update Type Hints (if applicable)**

```python
# OLD
def fetch_data() -> pd.DataFrame:
    # Returns: old_column

# NEW
def fetch_data() -> pd.DataFrame:
    # Returns: new_column
```

**3. Update Documentation**

Update docstring with new schema:
```python
"""
Returns:
    DataFrame with columns: new_column, col2, col3  # Updated
"""
```

**4. Run Schema Validator**

```bash
./scripts/sync_frontend.sh validate-schema
```

This checks all SQL queries against current schema.

---

## Common Tasks

### Task 1: Change Edge Thresholds

**Goal:** Adjust what counts as "strong edge"

**Steps:**
1. Edit `dashboard/constants.py`:
```python
# Change from 2.0 to 3.0
STRONG_SPREAD_EDGE_THRESHOLD = 3.0
```

2. Restart dashboard
```bash
docker restart streamlit
# OR
streamlit run dashboard/app.py
```

3. Verify: Strong edges now require ≥3 points

**Time:** 2 minutes

---

### Task 2: Add New Visualization

**Goal:** Create a new chart for calibration

**Steps:**

1. Add visualization function to `dashboard/app.py`:
```python
def plot_new_calibration_chart(df: pd.DataFrame) -> go.Figure:
    """
    New calibration visualization.

    Args:
        df: DataFrame with columns: predicted, actual

    Returns:
        Plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['predicted'],
        y=df['actual'],
        mode='markers'
    ))
    fig.update_layout(
        title="New Calibration Chart",
        height=CHART_HEIGHT_DEFAULT  # From constants
    )
    return fig
```

2. Use in page:
```python
df = fetch_calibration_data()
fig = plot_new_calibration_chart(df)
st.plotly_chart(fig, use_container_width=True)
```

**Time:** 15-30 minutes

---

### Task 3: Update API Endpoint

**Goal:** Change API base URL

**Steps:**

1. Update environment variable:
```bash
export FASTAPI_URL="http://new-server:8000"
```

2. Restart dashboard (picks up new URL automatically)

**OR** hardcode in `dashboard/app.py`:
```python
api = APIClient(base_url="http://new-server:8000")
```

**Time:** 1 minute

---

### Task 4: Add New Filter to Page

**Goal:** Add season filter to predictions page

**Steps:**

1. Add sidebar filter:
```python
season = st.sidebar.selectbox("Season", [2024, 2023, 2022], index=0)
```

2. Pass to query:
```python
df = fetch_retrospective_metrics(season=season)  # Add season parameter
```

3. Update query method signature (if needed):
```python
@staticmethod
def retrospective_metrics(
    conn,
    model_version: Optional[str] = None,
    season: Optional[int] = None  # Add this
) -> pd.DataFrame:
```

**Time:** 5-10 minutes

---

### Task 5: Change Cache TTL

**Goal:** Make live game data refresh faster

**Steps:**

1. Edit `dashboard/constants.py`:
```python
# Change from 30 to 10 seconds
REDIS_CACHE_TTL_LIVE = 10
```

2. Update cache decorator:
```python
@cache_data(ttl=REDIS_CACHE_TTL_LIVE)  # Now uses 10s
def fetch_live_games():
    ...
```

3. Restart dashboard

**Time:** 2 minutes

---

## Troubleshooting

### Issue: Dashboard won't load

**Symptoms:** Blank page, "Please wait..." forever

**Check:**
1. **Database connection:**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"
```

2. **Environment variables:**
```bash
echo $DB_PASSWORD  # Should be set
```

3. **Connection pool:**
```python
# Add to app.py temporarily
pool = get_db_pool()
st.write(f"Pool status: {pool}")
```

**Fix:**
- Set `DB_PASSWORD` environment variable
- Check PostgreSQL is running: `docker ps | grep postgres`
- Restart dashboard

---

### Issue: API calls failing

**Symptoms:** "Could not load prediction" warnings

**Check:**
1. **API health:**
```bash
curl http://localhost:8000/health
```

2. **Network:**
```bash
telnet localhost 8000
```

3. **Logs:**
```bash
docker logs streamlit  # Dashboard logs
docker logs fastapi    # API logs
```

**Fix:**
- Start FastAPI: `docker restart fastapi`
- Check API URL in code: `api = APIClient("http://localhost:8000")`
- Verify API key (if auth enabled)

---

### Issue: Queries timing out

**Symptoms:** "Database error" after 10+ seconds

**Check:**
1. **Query performance:**
```sql
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
```

2. **Connection pool:**
```python
stats = pool.get_stats()
st.write(stats)  # Check for connection exhaustion
```

**Fix:**
- Add indexes to slow queries
- Increase pool size: `max_size=20`
- Add `LIMIT` to large queries
- Check for N+1 query pattern

---

### Issue: Stale data showing

**Symptoms:** Data doesn't update, old values persist

**Check:**
1. **Cache status:**
```python
st.cache_data.clear()  # Manual cache clear
```

2. **TTL configuration:**
```python
@cache_data(ttl=300)  # Verify TTL is reasonable
```

**Fix:**
- Click "Refresh" button (if available)
- Restart dashboard (clears all caches)
- Reduce cache TTL for dynamic data
- Check Redis connection: `redis-cli ping`

---

### Issue: Type errors in logs

**Symptoms:** TypeError, AttributeError in logs

**Check:**
1. **Type hints match:**
```python
def func(param: int) -> pd.DataFrame:  # Verify correct types
```

2. **API response structure:**
```python
# Add debug logging
st.write(prediction)  # Inspect actual structure
```

**Fix:**
- Update type hints to match actual types
- Add validation before using API responses
- Check APIClient TypedDict definitions

---

## Performance Optimization

### Current Performance Metrics

**Dashboard Load Time:**
- Initial load: 1.5-2.5s (with pooling)
- Page navigation: 0.5-1.0s (cached)
- Chart render: 0.2-0.5s

**Query Performance:**
- Small queries (<1000 rows): 20-50ms
- Large queries (10k+ rows): 100-300ms
- Aggregations: 50-150ms

**Cache Hit Rates:**
- Redis: 70-85%
- Streamlit: 90-95%

### Optimization Checklist

✅ **Connection Pooling** - DONE (21 queries pooled)
✅ **Query Centralization** - DONE (43 methods)
✅ **Type Hints** - DONE (81% coverage)
✅ **Constants Extraction** - DONE (68 constants)
✅ **Component Reuse** - DONE (8 components)

**Future Optimizations:**
- [ ] Implement query result pagination
- [ ] Add materialized views for expensive aggregations
- [ ] Lazy loading for charts (render on scroll)
- [ ] WebSocket for live game updates

---

## Testing Strategy

### Manual Testing Checklist

Before deploying dashboard changes:

**1. Smoke Test (5 min)**
- [ ] Dashboard loads without errors
- [ ] All pages accessible via navigation
- [ ] Database connection working
- [ ] API calls successful

**2. Functional Test (10 min)**
- [ ] Filters work correctly
- [ ] Charts render properly
- [ ] Data updates when expected
- [ ] No console errors

**3. Performance Test (5 min)**
- [ ] Page load <3 seconds
- [ ] Cache hit rate >70%
- [ ] No memory leaks (refresh 10x)

### Validation Tools

**1. Schema Validator**
```bash
./scripts/sync_frontend.sh validate-schema
```

**2. Connection Pool Stats**
```python
stats = get_db_pool().get_stats()
st.write(f"Connections: {stats['size']}/{stats['max_size']}")
```

**3. Query Profiler**
```python
import time
start = time.time()
df = fetch_data()
st.write(f"Query time: {time.time() - start:.2f}s")
```

---

## Security Considerations

### Current Security Posture

**✅ Implemented:**
- Connection pooling (no credential exposure)
- Environment variables for passwords
- API response validation
- Graceful error handling (no stack traces to user)

**⚠️ Local Dev Environment:**
- No authentication required (Streamlit default)
- Hardcoded localhost URLs
- Full database access

**Production Recommendations:**
1. Add Streamlit authentication
2. Use secrets manager (not env vars)
3. Implement rate limiting
4. Add audit logging
5. Use read-only database user

---

## Migration Guide

### From Old to New Architecture

**Before (Problematic):**
```python
# ❌ Hardcoded credentials
"password": "sicillionbillions"

# ❌ New connection every query
with psycopg.connect(**DB_CONFIG) as conn:
    df = pd.read_sql_query(query, conn)

# ❌ Raw HTML
st.markdown("<div style='...'>...</div>", unsafe_allow_html=True)

# ❌ Magic numbers
if edge > 2.0:  # What does 2.0 mean?

# ❌ Unvalidated API calls
data = response.json()
home_yards = data['home_prediction']['predicted_passing_yards']  # KeyError!
```

**After (Best Practice):**
```python
# ✅ Environment variables
"password": os.getenv("DB_PASSWORD")

# ✅ Connection pooling
pool = get_db_pool()
with pool.get_connection() as conn:
    df = PredictionQueries.retrospective_metrics(conn)

# ✅ Native components
game_prediction_card(game, prediction, spread_edge, total_edge)

# ✅ Named constants
if edge > STRONG_SPREAD_EDGE_THRESHOLD:

# ✅ Validated API client
prediction = api.get_game_prediction(game_id, season, week)
if prediction is None:
    return  # Graceful degradation
```

---

## Quick Reference

### File → Purpose Map

| File | Purpose | Update When |
|------|---------|-------------|
| `dashboard/app.py` | Main dashboard | Adding pages, charts, business logic |
| `dashboard/constants.py` | Configuration | Changing thresholds, colors, TTLs |
| `dashboard/queries/*.py` | SQL queries | Schema changes, new data needs |
| `dashboard/components/*.py` | UI components | New reusable UI elements |
| `dashboard/utils/api_client.py` | API communication | API contract changes |
| `py/utils/psycopg_pool.py` | Connection pooling | Database config changes |

### Common Imports

```python
# Queries
from dashboard.queries.predictions import PredictionQueries
from dashboard.queries.games import GameQueries
from dashboard.queries.models import ModelQueries

# Components
from dashboard.components import game_prediction_card, service_health_card

# Constants
from dashboard.constants import (
    STRONG_SPREAD_EDGE_THRESHOLD,
    REDIS_CACHE_TTL_MEDIUM,
    COLOR_SUCCESS
)

# API
from dashboard.utils.api_client import APIClient

# Connection Pool
from py.utils.psycopg_pool import get_connection_pool
```

### Critical Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `main()` | app.py:3400+ | Entry point, page router |
| `get_db_pool()` | app.py:150+ | Get connection pool |
| `fetch_retrospective_metrics()` | app.py:173+ | Main predictions query |
| `game_prediction_card()` | components/cards.py | Game display |
| `PredictionQueries.retrospective_metrics()` | queries/predictions.py | SQL query |

---

## Appendix

### A. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5544 | PostgreSQL port |
| `DB_NAME` | devdb01 | Database name |
| `DB_USER` | dro | Database user |
| `DB_PASSWORD` | (required) | Database password |
| `FASTAPI_URL` | http://localhost:8000 | API base URL |
| `REDIS_URL` | redis://localhost:6379 | Redis connection |

### B. Key Metrics

**Code Statistics:**
- Total dashboard code: ~5,500 lines
- Query layer: 1,695 lines (43 methods)
- Components: 287 lines (8 components)
- Constants: 68 constants
- Type hint coverage: 81% (40/49 functions)

**Performance:**
- Connection pool: 2-10 connections
- Cache hit rate: 70-85% (Redis), 90-95% (Streamlit)
- Average query time: 50ms (pooled), 150ms (non-pooled)
- Dashboard load time: 1.5-2.5s

### C. Dependencies

**Required:**
- streamlit >= 1.28
- psycopg >= 3.0
- pandas >= 1.5
- plotly >= 5.0
- requests >= 2.28

**Optional:**
- redis >= 4.5 (for caching)

### D. Related Documentation

- `docs/guides/DASHBOARD_CONNECTION_POOLING.md` - Connection pool guide
- `docs/guides/API_CLIENT_INTEGRATION_GUIDE.md` - API client usage
- `docs/dashboard/REPLACEMENT_GUIDE.md` - Component migration
- `DASHBOARD_QUERIES_QUICK_REFERENCE.md` - Query class reference
- `dashboard/queries/README.md` - Query module guide

---

**End of Architecture Documentation**

For questions or updates, see:
- Architecture documentation: This file
- Component catalog: `dashboard/components/`
- Query reference: `dashboard/queries/README.md`
- Constants reference: `dashboard/constants.py` (inline comments)
