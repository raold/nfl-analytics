# Dashboard Refactoring Plan
**TICKET-6: Refactor Dashboard Monolithic App.py**

**Current State:** 2,733-line monolithic file  
**Target State:** Modular page-based architecture  
**Estimated Effort:** 1 week  
**Risk:** LOW - Mechanical refactoring

---

## 1. Current Structure Analysis

### 1.1 Current File Organization

```
dashboard/
├── app.py (2,733 lines) ← NEEDS REFACTORING
├── pages/
│   ├── game_predictions.py (exists, but not used by main app)
│   ├── player_props.py (exists, but not used by main app)
│   └── paper_trading.py (exists, but not used by main app)
├── components/
│   ├── cards.py
│   ├── charts.py
│   ├── metrics.py
│   └── tables.py
├── queries/
│   ├── games.py (356 lines)
│   ├── predictions.py (407 lines)
│   ├── predictions_batched.py (363 lines)
│   ├── paper_trading.py (396 lines)
│   ├── models.py (319 lines)
│   └── system.py (349 lines)
└── utils/
    ├── api_client.py (660 lines)
    ├── api_client_batched.py (254 lines)
    ├── db.py (140 lines)
    └── schema_validator.py (375 lines)
```

### 1.2 Functions in app.py

**Main render functions identified:**
- `render_game_predictions_page()` - Game predictions view
- `render_system_status_page()` - System health monitoring
- `render_model_status_page()` - Model performance metrics
- `render_data_summary_page()` - Data quality dashboard
- Navigation sidebar logic
- Session state management
- Cache initialization

---

## 2. Target Architecture

### 2.1 Proposed Structure

```
dashboard/
├── app.py (~150 lines)              ← Main entry point only
│   ├── Page configuration
│   ├── Navigation sidebar
│   ├── Page routing
│   └── Session initialization
│
├── pages/
│   ├── __init__.py                  ← Page registry
│   ├── game_predictions.py          ← Move render_game_predictions_page()
│   ├── player_props.py              ← Already exists
│   ├── paper_trading.py             ← Already exists
│   ├── system_status.py (NEW)       ← Move render_system_status_page()
│   ├── model_status.py (NEW)        ← Move render_model_status_page()
│   └── data_summary.py (NEW)        ← Move render_data_summary_page()
│
├── components/                       ← Reusable UI components
│   ├── __init__.py
│   ├── cards.py
│   ├── charts.py
│   ├── metrics.py
│   ├── tables.py
│   └── navigation.py (NEW)          ← Extract sidebar logic
│
├── utils/
│   ├── __init__.py
│   ├── api_client.py
│   ├── api_client_batched.py
│   ├── db.py
│   ├── schema_validator.py
│   ├── session.py (NEW)             ← Session state management
│   └── ui_helpers.py (NEW)          ← Shared plotting/formatting
│
└── config/
    └── pages.py (NEW)               ← Page configuration & metadata
```

---

## 3. Step-by-Step Refactoring Plan

### Phase 1: Preparation (Day 1)

#### Step 1.1: Create New Files
```bash
# Create new files
touch dashboard/pages/__init__.py
touch dashboard/pages/system_status.py
touch dashboard/pages/model_status.py
touch dashboard/pages/data_summary.py
touch dashboard/components/navigation.py
touch dashboard/utils/session.py
touch dashboard/utils/ui_helpers.py
touch dashboard/config/pages.py
```

#### Step 1.2: Add Base Page Template
Create `dashboard/pages/__init__.py`:
```python
"""
Dashboard pages module.

Each page should implement:
- render() function as main entry point
- Page-specific helper functions
- Import dependencies at module level
"""

from typing import Protocol

class DashboardPage(Protocol):
    """Protocol for dashboard pages."""
    
    def render(self) -> None:
        """Render the page content."""
        ...

# Page registry
PAGES = {
    "game_predictions": "Game Predictions",
    "player_props": "Player Props", 
    "paper_trading": "Paper Trading",
    "system_status": "System Status",
    "model_status": "Model Performance",
    "data_summary": "Data Quality"
}
```

### Phase 2: Extract Navigation (Day 1-2)

#### Step 2.1: Create navigation.py
```python
# dashboard/components/navigation.py
import streamlit as st
from dashboard.pages import PAGES

def render_sidebar() -> str:
    """
    Render navigation sidebar.
    
    Returns:
        str: Selected page key
    """
    st.sidebar.title("NFL Analytics Dashboard")
    
    # Page selection
    page = st.sidebar.radio(
        "Navigation",
        options=list(PAGES.keys()),
        format_func=lambda x: PAGES[x]
    )
    
    # System info
    st.sidebar.divider()
    st.sidebar.caption("System Status")
    
    # Add health indicators here
    
    return page
```

#### Step 2.2: Update app.py to use navigation
```python
# dashboard/app.py (new simplified version)
import streamlit as st
from dashboard.components.navigation import render_sidebar
from dashboard.utils.session import initialize_session
from dashboard.pages import game_predictions, player_props, paper_trading
from dashboard.pages import system_status, model_status, data_summary

# Page configuration
st.set_page_config(
    page_title="NFL Analytics Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session()

# Render navigation and get selected page
page = render_sidebar()

# Route to appropriate page
if page == "game_predictions":
    game_predictions.render()
elif page == "player_props":
    player_props.render()
elif page == "paper_trading":
    paper_trading.render()
elif page == "system_status":
    system_status.render()
elif page == "model_status":
    model_status.render()
elif page == "data_summary":
    data_summary.render()
else:
    st.error(f"Unknown page: {page}")
```

### Phase 3: Extract Session Management (Day 2)

#### Step 3.1: Create session.py
```python
# dashboard/utils/session.py
import streamlit as st
from typing import Any, Dict

DEFAULT_STATE = {
    "db_connection": None,
    "api_client": None,
    "cache_initialized": False,
    "last_refresh": None,
    "user_preferences": {}
}

def initialize_session() -> None:
    """Initialize session state with defaults."""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_session_value(key: str, default: Any = None) -> Any:
    """Get value from session state."""
    return st.session_state.get(key, default)

def set_session_value(key: str, value: Any) -> None:
    """Set value in session state."""
    st.session_state[key] = value
```

### Phase 4: Extract UI Helpers (Day 2-3)

#### Step 4.1: Create ui_helpers.py
```python
# dashboard/utils/ui_helpers.py
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Any

def render_metric_card(
    title: str,
    value: Any,
    delta: Any = None,
    help_text: str = None
) -> None:
    """Render a metric card with optional delta."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )

def render_status_indicator(
    status: str,
    label: str = None
) -> None:
    """Render colored status indicator."""
    colors = {
        "healthy": "🟢",
        "warning": "🟡",
        "error": "🔴",
        "unknown": "⚪"
    }
    icon = colors.get(status.lower(), "⚪")
    text = f"{icon} {label or status.title()}"
    st.markdown(text)

def create_line_chart(
    data: Dict[str, List],
    x_col: str,
    y_col: str,
    title: str = None
) -> go.Figure:
    """Create standardized line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        name=y_col
    ))
    if title:
        fig.update_layout(title=title)
    return fig
```

### Phase 5: Extract Game Predictions Page (Day 3-4)

#### Step 5.1: Update game_predictions.py
```python
# dashboard/pages/game_predictions.py
import streamlit as st
import pandas as pd
from datetime import datetime
from dashboard.queries import games, predictions
from dashboard.utils.ui_helpers import render_metric_card, render_status_indicator
from dashboard.components import charts, tables

def render() -> None:
    """Render game predictions page."""
    st.title("🏈 NFL Game Predictions")
    
    # Week selector
    col1, col2 = st.columns([1, 3])
    with col1:
        season = st.selectbox("Season", [2024, 2023, 2022])
    with col2:
        week = st.selectbox("Week", range(1, 19))
    
    # Load data
    games_data = load_games(season, week)
    predictions_data = load_predictions(games_data)
    
    # Render sections
    render_summary_metrics(predictions_data)
    render_games_table(games_data, predictions_data)
    render_edge_analysis(predictions_data)

def load_games(season: int, week: int) -> pd.DataFrame:
    """Load games for specified week."""
    return games.get_games_by_week(season, week)

def load_predictions(games_df: pd.DataFrame) -> pd.DataFrame:
    """Load predictions for games."""
    return predictions.get_batch_predictions(games_df['game_id'].tolist())

def render_summary_metrics(predictions_df: pd.DataFrame) -> None:
    """Render summary metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            "Total Games",
            len(predictions_df),
            help_text="Games in this week"
        )
    
    with col2:
        edges = predictions_df['edge'].abs()
        render_metric_card(
            "Avg Edge",
            f"{edges.mean():.1f} pts",
            help_text="Average prediction edge vs. market"
        )
    
    # Add more metrics...

def render_games_table(
    games_df: pd.DataFrame,
    predictions_df: pd.DataFrame
) -> None:
    """Render games table with predictions."""
    st.subheader("Game Predictions")
    
    # Merge data
    combined = games_df.merge(predictions_df, on='game_id')
    
    # Format for display
    display_df = format_games_table(combined)
    
    # Render table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def render_edge_analysis(predictions_df: pd.DataFrame) -> None:
    """Render edge analysis charts."""
    st.subheader("Edge Analysis")
    
    # Create edge distribution chart
    fig = charts.create_edge_distribution(predictions_df)
    st.plotly_chart(fig, use_container_width=True)

def format_games_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format games table for display."""
    # Formatting logic here
    return df
```

### Phase 6: Extract System Status Page (Day 4)

#### Step 6.1: Create system_status.py
```python
# dashboard/pages/system_status.py
import streamlit as st
from dashboard.queries import system
from dashboard.utils.ui_helpers import render_status_indicator

def render() -> None:
    """Render system status page."""
    st.title("⚙️ System Status")
    
    # Health checks
    health_data = system.get_health_status()
    render_health_checks(health_data)
    
    # Database stats
    db_stats = system.get_database_stats()
    render_database_stats(db_stats)
    
    # API metrics
    api_metrics = system.get_api_metrics()
    render_api_metrics(api_metrics)

def render_health_checks(health_data: dict) -> None:
    """Render health check indicators."""
    st.subheader("Health Checks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_status_indicator(
            health_data['database'],
            "Database"
        )
    
    with col2:
        render_status_indicator(
            health_data['redis'],
            "Redis Cache"
        )
    
    with col3:
        render_status_indicator(
            health_data['api'],
            "API"
        )

# Additional functions...
```

### Phase 7: Extract Model Status Page (Day 4-5)

#### Step 7.1: Create model_status.py
```python
# dashboard/pages/model_status.py
import streamlit as st
from dashboard.queries import models as model_queries
from dashboard.components import charts

def render() -> None:
    """Render model status page."""
    st.title("📊 Model Performance")
    
    # Model registry
    models = model_queries.get_approved_models()
    render_model_registry(models)
    
    # Performance metrics
    performance = model_queries.get_performance_metrics()
    render_performance_metrics(performance)
    
    # Ensemble weights
    weights = model_queries.get_ensemble_weights()
    render_ensemble_weights(weights)

def render_model_registry(models: list) -> None:
    """Render model registry table."""
    st.subheader("Approved Models")
    
    # Format and display
    st.table(models)

def render_performance_metrics(performance: dict) -> None:
    """Render performance metrics."""
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ATS Coverage", f"{performance['ats_coverage']:.1%}")
    
    with col2:
        st.metric("MAE", f"{performance['mae']:.2f} pts")
    
    with col3:
        st.metric("ROI", f"{performance['roi']:.1%}")

def render_ensemble_weights(weights: dict) -> None:
    """Render ensemble weight visualization."""
    st.subheader("Ensemble Weights")
    
    fig = charts.create_weight_pie_chart(weights)
    st.plotly_chart(fig, use_container_width=True)
```

### Phase 8: Extract Data Summary Page (Day 5)

#### Step 8.1: Create data_summary.py
```python
# dashboard/pages/data_summary.py
import streamlit as st
from dashboard.queries import games, system

def render() -> None:
    """Render data summary page."""
    st.title("📈 Data Quality")
    
    # Data coverage
    coverage = system.get_data_coverage()
    render_data_coverage(coverage)
    
    # Data freshness
    freshness = system.get_data_freshness()
    render_data_freshness(freshness)
    
    # Data quality metrics
    quality = system.get_data_quality_metrics()
    render_quality_metrics(quality)

# Implementation functions...
```

---

## 4. Testing Strategy

### 4.1 Unit Tests

Create `tests/unit/test_dashboard_pages.py`:
```python
import pytest
from unittest.mock import Mock, patch
from dashboard.pages import game_predictions, system_status

def test_game_predictions_render():
    """Test game predictions page renders without errors."""
    with patch('streamlit.title'):
        with patch('streamlit.selectbox', return_value=1):
            # Mock data loading
            with patch.object(game_predictions, 'load_games', return_value=Mock()):
                game_predictions.render()

def test_system_status_render():
    """Test system status page renders without errors."""
    with patch('streamlit.title'):
        with patch.object(system_status, 'render_health_checks'):
            system_status.render()
```

### 4.2 Integration Tests

Create `tests/integration/test_dashboard_navigation.py`:
```python
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

@pytest.fixture
def browser():
    driver = webdriver.Chrome()
    yield driver
    driver.quit()

def test_page_navigation(browser):
    """Test navigation between pages works."""
    browser.get("http://localhost:8501")
    time.sleep(2)
    
    # Click on different pages
    pages = ["Game Predictions", "System Status", "Model Performance"]
    for page in pages:
        link = browser.find_element(By.LINK_TEXT, page)
        link.click()
        time.sleep(1)
        assert page in browser.title
```

### 4.3 Manual Testing Checklist

- [ ] All pages render without errors
- [ ] Navigation between pages works
- [ ] Session state persists across page changes
- [ ] Cached data loads correctly
- [ ] Real-time updates work (30s refresh)
- [ ] Error handling displays properly
- [ ] Responsive layout works on different screen sizes

---

## 5. Migration Checklist

### Pre-Migration
- [ ] Create git branch: `feature/dashboard-refactor`
- [ ] Backup current app.py: `cp app.py app_backup_pre_refactor.py`
- [ ] Run full test suite on current version
- [ ] Document any existing bugs

### Phase 1: Preparation
- [ ] Create all new files (Step 1.1)
- [ ] Add base page template (Step 1.2)
- [ ] Commit: "Add new page structure files"

### Phase 2: Navigation
- [ ] Extract navigation to components/navigation.py
- [ ] Update app.py to use new navigation
- [ ] Test: Navigation sidebar renders
- [ ] Commit: "Extract navigation component"

### Phase 3: Session
- [ ] Extract session management to utils/session.py
- [ ] Update all references in app.py
- [ ] Test: Session state works correctly
- [ ] Commit: "Extract session management"

### Phase 4: UI Helpers
- [ ] Create utils/ui_helpers.py with common functions
- [ ] Test: Helper functions work independently
- [ ] Commit: "Add UI helper utilities"

### Phase 5: Game Predictions
- [ ] Move render_game_predictions_page() to pages/game_predictions.py
- [ ] Add render() function as entry point
- [ ] Update imports in app.py
- [ ] Test: Game predictions page works
- [ ] Commit: "Refactor game predictions page"

### Phase 6: System Status
- [ ] Move render_system_status_page() to pages/system_status.py
- [ ] Update imports in app.py
- [ ] Test: System status page works
- [ ] Commit: "Refactor system status page"

### Phase 7: Model Status
- [ ] Move render_model_status_page() to pages/model_status.py
- [ ] Update imports in app.py
- [ ] Test: Model status page works
- [ ] Commit: "Refactor model status page"

### Phase 8: Data Summary
- [ ] Move render_data_summary_page() to pages/data_summary.py
- [ ] Update imports in app.py
- [ ] Test: Data summary page works
- [ ] Commit: "Refactor data summary page"

### Post-Migration
- [ ] Run full test suite
- [ ] Manual testing (all pages, all features)
- [ ] Performance testing (load times)
- [ ] Code review
- [ ] Update documentation
- [ ] Merge to main branch
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production

---

## 6. Rollback Plan

### If Issues Arise:

1. **Immediate Rollback:**
   ```bash
   git checkout main
   git revert <refactor-commit-hash>
   ```

2. **Keep Backup:**
   - `app_backup_pre_refactor.py` available
   - Can restore quickly: `cp app_backup_pre_refactor.py app.py`

3. **Gradual Rollback:**
   - Revert specific pages one at a time
   - Keep working pages, roll back problematic ones

---

## 7. Success Metrics

### Code Quality
- [ ] app.py reduced from 2,733 lines to <200 lines (93% reduction)
- [ ] Each page file <500 lines
- [ ] Cyclomatic complexity <10 per function
- [ ] Test coverage >80%

### Performance
- [ ] Page load time <2 seconds
- [ ] Navigation between pages <500ms
- [ ] No regression in dashboard functionality

### Maintainability
- [ ] New developers can add pages easily
- [ ] Pages can be tested independently
- [ ] Reduced git merge conflicts

---

## 8. Timeline

| Day | Tasks | Deliverables |
|-----|-------|-------------|
| 1 | Phases 1-2 | New file structure, navigation extracted |
| 2 | Phases 3-4 | Session management, UI helpers |
| 3 | Phase 5 | Game predictions page refactored |
| 4 | Phases 6-7 | System & model status pages refactored |
| 5 | Phase 8 | Data summary page refactored |
| 6-7 | Testing | All tests passing, manual QA complete |

**Total: 1 week (5-7 days)**

---

## 9. Post-Refactoring Improvements

### Future Enhancements (After Refactoring Complete):

1. **Add Page-Level Caching:**
   ```python
   @st.cache_data(ttl=60)
   def load_games(season: int, week: int):
       # Cached data loading
   ```

2. **Add Loading States:**
   ```python
   with st.spinner("Loading predictions..."):
       predictions = load_predictions()
   ```

3. **Add Error Boundaries:**
   ```python
   try:
       render_games_table()
   except Exception as e:
       st.error(f"Error loading games: {e}")
   ```

4. **Add User Preferences:**
   - Save preferred week/season
   - Custom column visibility
   - Theme selection

---

**Document Version:** 1.0  
**Created:** 2025-12-06  
**Related Ticket:** TICKET-6  
**Estimated Completion:** 2025-12-13
