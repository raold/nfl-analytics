# Dashboard Constants - Quick Reference

**File:** `dashboard/constants.py` (380 lines, 68 constants)

## Import Statement

```python
from dashboard.constants import (
    REDIS_CACHE_TTL_LIVE,
    REDIS_CACHE_TTL_SHORT,
    API_TIMEOUT_DEFAULT,
    API_TIMEOUT_LONG,
    SOCKET_TIMEOUT_SECONDS,
    # ... (see file for complete list)
)
```

## Categories & Values

### Cache & Timeout Configuration

| Constant | Value | Use Case |
|----------|-------|----------|
| `REDIS_CACHE_TTL_LIVE` | 30s | Live data (quotes, game status) |
| `REDIS_CACHE_TTL_SHORT` | 300s (5m) | Dynamic data (player props, predictions) |
| `REDIS_CACHE_TTL_MEDIUM` | 1800s (30m) | Stable data (historical stats) |
| `REDIS_CACHE_TTL_LONG` | 3600s (1h) | Very stable (schema info) |
| `API_TIMEOUT_DEFAULT` | 10s | Standard API calls |
| `API_TIMEOUT_LONG` | 15s | Long-running API calls |
| `SOCKET_TIMEOUT_SECONDS` | 5s | Socket connectivity checks |

**Usage:**
```python
@cache_data(ttl=REDIS_CACHE_TTL_SHORT)
def fetch_data():
    return query_database()

response = requests.get(url, timeout=API_TIMEOUT_DEFAULT)
```

### Prediction Thresholds (Edge Detection)

| Constant | Value | Meaning |
|----------|-------|---------|
| `STRONG_SPREAD_EDGE_THRESHOLD` | 2.0 pts | Confident spread recommendation (green) |
| `MODERATE_SPREAD_EDGE_THRESHOLD` | 1.0 pts | Marginal spread recommendation (yellow) |
| `STRONG_TOTAL_EDGE_THRESHOLD` | 3.0 pts | Confident total recommendation (green) |
| `MODERATE_TOTAL_EDGE_THRESHOLD` | 2.0 pts | Marginal total recommendation (yellow) |

**Usage:**
```python
if abs(spread_edge) >= STRONG_SPREAD_EDGE_THRESHOLD:
    color = COLOR_SUCCESS_BG  # Green
elif abs(spread_edge) >= MODERATE_SPREAD_EDGE_THRESHOLD:
    color = COLOR_WARNING_BG  # Yellow
else:
    color = COLOR_NEUTRAL_BG  # Gray
```

### Win Probability & Score Estimation

| Constant | Value | Formula |
|----------|-------|---------|
| `SPREAD_TO_PROB_LOGISTIC_FACTOR` | 7 | `P(win) = 1 / (1 + exp(-spread / 7))` |
| `YARDS_PER_10PCT_WIN_PROB` | 500.0 | `P(win) = 0.5 + (yards_diff / 500)` |
| `MIN_WIN_PROB` | 0.1 | Minimum probability bound |
| `MAX_WIN_PROB` | 0.9 | Maximum probability bound |
| `QB_YARDS_TO_SCORE_INTERCEPT` | 10.0 | Base team score |
| `QB_YARDS_TO_SCORE_SLOPE` | 20.0 | Yards per point |

**Usage:**
```python
# Win probability from spread
home_win_prob = 1 / (1 + math.exp(-spread / SPREAD_TO_PROB_LOGISTIC_FACTOR))

# Win probability from QB yards
home_win_prob = 0.5 + (yards_diff / YARDS_PER_10PCT_WIN_PROB)
home_win_prob = max(MIN_WIN_PROB, min(MAX_WIN_PROB, home_win_prob))

# Estimate team score from QB yards
team_score = QB_YARDS_TO_SCORE_INTERCEPT + (qb_yards / QB_YARDS_TO_SCORE_SLOPE)
```

### Display Configuration

| Constant | Value | Use Case |
|----------|-------|----------|
| `CHART_HEIGHT_DEFAULT` | 400px | Standard charts (line, bar, scatter) |
| `CHART_HEIGHT_COMPACT` | 300px | Sidebar/compact sections |
| `CHART_HEIGHT_MINI` | 200px | Small summary boxes |
| `HISTOGRAM_BINS_DEFAULT` | 30 | Standard histograms |
| `HISTOGRAM_BINS_DENSE` | 40 | High-detail histograms |
| `CALIBRATION_BINS` | 10 | Calibration decile plots |

**Usage:**
```python
fig = go.Figure()
fig.update_layout(height=CHART_HEIGHT_DEFAULT)

df['prob_bin'] = pd.cut(df['prob_win'], bins=CALIBRATION_BINS)
```

### Query Limits & Season Filters

| Constant | Value | Purpose |
|----------|-------|---------|
| `QUERY_LIMIT_DEFAULT` | 1000 | Default SQL LIMIT |
| `QUERY_LIMIT_RECENT` | 100 | Recent records |
| `QUERY_LIMIT_STATS` | 500 | Statistical summaries |
| `MIN_SEASON_FOR_STATS` | 2022 | Stats baseline year |
| `MIN_SEASON_FOR_FULL_HISTORY` | 2023 | Full history baseline |

**Usage:**
```python
query = f"SELECT * FROM games WHERE season >= {MIN_SEASON_FOR_STATS} LIMIT {QUERY_LIMIT_DEFAULT}"
```

### Performance Targets

| Constant | Value | Target/Metric |
|----------|-------|---------------|
| `BREAKEVEN_WIN_RATE` | 0.524 | 52.4% (breakeven at -110 odds) |
| `TARGET_CALIBRATION` | 0.90 | 90% calibration coverage target |
| `STATE_SPACE_MAE` | 47.77 | Model MAE (yards) |
| `STATE_SPACE_COVERAGE` | 0.865 | Calibration coverage |
| `STATE_SPACE_P_VALUE` | 0.0069 | Statistical significance |

**Usage:**
```python
fig.add_hline(y=BREAKEVEN_WIN_RATE * 100, name="Breakeven")
fig.add_hline(y=TARGET_CALIBRATION * 100, name="Target Calibration")
```

### Colors (Bootstrap Compatible)

| Constant | Value | Purpose |
|----------|-------|---------|
| `COLOR_SUCCESS` | #28a745 | Success/profitable (green) |
| `COLOR_WARNING` | #ffc107 | Warning/caution (yellow) |
| `COLOR_NEUTRAL` | #6c757d | Neutral/no status (gray) |
| `COLOR_DANGER` | #dc3545 | Danger/loss (red) |
| `COLOR_PRIMARY` | #1f77b4 | Primary (blue, Plotly) |
| `COLOR_SECONDARY` | #ff7f0e | Secondary (orange, Plotly) |

### Background Colors

| Constant | Value | Use |
|----------|-------|-----|
| `COLOR_SUCCESS_BG` | #d4edda | Profitable/confident background |
| `COLOR_WARNING_BG` | #fff3cd | Marginal/caution background |
| `COLOR_NEUTRAL_BG` | #f8f9fa | Neutral background |
| `COLOR_DANGER_BG` | #f8d7da | Loss/error background |

### Text & Border Colors

| Constant | Value | Use |
|----------|-------|-----|
| `COLOR_TEXT_DARK` | #212529 | Dark text on light backgrounds |
| `COLOR_TEXT_LIGHT` | #ffffff | Light text on dark backgrounds |
| `COLOR_BORDER_LIGHT` | #dee2e6 | Light borders |
| `COLOR_BORDER_DARK` | #495057 | Dark borders |

## Common Patterns

### Edge Detection & Styling

```python
# Determine edge color
if abs(spread_edge) >= STRONG_SPREAD_EDGE_THRESHOLD:
    card_style = f"background-color: {COLOR_SUCCESS_BG}; color: {COLOR_TEXT_DARK};"
elif abs(spread_edge) >= MODERATE_SPREAD_EDGE_THRESHOLD:
    card_style = f"background-color: {COLOR_WARNING_BG}; color: {COLOR_TEXT_DARK};"
else:
    card_style = f"background-color: {COLOR_NEUTRAL_BG}; color: {COLOR_TEXT_DARK};"
```

### Chart Configuration

```python
# Create chart with standard dimensions
fig = go.Figure()
fig.add_trace(go.Scatter(...))
fig.update_layout(
    height=CHART_HEIGHT_DEFAULT,
    title="My Chart",
)

# Add reference lines
fig.add_hline(y=BREAKEVEN_WIN_RATE * 100, line_dash="dash")
fig.add_hline(y=TARGET_CALIBRATION * 100, line_dash="dash")
```

### Cache Configuration

```python
# Cache for 30 seconds (live data)
@cache_data(ttl=REDIS_CACHE_TTL_LIVE)
def get_current_games():
    return fetch_games_api()

# Cache for 5 minutes (prediction data)
@cache_data(ttl=REDIS_CACHE_TTL_SHORT)
def get_predictions(season, week):
    return load_predictions(season, week)
```

### Probability Calculations

```python
# From spread
spread_margin = home_score - away_score - sportsbook_spread
home_win_prob = 1 / (1 + math.exp(-spread_margin / SPREAD_TO_PROB_LOGISTIC_FACTOR))
home_win_prob = max(MIN_WIN_PROB, min(MAX_WIN_PROB, home_win_prob))

# From QB yards
yards_diff = home_qb_yards - away_qb_yards
home_win_prob = 0.5 + (yards_diff / YARDS_PER_10PCT_WIN_PROB)
home_win_prob = max(MIN_WIN_PROB, min(MAX_WIN_PROB, home_win_prob))
```

### Score Estimation

```python
# Estimate team scores from QB yards
home_score = QB_YARDS_TO_SCORE_INTERCEPT + (home_qb_yards / QB_YARDS_TO_SCORE_SLOPE)
away_score = QB_YARDS_TO_SCORE_INTERCEPT + (away_qb_yards / QB_YARDS_TO_SCORE_SLOPE)
predicted_total = home_score + away_score
predicted_spread = home_score - away_score
```

## Modifying Constants

To adjust dashboard behavior, edit one value in `dashboard/constants.py`:

**Make dashboard more sensitive to small edges:**
```python
STRONG_SPREAD_EDGE_THRESHOLD = 1.5  # (was 2.0)
MODERATE_SPREAD_EDGE_THRESHOLD = 0.75  # (was 1.0)
```

**Cache data longer (reduce database load):**
```python
REDIS_CACHE_TTL_SHORT = 600  # (was 300, now 10 minutes)
REDIS_CACHE_TTL_LIVE = 60  # (was 30, now 1 minute)
```

**Adjust win probability model:**
```python
SPREAD_TO_PROB_LOGISTIC_FACTOR = 6  # (was 7, more sensitive)
YARDS_PER_10PCT_WIN_PROB = 450  # (was 500, higher sensitivity)
```

## File Statistics

- **Total Constants:** 68
- **Categories:** 11
- **Lines of Code:** 380
- **Documentation:** Comprehensive docstrings for each constant
- **Usage Examples:** See patterns section above

## Related Files

- **Dashboard Code:** `dashboard/app.py`, `dashboard/game_predictions_patch.py`, `dashboard/pages/paper_trading.py`
- **Documentation:** `docs/DASHBOARD_CONSTANTS_MIGRATION.md` (complete migration log)
- **Test Coverage:** All constants validated for syntax and type correctness

## Author & Date

- **Author:** Claude Code
- **Date:** November 22, 2025
- **Status:** ✅ Production Ready
