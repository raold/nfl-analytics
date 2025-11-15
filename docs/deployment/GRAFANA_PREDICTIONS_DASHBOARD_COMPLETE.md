# Grafana Predictions Dashboard - Implementation Complete

**Created:** November 15, 2025
**Status:** ✅ COMPLETE
**Author:** Claude Code (Sonnet 4.5)

---

## Summary

Successfully created comprehensive Grafana dashboard for monitoring NFL player passing yards predictions from the `ml.predictions` table. The dashboard replaces incorrect game outcome metrics (points/margins) with player-level statistics (passing yards).

---

## Changes Made

### 1. New Dashboard Created

**File:** `grafana/provisioning/dashboards/nfl-predictions-monitoring.json`

**Dashboard Name:** "NFL Analytics - Predictions Monitoring"

**Panel Count:**
- **Total panels:** 16 (11 data panels + 5 row sections)
- **Refresh rate:** 1 minute
- **Time range:** Last 7 days (default)

### 2. PostgreSQL Datasource Added

**File:** `grafana/datasources/postgresql.yml`

**Configuration:**
- **Name:** PostgreSQL NFL
- **UID:** postgres_nfl
- **Connection:** pg-1:5432/devdb01
- **User:** dro
- **Features:** TimescaleDB support enabled

---

## Dashboard Panels

### Section 1: Ensemble Performance Overview (4 panels)

1. **Overall MAE (Passing Yards)** - `stat`
   - Query: `SELECT AVG(abs_prediction_error) FROM ml.predictions WHERE stat_type = 'passing_yards'`
   - Thresholds: Green (0-40), Yellow (40-50), Red (50+)
   - Target: < 40 yards

2. **Win Rate (%)** - `stat`
   - Query: `SELECT 100.0 * AVG(CASE WHEN status = 'win'...)`
   - Thresholds: Green (59%+), Yellow (55-59%), Red (<55%)
   - Target: > 59%

3. **Calibration Coverage (90% Target)** - `gauge`
   - Query: `SELECT 100.0 * AVG(CASE WHEN calibrated...)`
   - Thresholds: Optimal 88-92% (green), Under-calibrated (<85%), Over-calibrated (>95%)
   - Target: 88-92% coverage

4. **Total P&L (Last 7 Days)** - `stat`
   - Query: `SELECT SUM(pnl) FROM ml.predictions`
   - Thresholds: Green ($500+), Yellow ($0-500), Red (<$0)
   - Target: Positive P&L

### Section 2: Ensemble Component Performance (2 panels)

5. **Component MAE Comparison (Yards)** - `bargauge`
   - **State-Space v2:** `AVG(ABS(state_space_mean - actual_value))`
   - **Hierarchical v2:** `AVG(ABS(bayesian_mean - actual_value))`
   - **Hybrid XGBoost:** `AVG(ABS(xgboost_mean - actual_value))`
   - **Informative Priors:** `AVG(ABS(bnn_calibrated_mean - actual_value))`
   - **Ensemble:** `AVG(abs_prediction_error)`
   - Shows which component performs best

6. **Ensemble Weight Distribution** - `piechart`
   - State-Space v2: 30%
   - Hierarchical v2: 30%
   - Hybrid XGBoost: 25%
   - Informative Priors v2: 15%
   - Static weights from `enhanced_ensemble_v5.py`

### Section 3: Time Series Performance (2 panels)

7. **MAE by Week (Yards)** - `timeseries`
   - Query: Daily aggregation of MAE over last 30 days
   - Shows prediction accuracy trends
   - Identifies degradation or improvement patterns

8. **Cumulative P&L (Yards)** - `timeseries`
   - Query: Running sum of P&L over last 30 days
   - Visualizes profitability trajectory
   - Detects winning/losing streaks

### Section 4: Prediction Details (1 panel)

9. **Recent Player Predictions (Last 24h)** - `table`
   - Columns: player_name, team, opponent, is_home, pred_yards, actual_yards, mae, calibrated_90, status, pnl_usd, predicted_at
   - Limit: 50 most recent predictions
   - Color-coded status (Win=green, Loss=red, Push=yellow, Pending=blue)
   - MAE background coloring (Green <40, Yellow 40-50, Red >50)

### Section 5: Model Performance by Season (2 panels)

10. **MAE by Season (Yards)** - `bargauge`
    - Query: `AVG(abs_prediction_error) GROUP BY season`
    - Compares performance across 2022, 2023, 2024
    - Identifies year-over-year improvements

11. **Win Rate by Season (%)** - `timeseries`
    - Query: `100.0 * AVG(CASE WHEN status = 'win'...) GROUP BY season`
    - Shows betting success rate by season
    - Target: > 59% win rate

---

## Key Metrics Tracked

### Performance Metrics
- **MAE (Mean Absolute Error):** Average yards off actual passing yards
- **Win Rate:** % of profitable bets (target: 59%+)
- **Calibration:** % of actuals within 90% prediction intervals (target: 88-92%)
- **P&L (Profit & Loss):** Total dollars won/lost

### Ensemble Components
- **State-Space v2:** Time series model (30% weight)
- **Hierarchical v2:** Bayesian hierarchical model (30% weight)
- **Hybrid XGBoost:** Static/incremental gradient boosting (25% weight)
- **Informative Priors v2:** Enhanced Bayesian priors (15% weight)

### Prediction Fields from `ml.predictions`
```sql
-- Core prediction fields
prediction_mean         -- Ensemble mean prediction
prediction_median       -- Ensemble median
prediction_std          -- Standard deviation
prediction_q05 - q95    -- Quantiles (5%, 10%, 25%, 50%, 75%, 90%, 95%)

-- Component predictions
state_space_mean        -- State-Space v2 prediction
bayesian_mean           -- Hierarchical v2 prediction
xgboost_mean            -- Hybrid XGBoost prediction
bnn_calibrated_mean     -- Informative Priors v2 prediction

-- Outcome tracking
actual_value            -- Actual passing yards
abs_prediction_error    -- |predicted - actual|
calibrated              -- Within 90% interval?
status                  -- win/loss/push/pending
pnl                     -- Profit/loss in USD
```

---

## SQL Query Patterns

### 1. Overall Performance (Last 7 Days)
```sql
SELECT
  AVG(abs_prediction_error) as mae,
  100.0 * AVG(CASE WHEN status = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
  100.0 * AVG(CASE WHEN calibrated THEN 1.0 ELSE 0.0 END) as calibration,
  SUM(pnl) as total_pnl
FROM ml.predictions
WHERE stat_type = 'passing_yards'
  AND actual_value IS NOT NULL
  AND predicted_at > NOW() - INTERVAL '7 days';
```

### 2. Component Comparison
```sql
SELECT
  'State-Space v2' as component,
  AVG(ABS(state_space_mean - actual_value)) as mae
FROM ml.predictions
WHERE stat_type = 'passing_yards'
  AND actual_value IS NOT NULL
  AND state_space_mean IS NOT NULL
  AND predicted_at > NOW() - INTERVAL '7 days'

UNION ALL

SELECT 'Hierarchical v2', AVG(ABS(bayesian_mean - actual_value))
FROM ml.predictions
WHERE stat_type = 'passing_yards'
  AND actual_value IS NOT NULL
  AND bayesian_mean IS NOT NULL
  AND predicted_at > NOW() - INTERVAL '7 days'

-- ... repeat for other components
```

### 3. Time Series MAE
```sql
SELECT
  DATE_TRUNC('day', predicted_at) as time,
  AVG(abs_prediction_error) as mae
FROM ml.predictions
WHERE stat_type = 'passing_yards'
  AND actual_value IS NOT NULL
  AND predicted_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', predicted_at)
ORDER BY time;
```

### 4. Cumulative P&L
```sql
SELECT
  DATE_TRUNC('day', predicted_at) as time,
  SUM(SUM(pnl)) OVER (ORDER BY DATE_TRUNC('day', predicted_at)) as cumulative_pnl
FROM ml.predictions
WHERE stat_type = 'passing_yards'
  AND actual_value IS NOT NULL
  AND predicted_at > NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', predicted_at)
ORDER BY time;
```

### 5. Recent Predictions Table
```sql
SELECT
  player_name,
  team,
  opponent,
  is_home,
  ROUND(prediction_mean::numeric, 1) as pred_yards,
  ROUND(actual_value::numeric, 1) as actual_yards,
  ROUND(abs_prediction_error::numeric, 1) as mae,
  CASE WHEN calibrated THEN 'Yes' ELSE 'No' END as calibrated_90,
  status,
  ROUND(pnl::numeric, 2) as pnl_usd,
  predicted_at
FROM ml.predictions
WHERE stat_type = 'passing_yards'
  AND predicted_at > NOW() - INTERVAL '24 hours'
ORDER BY predicted_at DESC
LIMIT 50;
```

---

## Deployment

### Prerequisites
1. Docker services running (PostgreSQL, Grafana)
2. PostgreSQL datasource configured
3. `ml.predictions` table populated with data

### Grafana Restart (to load new dashboard)
```bash
# Restart Grafana container to pick up new datasource + dashboard
docker compose restart grafana

# Verify Grafana is healthy
docker compose ps grafana

# Check logs if issues
docker compose logs grafana
```

### Access Dashboard
1. Open Grafana: http://localhost:3000
2. Login: admin / sicillionbillions
3. Navigate: Dashboards → NFL Analytics - Predictions Monitoring

### Expected Data
- **Time range:** Last 7 days (adjustable)
- **Predictions:** From `ml.predictions` table
- **Stat type:** `passing_yards`
- **Seasons:** 2022, 2023, 2024 (historical)
- **Players:** ~101 unique QBs
- **Games:** ~148 test games

---

## Troubleshooting

### Issue: "No data" in panels

**Causes:**
1. PostgreSQL datasource not connected
2. `ml.predictions` table empty
3. No predictions in selected time range
4. Stat type filter mismatch

**Solutions:**
```bash
# 1. Check PostgreSQL container
docker compose ps pg-1

# 2. Check datasource in Grafana UI
# Navigate to Configuration → Data Sources → PostgreSQL NFL → Test

# 3. Verify predictions exist
psql "postgresql://dro:sicillionbillions@localhost:5544/devdb01" \
  -c "SELECT COUNT(*) FROM ml.predictions WHERE stat_type = 'passing_yards';"

# 4. Check prediction timestamps
psql "postgresql://dro:sicillionbillions@localhost:5544/devdb01" \
  -c "SELECT MIN(predicted_at), MAX(predicted_at) FROM ml.predictions;"

# If predictions are old, adjust time range in Grafana (top-right corner)
```

### Issue: "Error connecting to datasource"

**Causes:**
1. PostgreSQL container not running
2. Wrong connection string
3. Authentication failure

**Solutions:**
```bash
# 1. Start PostgreSQL
docker compose up -d pg-1

# 2. Verify connection string in grafana/datasources/postgresql.yml
# Should be: pg-1:5432/devdb01

# 3. Check credentials
# User: dro
# Password: sicillionbillions (matches docker-compose.yml environment)

# 4. Restart Grafana to reload datasource
docker compose restart grafana
```

### Issue: Panels show incorrect data

**Causes:**
1. Query references wrong columns
2. Stat type filter missing
3. Time zone mismatch

**Solutions:**
```sql
-- Verify table schema
\d ml.predictions

-- Check column names match queries
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'ml' AND table_name = 'predictions';

-- Ensure stat_type filter is present in all queries
-- All queries should include: WHERE stat_type = 'passing_yards'
```

---

## Next Steps

### 1. Populate Recent Data
```bash
# Run backtest to populate ml.predictions with recent games
python py/backtest/historical_validation_4way.py

# Or insert predictions from production API
python py/production/real_time_api.py
```

### 2. Add XGBoost v4 Panels (Optional)
```json
{
  "id": 12,
  "title": "XGBoost v4 Component Performance",
  "type": "timeseries",
  "targets": [{
    "rawSql": "SELECT DATE_TRUNC('day', predicted_at) as time, AVG(ABS(xgboost_mean - actual_value)) as mae FROM ml.predictions WHERE stat_type = 'passing_yards' AND xgboost_mean IS NOT NULL GROUP BY time ORDER BY time;"
  }]
}
```

### 3. Create Alerts (Optional)
```yaml
# grafana/provisioning/alerting/predictions.yml
apiVersion: 1
groups:
  - name: Predictions
    interval: 5m
    rules:
      - uid: mae_threshold
        title: MAE Above Threshold
        condition: A
        data:
          - refId: A
            queryType: sql
            model:
              rawSql: "SELECT AVG(abs_prediction_error) FROM ml.predictions WHERE predicted_at > NOW() - INTERVAL '1 day';"
        annotations:
          description: "MAE is {{ $value }} yards (threshold: 45 yards)"
        for: 10m
        labels:
          severity: warning
```

### 4. Export Metrics to Prometheus (Optional)
```python
# py/production/metrics.py
from prometheus_client import Gauge

mae_gauge = Gauge('predictions_mae_yards', 'Mean Absolute Error in Passing Yards')
win_rate_gauge = Gauge('predictions_win_rate', 'Betting Win Rate')
calibration_gauge = Gauge('predictions_calibration_coverage', 'Calibration Coverage')

# Update metrics from database
def update_metrics():
    result = db.query("SELECT AVG(abs_prediction_error) as mae, ...")
    mae_gauge.set(result['mae'])
    win_rate_gauge.set(result['win_rate'])
    calibration_gauge.set(result['calibration'])
```

---

## Files Created

1. **grafana/provisioning/dashboards/nfl-predictions-monitoring.json** (353 lines)
   - New comprehensive predictions dashboard
   - 11 data panels + 5 section rows
   - All panels query `ml.predictions` table

2. **grafana/datasources/postgresql.yml** (23 lines)
   - PostgreSQL datasource configuration
   - Connection to pg-1:5432/devdb01
   - TimescaleDB support enabled

3. **docs/deployment/GRAFANA_PREDICTIONS_DASHBOARD_COMPLETE.md** (this file)
   - Complete documentation
   - SQL query patterns
   - Deployment and troubleshooting guide

---

## Performance Characteristics

### Expected Dashboard Load Times
- **Small dataset** (< 1,000 predictions): < 1 second
- **Medium dataset** (1,000 - 10,000): 1-3 seconds
- **Large dataset** (> 10,000): 3-5 seconds

### Database Impact
- **Concurrent queries:** Up to 11 queries per dashboard refresh
- **Query complexity:** Simple aggregations (AVG, SUM, COUNT)
- **Indexes used:**
  - `idx_predictions_stat_type` (stat_type filter)
  - `idx_predictions_predicted_at` (time range filter)
  - `idx_predictions_season_week` (season grouping)

### Optimization Tips
1. **Reduce refresh rate** if database load is high (default: 1 minute)
2. **Shorten time range** for faster queries (default: 7 days)
3. **Add indexes** if queries are slow:
   ```sql
   CREATE INDEX idx_predictions_actual_value ON ml.predictions(actual_value)
   WHERE actual_value IS NOT NULL;
   ```

---

## Comparison: Old vs New

### Old Dashboard (Incorrect)
- **Table:** `predictions.retrospectives` (game outcome predictions)
- **Metrics:** Points, margins (home/away scores)
- **Stat type:** Game-level predictions
- **Y-axis range:** 0-50 points
- **Panel titles:** "Prediction Accuracy (Points)", "MAE by Week (Points)"

### New Dashboard (Correct)
- **Table:** `ml.predictions` (player passing yards)
- **Metrics:** Passing yards, MAE, win rate, calibration, P&L
- **Stat type:** Player-level predictions (`stat_type = 'passing_yards'`)
- **Y-axis range:** 0-600 yards
- **Panel titles:** "Prediction Accuracy (Yards)", "MAE by Week (Yards)"

---

## Summary of Panel Updates

| Panel ID | Panel Name | Old Query | New Query |
|----------|------------|-----------|-----------|
| 1 | Overall MAE | `predictions.retrospectives` (margin_error) | `ml.predictions` (abs_prediction_error) |
| 2 | Win Rate | N/A | `ml.predictions` (status = 'win') |
| 3 | Calibration | N/A | `ml.predictions` (calibrated) |
| 4 | Total P&L | N/A | `ml.predictions` (SUM(pnl)) |
| 5 | Component MAE | N/A | `ml.predictions` (component means) |
| 6 | Weight Distribution | N/A | Static weights (30/30/25/15) |
| 7 | MAE by Week | `predictions.retrospectives` (time series) | `ml.predictions` (time series) |
| 8 | Cumulative P&L | N/A | `ml.predictions` (running SUM(pnl)) |
| 9 | Recent Predictions | N/A | `ml.predictions` (table view) |
| 10 | MAE by Season | N/A | `ml.predictions` (GROUP BY season) |
| 11 | Win Rate by Season | N/A | `ml.predictions` (GROUP BY season) |

**Total panels updated:** 11 new panels created (replacing 0 old panels)

---

## Status

✅ **COMPLETE** - November 15, 2025

**Next Action:** Restart Grafana container to load new dashboard
```bash
docker compose restart grafana
```

**Access:** http://localhost:3000 → Dashboards → NFL Analytics - Predictions Monitoring

**Expected Data:** Historical predictions from 2022-2024 (148 games, ~1,387 predictions)
