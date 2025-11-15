# ml.predictions Table Guide

**Last Updated:** November 15, 2025
**Schema:** ml
**Table:** predictions
**Purpose:** Store player-level predictions with full uncertainty quantification and ensemble tracking

## Overview

The `ml.predictions` table is the central storage for all player passing yards predictions from ensemble models. It supports:

- **Player-level predictions** with full uncertainty quantification (mean, std, quantiles)
- **Ensemble tracking** via `ensemble_version` and `model_components` columns
- **A/B testing** with version-based performance comparison
- **Component analysis** to understand individual model contributions
- **Betting recommendations** with edge, prob_win, Kelly sizing
- **Performance tracking** with actual outcomes and error metrics

## Schema Overview

**Total Columns:** 45
**Primary Key:** `prediction_id` (BIGSERIAL)
**Unique Constraint:** (model_id, game_id, player_id, stat_type)
**Foreign Keys:** model_id → ml.model_registry(model_id)

## Key Column Groups

### Identification & Context
- `prediction_id` - BIGSERIAL PRIMARY KEY
- `model_id` - INTEGER NOT NULL (FK to ml.model_registry)
- `game_id` - VARCHAR(20) NOT NULL
- `season` - INTEGER NOT NULL (1999-2030)
- `week` - INTEGER NOT NULL (1-22)
- `player_id` - VARCHAR(50) NOT NULL
- `player_name` - VARCHAR(100) NOT NULL
- `team` - VARCHAR(5) NOT NULL
- `opponent` - VARCHAR(5) NOT NULL
- `is_home` - BOOLEAN NOT NULL
- `stat_type` - VARCHAR(50) NOT NULL (e.g., 'passing_yards')

### **NEW: Ensemble Tracking** (Added Nov 15, 2025)
- `ensemble_version` - TEXT (e.g., 'v5', 'v6', 'v5.1')
  - Purpose: Track which ensemble version generated the prediction
  - Use case: A/B testing, performance comparison across versions
  - Example: 'v5.1', 'v6.0'

- `model_components` - JSONB
  - Purpose: Store component model weights/contributions
  - Use case: Component-level analysis, weight optimization
  - Example: `{"state_space_v2": 0.30, "hierarchical_v2": 0.30, "hybrid_xgb": 0.25, "informative_priors_v2": 0.15}`

### Predictions
- `prediction_mean` - NUMERIC(8,2) NOT NULL
- `prediction_median` - NUMERIC(8,2)
- `prediction_std` - NUMERIC(8,2) (uncertainty)
- `prediction_q05` through `prediction_q95` - NUMERIC(8,2) (quantiles)

### Component Model Predictions
- `state_space_mean` - NUMERIC(8,2)
- `bayesian_mean` - NUMERIC(8,2) (hierarchical models)
- `xgboost_mean` - NUMERIC(8,2)
- `bnn_mean` - NUMERIC(8,2)
- `bnn_calibrated_mean` - NUMERIC(8,2)

### Betting Recommendations
- `line` - NUMERIC(8,2) (prop line, e.g., 265.5)
- `side` - VARCHAR(10) ('over', 'under')
- `odds` - INTEGER (American odds, e.g., -110)
- `edge` - NUMERIC(8,4) (expected edge)
- `prob_win` - NUMERIC(8,4) (0.0-1.0)
- `kelly_fraction` - NUMERIC(8,4)
- `bet_size` - NUMERIC(12,2) (recommended dollars)
- `used_cql` - BOOLEAN (whether CQL bet sizer was used)

### Actual Outcomes
- `actual_value` - NUMERIC(8,2) (filled after game)
- `status` - VARCHAR(20) ('win', 'loss', 'push', 'pending')
- `pnl` - NUMERIC(12,2) (profit/loss in dollars)
- `prediction_error` - NUMERIC(8,2) (actual - mean)
- `abs_prediction_error` - NUMERIC(8,2) (MAE)
- `calibrated` - BOOLEAN (actual within 90% interval?)

### Timestamps
- `predicted_at` - TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
- `created_at` - TIMESTAMP WITH TIME ZONE DEFAULT NOW()
- `updated_at` - TIMESTAMP WITH TIME ZONE DEFAULT NOW()

## Important Indexes

### Performance Indexes
- `idx_predictions_season_week` - Fast weekly queries
- `idx_predictions_player` - Player history lookup
- `idx_predictions_game_player_stat` - Composite for game predictions
- `idx_predictions_model_season_stat` - Model performance analysis

### **NEW: Ensemble Indexes** (Added Nov 15, 2025)
- `idx_predictions_ensemble_version` - A/B testing queries
- `idx_predictions_ensemble_season_week` - Ensemble performance by week
- `idx_predictions_model_components` - GIN index for JSONB queries

## Usage Examples

### 1. Insert a Prediction with Ensemble Metadata

```sql
INSERT INTO ml.predictions (
  model_id, game_id, season, week, player_id, player_name,
  team, opponent, is_home, stat_type,
  prediction_mean, prediction_std,
  prediction_q05, prediction_q95,
  ensemble_version, model_components,
  state_space_mean, bayesian_mean, xgboost_mean
)
VALUES (
  35,  -- ensemble v6 model_id
  '2024_11_KC_BUF',
  2024, 11,
  'MAH339293',  -- Patrick Mahomes
  'Patrick Mahomes',
  'KC', 'BUF', TRUE,
  'passing_yards',
  287.5, 42.3,
  210.2, 364.8,
  'v6.0',  -- ensemble version
  '{"state_space_v2": 0.30, "hierarchical_v2": 0.30, "hybrid_xgb": 0.25, "informative_priors_v2": 0.15}'::jsonb,
  295.2, 282.1, 285.9
);
```

### 2. Update Ensemble Metadata for Existing Prediction

```sql
-- Using helper function
SELECT ml.set_ensemble_metadata(
  12345,  -- prediction_id
  'v5.1',
  '{"state_space_v2": 0.30, "hierarchical_v2": 0.30, "hybrid_xgb": 0.25, "informative_priors_v2": 0.15}'::jsonb
);

-- OR direct UPDATE
UPDATE ml.predictions
SET
  ensemble_version = 'v5.1',
  model_components = '{"state_space_v2": 0.30, "hierarchical_v2": 0.30, "hybrid_xgb": 0.25, "informative_priors_v2": 0.15}'::jsonb
WHERE prediction_id = 12345;
```

### 3. Query Predictions by Ensemble Version (A/B Testing)

```sql
-- Compare v5 vs v6 performance
SELECT
  ensemble_version,
  COUNT(*) as n_predictions,
  AVG(prediction_mean) as avg_prediction,
  AVG(actual_value) as avg_actual,
  AVG(abs_prediction_error) as mae,
  AVG(CASE WHEN status = 'win' THEN 1.0 ELSE 0.0 END) as win_rate
FROM ml.predictions
WHERE ensemble_version IN ('v5.1', 'v6.0')
  AND actual_value IS NOT NULL
  AND season = 2024
GROUP BY ensemble_version;
```

### 4. Analyze Component Model Contributions

```sql
-- Extract component weights and predictions
SELECT
  player_name,
  prediction_mean,
  actual_value,
  (model_components->>'state_space_v2')::numeric as ss_weight,
  (model_components->>'hierarchical_v2')::numeric as hier_weight,
  (model_components->>'hybrid_xgb')::numeric as xgb_weight,
  state_space_mean,
  bayesian_mean as hierarchical_mean,
  xgboost_mean as xgb_mean,
  prediction_error
FROM ml.predictions
WHERE ensemble_version = 'v6.0'
  AND season = 2024
  AND week = 11
  AND actual_value IS NOT NULL;
```

### 5. Use Pre-Built Views

```sql
-- Ensemble performance comparison (by version)
SELECT * FROM ml.ensemble_performance_comparison
WHERE season = 2024
ORDER BY ensemble_version, mae;

-- Component model contributions
SELECT * FROM ml.component_model_contributions
WHERE ensemble_version = 'v6.0'
  AND season = 2024
  AND week = 11;

-- Latest predictions for production
SELECT * FROM ml.latest_predictions
WHERE season = 2024 AND week = 11
  AND ensemble_version = 'v6.0';
```

### 6. Update Prediction with Actual Outcome

```sql
-- Using helper function (recommended)
SELECT ml.update_prediction_outcome(
  12345,     -- prediction_id
  312.0,     -- actual_value
  'win',     -- status
  100.00     -- pnl
);

-- Function automatically computes:
-- - prediction_error = actual_value - prediction_mean
-- - abs_prediction_error = ABS(prediction_error)
-- - calibrated = (actual_value BETWEEN prediction_q05 AND prediction_q95)
```

### 7. A/B Test Analysis

```sql
-- Statistical comparison between ensemble versions
WITH v5_stats AS (
  SELECT
    AVG(abs_prediction_error) as mae,
    STDDEV(abs_prediction_error) as std_mae,
    COUNT(*) as n
  FROM ml.predictions
  WHERE ensemble_version = 'v5.1'
    AND actual_value IS NOT NULL
    AND season = 2024
),
v6_stats AS (
  SELECT
    AVG(abs_prediction_error) as mae,
    STDDEV(abs_prediction_error) as std_mae,
    COUNT(*) as n
  FROM ml.predictions
  WHERE ensemble_version = 'v6.0'
    AND actual_value IS NOT NULL
    AND season = 2024
)
SELECT
  'v5.1' as version,
  v5.mae,
  v5.std_mae,
  v5.n,
  NULL::numeric as improvement_pct
FROM v5_stats v5
UNION ALL
SELECT
  'v6.0' as version,
  v6.mae,
  v6.std_mae,
  v6.n,
  ((v5.mae - v6.mae) / v5.mae * 100) as improvement_pct
FROM v5_stats v5, v6_stats v6;
```

## Helper Functions

### ml.set_ensemble_metadata(prediction_id, ensemble_version, model_components)
Updates ensemble metadata for a prediction.

**Example:**
```sql
SELECT ml.set_ensemble_metadata(
  12345,
  'v6.0',
  '{"state_space_v2": 0.30, "hierarchical_v2": 0.30, "hybrid_xgb": 0.25, "informative_priors_v2": 0.15}'::jsonb
);
```

### ml.update_prediction_outcome(prediction_id, actual_value, status, pnl)
Updates prediction with actual outcome and automatically computes error metrics.

**Example:**
```sql
SELECT ml.update_prediction_outcome(12345, 312.0, 'win', 100.00);
```

### ml.compute_model_metrics(model_id, season, week_start, week_end, stat_type)
Computes aggregate metrics and stores in ml.prediction_metrics.

**Example:**
```sql
SELECT ml.compute_model_metrics(35, 2024, 1, 11, 'passing_yards');
```

## Available Views

### ml.latest_predictions
Latest prediction per game/player/stat (useful for production).
- Automatically de-duplicates to most recent prediction
- Includes model metadata from ml.model_registry

### ml.ensemble_predictions
Ensemble aggregated predictions across all models.
- Aggregates component models by game/player/stat
- Computes ensemble mean, std, min/max, quantiles

### ml.ensemble_performance_comparison (NEW)
Compare performance across different ensemble versions.
- Groups by ensemble_version, stat_type, season
- Computes MAE, RMSE, calibration, win rate, PnL

### ml.component_model_contributions (NEW)
Detailed view of component model weights and predictions.
- Extracts individual component weights from JSONB
- Shows component-level predictions vs ensemble
- Useful for weight optimization analysis

### ml.model_leaderboard
Best performing models ranked by MAE.
- Ranks all models by performance
- Includes calibration, win rate, total PnL

## Python Integration Example

```python
import psycopg2
import json
from datetime import datetime

conn = psycopg2.connect(
    "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
)

# Insert prediction with ensemble metadata
with conn.cursor() as cur:
    cur.execute("""
        INSERT INTO ml.predictions (
          model_id, game_id, season, week, player_id, player_name,
          team, opponent, is_home, stat_type,
          prediction_mean, prediction_std,
          prediction_q05, prediction_q95,
          ensemble_version, model_components,
          state_space_mean, bayesian_mean, xgboost_mean
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING prediction_id
    """, (
        35, '2024_11_KC_BUF', 2024, 11,
        'MAH339293', 'Patrick Mahomes',
        'KC', 'BUF', True, 'passing_yards',
        287.5, 42.3,
        210.2, 364.8,
        'v6.0',
        json.dumps({
            "state_space_v2": 0.30,
            "hierarchical_v2": 0.30,
            "hybrid_xgb": 0.25,
            "informative_priors_v2": 0.15
        }),
        295.2, 282.1, 285.9
    ))
    prediction_id = cur.fetchone()[0]
    conn.commit()

print(f"Created prediction_id: {prediction_id}")

# Query A/B test results
with conn.cursor() as cur:
    cur.execute("""
        SELECT * FROM ml.ensemble_performance_comparison
        WHERE season = 2024
        ORDER BY ensemble_version, mae
    """)
    for row in cur.fetchall():
        print(row)
```

## Migration History

- **032_ml_predictions.sql** (Nov 9, 2025): Initial table creation with 43 columns
- **044_add_ensemble_tracking_columns.sql** (Nov 15, 2025): Added `ensemble_version` and `model_components` columns

## Related Tables

- **ml.model_registry** - Model metadata and versioning
- **ml.prediction_metrics** - Aggregate performance metrics by model/time period

## Best Practices

1. **Always set ensemble_version** when inserting predictions for version tracking
2. **Use helper functions** (`ml.set_ensemble_metadata`, `ml.update_prediction_outcome`) for consistency
3. **Leverage views** for common queries rather than writing custom SQL
4. **Use GIN index** on model_components for JSONB queries
5. **Update actual_value** after games complete for performance tracking
6. **Run ml.compute_model_metrics** periodically to refresh aggregate metrics

## Quick Reference

| Purpose | View/Function |
|---------|---------------|
| Latest predictions | `ml.latest_predictions` |
| A/B testing | `ml.ensemble_performance_comparison` |
| Component analysis | `ml.component_model_contributions` |
| Model ranking | `ml.model_leaderboard` |
| Update outcome | `ml.update_prediction_outcome()` |
| Set ensemble metadata | `ml.set_ensemble_metadata()` |
| Compute metrics | `ml.compute_model_metrics()` |

## Support

For schema questions or issues:
- See `db/migrations/032_ml_predictions.sql` for original schema
- See `db/migrations/044_add_ensemble_tracking_columns.sql` for ensemble tracking additions
- Check `docs/database/DATABASE_QUICK_REFERENCE.md` for database conventions
