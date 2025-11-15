-- ============================================================================
-- Grafana Dashboard SQL Queries for A/B Test Monitoring
-- ============================================================================
-- Ensemble v5 (control) vs v6 (treatment) Sagarin Integration A/B Test
--
-- Purpose: Monitor real-time performance and statistical significance
-- Created: November 14, 2025
-- Author: Claude Code (Sonnet 4.5)
--
-- Usage: Copy these queries into Grafana dashboard panels
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PANEL 1: A/B Test Overview
-- ----------------------------------------------------------------------------
-- Description: Current sample sizes and split ratio verification
-- Visualization: Table
-- Refresh: 1m

SELECT
    m.model_name,
    m.version,
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'A (Control)'
        WHEN m.model_name = 'ensemble_v6.0' THEN 'B (Treatment)'
    END as ab_group,
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN '4-way ensemble'
        WHEN m.model_name = 'ensemble_v6.0' THEN '5-way with ELO + adaptive weights'
    END as description,
    COUNT(p.prediction_id) as total_predictions,
    COUNT(p.actual_value) as completed_games,
    ROUND(COUNT(p.actual_value)::numeric / NULLIF(COUNT(p.prediction_id), 0) * 100, 1) as completion_pct,
    MIN(p.predicted_at) as first_prediction,
    MAX(p.predicted_at) as last_prediction,
    ROUND(EXTRACT(EPOCH FROM (MAX(p.predicted_at) - MIN(p.predicted_at)))/3600, 1) as hours_elapsed
FROM ml.model_registry m
LEFT JOIN ml.predictions p ON m.model_id = p.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
GROUP BY m.model_id, m.model_name, m.version
ORDER BY ab_group;

-- ----------------------------------------------------------------------------
-- PANEL 2: Performance Metrics (MAE, RMSE, Calibration)
-- ----------------------------------------------------------------------------
-- Description: Core performance metrics for both groups
-- Visualization: Table with conditional coloring
-- Refresh: 5m

SELECT
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'A (Control)'
        WHEN m.model_name = 'ensemble_v6.0' THEN 'B (Treatment)'
    END as group,
    m.version,
    COUNT(p.actual_value) as n_completed,

    -- MAE
    ROUND(AVG(ABS(p.prediction_mean - p.actual_value))::numeric, 2) as mae,

    -- RMSE
    ROUND(SQRT(AVG(POWER(p.prediction_mean - p.actual_value, 2)))::numeric, 2) as rmse,

    -- Calibration (90% prediction interval)
    ROUND(AVG(
        CASE
            WHEN p.actual_value BETWEEN p.prediction_q10 AND p.prediction_q90
            THEN 1.0
            ELSE 0.0
        END
    )::numeric * 100, 1) as calibration_90_pct,

    -- Average uncertainty
    ROUND(AVG(p.prediction_std)::numeric, 1) as avg_uncertainty,

    -- Bias (mean prediction error)
    ROUND(AVG(p.prediction_mean - p.actual_value)::numeric, 2) as bias
FROM ml.model_registry m
JOIN ml.predictions p ON m.model_id = p.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.actual_value IS NOT NULL
  AND p.stat_type = 'passing_yards'
GROUP BY m.model_name, m.version
ORDER BY group;

-- ----------------------------------------------------------------------------
-- PANEL 3: MAE Over Time (Rolling 20-game window)
-- ----------------------------------------------------------------------------
-- Description: MAE trend to detect performance drift
-- Visualization: Time series line chart
-- Refresh: 5m

WITH numbered_predictions AS (
    SELECT
        m.model_name,
        p.predicted_at,
        ABS(p.prediction_mean - p.actual_value) as abs_error,
        ROW_NUMBER() OVER (PARTITION BY m.model_name ORDER BY p.predicted_at) as rn
    FROM ml.model_registry m
    JOIN ml.predictions p ON m.model_id = p.model_id
    WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
      AND p.actual_value IS NOT NULL
)
SELECT
    predicted_at as time,
    model_name as metric,
    ROUND(AVG(abs_error) OVER (
        PARTITION BY model_name
        ORDER BY rn
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    )::numeric, 2) as rolling_mae_20
FROM numbered_predictions
WHERE rn >= 20  -- Start after we have 20 observations
ORDER BY time;

-- ----------------------------------------------------------------------------
-- PANEL 4: Statistical Significance Test
-- ----------------------------------------------------------------------------
-- Description: Welch's t-test for MAE difference
-- Visualization: Stat panel (single value with threshold coloring)
-- Refresh: 5m

WITH group_stats AS (
    SELECT
        CASE
            WHEN m.model_name = 'ensemble_v5.0' THEN 'A'
            WHEN m.model_name = 'ensemble_v6.0' THEN 'B'
        END as ab_group,
        AVG(ABS(p.prediction_mean - p.actual_value)) as mean_mae,
        STDDEV(ABS(p.prediction_mean - p.actual_value)) as std_mae,
        COUNT(*) as n
    FROM ml.model_registry m
    JOIN ml.predictions p ON m.model_id = p.model_id
    WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
      AND p.actual_value IS NOT NULL
    GROUP BY m.model_name
),
t_stat AS (
    SELECT
        a.mean_mae as control_mae,
        b.mean_mae as treatment_mae,
        a.n as n_a,
        b.n as n_b,
        (a.mean_mae - b.mean_mae) /
        SQRT(POWER(a.std_mae, 2)/a.n + POWER(b.std_mae, 2)/b.n) as t_statistic,
        POWER(a.std_mae, 2)/a.n + POWER(b.std_mae, 2)/b.n as se_squared
    FROM
        (SELECT * FROM group_stats WHERE ab_group = 'A') a,
        (SELECT * FROM group_stats WHERE ab_group = 'B') b
)
SELECT
    ROUND((control_mae - treatment_mae)::numeric, 2) as mae_diff,
    ROUND(((control_mae - treatment_mae) / control_mae * 100)::numeric, 2) as mae_improvement_pct,
    ROUND(t_statistic::numeric, 3) as t_statistic,
    CASE
        WHEN ABS(t_statistic) > 1.96 THEN 'Significant (p < 0.05)'
        WHEN ABS(t_statistic) > 1.65 THEN 'Marginally Significant (p < 0.10)'
        ELSE 'Not Significant'
    END as significance_level,
    CASE
        WHEN n_a >= 100 AND n_b >= 100 THEN 'Sufficient Power'
        ELSE 'Insufficient Sample Size'
    END as power_status
FROM t_stat;

-- ----------------------------------------------------------------------------
-- PANEL 5: Prediction Volume by Group
-- ----------------------------------------------------------------------------
-- Description: Verify 50/50 split is maintained
-- Visualization: Time series bar chart (stacked)
-- Refresh: 5m

SELECT
    DATE_TRUNC('hour', p.predicted_at) as time,
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'Group A (v5)'
        WHEN m.model_name = 'ensemble_v6.0' THEN 'Group B (v6)'
    END as metric,
    COUNT(*) as value
FROM ml.model_registry m
JOIN ml.predictions p ON m.model_id = p.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.predicted_at >= NOW() - INTERVAL '24 hours'
GROUP BY time, m.model_name
ORDER BY time;

-- ----------------------------------------------------------------------------
-- PANEL 6: Deployment Decision Criteria
-- ----------------------------------------------------------------------------
-- Description: 5-criteria checklist for promoting v6 to production
-- Visualization: Table with status indicators
-- Refresh: 5m

WITH metrics AS (
    SELECT
        m.model_name,
        COUNT(p.actual_value) as n_completed,
        AVG(ABS(p.prediction_mean - p.actual_value)) as mae,
        SQRT(AVG(POWER(p.prediction_mean - p.actual_value, 2))) as rmse,
        AVG(
            CASE
                WHEN p.actual_value BETWEEN p.prediction_q10 AND p.prediction_q90
                THEN 1.0
                ELSE 0.0
            END
        ) as calibration_90
    FROM ml.model_registry m
    JOIN ml.predictions p ON m.model_id = p.model_id
    WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
      AND p.actual_value IS NOT NULL
    GROUP BY m.model_name
)
SELECT
    'Minimum Sample Size' as criterion,
    '>= 100 games per group' as threshold,
    MIN(n_completed)::text as current_value,
    CASE WHEN MIN(n_completed) >= 100 THEN '✅ PASS' ELSE '❌ FAIL' END as status
FROM metrics
UNION ALL
SELECT
    'MAE Improvement' as criterion,
    '> 2%' as threshold,
    ROUND((MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN mae END) -
           MAX(CASE WHEN model_name = 'ensemble_v6.0' THEN mae END)) /
          MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN mae END) * 100, 2)::text || '%' as current_value,
    CASE
        WHEN (MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN mae END) -
              MAX(CASE WHEN model_name = 'ensemble_v6.0' THEN mae END)) /
             MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN mae END) > 0.02
        THEN '✅ PASS'
        ELSE '❌ FAIL'
    END as status
FROM metrics
UNION ALL
SELECT
    'Calibration (v6)' as criterion,
    '>= 89%' as threshold,
    ROUND(MAX(CASE WHEN model_name = 'ensemble_v6.0' THEN calibration_90 END) * 100, 1)::text || '%' as current_value,
    CASE
        WHEN MAX(CASE WHEN model_name = 'ensemble_v6.0' THEN calibration_90 END) >= 0.89
        THEN '✅ PASS'
        ELSE '❌ FAIL'
    END as status
FROM metrics
UNION ALL
SELECT
    'RMSE Stability' as criterion,
    '< 5% degradation' as threshold,
    ROUND((MAX(CASE WHEN model_name = 'ensemble_v6.0' THEN rmse END) -
           MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN rmse END)) /
          MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN rmse END) * 100, 2)::text || '%' as current_value,
    CASE
        WHEN (MAX(CASE WHEN model_name = 'ensemble_v6.0' THEN rmse END) -
              MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN rmse END)) /
             MAX(CASE WHEN model_name = 'ensemble_v5.0' THEN rmse END) < 0.05
        THEN '✅ PASS'
        ELSE '❌ FAIL'
    END as status
FROM metrics;

-- ----------------------------------------------------------------------------
-- PANEL 7: Error Distribution Comparison
-- ----------------------------------------------------------------------------
-- Description: Histogram of prediction errors (v5 vs v6)
-- Visualization: Histogram (overlapping)
-- Refresh: 10m

SELECT
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'Group A (v5)'
        WHEN m.model_name = 'ensemble_v6.0' THEN 'Group B (v6)'
    END as series,
    ROUND((p.prediction_mean - p.actual_value) / 10) * 10 as error_bucket,
    COUNT(*) as frequency
FROM ml.model_registry m
JOIN ml.predictions p ON m.model_id = p.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.actual_value IS NOT NULL
GROUP BY m.model_name, error_bucket
HAVING COUNT(*) > 0
ORDER BY error_bucket;

-- ----------------------------------------------------------------------------
-- PANEL 8: Recent Predictions (Last 50)
-- ----------------------------------------------------------------------------
-- Description: Detailed view of recent predictions for debugging
-- Visualization: Table
-- Refresh: 1m

SELECT
    p.predicted_at,
    p.player_name,
    p.team,
    p.opponent,
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'A'
        WHEN m.model_name = 'ensemble_v6.0' THEN 'B'
    END as group,
    m.version,
    ROUND(p.prediction_mean::numeric, 1) as prediction,
    ROUND(p.prediction_std::numeric, 1) as uncertainty,
    ROUND(p.actual_value::numeric, 1) as actual,
    CASE
        WHEN p.actual_value IS NULL THEN NULL
        ELSE ROUND(ABS(p.prediction_mean - p.actual_value)::numeric, 1)
    END as abs_error,
    CASE
        WHEN p.actual_value IS NULL THEN 'Pending'
        WHEN p.actual_value BETWEEN p.prediction_q10 AND p.prediction_q90 THEN '✅ Calibrated'
        ELSE '❌ Miscalibrated'
    END as calibration_status
FROM ml.predictions p
JOIN ml.model_registry m ON p.model_id = m.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.stat_type = 'passing_yards'
ORDER BY p.predicted_at DESC
LIMIT 50;

-- ============================================================================
-- OPTIONAL ADVANCED QUERIES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PANEL 9: ROI Simulation (Requires Odds Data)
-- ----------------------------------------------------------------------------
-- Description: Simulated betting ROI if odds available
-- Note: Requires joining with odds data (not shown here)

-- SELECT
--     m.model_name,
--     COUNT(*) as n_bets,
--     SUM(CASE WHEN <prediction_correct> THEN 1 ELSE 0 END) as wins,
--     ROUND(SUM(<profit>) / SUM(<stake>) * 100, 2) as roi_pct
-- FROM ml.model_registry m
-- JOIN ml.predictions p ON m.model_id = p.model_id
-- JOIN odds_data o ON p.player_id = o.player_id AND p.week = o.week
-- WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
-- GROUP BY m.model_name;

-- ----------------------------------------------------------------------------
-- PANEL 10: Subgroup Analysis (Home vs Away)
-- ----------------------------------------------------------------------------
-- Description: MAE by home/away status

SELECT
    CASE
        WHEN m.model_name = 'ensemble_v5.0' THEN 'A (Control)'
        WHEN m.model_name = 'ensemble_v6.0' THEN 'B (Treatment)'
    END as group,
    p.is_home,
    COUNT(*) as n_games,
    ROUND(AVG(ABS(p.prediction_mean - p.actual_value))::numeric, 2) as mae,
    ROUND(STDDEV(ABS(p.prediction_mean - p.actual_value))::numeric, 2) as mae_std
FROM ml.model_registry m
JOIN ml.predictions p ON m.model_id = p.model_id
WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v6.0')
  AND p.actual_value IS NOT NULL
GROUP BY m.model_name, p.is_home
ORDER BY group, p.is_home;
