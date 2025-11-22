-- Migration: A/B Test Monitoring Infrastructure
-- Created: November 22, 2025
-- Description: Add tables for A/B test monitoring, guardrail enforcement, and experiment tracking
--
-- Usage:
--   psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -f db/migrations/007_ab_test_monitoring.sql  # pragma: allowlist secret

BEGIN;

-- ============================================================================
-- A/B Test Results Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.ab_test_results (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    ab_group VARCHAR(20) NOT NULL CHECK (ab_group IN ('control', 'treatment')),
    predicted_value FLOAT NOT NULL,
    actual_value FLOAT,  -- NULL until game completed
    error FLOAT,         -- ABS(predicted_value - actual_value), calculated post-game
    win BOOLEAN,         -- Bet outcome (TRUE if prediction correct)
    roi FLOAT,           -- Return on investment (1.0 for win, -1.1 for loss)
    model_version VARCHAR(50),  -- e.g., 'ensemble_v5', 'ensemble_v6'
    confidence_interval_lower FLOAT,  -- 90% CI lower bound
    confidence_interval_upper FLOAT,  -- 90% CI upper bound
    metadata JSONB,  -- Additional metadata (features, player info, etc.)
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ab_results_group ON ml.ab_test_results(ab_group);
CREATE INDEX IF NOT EXISTS idx_ab_results_date ON ml.ab_test_results(prediction_date);
CREATE INDEX IF NOT EXISTS idx_ab_results_game ON ml.ab_test_results(game_id);
CREATE INDEX IF NOT EXISTS idx_ab_results_model ON ml.ab_test_results(model_version);

COMMENT ON TABLE ml.ab_test_results IS 'Individual predictions for A/B test groups (control vs treatment)';
COMMENT ON COLUMN ml.ab_test_results.ab_group IS 'Group assignment (control or treatment) via MD5 hash';
COMMENT ON COLUMN ml.ab_test_results.error IS 'Absolute error, calculated after game completion';
COMMENT ON COLUMN ml.ab_test_results.win IS 'Bet outcome (TRUE if prediction beats actual)';
COMMENT ON COLUMN ml.ab_test_results.roi IS 'Return on investment (1.0 for win, -1.1 for loss at -110 odds)';

-- ============================================================================
-- A/B Test Status Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.ab_test_status (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN (
        'running',
        'stopped_guardrail',
        'stopped_manual',
        'completed',
        'promoted'
    )),
    reason TEXT,  -- Detailed reason for status change (e.g., "MAE degradation 12% exceeds limit 10%")
    metrics JSONB,  -- Snapshot of metrics at time of status change
    timestamp TIMESTAMP NOT NULL,
    created_by VARCHAR(100) DEFAULT 'system',  -- User or system that triggered status change
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ab_status_name ON ml.ab_test_status(test_name);
CREATE INDEX IF NOT EXISTS idx_ab_status_timestamp ON ml.ab_test_status(timestamp);
CREATE INDEX IF NOT EXISTS idx_ab_status_status ON ml.ab_test_status(status);

COMMENT ON TABLE ml.ab_test_status IS 'Audit log for A/B test lifecycle (start, stop, promote)';
COMMENT ON COLUMN ml.ab_test_status.status IS 'Test status (running, stopped_guardrail, stopped_manual, completed, promoted)';
COMMENT ON COLUMN ml.ab_test_status.reason IS 'Detailed reason for status change (for audit trail)';
COMMENT ON COLUMN ml.ab_test_status.metrics IS 'JSON snapshot of metrics at time of status change';

-- ============================================================================
-- Guardrail Violations Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.ab_test_guardrail_violations (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    violation_type VARCHAR(50) NOT NULL,  -- 'mae_degradation', 'roi_drop', 'win_rate_drop', 'calibration_drop'
    threshold_value FLOAT NOT NULL,  -- Configured threshold
    actual_value FLOAT NOT NULL,  -- Actual value that violated threshold
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('warning', 'critical')),
    auto_stopped BOOLEAN DEFAULT FALSE,  -- Whether test was auto-stopped
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_guardrail_test ON ml.ab_test_guardrail_violations(test_name);
CREATE INDEX IF NOT EXISTS idx_guardrail_type ON ml.ab_test_guardrail_violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_guardrail_timestamp ON ml.ab_test_guardrail_violations(timestamp);

COMMENT ON TABLE ml.ab_test_guardrail_violations IS 'Log of all guardrail violations for audit and analysis';
COMMENT ON COLUMN ml.ab_test_guardrail_violations.violation_type IS 'Type of violation (mae_degradation, roi_drop, etc.)';
COMMENT ON COLUMN ml.ab_test_guardrail_violations.auto_stopped IS 'TRUE if test was automatically stopped due to violation';

-- ============================================================================
-- Significance Tests Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.ab_test_significance (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    metric VARCHAR(50) NOT NULL,  -- 'mae', 'roi', 'win_rate'
    test_type VARCHAR(50) NOT NULL,  -- 't-test', 'bootstrap', 'bayesian'
    control_mean FLOAT NOT NULL,
    treatment_mean FLOAT NOT NULL,
    difference FLOAT NOT NULL,
    difference_pct FLOAT NOT NULL,
    p_value FLOAT NOT NULL,
    significant BOOLEAN NOT NULL,
    confidence_interval_lower FLOAT NOT NULL,
    confidence_interval_upper FLOAT NOT NULL,
    effect_size FLOAT,  -- Cohen's d
    sample_size_control INTEGER NOT NULL,
    sample_size_treatment INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_significance_test ON ml.ab_test_significance(test_name);
CREATE INDEX IF NOT EXISTS idx_significance_metric ON ml.ab_test_significance(metric);
CREATE INDEX IF NOT EXISTS idx_significance_timestamp ON ml.ab_test_significance(timestamp);

COMMENT ON TABLE ml.ab_test_significance IS 'Historical record of statistical significance tests';
COMMENT ON COLUMN ml.ab_test_significance.p_value IS 'P-value from statistical test (lower = more significant)';
COMMENT ON COLUMN ml.ab_test_significance.effect_size IS 'Cohen''s d (standardized effect size)';

-- ============================================================================
-- Helper Views
-- ============================================================================

-- View: Current A/B test metrics
CREATE OR REPLACE VIEW ml.ab_test_current_metrics AS
SELECT
    ab_group,
    COUNT(*) as sample_size,
    AVG(error) as mae,
    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
    AVG(roi) * 100 as roi_pct,
    MIN(prediction_date) as first_prediction,
    MAX(prediction_date) as last_prediction
FROM ml.ab_test_results
WHERE actual_value IS NOT NULL  -- Only completed games
GROUP BY ab_group;

COMMENT ON VIEW ml.ab_test_current_metrics IS 'Real-time summary metrics for control and treatment groups';

-- View: Daily A/B test metrics
CREATE OR REPLACE VIEW ml.ab_test_daily_metrics AS
SELECT
    DATE(prediction_date) as date,
    ab_group,
    COUNT(*) as n_predictions,
    AVG(error) as mae,
    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
    AVG(roi) * 100 as roi_pct
FROM ml.ab_test_results
WHERE actual_value IS NOT NULL
GROUP BY DATE(prediction_date), ab_group
ORDER BY date, ab_group;

COMMENT ON VIEW ml.ab_test_daily_metrics IS 'Daily rollup of A/B test metrics for trend analysis';

-- View: Latest test status
CREATE OR REPLACE VIEW ml.ab_test_latest_status AS
SELECT DISTINCT ON (test_name)
    test_name,
    status,
    reason,
    timestamp
FROM ml.ab_test_status
ORDER BY test_name, timestamp DESC;

COMMENT ON VIEW ml.ab_test_latest_status IS 'Most recent status for each A/B test';

-- ============================================================================
-- Sample Queries
-- ============================================================================

COMMENT ON SCHEMA ml IS 'Machine learning models, training runs, and A/B test monitoring';

-- Example: Get current MAE difference
-- SELECT
--     (SELECT mae FROM ml.ab_test_current_metrics WHERE ab_group = 'treatment') -
--     (SELECT mae FROM ml.ab_test_current_metrics WHERE ab_group = 'control')
-- AS mae_difference;

-- Example: Count guardrail violations by type
-- SELECT violation_type, COUNT(*)
-- FROM ml.ab_test_guardrail_violations
-- WHERE test_name = 'ensemble_v5_vs_v6'
-- GROUP BY violation_type;

-- Example: Track significance over time
-- SELECT DATE(timestamp) as date, metric, p_value
-- FROM ml.ab_test_significance
-- WHERE test_name = 'ensemble_v5_vs_v6'
-- ORDER BY timestamp;

COMMIT;

-- ============================================================================
-- Verification
-- ============================================================================

DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'ml'
    AND table_name IN ('ab_test_results', 'ab_test_status', 'ab_test_guardrail_violations', 'ab_test_significance');

    IF table_count = 4 THEN
        RAISE NOTICE '✅ Migration successful: 4 tables created';
    ELSE
        RAISE WARNING '⚠️  Expected 4 tables, found %', table_count;
    END IF;
END $$;

-- List created objects
\dt ml.ab_test*
\dv ml.ab_test*
