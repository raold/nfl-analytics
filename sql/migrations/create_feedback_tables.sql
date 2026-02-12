-- Feedback Loop and Meta-Learning Tables
-- Purpose: Track feature attribution, weight adjustments, and learning cycles
-- Author: Claude Code
-- Date: 2025-12-22

-- ============================================================================
-- FEEDBACK RUNS
-- Track each weekly feedback loop execution
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.feedback_runs (
    run_id SERIAL PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Outcomes updated
    games_updated INTEGER DEFAULT 0,
    games_with_outcomes INTEGER DEFAULT 0,

    -- Attribution analysis
    top_semantic_features JSONB,  -- {feature: attribution_score}
    semantic_contribution_score FLOAT,  -- Overall semantic lift

    -- Trigger status
    triggers_fired TEXT[],        -- ['mae_degradation', ...]
    trigger_details JSONB,        -- Details about each trigger

    -- Weight adjustments
    weight_adjustments JSONB,     -- {model: {old: X, new: Y}}
    weights_applied BOOLEAN DEFAULT FALSE,

    -- Annotations generated
    sentiment_annotations INTEGER DEFAULT 0,
    ner_annotations INTEGER DEFAULT 0,

    -- Metadata
    dry_run BOOLEAN DEFAULT FALSE,
    run_duration_seconds FLOAT,
    error_message TEXT,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_runs_season_week
    ON ml.feedback_runs(season, week);
CREATE INDEX IF NOT EXISTS idx_feedback_runs_timestamp
    ON ml.feedback_runs(run_timestamp DESC);

COMMENT ON TABLE ml.feedback_runs IS
    'Records each execution of the weekly feedback loop. Tracks outcomes updated, '
    'feature attribution analysis, retrain triggers, and weight adjustments.';

-- ============================================================================
-- SEMANTIC ATTRIBUTION
-- Track per-game feature contribution to predictions
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.semantic_attribution (
    attribution_id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    -- Prediction outcome
    prediction_correct BOOLEAN,  -- Spread cover prediction correct
    prediction_error FLOAT,       -- Absolute spread error
    predicted_spread FLOAT,
    actual_spread FLOAT,

    -- Feature values at prediction time
    semantic_features JSONB,      -- {feature: value}

    -- Attribution scores (SHAP-like)
    feature_attributions JSONB,   -- {feature: contribution}
    total_semantic_contribution FLOAT,  -- Sum of all semantic attributions

    -- Metadata
    model_version VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sem_attr_game
    ON ml.semantic_attribution(game_id);
CREATE INDEX IF NOT EXISTS idx_sem_attr_season_week
    ON ml.semantic_attribution(season, week);
CREATE INDEX IF NOT EXISTS idx_sem_attr_correct
    ON ml.semantic_attribution(prediction_correct);

COMMENT ON TABLE ml.semantic_attribution IS
    'Per-game analysis of how semantic features contributed to prediction accuracy. '
    'Uses SHAP-like attribution to measure marginal contribution of each feature.';

-- ============================================================================
-- FEATURE PERFORMANCE MEMORY
-- Track rolling performance of each feature over time
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.feature_performance (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    feature_name VARCHAR(100) NOT NULL,
    feature_value FLOAT,

    -- Outcome
    prediction_error FLOAT,
    correct_prediction BOOLEAN,

    -- Computed contribution
    marginal_contribution FLOAT,  -- SHAP-like value
    contribution_rank INTEGER,    -- Rank among all features for this game

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feat_perf_feature
    ON ml.feature_performance(feature_name);
CREATE INDEX IF NOT EXISTS idx_feat_perf_season_week
    ON ml.feature_performance(season, week);
CREATE INDEX IF NOT EXISTS idx_feat_perf_game
    ON ml.feature_performance(game_id);

COMMENT ON TABLE ml.feature_performance IS
    'Granular tracking of each feature''s contribution to each prediction. '
    'Enables rolling analysis of feature effectiveness over time.';

-- ============================================================================
-- FEATURE SELECTION STATE
-- Current state of each feature (active, disabled, probation)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.feature_selection_state (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL UNIQUE,
    feature_category VARCHAR(50),  -- 'semantic', 'base', 'synthetic'
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- 'active', 'disabled', 'probation'
    weight_multiplier FLOAT DEFAULT 1.0,

    -- Performance metrics (rolling)
    rolling_contribution FLOAT,
    contribution_std FLOAT,
    games_evaluated INTEGER DEFAULT 0,
    weeks_active INTEGER DEFAULT 0,
    weeks_since_change INTEGER DEFAULT 0,

    -- History
    last_status_change TIMESTAMPTZ,
    change_reason TEXT,
    previous_status VARCHAR(20),

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feature_status
    ON ml.feature_selection_state(status);
CREATE INDEX IF NOT EXISTS idx_feature_category
    ON ml.feature_selection_state(feature_category);

COMMENT ON TABLE ml.feature_selection_state IS
    'Current state of each feature in the selection pipeline. Features can be '
    'active (used in predictions), disabled (excluded), or on probation (monitored).';

-- ============================================================================
-- PROMPT VERSIONS
-- Track LLM prompt evolution for A/B testing
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.prompt_versions (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    prompt_type VARCHAR(50) NOT NULL,  -- 'sentiment', 'ner', 'severity'
    prompt_version VARCHAR(20) NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_template JSONB,  -- Structured template with variables

    -- Performance metrics
    games_tested INTEGER DEFAULT 0,
    accuracy_rate FLOAT,
    contribution_score FLOAT,
    avg_latency_ms FLOAT,

    -- Status
    is_active BOOLEAN DEFAULT FALSE,
    is_baseline BOOLEAN DEFAULT FALSE,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retired_at TIMESTAMPTZ,
    retire_reason TEXT,

    CONSTRAINT unique_prompt_version UNIQUE (feature_name, prompt_type, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_prompt_active
    ON ml.prompt_versions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_prompt_feature
    ON ml.prompt_versions(feature_name);

COMMENT ON TABLE ml.prompt_versions IS
    'Tracks evolution of LLM prompts used for semantic feature extraction. '
    'Enables A/B testing of prompt improvements based on prediction outcomes.';

-- ============================================================================
-- RETRAIN TRIGGERS LOG
-- History of when retrain triggers fired
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml.retrain_triggers (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    -- Trigger details
    trigger_type VARCHAR(50) NOT NULL,  -- 'mae_degradation', 'accuracy_drop', etc.
    trigger_value FLOAT,                -- Actual value that triggered
    threshold_value FLOAT,              -- Threshold that was exceeded

    -- Context
    baseline_value FLOAT,               -- Historical baseline for comparison
    lookback_weeks INTEGER,
    games_evaluated INTEGER,

    -- Action taken
    action_taken VARCHAR(50),           -- 'retraining_scheduled', 'ignored', 'manual_review'
    retraining_job_id VARCHAR(50),

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_retrain_model
    ON ml.retrain_triggers(model_id);
CREATE INDEX IF NOT EXISTS idx_retrain_season_week
    ON ml.retrain_triggers(season, week);
CREATE INDEX IF NOT EXISTS idx_retrain_type
    ON ml.retrain_triggers(trigger_type);

COMMENT ON TABLE ml.retrain_triggers IS
    'Log of all retrain triggers that fired, including context and actions taken.';

-- ============================================================================
-- AGGREGATED FEATURE STATS (Materialized View)
-- Pre-computed rolling feature performance
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS ml.feature_stats_rolling AS
SELECT
    feature_name,
    COUNT(*) AS games_count,
    AVG(marginal_contribution) AS avg_contribution,
    STDDEV(marginal_contribution) AS std_contribution,
    SUM(CASE WHEN correct_prediction THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS accuracy_when_positive,
    AVG(CASE WHEN correct_prediction THEN marginal_contribution ELSE NULL END) AS contrib_when_correct,
    AVG(CASE WHEN NOT correct_prediction THEN marginal_contribution ELSE NULL END) AS contrib_when_incorrect,
    MAX(created_at) AS last_updated
FROM ml.feature_performance
WHERE created_at >= NOW() - INTERVAL '8 weeks'
GROUP BY feature_name;

CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_stats_name
    ON ml.feature_stats_rolling(feature_name);

COMMENT ON MATERIALIZED VIEW ml.feature_stats_rolling IS
    'Rolling 8-week statistics for each feature. Refresh weekly after feedback loop.';

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get feature contribution summary for a week
CREATE OR REPLACE FUNCTION ml.get_week_feature_summary(
    p_season INTEGER,
    p_week INTEGER
)
RETURNS TABLE (
    feature_name VARCHAR(100),
    avg_contribution FLOAT,
    games_count BIGINT,
    accuracy_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fp.feature_name,
        AVG(fp.marginal_contribution)::FLOAT AS avg_contribution,
        COUNT(*)::BIGINT AS games_count,
        (SUM(CASE WHEN fp.correct_prediction THEN 1 ELSE 0 END)::FLOAT / COUNT(*))::FLOAT AS accuracy_rate
    FROM ml.feature_performance fp
    WHERE fp.season = p_season AND fp.week = p_week
    GROUP BY fp.feature_name
    ORDER BY avg_contribution DESC;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION ml.get_week_feature_summary IS
    'Get summary of feature contributions for a specific week.';

-- Function to check if retrain is needed
CREATE OR REPLACE FUNCTION ml.check_retrain_needed(
    p_model_id VARCHAR(50),
    p_season INTEGER,
    p_current_week INTEGER,
    p_lookback_weeks INTEGER DEFAULT 4
)
RETURNS TABLE (
    trigger_type VARCHAR(50),
    current_value FLOAT,
    baseline_value FLOAT,
    threshold FLOAT,
    triggered BOOLEAN
) AS $$
DECLARE
    v_baseline_mae FLOAT;
    v_recent_mae FLOAT;
    v_baseline_accuracy FLOAT;
    v_recent_accuracy FLOAT;
BEGIN
    -- Get baseline MAE (first 4 weeks of season)
    SELECT AVG(abs_spread_error) INTO v_baseline_mae
    FROM ml.game_predictions
    WHERE season = p_season AND week <= 4 AND model_id = p_model_id;

    -- Get recent MAE (last N weeks)
    SELECT AVG(abs_spread_error) INTO v_recent_mae
    FROM ml.game_predictions
    WHERE season = p_season
      AND week > p_current_week - p_lookback_weeks
      AND week <= p_current_week
      AND model_id = p_model_id;

    -- MAE degradation check
    RETURN QUERY SELECT
        'mae_degradation'::VARCHAR(50),
        v_recent_mae,
        v_baseline_mae,
        0.5::FLOAT,  -- Threshold: 0.5 point MAE increase
        (v_recent_mae - COALESCE(v_baseline_mae, v_recent_mae)) > 0.5;

    -- Get baseline accuracy
    SELECT AVG(CASE WHEN (predicted_spread > 0) = (actual_spread > 0) THEN 1.0 ELSE 0.0 END)
    INTO v_baseline_accuracy
    FROM ml.game_predictions
    WHERE season = p_season AND week <= 4 AND model_id = p_model_id
      AND actual_spread IS NOT NULL;

    -- Get recent accuracy
    SELECT AVG(CASE WHEN (predicted_spread > 0) = (actual_spread > 0) THEN 1.0 ELSE 0.0 END)
    INTO v_recent_accuracy
    FROM ml.game_predictions
    WHERE season = p_season
      AND week > p_current_week - p_lookback_weeks
      AND week <= p_current_week
      AND model_id = p_model_id
      AND actual_spread IS NOT NULL;

    -- Accuracy drop check
    RETURN QUERY SELECT
        'accuracy_drop'::VARCHAR(50),
        v_recent_accuracy,
        v_baseline_accuracy,
        0.05::FLOAT,  -- Threshold: 5% accuracy drop
        (COALESCE(v_baseline_accuracy, v_recent_accuracy) - v_recent_accuracy) > 0.05;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION ml.check_retrain_needed IS
    'Check if model performance has degraded enough to trigger retraining.';

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml TO dro;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml TO dro;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA ml TO dro;
