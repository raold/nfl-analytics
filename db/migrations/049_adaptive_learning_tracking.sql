-- Migration 049: Adaptive Learning Tracking Tables
-- Created: November 22, 2025
-- Purpose: Track ensemble weight evolution and feature importance over time for adaptive learning

-- ===========================================================================
-- TABLE 1: ml.ensemble_weight_history
-- ===========================================================================
-- Tracks weekly ensemble weight re-optimization results
-- Enables visualization of weight evolution and performance correlation
-- ===========================================================================

CREATE TABLE IF NOT EXISTS ml.ensemble_weight_history (
    history_id SERIAL PRIMARY KEY,

    -- Ensemble identification
    ensemble_id INTEGER REFERENCES ml.ensemble_versions(ensemble_id),
    ensemble_version TEXT NOT NULL,           -- 'v6.0', 'v7.0', etc.

    -- Temporal scope
    effective_season INTEGER NOT NULL,         -- Season these weights apply to
    effective_week INTEGER NOT NULL,           -- Week these weights were deployed
    training_start_week INTEGER,               -- First week used in optimization
    training_end_week INTEGER,                 -- Last week used in optimization
    training_window_size INTEGER,              -- Number of weeks used (e.g., 6)

    -- Component weights (5-way ensemble)
    state_space_weight NUMERIC(5,4),
    hierarchical_weight NUMERIC(5,4),
    xgboost_weight NUMERIC(5,4),
    informative_priors_weight NUMERIC(5,4),
    hybrid_weight NUMERIC(5,4),

    -- Full weight vector (JSONB for flexibility if ensemble expands)
    component_weights JSONB NOT NULL,          -- {"state_space_v2": 0.27, ...}

    -- Optimization method details
    optimization_method TEXT NOT NULL,         -- 'wls_grid_search', 'equal_weight', 'meta_learning'
    halflife NUMERIC(4,2),                     -- WLS recency halflife (2-8 weeks)
    loss_function TEXT,                        -- 'mae', 'mse', 'huber'
    constraints_applied JSONB,                 -- Per-component bounds if used

    -- Performance metrics (on validation set)
    validation_mae NUMERIC(8,4),
    validation_rmse NUMERIC(8,4),
    validation_calibration NUMERIC(4,3),       -- % within 90% interval
    validation_games INTEGER,                  -- Number of games in validation

    -- Weight change tracking
    weight_change_magnitude NUMERIC(5,4),      -- L1 norm of weight changes from previous
    max_weight_change NUMERIC(5,4),            -- Largest single component change
    changed_component TEXT,                    -- Component with largest change

    -- Deployment tracking
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deployed_by TEXT DEFAULT 'system',         -- 'system', 'admin', 'cron'
    is_active BOOLEAN DEFAULT TRUE,            -- Currently in production?

    -- Audit
    notes TEXT,                                -- Manual notes about this optimization

    -- Constraints
    CONSTRAINT check_weights_sum_to_one
        CHECK (ABS((state_space_weight + hierarchical_weight + xgboost_weight +
                    informative_priors_weight + hybrid_weight) - 1.0) < 0.001),
    CONSTRAINT check_positive_weights
        CHECK (state_space_weight >= 0 AND hierarchical_weight >= 0 AND
               xgboost_weight >= 0 AND informative_priors_weight >= 0 AND
               hybrid_weight >= 0),
    CONSTRAINT check_week_order
        CHECK (training_start_week <= training_end_week),
    UNIQUE (ensemble_version, effective_season, effective_week)
);

-- Indexes for efficient queries
CREATE INDEX idx_weight_history_ensemble ON ml.ensemble_weight_history(ensemble_id);
CREATE INDEX idx_weight_history_season_week ON ml.ensemble_weight_history(effective_season, effective_week);
CREATE INDEX idx_weight_history_active ON ml.ensemble_weight_history(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_weight_history_deployed ON ml.ensemble_weight_history(deployed_at DESC);

-- Comments
COMMENT ON TABLE ml.ensemble_weight_history IS 'Tracks weekly ensemble weight re-optimization for adaptive learning';
COMMENT ON COLUMN ml.ensemble_weight_history.halflife IS 'WLS exponential recency halflife in weeks (2-8 typical)';
COMMENT ON COLUMN ml.ensemble_weight_history.weight_change_magnitude IS 'L1 norm ||w_new - w_old||_1 for tracking stability';
COMMENT ON COLUMN ml.ensemble_weight_history.is_active IS 'TRUE if these weights are currently in production';


-- ===========================================================================
-- TABLE 2: ml.feature_importance_history
-- ===========================================================================
-- Tracks feature importance evolution over time for each model
-- Enables detection of feature drift and importance shifts
-- ===========================================================================

CREATE TABLE IF NOT EXISTS ml.feature_importance_history (
    importance_id SERIAL PRIMARY KEY,

    -- Model identification
    model_id INTEGER REFERENCES ml.model_registry(model_id),
    model_name TEXT NOT NULL,                 -- 'xgboost_v4_ridge', etc.

    -- Temporal scope
    computed_season INTEGER NOT NULL,
    computed_week INTEGER NOT NULL,           -- Week this importance was computed
    training_data_start DATE,                 -- First date in training data
    training_data_end DATE,                   -- Last date in training data

    -- Feature details
    feature_name TEXT NOT NULL,
    feature_category TEXT,                    -- 'player_stats', 'opponent', 'context', etc.

    -- Importance metrics (multiple methods)
    importance_gain NUMERIC(10,6),            -- XGBoost gain
    importance_weight NUMERIC(10,6),          -- XGBoost weight (split count)
    importance_cover NUMERIC(10,6),           -- XGBoost cover
    shap_mean_abs NUMERIC(10,6),              -- Mean absolute SHAP value
    permutation_importance NUMERIC(10,6),     -- Permutation importance

    -- Composite score (weighted average of methods)
    composite_score NUMERIC(10,6),            -- Normalized 0-1 composite
    importance_rank INTEGER,                  -- Rank among all features (1 = most important)
    rank_percentile NUMERIC(5,2),             -- Percentile rank (99 = top 1%)

    -- Change tracking
    previous_rank INTEGER,                    -- Rank in previous week
    rank_change INTEGER,                      -- Change from previous week
    importance_trend TEXT,                    -- 'increasing', 'decreasing', 'stable'

    -- Statistical significance
    importance_std NUMERIC(10,6),             -- Std dev if bootstrap/CV used
    is_significant BOOLEAN,                   -- TRUE if importance > noise threshold

    -- Metadata
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    computation_method TEXT,                  -- 'xgboost_native', 'shap', 'permutation'
    num_samples INTEGER,                      -- Number of samples used

    -- Constraints
    CONSTRAINT check_positive_importance
        CHECK (composite_score >= 0 AND composite_score <= 1),
    CONSTRAINT check_valid_rank
        CHECK (importance_rank > 0),
    UNIQUE (model_id, computed_season, computed_week, feature_name)
);

-- Indexes for efficient queries
CREATE INDEX idx_importance_history_model ON ml.feature_importance_history(model_id);
CREATE INDEX idx_importance_history_season_week ON ml.feature_importance_history(computed_season, computed_week);
CREATE INDEX idx_importance_history_feature ON ml.feature_importance_history(feature_name);
CREATE INDEX idx_importance_history_rank ON ml.feature_importance_history(importance_rank);
CREATE INDEX idx_importance_history_computed ON ml.feature_importance_history(computed_at DESC);

-- Comments
COMMENT ON TABLE ml.feature_importance_history IS 'Tracks feature importance evolution for drift detection';
COMMENT ON COLUMN ml.feature_importance_history.composite_score IS 'Weighted average of gain (50%), weight (30%), cover (20%)';
COMMENT ON COLUMN ml.feature_importance_history.importance_trend IS 'Direction of importance change: increasing, decreasing, stable';


-- ===========================================================================
-- VIEW 1: Latest Weights Summary
-- ===========================================================================
-- Quick access to current production weights for each ensemble version
-- ===========================================================================

CREATE OR REPLACE VIEW ml.latest_ensemble_weights AS
SELECT DISTINCT ON (ensemble_version)
    history_id,
    ensemble_version,
    effective_season,
    effective_week,
    state_space_weight,
    hierarchical_weight,
    xgboost_weight,
    informative_priors_weight,
    hybrid_weight,
    component_weights,
    optimization_method,
    halflife,
    validation_mae,
    validation_calibration,
    deployed_at,
    is_active
FROM ml.ensemble_weight_history
WHERE is_active = TRUE
ORDER BY ensemble_version, deployed_at DESC;

COMMENT ON VIEW ml.latest_ensemble_weights IS 'Current production weights for each ensemble version';


-- ===========================================================================
-- VIEW 2: Weight Evolution Over Time
-- ===========================================================================
-- Time series of weight changes for visualization in Grafana
-- ===========================================================================

CREATE OR REPLACE VIEW ml.ensemble_weight_evolution AS
SELECT
    history_id,
    ensemble_version,
    effective_season,
    effective_week,
    CONCAT('S', effective_season, 'W', LPAD(effective_week::TEXT, 2, '0')) as season_week,
    deployed_at,

    -- Component weights
    state_space_weight,
    hierarchical_weight,
    xgboost_weight,
    informative_priors_weight,
    hybrid_weight,

    -- Performance
    validation_mae,
    validation_calibration,

    -- Changes
    weight_change_magnitude,
    max_weight_change,
    changed_component,

    -- Method
    optimization_method,
    halflife
FROM ml.ensemble_weight_history
ORDER BY effective_season, effective_week;

COMMENT ON VIEW ml.ensemble_weight_evolution IS 'Time series of ensemble weight changes for Grafana dashboards';


-- ===========================================================================
-- VIEW 3: Top Features by Model
-- ===========================================================================
-- Current top-10 most important features for each model
-- ===========================================================================

CREATE OR REPLACE VIEW ml.top_features_current AS
WITH latest_computation AS (
    SELECT DISTINCT ON (model_name)
        model_name,
        computed_season,
        computed_week
    FROM ml.feature_importance_history
    ORDER BY model_name, computed_season DESC, computed_week DESC
)
SELECT
    fih.model_name,
    fih.computed_season,
    fih.computed_week,
    fih.feature_name,
    fih.feature_category,
    fih.composite_score,
    fih.importance_rank,
    fih.rank_percentile,
    fih.rank_change,
    fih.importance_trend
FROM ml.feature_importance_history fih
INNER JOIN latest_computation lc
    ON fih.model_name = lc.model_name
    AND fih.computed_season = lc.computed_season
    AND fih.computed_week = lc.computed_week
WHERE fih.importance_rank <= 10
ORDER BY fih.model_name, fih.importance_rank;

COMMENT ON VIEW ml.top_features_current IS 'Top 10 most important features for each model';


-- ===========================================================================
-- FUNCTION: Get Weight Change Between Weeks
-- ===========================================================================
-- Utility function to compare weights across time periods
-- ===========================================================================

CREATE OR REPLACE FUNCTION ml.get_weight_change(
    p_ensemble_version TEXT,
    p_season INTEGER,
    p_week_start INTEGER,
    p_week_end INTEGER
)
RETURNS TABLE (
    component TEXT,
    weight_start NUMERIC,
    weight_end NUMERIC,
    absolute_change NUMERIC,
    percent_change NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH start_weights AS (
        SELECT component_weights
        FROM ml.ensemble_weight_history
        WHERE ensemble_version = p_ensemble_version
          AND effective_season = p_season
          AND effective_week = p_week_start
        LIMIT 1
    ),
    end_weights AS (
        SELECT component_weights
        FROM ml.ensemble_weight_history
        WHERE ensemble_version = p_ensemble_version
          AND effective_season = p_season
          AND effective_week = p_week_end
        LIMIT 1
    )
    SELECT
        key::TEXT as component,
        (sw.component_weights->key)::NUMERIC as weight_start,
        (ew.component_weights->key)::NUMERIC as weight_end,
        ((ew.component_weights->key)::NUMERIC - (sw.component_weights->key)::NUMERIC) as absolute_change,
        (((ew.component_weights->key)::NUMERIC - (sw.component_weights->key)::NUMERIC) /
         NULLIF((sw.component_weights->key)::NUMERIC, 0) * 100) as percent_change
    FROM start_weights sw, end_weights ew,
         jsonb_object_keys(sw.component_weights) key
    ORDER BY ABS(((ew.component_weights->key)::NUMERIC - (sw.component_weights->key)::NUMERIC)) DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION ml.get_weight_change IS 'Compare ensemble weights between two weeks to track evolution';


-- ===========================================================================
-- SAMPLE DATA: Initialize with v6.0 baseline weights
-- ===========================================================================

INSERT INTO ml.ensemble_weight_history (
    ensemble_id,
    ensemble_version,
    effective_season,
    effective_week,
    training_start_week,
    training_end_week,
    training_window_size,
    state_space_weight,
    hierarchical_weight,
    xgboost_weight,
    informative_priors_weight,
    hybrid_weight,
    component_weights,
    optimization_method,
    validation_mae,
    validation_calibration,
    deployed_at,
    deployed_by,
    is_active,
    notes
)
SELECT
    ensemble_id,
    ensemble_version,
    2024,                      -- effective_season
    13,                         -- effective_week (current week)
    7,                          -- training_start_week (last 6 weeks)
    12,                         -- training_end_week
    6,                          -- training_window_size
    0.27,                       -- state_space_weight
    0.27,                       -- hierarchical_weight
    0.23,                       -- xgboost_weight
    0.13,                       -- informative_priors_weight
    0.10,                       -- hybrid_weight
    '{"state_space_v2": 0.27, "hierarchical_v2": 0.27, "xgboost_v4_ridge": 0.23, "informative_priors_v2": 0.13, "hybrid_static_incremental": 0.10}'::JSONB,
    'manual_baseline',          -- optimization_method
    NULL,                       -- validation_mae (not computed yet)
    NULL,                       -- validation_calibration
    CURRENT_TIMESTAMP,          -- deployed_at
    'migration_049',            -- deployed_by
    TRUE,                       -- is_active
    'v6.0 production baseline weights from November 15, 2025 deployment'
FROM ml.ensemble_versions
WHERE ensemble_version = '6.0'
  AND is_production = TRUE
ON CONFLICT (ensemble_version, effective_season, effective_week) DO NOTHING;


-- ===========================================================================
-- GRANT PERMISSIONS
-- ===========================================================================

GRANT SELECT, INSERT, UPDATE ON ml.ensemble_weight_history TO dro;
GRANT SELECT, INSERT, UPDATE ON ml.feature_importance_history TO dro;
GRANT SELECT ON ml.latest_ensemble_weights TO dro;
GRANT SELECT ON ml.ensemble_weight_evolution TO dro;
GRANT SELECT ON ml.top_features_current TO dro;
GRANT EXECUTE ON FUNCTION ml.get_weight_change TO dro;

-- ===========================================================================
-- MIGRATION COMPLETE
-- ===========================================================================
