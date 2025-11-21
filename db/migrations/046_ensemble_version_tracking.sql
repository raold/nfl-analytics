-- Migration 046: Ensemble Version Tracking and Validation Status
-- =================================================================
--
-- Purpose: Track ensemble configurations (v6, v7, etc.) with their
--          component model compositions, weights, and validation results
--
-- Tables Created:
-- 1. ml.ensemble_versions - Ensemble configuration tracking
-- 2. ml.ensemble_components - Component models and weights
-- 3. ml.ensemble_validation_results - Multi-season validation metrics
--
-- Author: Claude Code (Sonnet 4.5)
-- Date: November 19, 2025

-- ============================================================================
-- 1. ENSEMBLE VERSIONS TABLE
-- ============================================================================
-- Tracks different ensemble configurations (v6_baseline, v7_full, etc.)

CREATE TABLE IF NOT EXISTS ml.ensemble_versions (
    ensemble_id SERIAL PRIMARY KEY,
    ensemble_name TEXT NOT NULL UNIQUE,  -- e.g., "v6_baseline", "v7_full"
    ensemble_version TEXT NOT NULL,       -- e.g., "6.0", "7.0"
    description TEXT,

    -- Configuration flags
    enable_meta_learning BOOLEAN NOT NULL DEFAULT FALSE,
    enable_wls_recency BOOLEAN NOT NULL DEFAULT FALSE,
    enable_temperature_scaling BOOLEAN NOT NULL DEFAULT TRUE,

    -- WLS parameters
    wls_halflife REAL DEFAULT 4.0,        -- Exponential decay halflife (weeks)
    wls_min_history INTEGER DEFAULT 10,   -- Minimum games for WLS

    -- Meta-learning parameters
    meta_model_path TEXT,                 -- Path to meta-model file
    meta_blend_ratio REAL DEFAULT 0.6,    -- Meta vs WLS blend (0.6 = 60% meta)

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'development',  -- development, staging, production, deprecated
    validation_status TEXT,               -- pending, running, complete, failed
    is_production BOOLEAN DEFAULT FALSE,
    deployment_date TIMESTAMP WITH TIME ZONE,

    -- Expected performance (from component estimates)
    expected_mae REAL,
    expected_rmse REAL,
    expected_calibration REAL,

    -- Metadata
    created_by TEXT DEFAULT 'system',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT
);

-- Indexes
CREATE INDEX idx_ensemble_versions_status ON ml.ensemble_versions(status);
CREATE INDEX idx_ensemble_versions_production ON ml.ensemble_versions(is_production);
CREATE INDEX idx_ensemble_versions_validation_status ON ml.ensemble_versions(validation_status);

-- ============================================================================
-- 2. ENSEMBLE COMPONENTS TABLE
-- ============================================================================
-- Tracks which component models are in each ensemble and their weights

CREATE TABLE IF NOT EXISTS ml.ensemble_components (
    component_id SERIAL PRIMARY KEY,
    ensemble_id INTEGER NOT NULL REFERENCES ml.ensemble_versions(ensemble_id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES ml.model_registry(model_id) ON DELETE RESTRICT,

    -- Component details
    component_name TEXT NOT NULL,         -- e.g., "state_space_v2", "xgboost_v4_ridge"
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),

    -- Adaptive weights (for v7+)
    use_adaptive_weights BOOLEAN DEFAULT FALSE,
    min_weight REAL DEFAULT 0.0,
    max_weight REAL DEFAULT 1.0,

    -- Component-specific parameters
    parameters JSONB,                     -- Flexible parameter storage

    -- Performance metrics (from component validation)
    component_mae REAL,
    component_rmse REAL,
    component_r2 REAL,
    component_calibration REAL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(ensemble_id, model_id),
    UNIQUE(ensemble_id, component_name)
);

-- Indexes
CREATE INDEX idx_ensemble_components_ensemble ON ml.ensemble_components(ensemble_id);
CREATE INDEX idx_ensemble_components_model ON ml.ensemble_components(model_id);

-- Trigger to validate weights sum to 1.0 (within tolerance)
CREATE OR REPLACE FUNCTION ml.validate_ensemble_weights()
RETURNS TRIGGER AS $$
DECLARE
    weights_sum REAL;
BEGIN
    -- Calculate sum of weights for this ensemble
    SELECT SUM(weight) INTO weights_sum
    FROM ml.ensemble_components
    WHERE ensemble_id = NEW.ensemble_id;

    -- Allow tolerance of 0.01 (1%) for rounding
    IF weights_sum > 1.01 THEN
        RAISE EXCEPTION 'Component weights exceed 1.0 (sum = %)', weights_sum;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ensemble_components_validate_weights
    AFTER INSERT OR UPDATE ON ml.ensemble_components
    FOR EACH ROW
    EXECUTE FUNCTION ml.validate_ensemble_weights();

-- ============================================================================
-- 3. ENSEMBLE VALIDATION RESULTS TABLE
-- ============================================================================
-- Stores multi-season validation results for ensemble configurations

CREATE TABLE IF NOT EXISTS ml.ensemble_validation_results (
    validation_id SERIAL PRIMARY KEY,
    ensemble_id INTEGER NOT NULL REFERENCES ml.ensemble_versions(ensemble_id) ON DELETE CASCADE,

    -- Validation scope
    validation_name TEXT NOT NULL,        -- e.g., "2022-2024_full", "2024_w10-12_quick"
    season INTEGER NOT NULL,
    week_start INTEGER,
    week_end INTEGER,

    -- Validation metrics
    n_predictions INTEGER NOT NULL,
    n_failures INTEGER DEFAULT 0,

    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    r2 REAL NOT NULL,

    -- Calibration metrics
    calibration_90 REAL,                  -- % within 90% CI
    calibration_50 REAL,                  -- % within 50% CI

    -- Distribution metrics
    mean_prediction REAL,
    mean_actual REAL,
    mean_uncertainty REAL,

    -- Tail risk metrics
    p95_error REAL,                       -- 95th percentile absolute error
    p99_error REAL,                       -- 99th percentile absolute error
    max_error REAL,

    -- Execution metrics
    inference_time_avg_seconds REAL,
    inference_time_p95_seconds REAL,
    total_validation_time_seconds REAL,

    -- Metadata
    validation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    hardware TEXT,                        -- e.g., "Mac M4", "AWS r6g.2xlarge"
    git_commit_hash TEXT,
    notes TEXT
);

-- Indexes
CREATE INDEX idx_ensemble_validation_ensemble ON ml.ensemble_validation_results(ensemble_id);
CREATE INDEX idx_ensemble_validation_season ON ml.ensemble_validation_results(season);
CREATE INDEX idx_ensemble_validation_date ON ml.ensemble_validation_results(validation_date DESC);

-- Unique constraint: One validation per ensemble+season+validation_name
CREATE UNIQUE INDEX idx_ensemble_validation_unique
    ON ml.ensemble_validation_results(ensemble_id, season, validation_name);

-- ============================================================================
-- 4. STATISTICAL COMPARISON TABLE
-- ============================================================================
-- Stores pairwise statistical comparisons (Diebold-Mariano tests)

CREATE TABLE IF NOT EXISTS ml.ensemble_comparisons (
    comparison_id SERIAL PRIMARY KEY,

    -- Ensembles being compared
    ensemble_id_1 INTEGER NOT NULL REFERENCES ml.ensemble_versions(ensemble_id) ON DELETE CASCADE,
    ensemble_id_2 INTEGER NOT NULL REFERENCES ml.ensemble_versions(ensemble_id) ON DELETE CASCADE,

    -- Comparison scope
    season INTEGER NOT NULL,
    validation_name TEXT NOT NULL,

    -- MAE comparison
    mae_1 REAL NOT NULL,
    mae_2 REAL NOT NULL,
    mae_difference REAL NOT NULL,        -- mae_2 - mae_1 (negative = ensemble_2 better)

    -- Diebold-Mariano test
    dm_statistic REAL NOT NULL,
    p_value REAL NOT NULL,
    significant_01 BOOLEAN,              -- Significant at p < 0.01
    significant_05 BOOLEAN,              -- Significant at p < 0.05

    -- Other metrics comparison
    rmse_1 REAL,
    rmse_2 REAL,
    r2_1 REAL,
    r2_2 REAL,
    calibration_90_1 REAL,
    calibration_90_2 REAL,

    -- Metadata
    comparison_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT
);

-- Indexes
CREATE INDEX idx_ensemble_comparisons_ensemble1 ON ml.ensemble_comparisons(ensemble_id_1);
CREATE INDEX idx_ensemble_comparisons_ensemble2 ON ml.ensemble_comparisons(ensemble_id_2);
CREATE INDEX idx_ensemble_comparisons_season ON ml.ensemble_comparisons(season);
CREATE INDEX idx_ensemble_comparisons_significance ON ml.ensemble_comparisons(significant_05);

-- ============================================================================
-- 5. INSERT HISTORICAL ENSEMBLE VERSIONS
-- ============================================================================

-- v6.0 Baseline (Static Weights)
INSERT INTO ml.ensemble_versions (
    ensemble_name,
    ensemble_version,
    description,
    enable_meta_learning,
    enable_wls_recency,
    status,
    validation_status,
    is_production,
    deployment_date,
    expected_mae,
    expected_calibration,
    created_by,
    notes
) VALUES (
    'v6_baseline',
    '6.0',
    'Production ensemble with static weights (30% SS, 27% Hier, 23% XGB v4, 12% IP, 8% ELO)',
    FALSE,
    FALSE,
    'production',
    'pending',
    TRUE,
    '2025-11-15 08:00:00-08',
    35.0,
    90.0,
    'claude_sonnet_4.5',
    'Deployed Nov 15, 2025. A/B test vs v5.1 (Nov 14 - Dec 12, 2025)'
)
ON CONFLICT (ensemble_name) DO NOTHING;

-- v7.0 Meta-Learning Only
INSERT INTO ml.ensemble_versions (
    ensemble_name,
    ensemble_version,
    description,
    enable_meta_learning,
    enable_wls_recency,
    meta_blend_ratio,
    status,
    validation_status,
    expected_mae,
    created_by,
    notes
) VALUES (
    'v7_meta_only',
    '7.0',
    'Adaptive ensemble with meta-learning (XGBoost weight prediction)',
    TRUE,
    FALSE,
    1.0,  -- 100% meta-learning
    'staging',
    'running',
    33.0,
    'claude_sonnet_4.5',
    'Multi-season validation: Nov 18-19, 2025 (16 hours, 1,701 games)'
)
ON CONFLICT (ensemble_name) DO NOTHING;

-- v7.0 WLS Recency Only
INSERT INTO ml.ensemble_versions (
    ensemble_name,
    ensemble_version,
    description,
    enable_meta_learning,
    enable_wls_recency,
    wls_halflife,
    wls_min_history,
    meta_blend_ratio,
    status,
    validation_status,
    expected_mae,
    created_by,
    notes
) VALUES (
    'v7_wls_only',
    '7.0',
    'Adaptive ensemble with WLS recency adjustment (halflife=4 weeks)',
    FALSE,
    TRUE,
    4.0,
    10,
    0.0,  -- 0% meta, 100% WLS
    'staging',
    'running',
    32.5,
    'claude_sonnet_4.5',
    'Multi-season validation: Nov 18-19, 2025 (16 hours, 1,701 games)'
)
ON CONFLICT (ensemble_name) DO NOTHING;

-- v7.0 Full (Meta + WLS)
INSERT INTO ml.ensemble_versions (
    ensemble_name,
    ensemble_version,
    description,
    enable_meta_learning,
    enable_wls_recency,
    wls_halflife,
    wls_min_history,
    meta_blend_ratio,
    status,
    validation_status,
    expected_mae,
    created_by,
    notes
) VALUES (
    'v7_full',
    '7.0',
    'Full adaptive ensemble with meta-learning + WLS recency (60/40 blend)',
    TRUE,
    TRUE,
    4.0,
    10,
    0.6,  -- 60% meta, 40% WLS
    'staging',
    'running',
    32.0,
    'claude_sonnet_4.5',
    'Multi-season validation: Nov 18-19, 2025 (16 hours, 1,701 games)'
)
ON CONFLICT (ensemble_name) DO NOTHING;

-- ============================================================================
-- 6. GRANT PERMISSIONS
-- ============================================================================

GRANT SELECT, INSERT, UPDATE ON ml.ensemble_versions TO dro;
GRANT SELECT, INSERT, UPDATE ON ml.ensemble_components TO dro;
GRANT SELECT, INSERT, UPDATE, DELETE ON ml.ensemble_validation_results TO dro;
GRANT SELECT, INSERT, UPDATE ON ml.ensemble_comparisons TO dro;

GRANT USAGE ON SEQUENCE ml.ensemble_versions_ensemble_id_seq TO dro;
GRANT USAGE ON SEQUENCE ml.ensemble_components_component_id_seq TO dro;
GRANT USAGE ON SEQUENCE ml.ensemble_validation_results_validation_id_seq TO dro;
GRANT USAGE ON SEQUENCE ml.ensemble_comparisons_comparison_id_seq TO dro;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Verify migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 046 complete: Ensemble version tracking tables created';
    RAISE NOTICE '  - ml.ensemble_versions (4 initial versions inserted)';
    RAISE NOTICE '  - ml.ensemble_components';
    RAISE NOTICE '  - ml.ensemble_validation_results';
    RAISE NOTICE '  - ml.ensemble_comparisons';
END $$;
