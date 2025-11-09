-- Update model registry with canonical training window
-- Date: November 8, 2025
-- Purpose: Register state-space v2 model with approved status and canonical training window

-- ============================================================================
-- 1. Register state-space v2 model with canonical training window
-- ============================================================================

INSERT INTO ml.models (
    model_name,
    model_version,
    model_type,
    training_start_date,
    training_end_date,
    training_window_config,
    n_training_games,
    n_training_players,
    mae_2025_holdout,
    coverage_90_2025_holdout,
    status,
    approval_date,
    created_date,
    notes
) VALUES (
    'simplified_random_walk',
    'v2',
    'state_space',
    '2015-01-01',
    '2024-12-31',
    '2018-2024 full weight + 2015-2017 H=4 exponential decay',
    4504,
    91,
    47.77,
    86.5,
    'approved_production',
    '2025-11-08',
    '2025-11-08',
    'GO/NO-GO decision: APPROVED. Ablation study p=0.0069 vs v1 baseline. Canonical training window established via comprehensive empirical validation. Ready for 4-way ensemble integration. Training time: 14.2 min (MCMC, 4 chains). Convergence: Rhat=1.0021, ESS=1452, 0 divergences. Documentation: docs/STATE_SPACE_FINAL_DECISION.md, docs/TRAINING_WINDOW_ABLATION_RESULTS.md'
) ON CONFLICT (model_name, model_version) DO UPDATE SET
    status = EXCLUDED.status,
    approval_date = EXCLUDED.approval_date,
    mae_2025_holdout = EXCLUDED.mae_2025_holdout,
    coverage_90_2025_holdout = EXCLUDED.coverage_90_2025_holdout,
    notes = EXCLUDED.notes;

-- ============================================================================
-- 2. Create canonical training windows table for consistency across models
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml.canonical_training_windows (
    window_id SERIAL PRIMARY KEY,
    window_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    start_year INT NOT NULL,
    end_year INT NOT NULL,
    decay_formula TEXT,
    decay_halflife FLOAT,
    effective_sample_size FLOAT,
    validation_method TEXT,
    validation_date DATE,
    status VARCHAR(50),
    created_date TIMESTAMP DEFAULT NOW(),
    updated_date TIMESTAMP DEFAULT NOW()
);

-- Add comments for documentation
COMMENT ON TABLE ml.canonical_training_windows IS 'Canonical training windows for model consistency across project. Each window validated via ablation study.';
COMMENT ON COLUMN ml.canonical_training_windows.window_name IS 'Unique identifier for the training window configuration';
COMMENT ON COLUMN ml.canonical_training_windows.decay_formula IS 'Mathematical formula for exponential decay: w(s; t, H) = 0.5^((t-s)/H)';
COMMENT ON COLUMN ml.canonical_training_windows.decay_halflife IS 'Half-life parameter H for exponential decay: w(s; t, H) = 0.5^((t-s)/H)';
COMMENT ON COLUMN ml.canonical_training_windows.effective_sample_size IS 'Effective sample size accounting for decay weights: ESS = (Σw_i)² / Σw_i²';
COMMENT ON COLUMN ml.canonical_training_windows.validation_method IS 'Method used to validate this window (e.g., ablation study, cross-validation)';
COMMENT ON COLUMN ml.canonical_training_windows.status IS 'Status: active, deprecated, experimental';

-- ============================================================================
-- 3. Insert canonical training window v1 (established Nov 8, 2025)
-- ============================================================================

INSERT INTO ml.canonical_training_windows (
    window_name,
    description,
    start_year,
    end_year,
    decay_formula,
    decay_halflife,
    effective_sample_size,
    validation_method,
    validation_date,
    status
) VALUES (
    'canonical_v1_2025',
    'Empirically validated canonical training window for player performance models. Established via ablation study on 2025 weeks 1-6 hold-out (148 games, 37 QBs). Statistically significant improvement (p=0.0069) over 2020-2024 baseline. Core years: 2018-2024 (full weight). Decay years: 2015-2017 (exponential decay H=4). Formula: w(s; 2025, 4) = 0.5^((2025-s)/4). Effective weights: 2017=0.297, 2016=0.250, 2015=0.210.',
    2015,
    2024,
    'w(s; 2025, 4) = 0.5^((2025-s)/4)',
    4.0,
    4100.0,
    'Training window ablation study with Diebold-Mariano statistical test (4 scenarios: strict 2018-2024, 2018-2024+decay, 2020-2024 current, 2015-2024 dissertation)',
    '2025-11-08',
    'active'
) ON CONFLICT (window_name) DO UPDATE SET
    status = EXCLUDED.status,
    validation_date = EXCLUDED.validation_date,
    description = EXCLUDED.description,
    updated_date = NOW();

-- ============================================================================
-- 4. Add training window foreign key to models table (optional, for future)
-- ============================================================================

-- Check if column exists before adding
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'ml'
        AND table_name = 'models'
        AND column_name = 'canonical_window_id'
    ) THEN
        ALTER TABLE ml.models ADD COLUMN canonical_window_id INT REFERENCES ml.canonical_training_windows(window_id);
        COMMENT ON COLUMN ml.models.canonical_window_id IS 'Foreign key to canonical training window used for this model';
    END IF;
END $$;

-- ============================================================================
-- 5. Link state-space v2 to canonical window
-- ============================================================================

UPDATE ml.models
SET canonical_window_id = (
    SELECT window_id
    FROM ml.canonical_training_windows
    WHERE window_name = 'canonical_v1_2025'
)
WHERE model_name = 'simplified_random_walk'
  AND model_version = 'v2';

-- ============================================================================
-- 6. Verification queries
-- ============================================================================

-- Verify state-space v2 registration
SELECT
    model_name,
    model_version,
    status,
    mae_2025_holdout,
    coverage_90_2025_holdout,
    training_window_config,
    approval_date
FROM ml.models
WHERE model_name = 'simplified_random_walk'
ORDER BY created_date DESC;

-- Verify canonical window
SELECT
    window_name,
    start_year,
    end_year,
    decay_halflife,
    effective_sample_size,
    validation_date,
    status
FROM ml.canonical_training_windows
WHERE status = 'active'
ORDER BY validation_date DESC;

-- Verify linkage
SELECT
    m.model_name,
    m.model_version,
    m.status AS model_status,
    w.window_name,
    w.status AS window_status,
    w.decay_formula
FROM ml.models m
LEFT JOIN ml.canonical_training_windows w ON m.canonical_window_id = w.window_id
WHERE m.model_name = 'simplified_random_walk'
  AND m.model_version = 'v2';
