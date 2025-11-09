-- Migration: Fix Model Registry Convergence Status and File Paths
-- Date: November 9, 2025
-- Author: Model Management Agent
-- Purpose: Correct file paths, add convergence metrics, update production status
-- Reference: docs/milestones/MODEL_REGISTRY_VALIDATION.md
-- Database: devdb01 (PostgreSQL 14+)

BEGIN;

-- =============================================================================
-- STEP 1: Add convergence_metrics column to ml.model_registry
-- =============================================================================

ALTER TABLE ml.model_registry
ADD COLUMN IF NOT EXISTS convergence_metrics JSONB;

COMMENT ON COLUMN ml.model_registry.convergence_metrics IS
'Convergence diagnostics for Bayesian models: {"rhat_max": float, "ess_min": int, "divergences": int, "status": "CONVERGED"|"FAILED"}';

-- =============================================================================
-- STEP 2: Add performance_metrics column to ml.model_registry
-- =============================================================================

ALTER TABLE ml.model_registry
ADD COLUMN IF NOT EXISTS performance_metrics JSONB;

COMMENT ON COLUMN ml.model_registry.performance_metrics IS
'Model performance metrics: {"mae_yards": float, "rmse_yards": float, "calibration_90pct": float, "test_games": int}';

-- =============================================================================
-- STEP 3: Correct file paths (4 corrections)
-- =============================================================================

-- Fix state_space_v2 file path
UPDATE ml.model_registry
SET
    model_path = 'models/bayesian/simplified_random_walk_v2.rds',
    notes = REGEXP_REPLACE(notes, '⚠️ File not found:.*', '', 'g'),
    updated_at = NOW()
WHERE model_name = 'state_space_v2';

-- Fix hierarchical_v2 file path
UPDATE ml.model_registry
SET
    model_path = 'models/bayesian/passing_yards_hierarchical_v2.rds',
    notes = REGEXP_REPLACE(notes, '⚠️ File not found:.*', '', 'g'),
    updated_at = NOW()
WHERE model_name = 'hierarchical_v2';

-- Fix xgboost_baseline file path (and rename to xgboost_v2)
UPDATE ml.model_registry
SET
    model_name = 'xgboost_v2',
    model_path = 'models/xgboost/passing_xgboost_v2_canonical.pkl',
    status = 'production',
    notes = REGEXP_REPLACE(notes, '⚠️ File not found:.*', '', 'g') || E'\n✅ Promoted to production Nov 9, 2025. Weight increased to 0.45 after hierarchical_v2 exclusion.',
    updated_at = NOW()
WHERE model_name = 'xgboost_baseline';

-- =============================================================================
-- STEP 4: Add convergence metrics for Bayesian models
-- =============================================================================

-- state_space_v2: CONVERGED
UPDATE ml.model_registry
SET
    convergence_metrics = jsonb_build_object(
        'rhat_max', 1.0013,
        'ess_min', 2218,
        'divergences', 0,
        'status', 'CONVERGED',
        'notes', 'Max Rhat well below 1.01 threshold. No divergences. Approved for production Nov 8, 2025.'
    ),
    updated_at = NOW()
WHERE model_name = 'state_space_v2';

-- hierarchical_v2: FAILED
UPDATE ml.model_registry
SET
    convergence_metrics = jsonb_build_object(
        'rhat_max', 1.0337,
        'ess_min', 5316,
        'divergences', 107,
        'divergence_pct', 2.675,  -- 107/4000
        'status', 'FAILED',
        'reason', 'Rhat exceeds 1.01 threshold by 3.37%',
        'notes', 'Requires retraining with increased warmup iterations before production use.'
    ),
    updated_at = NOW()
WHERE model_name = 'hierarchical_v2';

-- informative_priors_v2: SEVERE FAILURE
UPDATE ml.model_registry
SET
    convergence_metrics = jsonb_build_object(
        'rhat_max', 1.0995,
        'ess_min', 5808,
        'divergences', 257,
        'divergence_pct', 6.425,  -- 257/4000
        'status', 'FAILED',
        'severity', 'SEVERE',
        'reason', 'Rhat exceeds 1.01 threshold by 9.95%. High divergence rate.',
        'notes', 'Requires major rework (simplify priors, increase chains) before production use.'
    ),
    updated_at = NOW()
WHERE model_name = 'informative_priors_v2';

-- =============================================================================
-- STEP 5: Update production status for convergence failures
-- =============================================================================

-- Demote hierarchical_v2 to staging
UPDATE ml.model_registry
SET
    status = 'staging',
    is_production = FALSE,
    notes = notes || E'\n\n⚠️ CONVERGENCE FAILURE (Nov 9, 2025): Max Rhat 1.0337 > 1.01 threshold.\nExcluded from production ensemble until retraining with improved convergence.',
    updated_at = NOW()
WHERE model_name = 'hierarchical_v2';

-- Ensure informative_priors_v2 is staging (already was, but add convergence note)
UPDATE ml.model_registry
SET
    status = 'staging',
    is_production = FALSE,
    notes = notes || E'\n\n⚠️ SEVERE CONVERGENCE FAILURE (Nov 9, 2025): Max Rhat 1.0995, divergences 6.0%.\nRequires major rework before production consideration.',
    updated_at = NOW()
WHERE model_name = 'informative_priors_v2';

-- =============================================================================
-- STEP 6: Add performance metrics for all production models
-- =============================================================================

-- state_space_v2
UPDATE ml.model_registry
SET
    performance_metrics = jsonb_build_object(
        'mae_yards', 47.77,
        'rmse_yards', 62.34,
        'calibration_90pct', 88.5,
        'test_games', 148,
        'p_value', 0.0069,
        'approved_date', '2025-11-08'
    ),
    updated_at = NOW()
WHERE model_name = 'state_space_v2';

-- bnn_multioutput_v3_1
UPDATE ml.model_registry
SET
    performance_metrics = jsonb_build_object(
        'mae_yards', 58.70,
        'rmse_yards', 76.12,
        'calibration_90pct', 89.5,
        'temperature_scaling', 0.75,
        'test_games', 562
    ),
    updated_at = NOW()
WHERE model_name = 'bnn_multioutput_v3_1';

-- hierarchical_v2 (even though staging, keep metrics for reference)
UPDATE ml.model_registry
SET
    performance_metrics = jsonb_build_object(
        'mae_yards', 52.34,
        'rmse_yards', 68.91,
        'calibration_90pct', 87.2,
        'test_games', 562,
        'note', 'Performance metrics from pre-convergence failure evaluation'
    ),
    updated_at = NOW()
WHERE model_name = 'hierarchical_v2';

-- xgboost_v2 (promoted to production)
UPDATE ml.model_registry
SET
    performance_metrics = jsonb_build_object(
        'mae_yards', 57.16,
        'rmse_yards', 73.45,
        'test_games', 182,
        'validation_set', '2025 weeks 1-7 hold-out'
    ),
    is_production = TRUE,
    updated_at = NOW()
WHERE model_name = 'xgboost_v2';

-- cql_bet_sizer
UPDATE ml.model_registry
SET
    performance_metrics = jsonb_build_object(
        'improvement_vs_baseline', 0.24,  -- +24%
        'policy_reward', 1.75,
        'baseline_reward', 1.41,
        'test_games', 1029
    ),
    updated_at = NOW()
WHERE model_name = 'cql_bet_sizer';

-- =============================================================================
-- STEP 7: Insert performance evaluations into ml.model_evaluations
-- =============================================================================

INSERT INTO ml.model_evaluations (
    model_id,
    evaluation_name,
    evaluation_type,
    dataset_name,
    dataset_size,
    metrics,
    notes
)
SELECT
    mr.model_id,
    'canonical_window_validation',
    'hold_out_test',
    'QB Passing Yards 2025 Weeks 1-7',
    CASE
        WHEN mr.model_name = 'state_space_v2' THEN 148
        WHEN mr.model_name = 'bnn_multioutput_v3_1' THEN 562
        WHEN mr.model_name = 'hierarchical_v2' THEN 562
        WHEN mr.model_name = 'xgboost_v2' THEN 182
    END,
    jsonb_build_object(
        'mae_yards', (mr.performance_metrics->>'mae_yards')::float,
        'rmse_yards', (mr.performance_metrics->>'rmse_yards')::float,
        'calibration_90pct', (mr.performance_metrics->>'calibration_90pct')::float
    ),
    'Performance metrics from November 2025 canonical window training. Source: MODEL_REGISTRY_VALIDATION.md'
FROM ml.model_registry mr
WHERE mr.model_name IN ('state_space_v2', 'hierarchical_v2', 'bnn_multioutput_v3_1', 'xgboost_v2')
ON CONFLICT DO NOTHING;  -- Prevent duplicate inserts if re-run

-- =============================================================================
-- STEP 8: Create convergence validation view
-- =============================================================================

CREATE OR REPLACE VIEW ml.model_convergence_status AS
SELECT
    mr.model_id,
    mr.model_name,
    mr.model_type,
    mr.status,
    mr.is_production,
    mr.convergence_metrics->>'status' as convergence_status,
    (mr.convergence_metrics->>'rhat_max')::float as rhat_max,
    (mr.convergence_metrics->>'ess_min')::int as ess_min,
    (mr.convergence_metrics->>'divergences')::int as divergences,
    (mr.convergence_metrics->>'divergence_pct')::float as divergence_pct,
    CASE
        WHEN mr.convergence_metrics->>'status' = 'CONVERGED' THEN '✅ PASS'
        WHEN mr.convergence_metrics->>'status' = 'FAILED' THEN '❌ FAIL'
        ELSE 'N/A'
    END as convergence_gate,
    mr.updated_at
FROM ml.model_registry mr
WHERE mr.model_type IN ('hierarchical_bayesian', 'state_space')
ORDER BY
    CASE mr.status WHEN 'production' THEN 1 WHEN 'staging' THEN 2 ELSE 3 END,
    mr.model_name;

COMMENT ON VIEW ml.model_convergence_status IS
'Real-time convergence validation for all Bayesian models. RED FLAG if production model has convergence_status = FAILED.';

-- =============================================================================
-- STEP 9: Create production readiness check function
-- =============================================================================

CREATE OR REPLACE FUNCTION ml.check_production_readiness(p_model_name text)
RETURNS TABLE(
    check_name text,
    status text,
    details text
) AS $$
BEGIN
    RETURN QUERY
    WITH model_checks AS (
        SELECT
            mr.model_name,
            mr.status,
            mr.model_path,
            mr.convergence_metrics,
            mr.performance_metrics
        FROM ml.model_registry mr
        WHERE mr.model_name = p_model_name
    )
    SELECT
        'Model Status'::text,
        CASE WHEN mc.status = 'production' THEN '✅ PASS' ELSE '❌ FAIL' END,
        'Status: ' || mc.status
    FROM model_checks mc

    UNION ALL

    SELECT
        'File Exists'::text,
        CASE WHEN mc.model_path IS NOT NULL THEN '⚠️ MANUAL CHECK' ELSE '❌ FAIL' END,
        'Path: ' || COALESCE(mc.model_path, 'NULL')
    FROM model_checks mc

    UNION ALL

    SELECT
        'Convergence (Bayesian)'::text,
        CASE
            WHEN mc.convergence_metrics->>'status' = 'CONVERGED' THEN '✅ PASS'
            WHEN mc.convergence_metrics->>'status' = 'FAILED' THEN '❌ FAIL'
            ELSE '⚠️ N/A (not Bayesian)'
        END,
        COALESCE(
            'Rhat: ' || (mc.convergence_metrics->>'rhat_max') || ', ESS: ' || (mc.convergence_metrics->>'ess_min'),
            'No convergence metrics'
        )
    FROM model_checks mc

    UNION ALL

    SELECT
        'Performance Metrics'::text,
        CASE WHEN mc.performance_metrics IS NOT NULL THEN '✅ PASS' ELSE '❌ FAIL' END,
        COALESCE(
            'MAE: ' || (mc.performance_metrics->>'mae_yards') || ' yards',
            'No performance metrics'
        )
    FROM model_checks mc;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION ml.check_production_readiness IS
'Quick production readiness check for a model. Usage: SELECT * FROM ml.check_production_readiness(''state_space_v2'');';

-- =============================================================================
-- STEP 10: Add migration metadata
-- =============================================================================

INSERT INTO ml.model_registry (
    model_name,
    model_type,
    model_family,
    version,
    model_path,
    status,
    description
) VALUES (
    'migration_033_convergence_fix',
    'metadata',
    'system',
    '1.0',
    'db/migrations/033_fix_convergence_status.sql',
    'completed',
    'Fixed convergence status and file paths for production models. See MODEL_REGISTRY_VALIDATION.md for details.'
) ON CONFLICT (model_name) DO NOTHING;

-- =============================================================================
-- VERIFICATION QUERIES (run these after migration)
-- =============================================================================

-- Verify convergence metrics populated
SELECT
    model_name,
    status,
    convergence_metrics->'status' as convergence_status,
    convergence_metrics->'rhat_max' as rhat_max,
    convergence_metrics->'ess_min' as ess_min
FROM ml.model_registry
WHERE model_type IN ('hierarchical_bayesian', 'state_space')
ORDER BY status, model_name;

-- Verify file paths corrected (manually check files exist)
SELECT
    model_name,
    model_path,
    CASE
        WHEN notes LIKE '%File not found%' THEN '❌ Still has error'
        ELSE '✅ Clean'
    END as notes_status
FROM ml.model_registry
WHERE status IN ('production', 'staging')
ORDER BY status, model_name;

-- Verify performance metrics populated
SELECT
    mr.model_name,
    mr.status,
    mr.performance_metrics->>'mae_yards' as mae_yards,
    mr.performance_metrics->>'calibration_90pct' as calibration,
    COUNT(me.evaluation_id) as evaluation_count
FROM ml.model_registry mr
LEFT JOIN ml.model_evaluations me ON mr.model_id = me.model_id
WHERE mr.status = 'production'
GROUP BY mr.model_name, mr.status, mr.performance_metrics
ORDER BY mr.model_name;

-- Check production readiness for all models
SELECT
    mr.model_name,
    mr.status,
    CASE WHEN mr.convergence_metrics->>'status' = 'FAILED' THEN '🚨 BLOCKED' ELSE '✅ READY' END as production_gate
FROM ml.model_registry mr
WHERE mr.status = 'production'
ORDER BY production_gate DESC, mr.model_name;

COMMIT;

-- =============================================================================
-- POST-MIGRATION NOTES
-- =============================================================================

/*
SUMMARY OF CHANGES:

1. Added convergence_metrics and performance_metrics JSONB columns
2. Corrected 3 file paths:
   - state_space_v2: ...canonical.rds → simplified_random_walk_v2.rds
   - hierarchical_v2: ...canonical.rds → passing_yards_hierarchical_v2.rds
   - xgboost_baseline → xgboost_v2, ...v2.pkl → ...v2_canonical.pkl

3. Updated convergence status:
   - state_space_v2: CONVERGED (Rhat 1.0013) ✅
   - hierarchical_v2: FAILED (Rhat 1.0337) → demoted to staging ❌
   - informative_priors_v2: FAILED (Rhat 1.0995) → remains staging ❌

4. Promoted xgboost_v2 to production with increased weight (0.45)

5. Populated performance metrics for all 5 core models

6. Created ml.model_convergence_status view for real-time monitoring

7. Created ml.check_production_readiness() function for validation

NEXT STEPS:

1. Run verification queries above to confirm migration success
2. Manually verify file paths exist (ls -lh models/bayesian/*.rds)
3. Update enhanced_ensemble_v5.py to match new 3-way ensemble
4. Retrain hierarchical_v2 with increased warmup (2000 → 4000 iterations)
5. Full 3-way ensemble backtest (2022-2024) to validate performance

ROLLBACK (if needed):

To rollback this migration, run:

    BEGIN;
    ALTER TABLE ml.model_registry DROP COLUMN IF EXISTS convergence_metrics;
    ALTER TABLE ml.model_registry DROP COLUMN IF EXISTS performance_metrics;
    DROP VIEW IF EXISTS ml.model_convergence_status;
    DROP FUNCTION IF EXISTS ml.check_production_readiness(text);
    DELETE FROM ml.model_evaluations WHERE notes LIKE '%MODEL_REGISTRY_VALIDATION%';
    DELETE FROM ml.model_registry WHERE model_name = 'migration_033_convergence_fix';
    -- Then restore from backup or manually revert status/path changes
    COMMIT;
*/
