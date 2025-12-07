-- Migration 102: Dirichlet alpha parameters by stat type
-- Purpose: Store separate Dirichlet concentration parameters for each stat type
-- Enables per-stat-type adaptive stacking (e.g., BNN better at rushing, Hierarchical at passing)
--
-- Usage: psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -f db/migrations/102_dirichlet_alpha_by_stat_type.sql

BEGIN;

-- Ensure ml schema exists
CREATE SCHEMA IF NOT EXISTS ml;

-- Per-stat-type Dirichlet parameters for adaptive stacking
CREATE TABLE IF NOT EXISTS ml.dirichlet_alpha_by_stat_type (
    alpha_id SERIAL PRIMARY KEY,

    -- Stat type identification
    stat_type VARCHAR(50) NOT NULL,  -- 'passing_yards', 'rushing_yards', 'receiving_yards', 'spread', 'total'

    -- Ensemble component (model) identification
    component_name VARCHAR(100) NOT NULL,  -- 'hierarchical', 'state_space', 'xgboost', 'informative_priors', 'hybrid'
    component_order INTEGER NOT NULL DEFAULT 0,  -- Display/iteration order

    -- Dirichlet parameters
    alpha NUMERIC(12,6) NOT NULL,          -- Current concentration parameter
    prior_alpha NUMERIC(12,6) NOT NULL,    -- Original prior (for reset/shrinkage)

    -- Derived weight (for convenience)
    current_weight NUMERIC(8,6) GENERATED ALWAYS AS (
        -- Will be computed by trigger/function since we need sum across components
        NULL
    ) STORED,

    -- Learning metadata
    n_updates INTEGER DEFAULT 0,
    cumulative_likelihood NUMERIC(20,10) DEFAULT 0,
    last_mae NUMERIC(10,4),
    last_rmse NUMERIC(10,4),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT positive_alpha CHECK (alpha > 0),
    CONSTRAINT positive_prior_alpha CHECK (prior_alpha > 0),

    -- Unique constraint: one alpha per stat_type/component pair
    UNIQUE(stat_type, component_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_das_stat_type ON ml.dirichlet_alpha_by_stat_type(stat_type);
CREATE INDEX IF NOT EXISTS idx_das_component ON ml.dirichlet_alpha_by_stat_type(component_name);
CREATE INDEX IF NOT EXISTS idx_das_updated ON ml.dirichlet_alpha_by_stat_type(updated_at DESC);

-- Comments
COMMENT ON TABLE ml.dirichlet_alpha_by_stat_type IS 'Per-stat-type Dirichlet concentration parameters for adaptive ensemble stacking';
COMMENT ON COLUMN ml.dirichlet_alpha_by_stat_type.alpha IS 'Current Dirichlet concentration parameter (updated via Bayesian learning)';
COMMENT ON COLUMN ml.dirichlet_alpha_by_stat_type.prior_alpha IS 'Original prior alpha (from grid search or expert knowledge)';
COMMENT ON COLUMN ml.dirichlet_alpha_by_stat_type.n_updates IS 'Number of Bayesian updates applied to this component';

-- Function to compute weights from alphas for a stat type
CREATE OR REPLACE FUNCTION ml.get_dirichlet_weights(p_stat_type VARCHAR(50))
RETURNS TABLE(component_name VARCHAR(100), weight NUMERIC(8,6))
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    total_alpha NUMERIC;
BEGIN
    -- Get total alpha for normalization
    SELECT SUM(alpha) INTO total_alpha
    FROM ml.dirichlet_alpha_by_stat_type
    WHERE stat_type = p_stat_type;

    IF total_alpha IS NULL OR total_alpha = 0 THEN
        RAISE EXCEPTION 'No Dirichlet alphas found for stat_type: %', p_stat_type;
    END IF;

    -- Return normalized weights
    RETURN QUERY
    SELECT
        d.component_name,
        (d.alpha / total_alpha)::NUMERIC(8,6) as weight
    FROM ml.dirichlet_alpha_by_stat_type d
    WHERE d.stat_type = p_stat_type
    ORDER BY d.component_order;
END;
$$;

-- Function to update alpha after observing prediction outcome
CREATE OR REPLACE FUNCTION ml.update_dirichlet_alpha(
    p_stat_type VARCHAR(50),
    p_component_name VARCHAR(100),
    p_likelihood NUMERIC,  -- How well this component predicted (0-1, higher = better)
    p_learning_rate NUMERIC DEFAULT 1.0
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE ml.dirichlet_alpha_by_stat_type
    SET
        alpha = alpha + (p_likelihood * p_learning_rate),
        n_updates = n_updates + 1,
        cumulative_likelihood = cumulative_likelihood + p_likelihood,
        updated_at = NOW()
    WHERE stat_type = p_stat_type
      AND component_name = p_component_name;

    IF NOT FOUND THEN
        RAISE WARNING 'No alpha found for stat_type=%, component=%', p_stat_type, p_component_name;
    END IF;
END;
$$;

-- Function to reset alphas to priors (for experimentation)
CREATE OR REPLACE FUNCTION ml.reset_dirichlet_alphas(p_stat_type VARCHAR(50) DEFAULT NULL)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    IF p_stat_type IS NULL THEN
        UPDATE ml.dirichlet_alpha_by_stat_type
        SET alpha = prior_alpha, n_updates = 0, cumulative_likelihood = 0, updated_at = NOW();
    ELSE
        UPDATE ml.dirichlet_alpha_by_stat_type
        SET alpha = prior_alpha, n_updates = 0, cumulative_likelihood = 0, updated_at = NOW()
        WHERE stat_type = p_stat_type;
    END IF;

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$;

-- Initialize with default weights from v6.0 (games) and v8.0 (props)
-- These are the production weights from CLAUDE.md

-- Game outcome weights (v6.0): State-Space 27%, Hierarchical 27%, XGBoost 23%, Informative 13%, Hybrid 10%
INSERT INTO ml.dirichlet_alpha_by_stat_type (stat_type, component_name, component_order, alpha, prior_alpha)
VALUES
    ('spread', 'state_space', 1, 2.7, 2.7),
    ('spread', 'hierarchical', 2, 2.7, 2.7),
    ('spread', 'xgboost', 3, 2.3, 2.3),
    ('spread', 'informative_priors', 4, 1.3, 1.3),
    ('spread', 'hybrid', 5, 1.0, 1.0),
    ('total', 'state_space', 1, 2.7, 2.7),
    ('total', 'hierarchical', 2, 2.7, 2.7),
    ('total', 'xgboost', 3, 2.3, 2.3),
    ('total', 'informative_priors', 4, 1.3, 1.3),
    ('total', 'hybrid', 5, 1.0, 1.0),
    ('win_prob', 'state_space', 1, 2.7, 2.7),
    ('win_prob', 'hierarchical', 2, 2.7, 2.7),
    ('win_prob', 'xgboost', 3, 2.3, 2.3),
    ('win_prob', 'informative_priors', 4, 1.3, 1.3),
    ('win_prob', 'hybrid', 5, 1.0, 1.0)
ON CONFLICT (stat_type, component_name) DO NOTHING;

-- Player prop weights (v8.0): Hierarchical 39.6%, Informative 39.5%, Hybrid 10.1%, State-Space 8.2%, XGBoost 2.6%
INSERT INTO ml.dirichlet_alpha_by_stat_type (stat_type, component_name, component_order, alpha, prior_alpha)
VALUES
    ('passing_yards', 'hierarchical', 1, 3.96, 3.96),
    ('passing_yards', 'informative_priors', 2, 3.95, 3.95),
    ('passing_yards', 'hybrid', 3, 1.01, 1.01),
    ('passing_yards', 'state_space', 4, 0.82, 0.82),
    ('passing_yards', 'xgboost', 5, 0.26, 0.26),
    ('rushing_yards', 'hierarchical', 1, 3.96, 3.96),
    ('rushing_yards', 'informative_priors', 2, 3.95, 3.95),
    ('rushing_yards', 'hybrid', 3, 1.01, 1.01),
    ('rushing_yards', 'state_space', 4, 0.82, 0.82),
    ('rushing_yards', 'xgboost', 5, 0.26, 0.26),
    ('receiving_yards', 'hierarchical', 1, 3.96, 3.96),
    ('receiving_yards', 'informative_priors', 2, 3.95, 3.95),
    ('receiving_yards', 'hybrid', 3, 1.01, 1.01),
    ('receiving_yards', 'state_space', 4, 0.82, 0.82),
    ('receiving_yards', 'xgboost', 5, 0.26, 0.26),
    ('receptions', 'hierarchical', 1, 3.96, 3.96),
    ('receptions', 'informative_priors', 2, 3.95, 3.95),
    ('receptions', 'hybrid', 3, 1.01, 1.01),
    ('receptions', 'state_space', 4, 0.82, 0.82),
    ('receptions', 'xgboost', 5, 0.26, 0.26),
    ('passing_tds', 'hierarchical', 1, 3.96, 3.96),
    ('passing_tds', 'informative_priors', 2, 3.95, 3.95),
    ('passing_tds', 'hybrid', 3, 1.01, 1.01),
    ('passing_tds', 'state_space', 4, 0.82, 0.82),
    ('passing_tds', 'xgboost', 5, 0.26, 0.26),
    ('rushing_tds', 'hierarchical', 1, 3.96, 3.96),
    ('rushing_tds', 'informative_priors', 2, 3.95, 3.95),
    ('rushing_tds', 'hybrid', 3, 1.01, 1.01),
    ('rushing_tds', 'state_space', 4, 0.82, 0.82),
    ('rushing_tds', 'xgboost', 5, 0.26, 0.26)
ON CONFLICT (stat_type, component_name) DO NOTHING;

-- View for current weights by stat type
CREATE OR REPLACE VIEW ml.current_dirichlet_weights AS
WITH totals AS (
    SELECT stat_type, SUM(alpha) as total_alpha
    FROM ml.dirichlet_alpha_by_stat_type
    GROUP BY stat_type
)
SELECT
    d.stat_type,
    d.component_name,
    d.alpha,
    d.prior_alpha,
    (d.alpha / t.total_alpha)::NUMERIC(8,6) as weight,
    (d.prior_alpha / t.total_alpha)::NUMERIC(8,6) as prior_weight,
    d.n_updates,
    d.updated_at
FROM ml.dirichlet_alpha_by_stat_type d
JOIN totals t ON d.stat_type = t.stat_type
ORDER BY d.stat_type, d.component_order;

-- Record migration
INSERT INTO schema_migrations (version, description)
VALUES ('102_dirichlet_alpha_by_stat_type', 'Per-stat-type Dirichlet concentration parameters for adaptive stacking')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- Verification
DO $$
DECLARE
    stat_count INTEGER;
    component_count INTEGER;
BEGIN
    SELECT COUNT(DISTINCT stat_type), COUNT(*) INTO stat_count, component_count
    FROM ml.dirichlet_alpha_by_stat_type;

    RAISE NOTICE 'Migration 102 complete: % stat types, % total components', stat_count, component_count;
    RAISE NOTICE 'Functions created: ml.get_dirichlet_weights(), ml.update_dirichlet_alpha(), ml.reset_dirichlet_alphas()';
    RAISE NOTICE 'View created: ml.current_dirichlet_weights';
END $$;
