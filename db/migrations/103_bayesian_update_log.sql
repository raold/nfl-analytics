-- Migration 103: Bayesian update audit log
-- Purpose: Track all Bayesian updates for transparency, debugging, and analysis
-- Records prior -> likelihood -> posterior transitions for both player priors and ensemble weights
--
-- Usage: psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -f db/migrations/103_bayesian_update_log.sql

BEGIN;

-- Ensure ml schema exists
CREATE SCHEMA IF NOT EXISTS ml;

-- Audit log for all Bayesian updates
CREATE TABLE IF NOT EXISTS ml.bayesian_update_log (
    log_id BIGSERIAL PRIMARY KEY,

    -- Context identification
    update_type VARCHAR(30) NOT NULL,  -- 'player_prior', 'ensemble_weight', 'dirichlet_alpha'
    player_id VARCHAR(50),              -- For player-level updates
    game_id VARCHAR(20),                -- Game that triggered this update
    stat_type VARCHAR(50) NOT NULL,     -- 'passing_yards', 'rushing_yards', etc.

    -- Temporal context
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    -- Bayesian update components (stored as JSON for flexibility)
    prior_params JSONB NOT NULL,        -- Prior distribution parameters
    likelihoods JSONB NOT NULL,         -- Likelihood contributions by component
    posterior_params JSONB NOT NULL,    -- Posterior distribution parameters

    -- Observed vs predicted (for analysis)
    actual_value NUMERIC(10,2),
    ensemble_prediction NUMERIC(10,2),
    prediction_error NUMERIC(10,2),     -- actual - predicted
    abs_error NUMERIC(10,2),            -- |actual - predicted|

    -- Component-level predictions (for debugging)
    component_predictions JSONB,        -- {hierarchical: 250.5, xgboost: 245.3, ...}

    -- Update metadata
    update_magnitude NUMERIC(10,6),     -- KL divergence or similar
    learning_rate_used NUMERIC(6,4) DEFAULT 1.0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Indexes inline
    CONSTRAINT valid_update_type CHECK (update_type IN ('player_prior', 'ensemble_weight', 'dirichlet_alpha'))
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_bul_game ON ml.bayesian_update_log(game_id);
CREATE INDEX IF NOT EXISTS idx_bul_player ON ml.bayesian_update_log(player_id) WHERE player_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_bul_stat_type ON ml.bayesian_update_log(stat_type);
CREATE INDEX IF NOT EXISTS idx_bul_season_week ON ml.bayesian_update_log(season, week);
CREATE INDEX IF NOT EXISTS idx_bul_created ON ml.bayesian_update_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bul_update_type ON ml.bayesian_update_log(update_type);

-- Partial index for large errors (for debugging)
CREATE INDEX IF NOT EXISTS idx_bul_large_errors ON ml.bayesian_update_log(abs_error DESC)
    WHERE abs_error > 50;  -- Large errors worth investigating

-- Comments
COMMENT ON TABLE ml.bayesian_update_log IS 'Audit trail for all Bayesian updates (player priors, ensemble weights, Dirichlet alphas)';
COMMENT ON COLUMN ml.bayesian_update_log.prior_params IS 'Prior distribution parameters as JSON (e.g., {"alpha": 10, "beta": 5} for Beta, {"mu": 250, "sigma": 30} for Normal)';
COMMENT ON COLUMN ml.bayesian_update_log.likelihoods IS 'Likelihood contributions by component (e.g., {"hierarchical": 0.85, "xgboost": 0.72})';
COMMENT ON COLUMN ml.bayesian_update_log.posterior_params IS 'Posterior distribution parameters after Bayesian update';
COMMENT ON COLUMN ml.bayesian_update_log.update_magnitude IS 'Measure of how much the posterior shifted (KL divergence or similar)';

-- Function to log a Bayesian update
CREATE OR REPLACE FUNCTION ml.log_bayesian_update(
    p_update_type VARCHAR(30),
    p_stat_type VARCHAR(50),
    p_season INTEGER,
    p_week INTEGER,
    p_prior_params JSONB,
    p_likelihoods JSONB,
    p_posterior_params JSONB,
    p_player_id VARCHAR(50) DEFAULT NULL,
    p_game_id VARCHAR(20) DEFAULT NULL,
    p_actual_value NUMERIC DEFAULT NULL,
    p_ensemble_prediction NUMERIC DEFAULT NULL,
    p_component_predictions JSONB DEFAULT NULL,
    p_learning_rate NUMERIC DEFAULT 1.0
)
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    new_log_id BIGINT;
    pred_error NUMERIC;
BEGIN
    -- Compute prediction error if both values provided
    IF p_actual_value IS NOT NULL AND p_ensemble_prediction IS NOT NULL THEN
        pred_error := p_actual_value - p_ensemble_prediction;
    END IF;

    INSERT INTO ml.bayesian_update_log (
        update_type, player_id, game_id, stat_type,
        season, week,
        prior_params, likelihoods, posterior_params,
        actual_value, ensemble_prediction, prediction_error, abs_error,
        component_predictions, learning_rate_used
    ) VALUES (
        p_update_type, p_player_id, p_game_id, p_stat_type,
        p_season, p_week,
        p_prior_params, p_likelihoods, p_posterior_params,
        p_actual_value, p_ensemble_prediction, pred_error, ABS(pred_error),
        p_component_predictions, p_learning_rate
    )
    RETURNING log_id INTO new_log_id;

    RETURN new_log_id;
END;
$$;

-- View for recent updates (debugging)
CREATE OR REPLACE VIEW ml.recent_bayesian_updates AS
SELECT
    log_id,
    update_type,
    stat_type,
    player_id,
    game_id,
    season,
    week,
    actual_value,
    ensemble_prediction,
    abs_error,
    created_at
FROM ml.bayesian_update_log
ORDER BY created_at DESC
LIMIT 100;

-- View for update statistics by stat type
CREATE OR REPLACE VIEW ml.bayesian_update_stats AS
SELECT
    stat_type,
    update_type,
    COUNT(*) as n_updates,
    AVG(abs_error) as avg_abs_error,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_error) as median_abs_error,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_error) as p90_abs_error,
    MIN(created_at) as first_update,
    MAX(created_at) as last_update
FROM ml.bayesian_update_log
WHERE abs_error IS NOT NULL
GROUP BY stat_type, update_type
ORDER BY stat_type, update_type;

-- View for player update history (for player-level analysis)
CREATE OR REPLACE VIEW ml.player_bayesian_history AS
SELECT
    player_id,
    stat_type,
    COUNT(*) as n_updates,
    AVG(abs_error) as avg_abs_error,
    STDDEV(prediction_error) as prediction_volatility,
    AVG(prediction_error) as avg_bias,  -- Positive = under-predicting
    MAX(created_at) as last_update
FROM ml.bayesian_update_log
WHERE player_id IS NOT NULL AND abs_error IS NOT NULL
GROUP BY player_id, stat_type
HAVING COUNT(*) >= 3;  -- Minimum sample size

-- Retention policy: keep 1 year of detailed logs, summarize older
-- (Note: Actual archival would require a cron job or pg_cron)
COMMENT ON TABLE ml.bayesian_update_log IS
    'Audit trail for Bayesian updates. Recommended: Archive logs older than 1 year to cold storage.';

-- Record migration
INSERT INTO schema_migrations (version, description)
VALUES ('103_bayesian_update_log', 'Bayesian update audit log for transparency and debugging')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- Verification
DO $$
BEGIN
    RAISE NOTICE 'Migration 103 complete: ml.bayesian_update_log table created';
    RAISE NOTICE 'Function created: ml.log_bayesian_update()';
    RAISE NOTICE 'Views created: ml.recent_bayesian_updates, ml.bayesian_update_stats, ml.player_bayesian_history';
END $$;
