-- Migration 024: Bayesian Hierarchical Player Props Predictions
-- Purpose: Store game-specific Bayesian prop predictions with full uncertainty quantification
-- Author: Claude Code
-- Date: 2025-10-12
--
-- This table differs from predictions.prop_predictions by:
-- 1. Storing full posterior distributions (not just point estimates)
-- 2. Including game-specific context adjustments
-- 3. Tracking hierarchical model components (league, position, team, player effects)
-- 4. Supporting proper Bayesian credible intervals (not confidence intervals)

-- Create predictions schema if not exists
CREATE SCHEMA IF NOT EXISTS predictions;

-- ==============================================================================
-- TABLE: bayesian_prop_predictions
-- Stores game-specific Bayesian hierarchical predictions with full uncertainty
-- ==============================================================================
CREATE TABLE IF NOT EXISTS predictions.bayesian_prop_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,

    -- Identifiers
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    stat_type TEXT NOT NULL CHECK (stat_type IN (
        'passing_yards', 'passing_tds', 'interceptions',
        'rushing_yards', 'rushing_tds', 'rushing_attempts',
        'receiving_yards', 'receiving_tds', 'receptions',
        'completions', 'pass_attempts', 'completion_pct'
    )),
    model_version TEXT NOT NULL,  -- e.g., 'hierarchical_v1.1'

    -- Core Bayesian predictions (in log-space for yards/attempts, natural scale for TDs/INTs)
    -- These are the raw posterior summaries from brms
    rating_mean DOUBLE PRECISION NOT NULL,           -- E[log(yards)] from fitted()
    rating_sd DOUBLE PRECISION NOT NULL,             -- SD[log(yards)] from predict()
    rating_q05 DOUBLE PRECISION NOT NULL,            -- 5th percentile from predict()
    rating_q50 DOUBLE PRECISION NOT NULL,            -- Median from fitted()
    rating_q95 DOUBLE PRECISION NOT NULL,            -- 95th percentile from predict()

    -- Transformed predictions (natural scale for user consumption)
    predicted_value DOUBLE PRECISION NOT NULL,       -- exp(rating_mean) = E[yards]
    predicted_q05 DOUBLE PRECISION NOT NULL,         -- exp(rating_q05)
    predicted_q95 DOUBLE PRECISION NOT NULL,         -- exp(rating_q95)

    -- Hierarchical model components (decomposition of prediction)
    league_intercept DOUBLE PRECISION,               -- Population-level mean
    position_group_effect DOUBLE PRECISION,          -- Position random effect (QB vs RB vs WR)
    team_effect DOUBLE PRECISION,                    -- Team random effect (offensive quality)
    vs_opponent_effect DOUBLE PRECISION,             -- Opponent random effect (defensive quality)
    player_effect DOUBLE PRECISION,                  -- Player-specific random effect (talent)

    -- Game context adjustments (covariates from model)
    log_attempts_adjustment DOUBLE PRECISION,        -- log(pass_attempts) or log(rush_attempts) effect
    home_field_adjustment DOUBLE PRECISION,          -- is_home effect
    favorite_adjustment DOUBLE PRECISION,            -- is_favored effect
    weather_adjustment DOUBLE PRECISION,             -- is_bad_weather + is_dome effects
    total_line_adjustment DOUBLE PRECISION,          -- total_line effect (game pace)
    spread_adjustment DOUBLE PRECISION,              -- spread_abs effect (game script)

    -- Prediction metadata
    n_posterior_draws INTEGER DEFAULT 4000,          -- Number of MCMC samples
    effective_sample_size DOUBLE PRECISION,          -- ESS for convergence diagnostics
    rhat DOUBLE PRECISION CHECK (rhat IS NULL OR rhat <= 1.1),  -- Convergence diagnostic

    -- Context snapshot (for debugging and feature engineering)
    game_context JSONB,  -- {
                         --   "is_home": true,
                         --   "is_favored": false,
                         --   "spread_abs": 3.5,
                         --   "total_line": 47.5,
                         --   "is_bad_weather": false,
                         --   "is_dome": true,
                         --   "log_attempts": 3.58,
                         --   "opponent": "BUF",
                         --   "position": "QB",
                         --   "experience_years": 5
                         -- }

    -- Observables (for model validation)
    actual_value DOUBLE PRECISION,                   -- Actual outcome (NULL until game completes)
    actual_in_ci BOOLEAN,                            -- Did actual fall in [q05, q95]?
    absolute_error DOUBLE PRECISION,                 -- |actual - predicted| (NULL until complete)
    log_likelihood DOUBLE PRECISION,                 -- Log p(actual | prediction) (NULL until complete)

    -- Betting integration
    book_line DOUBLE PRECISION,                      -- Sportsbook line at prediction time
    implied_edge DOUBLE PRECISION,                   -- predicted_value - book_line
    recommended_side TEXT CHECK (recommended_side IN ('over', 'under', 'pass', NULL)),
    kelly_fraction DOUBLE PRECISION CHECK (kelly_fraction IS NULL OR kelly_fraction >= 0),
    expected_value DOUBLE PRECISION,                 -- EV in dollars for $1 bet

    -- Timestamps
    predicted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    game_kickoff TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Foreign keys
    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE,

    -- Uniqueness: one prediction per (game, player, stat, model_version)
    UNIQUE(game_id, player_id, stat_type, model_version)
);

-- ==============================================================================
-- INDEXES: Fast lookups for common queries
-- ==============================================================================
CREATE INDEX idx_bayesian_props_game_id ON predictions.bayesian_prop_predictions(game_id);
CREATE INDEX idx_bayesian_props_player_id ON predictions.bayesian_prop_predictions(player_id);
CREATE INDEX idx_bayesian_props_stat_type ON predictions.bayesian_prop_predictions(stat_type);
CREATE INDEX idx_bayesian_props_model_version ON predictions.bayesian_prop_predictions(model_version);
CREATE INDEX idx_bayesian_props_predicted_at ON predictions.bayesian_prop_predictions(predicted_at DESC);
CREATE INDEX idx_bayesian_props_game_kickoff ON predictions.bayesian_prop_predictions(game_kickoff);

-- Index for finding predictions with available book lines
CREATE INDEX idx_bayesian_props_has_edge ON predictions.bayesian_prop_predictions(implied_edge DESC NULLS LAST)
WHERE book_line IS NOT NULL AND implied_edge IS NOT NULL;

-- Index for model validation queries
CREATE INDEX idx_bayesian_props_validation ON predictions.bayesian_prop_predictions(stat_type, model_version, actual_value)
WHERE actual_value IS NOT NULL;

-- Index for CI coverage analysis
CREATE INDEX idx_bayesian_props_ci_coverage ON predictions.bayesian_prop_predictions(stat_type, model_version, actual_in_ci)
WHERE actual_in_ci IS NOT NULL;

-- ==============================================================================
-- VIEWS: Convenience views for dashboard and analysis
-- ==============================================================================

-- View: Current week Bayesian prop predictions (natural scale, user-friendly)
CREATE OR REPLACE VIEW predictions.bayesian_props_current_week AS
SELECT
    bp.prediction_id,
    bp.game_id,
    g.kickoff as game_kickoff,
    g.home_team,
    g.away_team,
    g.week,
    g.season,
    g.game_type,

    -- Player info (get current team from rosters_weekly)
    bp.player_id,
    p.player_name,
    p.position as player_position,
    r.team as player_team,
    CASE
        WHEN r.team = g.home_team THEN 'home'
        WHEN r.team = g.away_team THEN 'away'
        ELSE 'unknown'
    END as home_away,
    CASE
        WHEN r.team = g.home_team THEN g.away_team
        WHEN r.team = g.away_team THEN g.home_team
        ELSE NULL
    END as opponent,

    -- Predictions (natural scale)
    bp.stat_type,
    bp.predicted_value,
    bp.predicted_q05,
    bp.predicted_q95,
    bp.predicted_q95 - bp.predicted_q05 as credible_interval_width,

    -- Uncertainty metrics
    bp.rating_sd as log_space_uncertainty,
    (bp.predicted_q95 - bp.predicted_q05) / (2 * 1.645) as implied_std,  -- Approximate SD in natural scale

    -- Model diagnostics
    bp.model_version,
    bp.rhat,
    bp.effective_sample_size,

    -- Betting info
    bp.book_line,
    bp.implied_edge,
    bp.recommended_side,
    bp.kelly_fraction,
    bp.expected_value,

    -- Metadata
    bp.predicted_at,
    bp.game_context
FROM predictions.bayesian_prop_predictions bp
JOIN games g ON bp.game_id = g.game_id
LEFT JOIN players p ON bp.player_id = p.player_id
LEFT JOIN LATERAL (
    SELECT DISTINCT ON (gsis_id) team
    FROM rosters_weekly
    WHERE gsis_id = bp.player_id
      AND season = g.season
      AND week <= g.week
    ORDER BY gsis_id, week DESC
) r ON TRUE
WHERE g.season = EXTRACT(YEAR FROM CURRENT_DATE)
  AND g.kickoff > NOW()
  AND g.kickoff < NOW() + INTERVAL '7 days'
ORDER BY g.kickoff, bp.predicted_value DESC;

-- View: Bayesian predictions with book lines (for finding edges)
CREATE OR REPLACE VIEW predictions.bayesian_edges AS
SELECT
    bp.prediction_id,
    bp.game_id,
    g.kickoff,
    g.home_team,
    g.away_team,
    g.week,

    -- Player (get team from rosters_weekly)
    bp.player_id,
    p.player_name,
    p.position,
    r.team as player_team,

    -- Prediction vs Line
    bp.stat_type,
    bp.predicted_value,
    bp.predicted_q05,
    bp.predicted_q95,
    bp.book_line,
    bp.implied_edge,

    -- Probability of going over line (using normal approximation)
    -- P(Y > line) = P(Z > (line - mean) / sd) where Z ~ N(0,1)
    CASE
        WHEN bp.book_line IS NOT NULL AND bp.rating_sd IS NOT NULL THEN
            1 - (0.5 * (1 + erf((bp.book_line - bp.predicted_value) / (bp.rating_sd * sqrt(2)))))
        ELSE NULL
    END as prob_over,

    -- Kelly criterion bet sizing
    bp.kelly_fraction,
    bp.expected_value,
    bp.recommended_side,

    -- Model quality
    bp.model_version,
    bp.predicted_q95 - bp.predicted_q05 as credible_interval_width,
    bp.rhat,

    -- Timestamps
    bp.predicted_at,
    g.kickoff as game_kickoff
FROM predictions.bayesian_prop_predictions bp
JOIN games g ON bp.game_id = g.game_id
LEFT JOIN players p ON bp.player_id = p.player_id
LEFT JOIN LATERAL (
    SELECT DISTINCT ON (gsis_id) team
    FROM rosters_weekly
    WHERE gsis_id = bp.player_id
      AND season = g.season
      AND week <= g.week
    ORDER BY gsis_id, week DESC
) r ON TRUE
WHERE bp.book_line IS NOT NULL
  AND bp.implied_edge IS NOT NULL
  AND ABS(bp.implied_edge) > 5  -- Only show edges > 5 yards
  AND g.kickoff > NOW()
ORDER BY ABS(bp.implied_edge) DESC;

-- View: Model validation metrics (post-game analysis)
CREATE OR REPLACE VIEW predictions.bayesian_validation_metrics AS
SELECT
    stat_type,
    model_version,
    COUNT(*) as n_predictions,
    COUNT(*) FILTER (WHERE actual_value IS NOT NULL) as n_completed,

    -- Accuracy metrics
    AVG(absolute_error) as mae,
    SQRT(AVG(absolute_error * absolute_error)) as rmse,
    STDDEV(actual_value - predicted_value) as prediction_sd,
    CORR(predicted_value, actual_value) as correlation,

    -- Calibration metrics
    AVG(actual_in_ci::INTEGER) as ci_coverage,  -- Should be ~0.90 for 90% CI
    AVG(log_likelihood) as avg_log_likelihood,

    -- Bias analysis
    AVG(actual_value - predicted_value) as mean_bias,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY actual_value - predicted_value) as median_bias,

    -- Convergence diagnostics
    AVG(rhat) as avg_rhat,
    MAX(rhat) as max_rhat,
    AVG(effective_sample_size) as avg_ess,
    MIN(effective_sample_size) as min_ess,

    -- Timestamp
    MAX(updated_at) as last_updated
FROM predictions.bayesian_prop_predictions
WHERE actual_value IS NOT NULL
GROUP BY stat_type, model_version
ORDER BY stat_type, model_version;

-- ==============================================================================
-- FUNCTIONS: Helper functions
-- ==============================================================================

-- Function: Update actual values and compute validation metrics
CREATE OR REPLACE FUNCTION predictions.update_bayesian_prop_actuals()
RETURNS TABLE (
    predictions_updated INTEGER,
    avg_mae DOUBLE PRECISION,
    ci_coverage DOUBLE PRECISION
) AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    -- Update passing yards predictions
    WITH actual_stats AS (
        SELECT
            g.game_id,
            p.player_id,
            p.passing_yards as actual_yards
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.passer_player_id IS NOT NULL
          AND g.game_type = 'REG'
          AND g.season >= 2024
        GROUP BY g.game_id, p.player_id, p.passing_yards
    ),
    updates AS (
        UPDATE predictions.bayesian_prop_predictions bp
        SET
            actual_value = a.actual_yards,
            actual_in_ci = (a.actual_yards BETWEEN bp.predicted_q05 AND bp.predicted_q95),
            absolute_error = ABS(a.actual_yards - bp.predicted_value),
            updated_at = NOW()
        FROM actual_stats a
        WHERE bp.game_id = a.game_id
          AND bp.player_id = a.player_id
          AND bp.stat_type = 'passing_yards'
          AND bp.actual_value IS NULL
        RETURNING 1
    )
    SELECT COUNT(*) INTO updated_count FROM updates;

    -- Return summary metrics
    RETURN QUERY
    SELECT
        updated_count,
        AVG(bp.absolute_error),
        AVG(bp.actual_in_ci::INTEGER)::DOUBLE PRECISION
    FROM predictions.bayesian_prop_predictions bp
    WHERE bp.actual_value IS NOT NULL
      AND bp.stat_type = 'passing_yards';
END;
$$ LANGUAGE plpgsql;

-- Function: Get prediction for specific player in upcoming game
CREATE OR REPLACE FUNCTION predictions.get_bayesian_prop_prediction(
    p_player_id TEXT,
    p_game_id TEXT,
    p_stat_type TEXT DEFAULT 'passing_yards'
)
RETURNS TABLE (
    player_name TEXT,
    stat_type TEXT,
    predicted_value DOUBLE PRECISION,
    credible_interval TEXT,  -- e.g., "[195, 312]"
    uncertainty_pct DOUBLE PRECISION,  -- Width / Mean
    opponent TEXT,
    game_kickoff TIMESTAMP WITH TIME ZONE,
    model_version TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pl.player_name,
        bp.stat_type,
        bp.predicted_value,
        FORMAT('[%.0f, %.0f]', bp.predicted_q05, bp.predicted_q95),
        (bp.predicted_q95 - bp.predicted_q05) / NULLIF(bp.predicted_value, 0) * 100,
        CASE
            WHEN r.team = g.home_team THEN g.away_team
            WHEN r.team = g.away_team THEN g.home_team
        END,
        g.kickoff,
        bp.model_version
    FROM predictions.bayesian_prop_predictions bp
    JOIN games g ON bp.game_id = g.game_id
    JOIN players pl ON bp.player_id = pl.player_id
    LEFT JOIN LATERAL (
        SELECT DISTINCT ON (gsis_id) team
        FROM rosters_weekly
        WHERE gsis_id = bp.player_id
          AND season = g.season
          AND week <= g.week
        ORDER BY gsis_id, week DESC
    ) r ON TRUE
    WHERE bp.player_id = p_player_id
      AND bp.game_id = p_game_id
      AND bp.stat_type = p_stat_type
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- TRIGGERS: Auto-update timestamps
-- ==============================================================================
CREATE OR REPLACE FUNCTION predictions.update_bayesian_props_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_bayesian_props_updated_at
    BEFORE UPDATE ON predictions.bayesian_prop_predictions
    FOR EACH ROW
    EXECUTE FUNCTION predictions.update_bayesian_props_timestamp();

-- ==============================================================================
-- GRANTS: Permissions
-- ==============================================================================
GRANT USAGE ON SCHEMA predictions TO dro;
GRANT ALL PRIVILEGES ON predictions.bayesian_prop_predictions TO dro;
GRANT USAGE, SELECT ON SEQUENCE predictions.bayesian_prop_predictions_prediction_id_seq TO dro;
GRANT SELECT ON predictions.bayesian_props_current_week TO dro;
GRANT SELECT ON predictions.bayesian_edges TO dro;
GRANT SELECT ON predictions.bayesian_validation_metrics TO dro;
GRANT EXECUTE ON FUNCTION predictions.update_bayesian_prop_actuals() TO dro;
GRANT EXECUTE ON FUNCTION predictions.get_bayesian_prop_prediction(TEXT, TEXT, TEXT) TO dro;

-- ==============================================================================
-- COMMENTS: Documentation
-- ==============================================================================
COMMENT ON TABLE predictions.bayesian_prop_predictions IS
'Game-specific Bayesian hierarchical prop predictions with full posterior uncertainty.
Stores both log-space (model native) and natural-scale (user-friendly) predictions.
Includes hierarchical decomposition and game context for interpretability.';

COMMENT ON COLUMN predictions.bayesian_prop_predictions.rating_mean IS
'Mean of posterior predictive distribution in log-space (from brms fitted()).
For yards: E[log(yards)]. Transform to natural scale with exp(rating_mean).';

COMMENT ON COLUMN predictions.bayesian_prop_predictions.rating_sd IS
'Standard deviation of posterior predictive distribution in log-space (from brms predict()).
Includes both parameter uncertainty AND observation noise.';

COMMENT ON COLUMN predictions.bayesian_prop_predictions.predicted_value IS
'Point prediction in natural scale: exp(rating_mean) for yards.
This is the expected value E[yards] accounting for log-normal distribution.';

COMMENT ON COLUMN predictions.bayesian_prop_predictions.rhat IS
'Gelman-Rubin convergence diagnostic. Should be <= 1.01 for reliable inference.
Values > 1.1 indicate poor MCMC convergence.';

COMMENT ON COLUMN predictions.bayesian_prop_predictions.actual_in_ci IS
'Boolean: did actual outcome fall within 90% credible interval [q05, q95]?
Used to validate model calibration. Should be TRUE ~90% of the time.';

COMMENT ON VIEW predictions.bayesian_edges IS
'Identifies betting opportunities where Bayesian prediction differs significantly
from sportsbook line (|edge| > 5 yards). Includes probability calculations.';

COMMENT ON FUNCTION predictions.update_bayesian_prop_actuals() IS
'Backfills actual outcomes for completed games and computes validation metrics.
Run this after games complete to track model performance.';

-- ==============================================================================
-- SAMPLE QUERIES
-- ==============================================================================
/*
-- 1. Get current week's predictions
SELECT * FROM predictions.bayesian_props_current_week
WHERE stat_type = 'passing_yards'
ORDER BY predicted_value DESC
LIMIT 10;

-- 2. Find the best betting edges
SELECT * FROM predictions.bayesian_edges
WHERE stat_type = 'passing_yards'
  AND ABS(implied_edge) > 10
ORDER BY ABS(implied_edge) DESC;

-- 3. Get prediction for specific player
SELECT * FROM predictions.get_bayesian_prop_prediction(
    '00-0019596',  -- Patrick Mahomes
    '2024_01_KC_BAL',
    'passing_yards'
);

-- 4. Model validation summary
SELECT * FROM predictions.bayesian_validation_metrics
ORDER BY stat_type, model_version;

-- 5. Check CI coverage by week (should be ~90%)
SELECT
    g.week,
    COUNT(*) as n_predictions,
    AVG(bp.actual_in_ci::INTEGER) as ci_coverage,
    AVG(bp.absolute_error) as mae
FROM predictions.bayesian_prop_predictions bp
JOIN games g ON bp.game_id = g.game_id
WHERE bp.actual_value IS NOT NULL
  AND bp.stat_type = 'passing_yards'
  AND g.season = 2024
GROUP BY g.week
ORDER BY g.week;

-- 6. Hierarchical effects analysis
SELECT
    player_id,
    predicted_value,
    league_intercept,
    position_group_effect,
    team_effect,
    player_effect,
    home_field_adjustment
FROM predictions.bayesian_prop_predictions
WHERE stat_type = 'passing_yards'
  AND game_id = '2024_01_KC_BAL'
ORDER BY predicted_value DESC;
*/

-- ==============================================================================
-- COMPLETE
-- ==============================================================================
-- Migration 024 complete! Run with:
-- PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -f db/migrations/024_bayesian_prop_predictions.sql
