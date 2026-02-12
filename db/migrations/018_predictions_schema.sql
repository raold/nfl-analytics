-- Migration 018: Predictions Infrastructure
-- Creates schema for storing predictions, tracking versioning, and learning from failures
--
-- Purpose: Enable systematic learning from prediction errors to improve models over time
-- This is the core of our RL feedback loop
--
-- Tables:
--   1. game_predictions: All model predictions with ensemble components
--   2. prediction_versions: Track how predictions changed over time
--   3. retrospectives: Post-game analysis of what went wrong/right
--   4. learning_loop: Aggregate patterns for model improvement

-- Create predictions schema
CREATE SCHEMA IF NOT EXISTS predictions;

-- ==============================================================================
-- TABLE 1: game_predictions
-- Stores all predictions from all models with full context
-- ==============================================================================
CREATE TABLE predictions.game_predictions (
    prediction_id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    model_version TEXT NOT NULL,  -- 'xgb_v2', 'bayesian_v1', 'ensemble_bayesian_xgb_v1'
    predicted_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Core predictions
    home_win_prob DECIMAL(5,4) CHECK (home_win_prob BETWEEN 0 AND 1),
    predicted_spread DECIMAL(5,2),
    predicted_total DECIMAL(5,2),

    -- Model component probabilities (for ensemble decomposition)
    xgb_prob DECIMAL(5,4) CHECK (xgb_prob IS NULL OR xgb_prob BETWEEN 0 AND 1),
    bayesian_prob DECIMAL(5,4) CHECK (bayesian_prob IS NULL OR bayesian_prob BETWEEN 0 AND 1),
    bayesian_sd DECIMAL(5,4) CHECK (bayesian_sd IS NULL OR bayesian_sd >= 0),

    -- Betting recommendation
    recommended_bet TEXT CHECK (recommended_bet IN ('home', 'away', 'over', 'under', 'none')),
    bet_confidence DECIMAL(5,4) CHECK (bet_confidence BETWEEN 0 AND 1),
    kelly_fraction DECIMAL(5,4) CHECK (kelly_fraction >= 0),
    edge_estimate DECIMAL(5,4),  -- Can be negative

    -- Market context at prediction time
    spread_close DECIMAL(5,2),
    total_close DECIMAL(5,2),
    days_before_game INTEGER CHECK (days_before_game >= 0),

    -- Feature snapshot (JSONB for flexibility as features evolve)
    feature_snapshot JSONB,

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

-- Indexes for fast lookups
CREATE INDEX idx_pred_game_id ON predictions.game_predictions(game_id);
CREATE INDEX idx_pred_model_version ON predictions.game_predictions(model_version);
CREATE INDEX idx_pred_predicted_at ON predictions.game_predictions(predicted_at DESC);
CREATE INDEX idx_pred_recommended_bet ON predictions.game_predictions(recommended_bet) WHERE recommended_bet != 'none';

-- ==============================================================================
-- TABLE 2: prediction_versions
-- Tracks how predictions evolved as new information became available
-- ==============================================================================
CREATE TABLE predictions.prediction_versions (
    version_id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    version_timestamp TIMESTAMP NOT NULL,
    version_label TEXT NOT NULL CHECK (version_label IN (
        '7_days_out', '5_days_out', '3_days_out', 'day_before', 'day_of', 'final'
    )),

    -- Prediction at this version
    home_win_prob DECIMAL(5,4) CHECK (home_win_prob BETWEEN 0 AND 1),
    predicted_spread DECIMAL(5,2),
    predicted_total DECIMAL(5,2),
    bet_confidence DECIMAL(5,4) CHECK (bet_confidence BETWEEN 0 AND 1),

    -- Changes since previous version
    prob_change DECIMAL(5,4),
    spread_change DECIMAL(5,2),
    confidence_change DECIMAL(5,4),

    -- What drove the changes? (narrative explanation)
    change_drivers JSONB,  -- {'injury': 'QB out', 'line_movement': 2.5, 'weather': 'snow forecast'}

    -- Link to main prediction
    prediction_id INTEGER,

    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions.game_predictions(prediction_id) ON DELETE CASCADE,

    UNIQUE(game_id, version_label)  -- One prediction per version label per game
);

CREATE INDEX idx_version_game ON predictions.prediction_versions(game_id);
CREATE INDEX idx_version_timestamp ON predictions.prediction_versions(version_timestamp DESC);
CREATE INDEX idx_version_label ON predictions.prediction_versions(version_label);

-- ==============================================================================
-- TABLE 3: retrospectives
-- Post-game analysis: Why did we get it right/wrong?
-- This is where we learn!
-- ==============================================================================
CREATE TABLE predictions.retrospectives (
    retro_id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL UNIQUE,  -- One retrospective per game
    analyzed_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Actual outcome
    actual_winner TEXT CHECK (actual_winner IN ('home', 'away', 'push')),
    actual_margin DECIMAL(5,2),
    actual_total DECIMAL(5,2),
    home_score INTEGER CHECK (home_score >= 0),
    away_score INTEGER CHECK (away_score >= 0),

    -- Our final prediction (from game_predictions)
    predicted_winner TEXT CHECK (predicted_winner IN ('home', 'away')),
    predicted_margin DECIMAL(5,2),
    predicted_total DECIMAL(5,2),
    prediction_confidence DECIMAL(5,4) CHECK (prediction_confidence BETWEEN 0 AND 1),

    -- Error metrics
    margin_error DECIMAL(5,2),  -- actual - predicted
    total_error DECIMAL(5,2),
    abs_margin_error DECIMAL(5,2) CHECK (abs_margin_error >= 0),

    -- Outcome classification
    outcome_type TEXT NOT NULL CHECK (outcome_type IN (
        'correct_high_conf',    -- Predicted correctly with >70% confidence
        'correct_low_conf',     -- Got it right but weren't sure
        'wrong_close',          -- Wrong but close (within 3 points)
        'wrong_upset',          -- Wrong direction, >7 point favorite lost
        'wrong_blowout',        -- Massive error (>14 points)
        'push'                  -- Landed exactly on spread
    )),

    surprise_factor DECIMAL(5,4) CHECK (surprise_factor BETWEEN 0 AND 1),  -- 0 = expected, 1 = shocking

    -- Root cause analysis
    primary_failure_mode TEXT CHECK (primary_failure_mode IN (
        'correct_prediction',
        'missed_injury_impact',
        'underestimated_home_field',
        'weather_not_captured',
        'backup_qb_uncertainty',
        'thursday_night_effect',
        'divisional_game_variance',
        'garbage_time_skew',
        'referee_impact',
        'narrative_overreaction',
        'model_uncertainty_high',
        'unknown'
    )),

    -- Narrative factors present (from game context)
    narrative_factors JSONB,  -- {'thursday_night': true, 'backup_qb': true, 'snow_game': true}

    -- Feature importance analysis (which features were most wrong?)
    feature_importance_diff JSONB,  -- {'home_epa_offense': -0.15, 'injury_load': 0.22}

    -- Lessons learned (human-written or LLM-generated)
    learning_notes TEXT,
    model_update_recommended BOOLEAN DEFAULT FALSE,
    update_priority TEXT CHECK (update_priority IN ('low', 'medium', 'high', 'critical') OR update_priority IS NULL),

    -- Status
    implemented BOOLEAN DEFAULT FALSE,
    implemented_at TIMESTAMP,

    FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
);

CREATE INDEX idx_retro_outcome_type ON predictions.retrospectives(outcome_type);
CREATE INDEX idx_retro_failure_mode ON predictions.retrospectives(primary_failure_mode);
CREATE INDEX idx_retro_surprise ON predictions.retrospectives(surprise_factor DESC);
CREATE INDEX idx_retro_not_implemented ON predictions.retrospectives(implemented) WHERE NOT implemented;

-- ==============================================================================
-- TABLE 4: learning_loop
-- Aggregate patterns discovered from multiple retrospectives
-- This drives model improvement!
-- ==============================================================================
CREATE TABLE predictions.learning_loop (
    learning_id SERIAL PRIMARY KEY,
    identified_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Pattern discovered
    pattern_name TEXT NOT NULL UNIQUE,  -- 'thursday_night_home_advantage', 'backup_qb_underdog_boost'
    pattern_description TEXT NOT NULL,
    category TEXT CHECK (category IN (
        'situational',      -- Game context (TNF, divisional, weather)
        'personnel',        -- Player-specific (backup QB, injured star)
        'temporal',         -- Time-based (early season, playoff)
        'market',           -- Betting market dynamics
        'model_bias'        -- Systematic model error
    )),

    -- Evidence (links to retrospectives)
    supporting_game_ids TEXT[],  -- Array of game_ids that show this pattern
    sample_size INTEGER CHECK (sample_size > 0),

    -- Statistical strength
    avg_prediction_error DECIMAL(5,2),  -- How much we're off by
    error_std_dev DECIMAL(5,2),
    statistical_significance DECIMAL(5,4) CHECK (statistical_significance BETWEEN 0 AND 1),  -- p-value
    effect_size DECIMAL(5,4),  -- Cohen's d or similar

    -- Recommendation
    feature_to_add TEXT,  -- 'thursday_night_home' or 'backup_qb_uncertainty'
    model_adjustment TEXT,  -- Human-readable description of what to change
    expected_improvement_pct DECIMAL(5,4),  -- Expected ROI improvement (e.g., 0.005 = 0.5%)

    -- Priority
    priority TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'critical')),

    -- Implementation tracking
    implemented BOOLEAN DEFAULT FALSE,
    implemented_at TIMESTAMP,
    implementation_notes TEXT,

    -- Validation results after implementation
    validation_result JSONB,  -- {'actual_improvement': 0.007, 'test_games': 45}
    validated BOOLEAN DEFAULT FALSE,
    validated_at TIMESTAMP
);

CREATE INDEX idx_learning_pattern ON predictions.learning_loop(pattern_name);
CREATE INDEX idx_learning_category ON predictions.learning_loop(category);
CREATE INDEX idx_learning_priority ON predictions.learning_loop(priority) WHERE NOT implemented;
CREATE INDEX idx_learning_pending ON predictions.learning_loop(implemented) WHERE NOT implemented;

-- ==============================================================================
-- VIEWS: Convenience views for common queries
-- ==============================================================================

-- View: Latest prediction for each game
CREATE VIEW predictions.latest_predictions AS
SELECT DISTINCT ON (game_id)
    prediction_id,
    game_id,
    model_version,
    predicted_at,
    home_win_prob,
    predicted_spread,
    recommended_bet,
    bet_confidence,
    edge_estimate,
    days_before_game
FROM predictions.game_predictions
ORDER BY game_id, predicted_at DESC;

-- View: Prediction accuracy summary
CREATE VIEW predictions.accuracy_summary AS
SELECT
    COUNT(*) as total_predictions,
    AVG(ABS(r.margin_error)) as avg_abs_error,
    STDDEV(r.margin_error) as error_std_dev,
    SUM(CASE WHEN r.outcome_type LIKE 'correct%' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as accuracy_rate,
    SUM(CASE WHEN r.surprise_factor > 0.7 THEN 1 ELSE 0 END) as high_surprise_count,
    AVG(r.surprise_factor) as avg_surprise_factor
FROM predictions.retrospectives r;

-- View: Learning opportunities (high-priority patterns not yet implemented)
CREATE VIEW predictions.learning_opportunities AS
SELECT
    learning_id,
    pattern_name,
    pattern_description,
    category,
    sample_size,
    avg_prediction_error,
    statistical_significance,
    expected_improvement_pct,
    priority
FROM predictions.learning_loop
WHERE NOT implemented
  AND priority IN ('high', 'critical')
ORDER BY priority DESC, expected_improvement_pct DESC;

-- ==============================================================================
-- FUNCTIONS: Helper functions
-- ==============================================================================

-- Function: Get prediction change summary for a game
CREATE OR REPLACE FUNCTION predictions.get_prediction_evolution(p_game_id TEXT)
RETURNS TABLE (
    version_label TEXT,
    version_timestamp TIMESTAMP,
    home_win_prob DECIMAL,
    predicted_spread DECIMAL,
    change_from_previous TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pv.version_label,
        pv.version_timestamp,
        pv.home_win_prob,
        pv.predicted_spread,
        CASE
            WHEN pv.prob_change IS NOT NULL
            THEN FORMAT('Prob: %s%s, Spread: %s%s',
                CASE WHEN pv.prob_change > 0 THEN '+' ELSE '' END,
                ROUND(pv.prob_change::NUMERIC, 3),
                CASE WHEN pv.spread_change > 0 THEN '+' ELSE '' END,
                ROUND(pv.spread_change::NUMERIC, 1))
            ELSE 'Initial prediction'
        END as change_description
    FROM predictions.prediction_versions pv
    WHERE pv.game_id = p_game_id
    ORDER BY pv.version_timestamp;
END;
$$ LANGUAGE plpgsql;

-- Function: Identify games needing retrospective analysis
CREATE OR REPLACE FUNCTION predictions.games_needing_retrospective()
RETURNS TABLE (
    game_id TEXT,
    kickoff TIMESTAMP,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        g.game_id,
        g.kickoff,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score
    FROM games g
    LEFT JOIN predictions.retrospectives r ON g.game_id = r.game_id
    WHERE g.home_score IS NOT NULL  -- Game completed
      AND r.retro_id IS NULL         -- No retrospective yet
      AND EXISTS (                    -- We made a prediction
          SELECT 1 FROM predictions.game_predictions p
          WHERE p.game_id = g.game_id
      )
    ORDER BY g.kickoff DESC;
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- GRANTS: Ensure application can access these tables
-- ==============================================================================
GRANT USAGE ON SCHEMA predictions TO dro;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA predictions TO dro;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA predictions TO dro;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA predictions TO dro;

-- ==============================================================================
-- COMMENTS: Documentation
-- ==============================================================================
COMMENT ON SCHEMA predictions IS 'Prediction storage, versioning, and learning system';

COMMENT ON TABLE predictions.game_predictions IS
'All model predictions with ensemble components. Multiple predictions per game allowed (versioning).';

COMMENT ON TABLE predictions.prediction_versions IS
'Tracks how predictions changed over time as new information (injuries, weather, line moves) became available.';

COMMENT ON TABLE predictions.retrospectives IS
'Post-game analysis of prediction accuracy. This is where we learn from failures! One per game.';

COMMENT ON TABLE predictions.learning_loop IS
'Aggregate patterns extracted from retrospectives. Drives systematic model improvement via RL feedback.';

COMMENT ON COLUMN predictions.retrospectives.surprise_factor IS
'How unexpected was this outcome? Calculated from prediction confidence and margin of error. 0=expected, 1=shocking.';

COMMENT ON COLUMN predictions.learning_loop.expected_improvement_pct IS
'Expected ROI improvement if this pattern is addressed. Format: 0.005 = 0.5% improvement.';

-- ==============================================================================
-- COMPLETE
-- ==============================================================================
-- Migration 018 complete! Run with:
-- psql -h localhost -p 5544 -U dro -d devdb01 -f db/migrations/018_predictions_schema.sql
