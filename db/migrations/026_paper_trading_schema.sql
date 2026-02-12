-- Paper Trading Schema
-- Created: November 7, 2025
-- Purpose: Track simulated positions and performance for ensemble validation

CREATE SCHEMA IF NOT EXISTS paper_trading;

-- Positions table: Track individual bets
CREATE TABLE IF NOT EXISTS paper_trading.positions (
    position_id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    prop_type TEXT NOT NULL,  -- 'passing_yards', 'rushing_yards', 'receiving_yards'
    side TEXT NOT NULL CHECK (side IN ('over', 'under')),
    line FLOAT NOT NULL,
    odds INT NOT NULL,  -- American odds (e.g., -110)
    bet_size FLOAT NOT NULL,
    kelly_fraction FLOAT,

    -- Model predictions
    model_prediction FLOAT NOT NULL,
    model_std FLOAT NOT NULL,
    model_prob_over FLOAT,
    ensemble_mean FLOAT,
    ensemble_std FLOAT,

    -- Individual model contributions
    bayesian_mean FLOAT,
    xgboost_mean FLOAT,
    bnn_mean FLOAT,
    bnn_calibrated_mean FLOAT,
    state_space_mean FLOAT,

    -- CQL adjustment
    cql_adjustment FLOAT DEFAULT 1.0,
    used_cql BOOLEAN DEFAULT FALSE,

    -- Timing
    placed_at TIMESTAMP DEFAULT NOW(),
    game_time TIMESTAMP,

    -- Results
    status TEXT DEFAULT 'open' CHECK (status IN ('open', 'win', 'loss', 'push', 'cancelled')),
    actual_result FLOAT,  -- Actual yards
    pnl FLOAT,  -- Profit/loss
    roi FLOAT,  -- Return on investment (pnl / bet_size)
    closed_at TIMESTAMP,

    -- Metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Performance summary table: Daily metrics
CREATE TABLE IF NOT EXISTS paper_trading.performance (
    date DATE PRIMARY KEY,
    starting_bankroll FLOAT NOT NULL,
    ending_bankroll FLOAT NOT NULL,
    total_bets INT NOT NULL DEFAULT 0,
    wins INT NOT NULL DEFAULT 0,
    losses INT NOT NULL DEFAULT 0,
    pushes INT NOT NULL DEFAULT 0,
    cancelled INT NOT NULL DEFAULT 0,

    -- Performance metrics
    win_rate FLOAT,  -- wins / (wins + losses)
    roi FLOAT,  -- (ending - starting) / starting
    profit FLOAT,  -- ending - starting
    avg_bet_size FLOAT,
    max_bet_size FLOAT,

    -- Risk metrics
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    max_drawdown_pct FLOAT,

    -- Calibration metrics
    calibration_90 FLOAT,  -- % of outcomes within 90% CI
    calibration_68 FLOAT,  -- % of outcomes within 68% CI
    avg_prediction_error FLOAT,  -- MAE

    -- CQL metrics
    avg_cql_adjustment FLOAT,
    cql_bets INT DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Bankroll history table: Track bankroll over time
CREATE TABLE IF NOT EXISTS paper_trading.bankroll_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    bankroll FLOAT NOT NULL,
    available_capital FLOAT NOT NULL,
    open_positions INT NOT NULL DEFAULT 0,
    daily_pnl FLOAT DEFAULT 0.0,
    cumulative_pnl FLOAT DEFAULT 0.0,
    roi FLOAT DEFAULT 0.0,
    notes TEXT
);

-- Odds history table: Track odds movement
CREATE TABLE IF NOT EXISTS paper_trading.odds_snapshots (
    id SERIAL PRIMARY KEY,
    position_id INT REFERENCES paper_trading.positions(position_id),
    timestamp TIMESTAMP DEFAULT NOW(),
    bookmaker TEXT NOT NULL,
    odds_over INT,
    odds_under INT,
    line FLOAT,

    -- Market context
    total_handle FLOAT,
    sharp_action BOOLEAN,
    line_movement FLOAT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_positions_game_id ON paper_trading.positions(game_id);
CREATE INDEX IF NOT EXISTS idx_positions_player_id ON paper_trading.positions(player_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON paper_trading.positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_placed_at ON paper_trading.positions(placed_at);
CREATE INDEX IF NOT EXISTS idx_positions_prop_type ON paper_trading.positions(prop_type);
CREATE INDEX IF NOT EXISTS idx_performance_date ON paper_trading.performance(date);
CREATE INDEX IF NOT EXISTS idx_bankroll_timestamp ON paper_trading.bankroll_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_odds_position ON paper_trading.odds_snapshots(position_id);

-- Views for reporting

-- Active positions view
CREATE OR REPLACE VIEW paper_trading.v_active_positions AS
SELECT
    p.*,
    g.home_team,
    g.away_team,
    g.season,
    g.week,
    EXTRACT(EPOCH FROM (game_time - NOW())) / 3600 AS hours_until_game
FROM paper_trading.positions p
LEFT JOIN games g ON p.game_id = g.game_id
WHERE p.status = 'open'
ORDER BY game_time;

-- Performance summary view
CREATE OR REPLACE VIEW paper_trading.v_performance_summary AS
SELECT
    COUNT(*) AS total_bets,
    SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN status = 'loss' THEN 1 ELSE 0 END) AS losses,
    SUM(CASE WHEN status = 'push' THEN 1 ELSE 0 END) AS pushes,
    ROUND(100.0 * SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN status IN ('win', 'loss') THEN 1 ELSE 0 END), 0), 2) AS win_rate_pct,
    ROUND(SUM(COALESCE(pnl, 0)), 2) AS total_pnl,
    ROUND(100.0 * SUM(COALESCE(pnl, 0)) / SUM(bet_size), 2) AS roi_pct,
    ROUND(AVG(bet_size), 2) AS avg_bet,
    ROUND(AVG(CASE WHEN used_cql THEN cql_adjustment ELSE NULL END), 3) AS avg_cql_adj,
    COUNT(CASE WHEN used_cql THEN 1 END) AS cql_bets
FROM paper_trading.positions
WHERE status != 'cancelled';

-- Model performance comparison view
CREATE OR REPLACE VIEW paper_trading.v_model_performance AS
SELECT
    prop_type,
    COUNT(*) AS n_bets,
    ROUND(AVG(ABS(actual_result - ensemble_mean)), 2) AS ensemble_mae,
    ROUND(AVG(ABS(actual_result - bayesian_mean)), 2) AS bayesian_mae,
    ROUND(AVG(ABS(actual_result - xgboost_mean)), 2) AS xgboost_mae,
    ROUND(AVG(ABS(actual_result - bnn_mean)), 2) AS bnn_mae,
    ROUND(AVG(ABS(actual_result - bnn_calibrated_mean)), 2) AS bnn_calibrated_mae,
    ROUND(AVG(ABS(actual_result - state_space_mean)), 2) AS state_space_mae
FROM paper_trading.positions
WHERE status IN ('win', 'loss', 'push')
  AND actual_result IS NOT NULL
GROUP BY prop_type;

-- Recent performance view (last 30 days)
CREATE OR REPLACE VIEW paper_trading.v_recent_performance AS
SELECT
    date,
    ending_bankroll,
    profit,
    roi * 100 AS roi_pct,
    win_rate * 100 AS win_rate_pct,
    total_bets,
    wins,
    losses,
    sharpe_ratio,
    max_drawdown_pct * 100 AS max_dd_pct,
    calibration_90 * 100 AS cal_90_pct,
    avg_cql_adjustment
FROM paper_trading.performance
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY date DESC;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION paper_trading.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER positions_updated_at
    BEFORE UPDATE ON paper_trading.positions
    FOR EACH ROW
    EXECUTE FUNCTION paper_trading.update_updated_at();

CREATE TRIGGER performance_updated_at
    BEFORE UPDATE ON paper_trading.performance
    FOR EACH ROW
    EXECUTE FUNCTION paper_trading.update_updated_at();

-- Initialize bankroll
INSERT INTO paper_trading.bankroll_history (bankroll, available_capital, open_positions)
VALUES (10000.0, 10000.0, 0)
ON CONFLICT DO NOTHING;

COMMENT ON SCHEMA paper_trading IS 'Paper trading system for ensemble validation (no real money)';
COMMENT ON TABLE paper_trading.positions IS 'Individual simulated bets with full model predictions';
COMMENT ON TABLE paper_trading.performance IS 'Daily performance summary metrics';
COMMENT ON TABLE paper_trading.bankroll_history IS 'Bankroll tracking over time';
COMMENT ON TABLE paper_trading.odds_snapshots IS 'Historical odds data for analysis';
