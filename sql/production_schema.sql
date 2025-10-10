-- production_schema.sql
-- Database schema for NFL betting production system

-- ============================================================================
-- Bets Table - Track all production bets
-- ============================================================================

CREATE TABLE IF NOT EXISTS bets (
    bet_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Game info
    game_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    -- Bet details
    bet_type VARCHAR(20) NOT NULL CHECK (bet_type IN ('spread', 'total', 'moneyline')),
    side VARCHAR(20) NOT NULL CHECK (side IN ('home', 'away', 'over', 'under')),
    line REAL NOT NULL,
    odds INTEGER NOT NULL,  -- American odds (e.g., -110, +150)
    stake REAL NOT NULL CHECK (stake > 0),
    is_paper_trade BOOLEAN NOT NULL DEFAULT FALSE,  -- True for paper trading, False for real money

    -- Model prediction
    model_name VARCHAR(50),  -- Which model made this prediction (xgboost, cql, iql)
    prediction REAL CHECK (prediction >= 0 AND prediction <= 1),  -- Model's win probability

    -- Result (NULL until game settles)
    result VARCHAR(10) CHECK (result IN ('win', 'loss', 'push', NULL)),
    payout REAL,  -- Net payout (profit/loss, not including stake)
    home_score INTEGER,
    away_score INTEGER,

    -- Line movement tracking
    closing_line REAL,  -- Closing line for CLV calculation
    clv REAL,  -- Closing Line Value (difference between our line and closing)

    -- Sportsbook tracking
    sportsbook VARCHAR(50),  -- Which book was used (FanDuel, DraftKings, etc.)

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_bets_timestamp ON bets(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bets_game_id ON bets(game_id);
CREATE INDEX IF NOT EXISTS idx_bets_season_week ON bets(season, week);
CREATE INDEX IF NOT EXISTS idx_bets_result ON bets(result);
CREATE INDEX IF NOT EXISTS idx_bets_model_name ON bets(model_name);
CREATE INDEX IF NOT EXISTS idx_bets_paper_trade ON bets(is_paper_trade);

-- ============================================================================
-- Model Stats Table - Thompson Sampling statistics
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_stats (
    model_name VARCHAR(50) PRIMARY KEY,

    -- Beta distribution parameters
    alpha REAL NOT NULL DEFAULT 1.0,  -- Wins + 1 (prior)
    beta REAL NOT NULL DEFAULT 1.0,   -- Losses + 1 (prior)

    -- Performance counters
    n_bets INTEGER NOT NULL DEFAULT 0,
    n_wins INTEGER NOT NULL DEFAULT 0,
    n_losses INTEGER NOT NULL DEFAULT 0,

    -- Metadata
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Initialize models
INSERT INTO model_stats (model_name, alpha, beta)
VALUES
    ('xgboost', 1.0, 1.0),
    ('cql', 1.0, 1.0),
    ('iql', 1.0, 1.0)
ON CONFLICT (model_name) DO NOTHING;

-- ============================================================================
-- Stress Tests Table - Weekly bootstrap results
-- ============================================================================

CREATE TABLE IF NOT EXISTS stress_tests (
    test_id SERIAL PRIMARY KEY,
    test_date TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Test parameters
    weeks_tested INTEGER NOT NULL,
    n_bets INTEGER NOT NULL,

    -- Actual performance
    actual_roi REAL NOT NULL,

    -- Bootstrap distribution
    bootstrap_mean_roi REAL NOT NULL,
    bootstrap_median_roi REAL NOT NULL,
    bootstrap_std_roi REAL NOT NULL,
    bootstrap_5th_percentile REAL NOT NULL,
    bootstrap_95th_percentile REAL NOT NULL,

    -- Analysis
    percentile_rank REAL NOT NULL,  -- Where actual ROI ranks (0-100)
    worst_case_roi REAL NOT NULL,   -- 1st percentile
    best_case_roi REAL NOT NULL,    -- 99th percentile

    -- Result
    passed BOOLEAN NOT NULL,  -- True if actual ROI > 5th percentile

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stress_tests_date ON stress_tests(test_date DESC);

-- ============================================================================
-- Line Movements Table - Track opening to closing lines
-- ============================================================================

CREATE TABLE IF NOT EXISTS line_movements (
    movement_id SERIAL PRIMARY KEY,

    -- Game info
    game_id VARCHAR(50) NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_date DATE NOT NULL,
    home_team VARCHAR(10) NOT NULL,
    away_team VARCHAR(10) NOT NULL,

    -- Spread movement
    opening_spread REAL,
    closing_spread REAL,
    line_movement REAL,  -- closing - opening

    -- Total movement
    opening_total REAL,
    closing_total REAL,
    total_movement REAL,

    -- Timestamps
    opening_timestamp TIMESTAMP,
    closing_timestamp TIMESTAMP,

    -- Sharp indicators
    sharp_indicators TEXT[],  -- Array of indicators: ['early_week_move', 'steam_move', etc.]
    steam_move_count INTEGER DEFAULT 0,
    reverse_line_move BOOLEAN DEFAULT FALSE,

    -- Public vs sharp
    public_side VARCHAR(10),  -- Which side has more public money
    sharp_side VARCHAR(10),   -- Which side has sharp money (if known)

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_line_movements_game_id ON line_movements(game_id);
CREATE INDEX IF NOT EXISTS idx_line_movements_date ON line_movements(game_date DESC);

-- ============================================================================
-- Performance Snapshots Table - Daily performance tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,

    -- Bankroll
    bankroll REAL NOT NULL,
    bankroll_change REAL NOT NULL,

    -- Performance metrics (all-time)
    total_bets INTEGER NOT NULL,
    total_wins INTEGER NOT NULL,
    total_losses INTEGER NOT NULL,
    total_pushes INTEGER NOT NULL,

    roi REAL NOT NULL,
    win_rate REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    current_drawdown REAL NOT NULL,

    avg_clv REAL,
    brier_score REAL,
    log_loss REAL,

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date ON performance_snapshots(snapshot_date DESC);

-- ============================================================================
-- Alerts Table - Track all alerts and warnings
-- ============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id SERIAL PRIMARY KEY,
    alert_date TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Alert details
    alert_type VARCHAR(50) NOT NULL,  -- losing_streak, large_drawdown, model_drift, etc.
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    message TEXT NOT NULL,

    -- Context
    metric_name VARCHAR(50),  -- Which metric triggered alert
    metric_value REAL,        -- Value that triggered alert
    threshold REAL,           -- Threshold that was exceeded

    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_note TEXT,

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(alert_date DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved, alert_date DESC);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Recent performance (last 30 days, real money only)
CREATE OR REPLACE VIEW recent_performance AS
SELECT
    COUNT(*) AS n_bets,
    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses,
    SUM(stake) AS total_staked,
    SUM(payout) AS total_payout,
    SUM(payout) / SUM(stake) AS roi,
    AVG(CASE WHEN result IN ('win', 'loss') THEN
        CASE WHEN result = 'win' THEN 1.0 ELSE 0.0 END
    END) AS win_rate,
    AVG(clv) AS avg_clv
FROM bets
WHERE timestamp > NOW() - INTERVAL '30 days'
    AND result IS NOT NULL
    AND is_paper_trade = FALSE;

-- Model comparison (real money only)
CREATE OR REPLACE VIEW model_comparison AS
SELECT
    model_name,
    COUNT(*) AS n_bets,
    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) AS losses,
    SUM(payout) / SUM(stake) AS roi,
    AVG(CASE WHEN result IN ('win', 'loss') THEN
        CASE WHEN result = 'win' THEN 1.0 ELSE 0.0 END
    END) AS win_rate,
    AVG(clv) AS avg_clv
FROM bets
WHERE result IS NOT NULL
    AND model_name IS NOT NULL
    AND is_paper_trade = FALSE
GROUP BY model_name;

-- Unresolved alerts
CREATE OR REPLACE VIEW active_alerts AS
SELECT *
FROM alerts
WHERE resolved = FALSE
ORDER BY
    CASE severity
        WHEN 'critical' THEN 1
        WHEN 'error' THEN 2
        WHEN 'warning' THEN 3
        WHEN 'info' THEN 4
    END,
    alert_date DESC;

-- ============================================================================
-- Functions
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
DROP TRIGGER IF EXISTS update_bets_updated_at ON bets;
CREATE TRIGGER update_bets_updated_at
    BEFORE UPDATE ON bets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_line_movements_updated_at ON line_movements;
CREATE TRIGGER update_line_movements_updated_at
    BEFORE UPDATE ON line_movements
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Sample Data (for testing)
-- ============================================================================

-- Uncomment to insert sample data:
/*
INSERT INTO bets (game_id, season, week, bet_type, side, line, odds, stake, model_name, prediction, result, payout, sportsbook)
VALUES
    ('2024_10_KC_SF', 2024, 10, 'spread', 'home', -3.5, -110, 250, 'xgboost', 0.58, 'win', 227.27, 'FanDuel'),
    ('2024_10_BUF_MIA', 2024, 10, 'total', 'over', 47.5, -110, 200, 'cql', 0.54, 'loss', -200, 'DraftKings'),
    ('2024_10_DAL_PHI', 2024, 10, 'spread', 'away', 3.0, -110, 300, 'iql', 0.56, 'win', 272.73, 'BetMGM');
*/
