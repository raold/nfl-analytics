-- Migration 027: Paper Trading Tables
-- Created: November 6, 2025
-- Purpose: Add paper trading infrastructure for risk-free validation

-- Table: paper_trades
-- Records individual paper trades (hypothetical bets without real money)
CREATE TABLE IF NOT EXISTS paper_trades (
    paper_trade_id SERIAL PRIMARY KEY,
    trade_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Game and player identification
    game_id TEXT,
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,

    -- Bet details
    prop_type TEXT NOT NULL,  -- 'passing_yards', 'rushing_yards', 'receiving_yards', etc.
    line_value DOUBLE PRECISION NOT NULL,  -- Betting line (e.g., 275.5 yards)
    bet_side TEXT NOT NULL,  -- 'over' or 'under'
    odds DOUBLE PRECISION NOT NULL,  -- American odds (e.g., -110, +120)
    bookmaker_key TEXT NOT NULL,  -- 'pinnacle', 'draftkings', etc.
    bet_amount DOUBLE PRECISION NOT NULL,  -- Bet size in dollars

    -- Model metadata
    edge DOUBLE PRECISION,  -- Model edge as decimal (e.g., 0.035 = 3.5%)
    model_prediction DOUBLE PRECISION,  -- Model's predicted value
    model_std DOUBLE PRECISION,  -- Model's uncertainty (standard deviation)
    confidence DOUBLE PRECISION,  -- Model confidence score (0-1)

    -- Settlement
    actual_result DOUBLE PRECISION,  -- Actual player stat (e.g., 312.0 yards)
    outcome TEXT NOT NULL DEFAULT 'pending',  -- 'win', 'loss', 'push', 'pending'
    settled_at TIMESTAMPTZ,
    profit_loss DOUBLE PRECISION,  -- Profit/loss amount in dollars

    -- Tracking metadata
    model_version TEXT,  -- Model version identifier
    strategy TEXT,  -- Strategy name (e.g., 'pinnacle_exchange', 'player_props')
    session_id TEXT NOT NULL,  -- Links to paper_trading_sessions

    -- Constraints
    CONSTRAINT paper_trades_outcome_check
        CHECK (outcome IN ('win', 'loss', 'push', 'pending')),
    CONSTRAINT paper_trades_bet_side_check
        CHECK (bet_side IN ('over', 'under')),
    CONSTRAINT paper_trades_bet_amount_positive
        CHECK (bet_amount > 0)
);

-- Indexes for paper_trades
CREATE INDEX idx_paper_trades_pending
    ON paper_trades(outcome, trade_timestamp DESC)
    WHERE outcome = 'pending';

CREATE INDEX idx_paper_trades_settled
    ON paper_trades(settled_at DESC)
    WHERE settled_at IS NOT NULL;

CREATE INDEX idx_paper_trades_session
    ON paper_trades(session_id, trade_timestamp DESC);

CREATE INDEX idx_paper_trades_player
    ON paper_trades(player_id, prop_type, trade_timestamp DESC);

CREATE INDEX idx_paper_trades_bookmaker
    ON paper_trades(bookmaker_key, outcome, trade_timestamp DESC);

-- Comment on table
COMMENT ON TABLE paper_trades IS
'Paper trades (hypothetical bets) for risk-free strategy validation. Each row represents a single bet that would have been placed in production, along with actual outcomes for performance analysis.';

COMMENT ON COLUMN paper_trades.edge IS
'Model edge as decimal (e.g., 0.035 = 3.5% edge). Calculated as model_prob_win - implied_prob_from_odds.';

COMMENT ON COLUMN paper_trades.model_prediction IS
'Model predicted value for the player stat (e.g., 290.5 passing yards). Used to determine if bet would trigger in production.';

COMMENT ON COLUMN paper_trades.model_std IS
'Model uncertainty (standard deviation). Used for risk-adjusted position sizing and confidence filtering.';

---

-- Table: paper_trading_sessions
-- Tracks overall performance metrics for paper trading sessions
CREATE TABLE IF NOT EXISTS paper_trading_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,

    -- Bankroll tracking
    initial_bankroll DOUBLE PRECISION NOT NULL,
    final_bankroll DOUBLE PRECISION,

    -- Performance metrics
    total_bets INTEGER,
    win_rate DOUBLE PRECISION,  -- Win rate as decimal (0-1)
    roi DOUBLE PRECISION,  -- Return on investment as decimal
    sharpe_ratio DOUBLE PRECISION,  -- Sharpe ratio (risk-adjusted returns)
    max_drawdown DOUBLE PRECISION,  -- Maximum drawdown as decimal

    -- Strategy identification
    strategy TEXT,

    -- Constraints
    CONSTRAINT paper_trading_sessions_initial_bankroll_positive
        CHECK (initial_bankroll > 0),
    CONSTRAINT paper_trading_sessions_final_bankroll_positive
        CHECK (final_bankroll IS NULL OR final_bankroll >= 0)
);

-- Indexes for paper_trading_sessions
CREATE INDEX idx_paper_trading_sessions_started
    ON paper_trading_sessions(started_at DESC);

CREATE INDEX idx_paper_trading_sessions_strategy
    ON paper_trading_sessions(strategy, started_at DESC);

CREATE INDEX idx_paper_trading_sessions_roi
    ON paper_trading_sessions(roi DESC NULLS LAST)
    WHERE ended_at IS NOT NULL;

-- Comment on table
COMMENT ON TABLE paper_trading_sessions IS
'Aggregated performance metrics for paper trading sessions. Each session represents a distinct testing period (e.g., one week of paper trading with specific strategy parameters).';

COMMENT ON COLUMN paper_trading_sessions.sharpe_ratio IS
'Sharpe ratio calculated from daily returns: mean(returns) / std(returns). Higher is better (>2.0 is excellent).';

COMMENT ON COLUMN paper_trading_sessions.max_drawdown IS
'Maximum peak-to-trough decline during the session, as fraction of initial bankroll. Lower is better (<0.15 is good).';

---

-- View: paper_trades_summary
-- Convenient view for analyzing paper trading performance
CREATE OR REPLACE VIEW paper_trades_summary AS
SELECT
    session_id,
    strategy,
    COUNT(*) as total_trades,
    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
    SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
    SUM(CASE WHEN outcome = 'pending' THEN 1 ELSE 0 END) as pending,
    AVG(CASE WHEN outcome IN ('win', 'loss') THEN
        CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END
    END) as win_rate,
    SUM(bet_amount) as total_wagered,
    SUM(profit_loss) as total_profit_loss,
    CASE
        WHEN SUM(bet_amount) > 0 THEN SUM(profit_loss) / SUM(bet_amount)
        ELSE 0
    END as roi,
    AVG(edge) as avg_edge,
    MIN(trade_timestamp) as first_trade,
    MAX(trade_timestamp) as last_trade
FROM paper_trades
GROUP BY session_id, strategy;

COMMENT ON VIEW paper_trades_summary IS
'Aggregated summary statistics for each paper trading session and strategy.';

---

-- View: paper_trades_by_prop_type
-- Performance breakdown by prop type
CREATE OR REPLACE VIEW paper_trades_by_prop_type AS
SELECT
    prop_type,
    COUNT(*) as total_trades,
    AVG(CASE WHEN outcome IN ('win', 'loss') THEN
        CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END
    END) as win_rate,
    SUM(bet_amount) as total_wagered,
    SUM(profit_loss) as total_profit_loss,
    CASE
        WHEN SUM(bet_amount) > 0 THEN SUM(profit_loss) / SUM(bet_amount)
        ELSE 0
    END as roi,
    AVG(edge) as avg_edge,
    AVG(model_std) as avg_model_std
FROM paper_trades
WHERE outcome IN ('win', 'loss', 'push')
GROUP BY prop_type;

COMMENT ON VIEW paper_trades_by_prop_type IS
'Performance breakdown by prop type (passing_yards, rushing_yards, etc.). Useful for identifying which props are most profitable.';

---

-- View: paper_trades_by_bookmaker
-- Performance breakdown by bookmaker
CREATE OR REPLACE VIEW paper_trades_by_bookmaker AS
SELECT
    bookmaker_key,
    COUNT(*) as total_trades,
    AVG(CASE WHEN outcome IN ('win', 'loss') THEN
        CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END
    END) as win_rate,
    SUM(bet_amount) as total_wagered,
    SUM(profit_loss) as total_profit_loss,
    CASE
        WHEN SUM(bet_amount) > 0 THEN SUM(profit_loss) / SUM(bet_amount)
        ELSE 0
    END as roi,
    AVG(edge) as avg_edge
FROM paper_trades
WHERE outcome IN ('win', 'loss', 'push')
GROUP BY bookmaker_key;

COMMENT ON VIEW paper_trades_by_bookmaker IS
'Performance breakdown by bookmaker. Useful for comparing Pinnacle vs DraftKings vs FanDuel performance.';

---

-- Grant permissions (adjust as needed)
GRANT SELECT, INSERT, UPDATE ON paper_trades TO dro;
GRANT SELECT, INSERT, UPDATE ON paper_trading_sessions TO dro;
GRANT USAGE, SELECT ON SEQUENCE paper_trades_paper_trade_id_seq TO dro;
GRANT SELECT ON paper_trades_summary TO dro;
GRANT SELECT ON paper_trades_by_prop_type TO dro;
GRANT SELECT ON paper_trades_by_bookmaker TO dro;
