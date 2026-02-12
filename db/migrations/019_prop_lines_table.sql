-- ============================================================================
-- Migration 019: Player Props Lines Table
-- ============================================================================
-- Creates table for storing player prop betting lines from sportsbooks
-- Includes TimescaleDB hypertable for efficient time-series queries
--
-- Usage:
--   psql -h localhost -p 5544 -U dro devdb01 -f db/migrations/019_prop_lines_table.sql

-- ============================================================================
-- Main Table: prop_lines_history
-- ============================================================================

CREATE TABLE IF NOT EXISTS prop_lines_history (
    -- Event identification
    event_id TEXT NOT NULL,
    game_id TEXT,
    sport_key TEXT DEFAULT 'americanfootball_nfl',
    commence_time TIMESTAMPTZ,

    -- Player identification
    player_id TEXT NOT NULL,                    -- GSIS player ID (link to players table)
    player_name TEXT NOT NULL,
    player_position TEXT,
    player_team TEXT,

    -- Prop details
    prop_type TEXT NOT NULL,                    -- e.g., 'passing_yards', 'rushing_tds', 'receptions'
    market_key TEXT NOT NULL,                   -- API market key (e.g., 'player_pass_yds')
    line_value DOUBLE PRECISION NOT NULL,       -- The prop line (e.g., 250.5 yards)

    -- Odds
    over_odds DOUBLE PRECISION,                 -- American odds for over
    under_odds DOUBLE PRECISION,                -- American odds for under

    -- Bookmaker info
    bookmaker_key TEXT NOT NULL,
    bookmaker_title TEXT,

    -- Metadata
    snapshot_at TIMESTAMPTZ NOT NULL,          -- When this snapshot was taken
    market_last_update TIMESTAMPTZ,            -- When book last updated this market
    fetched_at TIMESTAMPTZ DEFAULT NOW(),      -- When we fetched this data

    -- Derived fields (computed at ingestion)
    over_implied_prob DOUBLE PRECISION,        -- Implied probability of over
    under_implied_prob DOUBLE PRECISION,       -- Implied probability of under
    book_hold DOUBLE PRECISION,                -- Book hold/vig percentage
    line_move_since_open DOUBLE PRECISION,     -- How much line has moved (will update later)

    -- Primary key
    PRIMARY KEY (event_id, player_id, prop_type, bookmaker_key, snapshot_at)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_prop_lines_player_id ON prop_lines_history (player_id, prop_type, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_prop_lines_game_id ON prop_lines_history (game_id, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_prop_lines_event_id ON prop_lines_history (event_id, player_id, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_prop_lines_snapshot ON prop_lines_history (snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_prop_lines_prop_type ON prop_lines_history (prop_type, player_team, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_prop_lines_bookmaker ON prop_lines_history (bookmaker_key, prop_type, snapshot_at DESC);

-- BRIN index for time-based queries (very efficient for large time-series data)
CREATE INDEX IF NOT EXISTS idx_prop_lines_snapshot_brin ON prop_lines_history USING BRIN (snapshot_at) WITH (pages_per_range = 32);

-- Convert to TimescaleDB hypertable (partition by snapshot_at)
-- This enables much faster time-series queries and automatic compression
SELECT create_hypertable(
    'prop_lines_history',
    'snapshot_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Enable compression for older data (compress chunks older than 30 days)
ALTER TABLE prop_lines_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'player_id, prop_type, bookmaker_key',
    timescaledb.compress_orderby = 'snapshot_at DESC'
);

-- Add compression policy
SELECT add_compression_policy('prop_lines_history', INTERVAL '30 days', if_not_exists => TRUE);

COMMENT ON TABLE prop_lines_history IS 'Historical player prop betting lines from multiple sportsbooks. Time-series optimized with TimescaleDB.';

-- ============================================================================
-- View: latest_prop_lines
-- ============================================================================
-- Provides latest prop lines for each player/prop/book combination

CREATE OR REPLACE VIEW latest_prop_lines AS
SELECT DISTINCT ON (event_id, player_id, prop_type, bookmaker_key)
    event_id,
    game_id,
    commence_time,
    player_id,
    player_name,
    player_position,
    player_team,
    prop_type,
    market_key,
    line_value,
    over_odds,
    under_odds,
    bookmaker_key,
    bookmaker_title,
    snapshot_at,
    market_last_update,
    over_implied_prob,
    under_implied_prob,
    book_hold
FROM prop_lines_history
WHERE snapshot_at >= NOW() - INTERVAL '24 hours'  -- Only show recent lines
ORDER BY event_id, player_id, prop_type, bookmaker_key, snapshot_at DESC;

COMMENT ON VIEW latest_prop_lines IS 'Latest player prop lines from the past 24 hours';

-- ============================================================================
-- View: best_prop_lines
-- ============================================================================
-- Shows best available line for each player/prop (highest odds)

CREATE OR REPLACE VIEW best_prop_lines AS
WITH latest AS (
    SELECT DISTINCT ON (event_id, player_id, prop_type, bookmaker_key)
        *
    FROM prop_lines_history
    WHERE snapshot_at >= NOW() - INTERVAL '24 hours'
    ORDER BY event_id, player_id, prop_type, bookmaker_key, snapshot_at DESC
),
best_over AS (
    SELECT DISTINCT ON (event_id, player_id, prop_type)
        event_id,
        player_id,
        player_name,
        player_team,
        prop_type,
        line_value,
        over_odds AS best_over_odds,
        bookmaker_title AS best_over_book,
        snapshot_at
    FROM latest
    ORDER BY event_id, player_id, prop_type, over_odds DESC
),
best_under AS (
    SELECT DISTINCT ON (event_id, player_id, prop_type)
        event_id,
        player_id,
        prop_type,
        under_odds AS best_under_odds,
        bookmaker_title AS best_under_book
    FROM latest
    ORDER BY event_id, player_id, prop_type, under_odds DESC
)
SELECT
    o.event_id,
    o.player_id,
    o.player_name,
    o.player_team,
    o.prop_type,
    o.line_value,
    o.best_over_odds,
    o.best_over_book,
    u.best_under_odds,
    u.best_under_book,
    o.snapshot_at
FROM best_over o
JOIN best_under u ON o.event_id = u.event_id AND o.player_id = u.player_id AND o.prop_type = u.prop_type;

COMMENT ON VIEW best_prop_lines IS 'Best available odds for each player prop across all sportsbooks';

-- ============================================================================
-- Function: Calculate line movement
-- ============================================================================

CREATE OR REPLACE FUNCTION calculate_prop_line_movement(
    p_player_id TEXT,
    p_prop_type TEXT,
    p_bookmaker TEXT,
    p_hours_lookback INT DEFAULT 24
)
RETURNS TABLE (
    snapshot_at TIMESTAMPTZ,
    line_value DOUBLE PRECISION,
    over_odds DOUBLE PRECISION,
    under_odds DOUBLE PRECISION,
    line_change DOUBLE PRECISION,
    odds_change DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH line_history AS (
        SELECT
            pl.snapshot_at,
            pl.line_value,
            pl.over_odds,
            pl.under_odds,
            LAG(pl.line_value) OVER (ORDER BY pl.snapshot_at) AS prev_line,
            LAG(pl.over_odds) OVER (ORDER BY pl.snapshot_at) AS prev_over_odds
        FROM prop_lines_history pl
        WHERE pl.player_id = p_player_id
            AND pl.prop_type = p_prop_type
            AND pl.bookmaker_key = p_bookmaker
            AND pl.snapshot_at >= NOW() - (p_hours_lookback || ' hours')::INTERVAL
        ORDER BY pl.snapshot_at
    )
    SELECT
        lh.snapshot_at,
        lh.line_value,
        lh.over_odds,
        lh.under_odds,
        lh.line_value - COALESCE(lh.prev_line, lh.line_value) AS line_change,
        lh.over_odds - COALESCE(lh.prev_over_odds, lh.over_odds) AS odds_change
    FROM line_history lh;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION calculate_prop_line_movement IS 'Calculate line movement for a specific player prop over time';

-- ============================================================================
-- Table: prop_line_openings
-- ============================================================================
-- Track opening lines for CLV analysis

CREATE TABLE IF NOT EXISTS prop_line_openings (
    event_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    prop_type TEXT NOT NULL,
    bookmaker_key TEXT NOT NULL,
    opening_line DOUBLE PRECISION NOT NULL,
    opening_over_odds DOUBLE PRECISION,
    opening_under_odds DOUBLE PRECISION,
    opened_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (event_id, player_id, prop_type, bookmaker_key)
);

CREATE INDEX IF NOT EXISTS idx_prop_openings_player ON prop_line_openings (player_id, prop_type);

COMMENT ON TABLE prop_line_openings IS 'Opening lines for prop bets - used for Closing Line Value (CLV) analysis';

-- ============================================================================
-- Success Message
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Migration 019 completed successfully';
    RAISE NOTICE '   - Created prop_lines_history table (TimescaleDB hypertable)';
    RAISE NOTICE '   - Created latest_prop_lines view';
    RAISE NOTICE '   - Created best_prop_lines view';
    RAISE NOTICE '   - Created calculate_prop_line_movement function';
    RAISE NOTICE '   - Created prop_line_openings table';
    RAISE NOTICE '   - Added compression policy (30 days)';
END $$;
