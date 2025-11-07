-- Migration 026: Pinnacle Exchange Fields
-- Created: November 6, 2025
-- Purpose: Add exchange-specific fields for Pinnacle and other low-vig bookmakers

-- Add exchange-specific columns to prop_lines_history
ALTER TABLE prop_lines_history
    ADD COLUMN IF NOT EXISTS is_exchange BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS back_price DOUBLE PRECISION,  -- For exchange-style back pricing
    ADD COLUMN IF NOT EXISTS lay_price DOUBLE PRECISION,   -- For exchange-style lay pricing
    ADD COLUMN IF NOT EXISTS liquidity DOUBLE PRECISION;   -- Available volume/depth

-- Add bookmaker tier classification
ALTER TABLE prop_lines_history
    ADD COLUMN IF NOT EXISTS bookmaker_tier TEXT;  -- 'sharp', 'square', 'exchange'

COMMENT ON COLUMN prop_lines_history.is_exchange IS
'True if this is an exchange-style bookmaker (Betfair, Pinnacle sometimes acts like one)';

COMMENT ON COLUMN prop_lines_history.back_price IS
'Back price for exchange-style markets (price to bet on outcome)';

COMMENT ON COLUMN prop_lines_history.lay_price IS
'Lay price for exchange-style markets (price to bet against outcome)';

COMMENT ON COLUMN prop_lines_history.liquidity IS
'Available liquidity/volume at this price point. Useful for assessing market depth.';

COMMENT ON COLUMN prop_lines_history.bookmaker_tier IS
'Classification of bookmaker: sharp (Pinnacle, Circa), square (DraftKings, FanDuel), or exchange (Betfair)';

-- Create index for exchange queries
CREATE INDEX IF NOT EXISTS idx_prop_lines_exchange
    ON prop_lines_history(bookmaker_key, is_exchange, snapshot_at DESC)
    WHERE is_exchange = TRUE;

-- Create index for bookmaker tier queries
CREATE INDEX IF NOT EXISTS idx_prop_lines_bookmaker_tier
    ON prop_lines_history(bookmaker_tier, snapshot_at DESC)
    WHERE bookmaker_tier IS NOT NULL;

-- Create index for low-hold lines (Pinnacle, sharp books)
CREATE INDEX IF NOT EXISTS idx_prop_lines_low_hold
    ON prop_lines_history(book_hold, snapshot_at DESC)
    WHERE book_hold < 0.03;  -- Less than 3% hold

---

-- Update existing Pinnacle records to mark as sharp tier
UPDATE prop_lines_history
SET bookmaker_tier = 'sharp'
WHERE bookmaker_key IN ('pinnacle', 'pinnacleus', 'circa', 'bookmaker_eu')
  AND bookmaker_tier IS NULL;

-- Update existing square book records
UPDATE prop_lines_history
SET bookmaker_tier = 'square'
WHERE bookmaker_key IN ('draftkings', 'fanduel', 'betmgm', 'caesars', 'barstool')
  AND bookmaker_tier IS NULL;

-- Update exchange records
UPDATE prop_lines_history
SET bookmaker_tier = 'exchange',
    is_exchange = TRUE
WHERE bookmaker_key IN ('betfair', 'matchbook', 'smarkets')
  AND bookmaker_tier IS NULL;

---

-- Create view for best Pinnacle lines
CREATE OR REPLACE VIEW best_pinnacle_lines AS
SELECT
    player_id,
    player_name,
    player_position,
    player_team,
    prop_type,
    line_value,
    over_odds,
    under_odds,
    bookmaker_key,
    bookmaker_title,
    snapshot_at,
    over_implied_prob,
    under_implied_prob,
    book_hold,
    commence_time,
    game_id
FROM prop_lines_history
WHERE bookmaker_key = 'pinnacle'
  AND snapshot_at = (
      SELECT MAX(snapshot_at)
      FROM prop_lines_history
      WHERE bookmaker_key = 'pinnacle'
  )
ORDER BY commence_time, player_name, prop_type;

COMMENT ON VIEW best_pinnacle_lines IS
'Latest Pinnacle lines only. Pinnacle typically has the sharpest lines (lowest hold, 1.5-2%)';

---

-- Create view for comparing Pinnacle vs square books
CREATE OR REPLACE VIEW pinnacle_vs_square_comparison AS
WITH pinnacle_lines AS (
    SELECT
        player_id,
        prop_type,
        line_value as pinnacle_line,
        over_odds as pinnacle_over_odds,
        under_odds as pinnacle_under_odds,
        book_hold as pinnacle_hold,
        snapshot_at
    FROM best_pinnacle_lines
),
square_lines AS (
    SELECT
        player_id,
        prop_type,
        AVG(line_value) as square_avg_line,
        AVG(over_odds) as square_avg_over_odds,
        AVG(under_odds) as square_avg_under_odds,
        AVG(book_hold) as square_avg_hold
    FROM prop_lines_history
    WHERE bookmaker_tier = 'square'
      AND snapshot_at >= NOW() - INTERVAL '6 hours'
    GROUP BY player_id, prop_type
)
SELECT
    p.player_id,
    p.prop_type,
    p.pinnacle_line,
    s.square_avg_line,
    p.pinnacle_line - s.square_avg_line as line_diff,
    p.pinnacle_hold,
    s.square_avg_hold,
    s.square_avg_hold - p.pinnacle_hold as hold_savings,
    p.snapshot_at
FROM pinnacle_lines p
JOIN square_lines s ON p.player_id = s.player_id AND p.prop_type = s.prop_type;

COMMENT ON VIEW pinnacle_vs_square_comparison IS
'Compare Pinnacle lines vs average square book lines. Shows hold savings and line differences.';

---

-- Create materialized view for Pinnacle ROI analysis (refresh daily)
CREATE MATERIALIZED VIEW IF NOT EXISTS pinnacle_historical_performance AS
WITH pinnacle_bets AS (
    SELECT
        pt.player_id,
        pt.prop_type,
        pt.line_value,
        pt.bet_side,
        pt.odds,
        pt.actual_result,
        pt.outcome,
        pt.profit_loss,
        pt.bet_amount,
        pt.edge,
        pt.bookmaker_key,
        pt.settled_at
    FROM paper_trades pt
    WHERE pt.bookmaker_key = 'pinnacle'
      AND pt.outcome IN ('win', 'loss', 'push')
)
SELECT
    prop_type,
    COUNT(*) as total_bets,
    AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
    SUM(bet_amount) as total_wagered,
    SUM(profit_loss) as total_profit,
    CASE
        WHEN SUM(bet_amount) > 0 THEN SUM(profit_loss) / SUM(bet_amount)
        ELSE 0
    END as roi,
    AVG(edge) as avg_edge,
    MIN(settled_at) as first_bet,
    MAX(settled_at) as last_bet
FROM pinnacle_bets
GROUP BY prop_type;

CREATE INDEX IF NOT EXISTS idx_pinnacle_perf_roi
    ON pinnacle_historical_performance(roi DESC);

COMMENT ON MATERIALIZED VIEW pinnacle_historical_performance IS
'Historical performance of bets placed at Pinnacle. Refresh daily to track ROI improvement.';

---

-- Grant permissions
GRANT SELECT ON best_pinnacle_lines TO dro;
GRANT SELECT ON pinnacle_vs_square_comparison TO dro;
GRANT SELECT ON pinnacle_historical_performance TO dro;
