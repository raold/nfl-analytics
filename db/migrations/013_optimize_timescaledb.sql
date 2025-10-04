-- Migration 013: Optimize TimescaleDB Performance (Simplified)
-- Purpose: Improve compression and query performance for odds_history hypertable
-- Author: System
-- Date: 2025-01-04

-- Note: Running these outside a transaction for TimescaleDB operations

-- 1. Add BRIN index for time-range queries
-- BRIN indexes are perfect for append-only time-series data
CREATE INDEX IF NOT EXISTS idx_odds_history_snapshot_brin
ON odds_history USING BRIN(snapshot_at)
WITH (pages_per_range = 32);

-- 2. Add partial indexes for common query patterns
-- Index for recent odds - most queries target recent data
CREATE INDEX IF NOT EXISTS idx_odds_history_recent
ON odds_history(event_id, market_key, bookmaker_key, snapshot_at)
WHERE snapshot_at > '2024-01-01'::timestamptz;

-- Index for specific markets that are frequently queried
CREATE INDEX IF NOT EXISTS idx_odds_history_spreads
ON odds_history(event_id, bookmaker_key, snapshot_at)
WHERE market_key = 'spreads';

CREATE INDEX IF NOT EXISTS idx_odds_history_totals
ON odds_history(event_id, bookmaker_key, snapshot_at)
WHERE market_key = 'totals';

-- 3. Optimize chunk size for our workload
-- Default is 7 days, but for odds data 1 day chunks work better
SELECT set_chunk_time_interval('odds_history', INTERVAL '1 day');

-- 4. Add statistics for better query planning
ANALYZE odds_history;

-- 5. Create helper view for latest odds (commonly needed)
CREATE OR REPLACE VIEW latest_odds AS
WITH latest_snapshots AS (
    SELECT
        event_id,
        market_key,
        bookmaker_key,
        MAX(snapshot_at) as latest_snapshot
    FROM odds_history
    WHERE snapshot_at > CURRENT_DATE - INTERVAL '1 day'
    GROUP BY event_id, market_key, bookmaker_key
)
SELECT oh.*
FROM odds_history oh
INNER JOIN latest_snapshots ls
    ON oh.event_id = ls.event_id
    AND oh.market_key = ls.market_key
    AND oh.bookmaker_key = ls.bookmaker_key
    AND oh.snapshot_at = ls.latest_snapshot;

-- 6. Create index to support materialized view refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_game_features_refresh
ON mart.game_features_enhanced(game_id);

-- This allows REFRESH MATERIALIZED VIEW CONCURRENTLY
-- which doesn't lock the view during refresh

-- Report on improvements
DO $$
DECLARE
    chunk_count INTEGER;
    compressed_count INTEGER;
    total_size TEXT;
BEGIN
    SELECT COUNT(*) INTO chunk_count
    FROM timescaledb_information.chunks
    WHERE hypertable_name = 'odds_history';

    SELECT COUNT(*) INTO compressed_count
    FROM timescaledb_information.chunks
    WHERE hypertable_name = 'odds_history' AND is_compressed = true;

    SELECT pg_size_pretty(pg_total_relation_size('odds_history')) INTO total_size;

    RAISE NOTICE '';
    RAISE NOTICE '🚀 TimescaleDB Optimization Complete!';
    RAISE NOTICE '=====================================';
    RAISE NOTICE 'Hypertable: odds_history';
    RAISE NOTICE 'Chunks: % total, % compressed', chunk_count, compressed_count;
    RAISE NOTICE 'Total size: %', total_size;
    RAISE NOTICE '';
    RAISE NOTICE 'New features added:';
    RAISE NOTICE '  ✓ BRIN index for time-range queries';
    RAISE NOTICE '  ✓ Partial indexes for hot queries';
    RAISE NOTICE '  ✓ Optimized chunk interval (1 day)';
    RAISE NOTICE '  ✓ Helper view for latest odds';
    RAISE NOTICE '  ✓ Concurrent refresh support for marts';
    RAISE NOTICE '';
    RAISE NOTICE 'Note: Compression will occur automatically';
    RAISE NOTICE 'for chunks older than 7 days per existing policy';
    RAISE NOTICE '';
END $$;