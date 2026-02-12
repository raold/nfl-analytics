-- ============================================================================
-- Odds API Monitoring Dashboard Queries
-- ============================================================================
--
-- Collection of SQL queries for monitoring the odds fetching service.
-- These queries are designed for use with Grafana dashboards and can be
-- executed directly against PostgreSQL or via Prometheus postgres_exporter.
--
-- Author: Claude Code (Sonnet 4.5)
-- Date: November 8, 2025
-- Version: 1.0
-- ============================================================================

-- ============================================================================
-- 1. API USAGE METRICS
-- ============================================================================

-- Current month API usage summary
-- Returns: Current API calls made, quota limit, and percentage used
SELECT
    api_name,
    calls_made,
    quota_limit,
    ROUND(100.0 * calls_made / NULLIF(quota_limit, 0), 2) as pct_used,
    quota_limit - calls_made as calls_remaining,
    last_call_at,
    EXTRACT(EPOCH FROM (NOW() - last_call_at)) / 3600 as hours_since_last_call
FROM api_usage_tracker
WHERE month = DATE_TRUNC('month', CURRENT_DATE)
    AND api_name = 'the-odds-api';

-- API usage by month (historical trend)
-- Returns: Monthly API call trends
SELECT
    month,
    api_name,
    calls_made,
    quota_limit,
    ROUND(100.0 * calls_made / NULLIF(quota_limit, 0), 2) as pct_used,
    total_cost
FROM api_usage_tracker
WHERE api_name = 'the-odds-api'
ORDER BY month DESC
LIMIT 12;

-- API usage rate (calls per day)
-- Returns: Average API calls per day this month
SELECT
    api_name,
    calls_made as total_calls,
    EXTRACT(DAY FROM (NOW() - month)) + 1 as days_elapsed,
    ROUND(calls_made / (EXTRACT(DAY FROM (NOW() - month)) + 1), 2) as calls_per_day,
    quota_limit as monthly_quota,
    ROUND(quota_limit / 30.0, 2) as sustainable_daily_rate
FROM api_usage_tracker
WHERE month = DATE_TRUNC('month', CURRENT_DATE)
    AND api_name = 'the-odds-api';


-- ============================================================================
-- 2. ODDS SNAPSHOT FREQUENCY
-- ============================================================================

-- Odds snapshot frequency by game (upcoming games)
-- Returns: How often odds are being updated for each game
SELECT
    g.game_id,
    g.season,
    g.week,
    g.home_team || ' vs ' || g.away_team as matchup,
    g.kickoff,
    EXTRACT(EPOCH FROM (g.kickoff - NOW())) / 3600 as hours_until_game,
    COUNT(DISTINCT o.snapshot_at) as total_snapshots,
    COUNT(DISTINCT DATE(o.snapshot_at)) as days_with_snapshots,
    MAX(o.snapshot_at) as latest_snapshot,
    EXTRACT(EPOCH FROM (NOW() - MAX(o.snapshot_at))) / 3600 as hours_since_latest,
    COUNT(DISTINCT o.bookmaker_key) as bookmakers_tracked
FROM games g
LEFT JOIN odds_history o ON g.odds_api_event_id = o.event_id
WHERE g.kickoff > NOW()
    AND g.kickoff < NOW() + INTERVAL '7 days'
GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team, g.kickoff
ORDER BY g.kickoff;

-- Snapshot staleness alert (games with no recent odds)
-- Returns: Games that haven't had odds updated recently
SELECT
    g.game_id,
    g.home_team || ' vs ' || g.away_team as matchup,
    g.kickoff,
    EXTRACT(EPOCH FROM (g.kickoff - NOW())) / 3600 as hours_until_game,
    MAX(o.snapshot_at) as latest_snapshot,
    EXTRACT(EPOCH FROM (NOW() - MAX(o.snapshot_at))) / 3600 as hours_since_update,
    CASE
        WHEN EXTRACT(EPOCH FROM (g.kickoff - NOW())) / 3600 < 24 AND
             EXTRACT(EPOCH FROM (NOW() - MAX(o.snapshot_at))) / 3600 > 1
            THEN 'CRITICAL'
        WHEN EXTRACT(EPOCH FROM (g.kickoff - NOW())) / 3600 < 72 AND
             EXTRACT(EPOCH FROM (NOW() - MAX(o.snapshot_at))) / 3600 > 6
            THEN 'WARNING'
        ELSE 'OK'
    END as status
FROM games g
LEFT JOIN odds_history o ON g.odds_api_event_id = o.event_id
WHERE g.kickoff > NOW()
    AND g.kickoff < NOW() + INTERVAL '7 days'
GROUP BY g.game_id, g.home_team, g.away_team, g.kickoff
HAVING MAX(o.snapshot_at) < NOW() - INTERVAL '2 hours'
ORDER BY hours_since_update DESC;


-- ============================================================================
-- 3. BOOKMAKER COVERAGE
-- ============================================================================

-- Bookmaker availability by market
-- Returns: Which bookmakers are providing odds for each market
SELECT
    market_key,
    bookmaker_key,
    COUNT(DISTINCT event_id) as games_covered,
    COUNT(*) as total_odds,
    MIN(snapshot_at) as first_seen,
    MAX(snapshot_at) as last_seen,
    EXTRACT(EPOCH FROM (NOW() - MAX(snapshot_at))) / 3600 as hours_since_latest
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '7 days'
GROUP BY market_key, bookmaker_key
ORDER BY market_key, games_covered DESC;

-- Bookmaker reliability score
-- Returns: How consistently each bookmaker provides odds
WITH bookmaker_stats AS (
    SELECT
        bookmaker_key,
        COUNT(DISTINCT event_id) as events_covered,
        COUNT(DISTINCT DATE(snapshot_at)) as days_active,
        COUNT(*) as total_odds,
        MIN(snapshot_at) as first_odds,
        MAX(snapshot_at) as latest_odds
    FROM odds_history
    WHERE snapshot_at > NOW() - INTERVAL '30 days'
    GROUP BY bookmaker_key
),
all_events AS (
    SELECT COUNT(DISTINCT event_id) as total_events
    FROM odds_history
    WHERE snapshot_at > NOW() - INTERVAL '30 days'
)
SELECT
    bs.bookmaker_key,
    bs.events_covered,
    ae.total_events,
    ROUND(100.0 * bs.events_covered / ae.total_events, 1) as coverage_pct,
    bs.days_active,
    bs.total_odds,
    ROUND(bs.total_odds / bs.days_active, 0) as avg_odds_per_day,
    EXTRACT(EPOCH FROM (NOW() - bs.latest_odds)) / 3600 as hours_since_latest,
    CASE
        WHEN ROUND(100.0 * bs.events_covered / ae.total_events, 1) >= 90 THEN 'A'
        WHEN ROUND(100.0 * bs.events_covered / ae.total_events, 1) >= 75 THEN 'B'
        WHEN ROUND(100.0 * bs.events_covered / ae.total_events, 1) >= 50 THEN 'C'
        ELSE 'D'
    END as reliability_grade
FROM bookmaker_stats bs
CROSS JOIN all_events ae
ORDER BY coverage_pct DESC;


-- ============================================================================
-- 4. LINE MOVEMENT DETECTION
-- ============================================================================

-- Recent significant line movements
-- Returns: Props with line movements > 5% in last 24 hours
WITH line_movements AS (
    SELECT
        plh.event_id,
        plh.player_name,
        plh.prop_type,
        plh.bookmaker_key,
        plh.line_value,
        plh.snapshot_at,
        LAG(plh.line_value) OVER (
            PARTITION BY plh.event_id, plh.player_name, plh.prop_type, plh.bookmaker_key
            ORDER BY plh.snapshot_at
        ) as prev_line_value,
        LAG(plh.snapshot_at) OVER (
            PARTITION BY plh.event_id, plh.player_name, plh.prop_type, plh.bookmaker_key
            ORDER BY plh.snapshot_at
        ) as prev_snapshot_at
    FROM prop_lines_history plh
    WHERE plh.snapshot_at > NOW() - INTERVAL '24 hours'
)
SELECT
    lm.event_id,
    g.home_team || ' vs ' || g.away_team as matchup,
    g.kickoff,
    lm.player_name,
    lm.prop_type,
    lm.bookmaker_key,
    lm.prev_line_value as old_line,
    lm.line_value as new_line,
    lm.line_value - lm.prev_line_value as line_move,
    ROUND(100.0 * (lm.line_value - lm.prev_line_value) / NULLIF(lm.prev_line_value, 0), 2) as pct_change,
    lm.prev_snapshot_at as move_time,
    EXTRACT(EPOCH FROM (lm.snapshot_at - lm.prev_snapshot_at)) / 3600 as hours_between
FROM line_movements lm
JOIN games g ON lm.event_id = g.odds_api_event_id
WHERE lm.prev_line_value IS NOT NULL
    AND lm.prev_line_value != 0
    AND ABS(100.0 * (lm.line_value - lm.prev_line_value) / lm.prev_line_value) >= 5.0
ORDER BY ABS(pct_change) DESC, lm.snapshot_at DESC
LIMIT 50;

-- Market-wide line movement summary
-- Returns: Average line movements by prop type
SELECT
    prop_type,
    COUNT(*) as total_props,
    ROUND(AVG(ABS(line_move_since_open)), 2) as avg_line_move,
    ROUND(STDDEV(line_move_since_open), 2) as std_dev_line_move,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(line_move_since_open)), 2) as median_line_move,
    ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ABS(line_move_since_open)), 2) as p90_line_move
FROM prop_lines_history
WHERE snapshot_at > NOW() - INTERVAL '7 days'
    AND line_move_since_open IS NOT NULL
GROUP BY prop_type
ORDER BY avg_line_move DESC;


-- ============================================================================
-- 5. BOOKMAKER HOLD/VIG ANALYSIS
-- ============================================================================

-- Average book hold by bookmaker
-- Returns: How much juice/vig each bookmaker charges
SELECT
    bookmaker_key,
    prop_type,
    COUNT(*) as total_props,
    ROUND(AVG(book_hold) * 100, 2) as avg_hold_pct,
    ROUND(MIN(book_hold) * 100, 2) as min_hold_pct,
    ROUND(MAX(book_hold) * 100, 2) as max_hold_pct,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY book_hold) * 100, 2) as median_hold_pct
FROM prop_lines_history
WHERE snapshot_at > NOW() - INTERVAL '7 days'
    AND book_hold IS NOT NULL
    AND book_hold BETWEEN 0 AND 0.20  -- Filter outliers
GROUP BY bookmaker_key, prop_type
ORDER BY bookmaker_key, prop_type;

-- Low hold opportunities (best value bets)
-- Returns: Props with lowest bookmaker hold (best odds)
SELECT
    plh.event_id,
    g.home_team || ' vs ' || g.away_team as matchup,
    g.kickoff,
    plh.player_name,
    plh.prop_type,
    plh.bookmaker_key,
    plh.line_value,
    plh.over_odds,
    plh.under_odds,
    ROUND(plh.book_hold * 100, 2) as hold_pct,
    plh.snapshot_at,
    EXTRACT(EPOCH FROM (NOW() - plh.snapshot_at)) / 3600 as hours_ago
FROM prop_lines_history plh
JOIN games g ON plh.event_id = g.odds_api_event_id
WHERE plh.snapshot_at > NOW() - INTERVAL '24 hours'
    AND g.kickoff > NOW()
    AND plh.book_hold IS NOT NULL
    AND plh.book_hold < 0.03  -- Less than 3% hold
ORDER BY plh.book_hold, g.kickoff
LIMIT 100;


-- ============================================================================
-- 6. ODDS COVERAGE SUMMARY
-- ============================================================================

-- Overall odds coverage health check
-- Returns: Summary of odds data quality
SELECT
    'Total Games Next 7 Days' as metric,
    COUNT(DISTINCT g.game_id) as value,
    NULL as percentage
FROM games g
WHERE g.kickoff > NOW() AND g.kickoff < NOW() + INTERVAL '7 days'

UNION ALL

SELECT
    'Games with Odds Data' as metric,
    COUNT(DISTINCT g.game_id) as value,
    ROUND(100.0 * COUNT(DISTINCT g.game_id) / (
        SELECT COUNT(*) FROM games
        WHERE kickoff > NOW() AND kickoff < NOW() + INTERVAL '7 days'
    ), 1) as percentage
FROM games g
JOIN odds_history o ON g.odds_api_event_id = o.event_id
WHERE g.kickoff > NOW() AND g.kickoff < NOW() + INTERVAL '7 days'

UNION ALL

SELECT
    'Total Odds Snapshots (24h)' as metric,
    COUNT(*) as value,
    NULL as percentage
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '24 hours'

UNION ALL

SELECT
    'Unique Events Tracked' as metric,
    COUNT(DISTINCT event_id) as value,
    NULL as percentage
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '7 days'

UNION ALL

SELECT
    'Active Bookmakers' as metric,
    COUNT(DISTINCT bookmaker_key) as value,
    NULL as percentage
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '24 hours';


-- ============================================================================
-- 7. PROP LINES COVERAGE
-- ============================================================================

-- Player prop coverage by position
-- Returns: How many player props available by position
SELECT
    plh.player_position,
    plh.prop_type,
    COUNT(DISTINCT plh.player_id) as unique_players,
    COUNT(DISTINCT plh.event_id) as unique_games,
    COUNT(DISTINCT plh.bookmaker_key) as bookmakers,
    COUNT(*) as total_props,
    MAX(plh.snapshot_at) as latest_snapshot
FROM prop_lines_history plh
WHERE plh.snapshot_at > NOW() - INTERVAL '24 hours'
GROUP BY plh.player_position, plh.prop_type
ORDER BY plh.player_position, total_props DESC;

-- Top players by prop line volume
-- Returns: Players with most betting action (most lines available)
SELECT
    plh.player_name,
    plh.player_position,
    plh.player_team,
    COUNT(DISTINCT plh.prop_type) as prop_types,
    COUNT(DISTINCT plh.bookmaker_key) as bookmakers,
    COUNT(*) as total_lines,
    MAX(plh.snapshot_at) as latest_snapshot
FROM prop_lines_history plh
WHERE plh.snapshot_at > NOW() - INTERVAL '24 hours'
GROUP BY plh.player_name, plh.player_position, plh.player_team
ORDER BY total_lines DESC
LIMIT 50;


-- ============================================================================
-- 8. PROMETHEUS METRICS (For postgres_exporter)
-- ============================================================================

-- These queries are designed to be scraped by prometheus postgres_exporter
-- Configure in prometheus/queries.yaml

-- odds_api_calls_total
SELECT
    api_name as api,
    calls_made as value
FROM api_usage_tracker
WHERE month = DATE_TRUNC('month', CURRENT_DATE);

-- odds_api_quota_remaining
SELECT
    api_name as api,
    quota_limit - calls_made as value
FROM api_usage_tracker
WHERE month = DATE_TRUNC('month', CURRENT_DATE);

-- odds_snapshot_freshness_hours
SELECT
    g.game_id,
    g.home_team || '_' || g.away_team as game,
    EXTRACT(EPOCH FROM (NOW() - MAX(o.snapshot_at))) / 3600 as value
FROM games g
LEFT JOIN odds_history o ON g.odds_api_event_id = o.event_id
WHERE g.kickoff > NOW() AND g.kickoff < NOW() + INTERVAL '7 days'
GROUP BY g.game_id, g.home_team, g.away_team;

-- odds_bookmaker_coverage_pct
SELECT
    bookmaker_key as bookmaker,
    ROUND(100.0 * COUNT(DISTINCT event_id) / (
        SELECT COUNT(DISTINCT event_id)
        FROM odds_history
        WHERE snapshot_at > NOW() - INTERVAL '24 hours'
    ), 2) as value
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '24 hours'
GROUP BY bookmaker_key;

-- odds_line_movements_total
SELECT
    prop_type,
    COUNT(*) as value
FROM prop_lines_history
WHERE snapshot_at > NOW() - INTERVAL '1 hour'
    AND line_move_since_open IS NOT NULL
    AND ABS(line_move_since_open) >= 5.0
GROUP BY prop_type;


-- ============================================================================
-- 9. ALERTING QUERIES
-- ============================================================================

-- Critical: API quota nearly exhausted (>90%)
SELECT
    'API_QUOTA_CRITICAL' as alert_type,
    api_name,
    calls_made,
    quota_limit,
    ROUND(100.0 * calls_made / quota_limit, 1) as pct_used,
    'API quota usage above 90%' as message
FROM api_usage_tracker
WHERE month = DATE_TRUNC('month', CURRENT_DATE)
    AND api_name = 'the-odds-api'
    AND calls_made > 0.90 * quota_limit;

-- Warning: Stale odds for upcoming games
SELECT
    'STALE_ODDS_WARNING' as alert_type,
    g.game_id,
    g.home_team || ' vs ' || g.away_team as matchup,
    g.kickoff,
    MAX(o.snapshot_at) as last_update,
    EXTRACT(EPOCH FROM (NOW() - MAX(o.snapshot_at))) / 3600 as hours_stale,
    'Odds not updated recently for upcoming game' as message
FROM games g
LEFT JOIN odds_history o ON g.odds_api_event_id = o.event_id
WHERE g.kickoff > NOW()
    AND g.kickoff < NOW() + INTERVAL '48 hours'
GROUP BY g.game_id, g.home_team, g.away_team, g.kickoff
HAVING MAX(o.snapshot_at) < NOW() - INTERVAL '2 hours';

-- Warning: Missing bookmaker data
SELECT
    'MISSING_BOOKMAKER_WARNING' as alert_type,
    bookmaker_key,
    MAX(snapshot_at) as last_seen,
    EXTRACT(EPOCH FROM (NOW() - MAX(snapshot_at))) / 3600 as hours_since_last,
    'Bookmaker has not provided odds recently' as message
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '7 days'
GROUP BY bookmaker_key
HAVING MAX(snapshot_at) < NOW() - INTERVAL '24 hours';


-- ============================================================================
-- 10. DATA QUALITY CHECKS
-- ============================================================================

-- Check for duplicate odds records
SELECT
    event_id,
    bookmaker_key,
    market_key,
    outcome_name,
    snapshot_at,
    COUNT(*) as duplicate_count
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '24 hours'
GROUP BY event_id, bookmaker_key, market_key, outcome_name, snapshot_at
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;

-- Check for invalid odds values
SELECT
    'Invalid Odds Values' as issue,
    COUNT(*) as occurrences
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '24 hours'
    AND (
        outcome_price IS NULL
        OR outcome_price = 0
        OR ABS(outcome_price) > 10000  -- Extremely unlikely odds
    )

UNION ALL

SELECT
    'Missing Point Values' as issue,
    COUNT(*) as occurrences
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '24 hours'
    AND market_key IN ('spreads', 'totals')
    AND outcome_point IS NULL;

-- Check for unmatched events (events in odds but not in games table)
SELECT
    'Unmatched Events' as issue,
    COUNT(DISTINCT event_id) as count,
    STRING_AGG(DISTINCT event_id, ', ') as event_ids
FROM odds_history
WHERE snapshot_at > NOW() - INTERVAL '24 hours'
    AND NOT EXISTS (
        SELECT 1 FROM games g WHERE g.odds_api_event_id = odds_history.event_id
    );


-- ============================================================================
-- USAGE NOTES
-- ============================================================================

-- To use these queries with Grafana:
-- 1. Create a PostgreSQL data source in Grafana pointing to your database
-- 2. Create panels using these queries as the SQL query
-- 3. Use variables like $__timeFilter for time range filtering
-- 4. Set up alerts based on threshold values

-- To use with Prometheus postgres_exporter:
-- 1. Add queries to prometheus/postgres_exporter_queries.yaml
-- 2. Configure postgres_exporter to use the queries file
-- 3. Set up Prometheus scraping of postgres_exporter endpoint
-- 4. Create alerting rules in prometheus/alert_rules.yml

-- Example Grafana time series query:
-- SELECT
--     snapshot_at as time,
--     bookmaker_key as metric,
--     COUNT(*) as value
-- FROM odds_history
-- WHERE $__timeFilter(snapshot_at)
-- GROUP BY snapshot_at, bookmaker_key
-- ORDER BY snapshot_at;
