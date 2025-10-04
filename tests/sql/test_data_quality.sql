-- Data quality tests
-- Checks constraints, nulls, and data integrity

-- Test 1: Check for NULL game_ids
DO $$
DECLARE
    null_count INT;
BEGIN
    SELECT COUNT(*) INTO null_count FROM games WHERE game_id IS NULL;
    IF null_count > 0 THEN
        RAISE EXCEPTION 'Found % games with NULL game_id', null_count;
    END IF;
    RAISE NOTICE '✓ No NULL game_ids in games table';
END $$;

-- Test 2: Check for duplicate game_ids
DO $$
DECLARE
    dup_count INT;
BEGIN
    SELECT COUNT(*) INTO dup_count
    FROM (
        SELECT game_id, COUNT(*) as cnt
        FROM games
        GROUP BY game_id
        HAVING COUNT(*) > 1
    ) dups;
    
    IF dup_count > 0 THEN
        RAISE EXCEPTION 'Found % duplicate game_ids', dup_count;
    END IF;
    RAISE NOTICE '✓ No duplicate game_ids';
END $$;

-- Test 3: Check score consistency (home_score + away_score should be reasonable)
DO $$
DECLARE
    bad_scores INT;
BEGIN
    SELECT COUNT(*) INTO bad_scores
    FROM games
    WHERE home_score IS NOT NULL
    AND away_score IS NOT NULL
    AND (home_score < 0 OR away_score < 0 OR home_score + away_score > 150);
    
    IF bad_scores > 0 THEN
        RAISE WARNING 'Found % games with suspicious scores', bad_scores;
    ELSE
        RAISE NOTICE '✓ All scores are within reasonable bounds';
    END IF;
END $$;

-- Test 4: Check season values are reasonable (1999-2025)
DO $$
DECLARE
    bad_seasons INT;
BEGIN
    SELECT COUNT(*) INTO bad_seasons
    FROM games
    WHERE season < 1999 OR season > 2025;
    
    IF bad_seasons > 0 THEN
        RAISE EXCEPTION 'Found % games with invalid season', bad_seasons;
    END IF;
    RAISE NOTICE '✓ All seasons are in valid range (1999-2025)';
END $$;

-- Test 5: Check week values are reasonable (1-22)
DO $$
DECLARE
    bad_weeks INT;
BEGIN
    SELECT COUNT(*) INTO bad_weeks
    FROM games
    WHERE week < 1 OR week > 22;
    
    IF bad_weeks > 0 THEN
        RAISE WARNING 'Found % games with week outside 1-22', bad_weeks;
    ELSE
        RAISE NOTICE '✓ All week values are reasonable';
    END IF;
END $$;

-- Test 6: Check plays table foreign key integrity
DO $$
DECLARE
    orphan_plays INT;
BEGIN
    SELECT COUNT(*) INTO orphan_plays
    FROM plays p
    LEFT JOIN games g ON p.game_id = g.game_id
    WHERE g.game_id IS NULL;
    
    IF orphan_plays > 0 THEN
        RAISE EXCEPTION 'Found % plays with no matching game', orphan_plays;
    END IF;
    RAISE NOTICE '✓ All plays have matching games';
END $$;

-- Test 7: Check EPA values are reasonable (-10 to +10)
DO $$
DECLARE
    extreme_epa INT;
BEGIN
    SELECT COUNT(*) INTO extreme_epa
    FROM plays
    WHERE epa IS NOT NULL
    AND (epa < -10 OR epa > 10);
    
    IF extreme_epa > 0 THEN
        RAISE WARNING 'Found % plays with extreme EPA values', extreme_epa;
    ELSE
        RAISE NOTICE '✓ All EPA values are within reasonable bounds';
    END IF;
END $$;

-- Test 8: Check odds_history has no negative prices
DO $$
DECLARE
    negative_prices INT;
BEGIN
    SELECT COUNT(*) INTO negative_prices
    FROM odds_history
    WHERE outcome_price < 1.0;
    
    IF negative_prices > 0 THEN
        RAISE WARNING 'Found % odds with price < 1.0 (suspicious)', negative_prices;
    ELSE
        RAISE NOTICE '✓ All odds prices are >= 1.0';
    END IF;
END $$;

-- Test 9: Check for NULL bookmaker keys
DO $$
DECLARE
    null_bookmakers INT;
BEGIN
    SELECT COUNT(*) INTO null_bookmakers
    FROM odds_history
    WHERE bookmaker_key IS NULL;
    
    IF null_bookmakers > 0 THEN
        RAISE EXCEPTION 'Found % odds rows with NULL bookmaker_key', null_bookmakers;
    END IF;
    RAISE NOTICE '✓ No NULL bookmaker_keys';
END $$;

-- Test 10: Check snapshot_at is not in the future
DO $$
DECLARE
    future_snapshots INT;
BEGIN
    SELECT COUNT(*) INTO future_snapshots
    FROM odds_history
    WHERE snapshot_at > NOW() + INTERVAL '1 day';
    
    IF future_snapshots > 0 THEN
        RAISE WARNING 'Found % odds snapshots dated in the future', future_snapshots;
    ELSE
        RAISE NOTICE '✓ No future-dated snapshots';
    END IF;
END $$;

-- Test 11: Check mart.team_epa has reasonable values
DO $$
DECLARE
    row_count INT;
    extreme_epa INT;
BEGIN
    SELECT COUNT(*) INTO row_count FROM mart.team_epa;
    
    IF row_count = 0 THEN
        RAISE WARNING 'mart.team_epa is empty';
    ELSE
        SELECT COUNT(*) INTO extreme_epa
        FROM mart.team_epa
        WHERE total_epa < -1000 OR total_epa > 1000;
        
        IF extreme_epa > 0 THEN
            RAISE WARNING 'Found % teams with extreme total EPA', extreme_epa;
        ELSE
            RAISE NOTICE '✓ mart.team_epa values are reasonable';
        END IF;
    END IF;
END $$;

-- Summary: Data quality metrics
SELECT
    'games' AS table_name,
    COUNT(*) AS row_count,
    COUNT(*) FILTER (WHERE home_score IS NULL OR away_score IS NULL) AS missing_scores,
    COUNT(*) FILTER (WHERE spread_close IS NULL) AS missing_spreads
FROM games

UNION ALL

SELECT
    'plays',
    COUNT(*),
    COUNT(*) FILTER (WHERE epa IS NULL),
    COUNT(*) FILTER (WHERE wpa IS NULL)
FROM plays

UNION ALL

SELECT
    'odds_history',
    COUNT(*),
    COUNT(*) FILTER (WHERE outcome_price IS NULL),
    COUNT(*) FILTER (WHERE commence_time IS NULL)
FROM odds_history;

\echo '✅ Data quality checks complete'
