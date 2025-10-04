-- Schema validation tests
-- Run after migrations to verify structure

-- Test 1: Verify all expected tables exist
DO $$
DECLARE
    missing_tables TEXT[];
BEGIN
    SELECT ARRAY_AGG(table_name)
    INTO missing_tables
    FROM (VALUES 
        ('games'),
        ('plays'),
        ('odds_history'),
        ('weather'),
        ('injuries')
    ) AS expected(table_name)
    WHERE NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = expected.table_name
    );
    
    IF array_length(missing_tables, 1) > 0 THEN
        RAISE EXCEPTION 'Missing tables: %', missing_tables;
    END IF;
    
    RAISE NOTICE '✓ All expected tables exist';
END $$;

-- Test 2: Verify games table structure
DO $$
DECLARE
    expected_columns TEXT[] := ARRAY[
        'game_id', 'season', 'week', 'gameday', 'kickoff',
        'home_team', 'away_team', 'home_score', 'away_score',
        'spread_line', 'spread_close', 'total_line', 'total_close',
        'home_moneyline', 'away_moneyline', 'result'
    ];
    actual_columns TEXT[];
BEGIN
    SELECT ARRAY_AGG(column_name::TEXT ORDER BY ordinal_position)
    INTO actual_columns
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name = 'games';
    
    IF NOT expected_columns <@ actual_columns THEN
        RAISE EXCEPTION 'games table missing columns. Expected: %, Got: %',
            expected_columns, actual_columns;
    END IF;
    
    RAISE NOTICE '✓ games table has correct structure';
END $$;

-- Test 3: Verify plays table structure
DO $$
DECLARE
    required_columns TEXT[] := ARRAY[
        'game_id', 'play_id', 'posteam', 'defteam',
        'desc', 'epa', 'wpa', 'down', 'ydstogo', 'yardline_100', 'play_type'
    ];
    actual_columns TEXT[];
BEGIN
    SELECT ARRAY_AGG(column_name::TEXT ORDER BY ordinal_position)
    INTO actual_columns
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND table_name = 'plays';
    
    IF NOT required_columns <@ actual_columns THEN
        RAISE EXCEPTION 'plays table missing columns';
    END IF;
    
    RAISE NOTICE '✓ plays table has correct structure';
END $$;

-- Test 4: Verify odds_history table structure and constraints
DO $$
DECLARE
    pk_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_schema = 'public'
        AND table_name = 'odds_history'
        AND constraint_type = 'PRIMARY KEY'
    ) INTO pk_exists;
    
    IF NOT pk_exists THEN
        RAISE EXCEPTION 'odds_history missing PRIMARY KEY';
    END IF;
    
    RAISE NOTICE '✓ odds_history has primary key';
END $$;

-- Test 5: Verify TimescaleDB hypertable
DO $$
DECLARE
    is_hypertable BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'public'
        AND hypertable_name = 'odds_history'
    ) INTO is_hypertable;
    
    IF NOT is_hypertable THEN
        RAISE EXCEPTION 'odds_history is not a TimescaleDB hypertable';
    END IF;
    
    RAISE NOTICE '✓ odds_history is a TimescaleDB hypertable';
END $$;

-- Test 6: Verify mart schema exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.schemata
        WHERE schema_name = 'mart'
    ) THEN
        RAISE EXCEPTION 'mart schema does not exist';
    END IF;
    
    RAISE NOTICE '✓ mart schema exists';
END $$;

-- Test 7: Verify mart.team_epa table exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'mart'
        AND table_name = 'team_epa'
    ) THEN
        RAISE EXCEPTION 'mart.team_epa table does not exist';
    END IF;
    
    RAISE NOTICE '✓ mart.team_epa table exists';
END $$;

-- Test 8: Verify materialized view exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_matviews
        WHERE schemaname = 'mart'
        AND matviewname = 'game_summary'
    ) THEN
        RAISE EXCEPTION 'mart.game_summary materialized view does not exist';
    END IF;
    
    RAISE NOTICE '✓ mart.game_summary materialized view exists';
END $$;

-- Test 9: Verify indexes exist
DO $$
DECLARE
    expected_indexes TEXT[] := ARRAY[
        'idx_games_season_week',
        'idx_plays_game_id',
        'idx_plays_epa'
    ];
    missing_indexes TEXT[];
BEGIN
    SELECT ARRAY_AGG(index_name)
    INTO missing_indexes
    FROM unnest(expected_indexes) AS index_name
    WHERE NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
        AND indexname = index_name
    );
    
    IF array_length(missing_indexes, 1) > 0 THEN
        RAISE WARNING 'Missing indexes: %', missing_indexes;
    ELSE
        RAISE NOTICE '✓ All expected indexes exist';
    END IF;
END $$;

-- Summary
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname IN ('public', 'mart')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

\echo '✅ Schema validation complete'
