-- ============================================================================
-- COMPREHENSIVE DATABASE AUDIT SCRIPT
-- NFL Analytics Database - DBA Review
-- ============================================================================

-- 1. SCHEMA OVERVIEW
-- ----------------------------------------------------------------------------
SELECT '======== SCHEMA OVERVIEW ========' as section;

SELECT
    nspname as schema_name,
    COUNT(c.oid) as table_count
FROM pg_namespace n
LEFT JOIN pg_class c ON n.oid = c.relnamespace
    AND c.relkind = 'r'
WHERE nspname NOT IN ('pg_catalog', 'information_schema')
    AND nspname NOT LIKE '_timescaledb%'
    AND nspname != 'timescaledb_experimental'
    AND nspname != 'timescaledb_information'
GROUP BY nspname
ORDER BY nspname;

-- 2. ALL TABLES WITH RECORD COUNTS
-- ----------------------------------------------------------------------------
SELECT '======== TABLE INVENTORY ========' as section;

WITH table_sizes AS (
    SELECT
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
        n_live_tup as row_count
    FROM pg_stat_user_tables
    ORDER BY schemaname, tablename
)
SELECT * FROM table_sizes;

-- 3. PRIMARY KEY ANALYSIS
-- ----------------------------------------------------------------------------
SELECT '======== TABLES WITHOUT PRIMARY KEYS ========' as section;

SELECT
    t.schemaname,
    t.tablename
FROM pg_tables t
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_constraint c
    JOIN pg_namespace n ON n.oid = c.connamespace
    WHERE c.contype = 'p'
    AND c.conrelid = (t.schemaname||'.'||t.tablename)::regclass
)
AND t.schemaname IN ('public', 'mart', 'predictions', 'reference')
ORDER BY t.schemaname, t.tablename;

-- 4. FOREIGN KEY RELATIONSHIPS
-- ----------------------------------------------------------------------------
SELECT '======== FOREIGN KEY RELATIONSHIPS ========' as section;

SELECT
    tc.table_schema,
    tc.table_name,
    kcu.column_name,
    ccu.table_schema AS foreign_table_schema,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_schema, tc.table_name;

-- 5. PLAYER ID COLUMN ANALYSIS
-- ----------------------------------------------------------------------------
SELECT '======== PLAYER ID COLUMNS ACROSS TABLES ========' as section;

SELECT
    table_schema,
    table_name,
    column_name,
    data_type,
    character_maximum_length
FROM information_schema.columns
WHERE column_name IN ('player_id', 'gsis_id', 'pfr_id', 'pfr_player_id', 'otc_id', 'espn_id')
    AND table_schema IN ('public', 'mart', 'predictions', 'reference')
ORDER BY column_name, table_schema, table_name;

-- 6. TEAM ABBREVIATION ANALYSIS
-- ----------------------------------------------------------------------------
SELECT '======== TEAM COLUMNS ACROSS TABLES ========' as section;

SELECT
    table_schema,
    table_name,
    column_name,
    data_type,
    character_maximum_length
FROM information_schema.columns
WHERE column_name IN ('team', 'home_team', 'away_team', 'posteam', 'defteam',
                      'team_abbr', 'club_code', 'player_team', 'opponent')
    AND table_schema IN ('public', 'mart', 'predictions', 'reference')
ORDER BY column_name, table_schema, table_name;

-- 7. CHECK TEAM CODE CONSISTENCY
-- ----------------------------------------------------------------------------
SELECT '======== TEAM CODE VALUES IN GAMES TABLE ========' as section;

SELECT DISTINCT home_team FROM games
UNION
SELECT DISTINCT away_team FROM games
ORDER BY 1;

-- 8. CHECK TEAM CODE CONSISTENCY IN PLAYS
-- ----------------------------------------------------------------------------
SELECT '======== TEAM CODE VALUES IN PLAYS TABLE ========' as section;

SELECT DISTINCT posteam FROM plays WHERE posteam IS NOT NULL
UNION
SELECT DISTINCT defteam FROM plays WHERE defteam IS NOT NULL
ORDER BY 1;

-- 9. GAME ID FORMAT ANALYSIS
-- ----------------------------------------------------------------------------
SELECT '======== GAME ID FORMATS ========' as section;

SELECT
    table_name,
    COUNT(DISTINCT game_id) as unique_games,
    MIN(game_id) as sample_min,
    MAX(game_id) as sample_max
FROM (
    SELECT 'games' as table_name, game_id FROM games
    UNION ALL
    SELECT 'plays' as table_name, game_id FROM plays
    UNION ALL
    SELECT 'officials' as table_name, game_id FROM officials
) t
GROUP BY table_name;

-- 10. DUPLICATE KEY ANALYSIS
-- ----------------------------------------------------------------------------
SELECT '======== POTENTIAL DUPLICATE RECORDS ========' as section;

-- Check for duplicate games
SELECT
    'games' as table_name,
    game_id,
    COUNT(*) as duplicate_count
FROM games
GROUP BY game_id
HAVING COUNT(*) > 1
LIMIT 10;

-- 11. NULL VALUE ANALYSIS FOR KEY COLUMNS
-- ----------------------------------------------------------------------------
SELECT '======== NULL VALUES IN KEY COLUMNS ========' as section;

SELECT
    'games' as table_name,
    COUNT(*) FILTER (WHERE game_id IS NULL) as null_game_id,
    COUNT(*) FILTER (WHERE home_team IS NULL) as null_home_team,
    COUNT(*) FILTER (WHERE away_team IS NULL) as null_away_team,
    COUNT(*) as total_rows
FROM games
UNION ALL
SELECT
    'plays' as table_name,
    COUNT(*) FILTER (WHERE game_id IS NULL) as null_game_id,
    COUNT(*) FILTER (WHERE posteam IS NULL) as null_posteam,
    COUNT(*) FILTER (WHERE defteam IS NULL) as null_defteam,
    COUNT(*) as total_rows
FROM plays
UNION ALL
SELECT
    'players' as table_name,
    COUNT(*) FILTER (WHERE gsis_id IS NULL) as null_gsis_id,
    0 as null_col2,
    0 as null_col3,
    COUNT(*) as total_rows
FROM players;

-- 12. COLUMN NAME INCONSISTENCIES
-- ----------------------------------------------------------------------------
SELECT '======== SIMILAR COLUMN NAMES (POTENTIAL INCONSISTENCIES) ========' as section;

WITH column_pairs AS (
    SELECT
        c1.table_schema as schema1,
        c1.table_name as table1,
        c1.column_name as column1,
        c2.table_schema as schema2,
        c2.table_name as table2,
        c2.column_name as column2
    FROM information_schema.columns c1
    JOIN information_schema.columns c2
        ON c1.column_name != c2.column_name
        AND (
            -- Player ID variations
            (c1.column_name IN ('player_id', 'gsis_id', 'pfr_id', 'pfr_player_id')
             AND c2.column_name IN ('player_id', 'gsis_id', 'pfr_id', 'pfr_player_id'))
            OR
            -- Team variations
            (c1.column_name IN ('team', 'team_abbr', 'club_code')
             AND c2.column_name IN ('team', 'team_abbr', 'club_code'))
        )
    WHERE c1.table_schema IN ('public', 'mart', 'predictions')
        AND c2.table_schema IN ('public', 'mart', 'predictions')
        AND c1.table_name < c2.table_name
)
SELECT DISTINCT * FROM column_pairs
ORDER BY column1, table1, table2;

-- 13. INDEX ANALYSIS
-- ----------------------------------------------------------------------------
SELECT '======== INDEX COVERAGE ========' as section;

SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname IN ('public', 'mart', 'predictions', 'reference')
ORDER BY schemaname, tablename, indexname;

-- 14. CONSTRAINT ANALYSIS
-- ----------------------------------------------------------------------------
SELECT '======== CHECK CONSTRAINTS ========' as section;

SELECT
    n.nspname as schema_name,
    c.relname as table_name,
    con.conname as constraint_name,
    pg_get_constraintdef(con.oid) as constraint_definition
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE con.contype = 'c'
    AND n.nspname IN ('public', 'mart', 'predictions', 'reference')
ORDER BY n.nspname, c.relname;

-- 15. DATA TYPE INCONSISTENCIES
-- ----------------------------------------------------------------------------
SELECT '======== DATA TYPE INCONSISTENCIES FOR SAME LOGICAL COLUMNS ========' as section;

WITH column_types AS (
    SELECT
        column_name,
        data_type,
        character_maximum_length,
        numeric_precision,
        COUNT(*) as usage_count,
        STRING_AGG(table_schema || '.' || table_name, ', ') as tables
    FROM information_schema.columns
    WHERE table_schema IN ('public', 'mart', 'predictions', 'reference')
        AND column_name IN ('season', 'week', 'game_id', 'player_id', 'gsis_id',
                           'team', 'home_team', 'away_team')
    GROUP BY column_name, data_type, character_maximum_length, numeric_precision
)
SELECT * FROM column_types
WHERE column_name IN (
    SELECT column_name
    FROM column_types
    GROUP BY column_name
    HAVING COUNT(*) > 1
)
ORDER BY column_name, usage_count DESC;