"""
System & Metadata Queries

SQL queries for database metadata, health checks, and system statistics.

Author: Claude Code
Date: November 2025
"""

import pandas as pd


class SystemQueries:
    """Centralized queries for system and database metadata."""

    @staticmethod
    def database_statistics(conn) -> dict:
        """
        Fetch basic database statistics.

        Returns:
            Dict with keys:
                - total_games, total_plays
                - season_min, season_max
                - db_size (human-readable)
                - latest_game (most recent completed game timestamp)
        """
        stats = {}

        # Total games
        result = conn.execute("SELECT COUNT(*) FROM games").fetchone()
        stats['total_games'] = result[0] if result else 0

        # Total plays
        result = conn.execute("SELECT COUNT(*) FROM plays").fetchone()
        stats['total_plays'] = result[0] if result else 0

        # Season coverage
        result = conn.execute("SELECT MIN(season), MAX(season) FROM games").fetchone()
        stats['season_min'] = result[0] if result and result[0] else 0
        stats['season_max'] = result[1] if result and result[1] else 0

        # Database size
        result = conn.execute("""
            SELECT pg_size_pretty(pg_database_size('devdb01'))
        """).fetchone()
        stats['db_size'] = result[0] if result else 'N/A'

        # Latest completed game
        result = conn.execute("""
            SELECT MAX(kickoff) FROM games
            WHERE kickoff IS NOT NULL AND kickoff < NOW()
        """).fetchone()
        stats['latest_game'] = result[0] if result and result[0] else None

        return stats

    @staticmethod
    def schema_statistics(conn) -> pd.DataFrame:
        """
        Get table count by schema.

        Returns:
            DataFrame with columns:
                - schemaname, table_count
        """
        query = """
        SELECT
            schemaname,
            COUNT(*) as table_count
        FROM pg_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        GROUP BY schemaname
        ORDER BY table_count DESC
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def recent_data_ingestion(
        conn,
        min_season: int = 2024
    ) -> pd.DataFrame:
        """
        Get recent data ingestion history by season and week.

        Args:
            conn: Database connection
            min_season: Minimum season to include (default: 2024)

        Returns:
            DataFrame with columns:
                - season, week, game_type, games (count)
                - latest_kickoff (most recent game in that group)
        """
        query = f"""
        SELECT
            season,
            week,
            game_type,
            COUNT(*) as games,
            MAX(kickoff) as latest_kickoff
        FROM games
        WHERE season >= {min_season}
        GROUP BY season, week, game_type
        ORDER BY season DESC, week DESC
        LIMIT 10
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def table_row_counts(conn) -> pd.DataFrame:
        """
        Get row counts for all major tables.

        Returns:
            DataFrame with columns:
                - schema, table_name, row_count
        """
        query = """
        SELECT
            schemaname as schema,
            tablename as table_name,
            n_live_tup as row_count
        FROM pg_stat_user_tables
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY schemaname, tablename
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def column_statistics(
        conn,
        schema: str = 'public'
    ) -> pd.DataFrame:
        """
        Get column metadata for all tables in a schema.

        Args:
            conn: Database connection
            schema: Schema name (default: 'public')

        Returns:
            DataFrame with columns:
                - table_name, column_name, data_type, is_nullable
        """
        query = """
        SELECT
            table_name,
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position
        """

        return pd.read_sql_query(query, conn, params=(schema,))

    @staticmethod
    def prediction_coverage(conn) -> dict:
        """
        Get coverage statistics for ml.predictions table.

        Returns:
            Dict with keys:
                - total_predictions, predictions_with_actuals
                - coverage_pct, avg_calibration
        """
        query = """
        SELECT
            COUNT(*) as total_predictions,
            COUNT(actual_value) as predictions_with_actuals,
            ROUND(COUNT(actual_value)::numeric / COUNT(*)::numeric * 100, 1) as coverage_pct,
            ROUND(AVG(CASE WHEN calibrated THEN 1.0 ELSE 0.0 END)::numeric * 100, 1) as avg_calibration
        FROM ml.predictions
        WHERE stat_type = 'passing_yards'
        """

        result = pd.read_sql_query(query, conn)

        if not result.empty:
            row = result.iloc[0]
            return {
                'total_predictions': int(row['total_predictions']),
                'predictions_with_actuals': int(row['predictions_with_actuals']),
                'coverage_pct': float(row['coverage_pct']),
                'avg_calibration': float(row['avg_calibration']),
            }

        return {}

    @staticmethod
    def model_coverage(conn) -> pd.DataFrame:
        """
        Get model coverage statistics from ml.model_registry.

        Returns:
            DataFrame with columns:
                - status, count, avg_age_days
        """
        query = """
        SELECT
            status,
            COUNT(*) as count,
            ROUND(AVG(EXTRACT(DAY FROM NOW() - updated_at))::numeric, 1) as avg_age_days
        FROM ml.model_registry
        GROUP BY status
        ORDER BY count DESC
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def slow_queries_log(
        conn,
        min_duration_ms: int = 1000
    ) -> pd.DataFrame:
        """
        Get slow queries from pg_stat_statements (requires extension enabled).

        Args:
            conn: Database connection
            min_duration_ms: Minimum query duration in milliseconds

        Returns:
            DataFrame with columns:
                - query, calls, mean_time_ms, total_time_ms
        """
        query = f"""
        SELECT
            LEFT(query, 100) as query,
            calls,
            ROUND(mean_exec_time::numeric, 2) as mean_time_ms,
            ROUND(total_exec_time::numeric, 2) as total_time_ms
        FROM pg_stat_statements
        WHERE mean_exec_time > {min_duration_ms}
        ORDER BY total_exec_time DESC
        LIMIT 20
        """

        try:
            return pd.read_sql_query(query, conn)
        except Exception:
            # pg_stat_statements might not be enabled
            return pd.DataFrame()

    @staticmethod
    def index_statistics(conn) -> pd.DataFrame:
        """
        Get statistics on database indexes.

        Returns:
            DataFrame with columns:
                - table_name, index_name, idx_scan, idx_tup_read, idx_tup_fetch
        """
        query = """
        SELECT
            t.tablename as table_name,
            i.indexname as index_name,
            ps.idx_scan,
            ps.idx_tup_read,
            ps.idx_tup_fetch
        FROM pg_tables t
        JOIN pg_indexes i ON t.tablename = i.tablename
        LEFT JOIN pg_stat_user_indexes ps ON i.indexname = ps.indexrelname
        WHERE t.schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY ps.idx_scan DESC NULLS LAST
        LIMIT 30
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def last_vacuum_analyze(conn) -> pd.DataFrame:
        """
        Get last vacuum and analyze times for tables.

        Returns:
            DataFrame with columns:
                - table_name, last_vacuum, last_autovacuum, last_analyze
        """
        query = """
        SELECT
            schemaname || '.' || relname as table_name,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY schemaname, relname
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def connection_stats(conn) -> dict:
        """
        Get current database connection statistics.

        Returns:
            Dict with keys:
                - active_connections, idle_connections, total_connections
                - max_connections
        """
        query = """
        SELECT
            COUNT(*) FILTER (WHERE state = 'active') as active_connections,
            COUNT(*) FILTER (WHERE state = 'idle') as idle_connections,
            COUNT(*) as total_connections,
            (SELECT setting FROM pg_settings WHERE name = 'max_connections') as max_connections
        FROM pg_stat_activity
        """

        result = pd.read_sql_query(query, conn)

        if not result.empty:
            row = result.iloc[0]
            return {
                'active_connections': int(row['active_connections']),
                'idle_connections': int(row['idle_connections']),
                'total_connections': int(row['total_connections']),
                'max_connections': int(row['max_connections']),
            }

        return {}

    @staticmethod
    def materialized_view_status(conn) -> pd.DataFrame:
        """
        Get status of materialized views.

        Returns:
            DataFrame with columns:
                - view_name, row_count, is_populated
        """
        query = """
        SELECT
            schemaname || '.' || matviewname as view_name,
            COALESCE(n_live_tup, 0) as row_count,
            CASE WHEN ispopulated THEN 'Yes' ELSE 'No' END as is_populated
        FROM pg_matviews
        LEFT JOIN pg_stat_user_tables ON pg_matviews.matviewname = pg_stat_user_tables.relname
        ORDER BY schemaname, matviewname
        """

        return pd.read_sql_query(query, conn)
