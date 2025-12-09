"""
Database query tool for NFL analytics PostgreSQL database.
Provides safe SQL execution with parameterized queries, timeouts, and row limits.

Redis Phase 2 Enhancement:
- Added Redis-backed caching for query results (5 min TTL)
- Cache key format: mcp:database_query:query_database:{args_hash}
- Graceful fallback to uncached execution if Redis unavailable
"""

import logging
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from mcp_server.config import DATABASE_URL, MAX_QUERY_ROWS, MAX_QUERY_TIME_SECONDS

try:
    from mcp_server.tools.cache_wrapper import cache_database_query

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Connection pool (lazy initialization)
_connection_pool: pool.SimpleConnectionPool | None = None


def _get_connection_pool() -> pool.SimpleConnectionPool:
    """Get or create the connection pool."""
    global _connection_pool

    if _connection_pool is None:
        logger.info("Initializing database connection pool")
        _connection_pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
        logger.info("Connection pool initialized")

    return _connection_pool


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn_pool = _get_connection_pool()
    conn = conn_pool.getconn()

    try:
        yield conn
    finally:
        conn_pool.putconn(conn)


def _query_database_uncached(sql: str, params: tuple | None = None) -> dict[str, Any]:
    """
    Execute SQL query without caching (internal implementation).

    See query_database() for public API documentation.
    """
    try:
        logger.info(f"Executing query: {sql[:100]}...")

        with get_db_connection() as conn:
            # Set statement timeout
            with conn.cursor() as timeout_cursor:
                timeout_cursor.execute(f"SET statement_timeout = {MAX_QUERY_TIME_SECONDS * 1000}")

            # Execute query with RealDictCursor for dict results
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)

                # Fetch results with row limit
                rows = cursor.fetchmany(MAX_QUERY_ROWS + 1)

                # Check if truncated
                truncated = len(rows) > MAX_QUERY_ROWS
                if truncated:
                    rows = rows[:MAX_QUERY_ROWS]
                    logger.warning(
                        f"Query returned more than {MAX_QUERY_ROWS} rows, " f"truncated to limit"
                    )

                # Convert to list of dicts
                result_rows = [dict(row) for row in rows]

                logger.info(f"Query successful: {len(result_rows)} rows returned")

                return {
                    "success": True,
                    "rows": result_rows,
                    "row_count": len(result_rows),
                    "truncated": truncated,
                }

    except psycopg2.errors.QueryCanceled:
        error_msg = f"Query exceeded timeout of {MAX_QUERY_TIME_SECONDS} seconds"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "rows": [], "row_count": 0}

    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg, "rows": [], "row_count": 0}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg, "rows": [], "row_count": 0}


def get_recent_games(
    team: str | None = None, limit: int = 10, season: int | None = None
) -> dict[str, Any]:
    """
    Convenience function to get recent games.

    Args:
        team: Team abbreviation (e.g., "PHI", "KC") - optional
        limit: Number of games to return (default 10)
        season: Season year (optional)

    Returns:
        Query result dictionary
    """
    conditions = []
    params = []

    if team:
        conditions.append("(home_team = %s OR away_team = %s)")
        params.extend([team, team])

    if season:
        conditions.append("season = %s")
        params.append(season)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    sql = f"""
        SELECT
            game_id,
            season,
            week,
            game_date,
            home_team,
            away_team,
            home_score,
            away_score,
            total_home_epa,
            total_away_epa
        FROM games
        WHERE {where_clause}
        ORDER BY game_date DESC
        LIMIT %s
    """
    params.append(limit)

    return query_database(sql, tuple(params))


def get_team_stats(team: str, season: int | None = None) -> dict[str, Any]:
    """
    Get aggregated statistics for a team.

    Args:
        team: Team abbreviation (e.g., "PHI")
        season: Season year (optional)

    Returns:
        Query result dictionary with team statistics
    """
    season_filter = f"AND season = {season}" if season else ""

    sql = f"""
        SELECT
            team,
            COUNT(*) as games_played,
            SUM(CASE WHEN won THEN 1 ELSE 0 END) as wins,
            AVG(points_for) as avg_points_for,
            AVG(points_against) as avg_points_against,
            AVG(total_epa) as avg_epa,
            AVG(pass_epa) as avg_pass_epa,
            AVG(rush_epa) as avg_rush_epa
        FROM (
            SELECT
                home_team as team,
                CASE WHEN home_score > away_score THEN true ELSE false END as won,
                home_score as points_for,
                away_score as points_against,
                total_home_epa as total_epa,
                pass_epa as pass_epa,
                rush_epa as rush_epa,
                season
            FROM games
            WHERE home_team = %s {season_filter}

            UNION ALL

            SELECT
                away_team as team,
                CASE WHEN away_score > home_score THEN true ELSE false END as won,
                away_score as points_for,
                home_score as points_against,
                total_away_epa as total_epa,
                pass_epa as pass_epa,
                rush_epa as rush_epa,
                season
            FROM games
            WHERE away_team = %s {season_filter}
        ) team_games
        GROUP BY team
    """

    return query_database(sql, (team, team))


def close_connection_pool():
    """Close all connections in the pool (call on shutdown)."""
    global _connection_pool

    if _connection_pool is not None:
        logger.info("Closing database connection pool")
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Connection pool closed")


# Public API with Redis caching
if CACHE_AVAILABLE:
    # Wrap with Redis cache (5 min TTL)
    query_database = cache_database_query(ttl=300)(_query_database_uncached)
    logger.info("Database query caching enabled (Redis, 5 min TTL)")
else:
    # No caching available - use uncached version directly
    query_database = _query_database_uncached
    logger.info("Database query caching disabled (Redis unavailable)")
