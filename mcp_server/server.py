"""
NFL Analytics MCP Server

FastMCP server implementing Chapter 9's Claude AI integration framework.
Provides tools for database queries, ESPN news fetching, and future extensions.

Run with:
    uv run python -m mcp_server.server

Or as a module:
    python -m mcp_server.server
"""

import logging
from typing import Optional
from mcp.server.fastmcp import FastMCP

from mcp_server.tools.database import (
    query_database,
    get_recent_games,
    get_team_stats,
    close_connection_pool
)
from mcp_server.tools.news_fetcher import (
    fetch_espn_news,
    search_all_nfl_news
)
from mcp_server.config import MCP_SERVER_PORT, logger

# Initialize FastMCP server
mcp = FastMCP("nfl-analytics")

logger.info("NFL Analytics MCP Server initializing...")


# ============================================================================
# Database Tools
# ============================================================================

@mcp.tool()
def query_db(sql: str, params: Optional[str] = None) -> dict:
    """
    Execute a SQL query against the NFL analytics database.

    Args:
        sql: SQL query string (use %s for parameterization)
        params: Optional comma-separated parameters for parameterized queries

    Returns:
        Dictionary with:
            - success: bool
            - rows: list of dictionaries (if successful)
            - row_count: number of rows returned
            - error: error message (if failed)
            - truncated: whether results were truncated to MAX_QUERY_ROWS

    Safety features:
        - Parameterized queries to prevent SQL injection
        - Query timeout enforcement (30 seconds)
        - Row limit enforcement (10,000 rows)
        - Connection pooling for efficiency

    Example:
        query_db("SELECT * FROM games WHERE home_team = %s LIMIT 5", "PHI")
    """
    # Parse params if provided
    parsed_params = None
    if params:
        parsed_params = tuple(p.strip() for p in params.split(","))

    return query_database(sql, parsed_params)


@mcp.tool()
def get_team_recent_games(
    team: str,
    limit: int = 10,
    season: Optional[int] = None
) -> dict:
    """
    Get recent games for an NFL team.

    Args:
        team: Team abbreviation (e.g., "PHI", "KC")
        limit: Number of games to return (default 10)
        season: Season year (optional)

    Returns:
        Query result dictionary with game data

    Example:
        get_team_recent_games("PHI", limit=5, season=2024)
    """
    return get_recent_games(team, limit, season)


@mcp.tool()
def get_team_statistics(team: str, season: Optional[int] = None) -> dict:
    """
    Get aggregated statistics for an NFL team.

    Args:
        team: Team abbreviation (e.g., "PHI")
        season: Season year (optional)

    Returns:
        Dictionary with:
            - games_played: Total games
            - wins: Total wins
            - avg_points_for: Average points scored
            - avg_points_against: Average points allowed
            - avg_epa: Average EPA per game
            - avg_pass_epa: Average passing EPA
            - avg_rush_epa: Average rushing EPA

    Example:
        get_team_statistics("PHI", season=2024)
    """
    return get_team_stats(team, season)


# ============================================================================
# News Tools
# ============================================================================

@mcp.tool()
def get_team_news(team: str, limit: int = 10) -> list:
    """
    Fetch recent ESPN news for an NFL team.

    Args:
        team: Team abbreviation (e.g., "PHI", "KC") or full name (e.g., "Eagles")
        limit: Maximum number of articles to return (default 10)

    Returns:
        List of article dictionaries with:
            - title: Article headline
            - url: Link to full article
            - published: Publication timestamp (if available)
            - description: Article summary (if available)
            - source: "ESPN"

    Features:
        - Automatic team name normalization
        - 1-hour cache to reduce API calls
        - Rate limiting (1 request/second)

    Example:
        get_team_news("Eagles", limit=5)
    """
    return fetch_espn_news(team, limit)


@mcp.tool()
def search_nfl_news(keywords: str, hours: int = 24) -> list:
    """
    Search all NFL news for keywords.

    Args:
        keywords: Comma-separated keywords to search for
        hours: Look back this many hours (default 24)

    Returns:
        List of matching articles

    Example:
        search_nfl_news("injury,quarterback", hours=48)
    """
    keyword_list = [k.strip() for k in keywords.split(",")]
    return search_all_nfl_news(keyword_list, hours)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the MCP server."""
    logger.info("=" * 70)
    logger.info("NFL Analytics MCP Server")
    logger.info("Chapter 9: Claude AI Integration Framework")
    logger.info("=" * 70)
    logger.info(f"Server running on port {MCP_SERVER_PORT}")
    logger.info("")
    logger.info("Available tools:")
    logger.info("  Database:")
    logger.info("    - query_db: Execute SQL queries")
    logger.info("    - get_team_recent_games: Get recent games for a team")
    logger.info("    - get_team_statistics: Get team statistics")
    logger.info("  News:")
    logger.info("    - get_team_news: Fetch ESPN news for a team")
    logger.info("    - search_nfl_news: Search NFL news by keywords")
    logger.info("")
    logger.info("Example usage:")
    logger.info('  "Show me recent Eagles news and their last 5 games EPA"')
    logger.info("=" * 70)

    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
