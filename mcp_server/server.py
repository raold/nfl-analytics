"""
NFL Analytics MCP Server

FastMCP server implementing Chapter 9's Claude AI integration framework.
Provides tools for database queries, ESPN news fetching, and future extensions.

Run with:
    uv run python -m mcp_server.server

Or as a module:
    python -m mcp_server.server
"""

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCP_SERVER_PORT, logger
from mcp_server.tools.database import get_recent_games, get_team_stats, query_database
from mcp_server.tools.database_write import (
    insert_model_metadata,
    insert_prediction_batch,
    update_model_status,
)
from mcp_server.tools.filesystem import get_file_info, list_files, read_file, read_json
from mcp_server.tools.github_ops import create_issue, create_pr, get_repo_stats, list_recent_issues
from mcp_server.tools.news_fetcher import fetch_espn_news, search_all_nfl_news

# Initialize FastMCP server
mcp = FastMCP("nfl-analytics")

logger.info("NFL Analytics MCP Server initializing...")


# ============================================================================
# Database Tools
# ============================================================================


@mcp.tool()
def query_db(sql: str, params: str | None = None) -> dict:
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
def get_team_recent_games(team: str, limit: int = 10, season: int | None = None) -> dict:
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
def get_team_statistics(team: str, season: int | None = None) -> dict:
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
# Database Write Tools (Model Registry & Predictions)
# ============================================================================


@mcp.tool()
def register_model(
    model_name: str,
    model_version: str,
    model_type: str,
    framework: str,
    description: str,
    performance_summary: dict,
    training_config: dict | None = None,
) -> dict:
    """
    Register a trained model in the model registry.

    Args:
        model_name: Model name (e.g., "bnn_passing_v3.1")
        model_version: Version string (e.g., "3.1.0")
        model_type: Type (e.g., "bayesian_neural_network")
        framework: Framework (e.g., "pymc", "brms", "xgboost")
        description: Human-readable description
        performance_summary: Dict with metrics (mae, calibration, etc.)
        training_config: Optional training configuration

    Returns:
        Success/error dictionary with model_id

    Example:
        register_model(
            "bnn_passing_v3.1", "3.1.0", "bayesian_neural_network",
            "pymc", "Calibrated BNN for passing yards",
            {"mae": 47.77, "calibration": 0.895}
        )
    """
    return insert_model_metadata(
        model_name,
        model_version,
        model_type,
        framework,
        description,
        performance_summary,
        training_config,
    )


@mcp.tool()
def update_model_registry_status(model_id: int, status: str, notes: str | None = None) -> dict:
    """
    Update model status in registry.

    Args:
        model_id: Model ID
        status: New status ('trained', 'validated', 'deployed', 'deprecated')
        notes: Optional notes

    Returns:
        Success/error dictionary

    Example:
        update_model_registry_status(42, "deployed", "Production deployment complete")
    """
    return update_model_status(model_id, status, notes)


@mcp.tool()
def save_predictions(model_id: int, game_week: str, predictions: list) -> dict:
    """
    Save a batch of predictions.

    Args:
        model_id: Model ID
        game_week: Week identifier (e.g., "2024-W09")
        predictions: List of prediction dicts

    Returns:
        Success/error dictionary

    Example:
        save_predictions(42, "2024-W09", [
            {"game_id": "2024_09_PHI_DAL", "player_id": "J.Hurts",
             "prop_type": "passing_yards", "prediction": 275.5}
        ])
    """
    return insert_prediction_batch(model_id, game_week, predictions)


# ============================================================================
# Filesystem Tools
# ============================================================================


@mcp.tool()
def list_directory(directory_type: str, relative_path: str = "", pattern: str = "*") -> dict:
    """
    List files in a directory.

    Args:
        directory_type: Directory type ('models', 'logs', 'data', 'cache', 'results')
        relative_path: Relative path within directory
        pattern: Glob pattern (e.g., "*.pkl", "*.log")

    Returns:
        Dictionary with file list

    Example:
        list_directory("models", "bayesian", "*.pkl")
    """
    return list_files(directory_type, relative_path, pattern)


@mcp.tool()
def read_text_file(directory_type: str, relative_path: str, max_lines: int | None = None) -> dict:
    """
    Read a text file.

    Args:
        directory_type: Directory type
        relative_path: File path within directory
        max_lines: Optional line limit

    Returns:
        Dictionary with file contents

    Example:
        read_text_file("logs", "training/bnn_passing_v3.1.log", max_lines=100)
    """
    return read_file(directory_type, relative_path, max_lines)


@mcp.tool()
def read_json_file(directory_type: str, relative_path: str) -> dict:
    """
    Read and parse a JSON file.

    Args:
        directory_type: Directory type
        relative_path: File path

    Returns:
        Dictionary with parsed JSON

    Example:
        read_json_file("models", "bayesian/bnn_passing_v3_metadata.json")
    """
    return read_json(directory_type, relative_path)


@mcp.tool()
def get_file_metadata(directory_type: str, relative_path: str) -> dict:
    """
    Get file metadata.

    Args:
        directory_type: Directory type
        relative_path: File path

    Returns:
        Dictionary with file info (size, modified time, etc.)

    Example:
        get_file_metadata("models", "bayesian/bnn_passing_v3.1.pkl")
    """
    return get_file_info(directory_type, relative_path)


# ============================================================================
# GitHub Tools
# ============================================================================


@mcp.tool()
def github_create_issue(
    title: str, body: str, labels: str | None = None, assignees: str | None = None
) -> dict:
    """
    Create a GitHub issue.

    Args:
        title: Issue title
        body: Issue description (markdown)
        labels: Comma-separated labels
        assignees: Comma-separated assignees

    Returns:
        Dictionary with issue URL

    Example:
        github_create_issue(
            "BNN calibration degradation",
            "Calibration dropped from 89.5% to 82%",
            "bug,high-priority"
        )
    """
    return create_issue(title, body, labels, assignees)


@mcp.tool()
def github_list_issues(limit: int = 10, state: str = "open", labels: str | None = None) -> dict:
    """
    List GitHub issues.

    Args:
        limit: Number of issues
        state: Issue state ('open', 'closed', 'all')
        labels: Filter by labels

    Returns:
        Dictionary with issues list

    Example:
        github_list_issues(limit=5, state="open", labels="bug")
    """
    return list_recent_issues(limit, state, labels)


@mcp.tool()
def github_create_pr(title: str, body: str, base: str = "main") -> dict:
    """
    Create a pull request from current branch.

    Args:
        title: PR title
        body: PR description (markdown)
        base: Base branch

    Returns:
        Dictionary with PR URL

    Example:
        github_create_pr(
            "feat: Add BNN v3.2 with improved calibration",
            "- Implements temperature scaling\\n- Improves calibration to 92%"
        )
    """
    return create_pr(title, body, base)


@mcp.tool()
def github_repo_stats() -> dict:
    """
    Get repository statistics.

    Returns:
        Dictionary with repo stats (stars, forks, issues)

    Example:
        github_repo_stats()
    """
    return get_repo_stats()


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
    logger.info("NFL Analytics MCP Server - Phase 1 Integration Complete")
    logger.info("Chapter 9: Claude AI Integration Framework")
    logger.info("=" * 70)
    logger.info(f"Server running on port {MCP_SERVER_PORT}")
    logger.info("")
    logger.info("Available tools (18 total):")
    logger.info("  Database (Read):")
    logger.info("    - query_db: Execute SQL queries")
    logger.info("    - get_team_recent_games: Get recent games")
    logger.info("    - get_team_statistics: Get team stats")
    logger.info("  Database (Write):")
    logger.info("    - register_model: Register model in registry")
    logger.info("    - update_model_registry_status: Update model status")
    logger.info("    - save_predictions: Save prediction batch")
    logger.info("  Filesystem:")
    logger.info("    - list_directory: List files in directory")
    logger.info("    - read_text_file: Read text file")
    logger.info("    - read_json_file: Read and parse JSON")
    logger.info("    - get_file_metadata: Get file info")
    logger.info("  GitHub:")
    logger.info("    - github_create_issue: Create issue")
    logger.info("    - github_list_issues: List issues")
    logger.info("    - github_create_pr: Create pull request")
    logger.info("    - github_repo_stats: Get repo statistics")
    logger.info("  News:")
    logger.info("    - get_team_news: Fetch ESPN news")
    logger.info("    - search_nfl_news: Search NFL news")
    logger.info("")
    logger.info("Example usage:")
    logger.info('  "Register BNN v3.2 model with calibration metrics"')
    logger.info('  "List all model artifacts in models/bayesian directory"')
    logger.info('  "Create GitHub issue for calibration degradation"')
    logger.info("=" * 70)

    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
