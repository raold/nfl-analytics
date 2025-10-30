"""
NFL Analytics MCP Server Tools

Tools for database queries, news fetching, and future extensions.
"""

from .database import query_database
from .news_fetcher import fetch_espn_news, search_all_nfl_news

__all__ = [
    "query_database",
    "fetch_espn_news",
    "search_all_nfl_news",
]
