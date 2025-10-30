"""
ESPN news fetcher for NFL teams.
Provides real-time news and updates with caching and rate limiting.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

from mcp_server.config import (
    ESPN_API_BASE_URL,
    MAX_NEWS_ARTICLES,
    NEWS_CACHE_SECONDS,
    MCP_CACHE_DIR
)

logger = logging.getLogger(__name__)

# Team abbreviation mapping (NFL standard codes to full names)
TEAM_MAPPING = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LAC": "Chargers", "LAR": "Rams", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders"
}

# Reverse mapping (full names to abbreviations)
TEAM_REVERSE_MAPPING = {v.lower(): k for k, v in TEAM_MAPPING.items()}

# Rate limiting
_last_request_time = 0
_min_request_interval = 1  # seconds


def _normalize_team_name(team: str) -> str:
    """
    Normalize team name to a searchable format.

    Args:
        team: Team abbreviation (e.g., "PHI") or full name (e.g., "Eagles")

    Returns:
        Normalized team name for searching
    """
    team_upper = team.upper()

    # If it's an abbreviation, convert to full name
    if team_upper in TEAM_MAPPING:
        return TEAM_MAPPING[team_upper]

    # If it's already a full name (case-insensitive)
    team_lower = team.lower()
    for full_name in TEAM_MAPPING.values():
        if team_lower in full_name.lower() or full_name.lower() in team_lower:
            return full_name

    # Return as-is if no mapping found
    logger.warning(f"No mapping found for team '{team}', using as-is")
    return team


def _get_cache_path(team: str) -> Path:
    """Get cache file path for a team."""
    return MCP_CACHE_DIR / f"news_{team.lower()}.json"


def _load_cache(team: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load cached news if available and not expired.

    Args:
        team: Team name

    Returns:
        Cached news articles or None if expired/missing
    """
    cache_path = _get_cache_path(team)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        # Check if cache is expired
        cached_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cached_time > timedelta(seconds=NEWS_CACHE_SECONDS):
            logger.info(f"Cache expired for {team}")
            return None

        logger.info(f"Using cached news for {team}")
        return cache_data['articles']

    except Exception as e:
        logger.error(f"Error loading cache for {team}: {e}")
        return None


def _save_cache(team: str, articles: List[Dict[str, Any]]) -> None:
    """
    Save articles to cache.

    Args:
        team: Team name
        articles: List of article dictionaries
    """
    cache_path = _get_cache_path(team)

    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'articles': articles
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Cached {len(articles)} articles for {team}")

    except Exception as e:
        logger.error(f"Error saving cache for {team}: {e}")


def _rate_limit() -> None:
    """Enforce rate limiting between requests."""
    global _last_request_time

    time_since_last = time.time() - _last_request_time
    if time_since_last < _min_request_interval:
        sleep_time = _min_request_interval - time_since_last
        logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)

    _last_request_time = time.time()


def fetch_espn_news(team: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent ESPN news for an NFL team.

    Args:
        team: Team abbreviation (e.g., "PHI") or full name (e.g., "Eagles")
        limit: Maximum number of articles to return (default 10)

    Returns:
        List of article dictionaries with keys:
            - title: Article headline
            - url: Link to full article
            - published: Publication timestamp (if available)
            - description: Article summary (if available)

    Features:
        - Automatic team name normalization
        - 1-hour cache to reduce API calls
        - Rate limiting (1 request/second)
        - Fallback to HTML scraping if API fails

    Example:
        >>> articles = fetch_espn_news("PHI", limit=5)
        >>> for article in articles:
        ...     print(article['title'])
    """
    # Normalize team name
    team_normalized = _normalize_team_name(team)
    logger.info(f"Fetching news for {team_normalized} (limit: {limit})")

    # Check cache first
    cached = _load_cache(team_normalized)
    if cached:
        return cached[:min(limit, MAX_NEWS_ARTICLES)]

    # Enforce limit
    limit = min(limit, MAX_NEWS_ARTICLES)

    # Rate limiting
    _rate_limit()

    try:
        # Try ESPN main NFL news page with team filter
        url = f"https://www.espn.com/nfl/team/_/name/{_get_team_slug(team_normalized)}"
        logger.info(f"Fetching from: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')

        articles = []

        # Find news articles in the page
        # ESPN structure: look for article links in news sections
        for article_elem in soup.find_all(['article', 'div'], class_=lambda x: x and 'article' in x.lower() if x else False):
            if len(articles) >= limit:
                break

            # Extract title
            title_elem = article_elem.find(['h1', 'h2', 'h3', 'a'])
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)

            # Extract URL
            link_elem = article_elem.find('a', href=True)
            if not link_elem:
                continue

            url = link_elem['href']
            if not url.startswith('http'):
                url = f"https://www.espn.com{url}"

            # Extract description if available
            description_elem = article_elem.find('p')
            description = description_elem.get_text(strip=True) if description_elem else ""

            articles.append({
                'title': title,
                'url': url,
                'description': description,
                'published': None,  # ESPN doesn't always provide timestamps in listings
                'source': 'ESPN'
            })

        # If we didn't find articles, try the main NFL news page
        if not articles:
            logger.warning("No articles found on team page, trying main NFL news")
            articles = _fetch_nfl_news_generic(team_normalized, limit)

        logger.info(f"Found {len(articles)} articles for {team_normalized}")

        # Save to cache
        if articles:
            _save_cache(team_normalized, articles)

        return articles

    except Exception as e:
        error_msg = f"Error fetching ESPN news for {team_normalized}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Return empty list on error
        return []


def _get_team_slug(team_name: str) -> str:
    """
    Convert team name to ESPN URL slug.

    Args:
        team_name: Full team name (e.g., "Eagles")

    Returns:
        URL slug (e.g., "phi")
    """
    # Get abbreviation
    team_lower = team_name.lower()
    for abbr, full_name in TEAM_MAPPING.items():
        if full_name.lower() == team_lower:
            return abbr.lower()

    # Fallback: use first 3 letters
    return team_name[:3].lower()


def _fetch_nfl_news_generic(team_filter: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch general NFL news from ESPN main page.

    Args:
        team_filter: Optional team name to filter by
        limit: Maximum number of articles

    Returns:
        List of article dictionaries
    """
    try:
        url = "https://www.espn.com/nfl/news"
        logger.info(f"Fetching from main NFL news: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        articles = []

        # Find news articles
        for article_elem in soup.find_all('article', limit=limit * 2):  # Get extra to filter
            if len(articles) >= limit:
                break

            # Extract title
            title_elem = article_elem.find(['h1', 'h2'])
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)

            # Filter by team if specified
            if team_filter and team_filter.lower() not in title.lower():
                continue

            # Extract URL
            link_elem = article_elem.find('a', href=True)
            if not link_elem:
                continue

            url = link_elem['href']
            if not url.startswith('http'):
                url = f"https://www.espn.com{url}"

            articles.append({
                'title': title,
                'url': url,
                'description': '',
                'published': None,
                'source': 'ESPN'
            })

        return articles

    except Exception as e:
        logger.error(f"Error fetching generic NFL news: {e}")
        return []


def search_all_nfl_news(keywords: List[str], hours: int = 24) -> List[Dict[str, Any]]:
    """
    Search all NFL news for keywords.

    Args:
        keywords: List of keywords to search for
        hours: Look back this many hours (default 24)

    Returns:
        List of matching articles

    Note:
        This is a simple implementation that searches recent headlines.
        For production, consider using ESPN's search API or RSS feeds.
    """
    logger.info(f"Searching NFL news for keywords: {keywords}")

    try:
        # Fetch general NFL news
        articles = _fetch_nfl_news_generic(limit=MAX_NEWS_ARTICLES)

        # Filter by keywords
        filtered = []
        for article in articles:
            title_lower = article['title'].lower()
            desc_lower = article.get('description', '').lower()

            # Check if any keyword matches
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in title_lower or keyword_lower in desc_lower:
                    filtered.append(article)
                    break

        logger.info(f"Found {len(filtered)} articles matching keywords")
        return filtered

    except Exception as e:
        logger.error(f"Error searching NFL news: {e}")
        return []
