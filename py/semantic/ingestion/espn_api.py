"""
ESPN JSON API Client for Semantic Data Ingestion.

This module provides a robust client for ESPN's undocumented JSON APIs,
which are more reliable than HTML scraping (no 403 errors).

Endpoints used:
- News: https://site.api.espn.com/apis/site/v2/sports/football/nfl/news
- Injuries: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries

Note: These are undocumented APIs. Use responsibly with rate limiting.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ESPNCategory(Enum):
    """News categories from ESPN API."""
    INJURY = "injury"
    LINEUP = "lineup"
    COACHING = "coaching"
    TRADE = "trade"
    GENERAL = "general"


@dataclass
class ESPNNewsItem:
    """Structured news item from ESPN API."""
    id: str
    title: str
    description: str
    url: str
    published_at: datetime
    source: str = "espn_api"
    teams: list[str] = field(default_factory=list)
    athletes: list[str] = field(default_factory=list)
    category: ESPNCategory = ESPNCategory.GENERAL
    raw_json: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA256 hash of title + description for deduplication."""
        content = f"{self.title}|{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ESPNInjuryItem:
    """Structured injury item from ESPN API."""
    id: str
    athlete_id: str
    athlete_name: str
    team_id: str
    team_abbrev: str
    status: str  # Questionable, Doubtful, Out, IR
    injury_type: str  # Groin, Knee, etc.
    injury_location: str
    updated_at: datetime
    fantasy_status: str
    source: str = "espn_api"
    raw_json: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA256 hash for deduplication."""
        content = f"{self.athlete_id}|{self.status}|{self.injury_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ESPN Team ID to abbreviation mapping
ESPN_TEAM_IDS = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE",
    6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
    11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA",
    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF",
    26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
    33: "BAL", 34: "HOU"
}

# Reverse mapping
ESPN_ABBREV_TO_ID = {v: k for k, v in ESPN_TEAM_IDS.items()}


class ESPNApiClient:
    """
    Client for ESPN's JSON APIs.

    Uses undocumented but stable JSON endpoints that don't suffer
    from the 403 errors that plague HTML scraping.
    """

    # API endpoints
    NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
    TEAM_INJURIES_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    INJURY_DETAIL_URL = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/athletes/{athlete_id}/injuries/{injury_id}"
    ATHLETE_URL = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/athletes/{athlete_id}"

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize ESPN API client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set headers to look like a browser
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        })

        logger.info("ESPNApiClient initialized (timeout=%.1fs, retries=%d)", timeout, max_retries)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict | None = None) -> dict:
        """Make a rate-limited GET request."""
        self._rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error("ESPN API request failed: %s - %s", url, e)
            raise

    def fetch_news(self, limit: int = 50) -> list[ESPNNewsItem]:
        """
        Fetch NFL news from ESPN API.

        Args:
            limit: Maximum number of articles to fetch (max 50 per request)

        Returns:
            List of ESPNNewsItem objects
        """
        logger.info("Fetching ESPN news (limit=%d)", limit)

        data = self._get(self.NEWS_URL, params={"limit": min(limit, 50)})

        items = []
        for article in data.get("articles", []):
            try:
                item = self._parse_news_article(article)
                items.append(item)
            except Exception as e:
                logger.warning("Failed to parse article %s: %s", article.get("id"), e)

        logger.info("Fetched %d news items from ESPN", len(items))
        return items

    def _parse_news_article(self, article: dict) -> ESPNNewsItem:
        """Parse a news article from the API response."""
        # Extract teams and athletes from categories
        teams = []
        athletes = []
        category = ESPNCategory.GENERAL

        for cat in article.get("categories", []):
            cat_type = cat.get("type", "")

            if cat_type == "team":
                team_id = cat.get("teamId")
                if team_id and team_id in ESPN_TEAM_IDS:
                    teams.append(ESPN_TEAM_IDS[team_id])

            elif cat_type == "athlete":
                athlete_desc = cat.get("description", "")
                if athlete_desc:
                    athletes.append(athlete_desc)

            elif cat_type == "topic":
                topic = cat.get("description", "").lower()
                if "injury" in topic:
                    category = ESPNCategory.INJURY
                elif "trade" in topic:
                    category = ESPNCategory.TRADE

        # Infer category from headline if not set
        headline = article.get("headline", "").lower()
        if category == ESPNCategory.GENERAL:
            if any(kw in headline for kw in ["injury", "injured", "hurt", "out", "questionable"]):
                category = ESPNCategory.INJURY
            elif any(kw in headline for kw in ["trade", "traded", "acquire"]):
                category = ESPNCategory.TRADE
            elif any(kw in headline for kw in ["coach", "hire", "fire"]):
                category = ESPNCategory.COACHING
            elif any(kw in headline for kw in ["lineup", "start", "bench"]):
                category = ESPNCategory.LINEUP

        # Parse timestamp
        published_str = article.get("published", "")
        try:
            published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            published_at = datetime.utcnow()

        # Build URL
        links = article.get("links", {})
        url = ""
        if "web" in links:
            url = links["web"].get("href", "")

        return ESPNNewsItem(
            id=str(article.get("id", "")),
            title=article.get("headline", ""),
            description=article.get("description", ""),
            url=url,
            published_at=published_at,
            teams=teams,
            athletes=athletes,
            category=category,
            raw_json=article,
        )

    def fetch_team_injuries(
        self,
        team: str,
        season: int = 2025,
        expand_details: bool = True,
    ) -> list[ESPNInjuryItem]:
        """
        Fetch injuries for a specific team.

        Args:
            team: Team abbreviation (e.g., "KC", "BUF")
            season: NFL season year
            expand_details: Whether to fetch full injury details (slower but richer)

        Returns:
            List of ESPNInjuryItem objects
        """
        team_id = ESPN_ABBREV_TO_ID.get(team.upper())
        if not team_id:
            logger.warning("Unknown team abbreviation: %s", team)
            return []

        logger.info("Fetching injuries for %s (team_id=%d)", team, team_id)

        url = self.TEAM_INJURIES_URL.format(team_id=team_id)
        data = self._get(url, params={"limit": 100})

        items = []
        for injury_ref in data.get("items", []):
            try:
                ref_url = injury_ref.get("$ref", "")
                if not ref_url:
                    continue

                if expand_details:
                    injury_data = self._get(ref_url)
                    item = self._parse_injury(injury_data, team)
                    if item:
                        items.append(item)
                else:
                    # Extract IDs from ref URL for minimal data
                    # URL format: .../athletes/{athlete_id}/injuries/{injury_id}
                    parts = ref_url.split("/")
                    if len(parts) >= 2:
                        athlete_id = parts[-3] if "athletes" in parts else ""
                        injury_id = parts[-1].split("?")[0]
                        items.append(ESPNInjuryItem(
                            id=injury_id,
                            athlete_id=athlete_id,
                            athlete_name="",
                            team_id=str(team_id),
                            team_abbrev=team,
                            status="Unknown",
                            injury_type="Unknown",
                            injury_location="",
                            updated_at=datetime.utcnow(),
                            fantasy_status="",
                        ))
            except Exception as e:
                logger.warning("Failed to parse injury: %s", e)

        logger.info("Fetched %d injuries for %s", len(items), team)
        return items

    def _parse_injury(self, data: dict, team: str) -> ESPNInjuryItem | None:
        """Parse an injury from the detailed API response."""
        try:
            # Extract injury details
            details = data.get("details", {})
            injury_type = details.get("type", "Unknown")
            injury_location = details.get("location", "")
            fantasy_status = details.get("fantasyStatus", {}).get("description", "")

            # Parse timestamp
            date_str = data.get("date", "")
            try:
                updated_at = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                updated_at = datetime.utcnow()

            # Get athlete info from ref
            athlete_ref = data.get("athlete", {}).get("$ref", "")
            athlete_id = ""
            if "/athletes/" in athlete_ref:
                parts = athlete_ref.split("/athletes/")
                if len(parts) > 1:
                    athlete_id = parts[1].split("/")[0].split("?")[0]

            # Try to get athlete name (would require another API call)
            # For now, leave blank - can be enriched later
            athlete_name = ""

            return ESPNInjuryItem(
                id=str(data.get("id", "")),
                athlete_id=athlete_id,
                athlete_name=athlete_name,
                team_id=str(ESPN_ABBREV_TO_ID.get(team, "")),
                team_abbrev=team,
                status=data.get("status", "Unknown"),
                injury_type=injury_type,
                injury_location=injury_location,
                updated_at=updated_at,
                fantasy_status=fantasy_status,
                raw_json=data,
            )
        except Exception as e:
            logger.error("Failed to parse injury data: %s", e)
            return None

    def fetch_all_teams_injuries(
        self,
        teams: list[str] | None = None,
        season: int = 2025,
    ) -> list[ESPNInjuryItem]:
        """
        Fetch injuries for all teams (or specified subset).

        Args:
            teams: List of team abbreviations, or None for all 32 teams
            season: NFL season year

        Returns:
            List of all ESPNInjuryItem objects
        """
        if teams is None:
            teams = list(ESPN_TEAM_IDS.values())

        logger.info("Fetching injuries for %d teams", len(teams))

        all_injuries = []
        for team in teams:
            try:
                injuries = self.fetch_team_injuries(team, season, expand_details=True)
                all_injuries.extend(injuries)
            except Exception as e:
                logger.error("Failed to fetch injuries for %s: %s", team, e)

        logger.info("Fetched %d total injuries across %d teams", len(all_injuries), len(teams))
        return all_injuries

    def get_health_status(self) -> dict[str, Any]:
        """Check if ESPN API is accessible."""
        try:
            start = time.time()
            data = self._get(self.NEWS_URL, params={"limit": 1})
            latency = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "articles_available": len(data.get("articles", [])) > 0,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Convenience function for quick testing
def test_espn_api():
    """Quick test of ESPN API endpoints."""
    client = ESPNApiClient()

    print("\n=== ESPN API Health Check ===")
    health = client.get_health_status()
    print(f"Status: {health}")

    print("\n=== News Fetch ===")
    news = client.fetch_news(limit=5)
    for item in news[:3]:
        print(f"  [{item.category.value}] {item.title[:60]}...")
        print(f"    Teams: {item.teams}, Published: {item.published_at}")

    print("\n=== Injuries Fetch (KC) ===")
    injuries = client.fetch_team_injuries("KC")
    for injury in injuries[:3]:
        print(f"  {injury.athlete_id}: {injury.status} ({injury.injury_type})")

    print(f"\n=== Summary ===")
    print(f"News items: {len(news)}")
    print(f"KC injuries: {len(injuries)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_espn_api()
