"""
RotoWire News Client using Crawl4AI.

Since RotoWire RSS feeds are discontinued, we use Crawl4AI to scrape
their fantasy football news page in an LLM-friendly format.

Crawl4AI provides:
- Headless browser rendering (handles JavaScript)
- Markdown conversion (LLM-ready output)
- Anti-detection measures (stealth mode)

Endpoints:
- News: https://www.rotowire.com/football/news.php
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RotoWireNewsItem:
    """Structured news item from RotoWire."""

    id: str
    title: str
    description: str
    url: str
    published_at: datetime
    source: str = "rotowire"
    teams: list[str] = field(default_factory=list)
    players: list[str] = field(default_factory=list)
    category: str = "general"
    raw_html: str = ""

    @property
    def content_hash(self) -> str:
        """SHA256 hash of title + description for deduplication."""
        content = f"{self.title}|{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# NFL team abbreviation patterns for extraction
NFL_TEAM_PATTERNS = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Raiders": "LV",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "49ers": "SF",
    "Seahawks": "SEA",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
    # Also match abbreviations directly
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAX",
    "KC": "KC",
    "LV": "LV",
    "LAC": "LAC",
    "LAR": "LAR",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NO": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SF": "SF",
    "SEA": "SEA",
    "TB": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
}

# Injury-related keywords
INJURY_KEYWORDS = [
    "injury",
    "injured",
    "hurt",
    "out",
    "questionable",
    "doubtful",
    "ir",
    "inactive",
    "dnp",
    "limited",
    "full practice",
    "concussion",
    "ankle",
    "knee",
    "hamstring",
    "shoulder",
    "back",
    "illness",
]


class RotoWireClient:
    """
    Client for scraping RotoWire fantasy football news.

    Uses Crawl4AI for reliable, JavaScript-rendered scraping with
    anti-detection features.
    """

    NEWS_URL = "https://www.rotowire.com/football/news.php"

    def __init__(
        self,
        timeout: float = 60.0,
        rate_limit_delay: float = 2.0,
    ):
        """
        Initialize RotoWire client.

        Args:
            timeout: Page load timeout in seconds
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._crawler = None
        logger.info("RotoWireClient initialized (timeout=%.1fs)", timeout)

    async def _get_crawler(self):
        """Lazy-initialize async crawler."""
        if self._crawler is None:
            from crawl4ai import AsyncWebCrawler

            self._crawler = AsyncWebCrawler(verbose=False)
        return self._crawler

    async def fetch_news(self, limit: int = 50) -> list[RotoWireNewsItem]:
        """
        Fetch NFL news from RotoWire.

        Args:
            limit: Maximum number of articles to return

        Returns:
            List of RotoWireNewsItem objects
        """
        logger.info("Fetching RotoWire news (limit=%d)", limit)

        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

            # Define extraction schema for RotoWire news items
            schema = {
                "name": "RotoWire News",
                "baseSelector": ".news-update",
                "fields": [
                    {"name": "player", "selector": ".news-update__player-link", "type": "text"},
                    {"name": "headline", "selector": ".news-update__headline", "type": "text"},
                    {"name": "team", "selector": ".news-update__team a", "type": "text"},
                    {"name": "position", "selector": ".news-update__pos", "type": "text"},
                    {"name": "injury", "selector": ".news-update__inj", "type": "text"},
                    {"name": "timestamp", "selector": ".news-update__timestamp", "type": "text"},
                    {"name": "analysis", "selector": ".news-update__analysis", "type": "text"},
                ],
            }

            extraction_strategy = JsonCssExtractionStrategy(schema)

            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=self.NEWS_URL,
                    config=CrawlerRunConfig(
                        extraction_strategy=extraction_strategy,
                    ),
                )

            if not result.success:
                logger.error("Crawl failed: %s", result.error_message)
                return []

            # Parse the extracted JSON
            items = self._parse_extracted_json(result.extracted_content, limit)
            logger.info("Fetched %d news items from RotoWire", len(items))
            return items

        except Exception as e:
            logger.error("RotoWire fetch failed: %s", e)
            return []

    def _parse_extracted_json(self, json_content: str, limit: int) -> list[RotoWireNewsItem]:
        """Parse JSON extracted by CSS strategy."""
        import json

        try:
            data = json.loads(json_content) if json_content else []
        except json.JSONDecodeError:
            logger.warning("Failed to parse extracted JSON")
            return []

        items = []
        for entry in data[:limit]:
            try:
                player = entry.get("player", "").strip()
                headline = entry.get("headline", "").strip()
                analysis = entry.get("analysis", "").strip()
                team = entry.get("team", "").strip()
                injury = entry.get("injury", "").strip()
                position = entry.get("position", "").strip()

                if not headline or len(headline) < 5:
                    continue

                # Build title from player and headline
                title = f"{player}: {headline}" if player else headline

                # Extract teams from text and explicit team field
                teams = []
                if team:
                    abbrev = self._team_name_to_abbrev(team)
                    if abbrev:
                        teams.append(abbrev)
                teams.extend(self._extract_teams(headline + " " + analysis))

                # Determine category
                category = "injury" if injury else self._infer_category(title + " " + analysis)

                # Generate ID from content
                content = f"{title}|{analysis}"
                content_id = hashlib.sha256(content.encode()).hexdigest()[:12]

                items.append(
                    RotoWireNewsItem(
                        id=content_id,
                        title=title,
                        description=analysis[:500] if analysis else headline,
                        url=self.NEWS_URL,
                        published_at=datetime.utcnow(),
                        teams=list(set(teams)),
                        players=[player] if player else [],
                        category=category,
                    )
                )
            except Exception as e:
                logger.debug("Failed to parse entry: %s", e)

        return items

    def _team_name_to_abbrev(self, team_name: str) -> str | None:
        """Convert full team name to abbreviation."""
        name_map = {
            "Arizona Cardinals": "ARI",
            "Atlanta Falcons": "ATL",
            "Baltimore Ravens": "BAL",
            "Buffalo Bills": "BUF",
            "Carolina Panthers": "CAR",
            "Chicago Bears": "CHI",
            "Cincinnati Bengals": "CIN",
            "Cleveland Browns": "CLE",
            "Dallas Cowboys": "DAL",
            "Denver Broncos": "DEN",
            "Detroit Lions": "DET",
            "Green Bay Packers": "GB",
            "Houston Texans": "HOU",
            "Indianapolis Colts": "IND",
            "Jacksonville Jaguars": "JAX",
            "Kansas City Chiefs": "KC",
            "Las Vegas Raiders": "LV",
            "Los Angeles Chargers": "LAC",
            "Los Angeles Rams": "LAR",
            "Miami Dolphins": "MIA",
            "Minnesota Vikings": "MIN",
            "New England Patriots": "NE",
            "New Orleans Saints": "NO",
            "New York Giants": "NYG",
            "New York Jets": "NYJ",
            "Philadelphia Eagles": "PHI",
            "Pittsburgh Steelers": "PIT",
            "San Francisco 49ers": "SF",
            "Seattle Seahawks": "SEA",
            "Tampa Bay Buccaneers": "TB",
            "Tennessee Titans": "TEN",
            "Washington Commanders": "WAS",
        }
        # Try exact match first
        if team_name in name_map:
            return name_map[team_name]
        # Try partial match
        for full_name, abbrev in name_map.items():
            if team_name.lower() in full_name.lower() or full_name.lower() in team_name.lower():
                return abbrev
        return None

    def _parse_news_markdown(self, markdown: str, limit: int) -> list[RotoWireNewsItem]:
        """
        Parse news items from Crawl4AI's markdown output.

        RotoWire's news format typically shows player name followed by analysis.
        """
        items = []

        # Split into sections (Crawl4AI often produces ## headers for articles)
        sections = re.split(r"\n(?=##?\s)", markdown)

        for section in sections[: limit * 2]:  # Check more than limit (some may be invalid)
            if len(items) >= limit:
                break

            try:
                item = self._parse_news_section(section)
                if item:
                    items.append(item)
            except Exception as e:
                logger.debug("Failed to parse section: %s", e)

        return items

    def _parse_news_section(self, section: str) -> RotoWireNewsItem | None:
        """Parse a single news section."""
        lines = section.strip().split("\n")
        if not lines:
            return None

        # Extract title from first line (often markdown header)
        title = lines[0].strip()
        title = re.sub(r"^#+\s*", "", title)  # Remove markdown headers
        title = re.sub(r"\[|\]|\(.*?\)", "", title)  # Remove markdown links

        if len(title) < 10 or len(title) > 300:
            return None

        # Description is the rest of the text
        description = " ".join(lines[1:]).strip()
        description = re.sub(r"\s+", " ", description)  # Normalize whitespace

        if len(description) < 20:
            return None

        # Extract teams from text
        teams = self._extract_teams(title + " " + description)

        # Extract player names (capitalized words that look like names)
        players = self._extract_players(title)

        # Determine category
        category = self._infer_category(title + " " + description)

        # Generate ID from content hash
        content = f"{title}|{description}"
        content_id = hashlib.sha256(content.encode()).hexdigest()[:12]

        return RotoWireNewsItem(
            id=content_id,
            title=title,
            description=description[:500],  # Truncate long descriptions
            url=self.NEWS_URL,
            published_at=datetime.utcnow(),  # RotoWire doesn't always show timestamps
            teams=list(set(teams)),
            players=players[:5],  # Limit to 5 players
            category=category,
        )

    def _extract_teams(self, text: str) -> list[str]:
        """Extract NFL team abbreviations from text."""
        teams = []
        for pattern, abbrev in NFL_TEAM_PATTERNS.items():
            if re.search(rf"\b{pattern}\b", text, re.IGNORECASE):
                if abbrev not in teams:
                    teams.append(abbrev)
        return teams

    def _extract_players(self, text: str) -> list[str]:
        """Extract player names from text (heuristic)."""
        # Look for capitalized name patterns (First Last)
        pattern = r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b"
        matches = re.findall(pattern, text)

        # Filter out common non-player words
        non_players = {"New York", "San Francisco", "Los Angeles", "Kansas City", "Green Bay"}
        players = [m for m in matches if m not in non_players]

        return players

    def _infer_category(self, text: str) -> str:
        """Infer news category from content using word boundaries."""
        text_lower = text.lower()

        def has_word(keywords: list[str]) -> bool:
            """Check if any keyword appears as a whole word."""
            return any(re.search(rf"\b{re.escape(kw)}\b", text_lower) for kw in keywords)

        # Check in order of specificity (trade/coaching first since they're less common)
        if has_word(["trade", "traded", "acquire", "signing"]):
            return "trade"
        elif has_word(["coach", "hire", "fire", "coordinator"]):
            return "coaching"
        elif has_word(["start", "bench", "lineup"]):
            return "lineup"
        elif has_word(INJURY_KEYWORDS):
            return "injury"
        elif has_word(["snap", "target", "touch", "usage"]):
            return "fantasy"

        return "general"

    async def get_health_status(self) -> dict[str, Any]:
        """Check if RotoWire is accessible."""
        try:
            import time

            start = time.time()

            crawler = await self._get_crawler()
            async with crawler:
                result = await crawler.arun(
                    url=self.NEWS_URL,
                    timeout=10000,
                )

            latency = (time.time() - start) * 1000

            return {
                "status": "healthy" if result.success else "unhealthy",
                "latency_ms": round(latency, 2),
                "content_length": len(result.markdown) if result.success else 0,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


async def test_rotowire_client():
    """Quick test of RotoWire client."""

    client = RotoWireClient()

    print("\n=== RotoWire Health Check ===")
    health = await client.get_health_status()
    print(f"Status: {health}")

    print("\n=== News Fetch ===")
    news = await client.fetch_news(limit=10)
    for item in news[:5]:
        print(f"  [{item.category}] {item.title[:60]}...")
        print(f"    Teams: {item.teams}, Players: {item.players[:2]}")

    print(f"\n=== Summary ===")
    print(f"News items: {len(news)}")


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_rotowire_client())
