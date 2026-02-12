"""
Enhanced NFL news ingestion for semantic processing.

Wraps the existing mcp_server news fetcher and adds:
- Multiple source aggregation (ESPN, NFL.com, team sites)
- Injury report parsing
- Structured output for LLM processing
- Priority scoring for betting-relevant news
- Real-time streaming via Redis pub/sub
"""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NewsCategory(Enum):
    """Categories for NFL news relevance to betting."""

    INJURY = "injury"
    LINEUP = "lineup"
    COACHING = "coaching"
    WEATHER = "weather"
    SUSPENSION = "suspension"
    TRADE = "trade"
    GENERAL = "general"


class NewsPriority(Enum):
    """Priority levels for news items."""

    CRITICAL = 1  # Game-changing (star player out, coach fired)
    HIGH = 2  # Significant (key player questionable, scheme change)
    MEDIUM = 3  # Notable (depth player injury, minor news)
    LOW = 4  # Background (general articles, analysis)


@dataclass
class NewsItem:
    """Structured news item for semantic processing."""

    id: str  # Unique hash of title + source
    title: str
    url: str
    source: str
    published: datetime | None
    description: str = ""
    category: NewsCategory = NewsCategory.GENERAL
    priority: NewsPriority = NewsPriority.MEDIUM
    teams: list[str] = field(default_factory=list)
    players: list[str] = field(default_factory=list)
    raw_content: str = ""
    processed: bool = False
    sentiment: float | None = None  # -1.0 to 1.0, filled by semantic processing

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            hash_input = f"{self.title}:{self.source}:{self.url}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["category"] = self.category.value
        d["priority"] = self.priority.value
        d["published"] = self.published.isoformat() if self.published else None
        return d


# Injury keywords for classification (use word boundaries for short words)
INJURY_KEYWORDS = [
    "injury",
    "injured",
    "hurt",
    "questionable",
    "doubtful",
    "probable",
    "limited",
    "dnp",
    "did not practice",
    "ankle",
    "knee",
    "hamstring",
    "concussion",
    "shoulder",
    "hip",
    "groin",
    "calf",
    "quad",
    "achilles",
    "torn",
    "sprain",
    "strain",
    "fracture",
    "surgery",
    "injured reserve",
    "pup",
    "nfi",
]

# Short keywords that need word boundary matching
INJURY_KEYWORDS_EXACT = ["out", "back", "ir"]

# Coaching/lineup keywords
COACHING_KEYWORDS = [
    "coach",
    "fired",
    "hired",
    "coordinator",
    "playcaller",
    "starter",
    "benched",
    "starting",
    "lineup",
    "depth chart",
    "promoted",
    "demoted",
]

# Weather keywords
WEATHER_KEYWORDS = [
    "weather",
    "snow",
    "rain",
    "wind",
    "cold",
    "dome",
    "outdoor",
    "temperature",
    "forecast",
    "conditions",
]

# Star player patterns (for priority boosting)
STAR_PLAYER_PATTERNS = [
    r"\b(mahomes|allen|burrow|jackson|hurts|herbert)\b",
    r"\b(kelce|hill|chase|jefferson|lamb|diggs)\b",
    r"\b(henry|mccaffrey|barkley|chubb|taylor)\b",
]


def classify_news(title: str, description: str = "") -> tuple[NewsCategory, NewsPriority]:
    """
    Classify news by category and priority based on keywords.

    Args:
        title: News headline
        description: Article description/summary

    Returns:
        Tuple of (category, priority)
    """
    text = f"{title} {description}".lower()

    # Check for trades FIRST (to avoid "acquire" triggering other categories)
    if "trade" in text or "traded" in text or "acquired" in text:
        return NewsCategory.TRADE, NewsPriority.MEDIUM

    # Check for injuries (highest priority if star player)
    has_injury = any(kw in text for kw in INJURY_KEYWORDS)
    # Check exact match keywords with word boundaries
    if not has_injury:
        has_injury = any(re.search(rf"\b{kw}\b", text) for kw in INJURY_KEYWORDS_EXACT)

    if has_injury:
        category = NewsCategory.INJURY
        # Check if star player mentioned
        is_star = any(re.search(p, text, re.IGNORECASE) for p in STAR_PLAYER_PATTERNS)
        priority = NewsPriority.CRITICAL if is_star else NewsPriority.HIGH
        return category, priority

    # Check for coaching/lineup changes
    if any(kw in text for kw in COACHING_KEYWORDS):
        return NewsCategory.COACHING, NewsPriority.HIGH

    # Check for weather
    if any(kw in text for kw in WEATHER_KEYWORDS):
        return NewsCategory.WEATHER, NewsPriority.MEDIUM

    # Check for suspensions
    if "suspend" in text or "suspension" in text:
        return NewsCategory.SUSPENSION, NewsPriority.HIGH

    # Default to general/low
    return NewsCategory.GENERAL, NewsPriority.LOW


def extract_teams_from_text(text: str) -> list[str]:
    """
    Extract team mentions from text using known team patterns.

    Args:
        text: News text

    Returns:
        List of team abbreviations found
    """
    # Import team mapping
    try:
        from mcp_server.tools.news_fetcher import TEAM_MAPPING
    except ImportError:
        # Fallback team list
        TEAM_MAPPING = {
            "ARI": "Cardinals",
            "ATL": "Falcons",
            "BAL": "Ravens",
            "BUF": "Bills",
            "CAR": "Panthers",
            "CHI": "Bears",
            "CIN": "Bengals",
            "CLE": "Browns",
            "DAL": "Cowboys",
            "DEN": "Broncos",
            "DET": "Lions",
            "GB": "Packers",
            "HOU": "Texans",
            "IND": "Colts",
            "JAX": "Jaguars",
            "KC": "Chiefs",
            "LAC": "Chargers",
            "LAR": "Rams",
            "LV": "Raiders",
            "MIA": "Dolphins",
            "MIN": "Vikings",
            "NE": "Patriots",
            "NO": "Saints",
            "NYG": "Giants",
            "NYJ": "Jets",
            "PHI": "Eagles",
            "PIT": "Steelers",
            "SEA": "Seahawks",
            "SF": "49ers",
            "TB": "Buccaneers",
            "TEN": "Titans",
            "WAS": "Commanders",
        }

    # Abbreviations that could be common words - require UPPERCASE
    AMBIGUOUS_ABBRS = {"WAS", "NO", "NE", "DEN", "CAR", "MIN", "TEN", "IND", "LV"}

    text_lower = text.lower()
    found_teams = []

    for abbr, name in TEAM_MAPPING.items():
        # Check for full team name first (most reliable)
        if name.lower() in text_lower:
            if abbr not in found_teams:
                found_teams.append(abbr)
            continue

        # For ambiguous abbreviations, require uppercase match
        if abbr in AMBIGUOUS_ABBRS:
            if re.search(rf"\b{abbr}\b", text):  # Case-sensitive
                if abbr not in found_teams:
                    found_teams.append(abbr)
        else:
            # Other abbreviations can be case-insensitive
            if re.search(rf"\b{abbr}\b", text, re.IGNORECASE):
                if abbr not in found_teams:
                    found_teams.append(abbr)

    return found_teams


class NewsIngester:
    """
    Enhanced news ingester for semantic processing.

    Aggregates news from multiple sources, classifies by importance,
    and prepares for LLM-based entity extraction and sentiment analysis.
    """

    def __init__(
        self,
        redis_client=None,
        cache_ttl: int = 300,
    ):
        """
        Initialize the news ingester.

        Args:
            redis_client: Optional Redis client for caching/pub-sub
            cache_ttl: Cache time-to-live in seconds (default 5 min)
        """
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self._news_buffer: list[NewsItem] = []

    def fetch_team_news(
        self,
        team: str,
        limit: int = 10,
        min_priority: NewsPriority = NewsPriority.LOW,
    ) -> list[NewsItem]:
        """
        Fetch and classify news for a specific team.

        Args:
            team: Team abbreviation or name
            limit: Maximum articles to return
            min_priority: Minimum priority level to include

        Returns:
            List of classified NewsItem objects
        """
        try:
            from mcp_server.tools.news_fetcher import fetch_espn_news
        except ImportError:
            logger.error("ESPN news fetcher not available")
            return []

        # Fetch raw news
        raw_articles = fetch_espn_news(team, limit=limit * 2)  # Fetch extra for filtering

        # Convert to NewsItem with classification
        items = []
        for article in raw_articles:
            category, priority = classify_news(
                article.get("title", ""),
                article.get("description", ""),
            )

            # Filter by priority
            if priority.value > min_priority.value:
                continue

            # Extract teams mentioned
            teams = extract_teams_from_text(
                f"{article.get('title', '')} {article.get('description', '')}"
            )

            item = NewsItem(
                id="",
                title=article.get("title", ""),
                url=article.get("url", ""),
                source=article.get("source", "ESPN"),
                published=None,  # Parse if available
                description=article.get("description", ""),
                category=category,
                priority=priority,
                teams=teams,
            )

            items.append(item)

            if len(items) >= limit:
                break

        logger.info(f"Fetched {len(items)} news items for {team}")
        return items

    def fetch_all_teams_news(
        self,
        teams: list[str] | None = None,
        limit_per_team: int = 5,
        min_priority: NewsPriority = NewsPriority.MEDIUM,
    ) -> list[NewsItem]:
        """
        Fetch news for multiple teams, deduplicated.

        Args:
            teams: List of team abbreviations (None = all 32 teams)
            limit_per_team: Max articles per team
            min_priority: Minimum priority to include

        Returns:
            Deduplicated list of NewsItems sorted by priority
        """
        if teams is None:
            # All 32 teams - but this would be slow, so default to a subset
            teams = ["KC", "SF", "PHI", "BUF", "BAL", "DAL", "DET", "MIA"]

        all_items: dict[str, NewsItem] = {}  # Keyed by ID for dedup

        for team in teams:
            try:
                items = self.fetch_team_news(team, limit_per_team, min_priority)
                for item in items:
                    if item.id not in all_items:
                        all_items[item.id] = item
            except Exception as e:
                logger.warning(f"Failed to fetch news for {team}: {e}")

        # Sort by priority (lower = more important)
        sorted_items = sorted(all_items.values(), key=lambda x: x.priority.value)

        logger.info(f"Fetched {len(sorted_items)} total news items for {len(teams)} teams")
        return sorted_items

    def fetch_injury_news(self, limit: int = 20) -> list[NewsItem]:
        """
        Fetch injury-specific news across the league.

        Args:
            limit: Maximum items to return

        Returns:
            List of injury-related NewsItems
        """
        try:
            from mcp_server.tools.news_fetcher import search_all_nfl_news
        except ImportError:
            logger.error("ESPN news search not available")
            return []

        # Search for injury-related keywords
        keywords = ["injury", "injured", "questionable", "doubtful", "out"]
        raw_articles = search_all_nfl_news(keywords, hours=48)

        items = []
        for article in raw_articles[:limit]:
            category, priority = classify_news(
                article.get("title", ""),
                article.get("description", ""),
            )

            teams = extract_teams_from_text(
                f"{article.get('title', '')} {article.get('description', '')}"
            )

            item = NewsItem(
                id="",
                title=article.get("title", ""),
                url=article.get("url", ""),
                source=article.get("source", "ESPN"),
                published=None,
                description=article.get("description", ""),
                category=category,
                priority=priority,
                teams=teams,
            )
            items.append(item)

        return items

    def get_high_priority_news(self, hours: int = 24) -> list[NewsItem]:
        """
        Get high-priority news from the last N hours.

        Args:
            hours: Lookback window

        Returns:
            List of CRITICAL and HIGH priority news items
        """
        # This is a convenience method for the betting system
        all_news = self.fetch_all_teams_news(min_priority=NewsPriority.HIGH)
        return [n for n in all_news if n.priority.value <= NewsPriority.HIGH.value]

    def prepare_for_llm(self, news_items: list[NewsItem]) -> list[dict[str, Any]]:
        """
        Prepare news items for LLM processing.

        Args:
            news_items: List of NewsItem objects

        Returns:
            List of dicts optimized for LLM context
        """
        return [
            {
                "id": item.id,
                "text": f"{item.title}. {item.description}".strip(),
                "category": item.category.value,
                "priority": item.priority.name,
                "teams": item.teams,
                "source": item.source,
            }
            for item in news_items
        ]

    def to_json(self, news_items: list[NewsItem]) -> str:
        """Serialize news items to JSON string."""
        return json.dumps([item.to_dict() for item in news_items], indent=2)


# Convenience function for quick testing
def test_ingester():
    """Test the news ingester."""
    ingester = NewsIngester()

    print("Fetching Chiefs news...")
    items = ingester.fetch_team_news("KC", limit=5)
    for item in items:
        print(f"  [{item.priority.name}] {item.category.value}: {item.title[:60]}...")

    print("\nFetching high-priority league news...")
    high_priority = ingester.get_high_priority_news()
    print(f"Found {len(high_priority)} high-priority items")


if __name__ == "__main__":
    test_ingester()
