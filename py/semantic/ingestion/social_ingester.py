"""
Social media ingestion for NFL semantic analysis.

Handles Twitter/X integration for:
- Beat reporter tweets (verified sources)
- Official team accounts
- NFL insider accounts (Schefter, Rapoport, etc.)
- Sentiment aggregation from social signals

Note: Requires Twitter/X API v2 access for production.
This module provides the infrastructure and mock data for development.

Usage:
    ingester = SocialIngester(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))
    tweets = ingester.fetch_insider_tweets(hours=24)
    for tweet in tweets:
        print(f"[{tweet.source}] {tweet.text[:80]}...")
"""

import hashlib
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TweetCategory(Enum):
    """Categories for NFL tweets."""

    INJURY_UPDATE = "injury_update"
    LINEUP_NEWS = "lineup_news"
    PRACTICE_REPORT = "practice_report"
    GAME_PREVIEW = "game_preview"
    BREAKING_NEWS = "breaking_news"
    ANALYSIS = "analysis"
    RUMOR = "rumor"
    GENERAL = "general"


class SourceTier(Enum):
    """Credibility tiers for sources."""

    TIER_1 = 1  # Official team accounts, NFL.com
    TIER_2 = 2  # National insiders (Schefter, Rapoport, Pelissero)
    TIER_3 = 3  # Beat reporters, local media
    TIER_4 = 4  # Fan accounts, aggregators (use with caution)


@dataclass
class SocialPost:
    """Structured social media post for processing."""

    id: str
    text: str
    author: str
    author_handle: str
    source_tier: SourceTier
    category: TweetCategory
    timestamp: datetime
    likes: int = 0
    retweets: int = 0
    teams: list[str] = field(default_factory=list)
    players: list[str] = field(default_factory=list)
    url: str = ""
    is_verified: bool = False
    sentiment: float | None = None  # Filled by LLM processing

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            hash_input = f"{self.text}:{self.author}:{self.timestamp}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score (likes + 2*retweets)."""
        return self.likes + 2 * self.retweets

    @property
    def credibility_weight(self) -> float:
        """Weight based on source tier (higher = more credible)."""
        return {
            SourceTier.TIER_1: 1.0,
            SourceTier.TIER_2: 0.9,
            SourceTier.TIER_3: 0.7,
            SourceTier.TIER_4: 0.3,
        }.get(self.source_tier, 0.5)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["source_tier"] = self.source_tier.value
        d["category"] = self.category.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


# Verified NFL insider accounts (Tier 2)
NFL_INSIDERS = {
    "AdamSchefter": {"name": "Adam Schefter", "tier": SourceTier.TIER_2},
    "RapSheet": {"name": "Ian Rapoport", "tier": SourceTier.TIER_2},
    "TomPelissero": {"name": "Tom Pelissero", "tier": SourceTier.TIER_2},
    "JayGlazer": {"name": "Jay Glazer", "tier": SourceTier.TIER_2},
    "JosinaAnderson": {"name": "Josina Anderson", "tier": SourceTier.TIER_2},
    "FieldYates": {"name": "Field Yates", "tier": SourceTier.TIER_2},
    "JeffDarlington": {"name": "Jeff Darlington", "tier": SourceTier.TIER_2},
    "MikeGarafolo": {"name": "Mike Garafolo", "tier": SourceTier.TIER_2},
    "daborkowski": {"name": "Dianna Russini", "tier": SourceTier.TIER_2},
}

# Official team accounts (Tier 1)
OFFICIAL_TEAM_ACCOUNTS = {
    "Chiefs": "Chiefs",
    "Ravens": "Ravens",
    "49ers": "49ers",
    "Eagles": "Eagles",
    "Bills": "BuffaloBills",
    "Cowboys": "dallascowboys",
    "Lions": "Lions",
    "Dolphins": "MiamiDolphins",
    "Bengals": "Bengals",
    "Packers": "packers",
    "Texans": "HoustonTexans",
    "Browns": "Browns",
    "Steelers": "steelers",
    "Jets": "nyjets",
    "Patriots": "Patriots",
    "Chargers": "chargers",
    "Broncos": "Broncos",
    "Raiders": "Raiders",
    "Cardinals": "AZCardinals",
    "Rams": "RamsNFL",
    "Seahawks": "Seahawks",
    "Buccaneers": "Buccaneers",
    "Saints": "Saints",
    "Falcons": "AtlantaFalcons",
    "Panthers": "Panthers",
    "Vikings": "Vikings",
    "Bears": "ChicagoBears",
    "Titans": "Titans",
    "Colts": "Colts",
    "Jaguars": "Jaguars",
    "Giants": "Giants",
    "Commanders": "Commanders",
}

# Breaking news keywords
BREAKING_KEYWORDS = [
    "breaking", "just in", "confirmed", "official", "announced",
    "source:", "per source", "i'm told", "hearing", "expected to",
]

# Injury keywords for classification
INJURY_KEYWORDS = [
    "injury", "injured", "questionable", "doubtful", "out",
    "dnp", "limited", "full practice", "ir", "pup",
]


def classify_tweet(text: str) -> TweetCategory:
    """
    Classify a tweet by category.

    Args:
        text: Tweet text

    Returns:
        TweetCategory enum value
    """
    text_lower = text.lower()

    # Check for breaking news
    if any(kw in text_lower for kw in BREAKING_KEYWORDS):
        return TweetCategory.BREAKING_NEWS

    # Check for injury updates
    if any(kw in text_lower for kw in INJURY_KEYWORDS):
        return TweetCategory.INJURY_UPDATE

    # Check for practice reports
    if "practice" in text_lower or "wednesday" in text_lower or "dnp" in text_lower:
        return TweetCategory.PRACTICE_REPORT

    # Check for lineup news
    if any(kw in text_lower for kw in ["starting", "starter", "benched", "inactive"]):
        return TweetCategory.LINEUP_NEWS

    # Check for game preview
    if any(kw in text_lower for kw in ["preview", "matchup", "keys to the game"]):
        return TweetCategory.GAME_PREVIEW

    # Check for rumors
    if any(kw in text_lower for kw in ["rumor", "might", "could", "possibly"]):
        return TweetCategory.RUMOR

    return TweetCategory.GENERAL


class SocialIngester:
    """
    Social media ingester for NFL betting signals.

    Handles Twitter/X API integration with:
    - Rate limiting and caching
    - Source credibility scoring
    - Category classification
    - Deduplication
    """

    def __init__(
        self,
        bearer_token: str | None = None,
        cache_ttl: int = 300,
    ):
        """
        Initialize the social ingester.

        Args:
            bearer_token: Twitter API v2 bearer token
            cache_ttl: Cache time-to-live in seconds
        """
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        self.cache_ttl = cache_ttl
        self._tweet_cache: dict[str, SocialPost] = {}
        self._api_available = bool(self.bearer_token)

        if not self._api_available:
            logger.warning(
                "Twitter API not configured. Set TWITTER_BEARER_TOKEN for live data. "
                "Using mock data for development."
            )

    @property
    def is_available(self) -> bool:
        """Check if Twitter API is available."""
        return self._api_available

    def fetch_insider_tweets(
        self,
        hours: int = 24,
        min_tier: SourceTier = SourceTier.TIER_3,
    ) -> list[SocialPost]:
        """
        Fetch tweets from NFL insiders.

        Args:
            hours: Lookback window in hours
            min_tier: Minimum source tier to include

        Returns:
            List of SocialPost objects
        """
        if not self._api_available:
            logger.info("Using mock insider tweets (API not configured)")
            return self._get_mock_insider_tweets()

        # Real API implementation would go here
        return self._fetch_twitter_api(
            accounts=list(NFL_INSIDERS.keys()),
            hours=hours,
            min_tier=min_tier,
        )

    def fetch_team_tweets(
        self,
        teams: list[str],
        hours: int = 24,
    ) -> list[SocialPost]:
        """
        Fetch tweets from official team accounts.

        Args:
            teams: List of team names or abbreviations
            hours: Lookback window

        Returns:
            List of SocialPost objects
        """
        if not self._api_available:
            logger.info("Using mock team tweets (API not configured)")
            return self._get_mock_team_tweets(teams)

        # Map team names to handles
        handles = []
        for team in teams:
            team_upper = team.upper()
            # Try direct match first
            if team in OFFICIAL_TEAM_ACCOUNTS:
                handles.append(OFFICIAL_TEAM_ACCOUNTS[team])
            # Try by team name
            else:
                for name, handle in OFFICIAL_TEAM_ACCOUNTS.items():
                    if team_upper in name.upper():
                        handles.append(handle)
                        break

        return self._fetch_twitter_api(accounts=handles, hours=hours)

    def _fetch_twitter_api(
        self,
        accounts: list[str],
        hours: int,
        min_tier: SourceTier = SourceTier.TIER_4,
    ) -> list[SocialPost]:
        """
        Fetch tweets from Twitter API v2.

        Args:
            accounts: List of Twitter handles
            hours: Lookback window
            min_tier: Minimum tier to include

        Returns:
            List of SocialPost objects
        """
        try:
            import httpx

            # Twitter API v2 endpoint
            base_url = "https://api.twitter.com/2"

            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
            }

            posts = []
            start_time = datetime.utcnow() - timedelta(hours=hours)

            for handle in accounts:
                try:
                    # Get user ID
                    user_resp = httpx.get(
                        f"{base_url}/users/by/username/{handle}",
                        headers=headers,
                        timeout=10.0,
                    )
                    if user_resp.status_code != 200:
                        logger.warning(f"Failed to get user {handle}: {user_resp.status_code}")
                        continue

                    user_data = user_resp.json().get("data", {})
                    user_id = user_data.get("id")
                    if not user_id:
                        continue

                    # Get recent tweets
                    tweets_resp = httpx.get(
                        f"{base_url}/users/{user_id}/tweets",
                        headers=headers,
                        params={
                            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "max_results": 20,
                            "tweet.fields": "created_at,public_metrics",
                        },
                        timeout=10.0,
                    )

                    if tweets_resp.status_code != 200:
                        logger.warning(f"Failed to get tweets for {handle}")
                        continue

                    tweets_data = tweets_resp.json().get("data", [])

                    # Determine tier
                    if handle in NFL_INSIDERS:
                        tier = NFL_INSIDERS[handle]["tier"]
                        name = NFL_INSIDERS[handle]["name"]
                    elif handle in OFFICIAL_TEAM_ACCOUNTS.values():
                        tier = SourceTier.TIER_1
                        name = handle
                    else:
                        tier = SourceTier.TIER_4
                        name = handle

                    # Filter by tier
                    if tier.value > min_tier.value:
                        continue

                    for tweet in tweets_data:
                        metrics = tweet.get("public_metrics", {})
                        post = SocialPost(
                            id=tweet.get("id", ""),
                            text=tweet.get("text", ""),
                            author=name,
                            author_handle=handle,
                            source_tier=tier,
                            category=classify_tweet(tweet.get("text", "")),
                            timestamp=datetime.fromisoformat(
                                tweet.get("created_at", "").replace("Z", "+00:00")
                            ),
                            likes=metrics.get("like_count", 0),
                            retweets=metrics.get("retweet_count", 0),
                            is_verified=True,
                        )
                        posts.append(post)

                except Exception as e:
                    logger.warning(f"Error fetching tweets for {handle}: {e}")
                    continue

            return posts

        except ImportError:
            logger.error("httpx not installed for Twitter API")
            return []

    def _get_mock_insider_tweets(self) -> list[SocialPost]:
        """Get mock insider tweets for development."""
        now = datetime.now()

        return [
            SocialPost(
                id="mock_1",
                text="BREAKING: Patrick Mahomes (ankle) is officially questionable for Sunday's game vs Ravens. Source says he's 'day-to-day' and Chiefs are cautiously optimistic.",
                author="Adam Schefter",
                author_handle="AdamSchefter",
                source_tier=SourceTier.TIER_2,
                category=TweetCategory.BREAKING_NEWS,
                timestamp=now - timedelta(hours=2),
                likes=15420,
                retweets=8234,
                teams=["KC"],
                players=["Patrick Mahomes"],
                is_verified=True,
            ),
            SocialPost(
                id="mock_2",
                text="Ravens injury report: Lamar Jackson (knee) - LIMITED. Derrick Henry (rest) - FULL. Mark Andrews (knee) - DNP.",
                author="Ian Rapoport",
                author_handle="RapSheet",
                source_tier=SourceTier.TIER_2,
                category=TweetCategory.INJURY_UPDATE,
                timestamp=now - timedelta(hours=4),
                likes=8932,
                retweets=3421,
                teams=["BAL"],
                players=["Lamar Jackson", "Derrick Henry", "Mark Andrews"],
                is_verified=True,
            ),
            SocialPost(
                id="mock_3",
                text="Hearing the 49ers are expected to activate Christian McCaffrey off IR for Saturday's playoff game. He's been practicing in full this week.",
                author="Tom Pelissero",
                author_handle="TomPelissero",
                source_tier=SourceTier.TIER_2,
                category=TweetCategory.BREAKING_NEWS,
                timestamp=now - timedelta(hours=6),
                likes=12543,
                retweets=5678,
                teams=["SF"],
                players=["Christian McCaffrey"],
                is_verified=True,
            ),
        ]

    def _get_mock_team_tweets(self, teams: list[str]) -> list[SocialPost]:
        """Get mock team tweets for development."""
        now = datetime.now()

        mock_tweets = {
            "KC": SocialPost(
                id="team_kc_1",
                text="Injury Report | Week 16\n\nQuestionable:\n- Patrick Mahomes (ankle)\n- Chris Jones (calf)\n\nOut:\n- Hollywood Brown (shoulder)",
                author="Kansas City Chiefs",
                author_handle="Chiefs",
                source_tier=SourceTier.TIER_1,
                category=TweetCategory.INJURY_UPDATE,
                timestamp=now - timedelta(hours=3),
                likes=5432,
                retweets=1234,
                teams=["KC"],
                is_verified=True,
            ),
            "BAL": SocialPost(
                id="team_bal_1",
                text="OFFICIAL: Lamar Jackson has been named AFC Offensive Player of the Week for Week 15.",
                author="Baltimore Ravens",
                author_handle="Ravens",
                source_tier=SourceTier.TIER_1,
                category=TweetCategory.GENERAL,
                timestamp=now - timedelta(hours=12),
                likes=8765,
                retweets=2345,
                teams=["BAL"],
                players=["Lamar Jackson"],
                is_verified=True,
            ),
        }

        result = []
        for team in teams:
            team_upper = team.upper()
            if team_upper in mock_tweets:
                result.append(mock_tweets[team_upper])

        return result

    def get_sentiment_weighted_score(
        self,
        posts: list[SocialPost],
    ) -> dict[str, float]:
        """
        Calculate sentiment-weighted score per team.

        Weighs sentiment by:
        - Source credibility (tier)
        - Engagement (likes + retweets)
        - Recency (decay over time)

        Args:
            posts: List of SocialPost objects with sentiment filled

        Returns:
            Dict of team -> weighted sentiment score (-1.0 to 1.0)
        """
        from collections import defaultdict

        team_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
        now = datetime.now()

        for post in posts:
            if post.sentiment is None:
                continue

            # Calculate time decay (half-life of 12 hours)
            hours_old = (now - post.timestamp).total_seconds() / 3600
            time_weight = 0.5 ** (hours_old / 12)

            # Calculate engagement weight (log scale)
            engagement_weight = 1.0 + 0.1 * (post.engagement_score ** 0.5)

            # Combine weights
            total_weight = (
                post.credibility_weight *
                time_weight *
                engagement_weight
            )

            for team in post.teams:
                team_scores[team].append((post.sentiment, total_weight))

        # Calculate weighted average per team
        result = {}
        for team, scores in team_scores.items():
            if scores:
                total_weight = sum(w for _, w in scores)
                weighted_sum = sum(s * w for s, w in scores)
                result[team] = weighted_sum / total_weight if total_weight > 0 else 0.0

        return result

    def prepare_for_llm(self, posts: list[SocialPost]) -> list[dict[str, Any]]:
        """
        Prepare social posts for LLM processing.

        Args:
            posts: List of SocialPost objects

        Returns:
            List of dicts optimized for LLM context
        """
        return [
            {
                "id": post.id,
                "text": post.text,
                "author": post.author,
                "tier": post.source_tier.name,
                "category": post.category.value,
                "teams": post.teams,
                "engagement": post.engagement_score,
                "hours_ago": round(
                    (datetime.now() - post.timestamp).total_seconds() / 3600, 1
                ),
            }
            for post in posts
        ]


def test_social_ingester():
    """Test the social ingester."""
    ingester = SocialIngester()

    print("=== Testing Social Ingester ===")
    print(f"API available: {ingester.is_available}")
    print()

    print("Fetching insider tweets...")
    tweets = ingester.fetch_insider_tweets(hours=24)
    for tweet in tweets:
        print(f"  [{tweet.source_tier.name}] @{tweet.author_handle}: {tweet.text[:60]}...")
    print()

    print("Fetching team tweets for KC, BAL...")
    team_tweets = ingester.fetch_team_tweets(["KC", "BAL"])
    for tweet in team_tweets:
        print(f"  [{tweet.category.value}] {tweet.text[:60]}...")


if __name__ == "__main__":
    test_social_ingester()
