"""
Semantic feature extraction for NFL betting models.

Computes LLM-derived features from news and social media:
- Team sentiment scores (24h, 7d trends)
- Injury impact from NER extraction
- Media buzz volume and trends
- Coaching/scheme change detection
- Public bias indicators

These features integrate with GameFeatureExtractor to enhance the ensemble.

Usage:
    extractor = SemanticFeatureExtractor()
    features = extractor.extract_features('KC', 'BAL', 2025, 16)
    # Returns: {'home_sentiment_24h': 0.35, 'away_injury_llm_severity': 0.8, ...}
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SemanticFeatureConfig:
    """Configuration for semantic feature extraction."""

    # Lookback windows
    news_lookback_hours: int = 48
    social_lookback_hours: int = 24

    # LLM settings
    use_llm_sentiment: bool = True
    use_llm_ner: bool = True
    llm_timeout: float = 240.0

    # Caching
    cache_ttl: int = 300  # 5 minutes

    # Feature weights (for meta-features)
    sentiment_weight: float = 0.4
    injury_weight: float = 0.4
    buzz_weight: float = 0.2


@dataclass
class TeamSemanticState:
    """Semantic state for a single team at a point in time."""

    team: str
    timestamp: datetime

    # Sentiment features
    sentiment_24h: float = 0.0  # -1.0 to 1.0
    sentiment_7d_trend: float = 0.0  # Change over 7 days
    sentiment_confidence: float = 0.5

    # Injury features (LLM-derived)
    injury_mentions: int = 0
    injury_severity_llm: float = 0.0  # 0.0-1.0 (LLM-assessed)
    key_player_risk: float = 0.0  # 0.0-1.0

    # Media features
    media_buzz_volume: float = 0.0  # Normalized 0-1
    media_buzz_trend: float = 0.0  # Change vs prior week

    # Coaching/scheme features
    coaching_change_detected: bool = False
    scheme_mention_count: int = 0

    # Public perception
    public_overreaction_score: float = 0.0  # 0-1, higher = more reactive

    def to_feature_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to feature dictionary with optional prefix."""
        return {
            f"{prefix}sentiment_24h": self.sentiment_24h,
            f"{prefix}sentiment_7d_trend": self.sentiment_7d_trend,
            f"{prefix}sentiment_confidence": self.sentiment_confidence,
            f"{prefix}injury_mentions": float(self.injury_mentions),
            f"{prefix}injury_severity_llm": self.injury_severity_llm,
            f"{prefix}key_player_risk": self.key_player_risk,
            f"{prefix}media_buzz_volume": self.media_buzz_volume,
            f"{prefix}media_buzz_trend": self.media_buzz_trend,
            f"{prefix}coaching_change": float(self.coaching_change_detected),
            f"{prefix}scheme_mentions": float(self.scheme_mention_count),
            f"{prefix}public_overreaction": self.public_overreaction_score,
        }


class SemanticFeatureExtractor:
    """
    Extract semantic features for NFL game matchups.

    Combines news sentiment, social media signals, and LLM-derived
    insights into features for the betting model ensemble.

    Features per team (11 each, 22 total + 4 matchup):
    - sentiment_24h: Weighted sentiment from last 24h
    - sentiment_7d_trend: Sentiment trend direction
    - sentiment_confidence: LLM confidence in sentiment
    - injury_mentions: Count of injury-related news
    - injury_severity_llm: LLM-assessed severity (0-1)
    - key_player_risk: Risk of key player absence (0-1)
    - media_buzz_volume: Normalized media attention
    - media_buzz_trend: Change in buzz vs prior week
    - coaching_change: Binary coaching change flag
    - scheme_mentions: Count of scheme-related news
    - public_overreaction: Overreaction indicator (0-1)

    Matchup features (4):
    - sentiment_differential: home - away sentiment
    - injury_advantage: home vs away injury severity
    - media_attention_ratio: relative media coverage
    - narrative_momentum: combined narrative direction
    """

    FEATURE_COUNT = 26  # 11 per team + 4 matchup

    def __init__(
        self,
        config: SemanticFeatureConfig | None = None,
        inference_client: Any | None = None,
        repository: Any | None = None,
    ):
        """
        Initialize semantic feature extractor.

        Args:
            config: Feature extraction configuration
            inference_client: SemanticInferenceClient instance
            repository: SemanticRepository for database access (preferred)
        """
        self.config = config or SemanticFeatureConfig()
        self._inference = inference_client
        self._cache: dict[str, tuple[datetime, dict]] = {}

        # Database repository (preferred) or legacy ingesters
        self._repository = repository
        self._news_ingester = None
        self._social_ingester = None

    def _get_repository(self):
        """Lazy-load database repository."""
        if self._repository is None:
            try:
                from py.database.semantic_repository import SemanticRepository
                self._repository = SemanticRepository()
                logger.info("Using database for semantic features")
            except Exception as e:
                logger.warning(f"Failed to initialize repository: {e}")
        return self._repository

    def _get_inference_client(self):
        """Lazy-load inference client."""
        if self._inference is None:
            try:
                from py.semantic.models.inference import SemanticInferenceClient
                self._inference = SemanticInferenceClient(
                    ollama_timeout=self.config.llm_timeout
                )
            except Exception as e:
                logger.warning(f"Failed to initialize inference client: {e}")
                self._inference = None
        return self._inference

    def _get_news_ingester(self):
        """Lazy-load news ingester."""
        if self._news_ingester is None:
            try:
                from py.semantic.ingestion.news_ingester import NewsIngester
                self._news_ingester = NewsIngester()
            except Exception as e:
                logger.warning(f"Failed to initialize news ingester: {e}")
        return self._news_ingester

    def _get_social_ingester(self):
        """Lazy-load social ingester."""
        if self._social_ingester is None:
            try:
                from py.semantic.ingestion.social_ingester import SocialIngester
                self._social_ingester = SocialIngester()
            except Exception as e:
                logger.warning(f"Failed to initialize social ingester: {e}")
        return self._social_ingester

    def extract_features(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        use_cache: bool = True,
        as_of: datetime | None = None,
    ) -> dict[str, float]:
        """
        Extract all semantic features for a game matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number
            use_cache: Whether to use cached results
            as_of: Extract features as of this timestamp (for temporal safety)
                   If None, uses current time (for live predictions)

        Returns:
            Dictionary with 26 semantic features
        """
        cache_key = f"{home_team}_{away_team}_{season}_{week}"

        # Check cache
        if use_cache and cache_key in self._cache:
            cached_time, cached_features = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.config.cache_ttl:
                logger.debug(f"Using cached semantic features for {cache_key}")
                return cached_features

        start_time = time.time()

        # Use current time if as_of not specified
        query_time = as_of or datetime.now()

        # Extract team-level semantic states
        home_state = self._extract_team_state(home_team, as_of=query_time)
        away_state = self._extract_team_state(away_team, as_of=query_time)

        # Build feature dictionary
        features = {}

        # Home team features (prefix: home_)
        features.update(home_state.to_feature_dict("home_"))

        # Away team features (prefix: away_)
        features.update(away_state.to_feature_dict("away_"))

        # Matchup features
        features["sentiment_differential"] = (
            home_state.sentiment_24h - away_state.sentiment_24h
        )
        features["injury_advantage"] = (
            away_state.injury_severity_llm - home_state.injury_severity_llm
        )  # Positive = home advantage
        features["media_attention_ratio"] = (
            home_state.media_buzz_volume / (away_state.media_buzz_volume + 0.01)
        )
        features["narrative_momentum"] = (
            (home_state.sentiment_7d_trend + 1) / 2 -
            (away_state.sentiment_7d_trend + 1) / 2
        )

        # Cache results
        self._cache[cache_key] = (datetime.now(), features)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Extracted {len(features)} semantic features in {elapsed:.1f}ms")

        return features

    def _extract_team_state(
        self,
        team: str,
        as_of: datetime | None = None,
    ) -> TeamSemanticState:
        """
        Extract semantic state for a single team.

        Args:
            team: Team abbreviation
            as_of: Only use data published BEFORE this time (temporal safety)

        Returns:
            TeamSemanticState with all features populated
        """
        query_time = as_of or datetime.now()
        state = TeamSemanticState(team=team, timestamp=query_time)

        # Try database first (preferred for temporal safety)
        repository = self._get_repository()
        if repository:
            try:
                # Get news from database (temporal-safe query)
                news_items = repository.get_recent_news(
                    team=team,
                    before=query_time,
                    hours=self.config.news_lookback_hours,
                    limit=20,
                )

                if news_items:
                    # Count injury mentions
                    state.injury_mentions = sum(
                        1 for n in news_items
                        if n.category == "injury"
                    )

                    # Calculate media buzz (normalized by expected volume)
                    state.media_buzz_volume = min(1.0, len(news_items) / 20.0)

                    # Detect coaching changes
                    state.coaching_change_detected = any(
                        n.category == "coaching" for n in news_items
                    )

                    # Calculate scheme mentions
                    state.scheme_mention_count = sum(
                        1 for n in news_items
                        if "scheme" in n.title.lower() or "playbook" in n.title.lower()
                    )

                    logger.debug(f"Got {len(news_items)} news items for {team} from database")

                # Get sentiment from database
                sentiment = repository.get_team_sentiment(
                    team=team,
                    before=query_time,
                    hours=24,
                )
                if sentiment:
                    state.sentiment_24h = sentiment.sentiment
                    state.sentiment_confidence = sentiment.confidence
                    logger.debug(f"Got sentiment {sentiment.sentiment:.2f} for {team}")

                # Get injuries from database
                injuries = repository.get_team_injuries(team=team, before=query_time)
                if injuries:
                    # Calculate injury severity based on status
                    severity_map = {"Out": 1.0, "IR": 1.0, "Doubtful": 0.8, "Questionable": 0.4}
                    total_severity = sum(
                        severity_map.get(inj.status, 0.2) for inj in injuries[:10]
                    )
                    state.injury_severity_llm = min(1.0, total_severity / 5.0)
                    state.key_player_risk = min(1.0, len(injuries) / 20.0)

                return state

            except Exception as e:
                logger.warning(f"Database query failed for {team}, falling back: {e}")

        # Fallback to legacy live sources (if database unavailable)
        news_ingester = self._get_news_ingester()
        if news_ingester:
            try:
                from py.semantic.ingestion.news_ingester import NewsPriority

                news_items = news_ingester.fetch_team_news(
                    team=team,
                    limit=10,
                    min_priority=NewsPriority.MEDIUM,
                )

                # Count injury mentions
                state.injury_mentions = sum(
                    1 for n in news_items
                    if n.category.value == "injury"
                )

                # Calculate media buzz (normalized by expected volume)
                state.media_buzz_volume = min(1.0, len(news_items) / 10.0)

                # Detect coaching changes
                state.coaching_change_detected = any(
                    n.category.value == "coaching" for n in news_items
                )

                # Process with LLM if available
                if self.config.use_llm_sentiment and news_items:
                    state = self._enrich_with_llm(state, news_items)

            except Exception as e:
                logger.warning(f"Failed to fetch news for {team}: {e}")

        # Get social signals (legacy fallback)
        social_ingester = self._get_social_ingester()
        if social_ingester:
            try:
                tweets = social_ingester.fetch_team_tweets([team], hours=24)

                # Calculate engagement-weighted buzz
                if tweets:
                    total_engagement = sum(t.engagement_score for t in tweets)
                    state.media_buzz_volume = min(
                        1.0,
                        state.media_buzz_volume + total_engagement / 100000
                    )

            except Exception as e:
                logger.warning(f"Failed to fetch social for {team}: {e}")

        return state

    def _enrich_with_llm(
        self,
        state: TeamSemanticState,
        news_items: list,
    ) -> TeamSemanticState:
        """
        Enrich team state with LLM-derived features.

        Args:
            state: Current team state
            news_items: News items to analyze

        Returns:
            Enriched TeamSemanticState
        """
        inference = self._get_inference_client()
        if not inference:
            return state

        try:
            # Combine news text for sentiment
            combined_text = " ".join([
                f"{n.title}. {n.description}" for n in news_items[:5]
            ])

            # Get sentiment
            sentiment_result = inference.analyze_sentiment(combined_text)
            sentiment_map = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}
            base_sentiment = sentiment_map.get(
                sentiment_result.get("sentiment", "neutral"), 0.0
            )
            confidence = sentiment_result.get("confidence", 0.5)

            # Scale by confidence
            state.sentiment_24h = base_sentiment * confidence
            state.sentiment_confidence = confidence

            # Analyze injury severity with NER
            injury_news = [n for n in news_items if n.category.value == "injury"]
            if injury_news:
                injury_text = " ".join([
                    f"{n.title}. {n.description}" for n in injury_news[:3]
                ])
                entities = inference.extract_entities(injury_text)

                # Count player entities in injury context
                players = [
                    e for e in entities.get("entities", [])
                    if e.get("type") == "PLAYER"
                ]
                injuries = [
                    e for e in entities.get("entities", [])
                    if e.get("type") == "INJURY"
                ]

                # Severity based on player count and injury mentions
                if players and injuries:
                    state.injury_severity_llm = min(1.0, len(players) * 0.3)
                    state.key_player_risk = min(1.0, len(players) * 0.2)

        except Exception as e:
            logger.warning(f"LLM enrichment failed for {state.team}: {e}")

        return state

    def get_feature_names(self) -> list[str]:
        """Get list of all semantic feature names."""
        prefixes = ["home_", "away_"]
        team_features = [
            "sentiment_24h", "sentiment_7d_trend", "sentiment_confidence",
            "injury_mentions", "injury_severity_llm", "key_player_risk",
            "media_buzz_volume", "media_buzz_trend", "coaching_change",
            "scheme_mentions", "public_overreaction",
        ]

        names = []
        for prefix in prefixes:
            for feat in team_features:
                names.append(f"{prefix}{feat}")

        # Matchup features
        names.extend([
            "sentiment_differential",
            "injury_advantage",
            "media_attention_ratio",
            "narrative_momentum",
        ])

        return names

    def get_default_features(self) -> dict[str, float]:
        """Get default (neutral) feature values for fallback."""
        return {name: 0.0 for name in self.get_feature_names()}


def test_semantic_features():
    """Test semantic feature extraction."""
    print("=== Testing Semantic Feature Extractor ===")

    extractor = SemanticFeatureExtractor()

    print(f"Feature count: {extractor.FEATURE_COUNT}")
    print(f"Feature names: {len(extractor.get_feature_names())}")
    print()

    print("Extracting features for KC vs BAL...")
    start = time.time()
    features = extractor.extract_features("KC", "BAL", 2025, 16)
    elapsed = (time.time() - start) * 1000
    print(f"Extraction time: {elapsed:.1f}ms")
    print()

    print("Features extracted:")
    for name, value in sorted(features.items()):
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    test_semantic_features()
