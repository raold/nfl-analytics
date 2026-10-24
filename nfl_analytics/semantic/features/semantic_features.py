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
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Injury-specific decay rates (λ per day) based on typical recovery times
# Formula: half_life = ln(2) / λ
INJURY_DECAY_RATES = {
    # Season-ending injuries (very slow decay)
    "ACL": 0.005,  # ~140 day half-life
    "Achilles": 0.006,  # ~115 day half-life
    "Torn": 0.008,  # ~87 day half-life (generic torn)
    # Multi-week injuries
    "Knee": 0.03,  # ~23 day half-life
    "Shoulder": 0.04,  # ~17 day half-life
    "Ribs": 0.035,  # ~20 day half-life
    # 1-2 week injuries
    "Concussion": 0.15,  # ~5 day half-life (protocol-based)
    "Hamstring": 0.08,  # ~9 day half-life
    "Ankle": 0.10,  # ~7 day half-life
    "Calf": 0.09,  # ~8 day half-life
    "Quadriceps": 0.08,  # ~9 day half-life
    "Groin": 0.07,  # ~10 day half-life
    # Minor injuries (fast decay)
    "Rest": 0.50,  # ~1.4 day half-life
    "Illness": 0.40,  # ~1.7 day half-life
    "Personal": 0.60,  # ~1.2 day half-life
    # Default for unknown injuries
    "Default": 0.05,  # ~14 day half-life
}


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

    # =========================================================================
    # TEMPORAL DYNAMICS FEATURES (ODE-inspired)
    # =========================================================================
    # Injury dynamics: dI/dt = -λ*I + S(t) where λ varies by injury type
    injury_impact_decay: float = 0.0  # Σ(severity × e^(-λ×days)) integrated
    injury_velocity: float = 0.0  # d(impact)/dt - rate of injury change
    recovery_momentum: float = 0.0  # Players returning from injury

    # Sentiment dynamics: dσ/dt = α*(news) - β*σ (damped oscillator)
    sentiment_velocity: float = 0.0  # dσ/dt - first derivative
    sentiment_acceleration: float = 0.0  # d²σ/dt² - second derivative

    # Media attention decay
    media_half_life: float = 0.0  # Days since peak media coverage

    def to_feature_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to feature dictionary with optional prefix."""
        return {
            # Original 11 features
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
            # Temporal dynamics (6 new features per team = 12 total)
            f"{prefix}injury_impact_decay": self.injury_impact_decay,
            f"{prefix}injury_velocity": self.injury_velocity,
            f"{prefix}recovery_momentum": self.recovery_momentum,
            f"{prefix}sentiment_velocity": self.sentiment_velocity,
            f"{prefix}sentiment_acceleration": self.sentiment_acceleration,
            f"{prefix}media_half_life": self.media_half_life,
        }


class SemanticFeatureExtractor:
    """
    Extract semantic features for NFL game matchups.

    Combines news sentiment, social media signals, and LLM-derived
    insights into features for the betting model ensemble.

    Features per team (17 each, 34 total + 4 matchup = 38 total):

    Original features (11):
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

    Temporal dynamics features (6, ODE-inspired):
    - injury_impact_decay: Σ(severity × e^(-λ×days)) with injury-specific λ
    - injury_velocity: d(impact)/dt, rate of injury situation change
    - recovery_momentum: Players returning from injury
    - sentiment_velocity: dσ/dt, first derivative of sentiment
    - sentiment_acceleration: d²σ/dt², second derivative of sentiment
    - media_half_life: Days since peak media attention

    Matchup features (4):
    - sentiment_differential: home - away sentiment
    - injury_advantage: home vs away injury severity
    - media_attention_ratio: relative media coverage
    - narrative_momentum: combined narrative direction
    """

    FEATURE_COUNT = 38  # 17 per team (11 original + 6 temporal) + 4 matchup

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

                self._inference = SemanticInferenceClient(ollama_timeout=self.config.llm_timeout)
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

    # =========================================================================
    # TEMPORAL DYNAMICS COMPUTATION (ODE-Inspired)
    # =========================================================================

    def _get_decay_rate(self, injury_type: str) -> float:
        """
        Get injury-specific decay rate (λ) for exponential decay.

        Args:
            injury_type: Injury type string from database

        Returns:
            Decay rate λ (per day), where half-life = ln(2)/λ
        """
        if not injury_type:
            return INJURY_DECAY_RATES["Default"]

        injury_upper = injury_type.upper()
        for key, rate in INJURY_DECAY_RATES.items():
            if key.upper() in injury_upper:
                return rate

        return INJURY_DECAY_RATES["Default"]

    def _compute_injury_decay(
        self,
        injuries: list,
        as_of: datetime,
    ) -> tuple[float, float, float]:
        """
        Compute exponentially decayed injury impact using first-order ODE.

        Solves: dI/dt = -λ*I + S(t)
        Where:
          - I(t) = total injury impact at time t
          - λ = injury-specific healing rate
          - S(t) = shock function (new injuries)

        The solution is: I(t) = Σ(severity × e^(-λ×days_since))

        Args:
            injuries: List of InjuryItem objects with severity_score and updated_at
            as_of: Reference time for decay calculation

        Returns:
            Tuple of (total_decay_impact, injury_velocity, recovery_momentum)
            - total_decay_impact: Σ(severity × e^(-λ×days))
            - injury_velocity: d(impact)/dt using 7-day lookback
            - recovery_momentum: Rate of players recovering (returning)
        """
        if not injuries:
            return (0.0, 0.0, 0.0)

        total_impact = 0.0
        impact_7d_ago = 0.0
        recovering_count = 0

        for injury in injuries:
            # Get severity (default to 0.5 if not scored)
            severity = injury.severity_score if injury.severity_score is not None else 0.5

            # Calculate days since injury update
            if injury.updated_at:
                delta = as_of - injury.updated_at
                days_old = max(0, delta.total_seconds() / 86400)
            else:
                days_old = 7  # Default if no timestamp

            # Get decay rate for this injury type
            λ = self._get_decay_rate(injury.injury_type)

            # Apply exponential decay: I(t) = I₀ × e^(-λt)
            decayed_impact = severity * math.exp(-λ * days_old)
            total_impact += decayed_impact

            # Calculate what impact would be 7 days ago for velocity
            days_old_7 = days_old + 7
            impact_7d_ago += severity * math.exp(-λ * days_old_7)

            # Count players in recovery status (Questionable/Probable moving to Active)
            if injury.status in ("Questionable", "Probable"):
                recovering_count += 1

        # Normalize impacts
        total_impact = min(1.0, total_impact / 5.0)  # Cap at 5 severe injuries
        impact_7d_ago = min(1.0, impact_7d_ago / 5.0)

        # Velocity: d(impact)/dt ≈ (current - 7d_ago) / 7
        injury_velocity = (total_impact - impact_7d_ago) / 7.0

        # Recovery momentum: fraction of roster potentially returning
        recovery_momentum = min(1.0, recovering_count / 10.0)

        return (
            round(total_impact, 4),
            round(injury_velocity, 4),
            round(recovery_momentum, 4),
        )

    def _compute_sentiment_derivatives(
        self,
        team: str,
        as_of: datetime,
    ) -> tuple[float, float, float]:
        """
        Compute sentiment velocity and acceleration using finite differences.

        Models sentiment as damped oscillator: dσ/dt = α*(news) - β*σ
        Uses central difference for derivatives:
          - velocity: dσ/dt ≈ (σ(t+h) - σ(t-h)) / 2h
          - acceleration: d²σ/dt² ≈ (σ(t+h) - 2σ(t) + σ(t-h)) / h²

        Args:
            team: Team abbreviation
            as_of: Reference time for calculation

        Returns:
            Tuple of (sentiment_velocity, sentiment_acceleration, media_half_life)
            - sentiment_velocity: dσ/dt (rate of sentiment change)
            - sentiment_acceleration: d²σ/dt² (momentum of change)
            - media_half_life: Days since peak media attention
        """
        repository = self._get_repository()
        if not repository:
            return (0.0, 0.0, 0.0)

        try:
            # Get daily sentiment values for past 7 days
            timeseries = repository.get_sentiment_timeseries(
                team=team,
                before=as_of,
                points=7,
            )

            if len(timeseries) < 3:
                # Not enough data for derivatives
                return (0.0, 0.0, 0.0)

            # Sort by date (oldest to newest for derivative calculation)
            timeseries_sorted = sorted(timeseries, key=lambda x: x[0])
            sentiments = [s for _, s in timeseries_sorted]

            # Central difference for velocity (use most recent 3 points)
            # dσ/dt ≈ (σ(t) - σ(t-2h)) / 2h where h = 1 day
            if len(sentiments) >= 3:
                velocity = (sentiments[-1] - sentiments[-3]) / 2.0
            else:
                velocity = sentiments[-1] - sentiments[0]

            # Central difference for acceleration
            # d²σ/dt² ≈ (σ(t) - 2σ(t-h) + σ(t-2h)) / h²
            if len(sentiments) >= 3:
                acceleration = sentiments[-1] - 2 * sentiments[-2] + sentiments[-3]
            else:
                acceleration = 0.0

            # Media half-life: find peak and calculate days since
            max_idx = sentiments.index(max(sentiments, key=abs))
            days_since_peak = len(sentiments) - 1 - max_idx

            # Normalize to 0-1 scale (7 days = 1.0)
            media_half_life = min(1.0, days_since_peak / 7.0)

            return (
                round(velocity, 4),
                round(acceleration, 4),
                round(media_half_life, 4),
            )

        except Exception as e:
            logger.warning(f"Failed to compute sentiment derivatives for {team}: {e}")
            return (0.0, 0.0, 0.0)

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
            Dictionary with 38 semantic features (17 per team + 4 matchup)
        """
        cache_key = f"{home_team}_{away_team}_{season}_{week}"

        # Check cache
        if use_cache and cache_key in self._cache:
            cached_time, cached_features = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).seconds < self.config.cache_ttl:
                logger.debug(f"Using cached semantic features for {cache_key}")
                return cached_features

        start_time = time.time()

        # Use current time if as_of not specified (timezone-aware for DB compat)
        query_time = as_of or datetime.now(timezone.utc)

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
        features["sentiment_differential"] = home_state.sentiment_24h - away_state.sentiment_24h
        features["injury_advantage"] = (
            away_state.injury_severity_llm - home_state.injury_severity_llm
        )  # Positive = home advantage
        features["media_attention_ratio"] = home_state.media_buzz_volume / (
            away_state.media_buzz_volume + 0.01
        )
        features["narrative_momentum"] = (home_state.sentiment_7d_trend + 1) / 2 - (
            away_state.sentiment_7d_trend + 1
        ) / 2

        # Cache results
        self._cache[cache_key] = (datetime.now(timezone.utc), features)

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
        query_time = as_of or datetime.now(timezone.utc)
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
                    state.injury_mentions = sum(1 for n in news_items if n.category == "injury")

                    # Calculate media buzz (normalized by expected volume)
                    state.media_buzz_volume = min(1.0, len(news_items) / 20.0)

                    # Detect coaching changes
                    state.coaching_change_detected = any(
                        n.category == "coaching" for n in news_items
                    )

                    # Calculate scheme mentions
                    state.scheme_mention_count = sum(
                        1
                        for n in news_items
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
                    total_severity = sum(severity_map.get(inj.status, 0.2) for inj in injuries[:10])
                    state.injury_severity_llm = min(1.0, total_severity / 5.0)
                    state.key_player_risk = min(1.0, len(injuries) / 20.0)

                # =========================================================
                # TEMPORAL DYNAMICS (ODE-inspired features)
                # =========================================================

                # Get injury history for temporal decay calculation
                injury_history = repository.get_injury_history(
                    team=team,
                    before=query_time,
                    days=14,  # 2-week lookback for decay
                )

                # Compute injury decay, velocity, and recovery momentum
                (
                    state.injury_impact_decay,
                    state.injury_velocity,
                    state.recovery_momentum,
                ) = self._compute_injury_decay(injury_history, query_time)

                # Compute sentiment derivatives (velocity, acceleration)
                (
                    state.sentiment_velocity,
                    state.sentiment_acceleration,
                    state.media_half_life,
                ) = self._compute_sentiment_derivatives(team, query_time)

                logger.debug(
                    f"Temporal dynamics for {team}: "
                    f"injury_decay={state.injury_impact_decay:.3f}, "
                    f"sentiment_vel={state.sentiment_velocity:.3f}"
                )

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
                state.injury_mentions = sum(1 for n in news_items if n.category.value == "injury")

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
                        1.0, state.media_buzz_volume + total_engagement / 100000
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
            combined_text = " ".join([f"{n.title}. {n.description}" for n in news_items[:5]])

            # Get sentiment
            sentiment_result = inference.analyze_sentiment(combined_text)
            sentiment_map = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}
            base_sentiment = sentiment_map.get(sentiment_result.get("sentiment", "neutral"), 0.0)
            confidence = sentiment_result.get("confidence", 0.5)

            # Scale by confidence
            state.sentiment_24h = base_sentiment * confidence
            state.sentiment_confidence = confidence

            # Analyze injury severity with NER
            injury_news = [n for n in news_items if n.category.value == "injury"]
            if injury_news:
                injury_text = " ".join([f"{n.title}. {n.description}" for n in injury_news[:3]])
                entities = inference.extract_entities(injury_text)

                # Count player entities in injury context
                players = [e for e in entities.get("entities", []) if e.get("type") == "PLAYER"]
                injuries = [e for e in entities.get("entities", []) if e.get("type") == "INJURY"]

                # Severity based on player count and injury mentions
                if players and injuries:
                    state.injury_severity_llm = min(1.0, len(players) * 0.3)
                    state.key_player_risk = min(1.0, len(players) * 0.2)

        except Exception as e:
            logger.warning(f"LLM enrichment failed for {state.team}: {e}")

        return state

    def get_feature_names(self) -> list[str]:
        """Get list of all semantic feature names (38 total)."""
        prefixes = ["home_", "away_"]

        # Original 11 features per team
        team_features = [
            "sentiment_24h",
            "sentiment_7d_trend",
            "sentiment_confidence",
            "injury_mentions",
            "injury_severity_llm",
            "key_player_risk",
            "media_buzz_volume",
            "media_buzz_trend",
            "coaching_change",
            "scheme_mentions",
            "public_overreaction",
        ]

        # Temporal dynamics features (6 per team, ODE-inspired)
        temporal_features = [
            "injury_impact_decay",  # Σ(severity × e^(-λ×days))
            "injury_velocity",  # d(impact)/dt
            "recovery_momentum",  # Players returning
            "sentiment_velocity",  # dσ/dt
            "sentiment_acceleration",  # d²σ/dt²
            "media_half_life",  # Days since peak attention
        ]

        names = []
        for prefix in prefixes:
            for feat in team_features + temporal_features:
                names.append(f"{prefix}{feat}")

        # Matchup features (4)
        names.extend(
            [
                "sentiment_differential",
                "injury_advantage",
                "media_attention_ratio",
                "narrative_momentum",
            ]
        )

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
