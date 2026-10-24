"""Base class for feature extraction strategies.

This module defines the Strategy pattern interface for all feature extractors
in the NFL Analytics pipeline. By implementing this interface, extractors can
be registered with the FeatureRegistry and orchestrated uniformly.

Design Principles:
1. Single Responsibility: Each extractor handles one category of features
2. Open/Closed: New extractors can be added without modifying existing code
3. Liskov Substitution: All extractors are interchangeable through the interface
4. Interface Segregation: Minimal interface with optional hooks

Created: December 13, 2025
Phase 2.1 of code quality remediation plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import psycopg2


class ExtractionLevel(Enum):
    """Defines the scope of feature extraction."""

    TEAM = "team"  # Features for a single team (home or away)
    GAME = "game"  # Features for a game matchup (requires both teams)


class FeaturePriority(Enum):
    """Priority level for feature extraction.

    Higher priority extractors run first and their failure can be critical.
    Lower priority extractors can fail gracefully without blocking.
    """

    CRITICAL = 1  # Must succeed (base offensive/defensive)
    HIGH = 2  # Should succeed (NGS, QBR)
    MEDIUM = 3  # Nice to have (synthetic features)
    LOW = 4  # Optional (experimental features)


@dataclass
class FeatureExtractionResult:
    """Result of a feature extraction operation.

    Attributes:
        features: Dictionary of extracted feature names to values
        success: Whether extraction completed successfully
        error_message: Description of error if extraction failed
        extractor_name: Name of the extractor that produced this result
        extraction_time_ms: Time taken to extract features in milliseconds
        feature_count: Number of features extracted
    """

    features: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    extractor_name: str = ""
    extraction_time_ms: float = 0.0
    feature_count: int = 0

    def __post_init__(self):
        self.feature_count = len(self.features)


class FeatureExtractionError(Exception):
    """Raised when feature extraction fails and cannot recover."""

    def __init__(
        self, extractor_name: str, message: str, original_error: Optional[Exception] = None
    ):
        self.extractor_name = extractor_name
        self.message = message
        self.original_error = original_error
        super().__init__(f"[{extractor_name}] {message}")


class FeatureExtractorStrategy(ABC):
    """Abstract base class for all feature extractors.

    This interface defines the contract that all feature extractors must fulfill.
    Implementing this interface allows extractors to be:
    1. Registered with the FeatureRegistry
    2. Orchestrated uniformly by GameFeatureExtractor
    3. Tested independently with consistent patterns

    Example Implementation:
        class MyFeatureExtractor(FeatureExtractorStrategy):
            @property
            def name(self) -> str:
                return "my_features"

            @property
            def prefix(self) -> str:
                return "mf"  # Features will be: mf_feature1, mf_feature2, etc.

            def extract(self, home_team, away_team, season, week, conn=None) -> dict:
                # Extract and return features
                return {"feature1": 0.5, "feature2": 0.8}

            def get_feature_names(self) -> list[str]:
                return ["feature1", "feature2"]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this extractor.

        Returns:
            String name like "offensive_features" or "nextgen_stats"
        """
        pass

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Prefix to prepend to all feature names.

        This ensures feature names don't collide across extractors.
        For team-level extractors, prefixes like 'off_', 'def_', 'ngs_' are typical.
        For game-level extractors, no prefix may be needed.

        Returns:
            String prefix like "off_" or "ngs_" or ""
        """
        pass

    @property
    def level(self) -> ExtractionLevel:
        """Whether this extractor operates on team or game level.

        Team-level extractors are called twice (home and away).
        Game-level extractors are called once with both teams.

        Returns:
            ExtractionLevel.TEAM or ExtractionLevel.GAME
        """
        return ExtractionLevel.TEAM

    @property
    def priority(self) -> FeaturePriority:
        """Priority of this extractor.

        Higher priority extractors run first. Critical extractors must succeed.

        Returns:
            FeaturePriority enum value
        """
        return FeaturePriority.MEDIUM

    @property
    def requires_opponent(self) -> bool:
        """Whether this extractor needs the opponent team for context.

        Some extractors (e.g., opponent-adjusted features) need to know
        who the opponent is even for team-level extraction.

        Returns:
            True if opponent team is needed
        """
        return False

    @abstractmethod
    def extract(
        self,
        team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
        opponent: Optional[str] = None,
    ) -> dict[str, float]:
        """Extract features for a team in a specific game context.

        This is the main extraction method. Implementations should:
        1. Query necessary data from the database
        2. Compute feature values
        3. Return features WITHOUT the prefix (registry adds prefix)

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Target season year
            week: Target week number
            conn: Optional database connection (will create if not provided)
            opponent: Optional opponent team (required if requires_opponent=True)

        Returns:
            Dictionary mapping feature names to float values.
            Feature names should NOT include the prefix.

        Raises:
            FeatureExtractionError: If extraction fails critically
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of all feature names this extractor produces.

        Feature names should NOT include the prefix.

        Returns:
            List of feature names like ["epa_per_play", "success_rate"]
        """
        pass

    def get_prefixed_feature_names(self) -> list[str]:
        """Return feature names with prefix applied.

        Returns:
            List of prefixed names like ["off_epa_per_play", "off_success_rate"]
        """
        prefix = self.prefix
        if prefix and not prefix.endswith("_"):
            prefix = f"{prefix}_"
        return [f"{prefix}{name}" if prefix else name for name in self.get_feature_names()]

    def get_default_values(self) -> dict[str, float]:
        """Return default values for all features.

        Used when extraction fails and we need fallback values.
        Default implementation returns 0.0 for all features.

        Returns:
            Dictionary mapping feature names to default values
        """
        return {name: 0.0 for name in self.get_feature_names()}

    def validate_inputs(self, team: str, season: int, week: int) -> None:
        """Validate extraction inputs.

        Raises:
            ValueError: If inputs are invalid
        """
        if not team or not isinstance(team, str):
            raise ValueError(f"Invalid team: {team}")
        if not isinstance(season, int) or season < 1999 or season > 2100:
            raise ValueError(f"Invalid season: {season}")
        if not isinstance(week, int) or week < 1 or week > 22:
            raise ValueError(f"Invalid week: {week}")

    def safe_extract(
        self,
        team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
        opponent: Optional[str] = None,
    ) -> FeatureExtractionResult:
        """Extract features with error handling and timing.

        This is a wrapper around extract() that provides:
        1. Input validation
        2. Error handling (returns defaults on failure)
        3. Timing metrics
        4. Structured result object

        Args:
            Same as extract()

        Returns:
            FeatureExtractionResult with features and metadata
        """
        import time

        start_time = time.time()
        result = FeatureExtractionResult(extractor_name=self.name)

        try:
            self.validate_inputs(team, season, week)
            features = self.extract(team, season, week, conn, opponent)
            result.features = features
            result.success = True
        except FeatureExtractionError as e:
            result.success = False
            result.error_message = str(e)
            # Use default values on failure
            result.features = self.get_default_values()
        except Exception as e:
            result.success = False
            result.error_message = f"Unexpected error: {e}"
            result.features = self.get_default_values()

        result.extraction_time_ms = (time.time() - start_time) * 1000
        result.feature_count = len(result.features)
        return result


class GameLevelExtractorStrategy(FeatureExtractorStrategy):
    """Base class for extractors that operate at the game level.

    Game-level extractors receive both teams and produce features
    that are inherently about the matchup (not individual teams).

    Examples:
    - Offensive mismatch (home offense vs away defense)
    - Primetime game indicators
    - Rest differential
    """

    @property
    def level(self) -> ExtractionLevel:
        return ExtractionLevel.GAME

    @abstractmethod
    def extract_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
    ) -> dict[str, float]:
        """Extract features for a game matchup.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Target season year
            week: Target week number
            conn: Optional database connection

        Returns:
            Dictionary mapping feature names to values
        """
        pass

    def extract(
        self,
        team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
        opponent: Optional[str] = None,
    ) -> dict[str, float]:
        """Adapter to make game-level extractors work with team interface.

        For game-level extractors, 'team' is treated as home_team and
        'opponent' is required as away_team.
        """
        if opponent is None:
            raise FeatureExtractionError(self.name, "Game-level extractors require opponent team")
        return self.extract_game(team, opponent, season, week, conn)
