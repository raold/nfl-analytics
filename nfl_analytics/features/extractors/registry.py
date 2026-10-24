"""Feature Registry for orchestrating multiple extractors.

This module provides the FeatureRegistry class that manages multiple
FeatureExtractorStrategy implementations and coordinates their execution.

Design Goals:
1. Centralized registration of all feature extractors
2. Parallel execution where possible
3. Graceful failure handling (non-critical extractors can fail)
4. Consistent prefixing and feature naming
5. Performance monitoring and logging

Created: December 13, 2025
Phase 2.2 of code quality remediation plan.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import psycopg2

from nfl_analytics.features.extractors.base import (
    ExtractionLevel,
    FeatureExtractionError,
    FeatureExtractionResult,
    FeatureExtractorStrategy,
    FeaturePriority,
    GameLevelExtractorStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionSummary:
    """Summary of a full feature extraction run.

    Attributes:
        features: Combined features from all extractors
        total_features: Total number of features extracted
        extraction_time_ms: Total time for all extractions
        extractor_results: Results from each individual extractor
        failed_extractors: Names of extractors that failed
        success: True if all critical extractors succeeded
    """

    features: dict[str, float] = field(default_factory=dict)
    total_features: int = 0
    extraction_time_ms: float = 0.0
    extractor_results: list[FeatureExtractionResult] = field(default_factory=list)
    failed_extractors: list[str] = field(default_factory=list)
    success: bool = True


class FeatureRegistry:
    """Registry for managing and orchestrating feature extractors.

    The registry provides:
    1. Registration of extractors by name
    2. Ordered execution based on priority
    3. Parallel execution of independent extractors
    4. Aggregation of features with consistent naming

    Example:
        registry = FeatureRegistry()

        # Register extractors
        registry.register(OffensiveFeatureExtractor())
        registry.register(DefensiveFeatureExtractor())
        registry.register(NextGenStatsExtractor())

        # Extract all features for a game
        import datetime
        current_season = datetime.date.today().year
        summary = registry.extract_all(
            home_team='KC',
            away_team='BUF',
            season=current_season,
            week=10
        )

        print(f"Extracted {summary.total_features} features")
        print(summary.features)
    """

    def __init__(
        self,
        db_config: Optional[dict] = None,
        parallel: bool = True,
        max_workers: int = 4,
        fail_on_critical: bool = True,
    ):
        """Initialize the feature registry.

        Args:
            db_config: Database connection parameters
            parallel: Whether to run independent extractors in parallel
            max_workers: Maximum parallel workers for extraction
            fail_on_critical: Whether to raise error if critical extractor fails
        """
        self._extractors: dict[str, FeatureExtractorStrategy] = {}
        self._extraction_order: list[str] = []
        self.db_config = db_config or self._default_db_config()
        self.parallel = parallel
        self.max_workers = max_workers
        self.fail_on_critical = fail_on_critical

    def _default_db_config(self) -> dict:
        """Get default database configuration."""
        import os

        return {
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": int(os.environ.get("POSTGRES_PORT", "5544")),
            "dbname": os.environ.get("POSTGRES_DB", "devdb01"),
            "user": os.environ.get("POSTGRES_USER", "dro"),
            "password": os.environ.get("POSTGRES_PASSWORD", "sicillionbillions"),
        }

    def register(self, extractor: FeatureExtractorStrategy) -> None:
        """Register a feature extractor.

        Args:
            extractor: The extractor to register

        Raises:
            ValueError: If an extractor with the same name is already registered
        """
        name = extractor.name
        if name in self._extractors:
            raise ValueError(f"Extractor '{name}' is already registered")

        self._extractors[name] = extractor
        self._update_extraction_order()
        logger.debug(f"Registered extractor: {name} (priority: {extractor.priority.name})")

    def unregister(self, name: str) -> None:
        """Remove an extractor from the registry.

        Args:
            name: Name of the extractor to remove
        """
        if name in self._extractors:
            del self._extractors[name]
            self._update_extraction_order()
            logger.debug(f"Unregistered extractor: {name}")

    def _update_extraction_order(self) -> None:
        """Update the extraction order based on priorities."""
        self._extraction_order = sorted(
            self._extractors.keys(), key=lambda n: self._extractors[n].priority.value
        )

    def get_extractor(self, name: str) -> Optional[FeatureExtractorStrategy]:
        """Get an extractor by name."""
        return self._extractors.get(name)

    def list_extractors(self) -> list[str]:
        """List all registered extractor names in extraction order."""
        return list(self._extraction_order)

    def get_all_feature_names(self, include_home_away: bool = True) -> list[str]:
        """Get all feature names that will be produced.

        Args:
            include_home_away: Whether to include home/away prefixes for team-level features

        Returns:
            List of all feature names
        """
        all_names = []

        for extractor in self._extractors.values():
            prefixed_names = extractor.get_prefixed_feature_names()

            if extractor.level == ExtractionLevel.TEAM and include_home_away:
                # Team-level features get home_ and away_ prefixes
                all_names.extend([f"home_{n}" for n in prefixed_names])
                all_names.extend([f"away_{n}" for n in prefixed_names])
            else:
                # Game-level features don't get additional prefixes
                all_names.extend(prefixed_names)

        return all_names

    def extract_all(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
    ) -> ExtractionSummary:
        """Extract all features for a game matchup.

        This is the main orchestration method. It:
        1. Groups extractors by level (team vs game)
        2. Runs team-level extractors for both teams
        3. Runs game-level extractors once
        4. Combines all features with appropriate prefixes
        5. Handles errors gracefully

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Target season year
            week: Target week number
            conn: Optional shared database connection

        Returns:
            ExtractionSummary with all features and metadata
        """
        start_time = time.time()
        summary = ExtractionSummary()

        # Separate extractors by level
        team_extractors = []
        game_extractors = []

        for name in self._extraction_order:
            extractor = self._extractors[name]
            if extractor.level == ExtractionLevel.TEAM:
                team_extractors.append(extractor)
            else:
                game_extractors.append(extractor)

        # Create connection if not provided
        own_conn = False
        if conn is None:
            conn = psycopg2.connect(**self.db_config)
            own_conn = True

        try:
            # Extract team-level features
            home_features, home_results = self._extract_team_features(
                team_extractors, home_team, season, week, conn, away_team
            )
            away_features, away_results = self._extract_team_features(
                team_extractors, away_team, season, week, conn, home_team
            )

            # Add with home/away prefixes
            for name, value in home_features.items():
                summary.features[f"home_{name}"] = value
            for name, value in away_features.items():
                summary.features[f"away_{name}"] = value

            summary.extractor_results.extend(home_results)
            summary.extractor_results.extend(away_results)

            # Extract game-level features
            game_features, game_results = self._extract_game_features(
                game_extractors, home_team, away_team, season, week, conn
            )
            summary.features.update(game_features)
            summary.extractor_results.extend(game_results)

            # Check for failures
            for result in summary.extractor_results:
                if not result.success:
                    summary.failed_extractors.append(result.extractor_name)
                    extractor = self._extractors.get(result.extractor_name)
                    if extractor and extractor.priority == FeaturePriority.CRITICAL:
                        summary.success = False
                        if self.fail_on_critical:
                            raise FeatureExtractionError(
                                result.extractor_name,
                                f"Critical extractor failed: {result.error_message}",
                            )

        finally:
            if own_conn and conn:
                conn.close()

        summary.total_features = len(summary.features)
        summary.extraction_time_ms = (time.time() - start_time) * 1000

        if summary.failed_extractors:
            logger.warning(
                f"Feature extraction completed with {len(summary.failed_extractors)} failures: "
                f"{summary.failed_extractors}"
            )
        else:
            logger.debug(
                f"Feature extraction completed: {summary.total_features} features "
                f"in {summary.extraction_time_ms:.1f}ms"
            )

        return summary

    def _extract_team_features(
        self,
        extractors: list[FeatureExtractorStrategy],
        team: str,
        season: int,
        week: int,
        conn: psycopg2.extensions.connection,
        opponent: str,
    ) -> tuple[dict[str, float], list[FeatureExtractionResult]]:
        """Extract features from all team-level extractors for one team.

        Args:
            extractors: List of team-level extractors
            team: Team to extract features for
            season: Target season
            week: Target week
            conn: Database connection
            opponent: Opponent team (for extractors that need it)

        Returns:
            Tuple of (combined features dict, list of results)
        """
        features = {}
        results = []

        if self.parallel and len(extractors) > 1:
            # Parallel extraction
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_extractor = {
                    executor.submit(
                        extractor.safe_extract,
                        team,
                        season,
                        week,
                        conn,
                        opponent if extractor.requires_opponent else None,
                    ): extractor
                    for extractor in extractors
                }

                for future in as_completed(future_to_extractor):
                    extractor = future_to_extractor[future]
                    result = future.result()
                    results.append(result)

                    # Add prefixed features
                    prefix = extractor.prefix
                    if prefix and not prefix.endswith("_"):
                        prefix = f"{prefix}_"
                    for name, value in result.features.items():
                        features[f"{prefix}{name}" if prefix else name] = value
        else:
            # Sequential extraction
            for extractor in extractors:
                result = extractor.safe_extract(
                    team, season, week, conn, opponent if extractor.requires_opponent else None
                )
                results.append(result)

                # Add prefixed features
                prefix = extractor.prefix
                if prefix and not prefix.endswith("_"):
                    prefix = f"{prefix}_"
                for name, value in result.features.items():
                    features[f"{prefix}{name}" if prefix else name] = value

        return features, results

    def _extract_game_features(
        self,
        extractors: list[FeatureExtractorStrategy],
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        conn: psycopg2.extensions.connection,
    ) -> tuple[dict[str, float], list[FeatureExtractionResult]]:
        """Extract features from all game-level extractors.

        Args:
            extractors: List of game-level extractors
            home_team: Home team
            away_team: Away team
            season: Target season
            week: Target week
            conn: Database connection

        Returns:
            Tuple of (combined features dict, list of results)
        """
        features = {}
        results = []

        for extractor in extractors:
            if isinstance(extractor, GameLevelExtractorStrategy):
                result = extractor.safe_extract(home_team, season, week, conn, away_team)
            else:
                # Treat as game-level by passing opponent
                result = extractor.safe_extract(home_team, season, week, conn, away_team)

            results.append(result)

            # Add prefixed features
            prefix = extractor.prefix
            if prefix and not prefix.endswith("_"):
                prefix = f"{prefix}_"
            for name, value in result.features.items():
                features[f"{prefix}{name}" if prefix else name] = value

        return features, results


# Convenience function for creating a fully-loaded registry
def create_default_registry(
    db_config: Optional[dict] = None,
    parallel: bool = True,
) -> FeatureRegistry:
    """Create a registry with all standard extractors registered.

    This is a convenience function that sets up a registry with
    all the standard NFL Analytics feature extractors.

    Args:
        db_config: Optional database configuration
        parallel: Whether to enable parallel extraction

    Returns:
        Configured FeatureRegistry with all 5 standard extractors:
        - OffensiveFeatureExtractor (27 features, CRITICAL priority)
        - DefensiveFeatureExtractor (27 features, CRITICAL priority)
        - NextGenFeatureExtractor (23 features, HIGH priority)
        - QBRFeatureExtractor (10 features, HIGH priority)
        - OpponentAdjustedFeatureExtractor (7 features, MEDIUM priority, requires opponent)

    Example:
        import datetime
        current_season = datetime.date.today().year
        registry = create_default_registry()
        summary = registry.extract_all('KC', 'BUF', current_season, 10)
        print(f"Extracted {summary.total_features} features")
    """
    # Import here to avoid circular imports
    from nfl_analytics.features.extractors.offensive import OffensiveFeatureExtractor
    from nfl_analytics.features.extractors.defensive import DefensiveFeatureExtractor
    from nfl_analytics.features.extractors.nextgen import NextGenFeatureExtractor
    from nfl_analytics.features.extractors.qbr import QBRFeatureExtractor
    from nfl_analytics.features.extractors.opponent_adjusted import OpponentAdjustedFeatureExtractor

    registry = FeatureRegistry(db_config=db_config, parallel=parallel)

    # Register all standard extractors
    # Execution order determined by priority: CRITICAL > HIGH > MEDIUM
    registry.register(OffensiveFeatureExtractor(db_config=db_config))
    registry.register(DefensiveFeatureExtractor(db_config=db_config))
    registry.register(NextGenFeatureExtractor(db_config=db_config))
    registry.register(QBRFeatureExtractor(db_config=db_config))
    registry.register(OpponentAdjustedFeatureExtractor(db_config=db_config))

    logger.info(
        f"Created default registry with {len(registry.list_extractors())} extractors "
        f"({sum(1 for _ in registry.get_all_feature_names())} total features)"
    )

    return registry
