"""Opponent-Adjusted Feature Extractor Strategy wrapper.

This module wraps the existing OpponentAdjustedFeatures class in the
FeatureExtractorStrategy interface, allowing it to be registered
with the FeatureRegistry for unified orchestration.

Created: December 13, 2025
Phase 2.3 of code quality remediation plan.
"""

from __future__ import annotations

from typing import Optional

import psycopg2

from py.features.extractors.base import (
    FeatureExtractorStrategy,
    FeaturePriority,
)
from py.features.opponent_adjusted_features import OpponentAdjustedFeatures


class OpponentAdjustedFeatureExtractor(FeatureExtractorStrategy):
    """Strategy wrapper for OpponentAdjustedFeatures.

    This extractor produces 7 opponent-adjusted EPA features that
    account for opponent defensive strength.

    Why this matters:
    - Raw EPA doesn't tell the full story
    - +0.20 EPA vs Panthers (weak) is less impressive than +0.15 vs Ravens (elite)
    - DVOA-like normalization provides comparable metrics across matchups

    Features include:
    - Adjusted EPA (3): overall, pass, rush (adjusted for opponent defense)
    - DVOA approximation (4): offensive, pass, rush, defensive

    This extractor requires the opponent team to compute adjustments.

    Example:
        extractor = OpponentAdjustedFeatureExtractor()
        features = extractor.extract('KC', 2024, 10, opponent='BAL')
        # Returns: {'off_epa_opponent_adjusted': 0.12, 'dvoa_off_approx': 15.5, ...}
    """

    # Feature names produced by this extractor
    FEATURE_NAMES = [
        "off_epa_opponent_adjusted",
        "pass_epa_opponent_adjusted",
        "rush_epa_opponent_adjusted",
        "dvoa_off_approx",
        "dvoa_pass_approx",
        "dvoa_rush_approx",
        "dvoa_def_approx",
    ]

    def __init__(self, db_config: Optional[dict] = None):
        """Initialize the opponent-adjusted feature extractor.

        Args:
            db_config: Database configuration dictionary
        """
        self._db_config = db_config
        self._extractor: Optional[OpponentAdjustedFeatures] = None

    def _get_extractor(self) -> OpponentAdjustedFeatures:
        """Lazy initialization of the underlying extractor."""
        if self._extractor is None:
            if self._db_config:
                self._extractor = OpponentAdjustedFeatures(db_config=self._db_config)
            else:
                self._extractor = OpponentAdjustedFeatures()
        return self._extractor

    @property
    def name(self) -> str:
        """Human-readable name for this extractor."""
        return "opponent_adjusted_features"

    @property
    def prefix(self) -> str:
        """Prefix for feature names.

        Empty because feature names already include descriptive prefixes
        (off_epa_, dvoa_).
        """
        return ""

    @property
    def priority(self) -> FeaturePriority:
        """Extraction priority.

        MEDIUM: Opponent-adjusted features are valuable but not critical.
        Model can function with raw EPA features if these fail.
        """
        return FeaturePriority.MEDIUM

    @property
    def requires_opponent(self) -> bool:
        """This extractor requires the opponent team.

        Opponent-adjusted features need to know who the opponent is
        to compute defensive adjustments.
        """
        return True

    def extract(
        self,
        team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
        opponent: Optional[str] = None,
    ) -> dict[str, float]:
        """Extract opponent-adjusted features for a team.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Target season year
            week: Target week number
            conn: Optional database connection (not used - extractor manages own)
            opponent: Opponent team abbreviation (REQUIRED)

        Returns:
            Dictionary of 7 opponent-adjusted features

        Raises:
            FeatureExtractionError: If opponent is not provided
        """
        from py.features.extractors.base import FeatureExtractionError

        if opponent is None:
            raise FeatureExtractionError(
                self.name,
                "Opponent-adjusted features require opponent parameter"
            )

        extractor = self._get_extractor()
        return extractor.extract_features(team, opponent, season, week)

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names produced.

        Returns:
            List of 7 opponent-adjusted feature names
        """
        return list(self.FEATURE_NAMES)

    def get_default_values(self) -> dict[str, float]:
        """Return neutral defaults for all features.

        Used when extraction fails and we need fallback values.
        Neutral values (0.0) mean no adjustment from raw metrics.
        """
        return {
            "off_epa_opponent_adjusted": 0.0,
            "pass_epa_opponent_adjusted": 0.0,
            "rush_epa_opponent_adjusted": 0.0,
            "dvoa_off_approx": 0.0,
            "dvoa_pass_approx": 0.0,
            "dvoa_rush_approx": 0.0,
            "dvoa_def_approx": 0.0,
        }
