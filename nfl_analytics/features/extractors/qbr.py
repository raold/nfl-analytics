"""QBR Feature Extractor Strategy wrapper.

This module wraps the existing QBRFeatures class in the
FeatureExtractorStrategy interface, allowing it to be registered
with the FeatureRegistry for unified orchestration.

Created: December 13, 2025
Phase 2.3 of code quality remediation plan.
"""

from __future__ import annotations

from typing import Optional

import psycopg2

from nfl_analytics.features.extractors.base import (
    FeatureExtractorStrategy,
    FeaturePriority,
)
from nfl_analytics.features.qbr_features import QBRFeatures


class QBRFeatureExtractor(FeatureExtractorStrategy):
    """Strategy wrapper for QBRFeatures.

    This extractor produces 10 ESPN QBR features using
    rolling 4-game averages with exponential decay weighting.

    Features include:
    - Rolling QBR (2): 4wk average, points added
    - Season QBR (2): season average, season points added
    - Components (3): pass, run, sack contributions
    - Variance (1): consistency score
    - Raw EPA (2): passing EPA, EPA per dropback

    Example:
        extractor = QBRFeatureExtractor()
        features = extractor.extract('KC', 2024, 10)
        # Returns: {'qbr_4wk_avg': 72.5, 'qbr_consistency': 0.85, ...}
    """

    # Feature names produced by this extractor
    FEATURE_NAMES = [
        # Rolling QBR
        "qbr_4wk_avg",
        "qbr_pts_added_4wk",
        # Season QBR
        "qbr_season_avg",
        "qbr_season_pts_added",
        # Components
        "qbr_pass_component_4wk",
        "qbr_run_component_4wk",
        "qbr_sack_component_4wk",
        # Variance
        "qbr_consistency",
        # EPA-based (if available)
        "qbr_pass_epa",
        "qbr_epa_per_dropback",
    ]

    def __init__(self, db_config: Optional[dict] = None):
        """Initialize the QBR feature extractor.

        Args:
            db_config: Database configuration dictionary
        """
        self._db_config = db_config
        self._extractor: Optional[QBRFeatures] = None

    def _get_extractor(self) -> QBRFeatures:
        """Lazy initialization of the underlying extractor."""
        if self._extractor is None:
            if self._db_config:
                self._extractor = QBRFeatures(db_config=self._db_config)
            else:
                self._extractor = QBRFeatures()
        return self._extractor

    @property
    def name(self) -> str:
        """Human-readable name for this extractor."""
        return "qbr_features"

    @property
    def prefix(self) -> str:
        """Prefix for feature names.

        Empty because QBRFeatures already includes 'qbr_' prefix
        in its feature names.
        """
        return ""

    @property
    def priority(self) -> FeaturePriority:
        """Extraction priority.

        HIGH: QB performance is single most predictive factor for
        game outcomes. Model functions with defaults but loses accuracy.
        """
        return FeaturePriority.HIGH

    def extract(
        self,
        team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
        opponent: Optional[str] = None,
    ) -> dict[str, float]:
        """Extract QBR features for a team.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Target season year
            week: Target week number
            conn: Optional database connection (not used - extractor manages own)
            opponent: Optional opponent team (not used for QBR features)

        Returns:
            Dictionary of 10 QBR features
        """
        extractor = self._get_extractor()
        return extractor.extract_features(team, season, week)

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names produced.

        Returns:
            List of 10 QBR feature names
        """
        return list(self.FEATURE_NAMES)

    def get_default_values(self) -> dict[str, float]:
        """Return league-average defaults for all features.

        Used when extraction fails and we need fallback values.
        Values are based on QBRFeatures defaults.
        """
        extractor = self._get_extractor()
        return {
            "qbr_4wk_avg": extractor.DEFAULT_QBR,
            "qbr_pts_added_4wk": extractor.DEFAULT_PTS_ADDED,
            "qbr_season_avg": extractor.DEFAULT_QBR,
            "qbr_season_pts_added": extractor.DEFAULT_PTS_ADDED,
            "qbr_pass_component_4wk": extractor.DEFAULT_PASS_COMPONENT,
            "qbr_run_component_4wk": extractor.DEFAULT_RUN_COMPONENT,
            "qbr_sack_component_4wk": extractor.DEFAULT_SACK_COMPONENT,
            "qbr_consistency": extractor.DEFAULT_CONSISTENCY,
            "qbr_pass_epa": 0.0,
            "qbr_epa_per_dropback": 0.0,
        }
