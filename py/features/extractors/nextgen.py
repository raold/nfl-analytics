"""NextGen Stats Feature Extractor Strategy wrapper.

This module wraps the existing NextGenStatsFeatures class in the
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
from py.features.nextgen_features import NextGenStatsFeatures


class NextGenFeatureExtractor(FeatureExtractorStrategy):
    """Strategy wrapper for NextGenStatsFeatures.

    This extractor produces 23 Next Gen Stats features using
    rolling 4-game averages with exponential decay weighting.

    Features include:
    - Passing (5): time_to_throw, cpoe, aggressiveness, adot, qb_count
    - Rushing (4): rb_efficiency, ryoe, stacked_box_rate, rb_count
    - Receiving (4): separation, yac_above_exp, cushion, wr_count
    - Defensive (10): allowed versions of offensive metrics

    Example:
        extractor = NextGenFeatureExtractor()
        features = extractor.extract('KC', 2024, 10)
        # Returns: {'ngs_cpoe': 3.5, 'ngs_time_to_throw': 2.8, ...}
    """

    # Feature names produced by this extractor
    FEATURE_NAMES = [
        # Passing (Offensive - 5)
        "ngs_time_to_throw",
        "ngs_cpoe",
        "ngs_aggressiveness",
        "ngs_adot",
        "ngs_team_qb_count",
        # Rushing (Offensive - 4)
        "ngs_rb_efficiency",
        "ngs_ryoe",
        "ngs_stacked_box_rate",
        "ngs_team_rb_count",
        # Receiving (Offensive - 4)
        "ngs_separation",
        "ngs_yac_above_exp",
        "ngs_cushion",
        "ngs_team_wr_count",
        # Defensive (10)
        "def_ngs_cpoe_allowed",
        "def_ngs_time_to_throw_allowed",
        "def_ngs_separation_allowed",
        "def_ngs_yac_above_exp_allowed",
        "def_ngs_cushion_allowed",
        "def_ngs_ryoe_allowed",
        "def_ngs_efficiency_allowed",
        "def_ngs_stacked_box_rate_against",
        "def_ngs_adot_allowed",
        "def_ngs_aggressiveness_allowed",
    ]

    def __init__(self, db_config: Optional[dict] = None):
        """Initialize the NextGen feature extractor.

        Args:
            db_config: Database configuration dictionary
        """
        self._db_config = db_config
        self._extractor: Optional[NextGenStatsFeatures] = None

    def _get_extractor(self) -> NextGenStatsFeatures:
        """Lazy initialization of the underlying extractor."""
        if self._extractor is None:
            if self._db_config:
                self._extractor = NextGenStatsFeatures(db_config=self._db_config)
            else:
                self._extractor = NextGenStatsFeatures()
        return self._extractor

    @property
    def name(self) -> str:
        """Human-readable name for this extractor."""
        return "nextgen_stats"

    @property
    def prefix(self) -> str:
        """Prefix for feature names.

        Empty because NextGenStatsFeatures already includes prefixes
        in its feature names (ngs_, def_ngs_).
        """
        return ""

    @property
    def priority(self) -> FeaturePriority:
        """Extraction priority.

        HIGH: NextGen stats provide orthogonal signal to EPA metrics.
        Valuable but model can function with defaults if unavailable.
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
        """Extract NextGen stats features for a team.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Target season year
            week: Target week number
            conn: Optional database connection (not used - extractor manages own)
            opponent: Optional opponent team (not used for NGS features)

        Returns:
            Dictionary of 23 NextGen stats features
        """
        extractor = self._get_extractor()
        return extractor.get_team_ngs_features(team, season, week)

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names produced.

        Returns:
            List of 23 NextGen stats feature names
        """
        return list(self.FEATURE_NAMES)

    def get_default_values(self) -> dict[str, float]:
        """Return league-average defaults for all features.

        Used when extraction fails and we need fallback values.
        Values are based on NextGen stats defaults.
        """
        extractor = self._get_extractor()
        return {
            # Passing
            "ngs_time_to_throw": extractor.DEFAULT_TIME_TO_THROW,
            "ngs_cpoe": extractor.DEFAULT_CPOE,
            "ngs_aggressiveness": extractor.DEFAULT_AGGRESSIVENESS,
            "ngs_adot": extractor.DEFAULT_ADOT,
            "ngs_team_qb_count": extractor.DEFAULT_QB_COUNT,
            # Rushing
            "ngs_rb_efficiency": extractor.DEFAULT_RB_EFFICIENCY,
            "ngs_ryoe": extractor.DEFAULT_RYOE,
            "ngs_stacked_box_rate": extractor.DEFAULT_STACKED_BOX_RATE,
            "ngs_team_rb_count": extractor.DEFAULT_RB_COUNT,
            # Receiving
            "ngs_separation": extractor.DEFAULT_SEPARATION,
            "ngs_yac_above_exp": extractor.DEFAULT_YAC_ABOVE_EXP,
            "ngs_cushion": extractor.DEFAULT_CUSHION,
            "ngs_team_wr_count": extractor.DEFAULT_WR_COUNT,
            # Defensive (use offensive defaults for allowed metrics)
            "def_ngs_cpoe_allowed": extractor.DEFAULT_CPOE,
            "def_ngs_time_to_throw_allowed": extractor.DEFAULT_TIME_TO_THROW,
            "def_ngs_separation_allowed": extractor.DEFAULT_SEPARATION,
            "def_ngs_yac_above_exp_allowed": extractor.DEFAULT_YAC_ABOVE_EXP,
            "def_ngs_cushion_allowed": extractor.DEFAULT_CUSHION,
            "def_ngs_ryoe_allowed": extractor.DEFAULT_RYOE,
            "def_ngs_efficiency_allowed": extractor.DEFAULT_RB_EFFICIENCY,
            "def_ngs_stacked_box_rate_against": extractor.DEFAULT_STACKED_BOX_RATE,
            "def_ngs_adot_allowed": extractor.DEFAULT_ADOT,
            "def_ngs_aggressiveness_allowed": extractor.DEFAULT_AGGRESSIVENESS,
        }
