"""Defensive Feature Extractor Strategy wrapper.

This module wraps the existing TeamDefensiveFeatures class in the
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
from nfl_analytics.features.team_defensive_features import TeamDefensiveFeatures


class DefensiveFeatureExtractor(FeatureExtractorStrategy):
    """Strategy wrapper for TeamDefensiveFeatures.

    This extractor produces 27 team-level defensive features using
    rolling 4-game averages with exponential decay weighting.

    Features include:
    - EPA metrics (allowed per-play, pass, rush)
    - Efficiency metrics (success rate allowed, explosive plays, red zone)
    - Pass defense (yards allowed, completion pct, sacks, pressure)
    - Rush defense (yards allowed, success rate)
    - Game-level metrics (points allowed, yards)
    - Variance metrics (consistency scores)

    Example:
        extractor = DefensiveFeatureExtractor()
        features = extractor.extract('KC', 2024, 10)
        # Returns: {'def_epa_allowed_per_play': -0.05, ...}
    """

    # Feature names produced by this extractor
    FEATURE_NAMES = [
        # EPA-based features
        "def_epa_allowed_per_play",
        "def_success_rate_allowed",
        "def_pass_epa_allowed_per_play",
        "def_rush_epa_allowed_per_play",
        "def_turnover_rate",
        # Explosive play & efficiency
        "def_explosive_play_rate_allowed",
        "def_stuff_rate",
        "def_first_down_rate_allowed",
        "def_third_down_conv_rate_allowed",
        "def_red_zone_td_rate_allowed",
        # Pass defense
        "def_avg_yds_per_pass_allowed",
        "def_completion_pct_allowed",
        "def_pass_success_rate_allowed",
        "def_sack_rate",
        "def_pressure_rate",
        # Rush defense
        "def_avg_yds_per_rush_allowed",
        "def_rush_success_rate_allowed",
        # Game-level
        "def_points_allowed_per_game",
        "def_yds_allowed_per_game",
        "def_pass_yds_allowed_per_game",
        "def_rush_yds_allowed_per_game",
        "def_scoring_drives_pct_allowed",
        "def_penalty_rate",
        "def_blitz_rate",
        # Variance metrics
        "def_epa_std_4wk",
        "def_points_allowed_std_4wk",
        "def_epa_consistency_score",
    ]

    def __init__(self, db_config: Optional[dict] = None):
        """Initialize the defensive feature extractor.

        Args:
            db_config: Database configuration dictionary
        """
        self._db_config = db_config
        self._extractor: Optional[TeamDefensiveFeatures] = None

    def _get_extractor(self) -> TeamDefensiveFeatures:
        """Lazy initialization of the underlying extractor."""
        if self._extractor is None:
            if self._db_config:
                self._extractor = TeamDefensiveFeatures(db_config=self._db_config)
            else:
                self._extractor = TeamDefensiveFeatures()
        return self._extractor

    @property
    def name(self) -> str:
        """Human-readable name for this extractor."""
        return "defensive_features"

    @property
    def prefix(self) -> str:
        """Prefix for feature names.

        Empty because TeamDefensiveFeatures already includes 'def_' prefix
        in its feature names. This prevents double-prefixing.
        """
        return ""

    @property
    def priority(self) -> FeaturePriority:
        """Extraction priority.

        CRITICAL: Defensive features are fundamental for game predictions.
        Model cannot function without them.
        """
        return FeaturePriority.CRITICAL

    def extract(
        self,
        team: str,
        season: int,
        week: int,
        conn: Optional[psycopg2.extensions.connection] = None,
        opponent: Optional[str] = None,
    ) -> dict[str, float]:
        """Extract defensive features for a team.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Target season year
            week: Target week number
            conn: Optional database connection (not used - extractor manages own)
            opponent: Optional opponent team (not used for defensive features)

        Returns:
            Dictionary of 27 defensive features
        """
        extractor = self._get_extractor()
        return extractor.extract_features(team, season, week)

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names produced.

        Returns:
            List of 27 defensive feature names
        """
        return list(self.FEATURE_NAMES)

    def get_default_values(self) -> dict[str, float]:
        """Return league-average defaults for all features.

        Used when extraction fails and we need fallback values.
        Values are based on 2018-2024 NFL averages.
        """
        extractor = self._get_extractor()
        return {
            # EPA-based
            "def_epa_allowed_per_play": extractor.DEFAULT_DEF_EPA,
            "def_success_rate_allowed": extractor.DEFAULT_SUCCESS_RATE,
            "def_pass_epa_allowed_per_play": extractor.DEFAULT_PASS_EPA,
            "def_rush_epa_allowed_per_play": extractor.DEFAULT_RUSH_EPA,
            "def_turnover_rate": extractor.DEFAULT_TURNOVER_RATE,
            # Explosive play & efficiency
            "def_explosive_play_rate_allowed": extractor.DEFAULT_EXPLOSIVE_RATE,
            "def_stuff_rate": extractor.DEFAULT_STUFF_RATE,
            "def_first_down_rate_allowed": extractor.DEFAULT_FIRST_DOWN_RATE,
            "def_third_down_conv_rate_allowed": extractor.DEFAULT_THIRD_DOWN_RATE,
            "def_red_zone_td_rate_allowed": extractor.DEFAULT_RED_ZONE_TD_RATE,
            # Pass defense
            "def_avg_yds_per_pass_allowed": extractor.DEFAULT_YDS_PER_PASS,
            "def_completion_pct_allowed": extractor.DEFAULT_COMPLETION_PCT,
            "def_pass_success_rate_allowed": extractor.DEFAULT_PASS_SUCCESS,
            "def_sack_rate": extractor.DEFAULT_SACK_RATE,
            "def_pressure_rate": extractor.DEFAULT_PRESSURE_RATE,
            # Rush defense
            "def_avg_yds_per_rush_allowed": extractor.DEFAULT_YDS_PER_RUSH,
            "def_rush_success_rate_allowed": extractor.DEFAULT_RUSH_SUCCESS,
            # Game-level
            "def_points_allowed_per_game": extractor.DEFAULT_POINTS_ALLOWED,
            "def_yds_allowed_per_game": extractor.DEFAULT_YDS_ALLOWED,
            "def_pass_yds_allowed_per_game": extractor.DEFAULT_PASS_YDS_ALLOWED,
            "def_rush_yds_allowed_per_game": extractor.DEFAULT_RUSH_YDS_ALLOWED,
            "def_scoring_drives_pct_allowed": extractor.DEFAULT_SCORING_DRIVES_PCT,
            "def_penalty_rate": extractor.DEFAULT_PENALTY_RATE,
            "def_blitz_rate": extractor.DEFAULT_BLITZ_RATE,
            # Variance
            "def_epa_std_4wk": extractor.DEFAULT_DEF_EPA_STD,
            "def_points_allowed_std_4wk": extractor.DEFAULT_DEF_POINTS_STD,
            "def_epa_consistency_score": extractor.DEFAULT_DEF_CONSISTENCY_SCORE,
        }
