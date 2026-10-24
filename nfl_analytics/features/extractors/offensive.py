"""Offensive Feature Extractor Strategy wrapper.

This module wraps the existing TeamOffensiveFeatures class in the
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
from py.features.team_offensive_features import TeamOffensiveFeatures


class OffensiveFeatureExtractor(FeatureExtractorStrategy):
    """Strategy wrapper for TeamOffensiveFeatures.

    This extractor produces 27 team-level offensive features using
    rolling 4-game averages with exponential decay weighting.

    Features include:
    - EPA metrics (per-play, pass, rush)
    - Efficiency metrics (success rate, explosive plays, red zone)
    - Volume metrics (yards, completions, sacks)
    - Game-level metrics (points, time of possession)
    - Variance metrics (consistency scores)

    Example:
        extractor = OffensiveFeatureExtractor()
        features = extractor.extract('KC', 2024, 10)
        # Returns: {'off_epa_per_play': 0.15, 'off_success_rate': 0.48, ...}
    """

    # Feature names produced by this extractor (without prefix since
    # the underlying TeamOffensiveFeatures already includes 'off_' prefix)
    FEATURE_NAMES = [
        "off_epa_per_play",
        "off_success_rate",
        "off_explosive_play_rate",
        "off_stuff_rate",
        "off_first_down_rate",
        "off_third_down_conv_rate",
        "off_pass_epa_per_play",
        "off_rush_epa_per_play",
        "off_pass_success_rate",
        "off_rush_success_rate",
        "off_avg_yds_per_pass",
        "off_avg_yds_per_rush",
        "off_completion_pct",
        "off_sack_rate",
        "off_turnover_rate",
        "off_penalty_rate",
        "off_red_zone_td_rate",
        "off_time_of_possession",
        "off_plays_per_game",
        "off_points_per_game",
        "off_yds_per_game",
        "off_pass_yds_per_game",
        "off_rush_yds_per_game",
        "off_scoring_drives_pct",
        "off_epa_std_4wk",
        "off_scoring_std_4wk",
        "off_epa_consistency_score",
    ]

    def __init__(self, db_config: Optional[dict] = None):
        """Initialize the offensive feature extractor.

        Args:
            db_config: Database configuration dictionary
        """
        self._db_config = db_config
        self._extractor: Optional[TeamOffensiveFeatures] = None

    def _get_extractor(self) -> TeamOffensiveFeatures:
        """Lazy initialization of the underlying extractor."""
        if self._extractor is None:
            if self._db_config:
                self._extractor = TeamOffensiveFeatures(db_config=self._db_config)
            else:
                self._extractor = TeamOffensiveFeatures()
        return self._extractor

    @property
    def name(self) -> str:
        """Human-readable name for this extractor."""
        return "offensive_features"

    @property
    def prefix(self) -> str:
        """Prefix for feature names.

        Empty because TeamOffensiveFeatures already includes 'off_' prefix
        in its feature names. This prevents double-prefixing.
        """
        return ""

    @property
    def priority(self) -> FeaturePriority:
        """Extraction priority.

        CRITICAL: Offensive features are fundamental for game predictions.
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
        """Extract offensive features for a team.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Target season year
            week: Target week number
            conn: Optional database connection (not used - extractor manages own)
            opponent: Optional opponent team (not used for offensive features)

        Returns:
            Dictionary of 27 offensive features
        """
        extractor = self._get_extractor()
        return extractor.extract_features(team, season, week)

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names produced.

        Returns:
            List of 27 offensive feature names
        """
        return list(self.FEATURE_NAMES)

    def get_default_values(self) -> dict[str, float]:
        """Return league-average defaults for all features.

        Used when extraction fails and we need fallback values.
        Values are based on 2018-2024 NFL averages.
        """
        extractor = self._get_extractor()
        return {
            "off_epa_per_play": extractor.DEFAULT_EPA_PER_PLAY,
            "off_success_rate": extractor.DEFAULT_SUCCESS_RATE,
            "off_explosive_play_rate": extractor.DEFAULT_EXPLOSIVE_RATE,
            "off_stuff_rate": extractor.DEFAULT_STUFF_RATE,
            "off_first_down_rate": extractor.DEFAULT_FIRST_DOWN_RATE,
            "off_third_down_conv_rate": extractor.DEFAULT_THIRD_DOWN_CONV,
            "off_pass_epa_per_play": extractor.DEFAULT_EPA_PER_PLAY,
            "off_rush_epa_per_play": extractor.DEFAULT_EPA_PER_PLAY,
            "off_pass_success_rate": extractor.DEFAULT_SUCCESS_RATE,
            "off_rush_success_rate": extractor.DEFAULT_SUCCESS_RATE,
            "off_avg_yds_per_pass": extractor.DEFAULT_YDS_PER_PASS,
            "off_avg_yds_per_rush": extractor.DEFAULT_YDS_PER_RUSH,
            "off_completion_pct": extractor.DEFAULT_COMPLETION_PCT,
            "off_sack_rate": extractor.DEFAULT_SACK_RATE,
            "off_turnover_rate": extractor.DEFAULT_TURNOVER_RATE,
            "off_penalty_rate": extractor.DEFAULT_PENALTY_RATE,
            "off_red_zone_td_rate": extractor.DEFAULT_RED_ZONE_TD_RATE,
            "off_time_of_possession": extractor.DEFAULT_TIME_OF_POSSESSION,
            "off_plays_per_game": extractor.DEFAULT_PLAYS_PER_GAME,
            "off_points_per_game": extractor.DEFAULT_POINTS_PER_GAME,
            "off_yds_per_game": extractor.DEFAULT_YDS_PER_GAME,
            "off_pass_yds_per_game": extractor.DEFAULT_PASS_YDS_PER_GAME,
            "off_rush_yds_per_game": extractor.DEFAULT_RUSH_YDS_PER_GAME,
            "off_scoring_drives_pct": extractor.DEFAULT_SCORING_DRIVES_PCT,
            "off_epa_std_4wk": extractor.DEFAULT_EPA_STD,
            "off_scoring_std_4wk": extractor.DEFAULT_SCORING_STD,
            "off_epa_consistency_score": extractor.DEFAULT_CONSISTENCY_SCORE,
        }
