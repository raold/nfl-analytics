"""Unified as-of team/game features with configurable feature levels.

This module consolidates the asof_features chain into a single unified extractor
with a `feature_level` parameter to select the complexity level:

- 'base': Core features (~40) - EPA, margin, rest, QB/coach stability
- 'enhanced': + Play metrics (~60) - CPOE, success rate, air yards, explosives
- 'v2': + Game context (~80) - Division, Thursday, weather, defensive EPA trends

This replaces the inheritance chain of:
  asof_features.py → asof_features_enhanced.py → asof_features_v2.py

Example:
    from py.features.asof_features_unified import AsofFeatureExtractor

    # Use v2 features (most complete)
    extractor = AsofFeatureExtractor(feature_level='v2')
    features_df = extractor.extract_all(season_start=2020)

    # Use base features for compatibility
    extractor = AsofFeatureExtractor(feature_level='base')
    features_df = extractor.extract_all()

Created: December 13, 2025
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterable
from enum import Enum
from io import StringIO
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg

logger = logging.getLogger(__name__)


class FeatureLevel(Enum):
    """Feature extraction level."""

    BASE = "base"  # Core features only
    ENHANCED = "enhanced"  # + Play-level metrics
    V2 = "v2"  # + Game context features


# SQL Templates for each level
SQL_TEAM_BASE = """
WITH team_games AS (
    SELECT g.game_id,
           g.season,
           g.week,
           g.kickoff,
           g.home_team AS team,
           TRUE AS is_home,
           g.home_team,
           g.away_team,
           g.home_score,
           g.away_score,
           g.spread_close,
           g.total_close,
           g.home_score AS points_for,
           g.away_score AS points_against,
           g.stadium,
           g.roof,
           g.surface,
           g.home_qb_id AS qb_id,
           g.home_qb_name AS qb_name,
           g.home_coach AS coach_name
           {v2_home_columns}
    FROM games g
    UNION ALL
    SELECT g.game_id,
           g.season,
           g.week,
           g.kickoff,
           g.away_team AS team,
           FALSE AS is_home,
           g.home_team,
           g.away_team,
           g.home_score,
           g.away_score,
           g.spread_close,
           g.total_close,
           g.away_score AS points_for,
           g.home_score AS points_against,
           g.stadium,
           g.roof,
           g.surface,
           g.away_qb_id AS qb_id,
           g.away_qb_name AS qb_name,
           g.away_coach AS coach_name
           {v2_away_columns}
    FROM games g
),
team_stats AS (
    SELECT game_id,
           posteam AS team,
           COALESCE(SUM(epa), 0) AS epa_sum,
           COUNT(*) AS plays
           {enhanced_stats_columns}
    FROM plays
    WHERE posteam IS NOT NULL
    GROUP BY 1, 2
)
SELECT tg.game_id,
       tg.team,
       tg.is_home,
       tg.season,
       tg.week,
       tg.kickoff,
       tg.home_team,
       tg.away_team,
       tg.home_score,
       tg.away_score,
       tg.spread_close,
       tg.total_close,
       tg.points_for,
       tg.points_against,
       tg.stadium,
       tg.roof,
       tg.surface,
       tg.qb_id,
       tg.qb_name,
       tg.coach_name,
       tg.points_for - tg.points_against AS margin,
       COALESCE(ts.epa_sum, 0) AS epa_sum,
       COALESCE(ts.plays, 0) AS plays
       {v2_select_columns}
       {enhanced_select_columns}
FROM team_games tg
LEFT JOIN team_stats ts
  ON ts.game_id = tg.game_id AND ts.team = tg.team
ORDER BY tg.team, tg.season, tg.week, tg.kickoff NULLS FIRST, tg.game_id;
"""

# Enhanced stats columns (play-level metrics)
ENHANCED_STATS_COLUMNS = """,
           -- Success rate (EPA > 0)
           COALESCE(AVG(CASE WHEN success = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS success_rate,
           -- Passing efficiency
           COALESCE(AVG(CASE WHEN pass = 1 THEN air_yards END), 0) AS air_yards_mean,
           COALESCE(AVG(CASE WHEN pass = 1 THEN yards_after_catch END), 0) AS yac_mean,
           COALESCE(AVG(CASE WHEN pass = 1 THEN cpoe END), 0) AS cpoe_mean,
           COALESCE(AVG(CASE WHEN pass = 1 AND complete_pass = 1 THEN 1.0 ELSE 0.0 END), 0) AS completion_pct,
           -- Win probability
           COALESCE(MAX(wp), 0.5) AS wp_max,
           COALESCE(MIN(wp), 0.5) AS wp_min,
           COALESCE(AVG(CASE WHEN quarter = 4 THEN wp END), 0.5) AS wp_q4_mean,
           -- Situational play-calling
           COALESCE(AVG(CASE WHEN shotgun = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS shotgun_rate,
           COALESCE(AVG(CASE WHEN no_huddle = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS no_huddle_rate,
           -- Explosive plays
           COUNT(CASE WHEN pass = 1 AND yards_gained >= 20 THEN 1 END) AS explosive_pass,
           COUNT(CASE WHEN rush = 1 AND yards_gained >= 10 THEN 1 END) AS explosive_rush"""

ENHANCED_SELECT_COLUMNS = """,
       COALESCE(ts.success_rate, 0) AS success_rate,
       COALESCE(ts.air_yards_mean, 0) AS air_yards_mean,
       COALESCE(ts.yac_mean, 0) AS yac_mean,
       COALESCE(ts.cpoe_mean, 0) AS cpoe_mean,
       COALESCE(ts.completion_pct, 0) AS completion_pct,
       COALESCE(ts.wp_max, 0.5) AS wp_max,
       COALESCE(ts.wp_min, 0.5) AS wp_min,
       COALESCE(ts.wp_q4_mean, 0.5) AS wp_q4_mean,
       COALESCE(ts.shotgun_rate, 0) AS shotgun_rate,
       COALESCE(ts.no_huddle_rate, 0) AS no_huddle_rate,
       COALESCE(ts.explosive_pass, 0) AS explosive_pass,
       COALESCE(ts.explosive_rush, 0) AS explosive_rush"""

# V2 game-level columns
V2_HOME_COLUMNS = """,
           g.home_turnovers AS turnovers,
           g.away_turnovers AS turnovers_opp,
           g.home_penalties AS penalties,
           g.home_penalty_yards AS penalty_yards,
           g.home_rest AS rest_days,
           g.is_division_game,
           g.is_thursday_night,
           g.is_short_week,
           g.weather_condition,
           g.home_def_epa_last_4 AS def_epa_last_4,
           g.away_def_epa_last_4 AS opp_def_epa_last_4,
           g.division_h2h_variance"""

V2_AWAY_COLUMNS = """,
           g.away_turnovers AS turnovers,
           g.home_turnovers AS turnovers_opp,
           g.away_penalties AS penalties,
           g.away_penalty_yards AS penalty_yards,
           g.away_rest AS rest_days,
           g.is_division_game,
           g.is_thursday_night,
           g.is_short_week,
           g.weather_condition,
           g.away_def_epa_last_4 AS def_epa_last_4,
           g.home_def_epa_last_4 AS opp_def_epa_last_4,
           g.division_h2h_variance"""

V2_SELECT_COLUMNS = """,
       tg.turnovers,
       tg.turnovers_opp,
       tg.penalties,
       tg.penalty_yards,
       tg.rest_days AS rest_days_db,
       tg.is_division_game,
       tg.is_thursday_night,
       tg.is_short_week,
       tg.weather_condition,
       tg.def_epa_last_4,
       tg.opp_def_epa_last_4,
       tg.division_h2h_variance"""

# Feature column lists by level
BASE_FEATURE_COLUMNS = [
    "prior_games",
    "prior_epa_sum",
    "prior_plays",
    "prior_epa_mean",
    "epa_pp_last3",
    "epa_pp_last5",
    "prior_margin_avg",
    "margin_last3",
    "margin_last5",
    "prior_points_for_avg",
    "prior_points_against_avg",
    "points_for_last3",
    "points_against_last3",
    "prior_win_pct",
    "win_pct_last5",
    "season_win_pct",
    "season_point_diff_avg",
    "rest_days",
    "rest_lt_6",
    "rest_gt_13",
    "travel_change",
    "prev_result",
    "prev_margin",
    "prev_points_for",
    "prev_points_against",
    "prev_spread_close",
    "qb_change",
    "qb_team_games",
    "qb_total_games",
    "coach_change",
    "coach_team_games",
    "coach_total_games",
    "surface_grass",
    "surface_artificial",
    "roof_dome",
    "roof_outdoors",
]

ENHANCED_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "prior_turnovers_avg",
    "prior_turnovers_opp_avg",
    "prior_penalties_avg",
    "prior_penalty_yards_avg",
    "prior_success_rate",
    "success_rate_last3",
    "success_rate_last5",
    "prior_air_yards",
    "prior_yac",
    "prior_cpoe",
    "prior_completion_pct",
    "prior_wp_q4_mean",
    "prior_shotgun_rate",
    "prior_no_huddle_rate",
    "prior_explosive_pass_rate",
    "prior_explosive_rush_rate",
]

V2_FEATURE_COLUMNS = ENHANCED_FEATURE_COLUMNS + [
    "is_division_game",
    "is_thursday_night",
    "is_short_week",
    "weather_condition",
    "def_epa_last_4",
    "opp_def_epa_last_4",
    "division_h2h_variance",
    "prior_division_win_rate",
    "prior_thursday_avg_margin",
    "prior_short_week_avg_margin",
]

# Diff feature columns by level
BASE_DIFF_COLUMNS = [
    "prior_epa_mean",
    "prior_games",
    "epa_pp_last3",
    "epa_pp_last5",
    "prior_margin_avg",
    "margin_last3",
    "margin_last5",
    "prior_points_for_avg",
    "prior_points_against_avg",
    "points_for_last3",
    "points_against_last3",
    "prior_win_pct",
    "win_pct_last5",
    "season_win_pct",
    "season_point_diff_avg",
    "rest_days",
    "qb_change",
    "qb_team_games",
    "qb_total_games",
    "coach_change",
    "coach_team_games",
    "coach_total_games",
]

ENHANCED_DIFF_COLUMNS = BASE_DIFF_COLUMNS + [
    "prior_turnovers_avg",
    "prior_penalties_avg",
    "prior_success_rate",
    "success_rate_last3",
    "prior_air_yards",
    "prior_cpoe",
    "prior_completion_pct",
    "prior_shotgun_rate",
    "prior_explosive_pass_rate",
]

V2_DIFF_COLUMNS = ENHANCED_DIFF_COLUMNS + [
    "def_epa_last_4",
]


def get_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> psycopg.Connection:
    """Get database connection with defaults from environment."""
    return psycopg.connect(
        host=host or os.environ.get("POSTGRES_HOST", "localhost"),
        port=port or int(os.environ.get("POSTGRES_PORT", "5544")),
        dbname=dbname or os.environ.get("POSTGRES_DB", "devdb01"),
        user=user or os.environ.get("POSTGRES_USER", "dro"),
        password=password or os.environ.get("POSTGRES_PASSWORD", "sicillionbillions"),
    )


def rename_with_prefix(columns: Iterable[str], prefix: str) -> dict[str, str]:
    """Create column rename dict with prefix."""
    return {col: f"{prefix}_{col}" for col in columns}


class AsofFeatureExtractor:
    """Unified as-of feature extractor with configurable feature levels.

    This consolidates the asof_features chain into a single class that can
    produce features at any complexity level.

    Args:
        feature_level: 'base', 'enhanced', or 'v2'
        conn: Optional database connection (creates new if not provided)
        db_config: Optional dict with host, port, dbname, user, password

    Example:
        extractor = AsofFeatureExtractor(feature_level='enhanced')
        df = extractor.extract_all(season_start=2020)

    Note:
        The 'v2' level requires additional database columns (is_division_game,
        is_thursday_night, etc.) that may not be present. If these are missing,
        the extractor will automatically downgrade to 'enhanced' level.
    """

    # V2 columns that may not exist in all database schemas
    V2_OPTIONAL_COLUMNS = [
        "is_division_game",
        "is_thursday_night",
        "is_short_week",
        "weather_condition",
        "home_def_epa_last_4",
        "away_def_epa_last_4",
        "division_h2h_variance",
    ]

    def __init__(
        self,
        feature_level: str = "enhanced",  # Default to enhanced (works with current schema)
        conn: Optional[psycopg.Connection] = None,
        db_config: Optional[dict[str, Any]] = None,
    ):
        self._requested_level = FeatureLevel(feature_level)
        self._conn = conn
        self._db_config = db_config or {}
        self._owns_connection = False
        self._v2_columns_available: Optional[bool] = None
        self._checked_schema = False

    @property
    def feature_level(self) -> FeatureLevel:
        """Get effective feature level after schema check."""
        if self._requested_level == FeatureLevel.V2:
            if not self._checked_schema:
                self._check_v2_schema()
            if not self._v2_columns_available:
                return FeatureLevel.ENHANCED
        return self._requested_level

    def _check_v2_schema(self) -> None:
        """Check if v2 columns exist in database."""
        if self._checked_schema:
            return

        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'games'
            """)
            existing_cols = {r[0] for r in cur.fetchall()}
            self._v2_columns_available = all(
                col in existing_cols for col in self.V2_OPTIONAL_COLUMNS
            )
            if not self._v2_columns_available:
                missing = [c for c in self.V2_OPTIONAL_COLUMNS if c not in existing_cols]
                logger.warning(
                    f"V2 columns not found in database: {missing}. "
                    f"Downgrading to 'enhanced' level. Run Migration 019 to enable v2 features."
                )
        except Exception as e:
            logger.warning(f"Could not check schema: {e}. Using enhanced level.")
            self._v2_columns_available = False
        finally:
            self._checked_schema = True

    @property
    def conn(self) -> psycopg.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = get_connection(**self._db_config)
            self._owns_connection = True
        return self._conn

    def close(self) -> None:
        """Close connection if we own it."""
        if self._owns_connection and self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def feature_columns(self) -> list[str]:
        """Get feature columns for current level."""
        if self.feature_level == FeatureLevel.BASE:
            return BASE_FEATURE_COLUMNS
        elif self.feature_level == FeatureLevel.ENHANCED:
            return ENHANCED_FEATURE_COLUMNS
        else:
            return V2_FEATURE_COLUMNS

    @property
    def diff_columns(self) -> list[str]:
        """Get diff columns for current level."""
        if self.feature_level == FeatureLevel.BASE:
            return BASE_DIFF_COLUMNS
        elif self.feature_level == FeatureLevel.ENHANCED:
            return ENHANCED_DIFF_COLUMNS
        else:
            return V2_DIFF_COLUMNS

    def _build_sql(self) -> str:
        """Build SQL query for current feature level."""
        # Determine which SQL fragments to include
        include_enhanced = self.feature_level in (FeatureLevel.ENHANCED, FeatureLevel.V2)
        include_v2 = self.feature_level == FeatureLevel.V2

        return SQL_TEAM_BASE.format(
            v2_home_columns=V2_HOME_COLUMNS if include_v2 else "",
            v2_away_columns=V2_AWAY_COLUMNS if include_v2 else "",
            enhanced_stats_columns=ENHANCED_STATS_COLUMNS if include_enhanced else "",
            v2_select_columns=V2_SELECT_COLUMNS if include_v2 else "",
            enhanced_select_columns=ENHANCED_SELECT_COLUMNS if include_enhanced else "",
        )

    def fetch_team_game_features(self) -> pd.DataFrame:
        """Fetch raw team-game features from database."""
        sql = self._build_sql()
        return pd.read_sql(sql, self.conn)

    def compute_team_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling/cumulative features for team history."""
        team_df = df.copy()
        team_df["kickoff"] = pd.to_datetime(team_df["kickoff"], errors="coerce")
        team_df.sort_values(
            ["team", "season", "week", "kickoff", "game_id"],
            inplace=True,
            na_position="last",
        )

        # Cast numeric columns
        team_df["epa_sum"] = team_df["epa_sum"].astype(float)
        team_df["plays"] = team_df["plays"].astype(float)
        team_df["points_for"] = team_df["points_for"].astype(float)
        team_df["points_against"] = team_df["points_against"].astype(float)
        team_df["margin"] = team_df["margin"].astype(float)

        # Fill categorical NAs
        team_df["qb_id"] = team_df["qb_id"].fillna("UNKNOWN")
        team_df["coach_name"] = team_df["coach_name"].fillna("UNKNOWN")
        team_df["surface"] = team_df["surface"].fillna("").str.lower()
        team_df["roof"] = team_df["roof"].fillna("").str.lower()

        # Derived metrics
        team_df["epa_per_play"] = np.where(
            team_df["plays"] > 0, team_df["epa_sum"] / team_df["plays"], np.nan
        )
        team_df["points_total"] = team_df["points_for"] + team_df["points_against"]

        result_values = np.where(
            team_df["margin"] > 0, 1.0, np.where(team_df["margin"] < 0, 0.0, np.nan)
        )
        result_values = np.where(
            np.isnan(result_values) & team_df["margin"].notna(), 0.5, result_values
        )
        team_df["result_value"] = result_values

        # Helper columns for cumsum
        team_df["_points_for_f"] = team_df["points_for"].fillna(0.0)
        team_df["_points_against_f"] = team_df["points_against"].fillna(0.0)
        team_df["_margin_f"] = team_df["margin"].fillna(0.0)
        team_df["_result_f"] = team_df["result_value"].fillna(0.0)

        team_group = team_df.groupby("team", group_keys=False)

        # Base cumulative features
        team_df["prior_games"] = team_group.cumcount()
        team_df["prior_epa_sum"] = team_group["epa_sum"].cumsum() - team_df["epa_sum"]
        team_df["prior_plays"] = team_group["plays"].cumsum() - team_df["plays"]
        team_df["prior_points_for"] = team_group["_points_for_f"].cumsum() - team_df["_points_for_f"]
        team_df["prior_points_against"] = (
            team_group["_points_against_f"].cumsum() - team_df["_points_against_f"]
        )
        team_df["prior_margin_sum"] = team_group["_margin_f"].cumsum() - team_df["_margin_f"]
        team_df["prior_wins"] = team_group["_result_f"].cumsum() - team_df["_result_f"]

        with np.errstate(divide="ignore", invalid="ignore"):
            team_df["prior_epa_mean"] = np.where(
                team_df["prior_plays"] > 0, team_df["prior_epa_sum"] / team_df["prior_plays"], np.nan
            )
            team_df["prior_margin_avg"] = np.where(
                team_df["prior_games"] > 0, team_df["prior_margin_sum"] / team_df["prior_games"], np.nan
            )
            team_df["prior_points_for_avg"] = np.where(
                team_df["prior_games"] > 0, team_df["prior_points_for"] / team_df["prior_games"], np.nan
            )
            team_df["prior_points_against_avg"] = np.where(
                team_df["prior_games"] > 0,
                team_df["prior_points_against"] / team_df["prior_games"],
                np.nan,
            )
            team_df["prior_win_pct"] = np.where(
                team_df["prior_games"] > 0, team_df["prior_wins"] / team_df["prior_games"], np.nan
            )

        # Season-specific metrics
        season_group = team_df.groupby(["team", "season"], group_keys=False)
        team_df["season_games_prior"] = season_group.cumcount()
        team_df["season_margin_prior_sum"] = season_group["_margin_f"].cumsum() - team_df["_margin_f"]
        team_df["season_wins_prior"] = season_group["_result_f"].cumsum() - team_df["_result_f"]
        with np.errstate(divide="ignore", invalid="ignore"):
            team_df["season_point_diff_avg"] = np.where(
                team_df["season_games_prior"] > 0,
                team_df["season_margin_prior_sum"] / team_df["season_games_prior"],
                np.nan,
            )
            team_df["season_win_pct"] = np.where(
                team_df["season_games_prior"] > 0,
                team_df["season_wins_prior"] / team_df["season_games_prior"],
                np.nan,
            )

        # Rolling windows
        team_df["epa_pp_last3"] = team_group["epa_per_play"].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        team_df["epa_pp_last5"] = team_group["epa_per_play"].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )
        team_df["margin_last3"] = team_group["_margin_f"].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        team_df["margin_last5"] = team_group["_margin_f"].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )
        team_df["points_for_last3"] = team_group["points_for"].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        team_df["points_against_last3"] = team_group["points_against"].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
        team_df["win_pct_last5"] = team_group["_result_f"].transform(
            lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
        )

        # Rest days
        team_df["prev_kickoff"] = team_group["kickoff"].shift(1)
        team_df["rest_days"] = (team_df["kickoff"] - team_df["prev_kickoff"]).dt.total_seconds() / (
            24 * 3600
        )
        team_df["rest_days"] = team_df["rest_days"].fillna(7.0)
        team_df["rest_lt_6"] = (team_df["rest_days"] < 6).astype(float)
        team_df["rest_gt_13"] = (team_df["rest_days"] > 13).astype(float)

        # Previous game info
        team_df["prev_result"] = team_group["_result_f"].shift(1)
        team_df["prev_margin"] = team_group["_margin_f"].shift(1)
        team_df["prev_points_for"] = team_group["_points_for_f"].shift(1)
        team_df["prev_points_against"] = team_group["_points_against_f"].shift(1)
        team_df["prev_spread_close"] = team_group["spread_close"].shift(1)

        # Travel/schedule
        prev_is_home = team_group["is_home"].shift(1)
        team_df["travel_change"] = np.where(
            team_df["prior_games"] > 0, (prev_is_home != team_df["is_home"]).astype(float), 0.0
        )

        # QB stability
        prev_qb = team_group["qb_id"].shift(1)
        team_df["qb_change"] = np.where(
            team_df["prior_games"] > 0, (prev_qb != team_df["qb_id"]).astype(float), 0.0
        )
        team_df["qb_team_games"] = team_df.groupby(["team", "qb_id"]).cumcount().astype(float)
        team_df["qb_total_games"] = team_df.groupby("qb_id").cumcount().astype(float)

        # Coach stability
        prev_coach = team_group["coach_name"].shift(1)
        team_df["coach_change"] = np.where(
            team_df["prior_games"] > 0, (prev_coach != team_df["coach_name"]).astype(float), 0.0
        )
        team_df["coach_team_games"] = team_df.groupby(["team", "coach_name"]).cumcount().astype(float)
        team_df["coach_total_games"] = team_df.groupby("coach_name").cumcount().astype(float)

        # Surface/venue
        surface_lower = team_df["surface"]
        team_df["surface_grass"] = surface_lower.str.contains("grass").astype(float)
        team_df["surface_artificial"] = surface_lower.str.contains("turf").astype(float)

        roof_lower = team_df["roof"]
        team_df["roof_dome"] = roof_lower.isin(["dome", "closed"]).astype(float)
        team_df["roof_outdoors"] = roof_lower.isin(["outdoors", "open", "retractable"]).astype(float)

        # Enhanced features (if enabled)
        if self.feature_level in (FeatureLevel.ENHANCED, FeatureLevel.V2):
            team_df = self._add_enhanced_features(team_df, team_group)

        # V2 features (if enabled)
        if self.feature_level == FeatureLevel.V2:
            team_df = self._add_v2_features(team_df, team_group)

        # Cleanup helper columns
        helper_cols = ["_points_for_f", "_points_against_f", "_margin_f", "_result_f"]
        team_df.drop(columns=[c for c in helper_cols if c in team_df.columns], inplace=True)

        return team_df

    def _add_enhanced_features(
        self, team_df: pd.DataFrame, team_group: pd.core.groupby.DataFrameGroupBy
    ) -> pd.DataFrame:
        """Add enhanced play-level features."""
        # Fill turnover/penalty columns
        for col in ["turnovers", "turnovers_opp", "penalties", "penalty_yards"]:
            if col in team_df.columns:
                team_df[col] = team_df[col].fillna(0.0)

        # Cumulative turnover/penalty features
        if "turnovers" in team_df.columns:
            team_df["prior_turnovers_sum"] = team_group["turnovers"].cumsum() - team_df["turnovers"]
            team_df["prior_turnovers_opp_sum"] = (
                team_group["turnovers_opp"].cumsum() - team_df["turnovers_opp"]
            )
            team_df["prior_penalties_sum"] = team_group["penalties"].cumsum() - team_df["penalties"]
            team_df["prior_penalty_yards_sum"] = (
                team_group["penalty_yards"].cumsum() - team_df["penalty_yards"]
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                team_df["prior_turnovers_avg"] = np.where(
                    team_df["prior_games"] > 0,
                    team_df["prior_turnovers_sum"] / team_df["prior_games"],
                    np.nan,
                )
                team_df["prior_turnovers_opp_avg"] = np.where(
                    team_df["prior_games"] > 0,
                    team_df["prior_turnovers_opp_sum"] / team_df["prior_games"],
                    np.nan,
                )
                team_df["prior_penalties_avg"] = np.where(
                    team_df["prior_games"] > 0,
                    team_df["prior_penalties_sum"] / team_df["prior_games"],
                    np.nan,
                )
                team_df["prior_penalty_yards_avg"] = np.where(
                    team_df["prior_games"] > 0,
                    team_df["prior_penalty_yards_sum"] / team_df["prior_games"],
                    np.nan,
                )

        # Play-level efficiency features
        for col in [
            "success_rate", "air_yards_mean", "yac_mean", "cpoe_mean",
            "completion_pct", "wp_q4_mean", "shotgun_rate", "no_huddle_rate",
        ]:
            if col in team_df.columns:
                team_df[col] = team_df[col].fillna(0.0 if col != "wp_q4_mean" else 0.5)

        # Rolling averages for play metrics
        if "success_rate" in team_df.columns:
            team_df["prior_success_rate"] = team_group["success_rate"].transform(
                lambda s: s.shift(1).expanding().mean()
            )
            team_df["success_rate_last3"] = team_group["success_rate"].transform(
                lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
            )
            team_df["success_rate_last5"] = team_group["success_rate"].transform(
                lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
            )

        if "air_yards_mean" in team_df.columns:
            team_df["prior_air_yards"] = team_group["air_yards_mean"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        if "yac_mean" in team_df.columns:
            team_df["prior_yac"] = team_group["yac_mean"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        if "cpoe_mean" in team_df.columns:
            team_df["prior_cpoe"] = team_group["cpoe_mean"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        if "completion_pct" in team_df.columns:
            team_df["prior_completion_pct"] = team_group["completion_pct"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        if "wp_q4_mean" in team_df.columns:
            team_df["prior_wp_q4_mean"] = team_group["wp_q4_mean"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        if "shotgun_rate" in team_df.columns:
            team_df["prior_shotgun_rate"] = team_group["shotgun_rate"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        if "no_huddle_rate" in team_df.columns:
            team_df["prior_no_huddle_rate"] = team_group["no_huddle_rate"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        # Explosive play rates
        if "explosive_pass" in team_df.columns:
            team_df["explosive_pass"] = team_df["explosive_pass"].fillna(0.0)
            team_df["explosive_rush"] = team_df["explosive_rush"].fillna(0.0)

            with np.errstate(divide="ignore", invalid="ignore"):
                team_df["explosive_pass_rate"] = np.where(
                    team_df["plays"] > 0, team_df["explosive_pass"] / team_df["plays"], 0.0
                )
                team_df["explosive_rush_rate"] = np.where(
                    team_df["plays"] > 0, team_df["explosive_rush"] / team_df["plays"], 0.0
                )

            team_df["prior_explosive_pass_rate"] = team_group["explosive_pass_rate"].transform(
                lambda s: s.shift(1).expanding().mean()
            )
            team_df["prior_explosive_rush_rate"] = team_group["explosive_rush_rate"].transform(
                lambda s: s.shift(1).expanding().mean()
            )

        return team_df

    def _add_v2_features(
        self, team_df: pd.DataFrame, team_group: pd.core.groupby.DataFrameGroupBy
    ) -> pd.DataFrame:
        """Add v2 game context features."""
        # Fill v2 columns
        team_df["is_division_game"] = team_df["is_division_game"].fillna(False)
        team_df["is_thursday_night"] = team_df["is_thursday_night"].fillna(False)
        team_df["is_short_week"] = team_df["is_short_week"].fillna(False)
        team_df["weather_condition"] = team_df["weather_condition"].fillna("unknown")
        team_df["def_epa_last_4"] = team_df["def_epa_last_4"].fillna(0.0)
        team_df["opp_def_epa_last_4"] = team_df["opp_def_epa_last_4"].fillna(0.0)
        team_df["division_h2h_variance"] = team_df["division_h2h_variance"].fillna(0.0)

        # Division game history
        team_df["division_game_indicator"] = team_df["is_division_game"].astype(float)
        team_df["division_game_win"] = (
            team_df["is_division_game"] & (team_df["margin"] > 0)
        ).astype(float)

        team_df["prior_division_games"] = team_group["division_game_indicator"].transform(
            lambda s: s.shift(1).cumsum()
        )
        team_df["prior_division_wins"] = team_group["division_game_win"].transform(
            lambda s: s.shift(1).cumsum()
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            team_df["prior_division_win_rate"] = np.where(
                team_df["prior_division_games"] > 0,
                team_df["prior_division_wins"] / team_df["prior_division_games"],
                0.5,
            )

        # Thursday night history
        team_df["thursday_game_indicator"] = team_df["is_thursday_night"].astype(float)
        team_df["thursday_game_margin"] = np.where(
            team_df["is_thursday_night"], team_df["margin"], 0.0
        )

        team_df["prior_thursday_games"] = team_group["thursday_game_indicator"].transform(
            lambda s: s.shift(1).cumsum()
        )
        team_df["prior_thursday_margin_sum"] = team_group["thursday_game_margin"].transform(
            lambda s: s.shift(1).cumsum()
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            team_df["prior_thursday_avg_margin"] = np.where(
                team_df["prior_thursday_games"] > 0,
                team_df["prior_thursday_margin_sum"] / team_df["prior_thursday_games"],
                0.0,
            )

        # Short week history
        team_df["short_week_indicator"] = team_df["is_short_week"].astype(float)
        team_df["short_week_margin"] = np.where(team_df["is_short_week"], team_df["margin"], 0.0)

        team_df["prior_short_week_games"] = team_group["short_week_indicator"].transform(
            lambda s: s.shift(1).cumsum()
        )
        team_df["prior_short_week_margin_sum"] = team_group["short_week_margin"].transform(
            lambda s: s.shift(1).cumsum()
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            team_df["prior_short_week_avg_margin"] = np.where(
                team_df["prior_short_week_games"] > 0,
                team_df["prior_short_week_margin_sum"] / team_df["prior_short_week_games"],
                0.0,
            )

        return team_df

    def pivot_to_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot team features to game-level home/away format."""
        df = df.copy()
        df["kickoff"] = pd.to_datetime(df["kickoff"], errors="coerce")

        common_cols = [
            "game_id", "season", "week", "kickoff",
            "home_team", "away_team", "home_score", "away_score",
            "spread_close", "total_close",
        ]

        # V2 game-level features (not split by team)
        v2_game_features = []
        if self.feature_level == FeatureLevel.V2:
            v2_game_features = [
                "is_division_game", "is_thursday_night", "is_short_week",
                "weather_condition", "division_h2h_variance",
            ]

        home = df[df["is_home"]].copy()
        away = df[~df["is_home"]].copy()

        # Get team features (exclude game-level v2 features)
        team_features = [c for c in self.feature_columns if c not in v2_game_features]

        home_cols = common_cols + v2_game_features + [c for c in team_features if c in home.columns]
        away_cols = common_cols + [c for c in team_features if c in away.columns]

        home = home[home_cols].rename(
            columns=rename_with_prefix([c for c in team_features if c in home.columns], "home")
        )
        away = away[away_cols].rename(
            columns=rename_with_prefix([c for c in team_features if c in away.columns], "away")
        )

        merged = pd.merge(home, away, on=common_cols, how="inner", suffixes=("", "_dup"))
        merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]

        # Fill NA values
        for col in ["home_prior_epa_sum", "home_prior_plays", "away_prior_epa_sum", "away_prior_plays"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        for col in ["home_rest_days", "away_rest_days"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(7.0)

        # V2 feature fills
        if self.feature_level == FeatureLevel.V2:
            merged["is_division_game"] = merged["is_division_game"].fillna(False).astype(bool)
            merged["is_thursday_night"] = merged["is_thursday_night"].fillna(False).astype(bool)
            merged["is_short_week"] = merged["is_short_week"].fillna(False).astype(bool)
            merged["weather_condition"] = merged["weather_condition"].fillna("unknown")

            # One-hot encode weather
            weather_dummies = pd.get_dummies(merged["weather_condition"], prefix="weather")
            merged = pd.concat([merged, weather_dummies], axis=1)

        # Compute differentials
        for col in self.diff_columns:
            home_col = f"home_{col}"
            away_col = f"away_{col}"
            diff_col = f"{col}_diff"
            if home_col in merged.columns and away_col in merged.columns:
                merged[diff_col] = merged[home_col] - merged[away_col]

        # Legacy diff columns
        if "home_prior_epa_mean" in merged.columns and "away_prior_epa_mean" in merged.columns:
            merged["epa_diff_prior"] = merged["home_prior_epa_mean"] - merged["away_prior_epa_mean"]
        if "home_prior_plays" in merged.columns and "away_prior_plays" in merged.columns:
            merged["plays_diff_prior"] = merged["home_prior_plays"] - merged["away_prior_plays"]
        if "home_rest_days" in merged.columns and "away_rest_days" in merged.columns:
            merged["rest_diff"] = merged["home_rest_days"] - merged["away_rest_days"]

        # Betting outcomes
        merged["home_margin"] = merged["home_score"] - merged["away_score"]
        merged["home_win"] = (merged["home_margin"] > 0).astype(float)
        merged["home_cover"] = (merged["home_margin"] + merged["spread_close"] > 0).astype(float)
        merged["over_hit"] = (
            (merged["home_score"] + merged["away_score"]) > merged["total_close"]
        ).astype(float)
        merged["is_push"] = merged["home_cover"].isna()

        if self.feature_level == FeatureLevel.V2:
            merged["is_tie"] = (merged["home_margin"] == 0).astype(bool)
            merged["high_tie_prob"] = (merged["spread_close"].abs() < 1.5).fillna(False).astype(bool)

        merged.sort_values(["season", "week", "game_id"], inplace=True)
        return merged

    def extract_all(
        self,
        season_start: Optional[int] = None,
        season_end: Optional[int] = None,
        validate: bool = False,
    ) -> pd.DataFrame:
        """Extract all features and return game-level DataFrame.

        Args:
            season_start: First season to include (default: all)
            season_end: Last season to include (default: all)
            validate: Run validation checks

        Returns:
            Game-level DataFrame with home/away features
        """
        logger.info(f"Extracting asof features (level={self.feature_level.value})")

        # Fetch raw data
        team_df = self.fetch_team_game_features()
        logger.info(f"Fetched {len(team_df)} team-game records")

        # Compute history
        team_history = self.compute_team_history(team_df)

        if validate:
            self._validate_team_history(team_history)

        # Pivot to games
        games_wide = self.pivot_to_games(team_history)

        # Filter seasons
        if season_start is not None:
            games_wide = games_wide[games_wide["season"] >= season_start].copy()
        if season_end is not None:
            games_wide = games_wide[games_wide["season"] <= season_end].copy()

        logger.info(f"Extracted {len(games_wide)} games, {len(games_wide.columns)} columns")
        return games_wide

    def _validate_team_history(self, df: pd.DataFrame) -> None:
        """Validate team history for leakage."""
        expected_prior = df.groupby("team").cumcount()
        if not np.array_equal(df["prior_games"].to_numpy(), expected_prior.to_numpy()):
            raise ValueError("prior_games mismatch detected (possible leakage)")

        mask = df["prior_plays"] > 0
        if mask.any():
            ratio = df.loc[mask, "prior_epa_sum"] / df.loc[mask, "prior_plays"]
            if not np.allclose(df.loc[mask, "prior_epa_mean"], ratio, rtol=1e-6, atol=1e-9):
                raise ValueError("prior_epa_mean inconsistent with sums/plays")

        if (df["rest_days"] < 0).any():
            raise ValueError("Negative rest_days encountered")

        logger.info("Validation passed")


# Backward compatibility functions
def get_asof_features(
    feature_level: str = "enhanced",  # Default to enhanced (works with current schema)
    season_start: Optional[int] = None,
    season_end: Optional[int] = None,
    validate: bool = False,
    conn: Optional[psycopg.Connection] = None,
) -> pd.DataFrame:
    """Convenience function to get asof features.

    Args:
        feature_level: 'base', 'enhanced', or 'v2'
        season_start: First season to include
        season_end: Last season to include
        validate: Run validation checks
        conn: Optional database connection

    Returns:
        Game-level DataFrame with features
    """
    with AsofFeatureExtractor(feature_level=feature_level, conn=conn) as extractor:
        return extractor.extract_all(
            season_start=season_start,
            season_end=season_end,
            validate=validate,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate unified as-of features")
    parser.add_argument("--output", default="data/processed/features/asof_features_unified.csv")
    parser.add_argument("--level", choices=["base", "enhanced", "v2"], default="v2")
    parser.add_argument("--season-start", type=int, default=None)
    parser.add_argument("--season-end", type=int, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    print(f"=== Unified As-Of Feature Generation (level={args.level}) ===")
    df = get_asof_features(
        feature_level=args.level,
        season_start=args.season_start,
        season_end=args.season_end,
        validate=args.validate,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[SUCCESS] Wrote {len(df)} games, {len(df.columns)} columns to {args.output}")
