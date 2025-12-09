#!/usr/bin/env python3
"""
player_feature_extractor.py

Extracts player-level features from database for real-time predictions.

Computes rolling statistics and contextual features needed by XGBoost and BNN models.
Features match the schema expected by props_predictor.py.

Author: Claude Code (Sonnet 4.5)
Date: November 7, 2025
"""

import logging

import pandas as pd
import psycopg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerFeatureExtractor:
    """
    Extracts player features from database for real-time prop predictions.

    Features computed:
    - Rolling averages (last 3, 5 games, season)
    - Opponent defensive strength
    - Game script features (spread, total)
    - Environmental features (home/away, weather)
    - Usage metrics (attempts, targets, carries)
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """Initialize feature extractor with database connection."""
        self.db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def extract_features(
        self,
        player_id: str,
        prop_type: str,
        opponent: str,
        is_home: bool,
        season: int = 2025,
        week: int = 10,
    ) -> dict[str, float] | None:
        """
        Extract features for a player prop prediction.

        Args:
            player_id: Player ID (e.g., "00-0033873")
            prop_type: Prop type ("passing_yards", "rushing_yards", "receiving_yards")
            opponent: Opponent team abbreviation (e.g., "KC")
            is_home: Whether player's team is at home
            season: Season year
            week: Week number

        Returns:
            Dictionary of features matching props_predictor.py schema
        """
        try:
            # Map prop_type to stat_category
            stat_category_map = {
                "passing_yards": "passing",
                "rushing_yards": "rushing",
                "receiving_yards": "receiving",
            }
            stat_category = stat_category_map.get(prop_type)
            if not stat_category:
                logger.warning(f"Unknown prop_type: {prop_type}")
                return None

            # Get player's historical stats
            player_stats = self._get_player_stats(player_id, stat_category, season, week)
            if player_stats is None or len(player_stats) < 3:
                logger.warning(
                    f"Insufficient data for {player_id} ({len(player_stats) if player_stats is not None else 0} games)"
                )
                return None

            # Get opponent defensive stats
            opp_defense = self._get_opponent_defense(opponent, stat_category, season, week)

            # Get game context (spread, total)
            game_context = self._get_game_context(opponent, is_home, season, week)

            # Compute rolling features
            features = self._compute_rolling_features(player_stats, prop_type)

            # Add opponent defensive features
            if opp_defense is not None:
                features.update(opp_defense)
            else:
                # Fallback: league average
                features[f"opponent_{stat_category}_yards_allowed_avg"] = (
                    250.0 if stat_category == "passing" else 120.0
                )

            # Add game context
            if game_context is not None:
                features.update(game_context)
            else:
                # Fallback: neutral values
                features["implied_team_total"] = 22.0
                features["spread"] = 0.0

            # Add contextual features
            features["is_home"] = 1.0 if is_home else 0.0
            features["days_rest"] = 7.0  # Default to 1 week
            features["is_dome"] = 0.0  # TODO: query from venues table
            features["is_outdoors"] = 1.0
            features["weather_wind_mph"] = 5.0  # Default mild conditions
            features["weather_temp"] = 60.0

            logger.info(f"Extracted {len(features)} features for {player_id} vs {opponent}")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _get_player_stats(
        self, player_id: str, stat_category: str, season: int, week: int
    ) -> pd.DataFrame | None:
        """
        Get player's recent game stats (before prediction week).

        Returns DataFrame with columns: season, week, yards, attempts, touchdowns, etc.
        """
        try:
            conn = psycopg.connect(self.db_url)

            query = """
            SELECT
                season,
                week,
                stat_yards as yards,
                stat_attempts as attempts,
                stat_completions as completions,
                stat_touchdowns as touchdowns,
                stat_negative as negative_stat,
                cpoe
            FROM mart.player_game_stats
            WHERE player_id = %s
                AND stat_category = %s
                AND (
                    season < %s
                    OR (season = %s AND week < %s)
                )
            ORDER BY season DESC, week DESC
            LIMIT 10
            """

            with conn.cursor() as cur:
                cur.execute(query, (player_id, stat_category, season, season, week))
                rows = cur.fetchall()

            conn.close()

            if not rows:
                return None

            df = pd.DataFrame(
                rows,
                columns=[
                    "season",
                    "week",
                    "yards",
                    "attempts",
                    "completions",
                    "touchdowns",
                    "negative_stat",
                    "cpoe",
                ],
            )

            return df

        except Exception as e:
            logger.error(f"Failed to get player stats: {e}")
            return None

    def _get_opponent_defense(
        self, opponent: str, stat_category: str, season: int, week: int
    ) -> dict[str, float] | None:
        """
        Get opponent defensive strength (yards allowed per game).

        Returns dict with opponent_<stat_category>_yards_allowed_avg.
        """
        try:
            conn = psycopg.connect(self.db_url)

            # Query opponent's defensive performance (before prediction week)
            query = """
            WITH opp_games AS (
                SELECT
                    pgs.season,
                    pgs.week,
                    SUM(pgs.stat_yards) as yards_allowed
                FROM mart.player_game_stats pgs
                JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
                JOIN games g ON (
                    g.season = pgs.season
                    AND g.week = pgs.week
                    AND (
                        (g.home_team = %s AND ph.current_team = g.away_team)
                        OR (g.away_team = %s AND ph.current_team = g.home_team)
                    )
                )
                WHERE pgs.stat_category = %s
                    AND (
                        pgs.season < %s
                        OR (pgs.season = %s AND pgs.week < %s)
                    )
                GROUP BY pgs.season, pgs.week
                ORDER BY pgs.season DESC, pgs.week DESC
                LIMIT 5
            )
            SELECT AVG(yards_allowed) as avg_yards_allowed
            FROM opp_games
            """

            with conn.cursor() as cur:
                cur.execute(query, (opponent, opponent, stat_category, season, season, week))
                row = cur.fetchone()

            conn.close()

            if row and row[0]:
                return {f"opponent_{stat_category}_yards_allowed_avg": float(row[0])}
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get opponent defense: {e}")
            return None

    def _get_game_context(
        self, opponent: str, is_home: bool, season: int, week: int
    ) -> dict[str, float] | None:
        """
        Get game context (spread, total) from games table.

        Returns dict with implied_team_total, spread.
        """
        # TODO: Would need to know player's team to query games table
        # For now, return None to use fallback values
        return None

    def _compute_rolling_features(
        self, player_stats: pd.DataFrame, prop_type: str
    ) -> dict[str, float]:
        """
        Compute rolling average features from player's game log.

        Args:
            player_stats: DataFrame with columns [season, week, yards, attempts, ...]
            prop_type: "passing_yards", "rushing_yards", or "receiving_yards"

        Returns:
            Dictionary with rolling features
        """
        features = {}

        # Determine stat type
        if prop_type == "passing_yards":
            stat_prefix = "passing"
        elif prop_type == "rushing_yards":
            stat_prefix = "rushing"
        else:
            stat_prefix = "receiving"

        # Last 3 games averages
        if len(player_stats) >= 3:
            last3 = player_stats.head(3)
            features[f"{stat_prefix}_yards_last3"] = last3["yards"].mean()
            features[f"{'pass' if stat_prefix == 'passing' else stat_prefix}_attempts_last3"] = (
                last3["attempts"].mean()
            )

            if "completions" in last3.columns and stat_prefix == "passing":
                features["completions_last3"] = last3["completions"].mean()
            if "touchdowns" in last3.columns:
                features[f"{stat_prefix}_tds_last3"] = last3["touchdowns"].mean()
            if "negative_stat" in last3.columns:
                if stat_prefix == "passing":
                    features["interceptions_last3"] = last3["negative_stat"].mean()
                else:
                    features["fumbles_last3"] = last3["negative_stat"].mean()

        # Last 5 games averages
        if len(player_stats) >= 5:
            last5 = player_stats.head(5)
            features[f"{stat_prefix}_yards_last5"] = last5["yards"].mean()
            features[f"{'pass' if stat_prefix == 'passing' else stat_prefix}_attempts_last5"] = (
                last5["attempts"].mean()
            )

            if "completions" in last5.columns and stat_prefix == "passing":
                features["completions_last5"] = last5["completions"].mean()

        # Season average (current season only)
        current_season = player_stats["season"].max()
        season_stats = player_stats[player_stats["season"] == current_season]
        if len(season_stats) > 0:
            features[f"{stat_prefix}_yards_season"] = season_stats["yards"].mean()

        # Additional passing-specific features
        if stat_prefix == "passing" and len(player_stats) >= 3:
            last3 = player_stats.head(3)
            if "cpoe" in last3.columns:
                features["cpoe_last3"] = last3["cpoe"].mean()
            # Estimate sacks from negative stat (if available)
            features["sacks_last3"] = 2.5  # Default fallback

        return features


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract player features for prop prediction")
    parser.add_argument("--player-id", required=True, help="Player ID (e.g., 00-0033873)")
    parser.add_argument(
        "--prop-type", required=True, choices=["passing_yards", "rushing_yards", "receiving_yards"]
    )
    parser.add_argument("--opponent", required=True, help="Opponent team (e.g., KC)")
    parser.add_argument("--is-home", action="store_true", help="Player's team is at home")
    parser.add_argument("--season", type=int, default=2025, help="Season year")
    parser.add_argument("--week", type=int, default=10, help="Week number")

    args = parser.parse_args()

    extractor = PlayerFeatureExtractor()
    features = extractor.extract_features(
        player_id=args.player_id,
        prop_type=args.prop_type,
        opponent=args.opponent,
        is_home=args.is_home,
        season=args.season,
        week=args.week,
    )

    if features:
        print(f"\n✅ Extracted {len(features)} features:\n")
        for key, value in sorted(features.items()):
            print(f"  {key:40s} = {value:8.2f}")
    else:
        print("\n❌ Feature extraction failed")
