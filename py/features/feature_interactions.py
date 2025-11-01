#!/usr/bin/env python3
"""
Feature Interaction Engineering for NFL Analytics

Captures synergistic effects between features that boost predictive power and betting ROI.
Individual features may not be predictive, but combinations can reveal edge.

Interaction Categories:
1. Environmental × Travel × Venue (weather + travel + dome teams)
2. Rest × Distance (Thursday night + cross-country)
3. Vegas × Context (line efficiency in specific situations)
4. Matchup × Situation (strength interactions with game script)
5. Climate × Adaptation (warm vs cold weather teams)

Author: Claude Code
Date: 2025-10-30
"""

import logging
import numpy as np
import pandas as pd
import psycopg
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Team home climate classifications
WARM_WEATHER_TEAMS = {"MIA", "TB", "JAX", "NO", "ATL", "CAR", "HOU", "DAL", "ARI", "LA", "LV", "SF"}
COLD_WEATHER_TEAMS = {"BUF", "GB", "CHI", "DET", "MIN", "NE", "NYJ", "NYG", "CLE", "PIT", "BAL", "CIN"}
DOME_TEAMS = {"NO", "ATL", "DET", "MIN", "DAL", "ARI", "LA", "LV"}
HIGH_ALTITUDE_TEAMS = {"DEN"}


class FeatureInteractionEngine:
    """
    Generates interaction features that capture synergistic effects.

    Key insight: Individual features may have low predictive power,
    but combinations can reveal significant edge for betting.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_game_context(self) -> pd.DataFrame:
        """Load comprehensive game context with all relevant features."""
        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.kickoff,
            EXTRACT(DOW FROM g.kickoff) as day_of_week,
            EXTRACT(HOUR FROM g.kickoff) as game_hour,
            g.home_team,
            g.away_team,
            g.stadium,
            g.roof,
            g.surface,
            g.temp,
            g.wind,
            g.away_rest,
            g.home_rest,
            g.away_score,
            g.home_score,
            g.spread_close,
            g.total_close,
            g.home_moneyline,
            g.away_moneyline,
            g.is_thursday_night,
            t_home.team_abbr as home_abbr,
            t_home.team_conf as home_conference,
            t_home.team_division as home_division,
            t_away.team_abbr as away_abbr,
            t_away.team_conf as away_conference,
            t_away.team_division as away_division
        FROM games g
        LEFT JOIN teams t_home ON g.home_team = t_home.team_abbr
        LEFT JOIN teams t_away ON g.away_team = t_away.team_abbr
        WHERE g.season >= 2020 AND g.season <= 2025
        ORDER BY g.kickoff, g.game_id
        """

        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql_query(query, conn)

        # Derive weekday from day_of_week (0=Sunday, 1=Monday, ..., 6=Saturday)
        weekday_map = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
                      4: "Thursday", 5: "Friday", 6: "Saturday"}
        df["weekday"] = df["day_of_week"].map(weekday_map)

        # Create gametime column from hour (for prime time detection)
        df["gametime"] = df["game_hour"].apply(lambda x: f"{int(x)}:00" if pd.notna(x) else None)

        # Parse temperature and wind (stored as TEXT in database)
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df["wind"] = pd.to_numeric(df["wind"].astype(str).str.extract(r'(\d+)')[0], errors="coerce")

        self.logger.info(f"Loaded {len(df)} games with context")
        return df

    def _create_weather_travel_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interaction 1: Weather × Travel × Venue Adaptation

        Hypothesis: Weather effects are amplified for:
        - Dome teams playing outdoors
        - Teams traveling long distances
        - Warm weather teams in cold conditions (and vice versa)
        """
        result = df.copy()

        # Classify weather severity
        result["extreme_cold"] = (result["temp"] < 25).astype(int)
        result["extreme_heat"] = (result["temp"] > 85).astype(int)
        result["high_wind"] = (result["wind"] > 15).astype(int)
        result["outdoor_game"] = (result["roof"] == "outdoors").astype(int)

        # Team climate background
        result["away_is_dome_team"] = result["away_team"].isin(DOME_TEAMS).astype(int)
        result["home_is_dome_team"] = result["home_team"].isin(DOME_TEAMS).astype(int)
        result["away_is_warm_climate"] = result["away_team"].isin(WARM_WEATHER_TEAMS).astype(int)
        result["away_is_cold_climate"] = result["away_team"].isin(COLD_WEATHER_TEAMS).astype(int)

        # === PRIMARY INTERACTIONS ===

        # 1. Dome team playing outdoors in extreme weather
        result["dome_team_outdoor_cold"] = (
            result["away_is_dome_team"] *
            result["outdoor_game"] *
            result["extreme_cold"]
        )
        result["dome_team_outdoor_wind"] = (
            result["away_is_dome_team"] *
            result["outdoor_game"] *
            result["high_wind"]
        )

        # 2. Warm weather team in extreme cold (e.g., MIA/TB @ BUF/GB)
        result["warm_team_in_cold"] = (
            result["away_is_warm_climate"] *
            result["extreme_cold"]
        )

        # 3. Cold weather team in extreme heat (e.g., GB/NE @ MIA/TB)
        result["cold_team_in_heat"] = (
            result["away_is_cold_climate"] *
            result["extreme_heat"]
        )

        # 4. Weather intensity score (composite)
        result["weather_severity"] = (
            result["extreme_cold"].astype(float) +
            result["extreme_heat"].astype(float) +
            result["high_wind"].astype(float) * 0.5
        )

        return result

    def _create_rest_travel_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interaction 2: Rest Days × Travel Distance

        Hypothesis: Short rest + long travel = compounding fatigue
        Key case: Thursday night + cross-country travel
        """
        result = df.copy()

        # Calculate travel burden
        result["is_thursday_night"] = (result["weekday"] == "Thursday").astype(int)
        result["short_rest"] = (result["away_rest"] <= 5).astype(int)
        result["very_short_rest"] = (result["away_rest"] <= 4).astype(int)

        # Travel classification (would need team locations - simplified here)
        # Approximate: East (1), Central (2), Mountain (3), Pacific (4)
        zone_map = {
            **{team: 4 for team in ["SEA", "SF", "LA", "LV"]},
            **{team: 3 for team in ["DEN", "ARI"]},
            **{team: 2 for team in ["KC", "DAL", "HOU", "TEN", "NO", "MIN", "CHI", "GB", "DET"]},
            **{team: 1 for team in ["BUF", "NE", "NYJ", "NYG", "PHI", "WAS", "BAL", "PIT", "CLE", "CIN", "IND", "JAX", "MIA", "TB", "CAR", "ATL"]},
        }

        result["home_zone"] = result["home_team"].map(zone_map)
        result["away_zone"] = result["away_team"].map(zone_map)
        result["zone_distance"] = (result["home_zone"] - result["away_zone"]).abs()

        # === PRIMARY INTERACTIONS ===

        # 1. Thursday night + cross-country travel (2+ time zones)
        result["thursday_long_travel"] = (
            result["is_thursday_night"] *
            (result["zone_distance"] >= 2).astype(int)
        )

        # 2. Short rest + any significant travel
        result["short_rest_with_travel"] = (
            result["short_rest"] *
            result["zone_distance"]
        )

        # 3. Very short rest + cross-country
        result["critical_fatigue"] = (
            result["very_short_rest"] *
            (result["zone_distance"] >= 2).astype(int)
        )

        # 4. Rest disadvantage (away team worse rest than home)
        result["rest_disadvantage"] = np.maximum(
            0,
            result["home_rest"] - result["away_rest"]
        )

        return result

    def _create_altitude_rest_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interaction 3: Altitude × Rest

        Hypothesis: Playing in Denver (5,280 ft) is harder with short rest
        """
        result = df.copy()

        result["playing_at_altitude"] = (result["home_team"] == "DEN").astype(int)
        result["away_short_rest"] = (result["away_rest"] <= 5).astype(int)

        # === PRIMARY INTERACTION ===
        result["altitude_with_short_rest"] = (
            result["playing_at_altitude"] *
            result["away_short_rest"]
        )

        return result

    def _create_vegas_context_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interaction 4: Vegas Lines × Game Context

        Hypothesis: Line efficiency varies by context:
        - Division games (more familiarity)
        - Prime time games (more public action)
        - Weather games (line may not fully adjust)
        """
        result = df.copy()

        # Game context
        result["is_division_game"] = (
            result["home_division"] == result["away_division"]
        ).astype(int)
        result["is_prime_time"] = result["gametime"].isin(["8:20", "8:15"]).astype(int) if "gametime" in result.columns else 0

        # Line characteristics
        result["is_favorite"] = (result["spread_close"] < 0).astype(int)
        result["large_spread"] = (result["spread_close"].abs() >= 7).astype(int)
        result["small_spread"] = (result["spread_close"].abs() <= 3).astype(int)

        # === PRIMARY INTERACTIONS ===

        # 1. Division game + small spread (high variance)
        result["division_toss_up"] = (
            result["is_division_game"] *
            result["small_spread"]
        )

        # 2. Large favorite in bad weather (harder to cover)
        if "weather_severity" in result.columns:
            result["big_favorite_bad_weather"] = (
                result["large_spread"] *
                result["weather_severity"]
            )

        # 3. Prime time + public favorite (potential trap game)
        result["prime_time_favorite"] = (
            result["is_prime_time"] *
            result["is_favorite"]
        )

        return result

    def _create_matchup_situation_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interaction 5: Matchup Strength × Game Situation

        Hypothesis: Matchup advantages are amplified in specific situations:
        - Strong pass defense × bad weather (forces running)
        - Strong run defense × trailing team (forced to pass)
        """
        result = df.copy()

        # Situation indicators
        if "weather_severity" in result.columns:
            result["weather_favors_running"] = (result["weather_severity"] >= 1).astype(int)

        # Note: Would need defensive strength metrics from other modules
        # Placeholder for integration with defensive features
        result["pass_rush_advantage"] = 0  # To be filled from defensive metrics
        result["run_def_advantage"] = 0     # To be filled from defensive metrics

        return result

    def calculate_interaction_features(self) -> pd.DataFrame:
        """
        Generate all interaction features.

        Returns DataFrame with game_id and interaction features.
        """
        self.logger.info("=" * 70)
        self.logger.info("Feature Interaction Engineering")
        self.logger.info("=" * 70)

        # Load base data
        df = self._load_game_context()

        # Generate interaction categories
        df = self._create_weather_travel_interactions(df)
        df = self._create_rest_travel_interactions(df)
        df = self._create_altitude_rest_interactions(df)
        df = self._create_vegas_context_interactions(df)
        df = self._create_matchup_situation_interactions(df)

        self.logger.info(f"Generated {len(df)} game records with interaction features")

        return df

    def get_high_value_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to games with high-value interaction patterns.

        These are situations where feature interactions suggest betting edge.
        """
        # Define high-value patterns
        high_value = df[
            (df["dome_team_outdoor_cold"] == 1) |
            (df["warm_team_in_cold"] == 1) |
            (df["thursday_long_travel"] == 1) |
            (df["altitude_with_short_rest"] == 1) |
            (df["critical_fatigue"] == 1) |
            (df.get("big_favorite_bad_weather", 0) >= 1)
        ].copy()

        self.logger.info(f"Identified {len(high_value)} high-value interaction games")
        return high_value


def main():
    """Main execution function."""
    # Database connection
    db_url = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

    # Initialize engine
    engine = FeatureInteractionEngine(db_url)

    # Generate interaction features
    interactions_df = engine.calculate_interaction_features()

    # Display summary statistics
    logger.info("=" * 70)
    logger.info("INTERACTION FEATURE SUMMARY")
    logger.info("=" * 70)

    # Weather × Travel interactions
    logger.info("\n1. Weather × Travel Interactions:")
    logger.info(f"   Dome teams playing outdoor cold: {interactions_df['dome_team_outdoor_cold'].sum()}")
    logger.info(f"   Dome teams playing outdoor wind: {interactions_df['dome_team_outdoor_wind'].sum()}")
    logger.info(f"   Warm teams in extreme cold: {interactions_df['warm_team_in_cold'].sum()}")
    logger.info(f"   Cold teams in extreme heat: {interactions_df['cold_team_in_heat'].sum()}")

    # Rest × Travel interactions
    logger.info("\n2. Rest × Travel Interactions:")
    logger.info(f"   Thursday night + long travel: {interactions_df['thursday_long_travel'].sum()}")
    logger.info(f"   Critical fatigue situations: {interactions_df['critical_fatigue'].sum()}")
    logger.info(f"   Short rest with travel (avg): {interactions_df['short_rest_with_travel'].mean():.2f}")

    # Altitude interactions
    logger.info("\n3. Altitude Interactions:")
    logger.info(f"   Playing at altitude with short rest: {interactions_df['altitude_with_short_rest'].sum()}")

    # Vegas context interactions
    logger.info("\n4. Vegas × Context Interactions:")
    logger.info(f"   Division toss-up games: {interactions_df['division_toss_up'].sum()}")
    logger.info(f"   Prime time favorites: {interactions_df['prime_time_favorite'].sum()}")

    # High-value opportunities
    logger.info("\n5. High-Value Betting Opportunities:")
    high_value_games = engine.get_high_value_interactions(interactions_df)

    if len(high_value_games) > 0:
        logger.info(f"   Total high-value games: {len(high_value_games)}")
        logger.info("\n   Sample high-value games:")
        display_cols = ["game_id", "away_team", "home_team", "temp", "away_rest",
                        "dome_team_outdoor_cold", "thursday_long_travel"]
        available_cols = [col for col in display_cols if col in high_value_games.columns]
        logger.info(f"\n{high_value_games[available_cols].head(10).to_string()}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
