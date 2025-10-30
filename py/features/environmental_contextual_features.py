"""
Environmental & Contextual Feature Engineering (Phase 3 Extension)

Measures how environmental and contextual factors impact game outcomes.
Part of the feature engineering pipeline for predictive modeling.

This module quantifies environmental and situational factors:
- Thursday night home team advantage
- Travel direction (East to West, West to East - circadian effects)
- Indoor vs outdoor venue
- Weather conditions (temp, wind)
- Rest days advantage/disadvantage
- Division rivalry games
- Conference games
- Prime time games

Author: NFL Analytics System
Date: 2025-10-30
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from sqlalchemy import create_engine
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentalContextualEngine:
    """
    Engineer environmental and contextual game features.

    Tracks:
    - Thursday night effects (short rest, home advantage)
    - Travel direction and distance (circadian rhythm impacts)
    - Indoor/outdoor venue characteristics
    - Weather conditions
    - Rest days differentials
    - Rivalry and conference games
    """

    # NFL team timezones (for travel calculations)
    TEAM_TIMEZONES = {
        # Pacific (UTC-8)
        "SEA": "Pacific", "SF": "Pacific", "LA": "Pacific", "LV": "Pacific",
        # Mountain (UTC-7)
        "DEN": "Mountain", "ARI": "Mountain",
        # Central (UTC-6)
        "KC": "Central", "DAL": "Central", "HOU": "Central", "TEN": "Central",
        "NO": "Central", "MIN": "Central", "CHI": "Central", "GB": "Central",
        "DET": "Central",
        # Eastern (UTC-5)
        "BUF": "Eastern", "NE": "Eastern", "NYJ": "Eastern", "NYG": "Eastern",
        "PHI": "Eastern", "WAS": "Eastern", "BAL": "Eastern", "PIT": "Eastern",
        "CLE": "Eastern", "CIN": "Eastern", "IND": "Eastern", "JAX": "Eastern",
        "MIA": "Eastern", "TB": "Eastern", "CAR": "Eastern", "ATL": "Eastern",
    }

    # Timezone to numeric (for distance calculation)
    TZ_TO_NUMERIC = {
        "Pacific": -8,
        "Mountain": -7,
        "Central": -6,
        "Eastern": -5,
    }

    # Teams with indoor stadiums (climate controlled)
    INDOOR_TEAMS = {
        "ATL": "Dome",  # Mercedes-Benz Stadium
        "DET": "Dome",  # Ford Field
        "HOU": "Retractable",  # NRG Stadium
        "IND": "Dome",  # Lucas Oil Stadium
        "LA": "Dome",  # SoFi Stadium (though technically no roof, it's covered)
        "LV": "Dome",  # Allegiant Stadium
        "MIN": "Dome",  # U.S. Bank Stadium
        "NO": "Dome",  # Caesars Superdome
        "ARI": "Retractable",  # State Farm Stadium
        "DAL": "Retractable",  # AT&T Stadium
    }

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        """Initialize the environmental contextual engine."""
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_games_with_context(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """
        Load games with environmental and contextual information.

        Returns game-level data with venue, weather, and timing data.
        """
        logger.info(f"Loading games with context ({start_season}-{end_season})...")

        query = f"""
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.is_thursday_night,
            g.roof,
            g.surface,
            g.temp,
            g.wind,
            g.stadium,
            g.stadium_id,
            t_home.team_division as home_division,
            t_home.team_conf as home_conference,
            t_away.team_division as away_division,
            t_away.team_conf as away_conference
        FROM games g
        LEFT JOIN teams t_home ON g.home_team = t_home.team_abbr
        LEFT JOIN teams t_away ON g.away_team = t_away.team_abbr
        WHERE g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff
        """

        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} games with context")

        # Add derived features
        self._add_travel_features(df)
        self._add_venue_features(df)
        self._add_rivalry_features(df)
        self._add_weather_features(df)

        return df

    def _add_travel_features(self, df: pd.DataFrame) -> None:
        """
        Add travel direction and timezone features.

        Traveling from West to East (against the sun) has different effects
        than traveling East to West (with the sun).

        Research shows:
        - West to East travel (e.g., SEA to NYJ) = harder adjustment (losing time)
        - East to West travel (e.g., NE to LA) = easier adjustment (gaining time)
        """
        # Get timezone for each team
        df["home_tz"] = df["home_team"].map(self.TEAM_TIMEZONES)
        df["away_tz"] = df["away_team"].map(self.TEAM_TIMEZONES)

        df["home_tz_numeric"] = df["home_tz"].map(self.TZ_TO_NUMERIC)
        df["away_tz_numeric"] = df["away_tz"].map(self.TZ_TO_NUMERIC)

        # Travel direction
        # Positive = traveling East (away team is farther east)
        # Negative = traveling West (away team is farther west)
        df["tz_difference"] = df["home_tz_numeric"] - df["away_tz_numeric"]

        # Categorize travel
        df["travel_west_to_east"] = (df["tz_difference"] > 0).astype(int)  # Harder
        df["travel_east_to_west"] = (df["tz_difference"] < 0).astype(int)  # Easier
        df["no_timezone_travel"] = (df["tz_difference"] == 0).astype(int)

        # Number of timezone changes
        df["num_timezones_crossed"] = df["tz_difference"].abs()

        # Long distance travel (2+ timezones)
        df["long_distance_travel"] = (df["num_timezones_crossed"] >= 2).astype(int)

    def _add_venue_features(self, df: pd.DataFrame) -> None:
        """
        Add venue-related features (indoor/outdoor, dome, retractable).
        """
        # Indoor/outdoor classification
        df["venue_type"] = df["home_team"].map(
            lambda x: self.INDOOR_TEAMS.get(x, "Outdoor")
        )

        df["is_dome_game"] = (df["venue_type"] == "Dome").astype(int)
        df["is_retractable_roof"] = (df["venue_type"] == "Retractable").astype(int)
        df["is_outdoor_game"] = (df["venue_type"] == "Outdoor").astype(int)

        # Outdoor team playing indoors (adjustment)
        df["away_outdoor_at_indoor"] = (
            (df["venue_type"].isin(["Dome", "Retractable"])) &
            (~df["away_team"].isin(self.INDOOR_TEAMS.keys()))
        ).astype(int)

        # Indoor team playing outdoors (adjustment)
        df["away_indoor_at_outdoor"] = (
            (df["venue_type"] == "Outdoor") &
            (df["away_team"].isin(self.INDOOR_TEAMS.keys()))
        ).astype(int)

    def _add_rivalry_features(self, df: pd.DataFrame) -> None:
        """
        Add rivalry and conference features.

        Creates granular matchup categories:
        - Divisional (e.g., AFC North vs AFC North)
        - Conference non-divisional (e.g., AFC North vs AFC East)
        - Inter-conference (e.g., AFC North vs NFC South)
        - Specific division matchups (e.g., "AFC North vs NFC South")
        """
        # Division rivalry game (same division)
        df["is_division_game"] = (
            (df["home_division"] == df["away_division"]) &
            (df["home_conference"] == df["away_conference"])
        ).astype(int)

        # Conference game (same conference)
        df["is_conference_game"] = (
            df["home_conference"] == df["away_conference"]
        ).astype(int)

        # Inter-conference game (different conferences)
        df["is_interconference_game"] = (
            df["home_conference"] != df["away_conference"]
        ).astype(int)

        # Conference non-divisional (same conference, different division)
        df["is_conference_nondiv_game"] = (
            (df["home_conference"] == df["away_conference"]) &
            (df["home_division"] != df["away_division"])
        ).astype(int)

        # Create specific division matchup identifier
        # Format: "home_conf home_div vs away_conf away_div"
        df["division_matchup"] = (
            df["home_conference"].astype(str) + " " + df["home_division"].astype(str) +
            " vs " +
            df["away_conference"].astype(str) + " " + df["away_division"].astype(str)
        )

        # Create conference matchup identifier (ignoring division)
        # Format: "AFC vs NFC" or "AFC vs AFC"
        df["conference_matchup"] = (
            df["home_conference"].astype(str) + " vs " + df["away_conference"].astype(str)
        )

    def _add_weather_features(self, df: pd.DataFrame) -> None:
        """
        Add weather-related features.

        Note: Weather data may be missing for indoor games.
        """
        # Parse temperature (may be string or int)
        df["temp_numeric"] = pd.to_numeric(df["temp"], errors="coerce")

        # Parse wind (may have "mph" suffix)
        df["wind_numeric"] = df["wind"].astype(str).str.extract(r'(\d+)')[0].astype(float)

        # Weather categories
        df["is_cold_game"] = (df["temp_numeric"] < 40).astype(int)  # Below 40°F
        df["is_hot_game"] = (df["temp_numeric"] > 85).astype(int)   # Above 85°F
        df["is_windy_game"] = (df["wind_numeric"] > 15).astype(int)  # Wind > 15mph

        # Extreme weather
        df["extreme_weather"] = (
            (df["is_cold_game"] == 1) |
            (df["is_hot_game"] == 1) |
            (df["is_windy_game"] == 1)
        ).astype(int)

    def calculate_thursday_night_features(
        self,
        games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Thursday night game features.

        Thursday night games are known for:
        - Home team advantage (less travel impact)
        - Lower scoring (short rest, less preparation)
        - Higher variability
        """
        logger.info("Calculating Thursday night features...")

        results = []

        for game_id, row in games.iterrows():
            # Home team on Thursday night
            home_metrics = {
                "game_id": row["game_id"],
                "team": row["home_team"],
                "location": "home",
                "is_thursday_night": row["is_thursday_night"],
                "thursday_home_advantage": row["is_thursday_night"] if row["is_thursday_night"] else 0,
            }
            results.append(home_metrics)

            # Away team on Thursday night
            away_metrics = {
                "game_id": row["game_id"],
                "team": row["away_team"],
                "location": "away",
                "is_thursday_night": row["is_thursday_night"],
                "thursday_home_advantage": 0,  # Away team doesn't benefit
            }
            results.append(away_metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} Thursday night feature records")

        return df

    def calculate_travel_impact_features(
        self,
        games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate travel impact features for each team.

        Focus on circadian rhythm effects.
        """
        logger.info("Calculating travel impact features...")

        results = []

        for game_id, row in games.iterrows():
            # Home team (no travel)
            home_metrics = {
                "game_id": row["game_id"],
                "team": row["home_team"],
                "location": "home",
                "travel_west_to_east": 0,  # Home team doesn't travel
                "travel_east_to_west": 0,
                "num_timezones_crossed": 0,
                "long_distance_travel": 0,
            }
            results.append(home_metrics)

            # Away team (travels)
            away_metrics = {
                "game_id": row["game_id"],
                "team": row["away_team"],
                "location": "away",
                "travel_west_to_east": row["travel_west_to_east"],
                "travel_east_to_west": row["travel_east_to_west"],
                "num_timezones_crossed": row["num_timezones_crossed"],
                "long_distance_travel": row["long_distance_travel"],
            }
            results.append(away_metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} travel impact feature records")

        return df

    def calculate_venue_adaptation_features(
        self,
        games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate venue adaptation features.

        Indoor/outdoor transitions can impact team performance.
        """
        logger.info("Calculating venue adaptation features...")

        results = []

        for game_id, row in games.iterrows():
            # Home team venue characteristics
            home_metrics = {
                "game_id": row["game_id"],
                "team": row["home_team"],
                "location": "home",
                "is_dome_game": row["is_dome_game"],
                "is_outdoor_game": row["is_outdoor_game"],
                "venue_adaptation": 0,  # Home team is adapted
                "extreme_weather": row["extreme_weather"] if row["is_outdoor_game"] else 0,
            }
            results.append(home_metrics)

            # Away team venue adaptation
            away_metrics = {
                "game_id": row["game_id"],
                "team": row["away_team"],
                "location": "away",
                "is_dome_game": row["is_dome_game"],
                "is_outdoor_game": row["is_outdoor_game"],
                "venue_adaptation": row["away_outdoor_at_indoor"] or row["away_indoor_at_outdoor"],
                "extreme_weather": row["extreme_weather"] if row["is_outdoor_game"] else 0,
            }
            results.append(away_metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} venue adaptation feature records")

        return df

    def calculate_rivalry_features(
        self,
        games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rivalry and conference features.

        Division games are typically more competitive and lower scoring.
        Includes granular matchup permutations (e.g., AFC North vs NFC South).
        """
        logger.info("Calculating rivalry features...")

        results = []

        for game_id, row in games.iterrows():
            # Both teams have same rivalry context
            for team, opponent, location in [
                (row["home_team"], row["away_team"], "home"),
                (row["away_team"], row["home_team"], "away")
            ]:
                metrics = {
                    "game_id": row["game_id"],
                    "team": team,
                    "opponent": opponent,
                    "location": location,
                    "is_division_game": row["is_division_game"],
                    "is_conference_game": row["is_conference_game"],
                    "is_interconference_game": row["is_interconference_game"],
                    "is_conference_nondiv_game": row["is_conference_nondiv_game"],
                    "division_matchup": row["division_matchup"],
                    "conference_matchup": row["conference_matchup"],
                }
                results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} rivalry feature records")

        return df

    def generate_all_features(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all environmental and contextual features.

        Returns:
            Tuple of (thursday_night, travel_impact, venue_adaptation, rivalry)
        """
        logger.info("=" * 70)
        logger.info("Environmental & Contextual Feature Engineering")
        logger.info("=" * 70)

        # Load games with context
        games = self.load_games_with_context(start_season, end_season)

        # Calculate all features
        thursday_night = self.calculate_thursday_night_features(games)
        travel_impact = self.calculate_travel_impact_features(games)
        venue_adaptation = self.calculate_venue_adaptation_features(games)
        rivalry = self.calculate_rivalry_features(games)

        logger.info("=" * 70)
        logger.info("Feature generation complete!")
        logger.info(f"  Thursday night features: {len(thursday_night):,} records")
        logger.info(f"  Travel impact features: {len(travel_impact):,} records")
        logger.info(f"  Venue adaptation features: {len(venue_adaptation):,} records")
        logger.info(f"  Rivalry features: {len(rivalry):,} records")
        logger.info("=" * 70)

        return (
            thursday_night,
            travel_impact,
            venue_adaptation,
            rivalry
        )


def main():
    """Test the environmental contextual engine."""
    engine = EnvironmentalContextualEngine()

    # Generate ALL features for 2024-2025 seasons
    (
        thursday_night,
        travel_impact,
        venue_adaptation,
        rivalry
    ) = engine.generate_all_features(
        start_season=2024,
        end_season=2025
    )

    # Display sample statistics
    print("\n" + "=" * 70)
    print("THURSDAY NIGHT FEATURES SAMPLE")
    print("=" * 70)
    if len(thursday_night) > 0:
        thursday_games = thursday_night[thursday_night["is_thursday_night"] == True]
        print(f"Total Thursday night games: {len(thursday_games) // 2}")  # Divide by 2 (home + away)
        print(thursday_games.head(10).to_string())

    print("\n" + "=" * 70)
    print("TRAVEL IMPACT FEATURES SAMPLE")
    print("=" * 70)
    if len(travel_impact) > 0:
        long_travel = travel_impact[travel_impact["long_distance_travel"] == 1]
        print(f"Games with long distance travel (2+ timezones): {len(long_travel)}")
        print(long_travel.head(10).to_string())

        # Travel direction stats
        west_to_east = travel_impact["travel_west_to_east"].sum()
        east_to_west = travel_impact["travel_east_to_west"].sum()
        print(f"\nWest to East travel: {west_to_east} team-games")
        print(f"East to West travel: {east_to_west} team-games")

    print("\n" + "=" * 70)
    print("VENUE ADAPTATION FEATURES SAMPLE")
    print("=" * 70)
    if len(venue_adaptation) > 0:
        dome_games = venue_adaptation[venue_adaptation["is_dome_game"] == 1]
        outdoor_games = venue_adaptation[venue_adaptation["is_outdoor_game"] == 1]
        print(f"Dome games: {len(dome_games) // 2}")
        print(f"Outdoor games: {len(outdoor_games) // 2}")

        adaptation_needed = venue_adaptation[venue_adaptation["venue_adaptation"] == 1]
        print(f"Games requiring venue adaptation: {len(adaptation_needed)}")
        print(adaptation_needed.head(10).to_string())

    print("\n" + "=" * 70)
    print("RIVALRY FEATURES SAMPLE")
    print("=" * 70)
    if len(rivalry) > 0:
        division_games = rivalry[rivalry["is_division_game"] == 1]
        conference_nondiv_games = rivalry[rivalry["is_conference_nondiv_game"] == 1]
        interconference_games = rivalry[rivalry["is_interconference_game"] == 1]

        print(f"Division games: {len(division_games) // 2}")
        print(f"Conference non-divisional games: {len(conference_nondiv_games) // 2}")
        print(f"Inter-conference games: {len(interconference_games) // 2}")

        print("\nDivision games sample:")
        print(division_games.head(5).to_string())

        print("\n" + "=" * 70)
        print("UNIQUE DIVISION MATCHUPS")
        print("=" * 70)
        matchup_counts = rivalry.groupby("division_matchup").size().sort_values(ascending=False)
        print(matchup_counts.head(15).to_string())

        print("\n" + "=" * 70)
        print("CONFERENCE MATCHUP BREAKDOWN")
        print("=" * 70)
        conf_counts = rivalry.groupby("conference_matchup").size().sort_values(ascending=False)
        print(conf_counts.to_string())

    return (
        thursday_night,
        travel_impact,
        venue_adaptation,
        rivalry
    )


if __name__ == "__main__":
    main()
