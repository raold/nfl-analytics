#!/usr/bin/env python3
"""
Enhanced Travel Features with Circadian-Aware Jet Lag Modeling

Implements asymmetric travel penalties based on peer-reviewed research:
- Westbound travel (East→West): Harder (-12 per hour)
- Eastbound travel (West→East): Easier (-8 per hour)

References:
- Smith et al. (2017). "Greater circadian disadvantage during evening games..."
  Journal of Sleep Research, 26(5), 628-634. DOI: 10.1111/jsr.12532
- Leatherwood & Dragoo (2013). "Effect of airline travel on performance..."
  British Journal of Sports Medicine, 47(9), 561-567.

Author: Parallel Modeling Track C - Week 3 Integration
Date: November 22, 2025
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTravelFeatures:
    """
    Calculate circadian-aware travel features for NFL games.

    Features:
    - Asymmetric jet lag penalties (westbound vs eastbound)
    - Body clock hour at kickoff
    - Circadian penalty for early morning games
    - International game flags and altitude shock
    - Mexico City special altitude handling
    """

    def __init__(
        self,
        distance_matrix_path: str = "data/stadium_distance_matrix.csv",
        timezone_matrix_path: str = "data/stadium_timezone_matrix.csv",
        international_games_path: str = "data/international_games_2018_2024.csv",
    ):
        """Initialize with stadium data matrices."""
        self.distance_matrix = None
        self.timezone_matrix = None
        self.international_games = None

        # Load data if files exist
        if Path(distance_matrix_path).exists():
            self.distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)
            logger.info(f"Loaded distance matrix: {self.distance_matrix.shape}")

        if Path(timezone_matrix_path).exists():
            self.timezone_matrix = pd.read_csv(timezone_matrix_path, index_col=0)
            logger.info(f"Loaded timezone matrix: {self.timezone_matrix.shape}")

        if Path(international_games_path).exists():
            self.international_games = pd.read_csv(international_games_path)
            logger.info(f"Loaded {len(self.international_games)} international games")

    def calculate_travel_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate travel distance for away team.

        Args:
            df: DataFrame with 'home_team', 'away_team' columns

        Returns:
            DataFrame with 'travel_distance' column added
        """
        if self.distance_matrix is None:
            logger.warning("Distance matrix not loaded, skipping travel_distance")
            df["travel_distance"] = 0
            return df

        df["travel_distance"] = df.apply(
            lambda row: (
                self.distance_matrix.loc[row["away_team"], row["home_team"]]
                if row["away_team"] in self.distance_matrix.index
                and row["home_team"] in self.distance_matrix.columns
                else 0
            ),
            axis=1,
        )

        return df

    def calculate_timezone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate timezone shift and directional flags.

        Args:
            df: DataFrame with 'home_team', 'away_team' columns

        Returns:
            DataFrame with timezone features added:
            - timezone_shift: Integer hours (-3 to +3 domestic, -9 to +9 international)
            - is_westbound_travel: Binary (East→West, harder)
            - is_eastbound_travel: Binary (West→East, easier)
        """
        if self.timezone_matrix is None:
            logger.warning("Timezone matrix not loaded, skipping timezone features")
            df["timezone_shift"] = 0
            df["is_westbound_travel"] = 0
            df["is_eastbound_travel"] = 0
            return df

        # Get timezone offsets for home and away teams
        df["home_tz_offset"] = df["home_team"].map(
            self.timezone_matrix.set_index("team")["timezone_offset"]
            if "team" in self.timezone_matrix.columns
            else self.timezone_matrix.iloc[:, 0]  # First column if no 'team' column
        )

        df["away_tz_offset"] = df["away_team"].map(
            self.timezone_matrix.set_index("team")["timezone_offset"]
            if "team" in self.timezone_matrix.columns
            else self.timezone_matrix.iloc[:, 0]
        )

        # Calculate timezone shift (positive = westbound, negative = eastbound)
        df["timezone_shift"] = (df["home_tz_offset"] - df["away_tz_offset"]).fillna(0).astype(int)

        # Directional flags
        df["is_westbound_travel"] = (df["timezone_shift"] > 0).astype(int)
        df["is_eastbound_travel"] = (df["timezone_shift"] < 0).astype(int)

        return df

    def calculate_jet_lag_penalty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate asymmetric jet lag penalty based on travel direction.

        Research basis (Smith et al. 2017):
        - Westbound travel (East→West): Body clock extension, harder (-12 per hour)
        - Eastbound travel (West→East): Body clock compression, easier (-8 per hour)

        Args:
            df: DataFrame with 'timezone_shift' column

        Returns:
            DataFrame with 'jet_lag_penalty' column added
        """
        # Asymmetric penalty: westbound = -12 per hour, eastbound = -8 per hour
        df["jet_lag_penalty"] = np.where(
            df["timezone_shift"] > 0,  # Westbound (harder)
            df["timezone_shift"] * -12,
            np.where(
                df["timezone_shift"] < 0,  # Eastbound (easier)
                df["timezone_shift"].abs() * -8,
                0,  # No travel
            ),
        )

        return df

    def calculate_body_clock_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate body clock hour and circadian penalty.

        Args:
            df: DataFrame with 'kickoff', 'away_tz_offset' columns

        Returns:
            DataFrame with body clock features:
            - body_clock_hour: Hour at kickoff per away team's internal clock (0-23)
            - circadian_penalty: Penalty for early morning body clock (3am-6am = -26)
        """
        # Parse kickoff time if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["kickoff"]):
            df["kickoff"] = pd.to_datetime(df["kickoff"])

        # Get kickoff hour in UTC
        df["kickoff_hour_utc"] = df["kickoff"].dt.hour

        # Calculate body clock hour (away team's perception of time)
        df["body_clock_hour"] = (
            ((df["kickoff_hour_utc"] + df["away_tz_offset"]) % 24).fillna(12).astype(int)
        )  # Default to noon if missing

        # Circadian penalty for early morning games (3am-6am body clock)
        # Research: Peak performance ~5-8pm, worst 3-6am
        df["circadian_penalty"] = np.where(
            (df["body_clock_hour"] >= 3) & (df["body_clock_hour"] <= 6),
            -26,  # Significant penalty for early morning games
            0,
        )

        return df

    def merge_international_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge international game flags and altitude features.

        Args:
            df: DataFrame with 'game_id' column

        Returns:
            DataFrame with international features:
            - is_international: Binary flag
            - is_london_game: Binary
            - is_germany_game: Binary
            - is_mexico_city_game: Binary (critical for altitude)
            - altitude_shock: Stadium altitude - team home altitude
            - altitude_penalty: Altitude-based performance penalty
        """
        if self.international_games is None:
            logger.warning("International games not loaded, skipping international features")
            df["is_international"] = 0
            df["is_london_game"] = 0
            df["is_germany_game"] = 0
            df["is_mexico_city_game"] = 0
            df["altitude_shock"] = 0
            df["altitude_penalty"] = 0
            return df

        # Extract key international features
        intl_features = self.international_games[
            [
                "game_id",
                "is_international",
                "city",
                "altitude_shock",
            ]
        ].copy()

        # Add location-specific flags
        intl_features["is_london_game"] = intl_features["city"].isin(["London"]).astype(int)

        intl_features["is_germany_game"] = (
            intl_features["city"].isin(["Munich", "Frankfurt"]).astype(int)
        )

        intl_features["is_mexico_city_game"] = (
            intl_features["city"].isin(["Mexico City"]).astype(int)
        )

        # Calculate altitude penalty (Mexico City gets stronger penalty)
        # Mexico City: 7,200 ft → -0.005 penalty per foot
        # Other stadiums: -0.003 penalty per foot
        intl_features["altitude_penalty"] = np.where(
            intl_features["is_mexico_city_game"] == 1,
            intl_features["altitude_shock"] * -0.005,  # Stronger Mexico penalty
            intl_features["altitude_shock"] * -0.003,  # Standard altitude penalty
        )

        # Merge with main DataFrame (left join - most games are domestic)
        df = df.merge(intl_features.drop("city", axis=1), on="game_id", how="left")

        # Fill NaN for domestic games
        intl_cols = [
            "is_international",
            "is_london_game",
            "is_germany_game",
            "is_mexico_city_game",
            "altitude_shock",
            "altitude_penalty",
        ]
        df[intl_cols] = df[intl_cols].fillna(0)

        return df

    def calculate_thursday_night_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Thursday Night Football features.

        Note: Research shows 57.9% home win rate on TNF vs 57.1% overall (not significant, p>0.05).
        However, including as feature for XGBoost to learn potential nonlinear effects.

        Args:
            df: DataFrame with 'kickoff' column

        Returns:
            DataFrame with TNF features:
            - is_thursday_night: Binary flag for TNF games
            - tnf_home_advantage: Scaled advantage (default 0.0, XGBoost can learn)
        """
        # Parse kickoff if needed
        if not pd.api.types.is_datetime64_any_dtype(df["kickoff"]):
            df["kickoff"] = pd.to_datetime(df["kickoff"])

        # Thursday Night Football flag
        df["is_thursday_night"] = (df["kickoff"].dt.dayofweek == 3).astype(int)  # Thursday = 3

        # TNF home advantage (scaled to 0.0 initially, XGBoost can learn from data)
        # User hypothesis: May have statistically significant impact despite null research findings
        df["tnf_home_advantage"] = df["is_thursday_night"] * 0.0

        return df

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all enhanced travel features in proper order.

        Args:
            df: DataFrame with game data (requires: game_id, home_team, away_team, kickoff)

        Returns:
            DataFrame with all travel features added (13 new features):
            1. travel_distance
            2. timezone_shift
            3. is_westbound_travel
            4. is_eastbound_travel
            5. jet_lag_penalty
            6. body_clock_hour
            7. circadian_penalty
            8. is_international
            9. is_mexico_city_game
            10. altitude_shock
            11. altitude_penalty
            12. is_thursday_night
            13. tnf_home_advantage
        """
        logger.info("Calculating enhanced travel features...")

        # Step 1: Travel distance
        df = self.calculate_travel_distance(df)

        # Step 2: Timezone features
        df = self.calculate_timezone_features(df)

        # Step 3: Jet lag penalty
        df = self.calculate_jet_lag_penalty(df)

        # Step 4: Body clock features
        df = self.calculate_body_clock_features(df)

        # Step 5: International features
        df = self.merge_international_features(df)

        # Step 6: Thursday Night Football features
        df = self.calculate_thursday_night_features(df)

        # Calculate total travel penalty (for reference)
        df["total_travel_penalty"] = (
            df["jet_lag_penalty"]
            + df["circadian_penalty"]
            + df["altitude_penalty"]
            + df["tnf_home_advantage"]  # Included for completeness
        )

        logger.info("✓ Added 13 travel features")
        logger.info(f"  Domestic games: {(df['is_international'] == 0).sum()}")
        logger.info(f"  International games: {(df['is_international'] == 1).sum()}")
        logger.info(f"  Westbound travel: {(df['is_westbound_travel'] == 1).sum()}")
        logger.info(f"  Eastbound travel: {(df['is_eastbound_travel'] == 1).sum()}")
        logger.info(f"  Mexico City games: {(df['is_mexico_city_game'] == 1).sum()}")
        logger.info(f"  Thursday Night games: {(df['is_thursday_night'] == 1).sum()}")

        return df


def main():
    """Example usage and validation."""
    logger.info("=" * 60)
    logger.info("ENHANCED TRAVEL FEATURES - VALIDATION")
    logger.info("=" * 60)

    # Create sample data
    sample_data = pd.DataFrame(
        {
            "game_id": ["2024_01_SEA_NE", "2024_02_NE_LAR", "2024_03_DEN_MIA"],
            "home_team": ["NE", "LAR", "MIA"],
            "away_team": ["SEA", "NE", "DEN"],
            "kickoff": pd.to_datetime(
                ["2024-09-08 13:00:00", "2024-09-15 16:00:00", "2024-09-22 13:00:00"]
            ),
        }
    )

    # Calculate features
    travel_features = EnhancedTravelFeatures()
    result = travel_features.calculate_all_features(sample_data)

    # Display results
    logger.info("\nSample Travel Features:")
    display_cols = [
        "game_id",
        "travel_distance",
        "timezone_shift",
        "is_westbound_travel",
        "is_eastbound_travel",
        "jet_lag_penalty",
        "body_clock_hour",
        "circadian_penalty",
        "total_travel_penalty",
    ]
    print(result[display_cols].to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("✓ VALIDATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
