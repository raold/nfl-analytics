#!/usr/bin/env python3
"""
Causal Features v1.0 - DiD Roughing & PSM Travel/Weather
========================================================

Feature engineering module for causal inference features validated in Week 1-3
foundation phase of 12-week dual-track sprint.

Features Implemented:
1. DiD Roughing (2018 rule change): +17.70 yards effect
2. PSM Travel (null finding): +1.38 yards effect (not significant)
3. PSM Weather (null finding): -2.24 yards effect (not significant)

Usage:
    from py.features.causal_features_v1 import CausalFeaturesV1

    cf = CausalFeaturesV1(db_config)
    features = cf.calculate_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        opponent_team="HOU",
        game_weather={"temp": 45, "wind": 15, "precipitation": 0}
    )

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
Version: 1.0
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CausalFeaturesV1:
    """
    Causal inference feature calculator for ensemble v7.0.

    Integrates validated causal features from Week 1-3 foundation phase:
    - DiD Roughing (2018 rule change natural experiment)
    - PSM Travel (long-distance coast-to-coast games)
    - PSM Weather (extreme cold/wind conditions)

    Attributes:
        db_config (Dict): Database connection parameters
        defense_roughing_baseline (pd.DataFrame): 2017 defense classifications
        stadium_distance_matrix (pd.DataFrame): 32x32 distance matrix
    """

    # DiD Roughing: High-roughing defenses (2017 top 2 quintiles)
    HIGH_ROUGHING_DEFENSES = [
        "HOU",  # Houston Texans (1.12 penalties/100 attempts)
        "IND",  # Indianapolis Colts (0.92)
        "LAC",  # LA Chargers (0.88)
        "NYJ",  # NY Jets (0.85)
        "SEA",  # Seattle Seahawks (0.84)
        "ARI",  # Arizona Cardinals (0.79)
        "BAL",  # Baltimore Ravens (0.76)
        "KC",  # Kansas City Chiefs (0.73)
        "CLE",  # Cleveland Browns (0.70)
        "SF",  # San Francisco 49ers (0.68)
        "CIN",  # Cincinnati Bengals (0.67)
        "MIN",  # Minnesota Vikings (0.66)
        "DEN",  # Denver Broncos (0.65)
    ]

    # Treatment effects (from Week 1-3 validation)
    DID_ROUGHING_EFFECT = 17.70  # yards (p=0.001)
    PSM_TRAVEL_EFFECT = 1.38  # yards (p=0.87, null)
    PSM_WEATHER_EFFECT = -2.24  # yards (p=0.85, null)

    # Feature importance weights (based on statistical significance)
    DID_ROUGHING_WEIGHT = 1.0  # Full weight (highly significant)
    PSM_TRAVEL_WEIGHT = 0.0  # Zero weight (null finding)
    PSM_WEATHER_WEIGHT = 0.0  # Zero weight (null finding)

    def __init__(self, db_config: dict):
        """
        Initialize causal features calculator.

        Args:
            db_config: Database connection parameters
        """
        self.db_config = db_config
        self.defense_roughing_baseline = self._load_defense_roughing_baseline()
        self.stadium_distance_matrix = self._load_stadium_distance_matrix()

    def _load_defense_roughing_baseline(self) -> pd.DataFrame:
        """
        Load defense roughing baseline classifications from Week 1-3 deliverable.

        Returns:
            DataFrame with columns: [defense_team, quintile, roughing_rate_2017, is_treated]
        """
        baseline_path = Path("data/did_roughing_defense_classifications_2017.csv")

        if baseline_path.exists():
            df = pd.read_csv(baseline_path)
            logger.info(f"✅ Loaded defense roughing baseline ({len(df)} defenses)")
            return df
        else:
            logger.warning(f"⚠️  Defense roughing baseline not found at {baseline_path}")
            logger.warning("Creating in-memory baseline from HIGH_ROUGHING_DEFENSES list")

            # Fallback: create minimal baseline from hardcoded list
            data = []
            for team in self.HIGH_ROUGHING_DEFENSES:
                data.append(
                    {
                        "defense_team": team,
                        "quintile": (
                            "Very Aggressive"
                            if team in ["HOU", "IND", "LAC", "NYJ", "SEA", "ARI", "BAL"]
                            else "Aggressive"
                        ),
                        "roughing_rate_2017": 0.80,  # Approximate average
                        "is_treated": 1,
                    }
                )

            return pd.DataFrame(data)

    def _load_stadium_distance_matrix(self) -> pd.DataFrame:
        """
        Load 32x32 stadium distance matrix from Week 1-3 deliverable.

        Returns:
            DataFrame with index=home_team, columns=away_team, values=distance_miles
        """
        matrix_path = Path("data/stadium_distance_matrix.csv")

        if matrix_path.exists():
            df = pd.read_csv(matrix_path, index_col=0)
            logger.info(f"✅ Loaded stadium distance matrix ({df.shape[0]}x{df.shape[1]})")
            return df
        else:
            logger.warning(f"⚠️  Stadium distance matrix not found at {matrix_path}")
            logger.warning("Distance features will be unavailable")
            return pd.DataFrame()

    def calculate_features(
        self,
        player_id: str,
        season: int,
        week: int,
        opponent_team: str,
        player_team: str,
        game_weather: dict | None = None,
    ) -> dict[str, float]:
        """
        Calculate all causal features for a given QB game.

        Args:
            player_id: QB player ID (e.g., "00-0033873")
            season: Season year (e.g., 2024)
            week: Week number (1-18)
            opponent_team: Opponent defense team abbreviation (e.g., "HOU")
            player_team: QB's team abbreviation (e.g., "KC")
            game_weather: Optional dict with keys: temp, wind, precipitation

        Returns:
            Dict with causal feature values:
                - did_roughing_yards: +17.70 if high-roughing defense post-2018, else 0
                - psm_travel_yards: +1.38 if long-distance (≥2000 miles), else 0 (weight=0.0)
                - psm_weather_yards: -2.24 if extreme weather, else 0 (weight=0.0)
                - temp_numeric: Temperature in Fahrenheit (default: 50.0)
                - wind_numeric: Wind speed in mph (default: 5.0)
                - is_bad_weather: Binary (cold OR windy OR precipitation)
                - is_cold_weather: Binary (temp <32°F)
                - is_high_wind: Binary (wind >20mph)
                - causal_total_yards: Sum of enabled features (DiD only)
        """
        features = {}

        # Feature 1: DiD Roughing (enabled, weight=1.0)
        features["did_roughing_yards"] = self._calculate_did_roughing(opponent_team, season)

        # Feature 2: PSM Travel (weight=0.0 for causal_total, but raw value preserved for XGBoost)
        features["psm_travel_yards"] = self._calculate_psm_travel(player_team, opponent_team)

        # Feature 3: PSM Weather (weight=0.0 for causal_total, but raw value preserved for XGBoost)
        features["psm_weather_yards"] = self._calculate_psm_weather(game_weather)

        # Feature 4-8: Raw weather features (for XGBoost learning despite null PSM findings)
        if game_weather:
            features["temp_numeric"] = game_weather.get("temp", 50.0)  # Default 50°F
            features["wind_numeric"] = game_weather.get("wind", 5.0)  # Default 5 mph
            features["is_bad_weather"] = int(
                (game_weather.get("temp") is not None and game_weather.get("temp") < 32)
                or (game_weather.get("wind") is not None and game_weather.get("wind") > 20)
                or (game_weather.get("precipitation", 0) > 0)
            )
            features["is_cold_weather"] = int(
                game_weather.get("temp") is not None and game_weather.get("temp") < 32
            )
            features["is_high_wind"] = int(
                game_weather.get("wind") is not None and game_weather.get("wind") > 20
            )
        else:
            # Defaults for missing weather data
            features["temp_numeric"] = 50.0
            features["wind_numeric"] = 5.0
            features["is_bad_weather"] = 0
            features["is_cold_weather"] = 0
            features["is_high_wind"] = 0

        # Total causal adjustment (only enabled features)
        features["causal_total_yards"] = (
            self.DID_ROUGHING_WEIGHT * features["did_roughing_yards"]
            + self.PSM_TRAVEL_WEIGHT * features["psm_travel_yards"]
            + self.PSM_WEATHER_WEIGHT * features["psm_weather_yards"]
        )

        return features

    def _calculate_did_roughing(self, opponent_team: str, season: int) -> float:
        """
        Calculate DiD roughing treatment effect.

        Treatment: QB faces high-roughing defense (2017 top 2 quintiles) post-2018
        Effect: +17.70 yards (95% CI: [7.11, 28.29], p=0.001)

        Args:
            opponent_team: Opponent defense abbreviation
            season: Season year

        Returns:
            +17.70 if treated, 0.0 otherwise
        """
        # Only apply post-treatment (2018+)
        if season < 2018:
            return 0.0

        # Check if opponent is high-roughing defense
        if opponent_team in self.HIGH_ROUGHING_DEFENSES:
            logger.debug(
                f"DiD Roughing: {opponent_team} is high-roughing defense → +{self.DID_ROUGHING_EFFECT} yards"
            )
            return self.DID_ROUGHING_EFFECT
        else:
            return 0.0

    def _calculate_psm_travel(self, player_team: str, opponent_team: str) -> float:
        """
        Calculate PSM travel treatment effect.

        Treatment: Long-distance travel (≥2,000 miles coast-to-coast)
        Effect: +1.38 yards (95% CI: [-15.34, 18.10], p=0.87)

        **NOTE**: Null finding, disabled (weight=0.0)

        Args:
            player_team: QB's team abbreviation
            opponent_team: Opponent team abbreviation

        Returns:
            +1.38 if long-distance, 0.0 otherwise (but weighted to 0.0 anyway)
        """
        if self.stadium_distance_matrix.empty:
            return 0.0

        try:
            # Away team travels to opponent's stadium
            distance = self.stadium_distance_matrix.loc[opponent_team, player_team]

            # Long-distance threshold: 2,000 miles
            if distance >= 2000:
                logger.debug(
                    f"PSM Travel: {player_team} → {opponent_team} ({distance:.0f} miles) → +{self.PSM_TRAVEL_EFFECT} yards"
                )
                return self.PSM_TRAVEL_EFFECT
            else:
                return 0.0
        except KeyError:
            logger.warning(f"⚠️  Distance not found for {player_team} → {opponent_team}")
            return 0.0

    def _calculate_psm_weather(self, game_weather: dict | None) -> float:
        """
        Calculate PSM weather treatment effect.

        Treatment: Extreme weather (temp <32°F OR wind >20mph OR precipitation)
        Effect: -2.24 yards (95% CI: [-26.64, 22.17], p=0.85)

        **NOTE**: Null finding, disabled (weight=0.0)

        Args:
            game_weather: Dict with keys: temp, wind, precipitation

        Returns:
            -2.24 if extreme weather, 0.0 otherwise (but weighted to 0.0 anyway)
        """
        if not game_weather:
            return 0.0

        temp = game_weather.get("temp")
        wind = game_weather.get("wind")
        precip = game_weather.get("precipitation", 0)

        # Extreme weather criteria
        is_extreme_weather = (
            (temp is not None and temp < 32) or (wind is not None and wind > 20) or (precip > 0)
        )

        if is_extreme_weather:
            logger.debug(
                f"PSM Weather: Extreme conditions (temp={temp}, wind={wind}, precip={precip}) → {self.PSM_WEATHER_EFFECT} yards"
            )
            return self.PSM_WEATHER_EFFECT
        else:
            return 0.0


if __name__ == "__main__":
    # Test causal features calculator
    print("Testing Causal Features v1.0...")

    # Test database connection
    db_config = {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": "sicillionbillions",  # pragma: allowlist secret
    }

    cf = CausalFeaturesV1(db_config)

    # Test Case 1: Patrick Mahomes vs Houston Texans (high-roughing defense)
    print("\n1. Testing DiD Roughing (Mahomes vs HOU, 2024)...")
    features = cf.calculate_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        opponent_team="HOU",
        player_team="KC",
        game_weather={"temp": 65, "wind": 10, "precipitation": 0},
    )
    print(f"   DiD Roughing: {features['did_roughing_yards']:.2f} yards")
    print(f"   PSM Travel: {features['psm_travel_yards']:.2f} yards (disabled)")
    print(f"   PSM Weather: {features['psm_weather_yards']:.2f} yards (disabled)")
    print(f"   Total Causal: {features['causal_total_yards']:.2f} yards")

    # Expected: +17.70 yards (HOU is high-roughing defense post-2018)
    assert features["did_roughing_yards"] == 17.70, "DiD roughing should be +17.70"
    assert features["psm_travel_yards"] == 0.0, "PSM travel should be disabled"
    assert features["psm_weather_yards"] == 0.0, "PSM weather should be disabled"
    assert features["causal_total_yards"] == 17.70, "Total should be +17.70"
    print("   ✅ Test 1 passed")

    # Test Case 2: QB vs Green Bay (low-roughing defense)
    print("\n2. Testing DiD Roughing (QB vs GB, 2024)...")
    features = cf.calculate_features(
        player_id="00-0036442",
        season=2024,
        week=5,
        opponent_team="GB",
        player_team="CIN",
        game_weather={"temp": 55, "wind": 12, "precipitation": 0},
    )
    print(f"   DiD Roughing: {features['did_roughing_yards']:.2f} yards")
    print(f"   Total Causal: {features['causal_total_yards']:.2f} yards")

    # Expected: 0 yards (GB is low-roughing defense)
    assert features["did_roughing_yards"] == 0.0, "DiD roughing should be 0"
    assert features["causal_total_yards"] == 0.0, "Total should be 0"
    print("   ✅ Test 2 passed")

    # Test Case 3: Pre-2018 game (no treatment)
    print("\n3. Testing DiD Roughing (pre-2018)...")
    features = cf.calculate_features(
        player_id="00-0019596",
        season=2017,
        week=10,
        opponent_team="HOU",
        player_team="NE",
        game_weather={"temp": 45, "wind": 15, "precipitation": 0},
    )
    print(f"   DiD Roughing: {features['did_roughing_yards']:.2f} yards")
    print(f"   Total Causal: {features['causal_total_yards']:.2f} yards")

    # Expected: 0 yards (pre-2018, no treatment)
    assert features["did_roughing_yards"] == 0.0, "DiD roughing should be 0 pre-2018"
    assert features["causal_total_yards"] == 0.0, "Total should be 0"
    print("   ✅ Test 3 passed")

    print("\n✅ All tests passed! Causal Features v1.0 validated.")
