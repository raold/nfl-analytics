#!/usr/bin/env python3
"""
Test Weather Features Integration
==================================

Quick test script to validate weather features are correctly extracted.

Usage:
    python scripts/test_weather_features.py

Author: Claude Code (Sonnet 4.5)
Date: November 27, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.features.weather_features_integration import compute_weather_features

# Database config
db_config = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def test_weather_features():
    """Test weather feature extraction on known games."""

    # Test cases from weather table sample
    test_games = [
        ("2019_18_TEN_NE", "Cold playoff game (6.1°C, no wind)"),
        ("2019_18_SEA_PHI", "Windy playoff game (10.6°C, 46.8 km/h)"),
        ("2019_19_HOU_KC", "Very cold playoff game (-7.2°C, 27.36 km/h)"),
        ("2019_19_SEA_GB", "Extremely windy cold game (-6.1°C, 106.2 km/h)"),
        ("2019_21_SF_KC", "Super Bowl (9.4°C, 53.28 km/h)"),
    ]

    print("\n" + "=" * 80)
    print("WEATHER FEATURES INTEGRATION TEST")
    print("=" * 80)
    print()

    for game_id, description in test_games:
        print(f"\nGame: {game_id}")
        print(f"Description: {description}")
        print("-" * 80)

        features = compute_weather_features(game_id, db_config)

        print(f"  high_wind:      {features['high_wind']} (1.0 = wind > 24 km/h)")
        print(f"  cold_weather:   {features['cold_weather']} (1.0 = temp < 0°C)")
        print(f"  dome_game:      {features['dome_game']} (1.0 = dome/closed roof)")
        print(f"  precipitation:  {features['precipitation']} (1.0 = precip > 0)")
        print(f"  temp_c:         {features['temp_c']}")
        print(f"  wind_kph:       {features['wind_kph']}")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    test_weather_features()
