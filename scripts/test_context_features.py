#!/usr/bin/env python3
"""
Test Meta-Learning Context Feature Extraction
==============================================

Validate that all 15 context features extract correctly.

Expected Output:
- 15 features extracted for a known player-game
- Feature values in expected ranges
- Default features when data missing

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.features.meta_learning_context import MetaLearningContextEngine


def test_context_extraction():
    """
    Test context feature extraction for Patrick Mahomes Week 10 2024.
    """
    print("=" * 80)
    print("TEST: Meta-Learning Context Feature Extraction")
    print("=" * 80)

    # Initialize engine
    engine = MetaLearningContextEngine()

    # Test case: Patrick Mahomes, Week 10, 2024
    player_id = "00-0033873"  # Mahomes
    season = 2024
    week = 10

    print("\nExtracting features for:")
    print(f"  Player: {player_id} (Patrick Mahomes)")
    print(f"  Season: {season}")
    print(f"  Week: {week}")

    # Extract features
    features = engine.extract_context_features(
        player_id=player_id,
        season=season,
        week=week,
    )

    print(f"\n✅ Extracted {len(features.columns)} features:")
    print("\nQB Experience Features (5):")
    print(f"  qb_exp_rookie:          {features['qb_exp_rookie'].iloc[0]:.1f}")
    print(f"  qb_exp_developing:      {features['qb_exp_developing'].iloc[0]:.1f}")
    print(f"  qb_exp_veteran:         {features['qb_exp_veteran'].iloc[0]:.1f}")
    print(f"  qb_exp_elite_veteran:   {features['qb_exp_elite_veteran'].iloc[0]:.1f}")
    print(f"  log_games_experience:   {features['log_games_experience'].iloc[0]:.2f}")

    print("\nRoster Stability Features (1):")
    print(f"  roster_stability_l3:    {features['roster_stability_l3'].iloc[0]:.2%}")

    print("\nMarket Dispersion Features (2):")
    print(f"  market_dispersion:      {features['market_dispersion'].iloc[0]:.2f}")
    print(f"  high_dispersion_flag:   {features['high_dispersion_flag'].iloc[0]:.0f}")

    print("\nGame Context Features (7):")
    print(f"  is_home:                {features['is_home'].iloc[0]:.0f}")
    print(f"  is_dome_game:           {features['is_dome_game'].iloc[0]:.0f}")
    print(f"  is_division_game:       {features['is_division_game'].iloc[0]:.0f}")
    print(f"  temp_numeric:           {features['temp_numeric'].iloc[0]:.0f}°F")
    print(f"  wind_numeric:           {features['wind_numeric'].iloc[0]:.0f} mph")
    print(f"  is_cold_game:           {features['is_cold_game'].iloc[0]:.0f}")
    print(f"  is_windy_game:          {features['is_windy_game'].iloc[0]:.0f}")

    # Validate feature ranges
    print("\n" + "=" * 80)
    print("Feature Range Validation:")
    print("=" * 80)

    validations = [
        ("qb_exp_rookie", 0, 1),
        ("qb_exp_developing", 0, 1),
        ("qb_exp_veteran", 0, 1),
        ("qb_exp_elite_veteran", 0, 1),
        ("log_games_experience", 0, 6),
        ("roster_stability_l3", 0, 1),
        ("market_dispersion", 0, 5),
        ("high_dispersion_flag", 0, 1),
        ("is_home", 0, 1),
        ("is_dome_game", 0, 1),
        ("is_division_game", 0, 1),
        ("temp_numeric", 20, 110),
        ("wind_numeric", 0, 50),
        ("is_cold_game", 0, 1),
        ("is_windy_game", 0, 1),
    ]

    all_valid = True

    for feature, min_val, max_val in validations:
        val = features[feature].iloc[0]
        valid = min_val <= val <= max_val
        status = "✅" if valid else "❌"

        print(f"{status} {feature:25s}: {val:8.2f}  (expected [{min_val}, {max_val}])")

        if not valid:
            all_valid = False

    if all_valid:
        print("\n✅ All feature ranges valid!")
    else:
        print("\n❌ Some features out of range")

    print("=" * 80)


if __name__ == "__main__":
    test_context_extraction()
