#!/usr/bin/env python3
"""Quick API test for ensemble v5.1 integration"""

import json

import requests

API_BASE = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


def test_player_prediction():
    """Test player prediction"""
    print("=" * 60)
    print("TEST 2: Player Prediction (Jalen Hurts, 2024 Week 11)")
    print("=" * 60)

    payload = {
        "player_id": "00-0036389",
        "stat_type": "passing_yards",
        "week": 11,
        "season": 2024,
        "confidence_level": 0.90,
    }

    try:
        response = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))

            # Check if it's real ensemble prediction (not mocks)
            if "ensemble_mean" in result:
                print(
                    f"\n✅ Prediction: {result['ensemble_mean']:.1f} ± {result.get('ensemble_std', 0):.1f} yards"
                )
                print(
                    f"✅ 90% CI: [{result.get('ci_lower', 0):.1f}, {result.get('ci_upper', 0):.1f}]"
                )

                if "components" in result:
                    print("\n✅ Model Components (v5.1 integrated):")
                    for model, value in result["components"].items():
                        print(f"   {model}: {value:.1f} yards")
            else:
                print("\n⚠️  No ensemble_mean in response - may be using old format")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()


def test_game_prediction():
    """Test game prediction"""
    print("=" * 60)
    print("TEST 3: Game Prediction (2024 Week 12, BAL @ LAC)")
    print("=" * 60)

    payload = {"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12}

    try:
        response = requests.post(f"{API_BASE}/predict/game", json=payload, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            print("\n✅ Game prediction successful!")

            if "home_prediction" in result and "away_prediction" in result:
                home = result["home_prediction"]
                away = result["away_prediction"]
                print(
                    f"\nHome Team ({home.get('team')}): {home.get('predicted_passing_yards', 0):.1f} ± {home.get('predicted_passing_yards_std', 0):.1f} yards"
                )
                print(
                    f"Away Team ({away.get('team')}): {away.get('predicted_passing_yards', 0):.1f} ± {away.get('predicted_passing_yards_std', 0):.1f} yards"
                )
                print(f"Total Passing: {result.get('total_predicted_yards', 0):.1f} yards")
        elif response.status_code == 404:
            print("Game not found. Try different game_id.")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()


if __name__ == "__main__":
    test_health()
    test_player_prediction()
    test_game_prediction()

    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
