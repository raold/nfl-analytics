#!/usr/bin/env python3
"""
Test script for Enhanced Ensemble v5.1 integration in FastAPI.

Tests:
1. API health check
2. Player prediction (PHI QB Week 11)
3. Game prediction (WAS @ PHI Week 11)
"""

import sys

import requests

API_BASE = "http://localhost:8000"

def test_health():
    """Test API health endpoint."""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)

    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"   Models Loaded: {data['models_loaded']}")
            print(f"   Cache Available: {data['cache_available']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f}s")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_player_prediction():
    """Test player prediction endpoint."""
    print("\n" + "="*70)
    print("TEST 2: Player Prediction (PHI QB Week 11 2024)")
    print("="*70)

    payload = {
        "player_id": "00-0036389",
        "stat_type": "passing_yards",
        "week": 11,
        "season": 2024,
        "confidence_level": 0.90
    }

    try:
        response = requests.post(
            f"{API_BASE}/predict",
            json=payload,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("\n✅ PREDICTION SUCCESS")
            print(f"   Player: {data['player_id']}")
            print(f"   Prediction: {data['prediction']:.1f} ± {data['uncertainty']:.1f} yards")
            print(f"   90% CI: [{data['confidence_interval']['lower']:.1f}, {data['confidence_interval']['upper']:.1f}]")
            print("   Actual: 221 yards")
            print(f"   Processing Time: {data['processing_ms']:.1f}ms")
            print(f"   Cache Hit: {data['cache_hit']}")
            print(f"   Ensemble Version: {data['model_versions']['ensemble']}")
            print("\n   Weights:")
            for model, weight in data['ensemble_weights'].items():
                print(f"      {model}: {weight:.1%}")

            # Check if actual value falls in CI
            actual = 221
            ci_lower = data['confidence_interval']['lower']
            ci_upper = data['confidence_interval']['upper']

            if ci_lower <= actual <= ci_upper:
                print(f"\n   ✅ Actual value ({actual} yards) within 90% CI")
            else:
                print(f"\n   ⚠️  Actual value ({actual} yards) OUTSIDE 90% CI")

            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def test_game_prediction():
    """Test game prediction endpoint."""
    print("\n" + "="*70)
    print("TEST 3: Game Prediction (WAS @ PHI Week 11 2024)")
    print("="*70)

    payload = {
        "game_id": "2024_11_WAS_PHI",
        "season": 2024,
        "week": 11
    }

    try:
        response = requests.post(
            f"{API_BASE}/predict/game",
            json=payload,
            timeout=60
        )
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("\n✅ GAME PREDICTION SUCCESS")
            print(f"   Game: {data['away_team']} @ {data['home_team']}")
            print(f"   Season {data['season']}, Week {data['week']}")

            print(f"\n   Away ({data['away_team']}):")
            if data['away_prediction']['predicted_passing_yards']:
                print(f"      QB: {data['away_prediction']['qb_id']}")
                print(f"      Prediction: {data['away_prediction']['predicted_passing_yards']:.1f} ± {data['away_prediction']['predicted_passing_yards_std']:.1f} yards")
                print("      Actual: 191 yards")
            else:
                print("      No prediction available")

            print(f"\n   Home ({data['home_team']}):")
            if data['home_prediction']['predicted_passing_yards']:
                print(f"      QB: {data['home_prediction']['qb_id']}")
                print(f"      Prediction: {data['home_prediction']['predicted_passing_yards']:.1f} ± {data['home_prediction']['predicted_passing_yards_std']:.1f} yards")
                print("      Actual: 221 yards")
            else:
                print("      No prediction available")

            if data['total_predicted_yards']:
                print(f"\n   Total Predicted: {data['total_predicted_yards']:.1f} yards")
                print("   Total Actual: 412 yards")
                print(f"   Error: {abs(data['total_predicted_yards'] - 412):.1f} yards")

            print(f"\n   Processing Time: {data['processing_ms']:.1f}ms")

            return True
        else:
            print(f"❌ Game prediction failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Game prediction error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Enhanced Ensemble v5.1 FastAPI Integration Test")
    print("="*70)

    # Check if API is running
    try:
        requests.get(f"{API_BASE}/", timeout=2)
    except Exception as e:
        print(f"\n❌ API not running at {API_BASE}")
        print(f"   Error: {e}")
        print("\n   Start the API with:")
        print("   uvicorn py.api.main:app --reload --port 8000")
        sys.exit(1)

    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Player Prediction", test_player_prediction()))
    results.append(("Game Prediction", test_game_prediction()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
