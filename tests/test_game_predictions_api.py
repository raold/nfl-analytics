#!/usr/bin/env python3
"""
Test script for game prediction API endpoints.

Tests:
1. Database table creation
2. POST /api/v1/predict/game/full
3. POST /api/v1/predict/week
4. GET /api/v1/predictions/game/{game_id}

Author: Claude Code
Date: 2025-11-21
"""

import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import psycopg2
import requests

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",  # pragma: allowlist secret
}

API_BASE_URL = "http://localhost:8000/api/v1"


def test_database_table():
    """Test that ml.game_predictions table exists."""
    print("\n=== Test 1: Database Table ===")

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Check table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'ml'
            AND table_name = 'game_predictions'
        );
    """)
    exists = cursor.fetchone()[0]

    if not exists:
        print("❌ Table ml.game_predictions does not exist")
        return False

    # Check columns
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'ml'
        AND table_name = 'game_predictions'
        ORDER BY ordinal_position;
    """)
    columns = cursor.fetchall()

    print(f"✅ Table exists with {len(columns)} columns:")
    key_columns = [
        "game_id",
        "predicted_home_score",
        "predicted_away_score",
        "predicted_total",
        "predicted_spread",
        "predicted_home_win_prob",
        "spread_edge",
        "total_edge",
        "ml_edge",
    ]

    found_columns = [col[0] for col in columns]
    for col in key_columns:
        if col in found_columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} missing")

    cursor.close()
    conn.close()

    return True


def test_predict_game_full():
    """Test POST /api/v1/predict/game/full endpoint."""
    print("\n=== Test 2: POST /api/v1/predict/game/full ===")

    # Find a recent game
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT game_id, season, week, home_team, away_team
        FROM games
        WHERE season = 2024
        ORDER BY kickoff DESC
        LIMIT 1;
    """)
    game = cursor.fetchone()
    cursor.close()
    conn.close()

    if not game:
        print("❌ No games found in database")
        return False

    game_id, season, week, home_team, away_team = game
    print(f"Testing with game: {game_id} ({away_team}@{home_team})")

    # Make API request
    payload = {"game_id": game_id, "season": season, "week": week}

    try:
        response = requests.post(f"{API_BASE_URL}/predict/game/full", json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success (HTTP {response.status_code})")
            print(f"   Predicted Home Score: {data['predicted_home_score']}")
            print(f"   Predicted Away Score: {data['predicted_away_score']}")
            print(f"   Predicted Total: {data['predicted_total']}")
            print(f"   Predicted Spread: {data['predicted_spread']}")
            print(f"   Predicted Win Prob: {data['predicted_home_win_prob']:.2%}")
            print(f"   Processing Time: {data['processing_ms']:.1f}ms")

            if data.get("spread_edge") is not None:
                print(f"   Spread Edge: {data['spread_edge']:.2f}")
            if data.get("total_edge") is not None:
                print(f"   Total Edge: {data['total_edge']:.2f}")

            return True
        else:
            print(f"❌ Failed (HTTP {response.status_code})")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the API running? (uvicorn py.api.main:app --host 0.0.0.0 --port 8000)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict_week():
    """Test POST /api/v1/predict/week endpoint."""
    print("\n=== Test 3: POST /api/v1/predict/week ===")

    # Use recent week
    payload = {"season": 2024, "week": 1}

    try:
        response = requests.post(f"{API_BASE_URL}/predict/week", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success (HTTP {response.status_code})")
            print(f"   Total Games: {data['total_games']}")
            print(f"   Processing Time: {data['processing_ms']:.1f}ms")
            print(f"   Avg per Game: {data['processing_ms'] / data['total_games']:.1f}ms")

            # Show first 3 predictions
            print(f"\n   First 3 predictions:")
            for i, pred in enumerate(data["predictions"][:3]):
                print(f"      {i+1}. {pred['game_id']}: {pred['away_team']}@{pred['home_team']} "
                      f"({pred['predicted_total']:.1f} total, {pred['predicted_spread']:.1f} spread)")

            return True
        else:
            print(f"❌ Failed (HTTP {response.status_code})")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the API running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_get_game_prediction():
    """Test GET /api/v1/predictions/game/{game_id} endpoint."""
    print("\n=== Test 4: GET /api/v1/predictions/game/{game_id} ===")

    # Find a game with stored prediction
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT game_id
        FROM ml.game_predictions
        ORDER BY predicted_at DESC
        LIMIT 1;
    """)
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        print("⚠️  No stored predictions found - skipping test")
        return True

    game_id = result[0]
    print(f"Testing with game: {game_id}")

    try:
        response = requests.get(f"{API_BASE_URL}/predictions/game/{game_id}", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success (HTTP {response.status_code})")
            print(f"   Game: {data['away_team']}@{data['home_team']}")
            print(f"   Predicted Total: {data['predicted_total']}")
            print(f"   Predicted Spread: {data['predicted_spread']}")
            print(f"   Ensemble Version: {data['ensemble_version']}")
            print(f"   Predicted At: {data['predicted_at']}")
            return True
        else:
            print(f"❌ Failed (HTTP {response.status_code})")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the API running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Game Predictions API Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Database
    results.append(("Database Table", test_database_table()))

    # Test 2-4: API endpoints (skip if API not running)
    try:
        requests.get(f"{API_BASE_URL}/../health", timeout=2)
        api_running = True
    except:
        api_running = False
        print("\n⚠️  API not running - skipping endpoint tests")
        print("   Start API with: uvicorn py.api.main:app --host 0.0.0.0 --port 8000")

    if api_running:
        results.append(("POST /predict/game/full", test_predict_game_full()))
        results.append(("POST /predict/week", test_predict_week()))
        results.append(("GET /predictions/game/{game_id}", test_get_game_prediction()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
