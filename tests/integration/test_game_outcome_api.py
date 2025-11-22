#!/usr/bin/env python3
"""
Integration Test for Game Outcome API
======================================

Tests the full game prediction API endpoint with real Ensemble v6.1.

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import json
import sys
from pathlib import Path

import requests

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_game_prediction_api():
    """
    Test /api/v1/predict/game/full endpoint with 2025 Week 12 BUF @ HOU.

    Expected behavior:
    - Returns 200 OK
    - Contains predicted_home_score, predicted_away_score
    - Contains predicted_total, predicted_spread
    - Contains uncertainty estimates (spread_std, total_std)
    - ensemble_version = "v6.1"
    - Prediction stored in database
    """
    print("\n" + "="*70)
    print("INTEGRATION TEST: Game Outcome API (Ensemble v6.1)")
    print("="*70)

    # Test endpoint
    url = "http://localhost:8000/api/v1/predict/game/full"
    payload = {
        "game_id": "2025_12_BUF_HOU",
        "season": 2025,
        "week": 12
    }

    print(f"\n1. Sending POST request to: {url}")
    print(f"   Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"\n2. Response Status: {response.status_code}")

        if response.status_code != 200:
            print(f"   Error: {response.text}")
            return False

        # Parse response
        data = response.json()
        print("\n3. Response Data:")
        print(f"   Game ID: {data.get('game_id')}")
        print(f"   Teams: {data.get('away_team')} @ {data.get('home_team')}")
        print(f"   Home Score: {data.get('predicted_home_score')}")
        print(f"   Away Score: {data.get('predicted_away_score')}")
        print(f"   Total: {data.get('predicted_total')} ± {data.get('total_std')}")
        print(f"   Spread: {data.get('predicted_spread')} ± {data.get('spread_std')}")
        print(f"   Home Win Prob: {data.get('predicted_home_win_prob')}")
        print(f"   Ensemble Version: {data.get('ensemble_version')}")
        print(f"   Processing Time: {data.get('processing_ms'):.1f}ms")

        # Validation
        print("\n4. Validation:")
        checks = {
            "game_id matches": data.get('game_id') == payload['game_id'],
            "season matches": data.get('season') == payload['season'],
            "week matches": data.get('week') == payload['week'],
            "has predicted_home_score": 'predicted_home_score' in data,
            "has predicted_away_score": 'predicted_away_score' in data,
            "has predicted_total": 'predicted_total' in data,
            "has predicted_spread": 'predicted_spread' in data,
            "has predicted_home_win_prob": 'predicted_home_win_prob' in data,
            "has spread_std": 'spread_std' in data,
            "has total_std": 'total_std' in data,
            "ensemble_version is v6.1": data.get('ensemble_version') == 'v6.1',
            "home_score reasonable": 10 <= data.get('predicted_home_score', 0) <= 50,
            "away_score reasonable": 10 <= data.get('predicted_away_score', 0) <= 50,
            "total reasonable": 20 <= data.get('predicted_total', 0) <= 100,
            "spread reasonable": -30 <= data.get('predicted_spread', 0) <= 30,
            "win_prob valid": 0 <= data.get('predicted_home_win_prob', 0) <= 1,
            "spread_std > 0": data.get('spread_std', 0) > 0,
            "total_std > 0": data.get('total_std', 0) > 0,
        }

        all_passed = True
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status}: {check}")
            if not passed:
                all_passed = False

        print("\n" + "="*70)
        if all_passed:
            print("RESULT: ✅ ALL CHECKS PASSED")
            print("="*70)
            return True
        else:
            print("RESULT: ❌ SOME CHECKS FAILED")
            print("="*70)
            return False

    except requests.exceptions.Timeout:
        print("\n❌ ERROR: Request timed out (>60s)")
        return False
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API (is it running?)")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_game_prediction_api()
    sys.exit(0 if success else 1)
