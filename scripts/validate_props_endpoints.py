#!/usr/bin/env python3
"""
Validate Props API Endpoints

Tests all 3 new props endpoints with 2025 Week 12 data:
- /api/v1/predict/rushing_yards/{player_id}
- /api/v1/predict/receiving_yards/{player_id}
- /api/v1/predict/touchdown_prob/{player_id}

Validates:
1. Endpoint availability and response time (<500ms)
2. Response schema compliance
3. Prediction quality (reasonable ranges)
4. Error handling
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import logging
import time

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Test players for Week 12, 2025
TEST_PLAYERS = {
    "rushing_yards": [
        {"player_id": "00-0034844", "player_name": "Derrick Henry", "expected_range": (70, 120)},
        {
            "player_id": "00-0036945",
            "player_name": "Christian McCaffrey",
            "expected_range": (60, 100),
        },
        {"player_id": "00-0038120", "player_name": "Bijan Robinson", "expected_range": (65, 110)},
    ],
    "receiving_yards": [
        {"player_id": "00-0033077", "player_name": "Tyreek Hill", "expected_range": (70, 130)},
        {
            "player_id": "00-0037744",
            "player_name": "Justin Jefferson",
            "expected_range": (75, 125),
        },
        {
            "player_id": "00-0036389",
            "player_name": "CeeDee Lamb",
            "expected_range": (70, 120),
        },
    ],
    "touchdowns": [
        {"player_id": "00-0029701", "player_name": "Travis Kelce", "expected_range": (0.2, 0.6)},
        {"player_id": "00-0034844", "player_name": "Derrick Henry", "expected_range": (0.3, 0.7)},
        {"player_id": "00-0033077", "player_name": "Tyreek Hill", "expected_range": (0.25, 0.65)},
    ],
}


class PropsEndpointValidator:
    """Validates props API endpoints."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.results = {"passed": 0, "failed": 0, "warnings": 0}

    def check_health(self) -> bool:
        """Check API health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API health check passed")
                return True
            else:
                logger.error(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API health check error: {e}")
            return False

    def check_props_health(self) -> dict:
        """Check props-specific health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/props/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Props health check passed: {data['status']}")
                logger.info(f"   Models available: {data['models']}")
                return data
            else:
                logger.warning(f"⚠️ Props health check returned {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Props health endpoint not available: {e}")
            return None

    def validate_endpoint(
        self, endpoint: str, player_id: str, player_name: str, stat_type: str, expected_range: tuple
    ) -> bool:
        """
        Validate a single endpoint call.

        Args:
            endpoint: API endpoint path
            player_id: Player ID
            player_name: Player name (for logging)
            stat_type: Stat type (rushing_yards, receiving_yards, touchdowns)
            expected_range: (min, max) expected range for prediction

        Returns:
            True if validation passed, False otherwise
        """
        payload = {
            "player_id": player_id,
            "stat_type": stat_type,
            "week": 12,
            "season": 2025,
            "confidence_level": 0.90,
        }

        start_time = time.time()

        try:
            response = requests.post(endpoint, json=payload, timeout=5)
            elapsed_ms = (time.time() - start_time) * 1000

            # Check HTTP status
            if response.status_code != 200:
                logger.error(
                    f"❌ {player_name} ({stat_type}): HTTP {response.status_code} "
                    f"- {response.text[:100]}"
                )
                self.results["failed"] += 1
                return False

            # Parse response
            data = response.json()

            # Validate latency (<500ms target)
            if elapsed_ms > 500:
                logger.warning(
                    f"⚠️ {player_name} ({stat_type}): Latency {elapsed_ms:.0f}ms > 500ms target"
                )
                self.results["warnings"] += 1
            else:
                logger.info(f"✅ {player_name} ({stat_type}): Latency {elapsed_ms:.0f}ms OK")

            # Validate required fields
            required_fields = [
                "player_id",
                "stat_type",
                "week",
                "season",
                "prediction",
                "uncertainty",
                "confidence_level",
                "confidence_interval_lower",
                "confidence_interval_upper",
                "model_used",
                "processing_ms",
            ]

            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                logger.error(f"❌ {player_name} ({stat_type}): Missing fields: {missing_fields}")
                self.results["failed"] += 1
                return False

            # Validate prediction range
            prediction = data["prediction"]
            min_val, max_val = expected_range

            if not (min_val <= prediction <= max_val):
                logger.warning(
                    f"⚠️ {player_name} ({stat_type}): Prediction {prediction:.1f} "
                    f"outside expected range [{min_val}, {max_val}]"
                )
                self.results["warnings"] += 1

            # Validate confidence interval
            ci_lower = data["confidence_interval_lower"]
            ci_upper = data["confidence_interval_upper"]

            if not (ci_lower < prediction < ci_upper):
                logger.error(
                    f"❌ {player_name} ({stat_type}): Prediction {prediction:.1f} "
                    f"not within CI [{ci_lower:.1f}, {ci_upper:.1f}]"
                )
                self.results["failed"] += 1
                return False

            # Log success
            logger.info(
                f"✅ {player_name} ({stat_type}): "
                f"Pred={prediction:.1f}±{data['uncertainty']:.1f}, "
                f"CI=[{ci_lower:.1f}, {ci_upper:.1f}], "
                f"Model={data['model_used']}"
            )

            self.results["passed"] += 1
            return True

        except Exception as e:
            logger.error(f"❌ {player_name} ({stat_type}): Exception - {e}")
            self.results["failed"] += 1
            return False

    def run_validation(self) -> bool:
        """
        Run full validation suite across all endpoints.

        Returns:
            True if all validations passed, False if any failed
        """
        logger.info("=" * 70)
        logger.info("PROPS API ENDPOINT VALIDATION (Week 12, 2025)")
        logger.info("=" * 70)

        # Check API health
        if not self.check_health():
            logger.error("API not available. Start with: uvicorn py.api.main:app")
            return False

        # Check props health
        self.check_props_health()

        logger.info("\n" + "=" * 70)
        logger.info("TESTING RUSHING YARDS ENDPOINT")
        logger.info("=" * 70)

        for player in TEST_PLAYERS["rushing_yards"]:
            self.validate_endpoint(
                endpoint=f"{self.base_url}/api/v1/predict/rushing_yards/{player['player_id']}",
                player_id=player["player_id"],
                player_name=player["player_name"],
                stat_type="rushing_yards",
                expected_range=player["expected_range"],
            )

        logger.info("\n" + "=" * 70)
        logger.info("TESTING RECEIVING YARDS ENDPOINT")
        logger.info("=" * 70)

        for player in TEST_PLAYERS["receiving_yards"]:
            self.validate_endpoint(
                endpoint=f"{self.base_url}/api/v1/predict/receiving_yards/{player['player_id']}",
                player_id=player["player_id"],
                player_name=player["player_name"],
                stat_type="receiving_yards",
                expected_range=player["expected_range"],
            )

        logger.info("\n" + "=" * 70)
        logger.info("TESTING TOUCHDOWN PROBABILITY ENDPOINT")
        logger.info("=" * 70)

        for player in TEST_PLAYERS["touchdowns"]:
            self.validate_endpoint(
                endpoint=f"{self.base_url}/api/v1/predict/touchdown_prob/{player['player_id']}",
                player_id=player["player_id"],
                player_name=player["player_name"],
                stat_type="touchdowns",
                expected_range=player["expected_range"],
            )

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✅ Passed: {self.results['passed']}")
        logger.info(f"❌ Failed: {self.results['failed']}")
        logger.info(f"⚠️ Warnings: {self.results['warnings']}")

        all_passed = self.results["failed"] == 0

        if all_passed:
            logger.info("\n🎉 ALL VALIDATIONS PASSED!")
        else:
            logger.error(
                f"\n❌ VALIDATION FAILED: {self.results['failed']} failures, "
                f"{self.results['warnings']} warnings"
            )

        return all_passed


def main():
    """Run validation"""
    validator = PropsEndpointValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
