#!/usr/bin/env python3
"""
Test script for NFL Analytics Prediction API

Tests all endpoints and validates responses.

Usage:
    python api/test_api.py
    python api/test_api.py --base-url http://localhost:8000

Author: Claude Code
Date: 2025-11-08
"""

import argparse
import json
import sys
import time

import requests


class APITester:
    """Test harness for Prediction API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.tests_passed = 0
        self.tests_failed = 0

    def print_header(self, text: str):
        """Print section header"""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)

    def print_success(self, text: str):
        """Print success message"""
        print(f"✓ {text}")
        self.tests_passed += 1

    def print_failure(self, text: str):
        """Print failure message"""
        print(f"✗ {text}")
        self.tests_failed += 1

    def test_health(self) -> bool:
        """Test GET /health endpoint"""
        self.print_header("Test 1: Health Check")

        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()

            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Database Connected: {data['database_connected']}")
            print(f"Redis Connected: {data['redis_connected']}")
            print(f"Models Loaded: {data['models_loaded']}")
            print(f"Uptime: {data['uptime_seconds']:.2f}s")

            # Validate response
            assert data["status"] in ["healthy", "degraded", "unhealthy"], "Invalid status"
            assert isinstance(data["models_loaded"], int), "models_loaded must be int"
            assert data["models_loaded"] >= 0, "models_loaded must be >= 0"

            self.print_success("Health endpoint working")
            return True

        except Exception as e:
            self.print_failure(f"Health check failed: {e}")
            return False

    def test_root(self) -> bool:
        """Test GET / endpoint"""
        self.print_header("Test 2: Root Endpoint")

        try:
            response = self.session.get(f"{self.base_url}/", timeout=10)
            response.raise_for_status()

            data = response.json()
            print(f"Name: {data['name']}")
            print(f"Version: {data['version']}")
            print(f"Status: {data['status']}")

            assert "NFL Analytics" in data["name"], "Invalid API name"
            self.print_success("Root endpoint working")
            return True

        except Exception as e:
            self.print_failure(f"Root endpoint failed: {e}")
            return False

    def test_models_list(self) -> bool:
        """Test GET /models endpoint"""
        self.print_header("Test 3: List Models")

        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()

            data = response.json()
            print(f"Total Models: {data['total_models']}")
            print(f"Loaded Models: {data['loaded_models']}")
            print(f"Ensemble Ready: {data['ensemble_ready']}")

            # Print model details
            print("\nModels:")
            for model in data["models"]:
                print(f"  - {model['model_name']} ({model['model_type']})")
                print(f"    Version: {model['version']}")
                print(f"    Status: {model['status']}")
                print(f"    Size: {model['file_size_mb']:.2f} MB")

            # Validate response
            assert data["total_models"] >= 0, "total_models must be >= 0"
            assert data["loaded_models"] <= data["total_models"], "Invalid loaded count"

            self.print_success("Models endpoint working")
            return True

        except Exception as e:
            self.print_failure(f"Models list failed: {e}")
            return False

    def test_prediction(self) -> bool:
        """Test POST /predict endpoint"""
        self.print_header("Test 4: Generate Prediction")

        # Test payload
        payload = {
            "game_id": "2024_10_KC_LV",
            "player_id": "00-0033873",  # Patrick Mahomes
            "stat_type": "passing_yards"
        }

        try:
            print("\nRequest Payload:")
            print(json.dumps(payload, indent=2))

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            latency = time.time() - start_time

            response.raise_for_status()

            data = response.json()
            print("\nResponse:")
            print(f"Prediction: {data['prediction']:.1f}")
            print(f"Confidence: {data['confidence']:.2f}")
            print(f"90% CI: [{data['lower_bound']:.1f}, {data['upper_bound']:.1f}]")
            print(f"Ensemble Method: {data['ensemble_method']}")
            print(f"Latency: {latency:.3f}s")

            print("\nModel Contributions:")
            for model in data["models_used"]:
                print(f"  - {model['model_name']}: {model['prediction']:.1f} "
                      f"± {model['uncertainty']:.1f} (weight: {model['weight']:.2f})")

            # Validate response
            assert isinstance(data["prediction"], (int, float)), "prediction must be numeric"
            assert 0 <= data["confidence"] <= 1, "confidence must be in [0, 1]"
            assert data["lower_bound"] < data["upper_bound"], "Invalid credible interval"
            assert len(data["models_used"]) > 0, "No models used"

            # Validate weights sum to 1
            total_weight = sum(m["weight"] for m in data["models_used"])
            assert abs(total_weight - 1.0) < 0.01, f"Weights don't sum to 1 ({total_weight:.3f})"

            self.print_success(f"Prediction endpoint working (latency: {latency:.3f}s)")
            return True

        except Exception as e:
            self.print_failure(f"Prediction failed: {e}")
            return False

    def test_metrics(self) -> bool:
        """Test GET /metrics endpoint"""
        self.print_header("Test 5: Prometheus Metrics")

        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=10)
            response.raise_for_status()

            metrics_text = response.text

            # Check for key metrics
            required_metrics = [
                "predictions_total",
                "prediction_latency_seconds",
                "models_loaded_total",
                "api_requests_total"
            ]

            print("\nChecking for required metrics:")
            all_present = True
            for metric in required_metrics:
                if metric in metrics_text:
                    print(f"  ✓ {metric}")
                else:
                    print(f"  ✗ {metric} (missing)")
                    all_present = False

            # Print sample metrics
            print("\nSample Metrics:")
            for line in metrics_text.split("\n")[:20]:
                if line and not line.startswith("#"):
                    print(f"  {line}")

            assert all_present, "Some required metrics are missing"
            self.print_success("Metrics endpoint working")
            return True

        except Exception as e:
            self.print_failure(f"Metrics check failed: {e}")
            return False

    def test_invalid_request(self) -> bool:
        """Test error handling with invalid request"""
        self.print_header("Test 6: Error Handling")

        # Invalid payload (missing required field)
        invalid_payload = {
            "game_id": "2024_10_KC_LV",
            # Missing player_id and stat_type
        }

        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_payload,
                timeout=10
            )

            print(f"Status Code: {response.status_code}")

            # Should return 422 Unprocessable Entity
            assert response.status_code == 422, f"Expected 422, got {response.status_code}"

            error_data = response.json()
            print("Error Response:")
            print(json.dumps(error_data, indent=2))

            self.print_success("Error handling working correctly")
            return True

        except Exception as e:
            self.print_failure(f"Error handling test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and return success status"""
        print("\n" + "=" * 70)
        print("  NFL ANALYTICS PREDICTION API - TEST SUITE")
        print("=" * 70)
        print(f"\nBase URL: {self.base_url}")

        # Run tests
        tests = [
            self.test_health,
            self.test_root,
            self.test_models_list,
            self.test_prediction,
            self.test_metrics,
            self.test_invalid_request
        ]

        for test in tests:
            try:
                test()
            except KeyboardInterrupt:
                print("\n\nTests interrupted by user")
                sys.exit(1)
            except Exception as e:
                self.print_failure(f"Unexpected error in {test.__name__}: {e}")

        # Print summary
        self.print_header("TEST SUMMARY")
        total_tests = self.tests_passed + self.tests_failed
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {self.tests_passed} ✓")
        print(f"Failed: {self.tests_failed} ✗")
        print(f"Success Rate: {self.tests_passed / total_tests * 100:.1f}%")

        all_passed = self.tests_failed == 0
        if all_passed:
            print("\n✓ All tests passed!")
        else:
            print(f"\n✗ {self.tests_failed} test(s) failed")

        return all_passed


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test NFL Analytics Prediction API"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    tester = APITester(base_url=args.base_url)

    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Cannot connect to API at {args.base_url}")
        print("  Make sure the API server is running:")
        print("    docker compose up -d api")
        print("  or")
        print("    cd api && uvicorn main:app --reload")
        sys.exit(1)


if __name__ == "__main__":
    main()
