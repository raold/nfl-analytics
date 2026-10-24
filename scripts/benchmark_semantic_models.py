#!/usr/bin/env python3
"""
Benchmark script for Semantic Stack models.

Tests:
1. NER model accuracy on validation samples
2. MLX vs Ollama inference performance (latency, throughput)

Usage:
    python scripts/benchmark_semantic_models.py --test-ner
    python scripts/benchmark_semantic_models.py --compare-inference
    python scripts/benchmark_semantic_models.py --all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# NER test cases with expected entities
NER_TEST_CASES = [
    {
        "text": "Patrick Mahomes is questionable for Sunday with an ankle injury.",
        "expected": {"PLAYER": ["Patrick Mahomes"], "STATUS": ["questionable"], "INJURY": ["ankle"]},
    },
    {
        "text": "The Chiefs list Travis Kelce as doubtful with a knee issue.",
        "expected": {"TEAM": ["Chiefs"], "PLAYER": ["Travis Kelce"], "STATUS": ["doubtful"], "INJURY": ["knee"]},
    },
    {
        "text": "Ravens RB Derrick Henry practiced in full on Wednesday.",
        "expected": {"TEAM": ["Ravens"], "POSITION": ["RB"], "PLAYER": ["Derrick Henry"]},
    },
    {
        "text": "Josh Allen ruled out for Bills game against Dolphins.",
        "expected": {"PLAYER": ["Josh Allen"], "STATUS": ["out"], "TEAM": ["Bills", "Dolphins"]},
    },
    {
        "text": "49ers QB Brock Purdy probable with shoulder strain.",
        "expected": {"TEAM": ["49ers"], "POSITION": ["QB"], "PLAYER": ["Brock Purdy"], "STATUS": ["probable"], "INJURY": ["shoulder"]},
    },
    {
        "text": "Tyreek Hill limited in practice Thursday with hamstring tightness.",
        "expected": {"PLAYER": ["Tyreek Hill"], "INJURY": ["hamstring"]},
    },
    {
        "text": "Cowboys place Dak Prescott on IR with ankle fracture.",
        "expected": {"TEAM": ["Cowboys"], "PLAYER": ["Dak Prescott"], "STATUS": ["IR"], "INJURY": ["ankle", "fracture"]},
    },
    {
        "text": "Lamar Jackson full participant, expected to play Sunday.",
        "expected": {"PLAYER": ["Lamar Jackson"]},
    },
]

# Inference benchmark prompts
BENCHMARK_PROMPTS = [
    "Extract NFL entities from: Patrick Mahomes questionable with ankle injury.",
    "Analyze sentiment: The Chiefs are looking strong heading into playoffs.",
    "Summarize: The Ravens dominated the Bengals 35-10 with Lamar Jackson throwing 4 TDs.",
    "Extract entities: Travis Kelce listed as doubtful for Sunday's game against Raiders.",
    "Classify: Breaking news - Josh Allen ruled out for remainder of season.",
]


def test_ner_with_ollama(test_cases: list[dict], model: str = "qwen3:8b") -> dict[str, Any]:
    """Test NER accuracy using Ollama."""
    try:
        from nfl_analytics.semantic.models.ollama_client import OllamaClient
    except ImportError:
        logger.error("Could not import OllamaClient")
        return {"error": "Import failed"}

    client = OllamaClient(timeout=240.0, default_model=model)

    if not client.is_available():
        logger.error("Ollama is not available")
        return {"error": "Ollama not available"}

    results = {
        "model": model,
        "total_cases": len(test_cases),
        "passed": 0,
        "failed": 0,
        "partial": 0,
        "details": [],
        "latencies": [],
    }

    for i, case in enumerate(test_cases):
        text = case["text"]
        expected = case["expected"]

        prompt = f"""Extract NFL entities from the following text.
Identify: PLAYER (player names), TEAM (team names), INJURY (body parts/injuries),
STATUS (questionable, doubtful, out, IR, probable), POSITION (QB, RB, WR, etc.)

Text: {text}

Respond with a JSON object containing the entities found. Example:
{{"PLAYER": ["Name"], "TEAM": ["Team"], "INJURY": ["injury"], "STATUS": ["status"]}}
"""

        start = time.time()
        try:
            response = client.generate(prompt, temperature=0.1)
            latency = time.time() - start
            results["latencies"].append(latency)

            response_text = response.response.lower()

            # Check how many expected entities are found
            found_count = 0
            total_expected = 0
            missing = []

            for entity_type, entities in expected.items():
                for entity in entities:
                    total_expected += 1
                    if entity.lower() in response_text:
                        found_count += 1
                    else:
                        missing.append(f"{entity_type}:{entity}")

            # Classify result
            if found_count == total_expected:
                results["passed"] += 1
                status = "PASS"
            elif found_count > 0:
                results["partial"] += 1
                status = "PARTIAL"
            else:
                results["failed"] += 1
                status = "FAIL"

            results["details"].append({
                "case": i + 1,
                "text": text[:50] + "...",
                "status": status,
                "found": f"{found_count}/{total_expected}",
                "missing": missing,
                "latency": f"{latency:.2f}s",
            })

            logger.info(f"Case {i+1}: {status} ({found_count}/{total_expected}) - {latency:.2f}s")

        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "case": i + 1,
                "text": text[:50] + "...",
                "status": "ERROR",
                "error": str(e),
            })
            logger.error(f"Case {i+1}: ERROR - {e}")

    # Calculate metrics
    if results["latencies"]:
        results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
        results["min_latency"] = min(results["latencies"])
        results["max_latency"] = max(results["latencies"])

    results["accuracy"] = results["passed"] / results["total_cases"] if results["total_cases"] > 0 else 0
    results["partial_accuracy"] = (results["passed"] + results["partial"]) / results["total_cases"] if results["total_cases"] > 0 else 0

    return results


def compare_inference_performance(prompts: list[str], iterations: int = 3) -> dict[str, Any]:
    """Compare MLX vs Ollama inference performance."""
    results = {
        "ollama": {"available": False, "latencies": [], "errors": 0},
        "mlx": {"available": False, "latencies": [], "errors": 0},
    }

    # Test Ollama
    logger.info("Testing Ollama inference...")
    try:
        from nfl_analytics.semantic.models.ollama_client import OllamaClient
        client = OllamaClient(timeout=240.0, default_model="qwen3:8b")

        if client.is_available():
            results["ollama"]["available"] = True

            for iteration in range(iterations):
                for prompt in prompts:
                    start = time.time()
                    try:
                        response = client.generate(prompt, temperature=0.1)
                        latency = time.time() - start
                        results["ollama"]["latencies"].append(latency)
                    except Exception as e:
                        results["ollama"]["errors"] += 1
                        logger.warning(f"Ollama error: {e}")

            logger.info(f"Ollama: {len(results['ollama']['latencies'])} successful calls")
    except ImportError:
        logger.warning("Ollama client not available")

    # Test MLX
    logger.info("Testing MLX inference...")
    try:
        from nfl_analytics.semantic.models.mlx_client import MLXClient
        mlx_client = MLXClient()

        if mlx_client.is_available():
            results["mlx"]["available"] = True

            for iteration in range(iterations):
                for prompt in prompts:
                    start = time.time()
                    try:
                        response = mlx_client.generate(prompt, temperature=0.1)
                        latency = time.time() - start
                        results["mlx"]["latencies"].append(latency)
                    except Exception as e:
                        results["mlx"]["errors"] += 1
                        logger.warning(f"MLX error: {e}")

            logger.info(f"MLX: {len(results['mlx']['latencies'])} successful calls")
        else:
            logger.info("MLX not available (requires Apple Silicon with mlx-lm)")
    except ImportError:
        logger.warning("MLX client not available")

    # Calculate statistics
    for backend in ["ollama", "mlx"]:
        latencies = results[backend]["latencies"]
        if latencies:
            results[backend]["avg_latency"] = sum(latencies) / len(latencies)
            results[backend]["min_latency"] = min(latencies)
            results[backend]["max_latency"] = max(latencies)
            results[backend]["p50_latency"] = sorted(latencies)[len(latencies) // 2]
            results[backend]["total_calls"] = len(latencies)

    return results


def print_ner_report(results: dict) -> None:
    """Print NER test results."""
    print("\n" + "=" * 70)
    print("NER MODEL VALIDATION REPORT")
    print("=" * 70)

    if "error" in results:
        print(f"ERROR: {results['error']}")
        print("=" * 70)
        return

    print(f"Model: {results.get('model', 'unknown')}")
    print(f"Total cases: {results.get('total_cases', 0)}")
    print(f"Passed: {results['passed']} ({results['accuracy']:.1%})")
    print(f"Partial: {results['partial']}")
    print(f"Failed: {results['failed']}")
    print(f"Accuracy (strict): {results['accuracy']:.1%}")
    print(f"Accuracy (partial): {results['partial_accuracy']:.1%}")

    if "avg_latency" in results:
        print(f"\nLatency: avg={results['avg_latency']:.2f}s, min={results['min_latency']:.2f}s, max={results['max_latency']:.2f}s")

    print("-" * 70)
    print("Details:")
    for detail in results["details"]:
        status_icon = "✓" if detail["status"] == "PASS" else "~" if detail["status"] == "PARTIAL" else "✗"
        print(f"  [{status_icon}] Case {detail['case']}: {detail['status']} - {detail.get('found', 'N/A')}")
        if detail.get("missing"):
            print(f"      Missing: {', '.join(detail['missing'])}")

    print("=" * 70)


def print_inference_report(results: dict) -> None:
    """Print inference comparison results."""
    print("\n" + "=" * 70)
    print("MLX vs OLLAMA INFERENCE COMPARISON")
    print("=" * 70)

    for backend in ["ollama", "mlx"]:
        data = results[backend]
        print(f"\n{backend.upper()}:")
        print(f"  Available: {data['available']}")

        if data.get("total_calls"):
            print(f"  Total calls: {data['total_calls']}")
            print(f"  Errors: {data['errors']}")
            print(f"  Avg latency: {data['avg_latency']:.2f}s")
            print(f"  Min latency: {data['min_latency']:.2f}s")
            print(f"  Max latency: {data['max_latency']:.2f}s")
            print(f"  P50 latency: {data['p50_latency']:.2f}s")
            print(f"  Throughput: {data['total_calls'] / sum(data['latencies']):.2f} req/s")

    # Comparison
    if results["ollama"].get("avg_latency") and results["mlx"].get("avg_latency"):
        speedup = results["ollama"]["avg_latency"] / results["mlx"]["avg_latency"]
        faster = "MLX" if speedup > 1 else "Ollama"
        print(f"\n{faster} is {abs(speedup - 1) * 100 + 100 if speedup > 1 else 100 / speedup:.1f}% the speed of the other")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Semantic Stack models")
    parser.add_argument("--test-ner", action="store_true", help="Test NER model accuracy")
    parser.add_argument("--compare-inference", action="store_true", help="Compare MLX vs Ollama performance")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--iterations", type=int, default=2, help="Iterations for inference comparison")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    if not any([args.test_ner, args.compare_inference, args.all]):
        args.all = True

    all_results = {}

    if args.test_ner or args.all:
        logger.info("Running NER validation...")
        ner_results = test_ner_with_ollama(NER_TEST_CASES)
        all_results["ner"] = ner_results
        print_ner_report(ner_results)

    if args.compare_inference or args.all:
        logger.info("Running inference comparison...")
        inference_results = compare_inference_performance(BENCHMARK_PROMPTS, iterations=args.iterations)
        all_results["inference"] = inference_results
        print_inference_report(inference_results)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
