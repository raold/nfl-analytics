#!/usr/bin/env python
"""
Benchmark MLX vs Ollama inference performance for semantic stack.

This script compares:
- Latency (time per request, p50, p95, p99)
- Output quality (subjective comparison)
- Throughput (requests per second)

Results are saved to results/inference_benchmark.json for review.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --runs 5
    python scripts/benchmark_inference.py --ollama-only  # Skip MLX
    python scripts/benchmark_inference.py --mlx-only     # Skip Ollama
"""

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from inference clients
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mlx").setLevel(logging.WARNING)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    prompt_id: str
    prompt_type: str
    backend: str
    model: str
    latency_ms: float
    tokens_generated: int | None
    tokens_per_second: float | None
    output_text: str
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for a backend."""

    backend: str
    model: str
    total_runs: int
    successful_runs: int
    failed_runs: int

    # Latency stats (ms)
    latency_mean: float
    latency_median: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float

    # Throughput
    mean_tokens_per_second: float | None
    total_tokens: int | None

    # Quality (manual review needed)
    sample_outputs: dict[str, str]


# Representative prompts covering real semantic stack use cases
BENCHMARK_PROMPTS = {
    # Injury severity assessment (NER + severity scoring)
    "injury_1": {
        "type": "injury",
        "system": """You are an NFL injury impact analyst. Assess the severity and betting impact of injuries.
Output format: {"severity": 1-5, "impact": "high|medium|low", "reasoning": "brief explanation"}""",
        "prompt": "Patrick Mahomes - ankle injury, questionable for Sunday. Previously missed practice Wed/Thu, limited Friday.",
    },
    "injury_2": {
        "type": "injury",
        "system": """You are an NFL injury impact analyst. Assess the severity and betting impact of injuries.
Output format: {"severity": 1-5, "impact": "high|medium|low", "reasoning": "brief explanation"}""",
        "prompt": "Travis Kelce - knee, probable. Full participant in all practices this week.",
    },
    "injury_3": {
        "type": "injury",
        "system": """You are an NFL injury impact analyst. Assess the severity and betting impact of injuries.
Output format: {"severity": 1-5, "impact": "high|medium|low", "reasoning": "brief explanation"}""",
        "prompt": "Nick Bosa - torn ACL, out for season. Team defensive sack leader (15.5 sacks).",
    },

    # Sentiment analysis (news classification)
    "sentiment_1": {
        "type": "sentiment",
        "system": """You are an NFL sentiment analyst. Analyze team/player sentiment from news.
Output format: {"sentiment": "positive|neutral|negative", "confidence": 0.0-1.0, "reasoning": "brief"}""",
        "prompt": "Chiefs looking dominant after 45-10 blowout win. Mahomes threw 4 TDs with zero turnovers. Defense held opponent to 180 total yards.",
    },
    "sentiment_2": {
        "type": "sentiment",
        "system": """You are an NFL sentiment analyst. Analyze team/player sentiment from news.
Output format: {"sentiment": "positive|neutral|negative", "confidence": 0.0-1.0, "reasoning": "brief"}""",
        "prompt": "Raiders continue to struggle, now 3-11 on the season. Coaching staff under fire after fourth consecutive loss.",
    },
    "sentiment_3": {
        "type": "sentiment",
        "system": """You are an NFL sentiment analyst. Analyze team/player sentiment from news.
Output format: {"sentiment": "positive|neutral|negative", "confidence": 0.0-1.0, "reasoning": "brief"}""",
        "prompt": "Eagles-Cowboys rivalry game Sunday. Both teams 9-6, winner takes division lead heading into final two weeks.",
    },

    # Entity extraction (NER)
    "ner_1": {
        "type": "ner",
        "system": """You are an NFL entity extractor. Extract structured entities from text.
Categories: PLAYER, TEAM, INJURY, DATE, GAME, COACH, POSITION
Output format: {"entities": [{"text": "...", "type": "...", "confidence": 0.0-1.0}]}""",
        "prompt": "The Buffalo Bills will host the Miami Dolphins on Sunday Night Football. QB Josh Allen (shoulder) practiced fully Wednesday and is expected to start.",
    },
    "ner_2": {
        "type": "ner",
        "system": """You are an NFL entity extractor. Extract structured entities from text.
Categories: PLAYER, TEAM, INJURY, DATE, GAME, COACH, POSITION
Output format: {"entities": [{"text": "...", "type": "...", "confidence": 0.0-1.0}]}""",
        "prompt": "Head coach Kyle Shanahan confirmed RB Christian McCaffrey will return from injured reserve for Week 16 matchup against the Seahawks.",
    },
    "ner_3": {
        "type": "ner",
        "system": """You are an NFL entity extractor. Extract structured entities from text.
Categories: PLAYER, TEAM, INJURY, DATE, GAME, COACH, POSITION
Output format: {"entities": [{"text": "...", "type": "...", "confidence": 0.0-1.0}]}""",
        "prompt": "WR Tyreek Hill (ankle), CB Jalen Ramsey (hamstring), and LB Bradley Chubb (knee) all ruled out for Monday night's game vs Jets.",
    },

    # Complex reasoning (multi-factor analysis)
    "reasoning_1": {
        "type": "reasoning",
        "system": """You are an expert NFL betting analyst. Provide actionable insights considering multiple factors.
Focus on: injury impact, recent form, matchup dynamics, historical trends.
Output format: {"recommendation": "...", "confidence": 0.0-1.0, "key_factors": [...]}""",
        "prompt": """Analyze this matchup:
Chiefs (-7.5) at Raiders
- Chiefs: 11-4, won 5 of last 6
- Raiders: 6-9, lost 4 straight
- Chiefs missing: WR1 injured, LT questionable
- Raiders: Full roster, playing at home
- Historical: Chiefs 8-2 ATS in division games""",
    },
    "reasoning_2": {
        "type": "reasoning",
        "system": """You are an expert NFL betting analyst. Provide actionable insights considering multiple factors.
Focus on: injury impact, recent form, matchup dynamics, historical trends.
Output format: {"recommendation": "...", "confidence": 0.0-1.0, "key_factors": [...]}""",
        "prompt": """Analyze this total:
49ers vs Cowboys O/U 47.5
- 49ers: Top 5 offense, averaging 28.3 ppg
- Cowboys: Top 10 defense, allowing 21.1 ppg
- Weather: 45°F, 15mph winds, light rain
- Recent games: Under 4 of last 5 Cowboys games""",
    },

    # Coaching change detection
    "coaching_1": {
        "type": "coaching",
        "system": """You are an NFL coaching change detector. Identify coaching staff changes and assess impact.
Output format: {"change_detected": true|false, "type": "HC|OC|DC|position", "impact": "high|medium|low"}""",
        "prompt": "Panthers fire offensive coordinator Thomas Brown after team ranks 32nd in scoring. Former QB coach Chris Tabor promoted to interim OC.",
    },

    # Scheme analysis
    "scheme_1": {
        "type": "scheme",
        "system": """You are an NFL scheme analyst. Identify scheme changes and their betting relevance.
Output format: {"scheme_change": true|false, "description": "...", "expected_impact": "..."}""",
        "prompt": "Dolphins shifting to more run-heavy attack after ranking last in rushing. New run game coordinator hired from Shanahan tree.",
    },
}


def benchmark_ollama(
    client: Any,
    prompts: dict[str, dict],
    model: str = "qwen3:8b",
    temperature: float = 0.1,
    runs_per_prompt: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark Ollama inference."""
    logger.info(f"Benchmarking Ollama ({model})...")
    results = []

    for prompt_id, prompt_data in prompts.items():
        logger.info(f"  Running prompt: {prompt_id} (type: {prompt_data['type']})")

        for run in range(runs_per_prompt):
            try:
                start = time.time()

                # Use chat endpoint for instruct models (required for qwen3)
                messages = []
                if prompt_data.get("system"):
                    messages.append({"role": "system", "content": prompt_data["system"]})
                messages.append({"role": "user", "content": prompt_data["prompt"]})

                response = client.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                )

                elapsed = (time.time() - start) * 1000  # ms

                results.append(BenchmarkResult(
                    prompt_id=f"{prompt_id}_run{run+1}",
                    prompt_type=prompt_data["type"],
                    backend="ollama",
                    model=model,
                    latency_ms=elapsed,
                    tokens_generated=response.eval_count,
                    tokens_per_second=response.tokens_per_second,
                    output_text=response.response,
                ))

                logger.info(f"    Run {run+1}: {elapsed:.0f}ms, {response.tokens_per_second:.1f} tok/s")

            except Exception as e:
                logger.error(f"    Run {run+1} failed: {e}")
                results.append(BenchmarkResult(
                    prompt_id=f"{prompt_id}_run{run+1}",
                    prompt_type=prompt_data["type"],
                    backend="ollama",
                    model=model,
                    latency_ms=0.0,
                    tokens_generated=None,
                    tokens_per_second=None,
                    output_text="",
                    error=str(e),
                ))

    return results


def benchmark_mlx(
    client: Any,
    prompts: dict[str, dict],
    model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
    temperature: float = 0.1,
    runs_per_prompt: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark MLX inference."""
    logger.info(f"Benchmarking MLX ({model})...")
    results = []

    for prompt_id, prompt_data in prompts.items():
        logger.info(f"  Running prompt: {prompt_id} (type: {prompt_data['type']})")

        for run in range(runs_per_prompt):
            try:
                response = client.generate(
                    prompt=prompt_data["prompt"],
                    model=model,
                    system=prompt_data.get("system"),
                    temperature=temperature,
                )

                results.append(BenchmarkResult(
                    prompt_id=f"{prompt_id}_run{run+1}",
                    prompt_type=prompt_data["type"],
                    backend="mlx",
                    model=model,
                    latency_ms=response.total_time_ms,
                    tokens_generated=response.tokens_generated,
                    tokens_per_second=response.tokens_per_second,
                    output_text=response.text,
                ))

                logger.info(f"    Run {run+1}: {response.total_time_ms:.0f}ms, {response.tokens_per_second:.1f} tok/s")

            except Exception as e:
                logger.error(f"    Run {run+1} failed: {e}")
                results.append(BenchmarkResult(
                    prompt_id=f"{prompt_id}_run{run+1}",
                    prompt_type=prompt_data["type"],
                    backend="mlx",
                    model=model,
                    latency_ms=0.0,
                    tokens_generated=None,
                    tokens_per_second=None,
                    output_text="",
                    error=str(e),
                ))

    return results


def compute_summary(results: list[BenchmarkResult], backend: str) -> BenchmarkSummary:
    """Compute summary statistics for a backend."""
    backend_results = [r for r in results if r.backend == backend and r.error is None]

    if not backend_results:
        return BenchmarkSummary(
            backend=backend,
            model="N/A",
            total_runs=len([r for r in results if r.backend == backend]),
            successful_runs=0,
            failed_runs=len([r for r in results if r.backend == backend]),
            latency_mean=0.0,
            latency_median=0.0,
            latency_p95=0.0,
            latency_p99=0.0,
            latency_min=0.0,
            latency_max=0.0,
            mean_tokens_per_second=None,
            total_tokens=None,
            sample_outputs={},
        )

    latencies = [r.latency_ms for r in backend_results]
    latencies_sorted = sorted(latencies)

    tps_values = [r.tokens_per_second for r in backend_results if r.tokens_per_second is not None]
    token_counts = [r.tokens_generated for r in backend_results if r.tokens_generated is not None]

    # Sample outputs (one per prompt type)
    sample_outputs = {}
    for result in backend_results:
        if result.prompt_type not in sample_outputs:
            sample_outputs[result.prompt_type] = result.output_text[:200] + "..." if len(result.output_text) > 200 else result.output_text

    return BenchmarkSummary(
        backend=backend,
        model=backend_results[0].model,
        total_runs=len([r for r in results if r.backend == backend]),
        successful_runs=len(backend_results),
        failed_runs=len([r for r in results if r.backend == backend and r.error is not None]),
        latency_mean=statistics.mean(latencies),
        latency_median=statistics.median(latencies),
        latency_p95=latencies_sorted[int(len(latencies_sorted) * 0.95)] if len(latencies_sorted) > 20 else max(latencies_sorted),
        latency_p99=latencies_sorted[int(len(latencies_sorted) * 0.99)] if len(latencies_sorted) > 100 else max(latencies_sorted),
        latency_min=min(latencies),
        latency_max=max(latencies),
        mean_tokens_per_second=statistics.mean(tps_values) if tps_values else None,
        total_tokens=sum(token_counts) if token_counts else None,
        sample_outputs=sample_outputs,
    )


def print_summary(summary: BenchmarkSummary):
    """Print formatted summary."""
    print(f"\n{'='*80}")
    print(f"{summary.backend.upper()} BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Model: {summary.model}")
    print(f"Runs: {summary.successful_runs}/{summary.total_runs} successful")

    if summary.failed_runs > 0:
        print(f"FAILURES: {summary.failed_runs}")

    print(f"\nLATENCY (ms):")
    print(f"  Mean:   {summary.latency_mean:8.1f}")
    print(f"  Median: {summary.latency_median:8.1f}")
    print(f"  Min:    {summary.latency_min:8.1f}")
    print(f"  Max:    {summary.latency_max:8.1f}")
    print(f"  p95:    {summary.latency_p95:8.1f}")
    print(f"  p99:    {summary.latency_p99:8.1f}")

    if summary.mean_tokens_per_second:
        print(f"\nTHROUGHPUT:")
        print(f"  Mean tokens/sec: {summary.mean_tokens_per_second:.1f}")
        print(f"  Total tokens:    {summary.total_tokens}")

    print(f"\nSAMPLE OUTPUTS (first 200 chars):")
    for prompt_type, output in summary.sample_outputs.items():
        print(f"\n  [{prompt_type}]")
        print(f"  {output}")


def compare_backends(ollama_summary: BenchmarkSummary, mlx_summary: BenchmarkSummary):
    """Compare two backends."""
    print(f"\n{'='*80}")
    print("COMPARISON: MLX vs OLLAMA")
    print(f"{'='*80}")

    if ollama_summary.successful_runs == 0 or mlx_summary.successful_runs == 0:
        print("ERROR: One or both backends had no successful runs")
        return

    # Latency comparison
    speedup = ollama_summary.latency_median / mlx_summary.latency_median
    print(f"\nLATENCY (median):")
    print(f"  Ollama: {ollama_summary.latency_median:.1f}ms")
    print(f"  MLX:    {mlx_summary.latency_median:.1f}ms")
    if speedup > 1:
        print(f"  Winner: MLX ({speedup:.2f}x faster)")
    elif speedup < 1:
        print(f"  Winner: Ollama ({1/speedup:.2f}x faster)")
    else:
        print(f"  Tie: Same performance")

    # Throughput comparison
    if ollama_summary.mean_tokens_per_second and mlx_summary.mean_tokens_per_second:
        tps_ratio = mlx_summary.mean_tokens_per_second / ollama_summary.mean_tokens_per_second
        print(f"\nTHROUGHPUT (tokens/sec):")
        print(f"  Ollama: {ollama_summary.mean_tokens_per_second:.1f}")
        print(f"  MLX:    {mlx_summary.mean_tokens_per_second:.1f}")
        if tps_ratio > 1:
            print(f"  Winner: MLX ({tps_ratio:.2f}x faster)")
        elif tps_ratio < 1:
            print(f"  Winner: Ollama ({1/tps_ratio:.2f}x faster)")

    # Reliability
    print(f"\nRELIABILITY:")
    print(f"  Ollama: {ollama_summary.successful_runs}/{ollama_summary.total_runs} ({100*ollama_summary.successful_runs/ollama_summary.total_runs:.1f}%)")
    print(f"  MLX:    {mlx_summary.successful_runs}/{mlx_summary.total_runs} ({100*mlx_summary.successful_runs/mlx_summary.total_runs:.1f}%)")

    # Recommendation
    print(f"\nRECOMMENDATION:")
    if mlx_summary.latency_median < ollama_summary.latency_median * 0.7:
        print("  Use MLX for production (significantly faster)")
    elif mlx_summary.latency_median < ollama_summary.latency_median * 0.9:
        print("  Prefer MLX for production (faster with similar reliability)")
    elif ollama_summary.successful_runs > mlx_summary.successful_runs:
        print("  Use Ollama for production (more reliable)")
    else:
        print("  Either backend acceptable (similar performance)")
        print("  Consider: MLX for Mac M-series, Ollama for cross-platform")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX vs Ollama inference")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt (default: 3)")
    parser.add_argument("--ollama-only", action="store_true", help="Only benchmark Ollama")
    parser.add_argument("--mlx-only", action="store_true", help="Only benchmark MLX")
    parser.add_argument("--output", type=Path, default=Path("results/inference_benchmark.json"),
                        help="Output file for results")
    args = parser.parse_args()

    # Ensure results directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    summaries = {}

    # Benchmark Ollama
    if not args.mlx_only:
        try:
            from py.semantic.models.ollama_client import OllamaClient

            ollama = OllamaClient(timeout=240.0)

            if not ollama.is_available():
                logger.error("Ollama server not running. Start with: ollama serve")
                if args.ollama_only:
                    return
            else:
                ollama_results = benchmark_ollama(
                    ollama,
                    BENCHMARK_PROMPTS,
                    runs_per_prompt=args.runs,
                )
                all_results.extend(ollama_results)
                summaries["ollama"] = compute_summary(ollama_results, "ollama")
                print_summary(summaries["ollama"])
        except Exception as e:
            logger.error(f"Ollama benchmark failed: {e}")
            if args.ollama_only:
                return

    # Benchmark MLX
    if not args.ollama_only:
        try:
            from py.semantic.models.mlx_client import MLXClient, check_mlx_available

            if not check_mlx_available():
                logger.error("MLX not available. Requires Apple Silicon Mac with macOS 14+")
                if args.mlx_only:
                    return
            else:
                mlx = MLXClient()
                mlx_results = benchmark_mlx(
                    mlx,
                    BENCHMARK_PROMPTS,
                    runs_per_prompt=args.runs,
                )
                all_results.extend(mlx_results)
                summaries["mlx"] = compute_summary(mlx_results, "mlx")
                print_summary(summaries["mlx"])
        except Exception as e:
            logger.error(f"MLX benchmark failed: {e}")
            if args.mlx_only:
                return

    # Compare backends
    if "ollama" in summaries and "mlx" in summaries:
        compare_backends(summaries["ollama"], summaries["mlx"])

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "runs_per_prompt": args.runs,
            "total_prompts": len(BENCHMARK_PROMPTS),
        },
        "summaries": {k: asdict(v) for k, v in summaries.items()},
        "detailed_results": [asdict(r) for r in all_results],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")

    # Quality comparison note
    print(f"\n{'='*80}")
    print("QUALITY COMPARISON NOTE")
    print(f"{'='*80}")
    print("Review sample outputs above to assess quality differences.")
    print("Both models use similar Qwen architecture, so outputs should be comparable.")
    print(f"Detailed results with full outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
