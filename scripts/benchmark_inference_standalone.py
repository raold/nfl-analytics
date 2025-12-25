#!/usr/bin/env python
"""
Standalone inference benchmark for MLX vs Ollama.

This is a self-contained version that doesn't rely on complex imports.
"""

import argparse
import httpx
import json
import logging
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logs
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


# Test prompts
BENCHMARK_PROMPTS = {
    "injury_1": {
        "type": "injury",
        "system": "You are an NFL injury analyst. Assess severity (1-5) and impact (high/medium/low).",
        "prompt": "Patrick Mahomes - ankle injury, questionable for Sunday.",
    },
    "injury_2": {
        "type": "injury",
        "system": "You are an NFL injury analyst. Assess severity (1-5) and impact (high/medium/low).",
        "prompt": "Nick Bosa - torn ACL, out for season. Team defensive sack leader.",
    },
    "sentiment_1": {
        "type": "sentiment",
        "system": "You are an NFL sentiment analyst. Classify as positive/neutral/negative.",
        "prompt": "Chiefs looking dominant after 45-10 win. Mahomes threw 4 TDs.",
    },
    "sentiment_2": {
        "type": "sentiment",
        "system": "You are an NFL sentiment analyst. Classify as positive/neutral/negative.",
        "prompt": "Raiders continue to struggle, now 3-11. Coaching staff under fire.",
    },
    "ner_1": {
        "type": "ner",
        "system": "Extract entities: PLAYER, TEAM, INJURY, DATE. Output JSON.",
        "prompt": "Bills host Dolphins Sunday. QB Josh Allen (shoulder) practiced fully.",
    },
    "ner_2": {
        "type": "ner",
        "system": "Extract entities: PLAYER, TEAM, INJURY, DATE. Output JSON.",
        "prompt": "Kyle Shanahan confirmed RB Christian McCaffrey returns Week 16.",
    },
}


def benchmark_ollama(
    prompts: dict,
    model: str = "qwen3:8b",
    runs_per_prompt: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark Ollama inference."""
    logger.info(f"Benchmarking Ollama ({model})...")
    results = []

    client = httpx.Client(timeout=240.0)
    base_url = "http://localhost:11434"

    for prompt_id, prompt_data in prompts.items():
        logger.info(f"  Prompt: {prompt_id} ({prompt_data['type']})")

        for run in range(runs_per_prompt):
            try:
                messages = []
                if prompt_data.get("system"):
                    messages.append({"role": "system", "content": prompt_data["system"]})
                messages.append({"role": "user", "content": prompt_data["prompt"]})

                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1},
                }

                start = time.time()
                response = client.post(f"{base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                elapsed = (time.time() - start) * 1000

                output = data.get("message", {}).get("content", "")
                eval_count = data.get("eval_count", 0)
                total_duration_ns = data.get("total_duration", 0)

                tps = eval_count / (total_duration_ns / 1e9) if total_duration_ns > 0 else 0

                results.append(BenchmarkResult(
                    prompt_id=f"{prompt_id}_run{run+1}",
                    prompt_type=prompt_data["type"],
                    backend="ollama",
                    model=model,
                    latency_ms=elapsed,
                    tokens_generated=eval_count,
                    tokens_per_second=tps,
                    output_text=output,
                ))

                logger.info(f"    Run {run+1}: {elapsed:.0f}ms, {tps:.1f} tok/s")

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

    client.close()
    return results


def benchmark_mlx(
    prompts: dict,
    model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
    runs_per_prompt: int = 3,
) -> list[BenchmarkResult]:
    """Benchmark MLX inference."""
    logger.info(f"Benchmarking MLX ({model})...")

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        logger.error("MLX not installed: pip install mlx-lm")
        return []

    # Load model once
    logger.info(f"  Loading model: {model}")
    start_load = time.time()
    mlx_model, tokenizer = load(model)
    load_time = time.time() - start_load
    logger.info(f"  Model loaded in {load_time:.1f}s")

    results = []
    sampler = make_sampler(temp=0.1)

    for prompt_id, prompt_data in prompts.items():
        logger.info(f"  Prompt: {prompt_id} ({prompt_data['type']})")

        for run in range(runs_per_prompt):
            try:
                # Build prompt
                if prompt_data.get("system"):
                    full_prompt = f"<|im_start|>system\n{prompt_data['system']}<|im_end|>\n<|im_start|>user\n{prompt_data['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    full_prompt = prompt_data["prompt"]

                start = time.time()
                output = generate(
                    mlx_model,
                    tokenizer,
                    prompt=full_prompt,
                    max_tokens=512,
                    sampler=sampler,
                    verbose=False,
                )
                elapsed = (time.time() - start) * 1000

                tokens_generated = len(tokenizer.encode(output))
                tps = tokens_generated / (elapsed / 1000) if elapsed > 0 else 0

                results.append(BenchmarkResult(
                    prompt_id=f"{prompt_id}_run{run+1}",
                    prompt_type=prompt_data["type"],
                    backend="mlx",
                    model=model,
                    latency_ms=elapsed,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tps,
                    output_text=output.strip(),
                ))

                logger.info(f"    Run {run+1}: {elapsed:.0f}ms, {tps:.1f} tok/s")

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


def compute_summary(results: list[BenchmarkResult], backend: str) -> dict:
    """Compute summary statistics."""
    backend_results = [r for r in results if r.backend == backend and r.error is None]

    if not backend_results:
        return {
            "backend": backend,
            "total_runs": len([r for r in results if r.backend == backend]),
            "successful_runs": 0,
            "failed_runs": len([r for r in results if r.backend == backend]),
        }

    latencies = [r.latency_ms for r in backend_results]
    latencies_sorted = sorted(latencies)

    tps_values = [r.tokens_per_second for r in backend_results if r.tokens_per_second is not None]

    # Sample outputs
    sample_outputs = {}
    for result in backend_results:
        if result.prompt_type not in sample_outputs:
            output = result.output_text[:200] + "..." if len(result.output_text) > 200 else result.output_text
            sample_outputs[result.prompt_type] = output

    return {
        "backend": backend,
        "model": backend_results[0].model,
        "total_runs": len([r for r in results if r.backend == backend]),
        "successful_runs": len(backend_results),
        "failed_runs": len([r for r in results if r.backend == backend and r.error is not None]),
        "latency_mean": statistics.mean(latencies),
        "latency_median": statistics.median(latencies),
        "latency_p95": latencies_sorted[int(len(latencies_sorted) * 0.95)] if len(latencies_sorted) > 20 else max(latencies_sorted),
        "latency_min": min(latencies),
        "latency_max": max(latencies),
        "mean_tokens_per_second": statistics.mean(tps_values) if tps_values else None,
        "sample_outputs": sample_outputs,
    }


def print_summary(summary: dict):
    """Print formatted summary."""
    print(f"\n{'='*80}")
    print(f"{summary['backend'].upper()} BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Model: {summary.get('model', 'N/A')}")
    print(f"Runs: {summary['successful_runs']}/{summary['total_runs']} successful")

    if summary["failed_runs"] > 0:
        print(f"FAILURES: {summary['failed_runs']}")

    if summary['successful_runs'] > 0:
        print(f"\nLATENCY (ms):")
        print(f"  Mean:   {summary['latency_mean']:8.1f}")
        print(f"  Median: {summary['latency_median']:8.1f}")
        print(f"  Min:    {summary['latency_min']:8.1f}")
        print(f"  Max:    {summary['latency_max']:8.1f}")
        print(f"  p95:    {summary['latency_p95']:8.1f}")

        if summary.get('mean_tokens_per_second'):
            print(f"\nTHROUGHPUT:")
            print(f"  Mean tokens/sec: {summary['mean_tokens_per_second']:.1f}")

        print(f"\nSAMPLE OUTPUTS (first 200 chars):")
        for prompt_type, output in summary.get('sample_outputs', {}).items():
            print(f"\n  [{prompt_type}]")
            print(f"  {output}")


def compare_backends(ollama_summary: dict, mlx_summary: dict):
    """Compare two backends."""
    print(f"\n{'='*80}")
    print("COMPARISON: MLX vs OLLAMA")
    print(f"{'='*80}")

    if ollama_summary['successful_runs'] == 0 or mlx_summary['successful_runs'] == 0:
        print("ERROR: One or both backends had no successful runs")
        return

    # Latency comparison
    speedup = ollama_summary['latency_median'] / mlx_summary['latency_median']
    print(f"\nLATENCY (median):")
    print(f"  Ollama: {ollama_summary['latency_median']:.1f}ms")
    print(f"  MLX:    {mlx_summary['latency_median']:.1f}ms")
    if speedup > 1.1:
        print(f"  Winner: MLX ({speedup:.2f}x faster)")
    elif speedup < 0.9:
        print(f"  Winner: Ollama ({1/speedup:.2f}x faster)")
    else:
        print(f"  Tie: Similar performance")

    # Throughput comparison
    if ollama_summary.get('mean_tokens_per_second') and mlx_summary.get('mean_tokens_per_second'):
        tps_ratio = mlx_summary['mean_tokens_per_second'] / ollama_summary['mean_tokens_per_second']
        print(f"\nTHROUGHPUT (tokens/sec):")
        print(f"  Ollama: {ollama_summary['mean_tokens_per_second']:.1f}")
        print(f"  MLX:    {mlx_summary['mean_tokens_per_second']:.1f}")
        if tps_ratio > 1.1:
            print(f"  Winner: MLX ({tps_ratio:.2f}x faster)")
        elif tps_ratio < 0.9:
            print(f"  Winner: Ollama ({1/tps_ratio:.2f}x faster)")

    # Recommendation
    print(f"\nRECOMMENDATION:")
    if mlx_summary['latency_median'] < ollama_summary['latency_median'] * 0.7:
        print("  Use MLX for production (significantly faster)")
    elif mlx_summary['latency_median'] < ollama_summary['latency_median'] * 0.9:
        print("  Prefer MLX for production (faster)")
    elif ollama_summary['successful_runs'] > mlx_summary['successful_runs']:
        print("  Use Ollama for production (more reliable)")
    else:
        print("  Either backend acceptable (similar performance)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX vs Ollama")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt")
    parser.add_argument("--ollama-only", action="store_true")
    parser.add_argument("--mlx-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/inference_benchmark.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    summaries = {}

    # Benchmark Ollama
    if not args.mlx_only:
        try:
            ollama_results = benchmark_ollama(BENCHMARK_PROMPTS, runs_per_prompt=args.runs)
            all_results.extend(ollama_results)
            summaries["ollama"] = compute_summary(ollama_results, "ollama")
            print_summary(summaries["ollama"])
        except Exception as e:
            logger.error(f"Ollama benchmark failed: {e}")

    # Benchmark MLX
    if not args.ollama_only:
        try:
            mlx_results = benchmark_mlx(BENCHMARK_PROMPTS, runs_per_prompt=args.runs)
            all_results.extend(mlx_results)
            summaries["mlx"] = compute_summary(mlx_results, "mlx")
            print_summary(summaries["mlx"])
        except Exception as e:
            logger.error(f"MLX benchmark failed: {e}")

    # Compare
    if "ollama" in summaries and "mlx" in summaries:
        compare_backends(summaries["ollama"], summaries["mlx"])

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "runs_per_prompt": args.runs,
            "total_prompts": len(BENCHMARK_PROMPTS),
        },
        "summaries": summaries,
        "detailed_results": [asdict(r) for r in all_results],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
