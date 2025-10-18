#!/usr/bin/env python3
"""
Submit hyperparameter sweeps to the training queue.

Supports:
- YAML config files with parameter grids
- Command-line parameter override
- Sweep visualization and estimation

Usage:
    # Submit CQL alpha sweep
    python py/compute/submit_sweep.py \
        --model cql \
        --config sweeps/cql_alpha_sweep.yaml \
        --priority 10

    # Submit with custom parameters
    python py/compute/submit_sweep.py \
        --model iql \
        --base-config '{"batch_size": 256, "epochs": 200}' \
        --param alpha 0.1 0.5 1.0 \
        --param lr 1e-5 5e-5 1e-4 \
        --priority 8

    # Dry run (don't actually submit)
    python py/compute/submit_sweep.py \
        --model cql \
        --config sweeps/cql_alpha_sweep.yaml \
        --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute.tasks.training_task import HyperparameterSweep, get_queue_status, get_redis_client


def load_sweep_config(config_path: Path) -> dict[str, Any]:
    """
    Load sweep configuration from YAML file.

    Expected format:
        model_type: cql
        base_config:
          batch_size: 256
          epochs: 200
          data_path: data/rl_logged.csv
        param_grid:
          alpha: [0.1, 0.5, 1.0, 2.0, 5.0]
          lr: [1e-5, 5e-5, 1e-4]
          layers: [4, 5, 6]
        priority: 10
        min_gpu_memory: 16
        estimated_hours: 5.0

    Returns:
        Dict with sweep configuration
    """
    with config_path.open() as f:
        config = yaml.safe_load(f)

    return config


def estimate_sweep(sweep: HyperparameterSweep) -> dict[str, Any]:
    """
    Estimate sweep statistics.

    Returns:
        Dict with total configs, estimated time, etc.
    """
    configs = sweep.generate_configs()
    total_configs = len(configs)
    total_hours = total_configs * sweep.estimated_hours

    # Estimate wall-clock time on different setups
    estimate = {
        "total_configs": total_configs,
        "estimated_hours_per_config": sweep.estimated_hours,
        "total_gpu_hours": total_hours,
        "wall_clock_1xrtx": f"{total_hours:.1f} hours ({total_hours/24:.1f} days)",
        "wall_clock_2xrtx": f"{total_hours/2:.1f} hours ({total_hours/48:.1f} days)",
        "wall_clock_m4_only": f"{total_hours*10:.1f} hours ({total_hours*10/24:.1f} days)",  # ~10x slower
        "approx_cost_cloud": f"${total_hours * 32:.0f}",  # $32/hour for p4d.24xlarge
    }

    return estimate


def print_sweep_summary(sweep: HyperparameterSweep, estimate: dict[str, Any]):
    """Print formatted sweep summary."""
    print("=" * 60)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 60)
    print(f"Model Type: {sweep.model_type}")
    print(f"Priority: {sweep.priority}")
    print(f"Min GPU Memory: {sweep.min_gpu_memory} GB")
    print()

    print("Base Config:")
    for key, value in sweep.base_config.items():
        print(f"  {key}: {value}")
    print()

    print("Parameter Grid:")
    for key, values in sweep.param_grid.items():
        print(f"  {key}: {values}")
    print()

    print("Estimates:")
    print(f"  Total Configurations: {estimate['total_configs']}")
    print(f"  Hours per Config: {estimate['estimated_hours_per_config']}")
    print(f"  Total GPU-Hours: {estimate['total_gpu_hours']:.1f}")
    print()

    print("Wall-Clock Time:")
    print(f"  1× RTX 5090: {estimate['wall_clock_1xrtx']}")
    print(f"  2× RTX 5090: {estimate['wall_clock_2xrtx']}")
    print(f"  M4 MacBook: {estimate['wall_clock_m4_only']}")
    print()

    print(f"Cloud Cost (approx): {estimate['approx_cost_cloud']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Submit hyperparameter sweep to training queue")

    # Config source
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", type=Path, help="Path to YAML config file")
    config_group.add_argument("--model", help="Model type (for manual parameter specification)")

    # Manual config (if not using YAML)
    parser.add_argument(
        "--base-config",
        type=json.loads,
        default={},
        help="Base config as JSON string (e.g., '{\"batch_size\": 256}')",
    )
    parser.add_argument(
        "--param", action="append", nargs="+", help="Parameter grid: --param alpha 0.1 0.5 1.0"
    )
    parser.add_argument("--priority", type=int, default=5, help="Task priority (1-10, default: 5)")
    parser.add_argument(
        "--min-gpu-memory", type=int, default=8, help="Minimum GPU memory in GB (default: 8)"
    )
    parser.add_argument(
        "--estimated-hours",
        type=float,
        default=1.0,
        help="Estimated hours per config (default: 1.0)",
    )

    # Redis connection
    parser.add_argument("--redis-host", default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port (default: 6379)")

    # Actions
    parser.add_argument("--dry-run", action="store_true", help="Print summary without submitting")

    args = parser.parse_args()

    # Load or build sweep config
    if args.config:
        print(f"Loading sweep config from {args.config}...")
        config = load_sweep_config(args.config)

        sweep = HyperparameterSweep(
            model_type=config["model_type"],
            base_config=config["base_config"],
            param_grid=config["param_grid"],
            priority=config.get("priority", 5),
            min_gpu_memory=config.get("min_gpu_memory", 8),
            estimated_hours=config.get("estimated_hours", 1.0),
        )

    else:
        # Build from command-line args
        if not args.param:
            parser.error("--param required when not using --config")

        param_grid = {}
        for param_spec in args.param:
            param_name = param_spec[0]
            param_values = param_spec[1:]

            # Try to parse as numbers
            try:
                param_values = [float(v) if "." in v or "e" in v else int(v) for v in param_values]
            except ValueError:
                pass  # Keep as strings

            param_grid[param_name] = param_values

        sweep = HyperparameterSweep(
            model_type=args.model,
            base_config=args.base_config,
            param_grid=param_grid,
            priority=args.priority,
            min_gpu_memory=args.min_gpu_memory,
            estimated_hours=args.estimated_hours,
        )

    # Estimate sweep
    estimate = estimate_sweep(sweep)
    print_sweep_summary(sweep, estimate)

    # Dry run check
    if args.dry_run:
        print("\n[DRY RUN] Not submitting tasks")
        return

    # Confirm submission
    confirm = input("\nSubmit this sweep? (y/N): ")
    if confirm.lower() != "y":
        print("Cancelled")
        return

    # Connect to Redis and submit
    redis_client = get_redis_client(host=args.redis_host, port=args.redis_port)

    # Show queue status before
    print("\nQueue status before submission:")
    status = get_queue_status(redis_client)
    print(f"  Pending tasks: {status['total_pending']}")
    print(f"  Priority breakdown: {status['priority_breakdown']}")

    # Submit
    print("\nSubmitting tasks...")
    task_ids = sweep.submit(redis_client)

    # Show queue status after
    print("\nQueue status after submission:")
    status = get_queue_status(redis_client)
    print(f"  Pending tasks: {status['total_pending']}")
    print(f"  Priority breakdown: {status['priority_breakdown']}")

    print(f"\n✓ Successfully submitted {len(task_ids)} tasks")
    print("\nNext steps:")
    print("  1. Start workers: python py/compute/worker_enhanced.py --worker-id <id>")
    print("  2. Monitor progress: python py/compute/dashboard.py")
    print(f"  3. View results: python py/compute/model_registry.py list {sweep.model_type}")


if __name__ == "__main__":
    main()
