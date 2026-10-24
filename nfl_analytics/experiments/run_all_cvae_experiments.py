#!/usr/bin/env python3
"""
Master script to run all 7 cVAE experiments sequentially.

Each experiment is independent and can be run individually or as part of this suite.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import subprocess
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS = [
    {
        "name": "Experiment 1: cVAE + Environment BNN Ensemble",
        "script": "py/experiments/cvae_exp1_environment_bnn_ensemble.py",
        "description": "Test if trajectory features improve Environment BNN predictions",
    },
    {
        "name": "Experiment 2: Same-Game Parlay Pricing",
        "script": "py/experiments/cvae_exp2_same_game_parlay.py",
        "description": "Price complex parlays and validate with BNN calibration",
    },
    {
        "name": "Experiment 3: Causal Counterfactual Generation",
        "script": "py/experiments/cvae_exp3_causal_counterfactuals.py",
        "description": "Generate counterfactual trajectories for causal questions",
    },
    {
        "name": "Experiment 4: Live In-Game Trajectory Updates",
        "script": "py/experiments/cvae_exp4_live_trajectories.py",
        "description": "Generate remainder-of-game trajectories from current state",
    },
    {
        "name": "Experiment 5: Key Number Distribution Validation",
        "script": "py/experiments/cvae_exp5_key_numbers.py",
        "description": "Validate cVAE learns NFL key number patterns",
    },
    {
        "name": "Experiment 6: GNN-Enhanced Trajectory Conditioning",
        "script": "py/experiments/cvae_exp6_gnn_enhanced.py",
        "description": "Add GNN embeddings to cVAE conditioning",
    },
    {
        "name": "Experiment 7: Hierarchical Player-Game Trajectories",
        "script": "py/experiments/cvae_exp7_hierarchical_player.py",
        "description": "Combine cVAE with hierarchical BNN player models",
    },
]


def run_experiment(exp_config):
    """Run a single experiment"""
    logger.info("=" * 80)
    logger.info(f"{exp_config['name']}")
    logger.info("=" * 80)
    logger.info(f"Description: {exp_config['description']}")
    logger.info("")

    script_path = Path(exp_config["script"])

    if not script_path.exists():
        logger.error(f"✗ Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per experiment
        )

        if result.returncode == 0:
            logger.info("✓ Experiment completed successfully")
            logger.info("")
            return True
        else:
            logger.error(f"✗ Experiment failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("✗ Experiment timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"✗ Experiment failed with exception: {e}")
        return False


def main():
    logger.info("")
    logger.info("=" * 80)
    logger.info("cVAE EXPERIMENT SUITE - ALL 7 EXPERIMENTS")
    logger.info("=" * 80)
    logger.info("")

    results = {}

    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        results[exp["name"]] = "PASS" if success else "FAIL"

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUITE SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    for exp_name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        logger.info(f"{symbol} {exp_name}: {status}")

    total = len(results)
    passed = sum(1 for status in results.values() if status == "PASS")

    logger.info("")
    logger.info(f"Results: {passed}/{total} experiments passed")
    logger.info("")

    return all(status == "PASS" for status in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
