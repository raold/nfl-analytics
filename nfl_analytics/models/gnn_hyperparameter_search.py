#!/usr/bin/env python3
"""
GNN Hyperparameter Search using Optuna

Performs Bayesian optimization over GNN hyperparameters to find optimal configuration.
Uses Optuna's Tree-structured Parzen Estimator (TPE) for efficient search.

Usage:
    python nfl_analytics/models/gnn_hyperparameter_search.py --trials 50 --device cpu

Search Space:
    - embedding_dim: [16, 32, 64, 96, 128]
    - hidden_dim: [32, 64, 128, 192, 256]
    - num_message_rounds: [1, 2, 3, 4, 5]
    - learning_rate: [1e-4, 1e-3, 1e-2] (log scale)
    - batch_size: [32, 64, 128, 256]
    - use_chemistry: [True, False]

Objective: Minimize validation Brier score
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "nfl_analytics" / "models"))

from train_hierarchical_gnn import train_gnn_single_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Optuna objective function: train GNN with sampled hyperparameters.

    Returns:
        Best validation Brier score (to minimize)
    """

    # Sample hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64, 96, 128])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 192, 256])
    num_message_rounds = trial.suggest_int("num_message_rounds", 1, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    use_chemistry = trial.suggest_categorical("use_chemistry", [True, False])

    # Pruning constraint: skip if hidden_dim < embedding_dim (unstable)
    if hidden_dim < embedding_dim:
        raise optuna.TrialPruned("Hidden dim must be >= embedding dim")

    logger.info(f"\nTrial {trial.number}: Testing configuration:")
    logger.info(f"  embedding_dim: {embedding_dim}")
    logger.info(f"  hidden_dim: {hidden_dim}")
    logger.info(f"  num_message_rounds: {num_message_rounds}")
    logger.info(f"  learning_rate: {learning_rate:.6f}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  use_chemistry: {use_chemistry}")

    # Train GNN with sampled hyperparameters
    try:
        result = train_gnn_single_config(
            seasons=args.seasons,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_message_rounds=num_message_rounds,
            epochs=args.epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_chemistry=use_chemistry,
            device=args.device,
            output_dir=Path(args.output_dir) / f"trial_{trial.number}",
        )

        val_brier = result["best_val_brier"]
        test_brier = result["test_metrics"]["brier"]

        logger.info(f"  ✓ Completed: val_brier={val_brier:.4f}, test_brier={test_brier:.4f}")

        # Report intermediate values for pruning
        trial.report(val_brier, step=args.epochs)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_brier

    except Exception as e:
        logger.error(f"  ✗ Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_hyperparameter_search(args: argparse.Namespace):
    """Execute Optuna hyperparameter search."""

    logger.info("=" * 80)
    logger.info("GNN HYPERPARAMETER SEARCH WITH OPTUNA")
    logger.info("=" * 80)
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Epochs per trial: {args.epochs}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        study_name="gnn_hyperparameter_search",
        direction="minimize",  # Minimize validation Brier score
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args), n_trials=args.trials, show_progress_bar=True
    )

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH COMPLETE")
    logger.info("=" * 80)
    logger.info("")

    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(
        f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    logger.info(
        f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )
    logger.info("")

    # Best trial
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"  Best val Brier: {best_trial.value:.4f}")
    logger.info("")
    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    logger.info("")

    # Top 5 trials
    logger.info("Top 5 trials:")
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )[:5]

    for i, trial in enumerate(sorted_trials, 1):
        logger.info(f"{i}. Trial {trial.number}: {trial.value:.4f}")
        for key, value in trial.params.items():
            logger.info(f"     {key}: {value}")
        logger.info("")

    # Save results
    results = {
        "best_trial": {
            "number": best_trial.number,
            "val_brier": best_trial.value,
            "params": best_trial.params,
        },
        "top_5_trials": [
            {"number": t.number, "val_brier": t.value, "params": t.params} for t in sorted_trials
        ],
        "study_stats": {
            "n_trials": len(study.trials),
            "n_complete": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        },
    }

    results_path = output_dir / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Saved results to: {results_path}")
    logger.info("")

    # Plot optimization history
    try:

        # Optimization history
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig(output_dir / "optimization_history.png", dpi=300, bbox_inches="tight")
        logger.info("✓ Saved optimization history plot")

        # Parallel coordinate plot
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig.savefig(output_dir / "parallel_coordinate.png", dpi=300, bbox_inches="tight")
        logger.info("✓ Saved parallel coordinate plot")

        # Parameter importances
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.savefig(output_dir / "param_importances.png", dpi=300, bbox_inches="tight")
        logger.info("✓ Saved parameter importances plot")

    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Recommended next steps:")
    logger.info("  1. Retrain best config with more epochs for final model")
    logger.info("  2. Evaluate on test set with ensemble methods")
    logger.info("  3. Compare against XGBoost baseline")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="GNN Hyperparameter Search with Optuna")

    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials (default: 50)"
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        help="Seasons to include (default: 2016-2025)",
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Epochs per trial (default: 10 for speed)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device for training (default: cpu)",
    )

    parser.add_argument(
        "--output-dir", type=str, default="models/gnn/optuna", help="Output directory for results"
    )

    args = parser.parse_args()

    # Run search
    run_hyperparameter_search(args)


if __name__ == "__main__":
    main()
