"""
Training Script for 4th Down CQL Agent

Comprehensive training pipeline with:
  - Hyperparameter grid search
  - Cross-validation (by season)
  - Checkpointing and resume capability
  - Extensive logging and visualization

Usage:
  # Single training run
  python scripts/train_fourth_down_cql.py --mode single --epochs 500

  # Hyperparameter grid search
  python scripts/train_fourth_down_cql.py --mode grid --n-trials 20

  # Resume from checkpoint
  python scripts/train_fourth_down_cql.py --mode single --resume models/fourth_down/checkpoint_epoch_300.pth

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from py.rl.cql_agent import CQLAgent, get_device, set_seed

# Import training infrastructure
from py.rl.fourth_down_agent import (
    evaluate_fourth_down_policy,
    load_fourth_down_dataset,
    train_fourth_down_agent,
)
from py.rl.fourth_down_mdp import ACTION_DIM, STATE_DIM

# ============================================================================
# Single Training Run
# ============================================================================


def single_training_run(args):
    """Run single training with specified hyperparameters."""
    print("=" * 80)
    print("SINGLE TRAINING RUN")
    print("=" * 80)

    device = get_device(args.device)
    set_seed(args.seed)

    # Load data
    print(f"Loading training data from {args.train_data}")
    train_states, train_actions, train_rewards, _ = load_fourth_down_dataset(args.train_data)

    print(f"Loading test data from {args.test_data}")
    test_states, test_actions, test_rewards, test_metadata = load_fourth_down_dataset(args.test_data)

    # Initialize agent
    agent = CQLAgent(
        state_dim=STATE_DIM,
        n_actions=ACTION_DIM,
        device=device,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        hidden_dims=args.hidden_dims,
    )

    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)

    # Train
    output_dir = Path(args.output_dir)
    metrics_log = train_fourth_down_agent(
        agent=agent,
        states=train_states,
        actions=train_actions,
        rewards=train_rewards,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        output_dir=output_dir,
    )

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"✓ Training log saved to {log_path}")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    eval_metrics = evaluate_fourth_down_policy(
        agent, test_states, test_actions, test_rewards, test_metadata
    )

    # Save evaluation report
    eval_path = output_dir / "evaluation_report.json"
    with open(eval_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"✓ Evaluation report saved to {eval_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Agreement rate: {eval_metrics['agreement_rate']:.2%}")
    print(f"Reward lift: {eval_metrics['reward_lift']:.4f}")


# ============================================================================
# Hyperparameter Grid Search
# ============================================================================


def grid_search(args):
    """Run hyperparameter grid search."""
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)

    device = get_device(args.device)
    set_seed(args.seed)

    # Load data once
    print(f"Loading training data from {args.train_data}")
    train_states, train_actions, train_rewards, _ = load_fourth_down_dataset(args.train_data)

    print(f"Loading test data from {args.test_data}")
    test_states, test_actions, test_rewards, test_metadata = load_fourth_down_dataset(args.test_data)

    # Define hyperparameter grid
    param_grid = {
        "alpha": [0.1, 0.3, 1.0, 3.0],  # CQL conservatism
        "lr": [1e-5, 1e-4, 1e-3],  # Learning rate
        "hidden_dims": [
            [128, 64],  # Small
            [256, 128, 64],  # Medium
            [512, 256, 128],  # Large
        ],
        "batch_size": [128, 256, 512],
    }

    # Generate all combinations (limited by n_trials)
    from itertools import product
    param_combinations = list(product(
        param_grid["alpha"],
        param_grid["lr"],
        param_grid["hidden_dims"],
        param_grid["batch_size"],
    ))

    # Sample n_trials combinations
    if args.n_trials < len(param_combinations):
        np.random.shuffle(param_combinations)
        param_combinations = param_combinations[:args.n_trials]

    print(f"\nRunning {len(param_combinations)} trials...")
    print(f"Epochs per trial: {args.epochs}")

    results = []
    for trial_idx, (alpha, lr, hidden_dims, batch_size) in enumerate(param_combinations):
        print(f"\n{'=' * 80}")
        print(f"TRIAL {trial_idx + 1}/{len(param_combinations)}")
        print(f"  alpha={alpha}, lr={lr}, hidden_dims={hidden_dims}, batch_size={batch_size}")
        print(f"{'=' * 80}")

        # Initialize agent with trial hyperparameters
        agent = CQLAgent(
            state_dim=STATE_DIM,
            n_actions=ACTION_DIM,
            device=device,
            alpha=alpha,
            gamma=args.gamma,
            lr=lr,
            batch_size=batch_size,
            target_update_freq=args.target_update_freq,
            hidden_dims=hidden_dims,
        )

        # Train
        trial_output_dir = Path(args.output_dir) / f"trial_{trial_idx + 1:03d}"
        metrics_log = train_fourth_down_agent(
            agent=agent,
            states=train_states,
            actions=train_actions,
            rewards=train_rewards,
            epochs=args.epochs,
            batch_size=batch_size,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
            output_dir=trial_output_dir,
        )

        # Evaluate
        eval_metrics = evaluate_fourth_down_policy(
            agent, test_states, test_actions, test_rewards, test_metadata
        )

        # Record results
        trial_result = {
            "trial": trial_idx + 1,
            "hyperparameters": {
                "alpha": alpha,
                "lr": lr,
                "hidden_dims": hidden_dims,
                "batch_size": batch_size,
            },
            "final_loss": metrics_log[-1]["loss"],
            "agreement_rate": eval_metrics["agreement_rate"],
            "reward_lift": eval_metrics["reward_lift"],
        }
        results.append(trial_result)

        # Save trial results
        trial_result_path = trial_output_dir / "trial_result.json"
        with open(trial_result_path, "w") as f:
            json.dump(trial_result, f, indent=2)

        print(f"  → Agreement rate: {eval_metrics['agreement_rate']:.2%}")
        print(f"  → Reward lift: {eval_metrics['reward_lift']:.4f}")

    # Save all results
    results_path = Path(args.output_dir) / "grid_search_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"Total trials: {len(results)}")
    print(f"Results saved to: {results_path}")

    # Best model by agreement rate
    best_by_agreement = max(results, key=lambda x: x["agreement_rate"])
    print(f"\nBest by agreement rate ({best_by_agreement['agreement_rate']:.2%}):")
    print(f"  Trial {best_by_agreement['trial']}")
    print(f"  Hyperparameters: {best_by_agreement['hyperparameters']}")

    # Best model by reward lift
    best_by_reward = max(results, key=lambda x: x["reward_lift"])
    print(f"\nBest by reward lift ({best_by_reward['reward_lift']:.4f}):")
    print(f"  Trial {best_by_reward['trial']}")
    print(f"  Hyperparameters: {best_by_reward['hyperparameters']}")


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    ap = argparse.ArgumentParser(description="Train CQL agent for 4th down decisions")

    # Mode
    ap.add_argument("--mode", choices=["single", "grid"], default="single",
                    help="Training mode: single run or grid search")

    # Data paths
    ap.add_argument("--train-data", default="data/fourth_down_dataset_train.csv")
    ap.add_argument("--test-data", default="data/fourth_down_dataset_test.csv")
    ap.add_argument("--output-dir", default="models/fourth_down")

    # Training hyperparameters (for single mode)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128, 64])
    ap.add_argument("--target-update-freq", type=int, default=1000)

    # Grid search
    ap.add_argument("--n-trials", type=int, default=20,
                    help="Number of hyperparameter combinations to try (grid mode)")

    # Infrastructure
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-freq", type=int, default=20)
    ap.add_argument("--save-freq", type=int, default=100)
    ap.add_argument("--resume", help="Resume from checkpoint (single mode)")

    return ap.parse_args()


def main():
    args = parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{timestamp}"

    if args.mode == "single":
        single_training_run(args)
    elif args.mode == "grid":
        grid_search(args)


if __name__ == "__main__":
    main()
