#!/usr/bin/env python3
"""
CQL Hyperparameter Sweep for NFL Betting Policy Optimization.

Systematically sweeps CQL hyperparameters to improve win rate from 51% to 52.5%+.

Grid Search:
- alpha (CQL penalty): [0.1, 0.3, 1.0, 3.0, 10.0]
- lr (learning rate): [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
- hidden_dims: [[128, 64], [128, 64, 32], [256, 128, 64]]

Total configurations: 5 × 5 × 3 = 75 configs

Evaluation metrics:
- Match rate (policy accuracy)
- Estimated reward
- Q-value conservatism
- Action distribution

Usage:
    python py/rl/cql_sweep.py \
        --dataset data/rl_logged_2006_2024.csv \
        --output-dir models/cql/sweep \
        --epochs 500 \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from cql_agent import CQLAgent, evaluate_policy, load_dataset, populate_replay_buffer, train_cql


def create_param_grid() -> List[Dict]:
    """Create hyperparameter grid for CQL sweep."""
    alphas = [0.1, 0.3, 1.0, 3.0, 10.0]
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    hidden_dims_options = [
        [128, 64],
        [128, 64, 32],
        [256, 128, 64],
    ]

    grid = []
    config_id = 1
    for alpha in alphas:
        for lr in learning_rates:
            for hidden_dims in hidden_dims_options:
                grid.append({
                    'config_id': config_id,
                    'alpha': alpha,
                    'lr': lr,
                    'hidden_dims': hidden_dims,
                })
                config_id += 1

    return grid


def run_single_config(
    config: Dict,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    output_dir: Path,
) -> Dict:
    """Train and evaluate a single CQL configuration."""

    state_dim = states.shape[1]
    n_actions = 4

    # Initialize agent with config
    agent = CQLAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device,
        alpha=config['alpha'],
        lr=config['lr'],
        batch_size=batch_size,
        hidden_dims=config['hidden_dims'],
    )

    # Train
    metrics_log = train_cql(
        agent=agent,
        states=states,
        actions=actions,
        rewards=rewards,
        epochs=epochs,
        batch_size=batch_size,
        log_freq=50,
    )

    # Evaluate
    eval_metrics = evaluate_policy(agent, states, actions, rewards)

    # Save model
    model_path = output_dir / f"cql_config{config['config_id']}.pth"
    agent.save(str(model_path))

    # Combine results
    result = {
        **config,
        **eval_metrics,
        'final_loss': metrics_log[-1]['loss'],
        'final_td_loss': metrics_log[-1]['td_loss'],
        'final_cql_loss': metrics_log[-1]['cql_loss'],
    }

    return result


def run_sweep(
    dataset_path: str,
    output_dir: Path,
    epochs: int = 500,
    batch_size: int = 128,
    device_arg: str = 'auto',
    state_cols: List[str] = None,
) -> pd.DataFrame:
    """Run full CQL hyperparameter sweep."""

    if state_cols is None:
        state_cols = ['spread_close', 'total_close', 'epa_gap', 'market_prob', 'p_hat', 'edge']

    # Setup device
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    states, actions, rewards = load_dataset(dataset_path, state_cols)

    print(f"Dataset: {len(states)} samples, state_dim={states.shape[1]}")
    print(f"Mean reward: {rewards.mean():.4f}")

    # Create param grid
    param_grid = create_param_grid()
    print(f"\nHyperparameter grid: {len(param_grid)} configurations")
    print(f"Total training runs: {len(param_grid)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run sweep
    results = []

    for i, config in enumerate(param_grid):
        print(f"\n{'='*80}")
        print(f"Config {config['config_id']}/{len(param_grid)}")
        print(f"  alpha={config['alpha']}, lr={config['lr']}, hidden={config['hidden_dims']}")
        print(f"{'='*80}")

        try:
            result = run_single_config(
                config=config,
                states=states,
                actions=actions,
                rewards=rewards,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                output_dir=output_dir,
            )
            results.append(result)

            print(f"\nResults:")
            print(f"  Match rate: {result['match_rate']:.3f}")
            print(f"  Estimated reward: {result['estimated_policy_reward']:.4f}")
            print(f"  Avg Q-value: {result['avg_q_value']:.4f}")
            print(f"  Action dist: {result['action_distribution']}")

        except Exception as e:
            print(f"ERROR in config {config['config_id']}: {e}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_path = output_dir / "cql_sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n{'='*80}")
    print(f"Sweep complete! Results saved to {results_path}")
    print(f"{'='*80}")

    # Summary statistics
    print("\n=== Top 5 Configurations (by estimated reward) ===")
    top_5 = results_df.nlargest(5, 'estimated_policy_reward')
    for _, row in top_5.iterrows():
        print(f"Config {int(row['config_id'])}: "
              f"reward={row['estimated_policy_reward']:.4f}, "
              f"match_rate={row['match_rate']:.3f}, "
              f"alpha={row['alpha']}, lr={row['lr']}")

    print("\n=== Top 5 Configurations (by match rate) ===")
    top_5_match = results_df.nlargest(5, 'match_rate')
    for _, row in top_5_match.iterrows():
        print(f"Config {int(row['config_id'])}: "
              f"match_rate={row['match_rate']:.3f}, "
              f"reward={row['estimated_policy_reward']:.4f}, "
              f"alpha={row['alpha']}, lr={row['lr']}")

    # Save summary
    summary = {
        'total_configs': len(param_grid),
        'successful_configs': len(results),
        'best_config_by_reward': top_5.iloc[0].to_dict() if len(top_5) > 0 else None,
        'best_config_by_match': top_5_match.iloc[0].to_dict() if len(top_5_match) > 0 else None,
    }

    summary_path = output_dir / "cql_sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to {summary_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='CQL Hyperparameter Sweep')
    parser.add_argument('--dataset', required=True, help='Path to offline RL dataset CSV')
    parser.add_argument('--output-dir', type=Path, default=Path('models/cql/sweep'),
                        help='Output directory for sweep results')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs per config')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', default='auto', help='Device: auto/cpu/cuda/mps')
    parser.add_argument('--state-cols', nargs='+',
                        default=['spread_close', 'total_close', 'epa_gap', 'market_prob', 'p_hat', 'edge'],
                        help='State feature columns')

    args = parser.parse_args()

    results_df = run_sweep(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_arg=args.device,
        state_cols=args.state_cols,
    )

    print(f"\n[DONE] CQL sweep complete. Trained {len(results_df)} configurations.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
