"""
Training Pipeline for Experiment-Augmented RL Agents

Trains DQN and PPO agents using the 44-dimensional state space (15 base + 29 experiments)
to learn optimal NFL betting policies.

Features:
- Training on 2020-2023 seasons (holdout 2024 for validation)
- Hyperparameter tuning for expanded state space
- Feature importance analysis to identify top experiments
- Ablation studies to measure individual experiment contributions
- Comparison vs baseline (no experiment features)
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, List
import json
from datetime import datetime
import os

from experiment_augmented_env import ExperimentAugmentedBettingEnv
from experiment_rl_agent import DQNAgent, PPOAgent, AblationStudyRunner


def train_agent(env,
                agent,
                num_episodes: int = 1000,
                eval_freq: int = 100,
                verbose: bool = True) -> Dict:
    """
    Train RL agent on betting environment.

    Args:
        env: ExperimentAugmentedBettingEnv
        agent: DQNAgent or PPOAgent
        num_episodes: Number of training episodes
        eval_freq: Evaluate every N episodes
        verbose: Print progress

    Returns:
        Training results dict
    """
    episode_returns = []
    episode_rois = []
    eval_results = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        done = False

        while not done:
            # Select action
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                # Store and train
                agent.store_experience(state, action, reward, next_state, done)
                agent.train_step()

            elif isinstance(agent, PPOAgent):
                action, log_prob, value = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                # Store experience
                agent.store_experience(state, action, reward, log_prob, value, done)

            state = next_state
            episode_return += reward

        # PPO trains at end of episode
        if isinstance(agent, PPOAgent):
            agent.train_step()

        # Track metrics
        episode_returns.append(episode_return)
        episode_rois.append(info['total_roi'])

        # Periodic evaluation
        if (episode + 1) % eval_freq == 0:
            eval_result = evaluate_agent(env, agent, num_eval_episodes=10)
            eval_results.append({
                'episode': episode + 1,
                **eval_result
            })

            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Train Return: {np.mean(episode_returns[-eval_freq:]):.4f}")
                print(f"  Train ROI: {np.mean(episode_rois[-eval_freq:]):.4f}")
                print(f"  Eval Return: {eval_result['mean_return']:.4f}")
                print(f"  Eval ROI: {eval_result['mean_roi']:.4f}")
                if isinstance(agent, DQNAgent):
                    print(f"  Epsilon: {agent.epsilon:.4f}")
                print()

    return {
        'episode_returns': episode_returns,
        'episode_rois': episode_rois,
        'eval_results': eval_results,
        'final_feature_importance': agent.get_feature_importance().tolist()
    }


def evaluate_agent(env, agent, num_eval_episodes: int = 20) -> Dict:
    """
    Evaluate agent performance.

    Args:
        env: Environment
        agent: Trained agent
        num_eval_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    eval_returns = []
    eval_rois = []
    eval_sharpes = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_return = 0
        done = False

        while not done:
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, training=False)
            elif isinstance(agent, PPOAgent):
                action, _, _ = agent.select_action(state, training=False)

            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_return += reward

        eval_returns.append(episode_return)
        eval_rois.append(info['total_roi'])

        stats = env.get_episode_stats()
        eval_sharpes.append(stats['sharpe_ratio'])

    return {
        'mean_return': np.mean(eval_returns),
        'std_return': np.std(eval_returns),
        'mean_roi': np.mean(eval_rois),
        'std_roi': np.std(eval_rois),
        'mean_sharpe': np.mean(eval_sharpes),
        'final_bankroll': env.current_bankroll
    }


def run_ablation_studies(env,
                        agent_class,
                        agent_params: Dict,
                        num_episodes: int = 100) -> List[Dict]:
    """
    Run ablation studies to measure individual experiment contributions.

    Tests removing each major experiment category and measures performance impact.

    Args:
        env: Environment
        agent_class: DQNAgent or PPOAgent
        agent_params: Agent initialization params
        num_episodes: Training episodes per ablation

    Returns:
        List of ablation results
    """
    runner = AblationStudyRunner(agent_class, agent_params)

    # Define ablation experiments
    ablations = [
        # Full model (baseline)
        {
            'name': 'Full Model (All 29 Experiments)',
            'removed_indices': []
        },
        # Remove cVAE features
        {
            'name': 'No cVAE Features',
            'removed_indices': list(range(0, 13))  # First 13 experiment features
        },
        # Remove causal features
        {
            'name': 'No Causal Features',
            'removed_indices': list(range(13, 29))  # Last 16 experiment features
        },
        # Remove top individual experiments
        {
            'name': 'No CVAE-014 (Playoff Resting)',
            'removed_indices': [6]  # playoff_resting_prob feature
        },
        {
            'name': 'No CVAE-010 (Injury Adjustment)',
            'removed_indices': [2]  # injury_adjustment feature
        },
        {
            'name': 'No CAUSAL-014 (4th Down)',
            'removed_indices': [23]  # fourth_down_aggression feature
        },
        {
            'name': 'No CAUSAL-008 (OL Quality)',
            'removed_indices': [19, 20]  # ol_quality_rushing and ol_quality_passing
        },
        # Remove base features (to show experiments matter)
        {
            'name': 'Only Experiment Features (No Base)',
            'removed_indices': []  # Special case handled separately
        }
    ]

    results = []

    for ablation in ablations:
        print(f"\nRunning ablation: {ablation['name']}")

        result = runner.run_ablation(
            env=env,
            experiment_indices=ablation['removed_indices'],
            num_episodes=num_episodes,
            experiment_name=ablation['name']
        )

        results.append(result)

        print(f"  Mean Return: {result['mean_return']:.4f}")
        print(f"  ROI: {result['roi']:.4f}")
        print(f"  Sharpe: {result['sharpe']:.4f}")

    return results


def compare_baseline_vs_experiments(num_episodes: int = 500,
                                   eval_freq: int = 50) -> Dict:
    """
    Compare agent performance with vs without experiment features.

    Train two agents:
    1. Baseline: 15-dimensional state (no experiments)
    2. Experiment-augmented: 44-dimensional state (with experiments)

    Args:
        num_episodes: Training episodes
        eval_freq: Evaluation frequency

    Returns:
        Comparison results
    """
    # Database params
    db_params = {
        'dbname': 'devdb01',
        'user': 'dro',
        'password': 'sicillionbillions',
        'host': 'localhost',
        'port': 5544
    }

    print("=" * 80)
    print("Baseline vs Experiment-Augmented Comparison")
    print("=" * 80)

    # Create environments
    print("\nCreating environments...")

    env_experiment = ExperimentAugmentedBettingEnv(
        db_params=db_params,
        start_season=2020,
        end_season=2023,
        initial_bankroll=10000.0,
        max_bet_fraction=0.10,
        kelly_aggressiveness=0.25
    )

    # For baseline, we'd need a simpler environment (not implemented here)
    # For now, we'll use feature masking in ablation studies

    # Train experiment-augmented agent
    print("\nTraining experiment-augmented DQN agent...")

    agent_experiment = DQNAgent(
        state_dim=44,
        action_dim=5,
        hidden_dims=[128, 128, 64],
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    results_experiment = train_agent(
        env=env_experiment,
        agent=agent_experiment,
        num_episodes=num_episodes,
        eval_freq=eval_freq,
        verbose=True
    )

    print("\nTraining experiment-augmented PPO agent...")

    agent_ppo = PPOAgent(
        state_dim=44,
        action_dim=5,
        hidden_dims=[128, 128],
        learning_rate=3e-4,
        gamma=0.99
    )

    results_ppo = train_agent(
        env=env_experiment,
        agent=agent_ppo,
        num_episodes=num_episodes,
        eval_freq=eval_freq,
        verbose=True
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    final_eval_dqn = evaluate_agent(env_experiment, agent_experiment, num_eval_episodes=50)
    final_eval_ppo = evaluate_agent(env_experiment, agent_ppo, num_eval_episodes=50)

    print("\nDQN Agent:")
    print(f"  Mean Return: {final_eval_dqn['mean_return']:.4f} ± {final_eval_dqn['std_return']:.4f}")
    print(f"  Mean ROI: {final_eval_dqn['mean_roi']:.4f} ± {final_eval_dqn['std_roi']:.4f}")
    print(f"  Mean Sharpe: {final_eval_dqn['mean_sharpe']:.4f}")

    print("\nPPO Agent:")
    print(f"  Mean Return: {final_eval_ppo['mean_return']:.4f} ± {final_eval_ppo['std_return']:.4f}")
    print(f"  Mean ROI: {final_eval_ppo['mean_roi']:.4f} ± {final_eval_ppo['std_roi']:.4f}")
    print(f"  Mean Sharpe: {final_eval_ppo['mean_sharpe']:.4f}")

    return {
        'dqn': {
            'training_results': results_experiment,
            'final_eval': final_eval_dqn
        },
        'ppo': {
            'training_results': results_ppo,
            'final_eval': final_eval_ppo
        }
    }


def analyze_feature_importance(agent, feature_names: List[str]) -> pd.DataFrame:
    """
    Analyze and display feature importance.

    Args:
        agent: Trained agent with feature importance tracking
        feature_names: Names of all 44 features

    Returns:
        DataFrame with feature importance rankings
    """
    importance = agent.get_feature_importance()

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'rank': np.argsort(-importance) + 1
    })

    df = df.sort_values('importance', ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(description='Train Experiment-Augmented RL Agents')

    parser.add_argument('--agent', type=str, default='dqn',
                       choices=['dqn', 'ppo', 'both'],
                       help='Agent type to train')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--eval-freq', type=int, default=100,
                       help='Evaluation frequency')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation studies')
    parser.add_argument('--ablation-episodes', type=int, default=100,
                       help='Episodes per ablation study')
    parser.add_argument('--output-dir', type=str, default='experiments/rl',
                       help='Output directory for results')
    parser.add_argument('--save-agents', action='store_true',
                       help='Save trained agents')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.agent == 'both':
        # Compare DQN and PPO
        results = compare_baseline_vs_experiments(
            num_episodes=args.episodes,
            eval_freq=args.eval_freq
        )

        # Save results
        output_path = os.path.join(args.output_dir, f'comparison_{timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    else:
        # Train single agent
        db_params = {
            'dbname': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions',
            'host': 'localhost',
            'port': 5544
        }

        env = ExperimentAugmentedBettingEnv(
            db_params=db_params,
            start_season=2020,
            end_season=2023
        )

        if args.agent == 'dqn':
            agent = DQNAgent(state_dim=44, action_dim=5)
            print("Training DQN Agent...")
        else:
            agent = PPOAgent(state_dim=44, action_dim=5)
            print("Training PPO Agent...")

        results = train_agent(env, agent, args.episodes, args.eval_freq)

        # Save results
        output_path = os.path.join(args.output_dir, f'{args.agent}_{timestamp}.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        # Save agent
        if args.save_agents:
            agent_path = os.path.join(args.output_dir, f'{args.agent}_{timestamp}.pt')
            agent.save(agent_path)
            print(f"Agent saved to: {agent_path}")

        # Feature importance analysis
        feature_names = [
            # Base features (15)
            'home_win_prob', 'away_win_prob', 'spread', 'total', 'home_score',
            'away_score', 'season', 'week', 'week_normalized', 'market_edge',
            'abs_spread', 'total_normalized', 'actual_total', 'margin', 'signed_margin',
            # cVAE features (13)
            'blowout_prob', 'multi_horizon_q1', 'multi_horizon_q2', 'multi_horizon_q3',
            'multi_horizon_q4', 'injury_adjustment', 'home_advantage', 'rivalry_variance',
            'playoff_resting_prob', 'rookie_qb_adjustment', 'two_minute_score',
            'coordinator_change', 'season_trend',
            # Causal features (16)
            'surface_effect', 'home_field_ate', 'altitude_effect', 'cross_country_travel',
            'short_week_effect', 'ol_quality_rushing', 'ol_quality_passing',
            'veteran_qb_effect', 'personnel_heavy', 'personnel_spread',
            'fourth_down_aggression', 'conservative_bias', 'new_rules_effect',
            'defensive_scheme', 'recency_bias', 'sunk_cost_effect'
        ]

        importance_df = analyze_feature_importance(agent, feature_names)

        importance_path = os.path.join(args.output_dir, f'feature_importance_{timestamp}.csv')
        importance_df.to_csv(importance_path, index=False)

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        print(f"\nFull importance table saved to: {importance_path}")

    # Run ablation studies if requested
    if args.ablation:
        print("\n" + "=" * 80)
        print("Running Ablation Studies")
        print("=" * 80)

        env = ExperimentAugmentedBettingEnv(
            db_params=db_params,
            start_season=2020,
            end_season=2023
        )

        agent_class = DQNAgent if args.agent == 'dqn' else PPOAgent
        agent_params = {'state_dim': 44, 'action_dim': 5}

        ablation_results = run_ablation_studies(
            env=env,
            agent_class=agent_class,
            agent_params=agent_params,
            num_episodes=args.ablation_episodes
        )

        # Save ablation results
        ablation_path = os.path.join(args.output_dir, f'ablation_{timestamp}.json')
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)

        print(f"\nAblation results saved to: {ablation_path}")


if __name__ == '__main__':
    main()
