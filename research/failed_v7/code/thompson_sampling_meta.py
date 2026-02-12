#!/usr/bin/env python3
"""
Thompson Sampling Meta-Policy for Model Selection (Task 7).

Instead of fixed voting weights, use Bayesian bandits to dynamically select
which model (XGBoost, CQL, IQL) to use for each game based on empirical performance.

Algorithm:
1. Maintain Beta(α, β) posterior for each model's win rate
2. For each game:
   - Sample win rate θ ~ Beta(α, β) for each model
   - Select model with highest sampled win rate
   - Bet according to selected model's action
   - Update selected model's posterior: win → α+1, loss → β+1

Advantages over Task 6 fixed voting:
- Adapts online (learns which model is best as season progresses)
- Balances exploration (try all models) vs exploitation (use best model)
- Model-specific learning (doesn't average away each model's strengths)

Usage:
    python py/ensemble/thompson_sampling_meta.py \
        --xgb-model models/xgboost/v2_sweep/xgb_config18_season2024.json \
        --cql-model models/cql/sweep/cql_config4.pth \
        --iql-model models/iql/baseline_model.pth \
        --test-data data/processed/features/ensemble_features_2024.csv \
        --test-season 2024 \
        --prior-alpha 1.0 \
        --prior-beta 1.0 \
        --output results/ensemble/thompson_sampling_2024.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import ensemble predictors from Task 6
sys.path.append(str(Path(__file__).parent))
from ensemble_predictor import CQLPredictor, IQLPredictor, XGBoostPredictor


class ThompsonSamplingMeta:
    """
    Thompson Sampling for online model selection.

    Each model is a "bandit arm" with Beta(α, β) posterior over win rate.
    At each decision point, sample from posteriors and select best model.
    """

    def __init__(
        self,
        model_names: list[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        """
        Initialize Thompson Sampling with uniform Beta(1,1) priors.

        Args:
            model_names: List of model identifiers (e.g., ['xgboost', 'cql', 'iql'])
            prior_alpha: Prior pseudo-wins (default 1.0 = uniform prior)
            prior_beta: Prior pseudo-losses (default 1.0 = uniform prior)
        """
        self.model_names = model_names
        self.n_models = len(model_names)

        # Beta(α, β) posteriors for each model
        self.alphas = {name: prior_alpha for name in model_names}
        self.betas = {name: prior_beta for name in model_names}

        # Tracking
        self.selections = {name: 0 for name in model_names}
        self.wins = {name: 0 for name in model_names}
        self.losses = {name: 0 for name in model_names}
        self.history = []

    def select_model(self) -> str:
        """
        Thompson Sampling: sample from each Beta posterior and select best.

        Returns:
            Selected model name
        """
        samples = {}
        for name in self.model_names:
            # Sample win rate from Beta(α, β)
            theta = np.random.beta(self.alphas[name], self.betas[name])
            samples[name] = theta

        # Select model with highest sampled win rate
        selected = max(samples, key=samples.get)
        self.selections[selected] += 1

        return selected

    def update(self, model_name: str, won: bool):
        """
        Update posterior after observing outcome.

        Args:
            model_name: Model that was selected
            won: True if bet won, False if lost
        """
        if won:
            self.alphas[model_name] += 1
            self.wins[model_name] += 1
        else:
            self.betas[model_name] += 1
            self.losses[model_name] += 1

        self.history.append(
            {
                "model": model_name,
                "won": won,
                "alpha": self.alphas[model_name],
                "beta": self.betas[model_name],
            }
        )

    def get_stats(self) -> dict:
        """Get current statistics for all models."""
        stats = {}
        for name in self.model_names:
            alpha = self.alphas[name]
            beta = self.betas[name]
            n = self.selections[name]

            # Posterior mean = α / (α + β)
            mean_win_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

            # 95% credible interval
            low, high = 0.0, 1.0
            if alpha > 1 and beta > 1:
                # Approximate using normal distribution (valid for large α, β)
                mean = alpha / (alpha + beta)
                var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                std = np.sqrt(var)
                low = max(0, mean - 1.96 * std)
                high = min(1, mean + 1.96 * std)

            stats[name] = {
                "selections": n,
                "wins": self.wins[name],
                "losses": self.losses[name],
                "win_rate": self.wins[name] / n if n > 0 else 0.0,
                "posterior_mean": mean_win_rate,
                "credible_interval": [low, high],
                "alpha": alpha,
                "beta": beta,
            }

        return stats


def backtest_thompson_sampling(
    xgb_model: XGBoostPredictor,
    cql_model: CQLPredictor,
    iql_model: IQLPredictor,
    test_df: pd.DataFrame,
    xgb_features: list[str],
    rl_state_cols: list[str],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    bet_sizes: dict[int, float] = None,
) -> dict:
    """
    Backtest Thompson Sampling meta-policy.

    Args:
        xgb_model, cql_model, iql_model: Loaded model predictors
        test_df: Test data
        xgb_features: XGBoost feature names
        rl_state_cols: RL state feature names
        prior_alpha, prior_beta: Beta prior parameters
        bet_sizes: Bet size for each action

    Returns:
        Dict with backtest results
    """
    if bet_sizes is None:
        bet_sizes = {0: 0.0, 1: 0.01, 2: 0.02, 3: 0.05}

    # Initialize Thompson Sampling
    ts = ThompsonSamplingMeta(
        model_names=["xgboost", "cql", "iql"],
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )

    # Prepare features
    test_df[xgb_features].to_numpy()
    rl_states = test_df[rl_state_cols].to_numpy(dtype=np.float32)
    home_results = test_df["home_result"].to_numpy()
    spreads = test_df["spread_close"].to_numpy()

    # Market probabilities
    market_probs = 0.5 + (spreads / 27.0)
    market_probs = np.clip(market_probs, 0.01, 0.99)

    # Pre-compute all model predictions (offline, before Thompson Sampling)
    print("Computing model predictions...")
    xgb_probs, _ = xgb_model.predict(test_df[xgb_features])
    cql_actions, _ = cql_model.predict(rl_states)
    iql_actions, _ = iql_model.predict(rl_states)

    # Convert XGBoost probs to actions
    xgb_actions = np.array(
        [xgb_model.get_action(xgb_probs[i], market_probs[i]) for i in range(len(test_df))]
    )

    # Store predictions
    all_predictions = {
        "xgboost": xgb_actions,
        "cql": cql_actions,
        "iql": iql_actions,
        "xgb_probs": xgb_probs,
    }

    # Thompson Sampling online selection
    print("Running Thompson Sampling...")
    returns = []
    bets = []

    for i in range(len(test_df)):
        # Thompson Sampling: select model
        selected_model = ts.select_model()

        # Get action from selected model
        action = all_predictions[selected_model][i]

        # Skip if no-bet
        if action == 0:
            continue

        # Bet size
        bet_size = bet_sizes[action]

        # Determine bet direction (home or away)
        model_prob = xgb_probs[i]  # Use XGBoost prob for all models (simplification)
        bet_home = model_prob > market_probs[i]

        # Outcome
        home_won = home_results[i] == 1
        bet_won = (bet_home and home_won) or (not bet_home and not home_won)

        # Return (assuming -110 odds)
        if bet_won:
            ret = bet_size * 0.91
        else:
            ret = -bet_size

        returns.append(ret)

        # Update Thompson Sampling posterior
        ts.update(selected_model, bet_won)

        # Record bet
        bets.append(
            {
                "game_idx": i,
                "selected_model": selected_model,
                "action": action,
                "bet_size": bet_size,
                "bet_home": bet_home,
                "bet_won": bet_won,
                "return": ret,
                "model_prob": model_prob,
                "market_prob": market_probs[i],
            }
        )

    # Aggregate metrics
    if len(returns) == 0:
        return {
            "n_games": len(test_df),
            "n_bets": 0,
            "thompson_sampling_stats": ts.get_stats(),
        }

    returns = np.array(returns)

    # Betting statistics
    n_games = len(test_df)
    n_bets = len(returns)
    bet_rate = n_bets / n_games
    total_return = returns.sum()
    roi = (total_return / n_bets) * 100
    win_rate = np.mean([b["bet_won"] for b in bets]) * 100

    # Risk metrics
    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()

    # Model selection distribution
    model_selection_dist = {
        name: ts.selections[name] / n_bets if n_bets > 0 else 0.0 for name in ts.model_names
    }

    return {
        "n_games": n_games,
        "n_bets": n_bets,
        "bet_rate": bet_rate,
        "total_return": total_return,
        "roi": roi,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "thompson_sampling_stats": ts.get_stats(),
        "model_selection_distribution": model_selection_dist,
        "bets_detail": bets,
        "posterior_history": ts.history,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Thompson Sampling meta-policy (Task 7)")

    # Models
    ap.add_argument("--xgb-model", required=True, help="XGBoost model JSON")
    ap.add_argument("--cql-model", required=True, help="CQL model PTH")
    ap.add_argument("--iql-model", required=True, help="IQL model PTH")

    # Data
    ap.add_argument("--test-data", required=True, help="Test data CSV")
    ap.add_argument("--test-season", type=int, default=2024, help="Test season")

    # Thompson Sampling priors
    ap.add_argument("--prior-alpha", type=float, default=1.0, help="Beta prior pseudo-wins")
    ap.add_argument("--prior-beta", type=float, default=1.0, help="Beta prior pseudo-losses")

    # Features
    ap.add_argument(
        "--xgb-features",
        nargs="+",
        default=[
            "prior_epa_mean_diff",
            "epa_pp_last3_diff",
            "season_win_pct_diff",
            "win_pct_last5_diff",
            "prior_margin_avg_diff",
            "points_for_last3_diff",
            "points_against_last3_diff",
            "rest_diff",
            "week",
            "fourth_downs_diff",
            "fourth_down_epa_diff",
        ],
    )
    ap.add_argument(
        "--rl-state-cols",
        nargs="+",
        default=["spread_close", "total_close", "epa_gap", "market_prob", "p_hat", "edge"],
    )

    # Output
    ap.add_argument("--output", default="results/ensemble/thompson_sampling_2024.json")
    ap.add_argument("--device", default="cpu", help="Device for RL models")

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"{'='*80}")
    print("Thompson Sampling Meta-Policy (Task 7)")
    print(f"{'='*80}")
    print(f"Prior: Beta({args.prior_alpha}, {args.prior_beta})")
    print(f"Test season: {args.test_season}")

    # Load models
    print("\nLoading models...")
    print(f"  XGBoost: {args.xgb_model}")
    xgb_model = XGBoostPredictor(args.xgb_model)

    print(f"  CQL: {args.cql_model}")
    cql_model = CQLPredictor(args.cql_model, state_dim=len(args.rl_state_cols), device=args.device)

    print(f"  IQL: {args.iql_model}")
    iql_model = IQLPredictor(args.iql_model, state_dim=len(args.rl_state_cols), device=args.device)

    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    df = pd.read_csv(args.test_data)
    test_df = df[df["season"] == args.test_season].copy()
    print(f"  Test season {args.test_season}: {len(test_df)} games")

    # Check required columns
    required_cols = set(args.xgb_features + args.rl_state_cols + ["home_result", "spread_close"])
    missing_cols = required_cols - set(test_df.columns)
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return 1

    # Drop NaN
    test_df = test_df.dropna(subset=list(required_cols))
    print(f"  After dropping NaN: {len(test_df)} games")

    # Run Thompson Sampling backtest
    print("\nRunning Thompson Sampling backtest...")
    results = backtest_thompson_sampling(
        xgb_model=xgb_model,
        cql_model=cql_model,
        iql_model=iql_model,
        test_df=test_df,
        xgb_features=args.xgb_features,
        rl_state_cols=args.rl_state_cols,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
    )

    # Print results
    print(f"\n{'='*80}")
    print("Thompson Sampling Results")
    print(f"{'='*80}")
    print(f"Games: {results['n_games']}")
    print(f"Bets placed: {results['n_bets']} ({results['bet_rate']*100:.1f}% of games)")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Total return: {results['total_return']:+.2f} units")
    print(f"ROI: {results['roi']:+.2f}%")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {results['max_drawdown']:.2f} units")

    print("\nModel selection distribution:")
    for model, pct in sorted(results["model_selection_distribution"].items()):
        print(f"  {model}: {pct*100:.1f}%")

    print("\nFinal Thompson Sampling statistics:")
    for model, stats in results["thompson_sampling_stats"].items():
        print(f"\n  {model}:")
        print(f"    Selections: {stats['selections']}")
        print(f"    Win rate: {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['selections']})")
        print(f"    Posterior mean: {stats['posterior_mean']:.3f}")
        print(
            f"    95% CI: [{stats['credible_interval'][0]:.3f}, {stats['credible_interval'][1]:.3f}]"
        )
        print(f"    Beta({stats['alpha']:.1f}, {stats['beta']:.1f})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    results_copy = {
        k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
        for k, v in results.items()
        if k not in ["bets_detail", "posterior_history"]
    }

    # Convert bet details
    if "bets_detail" in results:
        results_copy["bets_detail"] = [
            {
                k: (float(v) if isinstance(v, (np.integer, np.floating, np.bool_)) else v)
                for k, v in bet.items()
            }
            for bet in results["bets_detail"]
        ]

    # Convert posterior history
    if "posterior_history" in results:
        results_copy["posterior_history"] = [
            {
                k: (float(v) if isinstance(v, (np.integer, np.floating, np.bool_)) else v)
                for k, v in entry.items()
            }
            for entry in results["posterior_history"]
        ]

    with open(output_path, "w") as f:
        json.dump(results_copy, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
