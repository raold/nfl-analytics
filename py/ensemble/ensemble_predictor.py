#!/usr/bin/env python3
"""
Ensemble Uncertainty Filtering for NFL Betting (Task 6).

Combines three models with uncertainty quantification and selective betting:
1. XGBoost v2 (Brier 0.1715, AUC 0.823) - probability calibration
2. CQL Config 4 (reward 0.0381, match 67.7%) - action selection
3. IQL Baseline (reward 0.0375, aggressive) - edge identification

Voting Strategies:
- Unanimous: Only bet when all 3 models agree on action
- Majority: Bet when 2+ models agree
- Weighted: XGBoost 50%, CQL 30%, IQL 20%

Uncertainty Filtering:
- XGBoost: prediction entropy (low = confident)
- CQL: Q-value spread (max - min across actions)
- IQL: advantage magnitude |Q(s,a) - V(s)|
- Only bet when uncertainty < threshold

Usage:
    python py/ensemble/ensemble_predictor.py \
        --xgb-model models/xgboost/v2_sweep/xgb_config18_season2024.json \
        --cql-model models/cql/sweep/cql_config4.pth \
        --iql-model models/iql/baseline_model.pth \
        --test-data data/processed/features/asof_team_features_v2.csv \
        --test-season 2024 \
        --strategy unanimous \
        --uncertainty-threshold 0.5 \
        --output results/ensemble_backtest_2024.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# Import RL agents
sys.path.append(str(Path(__file__).parent.parent / "rl"))
from cql_agent import CQLAgent
from iql_agent import IQLAgent

# ============================================================================
# XGBoost Model Loader
# ============================================================================


class XGBoostPredictor:
    """Wrapper for XGBoost v2 model with uncertainty quantification."""

    def __init__(self, model_path: str):
        """Load XGBoost model from JSON file."""
        model_path = Path(model_path)

        # Load model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Load config if available
        self.config = {}
        config_path = model_path.parent / f"{model_path.stem}_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict win probabilities with uncertainty.

        Returns:
            probs: (N,) array of win probabilities
            uncertainties: (N,) array of prediction entropy (higher = more uncertain)
        """
        dmatrix = xgb.DMatrix(features)
        probs = self.model.predict(dmatrix)

        # Uncertainty: binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)
        # Normalized to [0, 1] by dividing by max entropy (log(2))
        eps = 1e-10
        p_clipped = np.clip(probs, eps, 1 - eps)
        uncertainties = -(p_clipped * np.log2(p_clipped) + (1 - p_clipped) * np.log2(1 - p_clipped))

        return probs, uncertainties

    def get_action(self, prob: float, market_prob: float, edge_threshold: float = 0.02) -> int:
        """
        Convert probability to betting action.

        Action space: {0: no-bet, 1: small, 2: medium, 3: large}
        Based on edge magnitude relative to market.
        """
        edge = abs(prob - market_prob)

        if edge < edge_threshold:
            return 0  # no-bet (insufficient edge)
        elif edge < 0.05:
            return 1  # small bet
        elif edge < 0.10:
            return 2  # medium bet
        else:
            return 3  # large bet


# ============================================================================
# CQL Model Loader
# ============================================================================


class CQLPredictor:
    """Wrapper for CQL agent with uncertainty quantification."""

    def __init__(self, model_path: str, state_dim: int = 6, device: str = "cpu"):
        """Load CQL model from checkpoint."""
        self.device = torch.device(device)

        # Initialize agent (config will be overwritten by checkpoint)
        self.agent = CQLAgent(
            state_dim=state_dim,
            n_actions=4,
            device=self.device,
            alpha=0.1,  # Will be loaded from checkpoint
            lr=3e-5,
            hidden_dims=[128, 64],
        )

        # Load checkpoint
        self.agent.load(str(model_path))
        self.agent.q_network.eval()

    def predict(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict actions with uncertainty.

        Returns:
            actions: (N,) array of predicted actions
            uncertainties: (N,) array of Q-value spread (higher = more uncertain)
        """
        actions = []
        uncertainties = []

        with torch.no_grad():
            for state in states:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.agent.q_network(state_t).cpu().numpy()[0]

                # Action: argmax Q
                action = int(q_values.argmax())
                actions.append(action)

                # Uncertainty: Q-value spread (max - min)
                # Normalized to [0, 1] by assuming max spread of 10
                q_spread = (q_values.max() - q_values.min()) / 10.0
                uncertainties.append(q_spread)

        return np.array(actions), np.array(uncertainties)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions (for analysis)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.agent.q_network(state_t).cpu().numpy()[0]
        return q_values


# ============================================================================
# IQL Model Loader
# ============================================================================


class IQLPredictor:
    """Wrapper for IQL agent with uncertainty quantification."""

    def __init__(self, model_path: str, state_dim: int = 6, device: str = "cpu"):
        """Load IQL model from checkpoint."""
        self.device = torch.device(device)

        # Initialize agent (config will be overwritten by checkpoint)
        self.agent = IQLAgent(
            state_dim=state_dim,
            n_actions=4,
            device=self.device,
            expectile=0.9,
            temperature=3.0,
            lr_v=3e-4,
            lr_q=3e-4,
            hidden_dims=[128, 64, 32],
        )

        # Load checkpoint
        self.agent.load(str(model_path))
        self.agent.q_network.eval()
        self.agent.v_network.eval()

    def predict(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict actions with uncertainty.

        Returns:
            actions: (N,) array of predicted actions
            uncertainties: (N,) array of advantage magnitude (higher = more uncertain)
        """
        actions = []
        uncertainties = []

        with torch.no_grad():
            for state in states:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.agent.q_network(state_t).cpu().numpy()[0]
                v_value = self.agent.v_network(state_t).cpu().numpy()[0]

                # Advantages: A(s,a) = Q(s,a) - V(s)
                advantages = q_values - v_value

                # Action: softmax over advantages (implicit policy)
                logits = self.agent.temperature * advantages
                exp_logits = np.exp(logits - logits.max())  # numerical stability
                probs = exp_logits / exp_logits.sum()
                action = int(probs.argmax())
                actions.append(action)

                # Uncertainty: max advantage magnitude
                # Normalized to [0, 1] by assuming max advantage of 5
                max_adv = np.abs(advantages).max() / 5.0
                uncertainties.append(max_adv)

        return np.array(actions), np.array(uncertainties)

    def get_advantages(self, state: np.ndarray) -> np.ndarray:
        """Get advantages for all actions (for analysis)."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.agent.q_network(state_t).cpu().numpy()[0]
            v_value = self.agent.v_network(state_t).cpu().numpy()[0]
            advantages = q_values - v_value
        return advantages


# ============================================================================
# Ensemble Predictor
# ============================================================================


class EnsemblePredictor:
    """
    Ensemble of XGBoost, CQL, and IQL with uncertainty filtering.

    Voting strategies:
    - unanimous: Only bet when all 3 models agree
    - majority: Bet when 2+ models agree
    - weighted: XGBoost 50%, CQL 30%, IQL 20%

    Uncertainty filtering:
    - Only bet when uncertainty < threshold for all models
    """

    def __init__(
        self,
        xgb_model: XGBoostPredictor,
        cql_model: CQLPredictor,
        iql_model: IQLPredictor,
        strategy: str = "unanimous",
        uncertainty_threshold: float = 0.5,
        xgb_weight: float = 0.5,
        cql_weight: float = 0.3,
        iql_weight: float = 0.2,
    ):
        self.xgb = xgb_model
        self.cql = cql_model
        self.iql = iql_model
        self.strategy = strategy
        self.uncertainty_threshold = uncertainty_threshold

        # Weights for weighted voting (should sum to 1.0)
        total = xgb_weight + cql_weight + iql_weight
        self.xgb_weight = xgb_weight / total
        self.cql_weight = cql_weight / total
        self.iql_weight = iql_weight / total

    def predict(
        self,
        xgb_features: pd.DataFrame,
        rl_states: np.ndarray,
        market_probs: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Predict actions with ensemble voting and uncertainty filtering.

        Args:
            xgb_features: DataFrame with XGBoost features
            rl_states: (N, state_dim) array for RL agents
            market_probs: (N,) array of market-implied probabilities

        Returns:
            actions: (N,) array of ensemble actions
            metadata: dict with individual predictions and uncertainties
        """
        n_samples = len(xgb_features)

        # Get predictions from each model
        xgb_probs, xgb_uncertainties = self.xgb.predict(xgb_features)
        cql_actions, cql_uncertainties = self.cql.predict(rl_states)
        iql_actions, iql_uncertainties = self.iql.predict(rl_states)

        # Convert XGBoost probabilities to actions
        xgb_actions = np.array(
            [self.xgb.get_action(xgb_probs[i], market_probs[i]) for i in range(n_samples)]
        )

        # Initialize ensemble actions
        ensemble_actions = np.zeros(n_samples, dtype=int)

        # Apply voting strategy
        for i in range(n_samples):
            # Check uncertainty threshold
            if (
                xgb_uncertainties[i] > self.uncertainty_threshold
                or cql_uncertainties[i] > self.uncertainty_threshold
                or iql_uncertainties[i] > self.uncertainty_threshold
            ):
                # Too uncertain - no bet
                ensemble_actions[i] = 0
                continue

            # Voting
            if self.strategy == "unanimous":
                # All 3 must agree
                if xgb_actions[i] == cql_actions[i] == iql_actions[i]:
                    ensemble_actions[i] = xgb_actions[i]
                else:
                    ensemble_actions[i] = 0  # no-bet

            elif self.strategy == "majority":
                # At least 2 must agree
                votes = [xgb_actions[i], cql_actions[i], iql_actions[i]]
                action_counts = {a: votes.count(a) for a in set(votes)}
                max_count = max(action_counts.values())
                if max_count >= 2:
                    # Find action with most votes
                    ensemble_actions[i] = max(action_counts, key=action_counts.get)
                else:
                    ensemble_actions[i] = 0  # no-bet

            elif self.strategy == "weighted":
                # Weighted vote by confidence (inverse uncertainty)
                # Weight each action by (1 - uncertainty) * model_weight
                action_scores = {}

                # XGBoost vote
                xgb_conf = (1 - xgb_uncertainties[i]) * self.xgb_weight
                action_scores[xgb_actions[i]] = action_scores.get(xgb_actions[i], 0) + xgb_conf

                # CQL vote
                cql_conf = (1 - cql_uncertainties[i]) * self.cql_weight
                action_scores[cql_actions[i]] = action_scores.get(cql_actions[i], 0) + cql_conf

                # IQL vote
                iql_conf = (1 - iql_uncertainties[i]) * self.iql_weight
                action_scores[iql_actions[i]] = action_scores.get(iql_actions[i], 0) + iql_conf

                # Select action with highest score
                ensemble_actions[i] = max(action_scores, key=action_scores.get)

            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # Metadata
        metadata = {
            "xgb_probs": xgb_probs,
            "xgb_actions": xgb_actions,
            "xgb_uncertainties": xgb_uncertainties,
            "cql_actions": cql_actions,
            "cql_uncertainties": cql_uncertainties,
            "iql_actions": iql_actions,
            "iql_uncertainties": iql_uncertainties,
        }

        return ensemble_actions, metadata


# ============================================================================
# Backtesting
# ============================================================================


def backtest_ensemble(
    ensemble: EnsemblePredictor,
    test_df: pd.DataFrame,
    xgb_features: list[str],
    rl_state_cols: list[str],
    bet_sizes: dict[int, float] = None,
) -> dict:
    """
    Backtest ensemble on historical data.

    Args:
        ensemble: EnsemblePredictor instance
        test_df: DataFrame with test data (must have 'home_result', 'spread_close', etc.)
        xgb_features: List of feature names for XGBoost
        rl_state_cols: List of state feature names for RL agents
        bet_sizes: Dict mapping action → bet size (fraction of bankroll)

    Returns:
        Dict with backtest metrics
    """
    if bet_sizes is None:
        # Default bet sizes (Kelly-like)
        bet_sizes = {0: 0.0, 1: 0.01, 2: 0.02, 3: 0.05}

    # Prepare inputs
    xgb_feats = test_df[xgb_features]
    rl_states = test_df[rl_state_cols].to_numpy(dtype=np.float32)

    # Market probabilities (from closing spread)
    # Simplified: convert spread to implied probability using empirical rule
    # P(home win) ≈ 0.5 + (spread / 27)  [rough approximation]
    spreads = test_df["spread_close"].to_numpy()
    market_probs = 0.5 + (spreads / 27.0)
    market_probs = np.clip(market_probs, 0.01, 0.99)

    # Get ensemble predictions
    actions, metadata = ensemble.predict(xgb_feats, rl_states, market_probs)

    # Compute returns for each bet
    home_results = test_df["home_result"].to_numpy()  # 1 if home won, 0 otherwise

    returns = []
    bets = []

    for i in range(len(test_df)):
        action = actions[i]
        if action == 0:
            # No bet
            continue

        # Bet size
        bet_size = bet_sizes[action]

        # Direction: bet home if model prob > market prob
        model_prob = metadata["xgb_probs"][i]
        bet_home = model_prob > market_probs[i]

        # Outcome
        home_won = home_results[i] == 1
        bet_won = (bet_home and home_won) or (not bet_home and not home_won)

        # Return (simplified: assume -110 odds, so win = +0.91, loss = -1.0)
        if bet_won:
            ret = bet_size * 0.91  # win $0.91 per $1 risked
        else:
            ret = -bet_size  # lose $1 per $1 risked

        returns.append(ret)
        bets.append(
            {
                "action": action,
                "bet_size": bet_size,
                "bet_home": bet_home,
                "bet_won": bet_won,
                "return": ret,
                "model_prob": model_prob,
                "market_prob": market_probs[i],
                "edge": model_prob - market_probs[i],
            }
        )

    # Aggregate metrics
    if len(returns) == 0:
        return {
            "n_games": len(test_df),
            "n_bets": 0,
            "bet_rate": 0.0,
            "total_return": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "action_distribution": {},
            "agreement_rates": {},
            "uncertainty_stats": {},
            "avg_edge": 0.0,
        }

    returns = np.array(returns)

    # Betting statistics
    n_games = len(test_df)
    n_bets = len(returns)
    bet_rate = n_bets / n_games
    total_return = returns.sum()
    roi = (total_return / n_bets) * 100  # percent
    win_rate = np.mean([b["bet_won"] for b in bets]) * 100

    # Risk metrics
    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()

    # Action distribution
    action_dist = pd.Series(actions).value_counts(normalize=True).to_dict()

    # Agreement metrics
    xgb_cql_agree = (metadata["xgb_actions"] == metadata["cql_actions"]).mean()
    xgb_iql_agree = (metadata["xgb_actions"] == metadata["iql_actions"]).mean()
    cql_iql_agree = (metadata["cql_actions"] == metadata["iql_actions"]).mean()
    all_agree = (
        (metadata["xgb_actions"] == metadata["cql_actions"])
        & (metadata["cql_actions"] == metadata["iql_actions"])
    ).mean()

    # Uncertainty statistics
    uncertainty_stats = {
        "xgb_mean": metadata["xgb_uncertainties"].mean(),
        "cql_mean": metadata["cql_uncertainties"].mean(),
        "iql_mean": metadata["iql_uncertainties"].mean(),
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
        "action_distribution": action_dist,
        "agreement_rates": {
            "xgb_cql": xgb_cql_agree,
            "xgb_iql": xgb_iql_agree,
            "cql_iql": cql_iql_agree,
            "all_agree": all_agree,
        },
        "uncertainty_stats": uncertainty_stats,
        "avg_edge": np.mean([b["edge"] for b in bets]),
        "bets_detail": bets,
    }


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ensemble uncertainty filtering (Task 6)")

    # Model paths
    ap.add_argument("--xgb-model", required=True, help="XGBoost model JSON path")
    ap.add_argument("--cql-model", required=True, help="CQL model PTH path")
    ap.add_argument("--iql-model", required=True, help="IQL model PTH path")

    # Data
    ap.add_argument("--test-data", required=True, help="Test data CSV")
    ap.add_argument("--test-season", type=int, default=2024, help="Test season")

    # Ensemble config
    ap.add_argument(
        "--strategy",
        choices=["unanimous", "majority", "weighted"],
        default="unanimous",
        help="Voting strategy",
    )
    ap.add_argument(
        "--uncertainty-threshold", type=float, default=0.5, help="Max uncertainty to allow betting"
    )
    ap.add_argument(
        "--xgb-weight", type=float, default=0.5, help="XGBoost weight (weighted voting)"
    )
    ap.add_argument("--cql-weight", type=float, default=0.3, help="CQL weight (weighted voting)")
    ap.add_argument("--iql-weight", type=float, default=0.2, help="IQL weight (weighted voting)")

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
        help="XGBoost feature columns",
    )
    ap.add_argument(
        "--rl-state-cols",
        nargs="+",
        default=["spread_close", "total_close", "epa_gap", "market_prob", "p_hat", "edge"],
        help="RL state feature columns",
    )

    # Output
    ap.add_argument(
        "--output",
        default="results/ensemble_backtest.json",
        help="Output path for backtest results",
    )
    ap.add_argument("--device", default="cpu", help="Device: cpu/cuda/mps")

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"{'='*80}")
    print("Ensemble Uncertainty Filtering (Task 6)")
    print(f"{'='*80}")
    print(f"Strategy: {args.strategy}")
    print(f"Uncertainty threshold: {args.uncertainty_threshold}")
    print(f"Test season: {args.test_season}")

    # Load models
    print("\nLoading models...")
    print(f"  XGBoost: {args.xgb_model}")
    xgb_model = XGBoostPredictor(args.xgb_model)

    print(f"  CQL: {args.cql_model}")
    cql_model = CQLPredictor(args.cql_model, state_dim=len(args.rl_state_cols), device=args.device)

    print(f"  IQL: {args.iql_model}")
    iql_model = IQLPredictor(args.iql_model, state_dim=len(args.rl_state_cols), device=args.device)

    # Create ensemble
    ensemble = EnsemblePredictor(
        xgb_model=xgb_model,
        cql_model=cql_model,
        iql_model=iql_model,
        strategy=args.strategy,
        uncertainty_threshold=args.uncertainty_threshold,
        xgb_weight=args.xgb_weight,
        cql_weight=args.cql_weight,
        iql_weight=args.iql_weight,
    )

    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    df = pd.read_csv(args.test_data)

    # Filter to test season
    test_df = df[df["season"] == args.test_season].copy()
    print(f"  Test season {args.test_season}: {len(test_df)} games")

    # Check required columns
    required_cols = set(args.xgb_features + args.rl_state_cols + ["home_result", "spread_close"])
    missing_cols = required_cols - set(test_df.columns)
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return 1

    # Drop NaN rows
    test_df = test_df.dropna(subset=list(required_cols))
    print(f"  After dropping NaN: {len(test_df)} games")

    # Backtest
    print("\nRunning backtest...")
    results = backtest_ensemble(
        ensemble=ensemble,
        test_df=test_df,
        xgb_features=args.xgb_features,
        rl_state_cols=args.rl_state_cols,
    )

    # Print results
    print(f"\n{'='*80}")
    print("Backtest Results")
    print(f"{'='*80}")
    print(f"Games: {results['n_games']}")
    print(f"Bets placed: {results['n_bets']} ({results['bet_rate']*100:.1f}% of games)")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Total return: {results['total_return']:+.2f} units")
    print(f"ROI: {results['roi']:+.2f}%")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {results['max_drawdown']:.2f} units")
    print(f"Average edge: {results['avg_edge']:+.4f}")

    print("\nAction distribution:")
    for action, pct in sorted(results["action_distribution"].items()):
        action_name = {0: "no-bet", 1: "small", 2: "medium", 3: "large"}.get(
            action, f"action_{action}"
        )
        print(f"  {action_name}: {pct*100:.1f}%")

    print("\nModel agreement rates:")
    for pair, rate in results["agreement_rates"].items():
        print(f"  {pair}: {rate*100:.1f}%")

    print("\nUncertainty statistics:")
    for model, mean_unc in results["uncertainty_stats"].items():
        print(f"  {model}: {mean_unc:.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert bets_detail to serializable format
    results_copy = results.copy()
    results_copy["bets_detail"] = [
        {k: (float(v) if isinstance(v, np.generic) else v) for k, v in bet.items()}
        for bet in results["bets_detail"]
    ]

    with open(output_path, "w") as f:
        json.dump(results_copy, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
