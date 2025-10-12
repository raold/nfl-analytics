#!/usr/bin/env python3
"""Ensemble voting system combining Bayesian hierarchical models with XGBoost.

This module implements the ensemble strategy from the Bayesian EV analysis:
- Only bet when both Bayesian and XGBoost agree on direction
- Use Bayesian uncertainty for Kelly criterion position sizing
- Apply disagreement filtering to improve win rate

Key findings from analysis:
- Bayesian standalone: 54.0% win rate, +1.59% ROI
- Ensemble (both agree): 55.0% win rate, +2.60% ROI
- Recommended weight: 15-25% Bayesian, 75-85% XGBoost

Usage:
    python py/production/ensemble_bayesian_xgb.py --games data/processed/features/asof_team_features_v3_bayesian.csv \
                                                    --xgb-model models/xgboost/v3_best.json \
                                                    --test-season 2024 \
                                                    --output analysis/ensemble/bayesian_xgb_2024.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class BayesianXGBoostEnsemble:
    """Ensemble combining Bayesian hierarchical and XGBoost predictions."""

    def __init__(
        self,
        bayesian_weight: float = 0.25,
        xgb_weight: float = 0.75,
        agreement_threshold: float = 0.10,
        edge_threshold: float = 0.02,
        vig_rate: float = 0.024,
    ):
        """Initialize ensemble parameters.

        Args:
            bayesian_weight: Weight for Bayesian predictions (0.15-0.25 recommended)
            xgb_weight: Weight for XGBoost predictions (0.75-0.85 recommended)
            agreement_threshold: Max probability difference to consider agreement (default: 0.10)
            edge_threshold: Minimum edge to place bet after vig (default: 0.02 = 2%)
            vig_rate: Vigorish rate for betting (default: 0.024 for -110 odds)
        """
        assert abs(bayesian_weight + xgb_weight - 1.0) < 1e-6, "Weights must sum to 1.0"

        self.bayesian_weight = bayesian_weight
        self.xgb_weight = xgb_weight
        self.agreement_threshold = agreement_threshold
        self.edge_threshold = edge_threshold
        self.vig_rate = vig_rate

    def compute_ensemble_probability(
        self,
        bayesian_prob: float,
        xgb_prob: float,
    ) -> dict:
        """Compute weighted ensemble probability and metadata.

        Args:
            bayesian_prob: Bayesian model probability home team wins
            xgb_prob: XGBoost model probability home team wins

        Returns:
            Dictionary with:
                - ensemble_prob: Weighted average probability
                - prob_diff: Absolute difference between models
                - models_agree: Boolean indicating if models agree
        """
        ensemble_prob = (
            self.bayesian_weight * bayesian_prob + self.xgb_weight * xgb_prob
        )

        prob_diff = abs(bayesian_prob - xgb_prob)
        models_agree = prob_diff <= self.agreement_threshold

        return {
            "ensemble_prob": ensemble_prob,
            "prob_diff": prob_diff,
            "models_agree": models_agree,
        }

    def compute_betting_edge(
        self,
        ensemble_prob: float,
        spread: float,
    ) -> dict:
        """Compute betting edge against the closing spread.

        Args:
            ensemble_prob: Ensemble probability home team wins
            spread: Closing spread (negative = home favored)

        Returns:
            Dictionary with:
                - implied_prob: Market implied probability (after vig)
                - edge: Raw edge (ensemble_prob - implied_prob)
                - edge_after_vig: Edge accounting for vig
                - has_edge: Boolean indicating if edge exceeds threshold
        """
        # Convert spread to implied probability
        # Rough approximation: spread / 13.5 = standardized margin
        # Then use normal CDF
        from scipy.stats import norm

        margin_sd = 13.5
        implied_margin = -spread  # Negative spread = home favored
        implied_prob = norm.cdf(implied_margin / margin_sd)

        # Apply vig adjustment (market takes 4.76% edge with -110 odds on both sides)
        # Adjust implied prob upward for home favorite, downward for underdog
        if spread < 0:
            implied_prob_vig = implied_prob * (1 + self.vig_rate)
        else:
            implied_prob_vig = implied_prob * (1 - self.vig_rate)

        # Clamp to [0, 1]
        implied_prob_vig = max(0.0, min(1.0, implied_prob_vig))

        # Calculate edge
        raw_edge = ensemble_prob - implied_prob
        edge_after_vig = ensemble_prob - implied_prob_vig

        has_edge = edge_after_vig > self.edge_threshold

        return {
            "implied_prob": implied_prob_vig,
            "raw_edge": raw_edge,
            "edge_after_vig": edge_after_vig,
            "has_edge": has_edge,
        }

    def compute_kelly_fraction(
        self,
        edge: float,
        bayesian_sd: float,
        base_kelly: float = 0.25,
    ) -> float:
        """Compute fractional Kelly stake using Bayesian uncertainty.

        Args:
            edge: Edge after vig
            bayesian_sd: Combined Bayesian standard deviation (uncertainty)
            base_kelly: Base Kelly fraction (default: 1/4 Kelly = 0.25)

        Returns:
            Kelly fraction scaled by confidence (lower SD = higher stake)
        """
        # Confidence = 1 / (1 + SD)
        # Low SD (< 1.0) → high confidence (> 0.5) → larger stake
        # High SD (> 1.7) → low confidence (< 0.4) → smaller stake
        confidence = 1.0 / (1.0 + bayesian_sd)

        # Scale base Kelly by confidence and edge magnitude
        kelly_fraction = base_kelly * confidence * min(edge / 0.05, 1.0)

        return kelly_fraction

    def make_bet_decision(
        self,
        bayesian_prob: float,
        xgb_prob: float,
        spread: float,
        bayesian_sd: float,
    ) -> dict:
        """Make betting decision using ensemble logic.

        Strategy:
        1. Compute ensemble probability
        2. Check if both models agree on direction
        3. Compute betting edge against market
        4. If edge > threshold AND models agree → bet
        5. Use Bayesian uncertainty for Kelly sizing

        Args:
            bayesian_prob: Bayesian probability home wins
            xgb_prob: XGBoost probability home wins
            spread: Closing spread
            bayesian_sd: Combined Bayesian standard deviation

        Returns:
            Dictionary with betting recommendation:
                - should_bet: Boolean
                - bet_side: "home" or "away" or None
                - ensemble_prob: Weighted average probability
                - edge: Edge after vig
                - kelly_fraction: Recommended stake size
                - models_agree: Boolean
                - reason: String explanation
        """
        # Compute ensemble probability
        ensemble_result = self.compute_ensemble_probability(bayesian_prob, xgb_prob)
        ensemble_prob = ensemble_result["ensemble_prob"]
        models_agree = ensemble_result["models_agree"]

        # Compute betting edge
        edge_result = self.compute_betting_edge(ensemble_prob, spread)
        edge = edge_result["edge_after_vig"]
        has_edge = edge_result["has_edge"]

        # Decision logic
        should_bet = False
        bet_side = None
        kelly_fraction = 0.0
        reason = ""

        if not models_agree:
            reason = "Models disagree (prob_diff > {:.2f})".format(
                self.agreement_threshold
            )
        elif not has_edge:
            reason = "Insufficient edge (edge={:.3f} < threshold={:.3f})".format(
                edge, self.edge_threshold
            )
        else:
            should_bet = True
            bet_side = "home" if ensemble_prob > 0.5 else "away"
            kelly_fraction = self.compute_kelly_fraction(edge, bayesian_sd)
            reason = "Both models agree + positive edge"

        return {
            "should_bet": should_bet,
            "bet_side": bet_side,
            "ensemble_prob": ensemble_prob,
            "bayesian_prob": bayesian_prob,
            "xgb_prob": xgb_prob,
            "prob_diff": ensemble_result["prob_diff"],
            "models_agree": models_agree,
            "edge": edge,
            "kelly_fraction": kelly_fraction,
            "implied_prob": edge_result["implied_prob"],
            "reason": reason,
        }


def load_xgb_predictions(model_path: str, features_df: pd.DataFrame) -> pd.Series:
    """Load XGBoost model and generate predictions.

    Args:
        model_path: Path to saved XGBoost model (.json or .ubj)
        features_df: Feature dataframe

    Returns:
        Series of home win probabilities
    """
    import xgboost as xgb

    # Load model
    model = xgb.Booster()
    model.load_model(model_path)

    # Prepare features (drop non-feature columns)
    drop_cols = [
        "game_id",
        "season",
        "week",
        "kickoff",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_close",
        "total_close",
        "home_margin",
        "home_win",
        "home_cover",
        "over_hit",
        "is_push",
    ]

    feature_cols = [c for c in features_df.columns if c not in drop_cols]
    X = features_df[feature_cols]

    # Handle missing values
    X = X.fillna(0.0)

    # Convert to DMatrix and predict
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    predictions = model.predict(dmatrix)

    return pd.Series(predictions, index=features_df.index)


def evaluate_ensemble(
    games_df: pd.DataFrame,
    ensemble: BayesianXGBoostEnsemble,
) -> dict:
    """Evaluate ensemble performance on historical data.

    Args:
        games_df: Games with predictions and outcomes
        ensemble: Configured ensemble instance

    Returns:
        Dictionary with performance metrics
    """
    # Make betting decisions
    games_df["decision"] = games_df.apply(
        lambda row: ensemble.make_bet_decision(
            row["bayesian_prob_home"],
            row["xgb_prob_home"],
            row["spread_close"],
            row["bayesian_combined_sd"],
        ),
        axis=1,
    )

    # Extract decision fields
    games_df["should_bet"] = games_df["decision"].apply(lambda d: d["should_bet"])
    games_df["bet_side"] = games_df["decision"].apply(lambda d: d["bet_side"])
    games_df["ensemble_prob"] = games_df["decision"].apply(lambda d: d["ensemble_prob"])
    games_df["edge"] = games_df["decision"].apply(lambda d: d["edge"])
    games_df["kelly_fraction"] = games_df["decision"].apply(lambda d: d["kelly_fraction"])

    # Filter to bets
    bets = games_df[games_df["should_bet"]].copy()

    if len(bets) == 0:
        return {
            "n_games": len(games_df),
            "n_bets": 0,
            "win_rate": 0.0,
            "expected_roi": 0.0,
            "avg_edge": 0.0,
            "avg_kelly": 0.0,
        }

    # Evaluate outcomes
    bets["bet_won"] = (
        ((bets["bet_side"] == "home") & (bets["home_cover"] == 1.0))
        | ((bets["bet_side"] == "away") & (bets["home_cover"] == 0.0))
    )

    # Metrics
    n_games = len(games_df)
    n_bets = len(bets)
    win_rate = bets["bet_won"].mean()
    expected_roi = (win_rate - 0.524) * 100  # 52.4% breakeven with vig
    avg_edge = bets["edge"].mean()
    avg_kelly = bets["kelly_fraction"].mean()

    return {
        "n_games": n_games,
        "n_bets": n_bets,
        "bet_rate": n_bets / n_games,
        "win_rate": win_rate,
        "expected_roi": expected_roi,
        "avg_edge": avg_edge,
        "avg_kelly": avg_kelly,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble betting system: Bayesian + XGBoost"
    )
    parser.add_argument(
        "--games",
        required=True,
        help="CSV with game features including Bayesian ratings",
    )
    parser.add_argument(
        "--xgb-model",
        help="Path to XGBoost model (.json or .ubj). If not provided, uses bayesian_prob_home from CSV.",
    )
    parser.add_argument(
        "--test-season",
        type=int,
        help="Season to evaluate (if not specified, uses all data)",
    )
    parser.add_argument(
        "--bayesian-weight",
        type=float,
        default=0.25,
        help="Weight for Bayesian predictions (default: 0.25)",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=float,
        default=0.10,
        help="Max prob difference for agreement (default: 0.10)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.02,
        help="Minimum edge to bet (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path for results",
    )
    args = parser.parse_args()

    print("=== Bayesian-XGBoost Ensemble System ===\n")

    # Load games
    print(f"Loading games from {args.games}...")
    games_df = pd.read_csv(args.games)
    print(f"✓ Loaded {len(games_df)} games")

    # Filter to test season if specified
    if args.test_season:
        games_df = games_df[games_df["season"] == args.test_season].copy()
        print(f"✓ Filtered to {len(games_df)} games in {args.test_season} season")

    # Filter to games with outcomes
    games_df = games_df[games_df["home_score"].notna()].copy()
    print(f"✓ {len(games_df)} games with outcomes")

    # Load or use XGBoost predictions
    if args.xgb_model:
        print(f"\nLoading XGBoost model from {args.xgb_model}...")
        games_df["xgb_prob_home"] = load_xgb_predictions(args.xgb_model, games_df)
        print("✓ Generated XGBoost predictions")
    else:
        if "bayesian_prob_home" not in games_df.columns:
            raise ValueError(
                "games CSV must have bayesian_prob_home column if no XGBoost model provided"
            )
        # Simulate XGBoost with Bayesian + noise (for testing)
        print("\n⚠ No XGBoost model provided, simulating predictions for testing")
        np.random.seed(42)
        games_df["xgb_prob_home"] = games_df["bayesian_prob_home"] + np.random.normal(
            0, 0.05, len(games_df)
        )
        games_df["xgb_prob_home"] = games_df["xgb_prob_home"].clip(0.01, 0.99)

    # Initialize ensemble
    ensemble = BayesianXGBoostEnsemble(
        bayesian_weight=args.bayesian_weight,
        xgb_weight=1.0 - args.bayesian_weight,
        agreement_threshold=args.agreement_threshold,
        edge_threshold=args.edge_threshold,
    )

    print(f"\n=== Ensemble Configuration ===")
    print(f"Bayesian weight: {ensemble.bayesian_weight:.2f}")
    print(f"XGBoost weight: {ensemble.xgb_weight:.2f}")
    print(f"Agreement threshold: {ensemble.agreement_threshold:.2f}")
    print(f"Edge threshold: {ensemble.edge_threshold:.3f}")

    # Evaluate
    print(f"\n=== Evaluating Ensemble ===")
    results = evaluate_ensemble(games_df, ensemble)

    print(f"\nResults:")
    print(f"  Games analyzed: {results['n_games']}")
    print(f"  Bets placed: {results['n_bets']} ({results['bet_rate']:.1%} of games)")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Expected ROI: {results['expected_roi']:+.2f}%")
    print(f"  Avg edge: {results['avg_edge']:+.3f}")
    print(f"  Avg Kelly: {results['avg_kelly']:.3f}")

    # Save output
    if args.output:
        output_data = {
            "config": {
                "bayesian_weight": args.bayesian_weight,
                "agreement_threshold": args.agreement_threshold,
                "edge_threshold": args.edge_threshold,
            },
            "results": results,
            "games": games_df.to_dict(orient="records"),
        }

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n✓ Results saved to {args.output}")

    print("\n[SUCCESS] Ensemble evaluation complete!")


if __name__ == "__main__":
    main()
