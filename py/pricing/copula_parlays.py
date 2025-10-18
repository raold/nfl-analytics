"""
Task 10: Copula Models for Parlay Pricing

Uses Gaussian copulas to model correlation between game outcomes,
enabling accurate pricing of parlays and teasers.

Key insight: Games are not independent - divisional rivals, same-week games,
and conference matchups exhibit correlation that affects parlay EV.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


class GaussianCopulaParlay:
    """
    Gaussian copula for modeling correlated game outcomes.

    Models marginal distributions (individual game win probabilities)
    and dependence structure (correlation matrix) separately.
    """

    def __init__(self):
        self.correlation_matrix = None
        self.marginal_probs = None

    def fit_correlation(
        self, outcomes: np.ndarray, features: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Estimate correlation matrix from historical outcomes.

        Args:
            outcomes: (n_samples, n_games) binary outcomes
            features: Optional features for conditional correlation

        Returns:
            correlation_matrix: (n_games, n_games)
        """
        # Convert outcomes to standard normal via probability integral transform
        n_samples, n_games = outcomes.shape

        # Estimate marginal probabilities
        marginal_probs = outcomes.mean(axis=0)

        # Transform to uniform [0, 1]
        uniforms = np.zeros_like(outcomes, dtype=float)
        for i in range(n_games):
            # Empirical CDF
            uniforms[:, i] = (outcomes[:, i].argsort().argsort() + 1) / (n_samples + 1)

        # Transform to standard normal (inverse CDF)
        normals = stats.norm.ppf(uniforms)

        # Estimate correlation
        correlation_matrix = np.corrcoef(normals.T)

        # Handle numerical issues
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        np.fill_diagonal(correlation_matrix, 1.0)

        self.correlation_matrix = correlation_matrix
        self.marginal_probs = marginal_probs

        return correlation_matrix

    def simulate_parlay(
        self, marginal_probs: np.ndarray, correlation_matrix: np.ndarray, n_simulations: int = 10000
    ) -> np.ndarray:
        """
        Simulate correlated game outcomes for parlay pricing.

        Args:
            marginal_probs: (n_games,) individual win probabilities
            correlation_matrix: (n_games, n_games) correlation matrix
            n_simulations: Number of Monte Carlo simulations

        Returns:
            outcomes: (n_simulations, n_games) binary outcomes
        """
        n_games = len(marginal_probs)

        # Sample from multivariate normal
        mean = np.zeros(n_games)
        normals = np.random.multivariate_normal(
            mean=mean, cov=correlation_matrix, size=n_simulations
        )

        # Transform to uniform via standard normal CDF
        uniforms = stats.norm.cdf(normals)

        # Transform to binary outcomes via marginal probabilities
        outcomes = (uniforms < marginal_probs).astype(int)

        return outcomes

    def price_parlay(
        self,
        marginal_probs: np.ndarray,
        correlation_matrix: np.ndarray,
        parlay_odds: float,
        n_simulations: int = 10000,
    ) -> dict:
        """
        Price a parlay accounting for correlation.

        Args:
            marginal_probs: Individual game win probabilities
            correlation_matrix: Correlation between games
            parlay_odds: Offered parlay payout (e.g., 6.0 for +500)
            n_simulations: Monte Carlo samples

        Returns:
            Dict with EV, win_prob, and fair odds
        """
        # Simulate outcomes
        outcomes = self.simulate_parlay(marginal_probs, correlation_matrix, n_simulations)

        # Check if all games won
        all_won = outcomes.all(axis=1)
        win_prob = all_won.mean()

        # Compute expected value
        # Parlay pays out (parlay_odds - 1) if all win, loses 1 if any lose
        ev = win_prob * (parlay_odds - 1) - (1 - win_prob) * 1
        ev_pct = ev  # Already in decimal form

        # Fair odds (no vig)
        fair_odds = 1 / win_prob if win_prob > 0 else np.inf

        # Implied probability from offered odds
        implied_prob = 1 / parlay_odds

        return {
            "win_prob": win_prob,
            "fair_odds": fair_odds,
            "offered_odds": parlay_odds,
            "implied_prob": implied_prob,
            "edge": win_prob - implied_prob,
            "ev": ev,
            "ev_pct": ev_pct * 100,
        }

    def price_teaser(
        self,
        marginal_probs: np.ndarray,
        spreads: np.ndarray,
        teaser_points: float,
        correlation_matrix: np.ndarray,
        teaser_odds: float,
        n_simulations: int = 10000,
    ) -> dict:
        """
        Price a teaser (adjusted spreads, reduced payout).

        Args:
            marginal_probs: Base win probabilities
            spreads: Original point spreads
            teaser_points: Points to move spread (e.g., 6.0)
            correlation_matrix: Correlation matrix
            teaser_odds: Teaser payout odds
            n_simulations: Monte Carlo samples

        Returns:
            Dict with teaser EV and metrics
        """
        # Adjust probabilities for teaser
        # Rough approximation: each point is worth ~2.5% win probability
        prob_boost_per_point = 0.025
        adjusted_probs = marginal_probs + (teaser_points * prob_boost_per_point)
        adjusted_probs = np.clip(adjusted_probs, 0.01, 0.99)

        # Price as parlay with adjusted probabilities
        return self.price_parlay(adjusted_probs, correlation_matrix, teaser_odds, n_simulations)


def estimate_game_correlation(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate pairwise correlation between games based on features.

    Correlations arise from:
    1. Same week (common factors: weather, officiating trends)
    2. Same division (rivalry dynamics)
    3. Same conference (playoff implications)
    4. Shared opponents (transitive strength)
    """
    # Placeholder: simple correlation estimation
    # In production, would use more sophisticated methods

    correlations = []

    for idx1, game1 in games_df.iterrows():
        for idx2, game2 in games_df.iterrows():
            if idx1 >= idx2:
                continue

            corr = 0.0

            # Same week
            if game1["season"] == game2["season"] and game1["week"] == game2["week"]:
                corr += 0.05

            # Shared team
            shared_teams = set([game1["home_team"], game1["away_team"]]) & set(
                [game2["home_team"], game2["away_team"]]
            )
            if len(shared_teams) > 0:
                corr += 0.15

            # Same division
            # (Would need division lookup table)

            correlations.append(
                {
                    "game1_id": idx1,
                    "game2_id": idx2,
                    "correlation": corr,
                }
            )

    return pd.DataFrame(correlations)


def backtest_parlay_pricing(
    games_df: pd.DataFrame, model_probs: np.ndarray, parlay_size: int = 2, n_trials: int = 100
) -> dict:
    """
    Backtest parlay pricing with copula vs independence assumption.

    Args:
        games_df: Historical games with outcomes
        model_probs: Model-predicted win probabilities
        parlay_size: Number of games in parlay
        n_trials: Number of random parlays to test

    Returns:
        Dict comparing copula vs independence pricing
    """
    copula = GaussianCopulaParlay()

    # Prepare outcomes matrix for recent games (last 500)
    recent_df = games_df.tail(500).copy()
    (recent_df["home_score"] > recent_df["away_score"]).values.reshape(-1, 1)

    # Build correlation matrix (simplified: use same week as proxy)
    # In production, would use proper temporal correlation estimation
    len(recent_df)

    # For simplicity, assume small correlation (0.05) between all games
    # This is a placeholder - real implementation would estimate from data
    base_correlation = 0.05
    correlation_matrix = np.ones((parlay_size, parlay_size)) * base_correlation
    np.fill_diagonal(correlation_matrix, 1.0)

    results = {
        "independent": [],
        "copula": [],
    }

    for trial in range(n_trials):
        # Random sample of games
        sample_indices = np.random.choice(len(games_df), size=parlay_size, replace=False)
        sample_probs = model_probs[sample_indices]

        # True outcomes
        true_outcomes = (
            games_df.iloc[sample_indices]["home_score"]
            > games_df.iloc[sample_indices]["away_score"]
        ).values

        all_won = true_outcomes.all()

        # Independence assumption
        independent_prob = sample_probs.prod()
        independent_fair_odds = 1 / independent_prob if independent_prob > 0 else np.inf

        # Copula model
        copula_result = copula.price_parlay(
            marginal_probs=sample_probs,
            correlation_matrix=correlation_matrix,
            parlay_odds=independent_fair_odds,  # Compare against independence
            n_simulations=1000,
        )

        results["independent"].append(
            {
                "win_prob": independent_prob,
                "fair_odds": independent_fair_odds,
                "actual_won": all_won,
            }
        )

        results["copula"].append(
            {
                "win_prob": copula_result["win_prob"],
                "fair_odds": copula_result["fair_odds"],
                "actual_won": all_won,
            }
        )

    # Aggregate results
    independent_win_prob_mean = np.mean([r["win_prob"] for r in results["independent"]])
    copula_win_prob_mean = np.mean([r["win_prob"] for r in results["copula"]])

    actual_win_rate = np.mean([r["actual_won"] for r in results["independent"]])

    return {
        "n_trials": n_trials,
        "parlay_size": parlay_size,
        "actual_win_rate": actual_win_rate,
        "independent_predicted": independent_win_prob_mean,
        "copula_predicted": copula_win_prob_mean,
        "independent_error": abs(independent_win_prob_mean - actual_win_rate),
        "copula_error": abs(copula_win_prob_mean - actual_win_rate),
        "copula_improvement": abs(independent_win_prob_mean - actual_win_rate)
        - abs(copula_win_prob_mean - actual_win_rate),
    }


def main():
    """
    Example: Price parlays using Gaussian copula.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Copula parlay pricing")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to games CSV with model predictions"
    )
    parser.add_argument(
        "--output", type=str, default="results/copula/parlay_analysis.json", help="Output path"
    )
    parser.add_argument("--parlay-size", type=int, default=2, help="Parlay size (number of games)")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of backtests")

    args = parser.parse_args()

    print("=" * 80)
    print("Task 10: Copula Parlay Pricing")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Check for model probabilities
    if "model_prob" not in df.columns and "gnn_win_prob" in df.columns:
        df["model_prob"] = df["gnn_win_prob"]
    elif "model_prob" not in df.columns:
        print("ERROR: No model_prob column found")
        return

    print(f"  Loaded {len(df)} games")

    # Backtest
    print(f"\nBacktesting {args.parlay_size}-game parlays ({args.n_trials} trials)...")
    results = backtest_parlay_pricing(
        games_df=df,
        model_probs=df["model_prob"].values,
        parlay_size=args.parlay_size,
        n_trials=args.n_trials,
    )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Results")
    print(f"{'=' * 80}")
    print(f"\nActual win rate: {results['actual_win_rate']:.3f}")
    print("\nIndependence assumption:")
    print(f"  Predicted: {results['independent_predicted']:.3f}")
    print(f"  Error: {results['independent_error']:.4f}")
    print("\nCopula model:")
    print(f"  Predicted: {results['copula_predicted']:.3f}")
    print(f"  Error: {results['copula_error']:.4f}")
    print(f"\nImprovement: {results['copula_improvement']:.4f}")

    if results["copula_improvement"] > 0:
        print("✓ Copula model more accurate than independence")
    else:
        print("✗ Copula model not improving over independence")

    print(f"\n{'=' * 80}")
    print(f"Results saved to {args.output}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
