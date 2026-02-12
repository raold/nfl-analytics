#!/usr/bin/env python3
"""
Generate simulated metrics from copula-based game simulator.

This script:
1. Loads trained copula models (Dixon-Coles for marginals, Gaussian copula for dependence)
2. Simulates 10,000 games using the same parameters
3. Computes the same metrics as historical data for comparison

Output: analysis/results/simulated_metrics.json

Usage:
    python py/sim/generate_simulated_metrics.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm


def load_dixon_coles_params(path: str = "models/dixon_coles_params.json") -> dict:
    """
    Load Dixon-Coles parameters.

    Args:
        path: Path to saved parameters

    Returns:
        Dict with home_attack, home_defense, away_attack, away_defense, rho
    """
    if not Path(path).exists():
        print(f"⚠️  Dixon-Coles params not found at {path}")
        print("   Using default parameters from NFL averages")
        return {
            "home_advantage": 0.3,  # ~2.5 points
            "avg_attack": 1.3,  # ~23 points per team
            "avg_defense": 1.3,
            "rho": -0.15,  # Slight negative dependence
        }

    with open(path) as f:
        return json.load(f)


def simulate_game_copula(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    rho: float = -0.1,
    n_sims: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate game using Gaussian copula + Poisson marginals.

    Args:
        home_attack: Home team attack strength (log scale)
        home_defense: Home team defense strength (log scale, negative)
        away_attack: Away team attack strength (log scale)
        away_defense: Away team defense strength (log scale, negative)
        rho: Copula correlation parameter
        n_sims: Number of simulations

    Returns:
        (home_scores, away_scores) as numpy arrays
    """
    # Expected scores (attack and defense are already in log-scale)
    # Dixon-Coles: log(λ_home) = attack_home - defense_away + home_adv
    #              log(λ_away) = attack_away - defense_home
    # Defense parameters are NEGATIVE, so we subtract them (i.e., add their absolute value)
    lambda_home = np.exp(
        home_attack - away_defense
    )  # away_defense is negative, so subtracting adds
    lambda_away = np.exp(away_attack - home_defense)  # home_defense is negative

    # Gaussian copula
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    # Sample from copula
    z = np.random.multivariate_normal(mean, cov, size=n_sims)

    # Transform to uniform [0, 1]
    u_home = norm.cdf(z[:, 0])
    u_away = norm.cdf(z[:, 1])

    # Transform to Poisson quantiles
    # Use inverse CDF (quantile function)
    from scipy.stats import poisson

    home_scores = poisson.ppf(u_home, lambda_home)
    away_scores = poisson.ppf(u_away, lambda_away)

    return home_scores.astype(int), away_scores.astype(int)


def simulate_season(n_games: int = 1408, params: dict = None, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a full season of games.

    Args:
        n_games: Number of games to simulate
        params: Dixon-Coles parameters (with attack/defense dicts)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with home_score, away_score columns
    """
    np.random.seed(seed)

    if params is None:
        params = load_dixon_coles_params()

    # Extract parameters
    home_adv = params.get("home_advantage", 0.1)
    rho = params.get("rho", -0.1)

    # Get team strengths
    attack_params = params.get("attack", {})
    defense_params = params.get("defense", {})

    teams = list(attack_params.keys())
    if len(teams) == 0:
        raise ValueError("No team parameters found in Dixon-Coles params")

    # Simulate games
    home_scores = []
    away_scores = []

    for _ in range(n_games):
        # Sample two random teams
        home_team, away_team = np.random.choice(teams, size=2, replace=False)

        # Get their strengths
        home_attack = attack_params[home_team] + home_adv
        home_defense = defense_params[home_team]
        away_attack = attack_params[away_team]
        away_defense = defense_params[away_team]

        h_score, a_score = simulate_game_copula(
            home_attack, home_defense, away_attack, away_defense, rho, n_sims=1
        )

        home_scores.append(h_score[0])
        away_scores.append(a_score[0])

    return pd.DataFrame({"home_score": home_scores, "away_score": away_scores})


def compute_margin_distribution(df: pd.DataFrame) -> dict:
    """Compute margin distribution (same as historical)."""
    margins = df["home_score"] - df["away_score"]

    margin_counts = margins.value_counts().sort_index()
    total_games = len(margins)

    pmf = {}
    for margin, count in margin_counts.items():
        pmf[int(margin)] = float(count / total_games)

    return {
        "pmf": pmf,
        "mean": float(margins.mean()),
        "std": float(margins.std()),
        "min": int(margins.min()),
        "max": int(margins.max()),
        "n_games": total_games,
    }


def compute_key_number_masses(df: pd.DataFrame, key_numbers: list[int]) -> dict:
    """Compute key number masses (same as historical)."""
    margins = (df["home_score"] - df["away_score"]).abs()
    total_games = len(margins)

    masses = {}
    for key in key_numbers:
        count = (margins == key).sum()
        masses[key] = float(count / total_games)

    return masses


def compute_total_distribution(df: pd.DataFrame) -> dict:
    """Compute total distribution (same as historical)."""
    totals = df["home_score"] + df["away_score"]

    total_counts = totals.value_counts().sort_index()
    n_games = len(totals)

    pmf = {}
    for total, count in total_counts.items():
        pmf[int(total)] = float(count / n_games)

    return {
        "pmf": pmf,
        "mean": float(totals.mean()),
        "std": float(totals.std()),
        "min": int(totals.min()),
        "max": int(totals.max()),
        "n_games": n_games,
    }


def compute_score_dependence(df: pd.DataFrame) -> dict:
    """Compute score dependence (same as historical)."""
    home = df["home_score"].values
    away = df["away_score"].values

    tau, p_value = kendalltau(home, away)
    pearson = np.corrcoef(home, away)[0, 1]

    from scipy.stats import spearmanr

    spearman, _ = spearmanr(home, away)

    return {
        "kendall_tau": float(tau),
        "kendall_p_value": float(p_value),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "n_games": len(df),
    }


def compute_upset_rate(df: pd.DataFrame) -> float:
    """Compute home win rate (same as historical)."""
    home_wins = (df["home_score"] > df["away_score"]).sum()
    total_games = len(df)
    return float(home_wins / total_games)


def generate_simulated_metrics(
    n_games: int = 10000, output_path: str = "analysis/results/simulated_metrics.json"
):
    """
    Main function to generate simulated metrics.

    Args:
        n_games: Number of games to simulate
        output_path: Path to save JSON output
    """
    print("=" * 80)
    print("GENERATING SIMULATED METRICS")
    print("=" * 80)

    # Load parameters
    print("\n1. Loading copula parameters...")
    params = load_dixon_coles_params()
    print("✅ Parameters loaded")
    print(f"   Teams: {len(params.get('attack', {}))}")
    print(f"   Home advantage: {params.get('home_advantage', 'N/A'):.4f}")
    print(f"   Rho (dependence): {params.get('rho', 'N/A')}")

    # Simulate games
    print(f"\n2. Simulating {n_games} games...")
    df = simulate_season(n_games=n_games, params=params)
    print(f"✅ Simulated {len(df)} games")

    # Compute metrics
    print("\n3. Computing margin distribution...")
    margin_dist = compute_margin_distribution(df)
    print(f"✅ Mean margin: {margin_dist['mean']:.2f} ± {margin_dist['std']:.2f}")
    print(f"   Range: [{margin_dist['min']}, {margin_dist['max']}]")

    print("\n4. Computing key number masses...")
    key_numbers = [3, 6, 7, 10]
    key_masses = compute_key_number_masses(df, key_numbers)
    print("✅ Key number masses:")
    for key, mass in key_masses.items():
        print(f"   {key:2d} points: {mass*100:.2f}%")

    print("\n5. Computing total score distribution...")
    total_dist = compute_total_distribution(df)
    print(f"✅ Mean total: {total_dist['mean']:.2f} ± {total_dist['std']:.2f}")
    print(f"   Range: [{total_dist['min']}, {total_dist['max']}]")

    print("\n6. Computing score dependence...")
    dependence = compute_score_dependence(df)
    print(
        f"✅ Kendall's tau: {dependence['kendall_tau']:.4f} (p={dependence['kendall_p_value']:.4e})"
    )
    print(f"   Pearson: {dependence['pearson']:.4f}")
    print(f"   Spearman: {dependence['spearman']:.4f}")

    print("\n7. Computing home win rate...")
    home_win_rate = compute_upset_rate(df)
    print(f"✅ Home win rate: {home_win_rate*100:.1f}%")

    # Compile results
    results = {
        "metadata": {
            "n_games": len(df),
            "simulation_seed": 42,
            "description": "Simulated NFL games using Gaussian copula + Poisson marginals",
        },
        "parameters": params,
        "margin_distribution": margin_dist,
        "key_number_masses": key_masses,
        "total_distribution": total_dist,
        "score_dependence": dependence,
        "home_win_rate": home_win_rate,
    }

    # Save to file
    print(f"\n8. Saving results to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved to {output_path}")

    print("\n" + "=" * 80)
    print("SIMULATED METRICS COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  Games simulated: {len(df)}")
    print(f"  Mean margin: {margin_dist['mean']:.2f}")
    print(f"  Key mass (3pt): {key_masses[3]*100:.2f}%")
    print(f"  Key mass (7pt): {key_masses[7]*100:.2f}%")
    print(f"  Kendall's tau: {dependence['kendall_tau']:.4f}")
    print("\n✅ Ready for Phase 3: Acceptance test comparison")


if __name__ == "__main__":
    generate_simulated_metrics(n_games=10000)
