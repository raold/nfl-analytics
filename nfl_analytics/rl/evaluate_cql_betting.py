#!/usr/bin/env python3
"""
Evaluate CQL betting performance on test data.

Compares:
1. Random baseline (50% win rate, -4.5% ROI from vig)
2. Market implied probability baseline
3. Kelly-LCB (conservative Kelly criterion)
4. Single CQL model (best from training)
5. Ensemble CQL (20 models with uncertainty quantification)

Metrics:
- Win rate (target: 52-54%)
- ROI (return on investment, target: 3-5%)
- Sharpe ratio (risk-adjusted returns, target: >1.0)
- Max drawdown (worst peak-to-trough loss, target: <15%)
- Calibration (Q-value accuracy)

Usage:
    python nfl_analytics/rl/evaluate_cql_betting.py \\
        --best-model models/cql/805ae9f0 \\
        --ensemble-dir models/cql \\
        --data data/rl_logged.csv \\
        --output results/cql_betting_evaluation.json \\
        --test-split 0.2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn


# ============================================================================
# Database Connection & Outcome Loading
# ============================================================================


def get_connection() -> psycopg.Connection:
    """Get database connection using environment variables or defaults."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5544")
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def load_game_outcomes(season_start: int = 2019, season_end: int = 2024) -> pd.DataFrame:
    """
    Load actual game outcomes from database SEPARATE from features.

    This prevents target leakage by keeping outcomes out of the training pipeline
    until evaluation time. See ADR-016 for rationale.

    Spread convention in this database:
    - Positive spread_close = home is FAVORITE (giving points)
    - Negative spread_close = home is UNDERDOG (receiving points)

    Home covers when: (home_score - away_score) > spread_close
    Example: KC (home fav by 4, spread=+4) needs margin > 4 to cover
    Example: KC (home dog by 3, spread=-3) needs margin > -3 to cover (can lose by 2)

    Returns:
        DataFrame with game_id, season, week, home_score, away_score, spread_close, cover
    """
    query = """
        SELECT
            game_id,
            season,
            week,
            home_score,
            away_score,
            spread_close,
            CASE WHEN (home_score - away_score) > spread_close THEN 1 ELSE 0 END as cover
        FROM games
        WHERE season BETWEEN %s AND %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(season_start, season_end))
    return df

# ============================================================================
# Q-Network Architecture (must match cql_agent.py)
# ============================================================================


class QNetwork(nn.Module):
    """MLP for Q(s, a) estimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev_dim = state_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# ============================================================================
# Model Loading
# ============================================================================


def load_cql_model(model_path: Path, device: str = "cpu") -> tuple[QNetwork, dict]:
    """Load CQL model from checkpoint."""
    metadata_path = model_path / "metadata.json"
    checkpoint_path = model_path / "best_checkpoint.pth"

    with open(metadata_path) as f:
        metadata = json.load(f)

    config = metadata["config"]
    state_dim = len(config["state_cols"])
    action_dim = 4  # {no-bet (0), small (1), medium (2), large (3)}

    model = QNetwork(state_dim, action_dim, config["hidden_dims"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["q_network"])
    model.eval()

    return model, metadata


def load_ensemble(
    ensemble_dir: Path, ensemble_ids: list[str], device: str = "cpu"
) -> list[QNetwork]:
    """Load ensemble of CQL models."""
    models = []
    for model_id in ensemble_ids:
        model_path = ensemble_dir / model_id
        if model_path.exists():
            model, _ = load_cql_model(model_path, device)
            models.append(model)
    return models


# ============================================================================
# Baseline Policies
# ============================================================================


def random_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Random 50% betting."""
    df = df.copy()
    df["predicted_action"] = np.random.randint(0, 2, len(df))
    df["bet_size"] = 1.0  # Unit stake
    return df


def market_policy(df: pd.DataFrame, threshold: float = 0.52) -> pd.DataFrame:
    """Bet when market_prob > threshold."""
    df = df.copy()
    df["predicted_action"] = (df["market_prob"] > threshold).astype(int)
    df["bet_size"] = 1.0
    return df


def kelly_lcb_policy(df: pd.DataFrame, alpha: float = 0.1, threshold: float = 0.05) -> pd.DataFrame:
    """Kelly criterion with lower confidence bound."""
    df = df.copy()

    # Conservative probability estimate (LCB)
    df["p_lcb"] = df["p_hat"] - alpha
    df["p_lcb"] = df["p_lcb"].clip(0.1, 0.9)

    # Edge = p_lcb - market_implied
    df["kelly_edge"] = df["p_lcb"] - df["market_prob"]

    # Kelly fraction: f = edge / odds, clipped to 5% max
    b = 1.91  # Typical -110 odds
    df["kelly_frac"] = (df["p_lcb"] * b - (1 - df["p_lcb"])) / b
    df["kelly_frac"] = df["kelly_frac"].clip(0, 0.05)

    # Only bet when edge > threshold
    df["predicted_action"] = (df["kelly_edge"] > threshold).astype(int)
    df["bet_size"] = df["kelly_frac"]

    return df


# ============================================================================
# CQL Policies
# ============================================================================


def cql_single_policy(df: pd.DataFrame, model: QNetwork, state_cols: list[str]) -> pd.DataFrame:
    """Single CQL model policy."""
    df = df.copy()

    # Prepare states
    states = torch.FloatTensor(df[state_cols].values)

    # Predict Q-values for all 4 actions: {no-bet, small, medium, large}
    with torch.no_grad():
        q_values = model(states).numpy()

    # Action selection: choose action with highest Q-value
    df["q_no_bet"] = q_values[:, 0]
    df["q_small"] = q_values[:, 1]
    df["q_medium"] = q_values[:, 2]
    df["q_large"] = q_values[:, 3]

    # Best action
    best_actions = q_values.argmax(axis=1)
    df["predicted_action"] = (best_actions > 0).astype(int)  # 0 = no-bet, 1-3 = bet

    # Bet sizing based on chosen action
    bet_sizes = np.zeros(len(df))
    bet_sizes[best_actions == 1] = 0.01  # Small bet: 1%
    bet_sizes[best_actions == 2] = 0.03  # Medium bet: 3%
    bet_sizes[best_actions == 3] = 0.05  # Large bet: 5%
    df["bet_size"] = bet_sizes

    # Q-value advantage
    df["q_best"] = q_values.max(axis=1)
    df["q_advantage"] = df["q_best"] - df["q_no_bet"]

    return df


def cql_ensemble_policy(
    df: pd.DataFrame,
    models: list[QNetwork],
    state_cols: list[str],
    confidence_threshold: float = 0.05,
) -> pd.DataFrame:
    """Ensemble CQL policy with uncertainty filtering."""
    df = df.copy()

    states = torch.FloatTensor(df[state_cols].values)

    # Get predictions from all models (4 actions each)
    q_values_ensemble = []

    with torch.no_grad():
        for model in models:
            q_values = model(states).numpy()
            q_values_ensemble.append(q_values)

    # Stack: (n_models, n_samples, 4)
    q_values_ensemble = np.array(q_values_ensemble)

    # Ensemble statistics for each action
    q_mean = q_values_ensemble.mean(axis=0)  # (n_samples, 4)
    q_std = q_values_ensemble.std(axis=0)  # (n_samples, 4)

    df["q_no_bet_mean"] = q_mean[:, 0]
    df["q_small_mean"] = q_mean[:, 1]
    df["q_medium_mean"] = q_mean[:, 2]
    df["q_large_mean"] = q_mean[:, 3]

    # Best action according to ensemble mean
    best_actions = q_mean.argmax(axis=1)
    df["best_action"] = best_actions
    df["predicted_action"] = (best_actions > 0).astype(int)  # 0 = no-bet, 1-3 = bet

    # Bet sizing based on best action
    bet_sizes = np.zeros(len(df))
    bet_sizes[best_actions == 1] = 0.01  # Small: 1%
    bet_sizes[best_actions == 2] = 0.03  # Medium: 3%
    bet_sizes[best_actions == 3] = 0.05  # Large: 5%
    df["bet_size"] = bet_sizes

    # Uncertainty: std of best action Q-values
    best_q_std = q_std[np.arange(len(df)), best_actions]
    df["q_std"] = best_q_std
    df["confidence"] = 1 / (1 + best_q_std)  # Higher confidence when std is low

    # Filter by confidence: only bet when uncertainty is low
    high_confidence = best_q_std < confidence_threshold
    df["predicted_action"] = df["predicted_action"] & high_confidence

    # Q-value advantage
    df["q_best"] = q_mean.max(axis=1)
    df["q_advantage"] = df["q_best"] - df["q_no_bet_mean"]

    return df


# ============================================================================
# Performance Metrics
# ============================================================================


def calculate_metrics(df: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, float]:
    """
    Calculate betting performance metrics using SEPARATE outcomes data.

    This prevents target leakage by computing rewards only at evaluation time,
    not during dataset generation. See ADR-016 for rationale.

    Args:
        df: DataFrame with predictions (must have game_id, predicted_action, bet_size)
        outcomes: DataFrame from load_game_outcomes() (must have game_id, cover)

    Returns:
        dict of performance metrics
    """
    # Filter to actual bets
    bets = df[df["predicted_action"] == 1].copy()

    if len(bets) == 0:
        return {
            "win_rate": 0.0,
            "roi_pct": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "total_bets": 0,
            "total_staked": 0.0,
            "total_return": 0.0,
        }

    # CRITICAL: Merge with actual outcomes (loaded separately to prevent leakage)
    # Only keep bets where we have outcomes
    bets = bets.merge(outcomes[["game_id", "cover"]], on="game_id", how="inner")

    if len(bets) == 0:
        return {
            "win_rate": 0.0,
            "roi_pct": 0.0,
            "sharpe": 0.0,
            "max_dd_pct": 0.0,
            "total_bets": 0,
            "total_staked": 0.0,
            "total_return": 0.0,
        }

    # Compute reward from ACTUAL cover outcome (not from leaked r column)
    # Standard -110 odds: win pays +0.91, loss costs -1.0
    b = 0.91  # Net odds for -110
    bets["r_actual"] = bets["cover"] * b + (1 - bets["cover"]) * (-1.0)

    # Win/loss outcomes from actual cover
    wins = (bets["r_actual"] > 0).sum()
    total_bets = len(bets)
    win_rate = wins / total_bets

    # ROI calculation using actual rewards
    total_staked = bets["bet_size"].sum()
    total_return = (bets["r_actual"] * bets["bet_size"]).sum()
    roi_pct = (total_return / total_staked * 100) if total_staked > 0 else 0.0

    # Sharpe ratio using actual rewards
    returns = bets["r_actual"] * bets["bet_size"]
    mean_return = returns.mean()
    std_return = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe = mean_return / std_return if std_return > 0 else 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / (running_max + 1e-9)
    max_dd_pct = drawdown.max() * 100

    return {
        "win_rate": win_rate,
        "roi_pct": roi_pct,
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "total_bets": total_bets,
        "total_staked": float(total_staked),
        "total_return": float(total_return),
    }


# ============================================================================
# Main Evaluation
# ============================================================================


def evaluate_all_policies(
    df: pd.DataFrame,
    outcomes: pd.DataFrame,
    best_model: QNetwork | None,
    ensemble_models: list[QNetwork] | None,
    state_cols: list[str],
) -> dict[str, dict[str, float]]:
    """
    Evaluate all policies and return metrics.

    Args:
        df: Feature DataFrame (no outcome columns)
        outcomes: Separate outcomes DataFrame from load_game_outcomes()
        best_model: Single best CQL model (optional)
        ensemble_models: List of ensemble CQL models (optional)
        state_cols: List of state column names for CQL models
    """
    results = {}

    # Baseline 1: Random
    df_random = random_policy(df)
    results["random"] = calculate_metrics(df_random, outcomes)

    # Baseline 2: Market
    df_market = market_policy(df, threshold=0.52)
    results["market"] = calculate_metrics(df_market, outcomes)

    # Baseline 3: Kelly-LCB
    df_kelly = kelly_lcb_policy(df, alpha=0.1, threshold=0.05)
    results["kelly_lcb"] = calculate_metrics(df_kelly, outcomes)

    # CQL Single Model
    if best_model is not None:
        df_cql_single = cql_single_policy(df, best_model, state_cols)
        results["cql_single"] = calculate_metrics(df_cql_single, outcomes)

    # CQL Ensemble
    if ensemble_models is not None and len(ensemble_models) > 0:
        df_cql_ensemble = cql_ensemble_policy(
            df, ensemble_models, state_cols, confidence_threshold=0.05
        )
        results["cql_ensemble"] = calculate_metrics(df_cql_ensemble, outcomes)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CQL betting performance")
    parser.add_argument("--best-model", type=str, help="Path to best CQL model directory")
    parser.add_argument(
        "--ensemble-dir",
        type=str,
        default="models/cql",
        help="Directory containing ensemble models",
    )
    parser.add_argument("--data", type=str, required=True, help="Path to logged dataset CSV")
    parser.add_argument(
        "--output", type=str, default="results/cql_betting_evaluation.json", help="Output JSON path"
    )
    parser.add_argument(
        "--train-seasons",
        type=str,
        default="2019,2020,2021,2022",
        help="Comma-separated training seasons (default: 2019-2022)",
    )
    parser.add_argument(
        "--test-seasons",
        type=str,
        default="2023,2024",
        help="Comma-separated test seasons (default: 2023-2024)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")

    args = parser.parse_args()

    # Parse season lists
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    test_seasons = [int(s.strip()) for s in args.test_seasons.split(",")]

    # Load data
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples from {args.data}")

    # FIXED: Season-based temporal split (prevents data leakage across train/test)
    # This mirrors the XGBoost temporal validation approach per ADR-016
    df_train = df[df["season"].isin(train_seasons)]
    df_test = df[df["season"].isin(test_seasons)]
    print(f"Train seasons: {train_seasons} ({len(df_train)} games)")
    print(f"Test seasons: {test_seasons} ({len(df_test)} games)")

    # Load actual outcomes SEPARATELY from features (prevents target leakage)
    print("Loading game outcomes from database...")
    outcomes = load_game_outcomes(
        season_start=min(min(train_seasons), min(test_seasons)),
        season_end=max(max(train_seasons), max(test_seasons)),
    )
    print(f"Loaded {len(outcomes)} game outcomes")

    # Define state columns
    state_cols = ["spread_close", "total_close", "epa_gap", "market_prob", "p_hat", "edge"]

    # Load models
    best_model = None
    if args.best_model:
        best_model_path = Path(args.best_model)
        best_model, metadata = load_cql_model(best_model_path, args.device)
        print(f"Loaded best model: {best_model_path.name}")

    # Load ensemble (Phase 3 models, seed 42-61)
    ensemble_ids = [
        "ee237922",
        "a19dc3fe",
        "90fe41f9",
        "aa67f6f5",
        "c46a91c3",
        "cd7d1ed9",
        "dc57c8a2",
        "df2233d9",
        "fbaa0f3f",
        "fef28489",
        "655b2be4",
        "1e76793f",
        "3a5be1ef",
        "3ea3746c",
        "33ad8155",
        "487eb7aa",
        "74b1acbf",
        "80e26617",
        "88c895db",
        "090d1bd4",
    ]
    ensemble_dir = Path(args.ensemble_dir)
    ensemble_models = load_ensemble(ensemble_dir, ensemble_ids, args.device)
    print(f"Loaded ensemble: {len(ensemble_models)} models")

    # Evaluate on test set with SEPARATE outcomes (prevents leakage)
    print("\nEvaluating policies on test set...")
    results = evaluate_all_policies(df_test, outcomes, best_model, ensemble_models, state_cols)

    # Print results
    print("\n" + "=" * 80)
    print("BETTING PERFORMANCE EVALUATION (TEST SET)")
    print(f"Train: {train_seasons} | Test: {test_seasons}")
    print("=" * 80)

    for policy_name, metrics in results.items():
        print(f"\n{policy_name.upper()}")
        print(f"  Win rate:     {metrics['win_rate']*100:.1f}%")
        print(f"  ROI:          {metrics['roi_pct']:+.2f}%")
        print(f"  Sharpe:       {metrics['sharpe']:.2f}")
        print(f"  Max DD:       {metrics['max_dd_pct']:.1f}%")
        print(f"  Total bets:   {metrics['total_bets']}")
        print(f"  Total staked: ${metrics['total_staked']:.2f}")
        print(f"  Total return: ${metrics['total_return']:+.2f}")

    print("\n" + "=" * 80)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "dataset": args.data,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "policies": results,
        "ensemble_size": len(ensemble_models),
        "leakage_fix_applied": True,  # Flag for audit trail
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
