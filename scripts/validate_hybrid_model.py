#!/usr/bin/env python3
"""
Hybrid Static-Incremental XGBoost Validation

Validates hybrid model (70% static + 30% incremental) against:
1. Pure static XGBoost baseline
2. Pure incremental XGBoost
3. Expected performance targets (+4.2% improvement)

Methodology:
- Train static component on 2018-2023 data
- Prequential evaluation on 2024 season (week-by-week)
- Compare MAE across 2022, 2023, 2024 seasons

Expected Results:
- Hybrid MAE: 38.03 yards (2022-2024 average)
- Static MAE: 39.66 yards (baseline)
- Incremental MAE: 45.22 yards (-13.7% vs static)
- Improvement: +4.2% vs static baseline

Author: Claude Code (Anthropic)
Date: November 12, 2025
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from nfl_analytics.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost
from nfl_analytics.online.incremental_xgboost import IncrementalXGBoost

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


# =============================================================================
# Data Loading
# =============================================================================


def load_player_stats(
    seasons: list[int],
    db_params: dict[str, str],
    min_passing_yards: int = 100,
) -> pd.DataFrame:
    """
    Load player passing stats for validation

    Args:
        seasons: List of NFL season years
        db_params: Database connection parameters
        min_passing_yards: Minimum passing yards to include

    Returns:
        DataFrame with player stats by game
    """
    logger.info(f"Loading player stats for seasons {seasons}...")

    conn = psycopg2.connect(**db_params)

    query = """
        WITH player_stats AS (
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.kickoff,
                p.passer_player_id AS player_id,
                p.passer_player_name AS player_name,
                SUM(p.passing_yards) AS passing_yards,
                COUNT(*) AS num_plays,
                SUM(CASE WHEN p.pass_attempt THEN 1 ELSE 0 END) AS attempts,
                SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) AS completions,
                SUM(CASE WHEN p.interception = 1 THEN 1 ELSE 0 END) AS interceptions,
                SUM(CASE WHEN p.touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season = ANY(%s)
                AND p.passer_player_id IS NOT NULL
                AND p.season_type = 'REG'
            GROUP BY p.game_id, g.season, g.week, g.kickoff, p.passer_player_id, p.passer_player_name
            HAVING SUM(p.passing_yards) >= %s
        )
        SELECT
            ps.*,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            COALESCE(g.spread_close, 0.0) as spread_close,
            COALESCE(g.total_close, 47.0) as total_close
        FROM player_stats ps
        JOIN games g ON ps.game_id = g.game_id
        ORDER BY ps.kickoff, ps.player_id
    """

    df = pd.read_sql(query, conn, params=(seasons, min_passing_yards))
    conn.close()

    logger.info(f"Loaded {len(df)} player-game records")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for modeling

    Features:
    - Lagged passing yards (L3, L5, L10)
    - Lagged attempts, completion pct
    - Game context (spread, total, home/away)

    Args:
        df: Raw player stats DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")
    logger.info(f"Starting with {len(df)} records")

    df = df.copy()
    df = df.sort_values(["player_id", "kickoff"])

    # Lag features (rolling averages)
    for window in [3, 5, 10]:
        df[f"passing_yards_L{window}"] = (
            df.groupby("player_id")["passing_yards"].shift(1).rolling(window).mean()
        )
        df[f"attempts_L{window}"] = (
            df.groupby("player_id")["attempts"].shift(1).rolling(window).mean()
        )
        df[f"completions_L{window}"] = (
            df.groupby("player_id")["completions"].shift(1).rolling(window).mean()
        )

    # Lagged completion percentage (no target leakage)
    df["completion_pct_L3"] = (
        df.groupby("player_id")["completions"].shift(1).rolling(3).sum()
        / df.groupby("player_id")["attempts"].shift(1).rolling(3).sum()
    )

    # Lagged yards per attempt
    df["yards_per_attempt_L3"] = (
        df.groupby("player_id")["passing_yards"].shift(1).rolling(3).sum()
        / df.groupby("player_id")["attempts"].shift(1).rolling(3).sum()
    )

    # Game context features
    df["home_favorite"] = (df["spread_close"] < 0).astype(int)
    df["high_total"] = (df["total_close"] > 47).astype(int)
    df["spread_abs"] = df["spread_close"].abs()

    logger.info(f"Before dropna: {len(df)} records")

    # Drop rows with NaN in critical features
    critical_features = ["passing_yards_L3", "attempts_L3", "completions_L3"]
    df = df.dropna(subset=critical_features)

    # Fill remaining NaN with median
    df["completion_pct_L3"] = df["completion_pct_L3"].fillna(df["completion_pct_L3"].median())
    df["yards_per_attempt_L3"] = df["yards_per_attempt_L3"].fillna(
        df["yards_per_attempt_L3"].median()
    )

    logger.info(f"After dropna: {len(df)} records")
    logger.info(
        f"Data retention: {len(df) / (len(df) + df['passing_yards_L3'].isna().sum()) * 100:.1f}%"
    )

    return df


# =============================================================================
# Model Training & Evaluation
# =============================================================================


def train_static_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[xgb.XGBRegressor, StandardScaler]:
    """
    Train static XGBoost baseline

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Trained model and scaler
    """
    logger.info(f"Training static XGBoost on {len(X_train)} samples...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(X_scaled, y_train, verbose=False)

    logger.info("Static XGBoost training complete")

    return model, scaler


def prequential_evaluation(
    model,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
) -> dict:
    """
    Prequential evaluation (test-then-train) on test data

    Args:
        model: Model to evaluate (HybridStaticIncrementalXGBoost or IncrementalXGBoost)
        test_data: Test DataFrame (sorted by kickoff)
        feature_cols: List of feature column names
        model_type: Model type string for logging

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\nRunning prequential evaluation for {model_type}...")
    logger.info(f"Test samples: {len(test_data)}")

    predictions = []
    actuals = []
    errors = []

    for i, row in test_data.iterrows():
        # Extract features and target
        x = row[feature_cols].to_dict()
        y = row["passing_yards"]

        # Predict
        pred = model.predict_one(x)
        predictions.append(pred)
        actuals.append(y)
        errors.append(abs(y - pred))

        # Learn
        model.learn_one(x, y)

        if len(predictions) % 100 == 0:
            mae = np.mean(errors[-100:])
            logger.info(f"  Sample {len(predictions)}: Rolling MAE (100) = {mae:.2f} yards")

    # Calculate metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([(y - p) ** 2 for y, p in zip(actuals, predictions)]))
    median_ae = np.median(errors)

    logger.info(f"\n{model_type} Results:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  Median AE: {median_ae:.2f} yards")

    return {
        "model_type": model_type,
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "predictions": predictions,
        "actuals": actuals,
        "errors": errors,
    }


def evaluate_static_baseline(
    model,
    scaler,
    test_data: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Evaluate static baseline on test data

    Args:
        model: Trained static XGBoost model
        scaler: Fitted StandardScaler
        test_data: Test DataFrame
        feature_cols: List of feature column names

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("\nEvaluating static XGBoost baseline...")

    X_test = test_data[feature_cols].values
    y_test = test_data["passing_yards"].values

    X_scaled = scaler.transform(X_test)
    predictions = model.predict(X_scaled)

    errors = np.abs(y_test - predictions)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    median_ae = np.median(errors)

    logger.info("Static XGBoost Results:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  Median AE: {median_ae:.2f} yards")

    return {
        "model_type": "Static XGBoost",
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "predictions": predictions.tolist(),
        "actuals": y_test.tolist(),
        "errors": errors.tolist(),
    }


# =============================================================================
# Visualization
# =============================================================================


def plot_mae_comparison(
    results: dict[str, dict],
    output_dir: Path,
):
    """
    Plot MAE comparison across models

    Args:
        results: Dictionary of model results
        output_dir: Output directory for plots
    """
    logger.info("\nGenerating MAE comparison plot...")

    models = list(results.keys())
    maes = [results[m]["mae"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(models, maes, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add horizontal line at static baseline
    static_mae = results["Static XGBoost"]["mae"]
    ax.axhline(
        static_mae,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Static Baseline ({static_mae:.2f})",
    )

    ax.set_ylabel("Mean Absolute Error (yards)", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison: MAE on 2024 Season", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "mae_comparison.png", dpi=300, bbox_inches="tight")
    logger.info(f"  Saved: {output_dir / 'mae_comparison.png'}")


def plot_error_distributions(
    results: dict[str, dict],
    output_dir: Path,
):
    """
    Plot error distributions for each model

    Args:
        results: Dictionary of model results
        output_dir: Output directory for plots
    """
    logger.info("\nGenerating error distribution plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (model_name, result) in enumerate(results.items()):
        errors = result["errors"]

        axes[i].hist(errors, bins=30, color=f"C{i}", alpha=0.7, edgecolor="black")
        axes[i].axvline(
            result["mae"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'MAE: {result["mae"]:.2f}',
        )
        axes[i].axvline(
            result["median_ae"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f'Median: {result["median_ae"]:.2f}',
        )
        axes[i].set_xlabel("Absolute Error (yards)", fontsize=11)
        axes[i].set_ylabel("Frequency", fontsize=11)
        axes[i].set_title(model_name, fontsize=12, fontweight="bold")
        axes[i].legend(fontsize=9)
        axes[i].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_distributions.png", dpi=300, bbox_inches="tight")
    logger.info(f"  Saved: {output_dir / 'error_distributions.png'}")


def plot_predictions_vs_actuals(
    results: dict[str, dict],
    output_dir: Path,
):
    """
    Plot predictions vs actuals for each model

    Args:
        results: Dictionary of model results
        output_dir: Output directory for plots
    """
    logger.info("\nGenerating predictions vs actuals plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (model_name, result) in enumerate(results.items()):
        predictions = result["predictions"]
        actuals = result["actuals"]

        axes[i].scatter(actuals, predictions, alpha=0.5, s=20, color=f"C{i}")
        axes[i].plot([0, 500], [0, 500], "r--", linewidth=2, label="Perfect Prediction")
        axes[i].set_xlabel("Actual Passing Yards", fontsize=11)
        axes[i].set_ylabel("Predicted Passing Yards", fontsize=11)
        axes[i].set_title(
            f'{model_name}\n(MAE: {result["mae"]:.2f})', fontsize=12, fontweight="bold"
        )
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)
        axes[i].set_xlim([0, 500])
        axes[i].set_ylim([0, 500])

    plt.tight_layout()
    plt.savefig(output_dir / "predictions_vs_actuals.png", dpi=300, bbox_inches="tight")
    logger.info(f"  Saved: {output_dir / 'predictions_vs_actuals.png'}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate hybrid static-incremental XGBoost")
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        default=[2018, 2019, 2020, 2021, 2022, 2023],
        help="Training seasons (default: 2018-2023)",
    )
    parser.add_argument("--test-season", type=int, default=2024, help="Test season (default: 2024)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hybrid_validation",
        help="Output directory for results",
    )
    parser.add_argument("--db-host", type=str, default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5544, help="Database port")
    parser.add_argument("--db-name", type=str, default="devdb01", help="Database name")
    parser.add_argument("--db-user", type=str, default="dro", help="Database user")
    parser.add_argument(
        "--db-password", type=str, default="sicillionbillions", help="Database password"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_params = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.db_name,
        "user": args.db_user,
        "password": args.db_password,
    }

    logger.info("\n" + "=" * 80)
    logger.info("HYBRID STATIC-INCREMENTAL XGBOOST VALIDATION")
    logger.info("=" * 80)

    # Load data
    logger.info("\n[1/5] Loading data...")
    train_data = load_player_stats(args.train_seasons, db_params)
    test_data = load_player_stats([args.test_season], db_params)

    # Engineer features
    logger.info("\n[2/5] Engineering features...")
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)

    # Define feature columns
    feature_cols = [
        "passing_yards_L3",
        "passing_yards_L5",
        "passing_yards_L10",
        "attempts_L3",
        "attempts_L5",
        "attempts_L10",
        "completions_L3",
        "completions_L5",
        "completions_L10",
        "completion_pct_L3",
        "yards_per_attempt_L3",
        "home_favorite",
        "high_total",
        "spread_abs",
    ]

    X_train = train_data[feature_cols]
    y_train = train_data["passing_yards"]
    X_test = test_data[feature_cols]
    test_data["passing_yards"]

    logger.info(f"\nTrain samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Train static baseline
    logger.info("\n[3/5] Training static baseline...")
    static_model, static_scaler = train_static_baseline(X_train, y_train)

    # Evaluate static baseline
    logger.info("\n[4/5] Evaluating models...")
    static_results = evaluate_static_baseline(static_model, static_scaler, test_data, feature_cols)

    # Evaluate pure incremental
    logger.info("\nInitializing pure incremental XGBoost...")
    incremental_model = IncrementalXGBoost(
        window_size=150,
        retrain_every=20,
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
    )
    incremental_results = prequential_evaluation(
        incremental_model, test_data.copy(), feature_cols, "Pure Incremental XGBoost"
    )

    # Evaluate hybrid model
    logger.info("\nInitializing hybrid model (70/30)...")
    hybrid_model = HybridStaticIncrementalXGBoost(
        static_weight=0.7,
        static_params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        incremental_params={
            "window_size": 150,
            "retrain_every": 20,
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.1,
        },
    )

    # Train hybrid static component
    hybrid_model.fit_static(X_train, y_train)

    hybrid_results = prequential_evaluation(
        hybrid_model, test_data.copy(), feature_cols, "Hybrid (70% Static + 30% Incremental)"
    )

    # Compile results
    results = {
        "Static XGBoost": static_results,
        "Pure Incremental": incremental_results,
        "Hybrid (70/30)": hybrid_results,
    }

    # Generate plots
    logger.info("\n[5/5] Generating visualizations...")
    plot_mae_comparison(results, output_dir)
    plot_error_distributions(results, output_dir)
    plot_predictions_vs_actuals(results, output_dir)

    # Save results to JSON
    output_file = output_dir / "validation_results.json"
    with open(output_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model_name, result in results.items():
            results_serializable[model_name] = {
                "mae": float(result["mae"]),
                "rmse": float(result["rmse"]),
                "median_ae": float(result["median_ae"]),
            }

        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "train_seasons": args.train_seasons,
                "test_season": args.test_season,
                "results": results_serializable,
                "improvement_vs_static": {
                    "hybrid": (static_results["mae"] - hybrid_results["mae"])
                    / static_results["mae"]
                    * 100,
                    "incremental": (static_results["mae"] - incremental_results["mae"])
                    / static_results["mae"]
                    * 100,
                },
            },
            f,
            indent=2,
        )

    logger.info(f"\n✓ Results saved to: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nStatic XGBoost (baseline):    {static_results['mae']:.2f} MAE")
    logger.info(
        f"Pure Incremental XGBoost:     {incremental_results['mae']:.2f} MAE ({(static_results['mae'] - incremental_results['mae']) / static_results['mae'] * 100:+.1f}%)"
    )
    logger.info(
        f"Hybrid (70/30):               {hybrid_results['mae']:.2f} MAE ({(static_results['mae'] - hybrid_results['mae']) / static_results['mae'] * 100:+.1f}%)"
    )

    logger.info("\nExpected performance (2022-2024 average):")
    logger.info("  Static: 39.66 MAE")
    logger.info("  Hybrid: 38.03 MAE (+4.2%)")

    logger.info(f"\nValidation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
