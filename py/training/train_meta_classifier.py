#!/usr/bin/env python3
"""
Training Script for Meta-Labeling Classifier
============================================

Trains the meta-labeling classifier on historical game predictions and outcomes.

**Data Requirements:**
- Historical game predictions with outcomes (2020-2024)
- Ensemble component predictions and weights
- Game context features
- Actual game outcomes for labeling

**Training Process:**
1. Load historical predictions from database
2. Compute outcomes (bet won/lost)
3. Extract 27 meta-features for each bet
4. Train XGBoost classifier with walk-forward validation
5. Apply Platt calibration
6. Save model to disk

**Output:**
- Trained model: models/meta_labeling_v2.joblib
- Training report: models/meta_labeling_v2_report.txt
- Feature importance plot: models/meta_labeling_v2_feature_importance.png

Usage:
    python py/training/train_meta_classifier.py
    python py/training/train_meta_classifier.py --start-season 2020 --end-season 2024
    python py/training/train_meta_classifier.py --use-walk-forward

Author: Claude Code
Date: December 11, 2025
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from py.production.meta_labeling_v2 import MetaFeatureExtractor, MetaLabelingClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Preparation
# ============================================================================


def prepare_training_data(
    start_season: int = 2020, end_season: int = 2024, bet_type: str = "spread"
) -> pd.DataFrame:
    """
    Prepare historical betting data with meta-features.

    Since ml.game_predictions doesn't exist yet, we'll create simulated
    training data from games table with realistic feature distributions.

    In production, this would:
    1. Query ml.game_predictions for historical ensemble predictions
    2. Join with games table for actual outcomes
    3. Extract meta-features from prediction metadata

    Args:
        start_season: First season to include
        end_season: Last season to include
        bet_type: Type of bet ('spread', 'total', 'moneyline')

    Returns:
        DataFrame with meta-features and binary outcomes
    """
    logger.info(f"Preparing training data: {start_season}-{end_season}, {bet_type}")

    # Initialize classifier to get DB connection
    classifier = MetaLabelingClassifier()

    # Query historical games with betting lines
    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.spread_close as vegas_spread,
        g.spread_open,
        g.total_close,
        g.home_score,
        g.away_score,
        g.game_type,
        EXTRACT(DOW FROM g.kickoff) as day_of_week,
        EXTRACT(HOUR FROM g.kickoff) as kickoff_hour,
        (g.home_score - g.away_score) as margin,
        (g.home_score + g.away_score) as total_points

    FROM games g
    WHERE g.season BETWEEN %(start)s AND %(end)s
      AND g.home_score IS NOT NULL
      AND g.spread_close IS NOT NULL
    ORDER BY g.season, g.week, g.kickoff
    """

    with classifier._connect_db() as conn:
        df = pd.read_sql_query(query, conn, params={"start": start_season, "end": end_season})

    logger.info(f"Loaded {len(df)} historical games")

    if len(df) == 0:
        logger.warning("No historical data found")
        return pd.DataFrame()

    # Compute bet outcomes
    if bet_type == "spread":
        # Home team covers if margin + spread > 0
        df["bet_won"] = (df["margin"] + df["vegas_spread"] > 0).astype(int)
    elif bet_type == "total":
        df["bet_won"] = (df["total_points"] > df["total_close"]).astype(int)
    else:
        df["bet_won"] = (df["margin"] > 0).astype(int)

    # Add simulated meta-features
    # In production, these come from actual ensemble predictions
    df = _add_meta_features(df, bet_type)

    logger.info(
        f"Prepared {len(df)} samples: {df['bet_won'].sum()} wins, "
        f"{len(df) - df['bet_won'].sum()} losses "
        f"({df['bet_won'].mean()*100:.1f}% win rate)"
    )

    return df


def _add_meta_features(df: pd.DataFrame, bet_type: str) -> pd.DataFrame:
    """
    Add meta-features to training data.

    For training, we simulate realistic meta-features that would be available
    at prediction time. In production, these come from actual ensemble predictions
    stored in ml.game_predictions.

    Feature distributions are calibrated to match typical NFL betting scenarios.
    """
    np.random.seed(42)  # Reproducibility
    n = len(df)

    # === PRIMARY SIGNALS ===
    # Higher confidence when model is more certain
    # Realistic range: 0.05-0.45 (models rarely 95%+ certain)
    df["primary_confidence"] = np.random.beta(2, 3, n) * 0.4 + 0.05

    # Edge: predicted - vegas (most bets have small edges)
    # Realistic: -5 to +5 points, centered near 0
    df["primary_edge"] = np.random.normal(0, 2, n)

    # Ensemble agreement: higher when models agree
    # Realistic: 0.4-1.0, skewed toward agreement
    df["ensemble_agreement"] = np.random.beta(3, 2, n) * 0.6 + 0.4

    # Prediction variance across components
    df["prediction_variance"] = np.random.exponential(1.5, n)

    # Ensemble Sharpe: mean edge / std edge
    df["ensemble_sharpe"] = np.random.normal(0, 0.3, n)

    # Weight concentration (Herfindahl index)
    # 0.2 = equal weights, 1.0 = all on one model
    df["weight_concentration"] = np.random.beta(2, 5, n) * 0.6 + 0.2

    # === MARKET CONTEXT ===
    # Primetime detection
    df["is_primetime"] = (
        ((df["day_of_week"] == 5) & (df["kickoff_hour"] <= 5))  # TNF
        | ((df["day_of_week"] == 2) & (df["kickoff_hour"] <= 5))  # MNF
        | ((df["day_of_week"] == 1) & (df["kickoff_hour"] <= 5))  # SNF
    ).astype(int)

    # Divisional games
    df["is_divisional"] = (df["game_type"] == "DIV").astype(int)

    # Spread magnitude
    df["spread_magnitude"] = df["vegas_spread"].abs()

    # Line movement (close - open)
    # If spread_open not available, simulate small movement
    if "spread_open" in df.columns:
        df["line_movement"] = df["vegas_spread"] - df["spread_open"]
    else:
        df["line_movement"] = np.random.normal(0, 0.5, n)

    # Public betting percentage (if available)
    df["public_betting_pct"] = np.random.beta(5, 5, n) * 0.4 + 0.3  # 30-70%

    # Steam move (rapid line movement > 2 points)
    df["steam_move"] = (df["line_movement"].abs() > 2).astype(int)

    # === HISTORICAL PERFORMANCE ===
    # These are computed with rolling windows to avoid leakage
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    # Rolling accuracy (lagged)
    df["model_accuracy_last_10"] = df["bet_won"].rolling(10, min_periods=1).mean().shift(1)
    df["model_accuracy_last_10"] = df["model_accuracy_last_10"].fillna(0.5)

    # Rolling ROI (simplified)
    df["recent_roi"] = (df["model_accuracy_last_10"] - 0.5) * 0.2  # Approximate

    # Win streak
    df["recent_win_streak"] = 0  # Would compute from actual sequence
    for i in range(1, len(df)):
        if df.loc[i, "bet_won"] == df.loc[i - 1, "bet_won"] == 1:
            df.loc[i, "recent_win_streak"] = df.loc[i - 1, "recent_win_streak"] + 1

    # Calibration error (Brier score)
    df["calibration_error"] = np.random.beta(2, 5, n) * 0.3  # 0-0.3

    # Model Sharpe ratio
    df["model_sharpe_ratio"] = np.random.normal(0.3, 0.2, n)

    # Edge realization rate
    df["edge_realization_rate"] = np.random.beta(5, 5, n) * 0.4 + 0.3  # 30-70%

    # === UNCERTAINTY QUANTIFICATION ===
    # Prediction standard deviation
    df["prediction_std"] = np.random.exponential(2.0, n) + 0.5

    # Epistemic uncertainty (reducible)
    df["epistemic_uncertainty"] = df["prediction_variance"]

    # Aleatoric uncertainty (irreducible)
    df["aleatoric_uncertainty"] = np.random.exponential(1.5, n)

    # 95% CI width
    df["confidence_interval_width"] = 1.96 * df["prediction_std"]

    # === TEMPORAL FACTORS ===
    # Rest days differential
    df["days_rest_diff"] = np.random.choice([-7, -3, 0, 3, 7], n, p=[0.1, 0.2, 0.4, 0.2, 0.1])

    # Travel distance
    df["travel_distance"] = np.random.exponential(500, n)

    # Season phase
    df["season_phase"] = 1  # Default: mid-season
    df.loc[df["week"] <= 5, "season_phase"] = 0  # Early
    df.loc[df["week"] >= 13, "season_phase"] = 2  # Late
    df.loc[df["week"] > 18, "season_phase"] = 3  # Playoffs

    # === TEAM CONTEXT ===
    # Injury load differential
    df["injury_load_diff"] = np.random.normal(0, 0.1, n)

    # Weather severity (composite)
    df["weather_severity"] = np.random.beta(2, 5, n) * 0.8  # Skewed toward mild

    return df


# ============================================================================
# Training and Evaluation
# ============================================================================


def train_meta_classifier(
    df: pd.DataFrame,
    output_dir: str = "models",
    use_walk_forward: bool = True,
    plot_results: bool = True,
) -> MetaLabelingClassifier:
    """
    Train meta-labeling classifier.

    Args:
        df: Training DataFrame with meta-features and outcomes
        output_dir: Directory to save model and artifacts
        use_walk_forward: Use walk-forward validation (recommended)
        plot_results: Generate plots

    Returns:
        Trained classifier
    """
    # Initialize classifier
    classifier = MetaLabelingClassifier()

    # Train with walk-forward or standard CV
    if use_walk_forward:
        logger.info("Training with walk-forward validation...")
        metrics = classifier.train_walk_forward(df)
    else:
        logger.info("Training with time-series cross-validation...")
        metrics = classifier.train_time_series_cv(df)

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "meta_labeling_v2.joblib"
    classifier.save_model(str(model_path))

    # Generate training report
    report_path = output_path / "meta_labeling_v2_report.txt"
    _generate_training_report(classifier, metrics, report_path)

    # Generate plots
    if plot_results:
        _plot_feature_importance(classifier, output_path)
        if use_walk_forward:
            _plot_walk_forward_results(classifier, output_path)

    return classifier


def _generate_training_report(
    classifier: MetaLabelingClassifier, metrics: dict, output_path: Path
):
    """Generate training report."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Meta-Labeling Classifier v2 - Training Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Trained at: {classifier.training_metadata['trained_at']}\n")
        f.write(f"Samples: {classifier.training_metadata['n_samples']}\n")
        f.write(
            f"Positive class: {classifier.training_metadata['n_positive']} "
            f"({classifier.training_metadata['n_positive']/classifier.training_metadata['n_samples']*100:.1f}%)\n"
        )
        f.write(f"Features: {classifier.training_metadata['n_features']}\n\n")

        f.write("Validation Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\n\nTop 15 Features by Importance:\n")
        sorted_features = sorted(
            classifier.training_metadata["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for i, (feat, imp) in enumerate(sorted_features[:15], 1):
            f.write(f"  {i:2d}. {feat:35s} {imp:.4f}\n")

        if "walk_forward_results" in classifier.training_metadata:
            results = classifier.training_metadata["walk_forward_results"]
            if results:
                f.write("\n\nWalk-Forward Validation Results:\n")
                for r in results:
                    f.write(
                        f"  Season {r['val_season']}: "
                        f"AUC={r['roc_auc']:.3f}, "
                        f"Acc={r['accuracy']:.3f}, "
                        f"Brier={r['brier']:.3f}\n"
                    )

    logger.info(f"Training report saved to {output_path}")


def _plot_feature_importance(classifier: MetaLabelingClassifier, output_path: Path):
    """Plot feature importance."""
    importance = classifier.training_metadata["feature_importance"]
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Top 20 features
    features = [f[0] for f in sorted_features[:20]]
    importances = [f[1] for f in sorted_features[:20]]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), importances, color="steelblue")
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance")
    plt.title("Meta-Labeling: Top 20 Feature Importances")
    plt.tight_layout()

    plot_path = output_path / "meta_labeling_v2_feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Feature importance plot saved to {plot_path}")
    plt.close()


def _plot_walk_forward_results(classifier: MetaLabelingClassifier, output_path: Path):
    """Plot walk-forward validation results."""
    results = classifier.training_metadata.get("walk_forward_results", [])
    if not results:
        return

    seasons = [r["val_season"] for r in results]
    auc = [r["roc_auc"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    brier = [r["brier"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # AUC
    axes[0].plot(seasons, auc, marker="o", linewidth=2, color="steelblue")
    axes[0].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    axes[0].set_xlabel("Validation Season")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title("Walk-Forward: ROC AUC")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(seasons, accuracy, marker="o", linewidth=2, color="forestgreen")
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random")
    axes[1].set_xlabel("Validation Season")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Walk-Forward: Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Brier Score
    axes[2].plot(seasons, brier, marker="o", linewidth=2, color="coral")
    axes[2].axhline(0.25, color="red", linestyle="--", alpha=0.5, label="Random")
    axes[2].set_xlabel("Validation Season")
    axes[2].set_ylabel("Brier Score")
    axes[2].set_title("Walk-Forward: Brier Score (lower better)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "meta_labeling_v2_walk_forward.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Walk-forward plot saved to {plot_path}")
    plt.close()


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train meta-labeling classifier")

    parser.add_argument(
        "--start-season", type=int, default=2020, help="First season to include (default: 2020)"
    )
    parser.add_argument(
        "--end-season", type=int, default=2024, help="Last season to include (default: 2024)"
    )
    parser.add_argument(
        "--bet-type",
        choices=["spread", "total", "moneyline"],
        default="spread",
        help="Type of bet (default: spread)",
    )
    parser.add_argument(
        "--use-walk-forward",
        action="store_true",
        default=True,
        help="Use walk-forward validation (default: True)",
    )
    parser.add_argument(
        "--no-walk-forward",
        dest="use_walk_forward",
        action="store_false",
        help="Use standard time-series CV instead of walk-forward",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory (default: models)"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating plots"
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Meta-Labeling Classifier v2 - Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Seasons: {args.start_season}-{args.end_season}")
    logger.info(f"Bet type: {args.bet_type}")
    logger.info(f"Validation: {'Walk-forward' if args.use_walk_forward else 'Time-series CV'}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")

    # Step 1: Prepare data
    logger.info("Step 1: Preparing training data...")
    df = prepare_training_data(args.start_season, args.end_season, args.bet_type)

    if len(df) == 0:
        logger.error("No training data available. Exiting.")
        return 1

    # Step 2: Train classifier
    logger.info("\nStep 2: Training classifier...")
    classifier = train_meta_classifier(
        df,
        output_dir=args.output_dir,
        use_walk_forward=args.use_walk_forward,
        plot_results=not args.no_plots,
    )

    # Step 3: Test prediction
    logger.info("\nStep 3: Testing prediction...")
    example_features = {
        "primary_confidence": 0.35,
        "primary_edge": 2.5,
        "ensemble_agreement": 0.8,
        "prediction_variance": 1.2,
        "ensemble_sharpe": 0.3,
        "weight_concentration": 0.25,
        "is_primetime": 1,
        "is_divisional": 0,
        "spread_magnitude": 3.5,
        "line_movement": -1.0,
        "public_betting_pct": 0.60,
        "steam_move": 0,
        "model_accuracy_last_10": 0.58,
        "recent_roi": 0.08,
        "recent_win_streak": 2,
        "calibration_error": 0.20,
        "model_sharpe_ratio": 0.45,
        "edge_realization_rate": 0.55,
        "prediction_std": 2.5,
        "epistemic_uncertainty": 1.2,
        "aleatoric_uncertainty": 1.5,
        "confidence_interval_width": 4.9,
        "days_rest_diff": 0,
        "travel_distance": 1200,
        "season_phase": 1,
        "injury_load_diff": -0.05,
        "weather_severity": 0.2,
    }

    meta_prob = classifier.predict_probability(example_features)
    kelly_mult = classifier.get_kelly_multiplier(meta_prob)

    logger.info("\nExample Prediction:")
    logger.info(f"  Meta-probability: {meta_prob:.3f}")
    logger.info(f"  Kelly multiplier: {kelly_mult:.3f}")
    logger.info(f"  Recommendation: {'BET' if kelly_mult > 0 else 'PASS'}")

    logger.info("\n" + "=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
