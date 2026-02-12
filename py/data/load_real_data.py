#!/usr/bin/env python3
"""
Merge real predictions with game features and outcomes.

Joins multimodel_predictions.csv with asof_team_features.csv to create
a unified dataset for analysis with:
- Actual game outcomes (scores, margins, spread covers)
- Model predictions (GLM, XGBoost, state-space, ensemble)
- Rich feature set (105+ features: EPA, rest, QB changes, etc.)

Usage:
    python py/data/load_real_data.py --output data/processed/merged_predictions_features.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load multimodel predictions.

    Columns: game_id, season, week, actual, glm, xgb, state, ens_stack_glm_xgb_state
    """
    df = pd.read_csv(predictions_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "actual": "home_win_actual",
            "glm": "prob_glm",
            "xgb": "prob_xgb",
            "state": "prob_state",
            "ens_stack_glm_xgb_state": "prob_ensemble",
        }
    )

    # Keep only key columns
    cols_keep = [
        "game_id",
        "season",
        "week",
        "home_win_actual",
        "prob_glm",
        "prob_xgb",
        "prob_state",
        "prob_ensemble",
    ]

    return df[cols_keep]


def load_features(features_path: Path) -> pd.DataFrame:
    """
    Load game features with outcomes.

    Key columns: game_id, home_score, away_score, spread_close, home_margin, home_cover
    """
    df = pd.read_csv(features_path)

    # Ensure margin column exists
    if "home_margin" not in df.columns:
        df["home_margin"] = df["home_score"] - df["away_score"]

    # Ensure cover column exists (home covered spread?)
    if "home_cover" not in df.columns and "spread_close" in df.columns:
        df["home_cover"] = (df["home_margin"] + df["spread_close"]) > 0

    return df


def merge_predictions_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge predictions with features on game_id.

    Returns unified DataFrame with predictions, outcomes, and features.
    """
    merged = predictions.merge(features, on="game_id", how="inner", suffixes=("_pred", "_feat"))

    # Reconcile duplicate columns (season, week may appear in both)
    if "season_pred" in merged.columns and "season_feat" in merged.columns:
        merged["season"] = merged["season_pred"].fillna(merged["season_feat"])
        merged = merged.drop(columns=["season_pred", "season_feat"])

    if "week_pred" in merged.columns and "week_feat" in merged.columns:
        merged["week"] = merged["week_pred"].fillna(merged["week_feat"])
        merged = merged.drop(columns=["week_pred", "week_feat"])

    return merged


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for analysis."""

    # Convert probabilities to implied margins (rough approximation)
    # For spread betting, P(cover) ≈ 0.5 implies fair line
    # deviation from 0.5 suggests edge
    df["ensemble_edge"] = df["prob_ensemble"] - 0.5

    # Prediction error (for ensemble)
    df["ensemble_error"] = np.abs(df["home_win_actual"] - df["prob_ensemble"])

    # Brier score contribution (squared error)
    df["brier_contrib"] = (df["home_win_actual"] - df["prob_ensemble"]) ** 2

    # Predicted margin from ensemble probability (inverse logit approximation)
    # logit(p) ≈ margin / 14 (rough NFL calibration)
    # margin ≈ 14 * log(p / (1-p))
    p_clip = np.clip(df["prob_ensemble"], 0.01, 0.99)
    df["predicted_margin"] = 14 * np.log(p_clip / (1 - p_clip))

    return df


def validate_merge(df: pd.DataFrame) -> None:
    """Validate merged data quality."""

    print("\n" + "=" * 60)
    print("MERGE VALIDATION")
    print("=" * 60)

    # Check record count
    print(f"Total games: {len(df):,}")
    print(f"Seasons: {df['season'].min()}-{df['season'].max()}")
    print("Missing values:")
    missing = df[["home_score", "away_score", "spread_close", "prob_ensemble"]].isna().sum()
    print(missing)

    # Check Brier score (should match dissertation: 0.2515)
    brier = df["brier_contrib"].mean()
    print(f"\nBrier score: {brier:.4f} (expected ≈ 0.2515)")

    # Check ATS win rate
    ats_wins = df["home_cover"].sum()
    ats_rate = ats_wins / len(df)
    print(f"ATS win rate: {ats_rate:.1%} ({ats_wins:,}/{len(df):,})")

    # Check distribution of predictions
    print("\nEnsemble probability distribution:")
    print(df["prob_ensemble"].describe())

    # Check margin correlation
    if "home_margin" in df.columns and "predicted_margin" in df.columns:
        corr = df[["home_margin", "predicted_margin"]].corr().iloc[0, 1]
        print(f"\nMargin correlation (predicted vs actual): {corr:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge real predictions with features and outcomes"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("analysis/results/multimodel_predictions.csv"),
        help="Path to multimodel predictions CSV",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features/asof_team_features.csv"),
        help="Path to game features CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/merged_predictions_features.csv"),
        help="Output path for merged CSV",
    )
    parser.add_argument(
        "--include-all-features",
        action="store_true",
        help="Include all 105+ features (default: keep only key columns)",
    )

    args = parser.parse_args()

    print("Loading predictions...")
    predictions = load_predictions(args.predictions)
    print(f"✓ Loaded {len(predictions):,} prediction records")

    print("\nLoading features...")
    features = load_features(args.features)
    print(f"✓ Loaded {len(features):,} feature records")

    print("\nMerging datasets...")
    merged = merge_predictions_features(predictions, features)
    print(f"✓ Merged {len(merged):,} games")

    print("\nComputing derived metrics...")
    merged = compute_derived_metrics(merged)

    validate_merge(merged)

    # Select columns to export
    if not args.include_all_features:
        # Core columns for analysis
        cols_export = [
            "game_id",
            "season",
            "week",
            "kickoff",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_margin",
            "spread_close",
            "total_close",
            "home_cover",
            "prob_glm",
            "prob_xgb",
            "prob_state",
            "prob_ensemble",
            "predicted_margin",
            "ensemble_edge",
            "ensemble_error",
            "brier_contrib",
            # Key features for explainability
            "home_prior_epa_mean",
            "away_prior_epa_mean",
            "prior_epa_mean_diff",
            "home_epa_pp_last3",
            "away_epa_pp_last3",
            "epa_pp_last3_diff",
            "home_prior_margin_avg",
            "away_prior_margin_avg",
            "prior_margin_avg_diff",
            "home_rest_days",
            "away_rest_days",
            "rest_days_diff",
            "home_qb_change",
            "away_qb_change",
            "qb_change_diff",
            "home_prev_result",
            "away_prev_result",
        ]
        # Filter to available columns
        cols_export = [c for c in cols_export if c in merged.columns]
        merged_export = merged[cols_export]
    else:
        merged_export = merged

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged_export.to_csv(args.output, index=False)
    print(
        f"\n✓ Wrote {len(merged_export):,} rows × {len(merged_export.columns)} columns to {args.output}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
