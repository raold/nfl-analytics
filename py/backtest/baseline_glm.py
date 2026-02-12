#!/usr/bin/env python3
"""
Baseline logistic regression for NFL spread prediction with reliability diagrams.

Trains a simple logistic regression model using walk-forward validation and
generates calibration/reliability plots showing predicted vs observed frequencies.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

# Default features for spread prediction
DEFAULT_FEATURES = [
    "prior_epa_mean_diff",
    "epa_pp_last3_diff",
    "rest_diff",
    "season_win_pct_diff",
    "win_pct_last5_diff",
    "prior_margin_avg_diff",
    "points_for_last3_diff",
    "points_against_last3_diff",
]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load features CSV and prepare for modeling."""
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if "home_cover" not in df.columns:
        raise ValueError("CSV must contain 'home_cover' column")
    if "season" not in df.columns:
        raise ValueError("CSV must contain 'season' column")

    # Drop rows with missing values in target
    df = df.dropna(subset=["home_cover"])

    return df


def train_and_predict(
    df: pd.DataFrame,
    test_season: int,
    features: list[str],
    min_season: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train on seasons before test_season, predict on test_season.

    Returns:
        (y_true, y_pred_proba) for test season
    """
    # Split data
    if min_season is not None:
        train = df[(df.season >= min_season) & (df.season < test_season)]
    else:
        train = df[df.season < test_season]
    test = df[df.season == test_season]

    if len(train) == 0:
        raise ValueError(f"No training data for season {test_season}")
    if len(test) == 0:
        raise ValueError(f"No test data for season {test_season}")

    # Extract features and target
    X_train = train[features].fillna(0).values
    y_train = train["home_cover"].values
    X_test = test[features].fillna(0).values
    y_test = test["home_cover"].values

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return y_test, y_pred_proba


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    season: int = None,
):
    """Create and save reliability diagram."""
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
    )

    # Compute metrics
    brier = brier_score_loss(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)

    # Create figure (optimized for compact display)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)

    # Plot calibration curve (enhanced visibility)
    ax.plot(
        prob_pred,
        prob_true,
        "o-",
        color="#2a6fbb",
        linewidth=2.5,
        markersize=10,
        label=f"Model (n={len(y_true)})",
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Perfect")

    # Labels and formatting (optimized for space)
    title = "Reliability Diagram"
    if season:
        title += f" ({season})"
    ax.set_title(title, fontsize=12, pad=5)
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Frequency", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9)

    # Minimize margins for better data-ink ratio
    plt.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.10)

    # Add metrics as text
    metrics_text = f"Brier: {brier:.4f}\nLogLoss: {logloss:.4f}"
    ax.text(
        0.95,
        0.05,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Save (high DPI for sharp rendering)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved reliability diagram: {output_path}")
    print(f"   Brier: {brier:.4f}, LogLoss: {logloss:.4f}, n={len(y_true)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline GLM and generate reliability diagram"
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("data/processed/features/asof_team_features.csv"),
        help="Path to features CSV",
    )
    parser.add_argument("--start-season", type=int, required=True, help="Test season to predict")
    parser.add_argument(
        "--end-season",
        type=int,
        help="End season (for compatibility, uses start-season as test season)",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        help="Minimum training season (optional, uses all prior seasons if not set)",
    )
    parser.add_argument(
        "--cal-plot", type=Path, help="Output path for calibration/reliability plot PNG"
    )
    parser.add_argument(
        "--cal-bins",
        type=int,
        default=10,
        help="Number of bins for calibration curve (default: 10)",
    )
    parser.add_argument(
        "--features", nargs="+", default=DEFAULT_FEATURES, help="Feature columns to use"
    )

    args = parser.parse_args()

    # Use start_season as the test season
    test_season = args.start_season

    # Load data
    print(f"Loading data from: {args.features_csv}")
    df = load_data(args.features_csv)
    print(f"Loaded {len(df)} games ({df.season.min()}-{df.season.max()})")

    # Verify features exist
    missing_features = [f for f in args.features if f not in df.columns]
    if missing_features:
        print(f"ERROR: Missing features: {missing_features}")
        print(f"Available features: {df.columns.tolist()}")
        return 1

    # Train and predict
    print(f"\nTraining for season {test_season}...")
    try:
        y_true, y_pred_proba = train_and_predict(df, test_season, args.features, args.min_season)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Generate reliability diagram
    if args.cal_plot:
        plot_reliability_diagram(
            y_true, y_pred_proba, args.cal_plot, n_bins=args.cal_bins, season=test_season
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
