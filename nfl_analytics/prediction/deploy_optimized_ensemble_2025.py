"""
Deploy Optimized Ensemble for 2025 Season Predictions

Implements the production-ready optimized configuration:
- Increased regularization (alpha=5.0, lambda=5.0)
- Only beneficial Phase 5 features (away_short_rest, is_conference_game)
- Platt Scaling calibration
- Conservative betting criteria (5% min edge, 10% max Kelly)

Author: Claude Code
Date: 2025-10-30
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.ensemble.game_outcome_ensemble_realistic import RealisticGameOutcomeEnsemble
from py.models.state_space import fetch_games


class PlattScalingCalibrator:
    """Platt Scaling calibration for ensemble probabilities."""

    def __init__(self):
        self.calibrator = LogisticRegression()

    def fit(self, probabilities: np.ndarray, targets: np.ndarray):
        """
        Fit Platt Scaling calibrator.

        Args:
            probabilities: Uncalibrated probabilities (N,)
            targets: True binary labels (N,)
        """
        X = probabilities.reshape(-1, 1)
        self.calibrator.fit(X, targets)

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply Platt Scaling calibration.

        Args:
            probabilities: Uncalibrated probabilities (N,)

        Returns:
            Calibrated probabilities (N,)
        """
        X = probabilities.reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]


def create_optimized_ensemble() -> RealisticGameOutcomeEnsemble:
    """
    Create optimized ensemble with production settings.

    Returns:
        RealisticGameOutcomeEnsemble configured with optimal hyperparameters
    """
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # OPTIMIZED regularization
        "alpha": 5.0,  # L1 regularization (increased from 2.0)
        "lambda": 5.0,  # L2 regularization (increased from 2.0)
        "min_child_weight": 5,
        "gamma": 0.5,
        "n_jobs": -1,
    }

    ensemble = RealisticGameOutcomeEnsemble(xgb_params=xgb_params)

    # OPTIMIZED feature set: only beneficial Phase 5 features
    ensemble.legitimate_phase5_features = [
        "away_short_rest",  # Provides measurable benefit
        "is_conference_game",  # Minimal but positive benefit
    ]

    return ensemble


def apply_betting_criteria(
    predictions: pd.DataFrame,
    min_edge: float = 0.05,
    max_kelly_fraction: float = 0.10,
    prob_clip_low: float = 0.15,
    prob_clip_high: float = 0.85,
) -> pd.DataFrame:
    """
    Apply conservative betting criteria to predictions.

    Args:
        predictions: DataFrame with calibrated probabilities
        min_edge: Minimum edge threshold (default 5%)
        max_kelly_fraction: Maximum Kelly fraction (default 10%)
        prob_clip_low: Lower probability clip (default 15%)
        prob_clip_high: Upper probability clip (default 85%)

    Returns:
        DataFrame with betting recommendations
    """
    # Clip probabilities
    predictions["home_win_prob_clipped"] = predictions["home_win_prob_calibrated"].clip(
        prob_clip_low, prob_clip_high
    )

    # Calculate edge (assuming fair odds for simplicity)
    predictions["implied_prob"] = 0.5  # Placeholder - would use actual market odds
    predictions["edge"] = predictions["home_win_prob_clipped"] - predictions["implied_prob"]

    # Calculate Kelly fraction
    predictions["kelly_fraction"] = (
        predictions["home_win_prob_clipped"] * 2 - 1
    )  # For even money bets
    predictions["kelly_fraction"] = predictions["kelly_fraction"].clip(0, max_kelly_fraction)

    # Betting recommendation
    predictions["bet_recommended"] = (predictions["edge"].abs() >= min_edge) & (
        predictions["kelly_fraction"] > 0
    )

    # Betting side
    predictions["bet_side"] = "none"
    predictions.loc[predictions["bet_recommended"] & (predictions["edge"] > 0), "bet_side"] = "home"
    predictions.loc[predictions["bet_recommended"] & (predictions["edge"] < 0), "bet_side"] = "away"

    return predictions


def main():
    """Deploy optimized ensemble for 2025 predictions."""
    print("=" * 80)
    print("OPTIMIZED ENSEMBLE DEPLOYMENT - 2025 SEASON")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # STEP 1: Load data
    # -------------------------------------------------------------------------
    print("STEP 1: Loading data...")

    # Training: 2019-2023 (5 years)
    train_seasons = [2019, 2020, 2021, 2022, 2023]
    # Calibration: 2024 (most recent complete season)
    calibration_season = [2024]
    # Prediction: 2025 (current season)
    prediction_season = [2025]

    all_seasons = train_seasons + calibration_season + prediction_season
    df_all = fetch_games(all_seasons)

    train_df = df_all[df_all["season"].isin(train_seasons)]
    calibration_df = df_all[df_all["season"].isin(calibration_season)]
    prediction_df = df_all[df_all["season"].isin(prediction_season)]

    print(f"  Train:       {len(train_df):4d} games ({train_seasons[0]}-{train_seasons[-1]})")
    print(f"  Calibration: {len(calibration_df):4d} games ({calibration_season[0]})")
    print(f"  Prediction:  {len(prediction_df):4d} games ({prediction_season[0]})")
    print()

    # -------------------------------------------------------------------------
    # STEP 2: Train optimized ensemble
    # -------------------------------------------------------------------------
    print("STEP 2: Training optimized ensemble...")
    print("  Configuration:")
    print("    - XGBoost: alpha=5.0, lambda=5.0")
    print("    - Phase 5 features: away_short_rest, is_conference_game")
    print()

    ensemble = create_optimized_ensemble()
    train_metrics = ensemble.fit(train_df)

    print(f"  Train accuracy: {train_metrics['ensemble']['accuracy']:.1%}")
    print()

    # -------------------------------------------------------------------------
    # STEP 3: Calibrate with Platt Scaling
    # -------------------------------------------------------------------------
    print("STEP 3: Calibrating with Platt Scaling...")

    # Get uncalibrated predictions on calibration set
    calibration_target = (calibration_df["margin"] > 0).astype(int).values
    calibration_preds, _ = ensemble.predict(calibration_df)

    # Fit calibrator
    calibrator = PlattScalingCalibrator()
    calibrator.fit(calibration_preds, calibration_target)

    print(f"  Calibrator fitted on {len(calibration_df)} games")
    print()

    # -------------------------------------------------------------------------
    # STEP 4: Generate 2025 predictions
    # -------------------------------------------------------------------------
    print("STEP 4: Generating 2025 predictions...")

    # Get uncalibrated predictions
    prediction_preds_uncalibrated, _ = ensemble.predict(prediction_df)

    # Apply calibration
    prediction_preds_calibrated = calibrator.transform(prediction_preds_uncalibrated)

    # Create predictions DataFrame
    predictions = pd.DataFrame(
        {
            "week": prediction_df["week"].values,
            "home_team": prediction_df["home_team"].values,
            "away_team": prediction_df["away_team"].values,
            "home_win_prob_uncalibrated": prediction_preds_uncalibrated,
            "home_win_prob_calibrated": prediction_preds_calibrated,
        }
    )

    # Apply betting criteria
    predictions = apply_betting_criteria(
        predictions,
        min_edge=0.05,  # 5% minimum edge
        max_kelly_fraction=0.10,  # 10% max Kelly
        prob_clip_low=0.15,  # Clip at 15%
        prob_clip_high=0.85,  # Clip at 85%
    )

    print(f"  Generated {len(predictions)} predictions for 2025 season")
    print()

    # -------------------------------------------------------------------------
    # STEP 5: Summary statistics
    # -------------------------------------------------------------------------
    print("STEP 5: Prediction Summary")
    print("=" * 80)

    # Overall distribution
    print("\nProbability Distribution (Calibrated):")
    print(predictions["home_win_prob_calibrated"].describe())

    # Betting recommendations
    n_bets = predictions["bet_recommended"].sum()
    print("\nBetting Recommendations:")
    print(f"  Total games:        {len(predictions)}")
    print(f"  Bets recommended:   {n_bets} ({n_bets/len(predictions):.1%})")

    if n_bets > 0:
        print("\nBetting Breakdown:")
        print(f"  Home bets: {(predictions['bet_side'] == 'home').sum()}")
        print(f"  Away bets: {(predictions['bet_side'] == 'away').sum()}")
        print(
            f"\n  Average edge:        {predictions[predictions['bet_recommended']]['edge'].abs().mean():.1%}"
        )
        print(
            f"  Average Kelly:       {predictions[predictions['bet_recommended']]['kelly_fraction'].mean():.1%}"
        )

    # By week
    print("\nPredictions by Week:")
    weekly_summary = predictions.groupby("week").agg(
        {
            "home_win_prob_calibrated": "mean",
            "bet_recommended": "sum",
        }
    )
    print(weekly_summary)

    # -------------------------------------------------------------------------
    # STEP 6: Save outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: Saving outputs...")

    output_dir = Path("predictions/2025")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full predictions
    predictions_file = output_dir / "optimized_ensemble_predictions_2025.csv"
    predictions.to_csv(predictions_file, index=False)
    print(f"  ✓ Predictions saved to {predictions_file}")

    # Save recommended bets only
    if n_bets > 0:
        bets_file = output_dir / "recommended_bets_2025.csv"
        recommended_bets = predictions[predictions["bet_recommended"]].copy()
        recommended_bets.to_csv(bets_file, index=False)
        print(f"  ✓ Recommended bets saved to {bets_file}")

    # Save metadata
    metadata = {
        "deployment_date": "2025-10-30",
        "train_seasons": train_seasons,
        "calibration_season": calibration_season,
        "prediction_season": prediction_season,
        "n_train_games": len(train_df),
        "n_calibration_games": len(calibration_df),
        "n_prediction_games": len(predictions),
        "n_recommended_bets": int(n_bets),
        "configuration": {
            "xgb_alpha": 5.0,
            "xgb_lambda": 5.0,
            "phase5_features": ["away_short_rest", "is_conference_game"],
            "min_edge": 0.05,
            "max_kelly_fraction": 0.10,
            "prob_clip": [0.15, 0.85],
        },
        "train_accuracy": train_metrics["ensemble"]["accuracy"],
    }

    metadata_file = output_dir / "deployment_metadata_2025.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to {metadata_file}")

    print("\n" + "=" * 80)
    print("DEPLOYMENT COMPLETE")
    print("=" * 80)
    print("\nOptimized ensemble successfully deployed for 2025 season!")
    print(f"Review predictions at: {predictions_file}")


if __name__ == "__main__":
    main()
