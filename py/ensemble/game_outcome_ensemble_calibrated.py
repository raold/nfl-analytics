#!/usr/bin/env python3
"""
Calibrated Game Outcome Ensemble with Isotonic Regression

Extends the OptimizedGameOutcomeEnsemble with probability calibration to fix
overconfident predictions.

Key Innovation:
- Adds isotonic regression calibrator to meta-learner output
- Fits calibrator on validation set (avoids overfitting)
- Transforms predictions to match actual observed frequencies

Problem Solved:
- Uncalibrated model might predict 0.95 when true probability is 0.70
- Calibration adjusts predictions to be more realistic
- Better Expected Calibration Error (ECE)
- More reliable betting decisions

Author: Claude Code
Date: 2025-10-30
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.ensemble.game_outcome_ensemble_optimized import OptimizedGameOutcomeEnsemble


class CalibratedGameOutcomeEnsemble(OptimizedGameOutcomeEnsemble):
    """
    Optimized ensemble with isotonic regression calibration.

    Architecture:
    1-4. Same as OptimizedGameOutcomeEnsemble (State-Space, XGBoost, Bayesian LR, Neural Net)
    5. Meta-Learner (Stacking) with regularization
    6. 🆕 Isotonic Calibrator - transforms meta-learner output to calibrated probabilities

    Calibration Process:
    - Train base models on training set
    - Generate predictions on validation set
    - Fit isotonic regression: calibrated_prob = f(uncalibrated_prob)
    - Apply calibration to all future predictions

    Benefits:
    - Predictions match observed frequencies (reliability)
    - Reduced Expected Calibration Error (ECE)
    - Better risk management through honest uncertainty
    """

    def __init__(
        self,
        state_space_config: dict | None = None,
        xgb_params: dict | None = None,
        nn_params: dict | None = None,
    ):
        super().__init__(state_space_config, xgb_params, nn_params)

        # Isotonic regression calibrator (non-parametric, monotonic)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_calibrated = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Fit all components + calibrator.

        Args:
            train_df: Training games for base models
            val_df: Validation games for calibrator fitting (REQUIRED for calibration)

        Returns:
            Dictionary of metrics for each component (including calibrated ensemble)
        """
        if val_df is None:
            print("\n⚠️  WARNING: No validation set provided - calibration will NOT be applied")
            print("  Calibrator requires validation set to avoid overfitting")
            print("  Falling back to uncalibrated predictions\n")
            return super().fit(train_df, val_df)

        print(f"\n{'='*80}")
        print("TRAINING CALIBRATED 5-COMPONENT ENSEMBLE")
        print("(Optimized Features + Strong Regularization + Isotonic Calibration)")
        print(f"{'='*80}")
        print(f"Train: {len(train_df)} games")
        print(f"Val:   {len(val_df)} games (for calibration)")

        # Fit base ensemble on training set
        metrics = super().fit(train_df, val_df)

        # ====================================================================
        # Component 6: Isotonic Calibrator (on validation set)
        # ====================================================================
        print(f"\n{'-'*80}")
        print("[6/6] Fitting Isotonic Calibrator (Validation Set)")
        print(f"{'-'*80}")

        # Generate uncalibrated predictions on validation set
        val_target = (val_df["margin"] > 0).astype(int).values
        uncalibrated_preds, _ = super().predict(val_df)

        # Fit isotonic regression to map uncalibrated -> calibrated
        self.calibrator.fit(uncalibrated_preds, val_target)
        self.is_calibrated = True

        # Generate calibrated predictions
        calibrated_preds = self.calibrator.transform(uncalibrated_preds)

        # Calculate calibrated metrics
        calibrated_metrics = {
            "brier": float(brier_score_loss(val_target, calibrated_preds)),
            "logloss": float(log_loss(val_target, calibrated_preds)),
            "accuracy": float(np.mean((calibrated_preds >= 0.5) == val_target)),
        }

        # Compare uncalibrated vs calibrated
        uncalibrated_metrics = {
            "brier": float(brier_score_loss(val_target, uncalibrated_preds)),
            "logloss": float(log_loss(val_target, uncalibrated_preds)),
            "accuracy": float(np.mean((uncalibrated_preds >= 0.5) == val_target)),
        }

        print("  BEFORE Calibration:")
        print(f"    Val Brier: {uncalibrated_metrics['brier']:.4f}")
        print(f"    Val LogLoss: {uncalibrated_metrics['logloss']:.4f}")
        print(f"    Val Accuracy: {uncalibrated_metrics['accuracy']:.1%}")

        print("\n  AFTER Calibration:")
        print(f"    Val Brier: {calibrated_metrics['brier']:.4f}")
        print(f"    Val LogLoss: {calibrated_metrics['logloss']:.4f}")
        print(f"    Val Accuracy: {calibrated_metrics['accuracy']:.1%}")

        # Calculate improvement
        brier_improvement = uncalibrated_metrics['brier'] - calibrated_metrics['brier']
        logloss_improvement = uncalibrated_metrics['logloss'] - calibrated_metrics['logloss']

        print("\n  Calibration Impact:")
        print(f"    Brier Improvement: {brier_improvement:+.4f} ({brier_improvement/uncalibrated_metrics['brier']:+.1%})")
        print(f"    LogLoss Improvement: {logloss_improvement:+.4f} ({logloss_improvement/uncalibrated_metrics['logloss']:+.1%})")

        if brier_improvement < 0:
            print("    ⚠️  Warning: Calibration increased Brier score (may indicate insufficient validation data)")
        else:
            print("    ✓ Calibration improved probability estimates")

        print(f"\n{'='*80}")
        print("CALIBRATED TRAINING COMPLETE")
        print(f"{'='*80}\n")

        # Add calibrated metrics to return dict
        metrics["calibrated_ensemble"] = calibrated_metrics
        metrics["uncalibrated_ensemble"] = uncalibrated_metrics

        return metrics

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Generate calibrated ensemble predictions.

        Args:
            df: Games DataFrame

        Returns:
            (calibrated_ensemble_predictions, component_predictions_dict)

        Note: If calibrator not fitted, returns uncalibrated predictions
        """
        # Get uncalibrated ensemble predictions
        uncalibrated_preds, component_preds = super().predict(df)

        if not self.is_calibrated:
            print("⚠️  Calibrator not fitted - returning uncalibrated predictions")
            return uncalibrated_preds, component_preds

        # Apply calibration
        calibrated_preds = self.calibrator.transform(uncalibrated_preds)

        # Add both to component dict for comparison
        component_preds["uncalibrated_ensemble"] = uncalibrated_preds
        component_preds["calibrated_ensemble"] = calibrated_preds

        return calibrated_preds, component_preds

    def evaluate(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Evaluate all components including calibrated ensemble on test set.

        Returns metrics for:
        - Individual components (state_space, xgboost, bayesian_lr, neural_net)
        - Uncalibrated ensemble
        - Calibrated ensemble
        """
        target = (df["margin"] > 0).astype(int).values
        calibrated_preds, component_preds = self.predict(df)

        metrics = {}

        # Calibrated ensemble metrics (main result)
        metrics["calibrated_ensemble"] = {
            "brier": float(brier_score_loss(target, calibrated_preds)),
            "logloss": float(log_loss(target, calibrated_preds)),
            "accuracy": float(np.mean((calibrated_preds >= 0.5) == target)),
        }

        # Uncalibrated ensemble metrics (for comparison)
        if "uncalibrated_ensemble" in component_preds:
            uncalibrated_preds = component_preds["uncalibrated_ensemble"]
            metrics["uncalibrated_ensemble"] = {
                "brier": float(brier_score_loss(target, uncalibrated_preds)),
                "logloss": float(log_loss(target, uncalibrated_preds)),
                "accuracy": float(np.mean((uncalibrated_preds >= 0.5) == target)),
            }

        # Component metrics (exclude ensemble predictions from component_preds)
        for name, preds in component_preds.items():
            if name not in ["uncalibrated_ensemble", "calibrated_ensemble"]:
                metrics[name] = {
                    "brier": float(brier_score_loss(target, preds)),
                    "logloss": float(log_loss(target, preds)),
                    "accuracy": float(np.mean((preds >= 0.5) == target)),
                }

        return metrics


if __name__ == "__main__":
    # Example usage
    print("CalibratedGameOutcomeEnsemble")
    print("=" * 80)
    print("This module provides a calibrated version of the optimized ensemble.")
    print("\nUsage:")
    print("  from py.ensemble.game_outcome_ensemble_calibrated import CalibratedGameOutcomeEnsemble")
    print("")
    print("  ensemble = CalibratedGameOutcomeEnsemble()")
    print("  ensemble.fit(train_df, val_df)  # val_df required for calibration")
    print("  predictions, components = ensemble.predict(test_df)")
    print("\nFeatures:")
    print("  - Isotonic regression calibration")
    print("  - Fixes overconfident predictions")
    print("  - Improves Expected Calibration Error (ECE)")
    print("  - Better risk management for betting")
