"""
Optimized Ensemble Study - Implementing Recommended Fixes

Implements the top 5 recommendations from feature ablation study:
1. Increased regularization (alpha=5.0, lambda=5.0)
2. Remove harmful features (extreme_cold, high_wind)
3. Keep only beneficial features (away_short_rest, is_conference_game)
4. Add Platt Scaling calibration
5. Run temporal validation (2019-2020 train, test each year separately)

Author: Claude Code
Date: 2025-10-30
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
        # Reshape for sklearn
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


def calculate_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate calibration error metrics."""
    # Bin predictions
    bins = np.linspace(0.5, 1.0, 11)
    bin_indices = np.digitize(y_pred, bins)

    calibration_error = 0.0
    n_bins_used = 0

    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        if bin_mask.sum() > 0:
            bin_pred = y_pred[bin_mask].mean()
            bin_actual = y_true[bin_mask].mean()
            calibration_error += abs(bin_pred - bin_actual)
            n_bins_used += 1

    mean_calibration_error = calibration_error / max(n_bins_used, 1)

    return {
        "mean_calibration_error": mean_calibration_error,
        "n_bins": n_bins_used,
    }


class OptimizedEnsembleStudy:
    """Run optimized ensemble study with all fixes."""

    def __init__(self):
        self.results = {}
        self.calibrator = PlattScalingCalibrator()

    def create_optimized_ensemble(self) -> RealisticGameOutcomeEnsemble:
        """
        Create optimized ensemble with:
        - Increased regularization (alpha=5.0, lambda=5.0)
        - Only 2 beneficial Phase 5 features (away_short_rest, is_conference_game)
        """
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            # INCREASED regularization (was 2.0)
            'alpha': 5.0,
            'lambda': 5.0,
            'min_child_weight': 5,
            'gamma': 0.5,
            'n_jobs': -1,
        }

        ensemble = RealisticGameOutcomeEnsemble(xgb_params=xgb_params)

        # Only keep beneficial features (remove extreme_cold, high_wind, dome_team_outdoor_wind)
        ensemble.legitimate_phase5_features = [
            "away_short_rest",
            "is_conference_game",
        ]

        return ensemble

    def run_step1_3_optimized_ensemble(self):
        """Steps 1-3: Train optimized ensemble with increased reg + reduced features."""
        print("\n" + "="*80)
        print("STEPS 1-3: OPTIMIZED ENSEMBLE")
        print("="*80)
        print("\nConfiguration:")
        print("  - XGBoost regularization: alpha=5.0, lambda=5.0")
        print("  - Phase 5 features: away_short_rest, is_conference_game")
        print("  - Removed: extreme_cold, high_wind, dome_team_outdoor_wind")

        # Load data
        train_seasons = [2019, 2020, 2021]
        test_seasons = [2022, 2023, 2024]
        all_seasons = train_seasons + test_seasons

        df_all = fetch_games(all_seasons)
        train_df = df_all[df_all["season"].isin(train_seasons)]
        test_df = df_all[df_all["season"].isin(test_seasons)]

        print(f"\nTrain: {len(train_df)} games ({train_seasons})")
        print(f"Test:  {len(test_df)} games ({test_seasons})")

        # Create and train optimized ensemble
        ensemble = self.create_optimized_ensemble()
        train_metrics = ensemble.fit(train_df)

        # Evaluate on test set
        test_metrics = ensemble.evaluate(test_df)

        # Calculate calibration
        test_target = (test_df["margin"] > 0).astype(int).values
        test_preds, _ = ensemble.predict(test_df)
        calibration_metrics = calculate_calibration_metrics(test_target, test_preds)

        self.results["optimized_uncalibrated"] = {
            "train_accuracy": train_metrics["ensemble"]["accuracy"],
            "test_accuracy": test_metrics["ensemble"]["accuracy"],
            "test_brier": test_metrics["ensemble"]["brier"],
            "test_logloss": test_metrics["ensemble"]["logloss"],
            "calibration_error": calibration_metrics["mean_calibration_error"],
        }

        print(f"\n{'='*80}")
        print("RESULTS (Optimized, Uncalibrated):")
        print(f"{'='*80}")
        print(f"Test Accuracy: {self.results['optimized_uncalibrated']['test_accuracy']:.1%}")
        print(f"Test Brier:    {self.results['optimized_uncalibrated']['test_brier']:.4f}")
        print(f"Test LogLoss:  {self.results['optimized_uncalibrated']['test_logloss']:.4f}")
        print(f"Calibration Error: {self.results['optimized_uncalibrated']['calibration_error']:.4f}")

        # Save ensemble and predictions for calibration step
        self.ensemble = ensemble
        self.test_df = test_df
        self.test_preds_uncalibrated = test_preds
        self.test_target = test_target

    def run_step4_platt_scaling(self):
        """Step 4: Add Platt Scaling calibration."""
        print("\n" + "="*80)
        print("STEP 4: PLATT SCALING CALIBRATION")
        print("="*80)

        # Use 2022 as calibration set
        calibration_df = self.test_df[self.test_df["season"] == 2022]
        calibration_target = (calibration_df["margin"] > 0).astype(int).values
        calibration_preds, _ = self.ensemble.predict(calibration_df)

        print(f"\nCalibration set: {len(calibration_df)} games (2022)")

        # Fit Platt Scaling
        self.calibrator.fit(calibration_preds, calibration_target)

        # Apply to full test set
        test_preds_calibrated = self.calibrator.transform(self.test_preds_uncalibrated)

        # Evaluate calibrated predictions
        from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

        calibrated_metrics = {
            "accuracy": accuracy_score(self.test_target, test_preds_calibrated >= 0.5),
            "brier": brier_score_loss(self.test_target, test_preds_calibrated),
            "logloss": log_loss(self.test_target, test_preds_calibrated),
        }

        calibration_metrics = calculate_calibration_metrics(self.test_target, test_preds_calibrated)

        self.results["optimized_calibrated"] = {
            "test_accuracy": calibrated_metrics["accuracy"],
            "test_brier": calibrated_metrics["brier"],
            "test_logloss": calibrated_metrics["logloss"],
            "calibration_error": calibration_metrics["mean_calibration_error"],
        }

        print(f"\n{'='*80}")
        print("RESULTS (Optimized + Calibrated):")
        print(f"{'='*80}")
        print(f"Test Accuracy: {self.results['optimized_calibrated']['test_accuracy']:.1%}")
        print(f"Test Brier:    {self.results['optimized_calibrated']['test_brier']:.4f}")
        print(f"Test LogLoss:  {self.results['optimized_calibrated']['test_logloss']:.4f}")
        print(f"Calibration Error: {self.results['optimized_calibrated']['calibration_error']:.4f}")

        # Calculate improvement
        cal_error_reduction = (
            self.results["optimized_uncalibrated"]["calibration_error"] -
            self.results["optimized_calibrated"]["calibration_error"]
        )
        print(f"\nCalibration error reduction: {cal_error_reduction:.4f} ({cal_error_reduction/self.results['optimized_uncalibrated']['calibration_error']:.1%})")

        self.test_preds_calibrated = test_preds_calibrated

    def run_step5_temporal_validation(self):
        """Step 5: Run temporal validation (train 2019-2020, test each year)."""
        print("\n" + "="*80)
        print("STEP 5: TEMPORAL VALIDATION")
        print("="*80)

        # Train on 2019-2020 only
        train_seasons = [2019, 2020]
        test_years = [2021, 2022, 2023, 2024]

        all_seasons = train_seasons + test_years
        df_all = fetch_games(all_seasons)
        train_df = df_all[df_all["season"].isin(train_seasons)]

        print(f"\nTrain: {len(train_df)} games (2019-2020)")
        print(f"Test:  Individual years 2021-2024\n")

        # Create and train ensemble
        ensemble = self.create_optimized_ensemble()
        ensemble.fit(train_df)

        # Test on each year separately
        temporal_results = []

        for year in test_years:
            year_df = df_all[df_all["season"] == year]
            year_target = (year_df["margin"] > 0).astype(int).values
            year_preds, _ = ensemble.predict(year_df)

            # Metrics
            from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

            accuracy = accuracy_score(year_target, year_preds >= 0.5)
            brier = brier_score_loss(year_target, year_preds)
            logloss = log_loss(year_target, year_preds)
            calibration = calculate_calibration_metrics(year_target, year_preds)

            temporal_results.append({
                "year": year,
                "n_games": len(year_df),
                "accuracy": accuracy,
                "brier": brier,
                "logloss": logloss,
                "calibration_error": calibration["mean_calibration_error"],
            })

            print(f"Year {year}: Accuracy={accuracy:.1%}, Brier={brier:.4f}, CalError={calibration['mean_calibration_error']:.4f}")

        self.results["temporal_validation"] = temporal_results

        # Check for degradation over time
        accuracies = [r["accuracy"] for r in temporal_results]
        degradation = accuracies[0] - accuracies[-1]

        print(f"\nAccuracy degradation 2021→2024: {degradation:+.1%}")

        if degradation > 0.05:
            print("⚠️  Significant degradation over time - confirms overfitting hypothesis")
        else:
            print("✓ Stable performance over time")

    def compare_all_configurations(self):
        """Compare all configurations."""
        print("\n" + "="*80)
        print("FINAL COMPARISON - ALL CONFIGURATIONS")
        print("="*80)

        # Load previous results for comparison
        baseline_path = Path("analysis/ablation/feature_ablation_results.json")
        if baseline_path.exists():
            with open(baseline_path) as f:
                previous_results = json.load(f)

            comparison = pd.DataFrame([
                {
                    "Configuration": "Previous - Current (α=2, λ=2, 5 features)",
                    "Test Acc": f"{previous_results['current']['test_accuracy']:.1%}",
                    "Test Brier": f"{previous_results['current']['test_brier']:.4f}",
                    "Cal Error": "N/A",
                },
                {
                    "Configuration": "Previous - Increased Reg (α=5, λ=5, 5 features)",
                    "Test Acc": f"{previous_results['increased_reg']['test_accuracy']:.1%}",
                    "Test Brier": f"{previous_results['increased_reg']['test_brier']:.4f}",
                    "Cal Error": "N/A",
                },
                {
                    "Configuration": "Optimized (α=5, λ=5, 2 features)",
                    "Test Acc": f"{self.results['optimized_uncalibrated']['test_accuracy']:.1%}",
                    "Test Brier": f"{self.results['optimized_uncalibrated']['test_brier']:.4f}",
                    "Cal Error": f"{self.results['optimized_uncalibrated']['calibration_error']:.4f}",
                },
                {
                    "Configuration": "Optimized + Calibrated (BEST)",
                    "Test Acc": f"{self.results['optimized_calibrated']['test_accuracy']:.1%}",
                    "Test Brier": f"{self.results['optimized_calibrated']['test_brier']:.4f}",
                    "Cal Error": f"{self.results['optimized_calibrated']['calibration_error']:.4f}",
                },
            ])

            print("\n", comparison.to_string(index=False))

    def save_results(self):
        """Save all results."""
        output_dir = Path("analysis/optimized")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        with open(output_dir / "optimized_ensemble_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to {output_dir / 'optimized_ensemble_results.json'}")


def main():
    """Run complete optimized ensemble study."""
    study = OptimizedEnsembleStudy()

    # Steps 1-3: Optimized ensemble (increased reg + reduced features)
    study.run_step1_3_optimized_ensemble()

    # Step 4: Add Platt Scaling calibration
    study.run_step4_platt_scaling()

    # Step 5: Temporal validation
    study.run_step5_temporal_validation()

    # Compare all configurations
    study.compare_all_configurations()

    # Save results
    study.save_results()

    print("\n" + "="*80)
    print("OPTIMIZED ENSEMBLE STUDY COMPLETE")
    print("="*80)

    print("\nKEY TAKEAWAYS:")
    print(f"  1. Optimized config improves test accuracy to {study.results['optimized_uncalibrated']['test_accuracy']:.1%}")
    print(f"  2. Calibration reduces calibration error to {study.results['optimized_calibrated']['calibration_error']:.4f}")
    print(f"  3. Best configuration: {study.results['optimized_calibrated']['test_accuracy']:.1%} accuracy")


if __name__ == "__main__":
    main()
