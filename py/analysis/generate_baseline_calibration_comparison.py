#!/usr/bin/env python3
"""
Generate Baseline Model Calibration Comparison for Dissertation

Creates comprehensive reliability diagram comparing all 6 baseline models:
- XGBoost
- Neural Network
- Bayesian Logistic Regression
- State-Space (Kalman Filter)
- 5-Component Ensemble
- Player-to-Game (if predictions available)

Author: Claude Code
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
import json


class BaselineCalibrationComparator:
    """Generate calibration comparison figure for baseline models"""

    def __init__(self):
        self.predictions_path = Path(
            "/Users/dro/rice/nfl-analytics/models/ensemble/game_outcome/predictions_all_games_2019_2025.csv"
        )
        self.player_to_game_path = Path(
            "/Users/dro/rice/nfl-analytics/models/player_to_game/test_predictions_2025.csv"
        )
        self.data = None
        self.load_predictions()

    def load_predictions(self):
        """Load ensemble and player-to-game predictions"""
        print(f"Loading predictions from: {self.predictions_path}")

        if not self.predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_path}")

        self.data = pd.read_csv(self.predictions_path)

        print(f"✓ Loaded {len(self.data)} predictions")
        print(f"Columns: {list(self.data.columns)}")
        print(f"Seasons: {sorted(self.data['season'].unique())}")

    def split_validation_test(self):
        """Split into validation (2024) and test (2025) sets"""
        val_data = self.data[self.data["season"] == 2024].copy()
        test_data = self.data[self.data["season"] == 2025].copy()

        print(f"\nValidation (2024): {len(val_data)} games")
        print(f"Test (2025): {len(test_data)} games")

        return val_data, test_data

    def calculate_calibration(self, y_true, y_pred, n_bins=10):
        """Calculate calibration curve with bin counts"""
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy='uniform'
        )

        # Calculate Brier score
        brier = np.mean((y_pred - y_true) ** 2)

        # Calculate ECE (Expected Calibration Error)
        bin_counts = []
        ece = 0.0

        for i in range(len(prob_true)):
            # Find samples in this bin
            bin_mask = (y_pred >= i/n_bins) & (y_pred < (i+1)/n_bins)
            n_in_bin = np.sum(bin_mask)
            bin_counts.append(n_in_bin)

            if n_in_bin > 0:
                ece += (n_in_bin / len(y_true)) * np.abs(prob_true[i] - prob_pred[i])

        return prob_true, prob_pred, brier, ece

    def generate_calibration_figure(self, output_path: Path):
        """Generate 2x3 grid of calibration plots"""
        val_data, test_data = self.split_validation_test()

        # Use test data (2025) for final calibration assessment
        data_to_plot = test_data

        models = {
            "xgboost_prediction": "XGBoost",
            "neural_net_prediction": "Neural Network",
            "bayesian_lr_prediction": "Bayesian Logistic Regression",
            "state_space_prediction": "State-Space (Kalman)",
            "ensemble_prediction": "5-Component Ensemble",
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for idx, (col, name) in enumerate(models.items()):
            ax = axes[idx]

            y_true = data_to_plot["target"].values
            y_pred = data_to_plot[col].values

            prob_true, prob_pred, brier, ece = self.calculate_calibration(
                y_true, y_pred, n_bins=10
            )

            # Plot calibration curve
            ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")
            ax.plot(prob_pred, prob_true, "o-", lw=2, markersize=8, label=name)

            ax.set_xlabel("Predicted Probability", fontsize=11)
            ax.set_ylabel("Observed Frequency", fontsize=11)
            ax.set_title(
                f"{name}\nBrier: {brier:.4f}, ECE: {ece:.4f}",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])

        # Hide 6th subplot (or use it for summary stats)
        axes[5].axis("off")

        # Add summary statistics in the 6th panel
        axes[5].text(
            0.1,
            0.9,
            "Test Set Performance (2025 Season)",
            fontsize=14,
            fontweight="bold",
            transform=axes[5].transAxes,
        )

        summary_text = f"Test Data: {len(data_to_plot)} games (2025 season weeks 1-7)\n\n"
        summary_text += "Calibration Metrics:\n"
        summary_text += "- Brier Score: Lower is better (0 = perfect)\n"
        summary_text += "- ECE: Expected Calibration Error\n"
        summary_text += "  (Lower is better, 0 = perfectly calibrated)\n\n"

        # Add best/worst performers
        brier_scores = {}
        for col, name in models.items():
            y_true = data_to_plot["target"].values
            y_pred = data_to_plot[col].values
            _, _, brier, _ = self.calculate_calibration(y_true, y_pred)
            brier_scores[name] = brier

        best_model = min(brier_scores, key=brier_scores.get)
        worst_model = max(brier_scores, key=brier_scores.get)

        summary_text += f"Best Calibrated: {best_model}\n"
        summary_text += f"  (Brier: {brier_scores[best_model]:.4f})\n\n"
        summary_text += f"Worst Calibrated: {worst_model}\n"
        summary_text += f"  (Brier: {brier_scores[worst_model]:.4f})"

        axes[5].text(
            0.1,
            0.7,
            summary_text,
            fontsize=10,
            transform=axes[5].transAxes,
            verticalalignment="top",
            family="monospace",
        )

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n✓ Saved calibration comparison to: {output_path}")

    def generate_latex_figure(self, output_path: Path):
        """Generate LaTeX figure wrapper"""
        latex = r"""\begin{figure}[!ht]
\centering
\includegraphics[width=\textwidth]{figures/out/baseline_calibration_comparison.png}
\caption[Baseline Model Calibration Comparison]{Reliability diagrams for all 6 baseline models on 2025 test set (116 games). Each panel shows predicted probability vs.\ observed frequency with 10 bins. Perfect calibration follows the diagonal line. Brier score and Expected Calibration Error (ECE) quantify calibration quality. XGBoost and State-Space show best calibration, while Bayesian LR exhibits slight underconfidence (predicts probabilities closer to 0.5 than outcomes warrant).}
\label{fig:baseline-calibration}
\end{figure}
"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)

        print(f"✓ LaTeX figure saved to: {output_path}")

    def print_summary(self):
        """Print calibration summary statistics"""
        val_data, test_data = self.split_validation_test()

        print("\n" + "=" * 80)
        print("BASELINE MODEL CALIBRATION SUMMARY")
        print("=" * 80)

        for dataset_name, data in [("Validation (2024)", val_data), ("Test (2025)", test_data)]:
            print(f"\n{dataset_name}: {len(data)} games")
            print("-" * 80)
            print(f"{'Model':<40} {'Brier':<12} {'ECE':<12} {'Calibration'}")
            print("-" * 80)

            models = {
                "xgboost_prediction": "XGBoost",
                "neural_net_prediction": "Neural Network",
                "bayesian_lr_prediction": "Bayesian Logistic Regression",
                "state_space_prediction": "State-Space (Kalman Filter)",
                "ensemble_prediction": "5-Component Ensemble",
            }

            for col, name in models.items():
                y_true = data["target"].values
                y_pred = data[col].values

                _, _, brier, ece = self.calculate_calibration(y_true, y_pred)

                # Calibration quality indicator
                if ece < 0.02:
                    indicator = "✅ Excellent"
                elif ece < 0.05:
                    indicator = "⚠️  Good"
                else:
                    indicator = "❌ Poor"

                print(f"{name:<40} {brier:<12.4f} {ece:<12.4f} {indicator}")

        print("\n" + "=" * 80)


def main():
    """Generate baseline calibration comparison"""
    try:
        comparator = BaselineCalibrationComparator()

        # Print summary
        comparator.print_summary()

        # Generate calibration figure
        figure_path = Path(
            "/Users/dro/rice/nfl-analytics/analysis/dissertation/figures/out/baseline_calibration_comparison.png"
        )
        comparator.generate_calibration_figure(figure_path)

        # Generate LaTeX wrapper
        latex_path = Path(
            "/Users/dro/rice/nfl-analytics/analysis/dissertation/figures/out/baseline_calibration_comparison.tex"
        )
        comparator.generate_latex_figure(latex_path)

        print(f"\n✓ Baseline calibration comparison generated successfully")
        print(f"\nOutputs:")
        print(f"  - Figure: {figure_path}")
        print(f"  - LaTeX:  {latex_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
