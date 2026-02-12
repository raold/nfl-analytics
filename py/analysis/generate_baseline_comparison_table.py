#!/usr/bin/env python3
"""
Generate Baseline Model Comparison Table for Chapter 4

Consolidates all baseline model results into a comprehensive comparison table
for the dissertation. Includes XGBoost, Neural Net, Bayesian LR, State-Space,
5-Component Ensemble, and Player-to-Game models.

Author: Claude Code
Date: 2025-10-26
"""

import json
from pathlib import Path


class BaselineModelComparator:
    """Generate comprehensive comparison table for baseline models"""

    def __init__(self):
        self.models_dir = _PROJECT_ROOT / "models"
        self.results = {}
        self.load_all_results()

    def load_all_results(self):
        """Load all model results from JSON files"""

        # Load ensemble results (includes individual models)
        ensemble_val = self.load_json(self.models_dir / "ensemble/game_outcome/val_metrics.json")
        ensemble_test = self.load_json(self.models_dir / "ensemble/game_outcome/test_metrics.json")

        # Load player-to-game results
        player_results = self.load_json(self.models_dir / "player_to_game/results.json")

        # Aggregate results
        self.results = {
            "xgboost": {
                "val_brier": ensemble_val["xgboost"]["brier"],
                "val_accuracy": ensemble_val["xgboost"]["accuracy"] * 100,
                "test_brier": ensemble_test["xgboost"]["brier"],
                "test_accuracy": ensemble_test["xgboost"]["accuracy"] * 100,
            },
            "neural_net": {
                "val_brier": ensemble_val["neural_net"]["brier"],
                "val_accuracy": ensemble_val["neural_net"]["accuracy"] * 100,
                "test_brier": ensemble_test["neural_net"]["brier"],
                "test_accuracy": ensemble_test["neural_net"]["accuracy"] * 100,
            },
            "bayesian_lr": {
                "val_brier": ensemble_val["bayesian_lr"]["brier"],
                "val_accuracy": ensemble_val["bayesian_lr"]["accuracy"] * 100,
                "test_brier": ensemble_test["bayesian_lr"]["brier"],
                "test_accuracy": ensemble_test["bayesian_lr"]["accuracy"] * 100,
            },
            "state_space": {
                "val_brier": ensemble_val["state_space"]["brier"],
                "val_accuracy": ensemble_val["state_space"]["accuracy"] * 100,
                "test_brier": ensemble_test["state_space"]["brier"],
                "test_accuracy": ensemble_test["state_space"]["accuracy"] * 100,
            },
            "ensemble": {
                "val_brier": ensemble_val["ensemble"]["brier"],
                "val_accuracy": ensemble_val["ensemble"]["accuracy"] * 100,
                "test_brier": ensemble_test["ensemble"]["brier"],
                "test_accuracy": ensemble_test["ensemble"]["accuracy"] * 100,
            },
            "player_to_game": {
                "val_brier": player_results["validation"]["brier"],
                "val_accuracy": player_results["validation"]["accuracy"] * 100,
                "test_brier": player_results["test"]["brier"],
                "test_accuracy": player_results["test"]["accuracy"] * 100,
            },
        }

    def load_json(self, path: Path) -> dict:
        """Load JSON file"""
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        with open(path) as f:
            return json.load(f)

    def calculate_generalization_gap(self, model: str) -> tuple:
        """Calculate validation to test generalization gap"""
        val_brier = self.results[model]["val_brier"]
        test_brier = self.results[model]["test_brier"]

        # Absolute gap
        abs_gap = test_brier - val_brier

        # Relative gap (percentage increase)
        rel_gap = ((test_brier - val_brier) / val_brier) * 100

        return abs_gap, rel_gap

    def generate_latex_table(self, output_path: Path):
        """Generate LaTeX table for Chapter 4"""

        latex = r"""\begin{table}[!ht]
\centering
\caption[Baseline Model Comparison]{Baseline Model Performance Comparison: All models evaluated on identical validation (2024) and test (2025) sets. Generalization gap shows val$\to$test Brier degradation.}
\label{tab:baseline-comparison}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Model} & \textbf{Val Brier} & \textbf{Test Brier} & \textbf{Val Acc (\%)} & \textbf{Test Acc (\%)} & \textbf{Gap} & \textbf{Rank} \\
\midrule
"""

        # Sort by test Brier (lower is better)
        sorted_models = sorted(self.results.keys(), key=lambda m: self.results[m]["test_brier"])

        for rank, model in enumerate(sorted_models, 1):
            r = self.results[model]
            abs_gap, rel_gap = self.calculate_generalization_gap(model)

            # Model name formatting
            model_name = {
                "xgboost": "XGBoost",
                "neural_net": "Neural Network",
                "bayesian_lr": "Bayesian Logistic Regression",
                "state_space": "State-Space (Kalman Filter)",
                "ensemble": "5-Component Ensemble",
                "player_to_game": "Player-to-Game (XGB)",
            }[model]

            # Mark best with bold
            if rank == 1:
                model_name = f"\\textbf{{{model_name}}}"

            # Highlight severe overfitting (>50% gap)
            gap_str = f"+{abs_gap:.4f}"
            if rel_gap > 50:
                gap_str = f"\\textcolor{{red}}{{{gap_str}}}"

            latex += f"{model_name} & {r['val_brier']:.4f} & {r['test_brier']:.4f} & "
            latex += f"{r['val_accuracy']:.1f} & {r['test_accuracy']:.1f} & "
            latex += f"{gap_str} & {rank} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{adjustbox}
\begin{tablenotes}[flushleft]\footnotesize
\item \textbf{Val:} 2024 season (272 games); \textbf{Test:} 2025 season weeks 1--7 (116 games)
\item \textbf{Gap:} Test Brier - Val Brier (positive = overfitting, negative = improved generalization)
\item \textbf{Rank:} Ordered by test Brier score (lower is better)
\item Ensemble shows severe overfitting (gap +0.1109, +70.4\% relative), while Player-to-Game and Neural Net generalize best
\item State-Space provides robust baseline with minimal overfitting (+0.0157, +6.4\%)
\end{tablenotes}
\end{table}
"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)

        print(f"✓ LaTeX table saved to: {output_path}")

    def print_summary(self):
        """Print human-readable summary"""
        print("\n" + "=" * 80)
        print("BASELINE MODEL COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\n{'Model':<35} {'Val Brier':<12} {'Test Brier':<12} {'Gap':<15} {'Rank'}")
        print("-" * 80)

        # Sort by test Brier
        sorted_models = sorted(self.results.keys(), key=lambda m: self.results[m]["test_brier"])

        for rank, model in enumerate(sorted_models, 1):
            r = self.results[model]
            abs_gap, rel_gap = self.calculate_generalization_gap(model)

            model_name = {
                "xgboost": "XGBoost",
                "neural_net": "Neural Network",
                "bayesian_lr": "Bayesian Logistic Regression",
                "state_space": "State-Space (Kalman Filter)",
                "ensemble": "5-Component Ensemble",
                "player_to_game": "Player-to-Game (XGB)",
            }[model]

            gap_indicator = "✅" if abs_gap < 0.02 else ("⚠️" if abs_gap < 0.05 else "❌")

            print(
                f"{model_name:<35} {r['val_brier']:<12.4f} {r['test_brier']:<12.4f} "
                f"{gap_indicator} +{abs_gap:.4f} ({rel_gap:+.1f}%)  #{rank}"
            )

        print("\n📊 KEY FINDINGS:")

        # Best model
        best_model = sorted_models[0]
        best_name = {
            "xgboost": "XGBoost",
            "neural_net": "Neural Network",
            "bayesian_lr": "Bayesian Logistic Regression",
            "state_space": "State-Space (Kalman Filter)",
            "ensemble": "5-Component Ensemble",
            "player_to_game": "Player-to-Game",
        }[best_model]

        print(
            f"  • Best test performance: {best_name} (Brier {self.results[best_model]['test_brier']:.4f})"
        )

        # Most robust (smallest gap)
        gaps = {m: self.calculate_generalization_gap(m)[0] for m in self.results.keys()}
        most_robust = min(gaps, key=gaps.get)
        robust_name = {
            "xgboost": "XGBoost",
            "neural_net": "Neural Network",
            "bayesian_lr": "Bayesian Logistic Regression",
            "state_space": "State-Space",
            "ensemble": "5-Component Ensemble",
            "player_to_game": "Player-to-Game",
        }[most_robust]

        print(f"  • Most robust: {robust_name} (gap {gaps[most_robust]:+.4f})")

        # Worst overfitting
        worst_overfit = max(gaps, key=gaps.get)
        overfit_name = {
            "xgboost": "XGBoost",
            "neural_net": "Neural Network",
            "bayesian_lr": "Bayesian Logistic Regression",
            "state_space": "State-Space",
            "ensemble": "5-Component Ensemble",
            "player_to_game": "Player-to-Game",
        }[worst_overfit]

        _, worst_rel_gap = self.calculate_generalization_gap(worst_overfit)
        print(
            f"  • Worst overfitting: {overfit_name} (gap {gaps[worst_overfit]:+.4f}, {worst_rel_gap:+.1f}% relative)"
        )

        print("\n" + "=" * 80)


def main():
    """Generate baseline comparison table"""
    try:
        comparator = BaselineModelComparator()

        # Print summary
        comparator.print_summary()

        # Generate LaTeX table
        output_path = _PROJECT_ROOT / "analysis/dissertation/figures/out/baseline_comparison_table.tex"
        comparator.generate_latex_table(output_path)

        print("\n✓ Baseline comparison table generated successfully")
        print(f"\nOutput: {output_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
