"""
Feature Ablation Study for Realistic Ensemble

Tests multiple configurations to identify which features contribute to elevated win rate:

Configurations:
1. Baseline Only - No Phase 5 features (9 features)
2. Current (5 Phase 5 features) - With regularization alpha=2.0, lambda=2.0
3. Increased Regularization - alpha=5.0, lambda=5.0
4. Individual feature ablations - Remove each Phase 5 feature one at a time

This helps identify:
- Which features drive the elevated 63.2% win rate
- Whether increased regularization helps calibration
- Whether specific Phase 5 features cause overfitting

Author: Claude Code
Date: 2025-10-30
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.ensemble.game_outcome_ensemble_realistic import RealisticGameOutcomeEnsemble
from py.models.state_space import fetch_games


class FeatureAblationStudy:
    """Run comprehensive feature ablation study."""

    def __init__(self, train_seasons: list[int], test_seasons: list[int]):
        self.train_seasons = train_seasons
        self.test_seasons = test_seasons
        self.results = {}

    def create_ensemble_config(
        self,
        config_name: str,
        features_to_include: list[str] | None = None,
        alpha: float = 2.0,
        lambda_: float = 2.0,
    ) -> tuple[RealisticGameOutcomeEnsemble, str]:
        """
        Create ensemble with specific feature configuration.

        Args:
            config_name: Name of configuration
            features_to_include: List of Phase 5 features to include (None = all 5)
            alpha: XGBoost L1 regularization
            lambda_: XGBoost L2 regularization

        Returns:
            (ensemble, description)
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
            'alpha': alpha,
            'lambda': lambda_,
            'min_child_weight': 5,
            'gamma': 0.5,
            'n_jobs': -1,
        }

        ensemble = RealisticGameOutcomeEnsemble(xgb_params=xgb_params)

        # Override legitimate features if specified
        if features_to_include is not None:
            ensemble.legitimate_phase5_features = features_to_include

        description = f"{config_name} | Features: {len(features_to_include) if features_to_include else 5} | alpha={alpha}, lambda={lambda_}"
        return ensemble, description

    def train_and_evaluate(
        self,
        ensemble: RealisticGameOutcomeEnsemble,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config_name: str,
    ) -> dict:
        """Train ensemble and evaluate on test set."""
        print(f"\n{'='*80}")
        print(f"CONFIGURATION: {config_name}")
        print(f"{'='*80}")

        # Train
        train_metrics = ensemble.fit(train_df)

        # Evaluate on test set
        test_metrics = ensemble.evaluate(test_df)

        # Combine results
        results = {
            "config_name": config_name,
            "train_accuracy": train_metrics["ensemble"]["accuracy"],
            "train_brier": train_metrics["ensemble"]["brier"],
            "test_accuracy": test_metrics["ensemble"]["accuracy"],
            "test_brier": test_metrics["ensemble"]["brier"],
            "test_logloss": test_metrics["ensemble"]["logloss"],
            "features": ensemble.legitimate_phase5_features.copy(),
            "n_features": len(ensemble.legitimate_phase5_features) + 9,  # +9 baseline
        }

        print(f"\nRESULTS:")
        print(f"  Train Accuracy: {results['train_accuracy']:.1%}")
        print(f"  Test Accuracy:  {results['test_accuracy']:.1%}")
        print(f"  Test Brier:     {results['test_brier']:.4f}")
        print(f"  Features:       {results['n_features']} ({len(ensemble.legitimate_phase5_features)} Phase 5)")

        return results

    def run_full_study(self) -> dict:
        """Run complete ablation study."""
        print("="*80)
        print("FEATURE ABLATION STUDY")
        print("="*80)
        print(f"Train seasons: {self.train_seasons}")
        print(f"Test seasons:  {self.test_seasons}")

        # Load data
        all_seasons = self.train_seasons + self.test_seasons
        df_all = fetch_games(all_seasons)
        train_df = df_all[df_all["season"].isin(self.train_seasons)]
        test_df = df_all[df_all["season"].isin(self.test_seasons)]

        print(f"\nTrain: {len(train_df)} games")
        print(f"Test:  {len(test_df)} games")

        # All 5 legitimate Phase 5 features
        all_phase5_features = [
            "extreme_cold",
            "dome_team_outdoor_wind",
            "is_conference_game",
            "away_short_rest",
            "high_wind",
        ]

        # ====================================================================
        # Configuration 1: BASELINE ONLY (No Phase 5 features)
        # ====================================================================
        ensemble, desc = self.create_ensemble_config(
            "Baseline Only",
            features_to_include=[],  # No Phase 5 features
            alpha=2.0,
            lambda_=2.0,
        )
        self.results["baseline_only"] = self.train_and_evaluate(
            ensemble, train_df, test_df, "Baseline Only"
        )

        # ====================================================================
        # Configuration 2: CURRENT (All 5 Phase 5 features, standard reg)
        # ====================================================================
        ensemble, desc = self.create_ensemble_config(
            "Current (5 Phase 5 features)",
            features_to_include=all_phase5_features,
            alpha=2.0,
            lambda_=2.0,
        )
        self.results["current"] = self.train_and_evaluate(
            ensemble, train_df, test_df, "Current (5 Phase 5 features)"
        )

        # ====================================================================
        # Configuration 3: INCREASED REGULARIZATION (All 5, higher reg)
        # ====================================================================
        ensemble, desc = self.create_ensemble_config(
            "Increased Regularization",
            features_to_include=all_phase5_features,
            alpha=5.0,
            lambda_=5.0,
        )
        self.results["increased_reg"] = self.train_and_evaluate(
            ensemble, train_df, test_df, "Increased Regularization (alpha=5, lambda=5)"
        )

        # ====================================================================
        # Configuration 4-8: INDIVIDUAL FEATURE ABLATIONS
        # Remove each Phase 5 feature one at a time
        # ====================================================================
        for feature_to_remove in all_phase5_features:
            features_subset = [f for f in all_phase5_features if f != feature_to_remove]
            config_name = f"Without {feature_to_remove}"

            ensemble, desc = self.create_ensemble_config(
                config_name,
                features_to_include=features_subset,
                alpha=2.0,
                lambda_=2.0,
            )

            self.results[f"ablate_{feature_to_remove}"] = self.train_and_evaluate(
                ensemble, train_df, test_df, config_name
            )

        return self.results

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    def print_summary(self):
        """Print summary comparison table."""
        print("\n" + "="*80)
        print("FEATURE ABLATION STUDY - SUMMARY")
        print("="*80)

        # Create summary dataframe
        rows = []
        for config_key, result in self.results.items():
            rows.append({
                "Configuration": result["config_name"],
                "Features": result["n_features"],
                "Train Acc": f"{result['train_accuracy']:.1%}",
                "Test Acc": f"{result['test_accuracy']:.1%}",
                "Test Brier": f"{result['test_brier']:.4f}",
            })

        df_summary = pd.DataFrame(rows)
        print("\n", df_summary.to_string(index=False))

        # Highlight key findings
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)

        baseline_acc = self.results["baseline_only"]["test_accuracy"]
        current_acc = self.results["current"]["test_accuracy"]
        increased_reg_acc = self.results["increased_reg"]["test_accuracy"]

        print(f"\n1. BASELINE ONLY (no Phase 5 features):")
        print(f"   Test Accuracy: {baseline_acc:.1%}")

        print(f"\n2. CURRENT (5 Phase 5 features):")
        print(f"   Test Accuracy: {current_acc:.1%}")
        print(f"   Lift from baseline: {(current_acc - baseline_acc):.1%}")

        print(f"\n3. INCREASED REGULARIZATION:")
        print(f"   Test Accuracy: {increased_reg_acc:.1%}")
        print(f"   Change from current: {(increased_reg_acc - current_acc):.1%}")

        # Identify most important features
        print(f"\n4. INDIVIDUAL FEATURE IMPORTANCE (by ablation):")
        all_phase5_features = [
            "extreme_cold",
            "dome_team_outdoor_wind",
            "is_conference_game",
            "away_short_rest",
            "high_wind",
        ]

        for feature in all_phase5_features:
            ablate_key = f"ablate_{feature}"
            if ablate_key in self.results:
                ablate_acc = self.results[ablate_key]["test_accuracy"]
                impact = current_acc - ablate_acc
                print(f"   {feature:30s}: {impact:+.2%} (removing this feature)")


def main():
    """Run feature ablation study."""
    # Configuration
    train_seasons = [2019, 2020, 2021]
    test_seasons = [2022, 2023, 2024]

    # Run study
    study = FeatureAblationStudy(train_seasons, test_seasons)
    results = study.run_full_study()

    # Print summary
    study.print_summary()

    # Save results
    output_dir = Path("analysis/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    study.save_results(output_dir / "feature_ablation_results.json")

    print("\n" + "="*80)
    print("FEATURE ABLATION STUDY COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
