#!/usr/bin/env python3
"""
Feature Importance Analysis & Pruning

Analyzes v3 model feature importance using multiple methods:
- XGBoost gain/weight importance
- SHAP values (if available)
- Correlation analysis
- Redundancy detection

Recommends features for pruning and tests performance impact.

Usage:
    python py/analysis/feature_importance_analysis.py --model-path models/xgboost/v3_production/model.json
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Analyze feature importance and recommend pruning."""

    def __init__(self, model_path: str, features_csv: str):
        """Initialize analyzer."""
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.parent / "metadata.json"
        self.features_csv = features_csv

        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

        self.features = self.metadata["training_data"]["features"]
        logger.info(f"Model has {len(self.features)} features")

    def load_data(self) -> pd.DataFrame:
        """Load feature data."""
        logger.info(f"Loading data from {self.features_csv}")
        df = pd.read_csv(self.features_csv)

        # Filter to completed games with target
        df = df[df["home_score"].notna()].copy()
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

        logger.info(f"Loaded {len(df)} completed games")
        return df

    def get_xgboost_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        logger.info("Extracting XGBoost feature importance...")

        importance_types = ["weight", "gain", "cover"]
        importance_data = []

        for imp_type in importance_types:
            scores = self.model.get_score(importance_type=imp_type)
            for feature, score in scores.items():
                importance_data.append(
                    {"feature": feature, "importance_type": imp_type, "score": score}
                )

        df = pd.DataFrame(importance_data)

        # Pivot to wide format
        df_pivot = df.pivot(
            index="feature", columns="importance_type", values="score"
        ).reset_index()
        df_pivot = df_pivot.fillna(0)

        # Calculate composite score (weighted average)
        df_pivot["composite_score"] = (
            0.5 * df_pivot["gain"] / df_pivot["gain"].max()
            + 0.3 * df_pivot["weight"] / df_pivot["weight"].max()
            + 0.2 * df_pivot["cover"] / df_pivot["cover"].max()
        )

        df_pivot = df_pivot.sort_values("composite_score", ascending=False)

        logger.info(f"Extracted importance for {len(df_pivot)} features")
        return df_pivot

    def analyze_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature correlations to find redundancy."""
        logger.info("Analyzing feature correlations...")

        # Get numeric features
        numeric_features = [
            f
            for f in self.features
            if f in df.columns and df[f].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        if len(numeric_features) == 0:
            logger.warning("No numeric features found")
            return pd.DataFrame()

        # Compute correlation matrix
        corr_matrix = df[numeric_features].corr().abs()

        # Find highly correlated pairs (> 0.9)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.9:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )

        if high_corr_pairs:
            corr_df = pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False)
            logger.info(f"Found {len(corr_df)} highly correlated feature pairs (> 0.9)")
            return corr_df
        else:
            logger.info("No highly correlated features found (> 0.9)")
            return pd.DataFrame()

    def recommend_pruning(
        self, importance_df: pd.DataFrame, threshold_percentile: float = 10.0
    ) -> dict:
        """
        Recommend features to prune based on importance.

        Args:
            importance_df: Feature importance DataFrame
            threshold_percentile: Percentile below which to recommend pruning

        Returns:
            Dict with recommendations
        """
        logger.info(
            f"Generating pruning recommendations (threshold: {threshold_percentile}th percentile)..."
        )

        # Calculate threshold
        threshold = np.percentile(importance_df["composite_score"], threshold_percentile)

        # Features below threshold
        low_importance = importance_df[importance_df["composite_score"] < threshold]

        # Features to keep
        high_importance = importance_df[importance_df["composite_score"] >= threshold]

        recommendations = {
            "total_features": len(importance_df),
            "features_to_keep": len(high_importance),
            "features_to_prune": len(low_importance),
            "threshold_percentile": threshold_percentile,
            "threshold_value": float(threshold),
            "keep_list": high_importance["feature"].tolist(),
            "prune_list": low_importance["feature"].tolist(),
            "pruned_importance_sum": float(low_importance["gain"].sum()),
            "total_importance_sum": float(importance_df["gain"].sum()),
        }

        recommendations["pruned_importance_pct"] = (
            recommendations["pruned_importance_sum"] / recommendations["total_importance_sum"] * 100
        )

        logger.info(
            f"Recommendation: Keep {recommendations['features_to_keep']}, "
            f"Prune {recommendations['features_to_prune']} features"
        )
        logger.info(
            f"Pruned features represent {recommendations['pruned_importance_pct']:.2f}% of total importance"
        )

        return recommendations

    def test_pruned_model(
        self, df: pd.DataFrame, keep_features: list[str], test_season: int = 2024
    ) -> dict:
        """
        Test performance with pruned feature set.

        Args:
            df: Full dataset
            keep_features: Features to keep
            test_season: Season to use for testing

        Returns:
            Performance metrics
        """
        logger.info(f"Testing pruned model on {test_season} season...")

        # Split data
        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] == test_season].copy()

        logger.info(f"Train: {len(train_df)} games, Test: {len(test_df)} games")

        # Check which features are available
        available_features = [f for f in keep_features if f in df.columns]
        missing_features = [f for f in keep_features if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")

        # Prepare data
        X_test = test_df[available_features].values
        y_test = test_df["home_win"].values
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Predict with full model
        X_test_full = test_df[self.features].fillna(0).values
        dmatrix_full = xgb.DMatrix(X_test_full, feature_names=self.features)
        y_pred_full = self.model.predict(dmatrix_full)

        # For pruned model, we'd need to retrain, but for now just evaluate
        # what happens if we zero out pruned features
        X_test_pruned = test_df[self.features].fillna(0).copy()
        pruned_features = [f for f in self.features if f not in keep_features]
        for f in pruned_features:
            if f in X_test_pruned.columns:
                X_test_pruned[f] = 0

        dmatrix_pruned = xgb.DMatrix(X_test_pruned.values, feature_names=self.features)
        y_pred_pruned = self.model.predict(dmatrix_pruned)

        # Calculate metrics
        metrics = {
            "full_model": {
                "brier_score": float(brier_score_loss(y_test, y_pred_full)),
                "accuracy": float(accuracy_score(y_test, (y_pred_full > 0.5).astype(int))),
            },
            "pruned_model": {
                "brier_score": float(brier_score_loss(y_test, y_pred_pruned)),
                "accuracy": float(accuracy_score(y_test, (y_pred_pruned > 0.5).astype(int))),
            },
        }

        metrics["brier_diff"] = (
            metrics["pruned_model"]["brier_score"] - metrics["full_model"]["brier_score"]
        )
        metrics["accuracy_diff"] = (
            metrics["pruned_model"]["accuracy"] - metrics["full_model"]["accuracy"]
        )

        logger.info(
            f"Full Model - Brier: {metrics['full_model']['brier_score']:.4f}, "
            f"Accuracy: {metrics['full_model']['accuracy']:.1%}"
        )
        logger.info(
            f"Pruned Model - Brier: {metrics['pruned_model']['brier_score']:.4f}, "
            f"Accuracy: {metrics['pruned_model']['accuracy']:.1%}"
        )
        logger.info(
            f"Difference - Brier: {metrics['brier_diff']:+.4f}, "
            f"Accuracy: {metrics['accuracy_diff']:+.1%}"
        )

        return metrics

    def visualize_importance(self, importance_df: pd.DataFrame, output_dir: str, top_n: int = 30):
        """Create visualizations of feature importance."""
        logger.info(f"Creating importance visualizations (top {top_n})...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Top N features by composite score
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features["composite_score"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Composite Importance Score")
        plt.title(f"Top {top_n} Features by Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path / "feature_importance_top30.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved visualization to {output_path / 'feature_importance_top30.png'}")

    def generate_report(
        self,
        importance_df: pd.DataFrame,
        recommendations: dict,
        test_metrics: dict,
        output_path: str,
    ):
        """Generate comprehensive analysis report."""
        logger.info("Generating report...")

        report = []
        report.append("# Feature Importance Analysis Report\n")
        report.append(f"**Model**: {self.model_path}\n")
        report.append(f"**Total Features**: {len(self.features)}\n\n")

        report.append("## Summary\n")
        report.append(f"- **Features to Keep**: {recommendations['features_to_keep']}\n")
        report.append(f"- **Features to Prune**: {recommendations['features_to_prune']}\n")
        report.append(
            f"- **Pruned Importance**: {recommendations['pruned_importance_pct']:.2f}%\n\n"
        )

        report.append("## Performance Impact\n")
        report.append(f"- **Full Model Brier**: {test_metrics['full_model']['brier_score']:.4f}\n")
        report.append(
            f"- **Pruned Model Brier**: {test_metrics['pruned_model']['brier_score']:.4f}\n"
        )
        report.append(f"- **Difference**: {test_metrics['brier_diff']:+.4f}\n\n")

        report.append("## Top 20 Features\n")
        for i, row in importance_df.head(20).iterrows():
            report.append(
                f"{i+1}. **{row['feature']}** - Gain: {row['gain']:.1f}, Weight: {row['weight']:.0f}\n"
            )

        report.append("\n## Bottom 10 Features (Candidates for Pruning)\n")
        for i, row in importance_df.tail(10).iterrows():
            report.append(
                f"- **{row['feature']}** - Gain: {row['gain']:.1f}, Composite: {row['composite_score']:.4f}\n"
            )

        # Write report
        with open(output_path, "w") as f:
            f.writelines(report)

        logger.info(f"Report saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument("--model-path", default="models/xgboost/v3_production/model.json")
    parser.add_argument(
        "--features-csv", default="data/processed/features/asof_team_features_v3.csv"
    )
    parser.add_argument("--output-dir", default="analysis/feature_importance")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for pruning (default: 10)",
    )
    parser.add_argument(
        "--test-season", type=int, default=2024, help="Season to use for testing (default: 2024)"
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(args.model_path, args.features_csv)

    # Load data
    df = analyzer.load_data()

    # Get importance
    importance_df = analyzer.get_xgboost_importance()

    # Analyze correlations
    corr_df = analyzer.analyze_correlation(df)

    # Generate recommendations
    recommendations = analyzer.recommend_pruning(importance_df, args.threshold_percentile)

    # Test pruned model
    test_metrics = analyzer.test_pruned_model(df, recommendations["keep_list"], args.test_season)

    # Create visualizations
    analyzer.visualize_importance(importance_df, args.output_dir)

    # Generate report
    report_path = Path(args.output_dir) / "feature_importance_report.md"
    analyzer.generate_report(importance_df, recommendations, test_metrics, str(report_path))

    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(output_path / "feature_importance_detailed.csv", index=False)
    if not corr_df.empty:
        corr_df.to_csv(output_path / "high_correlation_pairs.csv", index=False)

    with open(output_path / "pruning_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    with open(output_path / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total Features: {recommendations['total_features']}")
    print(f"Recommended to Keep: {recommendations['features_to_keep']}")
    print(f"Recommended to Prune: {recommendations['features_to_prune']}")
    print("\nPerformance Impact:")
    print(f"  Brier Score Change: {test_metrics['brier_diff']:+.4f}")
    print(f"  Accuracy Change: {test_metrics['accuracy_diff']:+.1%}")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
