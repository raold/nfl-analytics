#!/usr/bin/env python3
"""
Model Version Comparison: v1.0 → v2.5 → v3.0

Comprehensive comparison of all model versions on 2024 holdout data:
- v1.0: Baseline hierarchical model (hierarchical_v1.0)
- v2.5: Informative priors model (passing_informative_priors_v1.rds)
- v3.0: Full ensemble with QB-WR chemistry, BNN, state-space

Outputs:
- Detailed performance metrics (MAE, RMSE, correlation, coverage)
- Comparative visualizations
- Model improvement breakdown
- ROI projections
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


class ModelVersionComparison:
    """Compare performance across model versions."""

    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.results = {}

    def fetch_actuals_2024(self, stat_type="passing_yards", aggregate_to_season=True):
        """
        Fetch actual 2024 outcomes for comparison.

        Args:
            stat_type: Type of stat to fetch
            aggregate_to_season: If True, aggregate game-level to season averages
                                to match prediction granularity
        """

        table_map = {
            "passing_yards": "nextgen_passing",
            "rushing_yards": "nextgen_rushing",
            "receiving_yards": "nextgen_receiving",
        }

        col_map = {
            "passing_yards": "pass_yards",
            "rushing_yards": "rush_yards",
            "receiving_yards": "receiving_yards",
        }

        table = table_map[stat_type]
        col = col_map[stat_type]

        if aggregate_to_season:
            # Aggregate to season-level per-game averages to match predictions
            query = f"""
            SELECT
                ngs.player_id,
                MAX(ngs.player_display_name) as player_display_name,
                MAX(ngs.player_position) as player_position,
                ngs.season,
                AVG(ngs.{col}) as actual_yards,  -- Per-game average
                COUNT(*) as n_games,
                MAX(ph.years_exp) as years_exp,
                MAX(ph.current_team) as current_team
            FROM {table} ngs
            JOIN mart.player_hierarchy ph ON ngs.player_id = ph.player_id
            WHERE ngs.season = 2024
                AND ngs.{col} IS NOT NULL
                AND ngs.week <= 17
            GROUP BY ngs.player_id, ngs.season
            HAVING COUNT(*) >= 3  -- At least 3 games for stable average
            ORDER BY ngs.player_id
            """
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"✓ Loaded {len(df)} season-aggregated actuals for {stat_type} (2024)")
        else:
            # Game-level data
            query = f"""
            SELECT
                ngs.player_id,
                ngs.player_display_name,
                ngs.player_position,
                ngs.season,
                ngs.week,
                ngs.{col} as actual_yards,
                ph.years_exp,
                ph.current_team
            FROM {table} ngs
            JOIN mart.player_hierarchy ph ON ngs.player_id = ph.player_id
            WHERE ngs.season = 2024
                AND ngs.{col} IS NOT NULL
                AND ngs.week <= 17
            ORDER BY ngs.player_id, ngs.week
            """
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"✓ Loaded {len(df)} game-level actuals for {stat_type} (2024)")

        return df

    def fetch_v1_predictions(self, stat_type="passing_yards"):
        """Fetch v1.0 baseline predictions from database."""

        query = """
        SELECT
            player_id,
            rating_mean,
            rating_sd,
            rating_q05,
            rating_q95,
            n_games_observed
        FROM mart.bayesian_player_ratings
        WHERE model_version = 'hierarchical_v1.0'
            AND stat_type = %s
            AND season = 2024
        """

        df = pd.read_sql_query(query, self.conn, params=(stat_type,))

        # Transform from log-space to yards
        df["pred_v1"] = np.exp(df["rating_mean"])
        df["uncertainty_v1"] = df["pred_v1"] * df["rating_sd"]
        df["ci_lower_v1"] = np.exp(df["rating_q05"])
        df["ci_upper_v1"] = np.exp(df["rating_q95"])

        logger.info(f"✓ Loaded {len(df)} v1.0 predictions")
        return df[
            [
                "player_id",
                "pred_v1",
                "uncertainty_v1",
                "ci_lower_v1",
                "ci_upper_v1",
                "n_games_observed",
            ]
        ]

    def generate_v2_5_predictions(self, actuals: pd.DataFrame):
        """Generate v2.5 predictions from informative priors model."""

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri

            pandas2ri.activate()

            # Load R model
            ro.r(
                """
            library(brms)
            model <- readRDS("models/bayesian/passing_informative_priors_v1.rds")
            """
            )

            logger.info("✓ Loaded v2.5 informative priors model")

            # For now, use same as v1 since we need proper R integration
            # This is a placeholder - proper implementation would use R to generate predictions
            logger.warning("⚠️  v2.5 predictions not yet implemented - using v1.0 as proxy")
            return None

        except ImportError:
            logger.warning("⚠️  rpy2 not available - cannot load R models")
            return None

    def load_bnn_predictions(self, actuals: pd.DataFrame):
        """Load BNN model and generate predictions."""

        bnn_path = Path("models/bayesian/bnn_passing_v1.pkl")
        if not bnn_path.exists():
            logger.warning(f"⚠️  BNN model not found at {bnn_path}")
            return None

        try:
            from py.models.bayesian_neural_network import BayesianNeuralNetwork

            logger.info("Loading BNN model...")
            BayesianNeuralNetwork.load(bnn_path)

            # Load metadata for feature engineering
            metadata_path = Path("models/bayesian/bnn_passing_v1_metadata.json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                logger.info(f"✓ Loaded BNN model (MAE: {metadata['test_mae']:.2f} yards)")

            # For now, return None until we implement feature engineering
            logger.warning("⚠️  BNN prediction pipeline not yet implemented")
            return None

        except Exception as e:
            logger.error(f"Error loading BNN: {e}")
            return None

    def calculate_metrics(
        self, actuals: np.ndarray, predictions: np.ndarray, model_name: str
    ) -> dict:
        """Calculate comprehensive performance metrics."""

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals.clip(min=1))) * 100
        correlation = np.corrcoef(actuals, predictions)[0, 1]

        # Residual analysis
        residuals = predictions - actuals
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        # Calculate skill score relative to baseline (mean prediction)
        baseline_mae = mean_absolute_error(actuals, np.full_like(actuals, actuals.mean()))
        skill_score = (baseline_mae - mae) / baseline_mae * 100

        metrics = {
            "model": model_name,
            "n_predictions": len(actuals),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "correlation": correlation,
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "skill_score": skill_score,
        }

        return metrics

    def run_comparison(self, stat_type="passing_yards"):
        """
        Run full comparison across all model versions.

        Note: v1.0 predictions are season-level per-game averages, so we aggregate
        actuals to match. This compares: predicted avg yards/game vs actual avg yards/game.
        """

        logger.info(f"\n{'='*80}")
        logger.info(f"MODEL VERSION COMPARISON: {stat_type}")
        logger.info(f"{'='*80}\n")

        # Load actuals (aggregated to season-level to match prediction granularity)
        actuals = self.fetch_actuals_2024(stat_type, aggregate_to_season=True)

        # Load v1.0 predictions
        v1_preds = self.fetch_v1_predictions(stat_type)

        # Merge with actuals
        comparison = actuals.merge(v1_preds, on="player_id", how="inner")

        logger.info(f"Merged {len(comparison)} predictions with actuals")

        if comparison.empty:
            logger.error("No matching predictions found")
            return None

        # Calculate v1.0 metrics
        v1_metrics = self.calculate_metrics(
            comparison["actual_yards"].values, comparison["pred_v1"].values, "v1.0 Baseline"
        )

        # Calculate CI coverage for v1.0
        in_ci_90 = (comparison["actual_yards"] >= comparison["ci_lower_v1"]) & (
            comparison["actual_yards"] <= comparison["ci_upper_v1"]
        )
        v1_metrics["ci_90_coverage"] = in_ci_90.mean()

        logger.info("\n✅ v1.0 Baseline Performance:")
        logger.info(f"  MAE: {v1_metrics['mae']:.2f} yards")
        logger.info(f"  RMSE: {v1_metrics['rmse']:.2f} yards")
        logger.info(f"  Correlation: {v1_metrics['correlation']:.3f}")
        logger.info(f"  90% CI Coverage: {v1_metrics['ci_90_coverage']:.1%}")
        logger.info(f"  Skill Score: {v1_metrics['skill_score']:.1f}%")

        # Store results
        self.results[stat_type] = {"comparison_df": comparison, "metrics": {"v1.0": v1_metrics}}

        return comparison

    def generate_comparison_report(self, stat_type="passing_yards"):
        """Generate comprehensive comparison visualizations."""

        if stat_type not in self.results:
            logger.error(f"No results found for {stat_type}")
            return

        comparison = self.results[stat_type]["comparison_df"]
        metrics = self.results[stat_type]["metrics"]

        output_dir = Path("reports/model_comparison_v3")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Actual vs Predicted (v1.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(comparison["actual_yards"], comparison["pred_v1"], alpha=0.5, s=20)
        max_val = max(comparison["actual_yards"].max(), comparison["pred_v1"].max())
        ax1.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect Prediction")
        ax1.set_xlabel("Actual Yards")
        ax1.set_ylabel("Predicted Yards")
        ax1.set_title(
            f'v1.0 Baseline\nMAE: {metrics["v1.0"]["mae"]:.1f}, r: {metrics["v1.0"]["correlation"]:.3f}'
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residual Distribution (v1.0)
        ax2 = fig.add_subplot(gs[0, 1])
        residuals_v1 = comparison["pred_v1"] - comparison["actual_yards"]
        ax2.hist(residuals_v1, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(0, color="red", linestyle="--", linewidth=2)
        ax2.axvline(
            residuals_v1.mean(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {residuals_v1.mean():.1f}",
        )
        ax2.set_xlabel("Prediction Error (yards)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"v1.0 Residual Distribution\nStd: {residuals_v1.std():.1f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. CI Coverage Analysis (v1.0)
        ax3 = fig.add_subplot(gs[0, 2])
        coverage_data = pd.DataFrame(
            {"Model": ["v1.0"], "Coverage": [metrics["v1.0"]["ci_90_coverage"]], "Expected": [0.90]}
        )
        x = np.arange(len(coverage_data))
        width = 0.35
        ax3.bar(x - width / 2, coverage_data["Coverage"], width, label="Actual", color="steelblue")
        ax3.bar(
            x + width / 2, coverage_data["Expected"], width, label="Expected", color="lightcoral"
        )
        ax3.set_ylabel("Coverage Rate")
        ax3.set_title("90% Credible Interval Coverage")
        ax3.set_xticks(x)
        ax3.set_xticklabels(coverage_data["Model"])
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Error by Experience Level
        ax4 = fig.add_subplot(gs[1, 0])
        comparison["error_v1"] = np.abs(comparison["pred_v1"] - comparison["actual_yards"])
        comparison["exp_category"] = pd.cut(
            comparison["years_exp"],
            bins=[0, 2, 5, 10, 20],
            labels=["Rookie (0-2)", "Young (3-5)", "Veteran (6-10)", "Experienced (10+)"],
        )
        exp_errors = comparison.groupby("exp_category")["error_v1"].mean().sort_values()
        ax4.barh(range(len(exp_errors)), exp_errors.values, color="steelblue")
        ax4.set_yticks(range(len(exp_errors)))
        ax4.set_yticklabels(exp_errors.index)
        ax4.set_xlabel("Mean Absolute Error (yards)")
        ax4.set_title("v1.0 Error by Experience Level")
        ax4.grid(True, alpha=0.3, axis="x")
        for i, v in enumerate(exp_errors.values):
            ax4.text(v + 1, i, f"{v:.1f}", va="center")

        # 5. Prediction Uncertainty vs Error
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(comparison["uncertainty_v1"], comparison["error_v1"], alpha=0.5, s=20)
        ax5.set_xlabel("Model Uncertainty (σ)")
        ax5.set_ylabel("Absolute Error (yards)")
        ax5.set_title("v1.0 Calibration: Uncertainty vs Error")
        ax5.grid(True, alpha=0.3)

        # Add correlation
        corr_unc_err = np.corrcoef(comparison["uncertainty_v1"], comparison["error_v1"])[0, 1]
        ax5.text(
            0.05,
            0.95,
            f"r = {corr_unc_err:.3f}",
            transform=ax5.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 6. Top 10 Best Predictions
        ax6 = fig.add_subplot(gs[1, 2])
        best_preds = comparison.nsmallest(10, "error_v1")[["player_display_name", "error_v1"]]
        ax6.barh(range(len(best_preds)), best_preds["error_v1"].values, color="green", alpha=0.7)
        ax6.set_yticks(range(len(best_preds)))
        ax6.set_yticklabels(best_preds["player_display_name"].values, fontsize=8)
        ax6.set_xlabel("Absolute Error (yards)")
        ax6.set_title("Top 10 Best Predictions (v1.0)")
        ax6.grid(True, alpha=0.3, axis="x")
        ax6.invert_yaxis()

        # 7. Top 10 Worst Predictions
        ax7 = fig.add_subplot(gs[2, 0])
        worst_preds = comparison.nlargest(10, "error_v1")[["player_display_name", "error_v1"]]
        ax7.barh(range(len(worst_preds)), worst_preds["error_v1"].values, color="red", alpha=0.7)
        ax7.set_yticks(range(len(worst_preds)))
        ax7.set_yticklabels(worst_preds["player_display_name"].values, fontsize=8)
        ax7.set_xlabel("Absolute Error (yards)")
        ax7.set_title("Top 10 Worst Predictions (v1.0)")
        ax7.grid(True, alpha=0.3, axis="x")
        ax7.invert_yaxis()

        # 8. Weekly Error Trend
        ax8 = fig.add_subplot(gs[2, 1])
        weekly_errors = (
            comparison.groupby("week")
            .agg({"error_v1": "mean", "actual_yards": "count"})
            .reset_index()
        )
        ax8.plot(
            weekly_errors["week"], weekly_errors["error_v1"], "o-", color="steelblue", linewidth=2
        )
        ax8.set_xlabel("Week")
        ax8.set_ylabel("Mean Absolute Error (yards)")
        ax8.set_title("v1.0 Weekly Error Trend (2024)")
        ax8.grid(True, alpha=0.3)
        ax8.set_xticks(range(1, 18))

        # 9. Summary Metrics Comparison
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis("off")
        summary_text = f"""
        v1.0 BASELINE PERFORMANCE
        ═══════════════════════════════

        Sample Size: {metrics['v1.0']['n_predictions']:,}

        Accuracy Metrics:
          • MAE: {metrics['v1.0']['mae']:.2f} yards
          • RMSE: {metrics['v1.0']['rmse']:.2f} yards
          • MAPE: {metrics['v1.0']['mape']:.1f}%
          • Correlation: {metrics['v1.0']['correlation']:.3f}

        Uncertainty Quantification:
          • 90% CI Coverage: {metrics['v1.0']['ci_90_coverage']:.1%}
          • Expected: 90.0%
          • Status: {'✅ Well Calibrated' if abs(metrics['v1.0']['ci_90_coverage'] - 0.9) < 0.05 else '⚠️ Needs Calibration'}

        Model Skill:
          • Skill Score: {metrics['v1.0']['skill_score']:.1f}%
          • Mean Bias: {metrics['v1.0']['mean_residual']:.2f} yards

        ═══════════════════════════════
        v2.5 & v3.0 PENDING INTEGRATION
        """
        ax9.text(
            0.1,
            0.9,
            summary_text,
            transform=ax9.transAxes,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="top",
        )

        plt.suptitle(
            f'{stat_type.replace("_", " ").title()} - Model Version Comparison (2024)',
            fontsize=16,
            fontweight="bold",
        )

        # Save figure
        output_path = output_dir / f"{stat_type}_model_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Saved comparison report to {output_path}")

        # Save detailed metrics
        metrics_path = output_dir / f"{stat_type}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save detailed results
        results_path = output_dir / f"{stat_type}_detailed_results.csv"
        comparison.to_csv(results_path, index=False)

        logger.info(f"✓ Saved detailed results to {results_path}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    """Run model version comparison."""

    print("\n" + "=" * 80)
    print("MODEL VERSION COMPARISON: v1.0 → v2.5 → v3.0")
    print("=" * 80 + "\n")

    comparator = ModelVersionComparison()

    try:
        # Run comparison for passing yards
        comparison_df = comparator.run_comparison("passing_yards")

        if comparison_df is not None:
            # Generate comprehensive report
            comparator.generate_comparison_report("passing_yards")

            print("\n" + "=" * 80)
            print("✅ MODEL COMPARISON COMPLETE")
            print("=" * 80)
            print("\nReports saved to: reports/model_comparison_v3/")
            print("\nNext Steps:")
            print("  1. Integrate v2.5 informative priors predictions")
            print("  2. Add v3.0 ensemble predictions (BNN + QB-WR chemistry)")
            print("  3. Run ROI simulation on historical betting lines")
            print("  4. Deploy best performing model to production")

    except Exception as e:
        logger.error(f"Error in comparison: {e}", exc_info=True)

    finally:
        comparator.close()


if __name__ == "__main__":
    main()
