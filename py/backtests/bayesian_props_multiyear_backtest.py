#!/usr/bin/env python3
"""
Multi-Year Backtest: Bayesian Hierarchical Player Props Models (2022-2024)

This script performs comprehensive backtesting across multiple seasons to evaluate
model performance, stability, and generalization.

Key Analyses:
- Year-over-year performance comparison
- Cross-season model stability
- Early vs late season accuracy
- Rookie vs veteran performance
- Position-specific trends
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


class MultiYearBacktest:
    """Multi-season backtesting for Bayesian player props models."""

    def __init__(self, db_config: dict = None):
        """Initialize backtester."""
        self.db_config = db_config or DB_CONFIG
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to database for multi-year backtesting")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def fetch_actual_outcomes_multiyear(
        self, stat_type: str, seasons: list[int], weeks: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Fetch actual player outcomes across multiple seasons.

        Args:
            stat_type: Type of stat ('passing_yards', 'rushing_yards', 'receiving_yards')
            seasons: List of seasons to fetch
            weeks: Optional list of weeks to include

        Returns:
            DataFrame with actual outcomes across all seasons
        """
        stat_column_map = {
            "passing_yards": ("nextgen_passing", "pass_yards", "attempts"),
            "rushing_yards": ("nextgen_rushing", "rush_yards", "carries"),
            "receiving_yards": ("nextgen_receiving", "receiving_yards", "targets"),
        }

        table, stat_col, volume_col = stat_column_map.get(stat_type)
        if not table:
            raise ValueError(f"Unknown stat type: {stat_type}")

        week_filter = f"AND week IN ({','.join(map(str, weeks))})" if weeks else ""

        query = f"""
        SELECT
            ngs.player_id,
            ngs.player_display_name,
            ngs.player_position,
            ngs.season,
            ngs.week,
            ph.current_team as team,
            ngs.{stat_col} as actual_yards,
            ngs.{volume_col} as volume
        FROM {table} ngs
        LEFT JOIN mart.player_hierarchy ph ON ngs.player_id = ph.player_id
        WHERE ngs.season IN ({','.join(map(str, seasons))})
            AND ngs.{stat_col} IS NOT NULL
            {week_filter}
        ORDER BY ngs.season, ngs.week, ngs.player_id
        """

        try:
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Fetched {len(df)} outcomes for {stat_type} across {len(seasons)} seasons")
            return df
        except Exception as e:
            logger.error(f"Error fetching outcomes: {e}")
            return pd.DataFrame()

    def get_bayesian_predictions_multiyear(
        self, stat_type: str, seasons: list[int]
    ) -> pd.DataFrame:
        """
        Fetch Bayesian predictions for multiple seasons.

        Note: This assumes models were trained on data up to but not including each season.
        For proper backtesting, we would train separate models for each season.
        """
        all_predictions = []

        for season in seasons:
            # Get all players with stats in this season
            query = f"""
            SELECT DISTINCT player_id
            FROM mart.player_game_stats
            WHERE season = {season}
                AND stat_category = '{stat_type.split('_')[0]}'
            """

            try:
                players_df = pd.read_sql_query(query, self.conn)
                players_df["player_id"].tolist()

                # Fetch Bayesian ratings for this season
                ratings_query = """
                SELECT
                    bpr.player_id,
                    bpr.season,
                    bpr.stat_type,
                    bpr.rating_mean,
                    bpr.rating_sd,
                    bpr.rating_q05,
                    bpr.rating_q50,
                    bpr.rating_q95,
                    bpr.position_group_mean,
                    bpr.team_effect,
                    bpr.n_games_observed,
                    bpr.effective_sample_size,
                    bpr.rhat,
                    p.player_name,
                    p.position,
                    ph.current_team,
                    ph.years_exp
                FROM mart.bayesian_player_ratings bpr
                JOIN players p ON bpr.player_id = p.player_id
                LEFT JOIN mart.player_hierarchy ph ON bpr.player_id = ph.player_id
                WHERE bpr.stat_type = %s
                    AND bpr.season = %s
                    AND bpr.rhat < 1.1
                """

                predictions = pd.read_sql_query(
                    ratings_query, self.conn, params=(stat_type, season)
                )

                if not predictions.empty:
                    all_predictions.append(predictions)
                    logger.info(f"Fetched {len(predictions)} predictions for {season}")

            except Exception as e:
                logger.warning(f"Error fetching predictions for {season}: {e}")
                continue

        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_season_metrics(self, results: pd.DataFrame, season: int) -> dict:
        """Calculate metrics for a specific season."""
        season_data = results[results["season"] == season]

        if season_data.empty:
            return {}

        metrics = {
            "season": season,
            "n_predictions": len(season_data),
            "mae": mean_absolute_error(
                season_data["actual_yards"], season_data["bayes_prediction"]
            ),
            "rmse": np.sqrt(
                mean_squared_error(season_data["actual_yards"], season_data["bayes_prediction"])
            ),
            "mape": np.mean(
                np.abs(
                    (season_data["actual_yards"] - season_data["bayes_prediction"])
                    / season_data["actual_yards"].clip(lower=1)
                )
            )
            * 100,
            "correlation": season_data[["actual_yards", "bayes_prediction"]].corr().iloc[0, 1],
        }

        # Coverage metrics
        if "in_ci_90" in season_data.columns:
            metrics["ci_90_coverage"] = season_data["in_ci_90"].mean()
            metrics["ci_68_coverage"] = (
                season_data["in_ci_68"].mean() if "in_ci_68" in season_data.columns else None
            )

        # Position breakdown
        for pos in season_data["player_position"].unique():
            pos_data = season_data[season_data["player_position"] == pos]
            if len(pos_data) >= 10:
                metrics[f"mae_{pos}"] = mean_absolute_error(
                    pos_data["actual_yards"], pos_data["bayes_prediction"]
                )

        return metrics

    def run_multiyear_backtest(
        self,
        stat_type: str,
        seasons: list[int] = [2022, 2023, 2024],
        start_week: int = 1,
        end_week: int = 17,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run backtest across multiple seasons.

        Args:
            stat_type: Type of stat to backtest
            seasons: List of seasons to include
            start_week: First week to include
            end_week: Last week to include

        Returns:
            Tuple of (results DataFrame, metrics by season)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Multi-Year Backtest: {stat_type}")
        logger.info(f"Seasons: {seasons}")
        logger.info(f"Weeks: {start_week}-{end_week}")
        logger.info("=" * 80)

        # Fetch actual outcomes for all seasons
        actuals = self.fetch_actual_outcomes_multiyear(
            stat_type=stat_type, seasons=seasons, weeks=list(range(start_week, end_week + 1))
        )

        if actuals.empty:
            logger.error("No actual outcomes found")
            return pd.DataFrame(), {}

        # Fetch Bayesian predictions for all seasons
        predictions = self.get_bayesian_predictions_multiyear(stat_type=stat_type, seasons=seasons)

        if predictions.empty:
            logger.warning("No Bayesian predictions found - using fallback approach")
            # Fallback: use position group means as baseline
            predictions = self._generate_baseline_predictions(actuals, stat_type)

        # Merge actuals with predictions
        results = actuals.merge(
            predictions, on=["player_id", "season"], how="inner", suffixes=("_actual", "_pred")
        )

        if results.empty:
            logger.error("No matching predictions found for actuals")
            return pd.DataFrame(), {}

        logger.info(f"Matched {len(results)} predictions with actuals")

        # Calculate derived metrics
        # CRITICAL: Transform from log-space to yards (model was trained on log(yards))
        if "rating_mean" in results.columns:
            results["bayes_prediction"] = np.exp(results["rating_mean"])
            # Transform uncertainty: use delta method for log-normal distribution
            # Var(exp(X)) ≈ exp(2μ + σ²)(exp(σ²) - 1) for X ~ N(μ, σ²)
            results["bayes_uncertainty"] = results["bayes_prediction"] * results.get(
                "rating_sd", 0.5
            )
            # Transform quantiles from log-space to yards
            if "rating_q05" in results.columns:
                results["bayes_ci_lower"] = np.exp(results["rating_q05"])
            else:
                results["bayes_ci_lower"] = (
                    results["bayes_prediction"] - 1.645 * results["bayes_uncertainty"]
                )
            if "rating_q95" in results.columns:
                results["bayes_ci_upper"] = np.exp(results["rating_q95"])
            else:
                results["bayes_ci_upper"] = (
                    results["bayes_prediction"] + 1.645 * results["bayes_uncertainty"]
                )
        else:
            results["bayes_prediction"] = results.get("bayes_prediction", 0)
            results["bayes_uncertainty"] = results.get("bayes_uncertainty", 50)
            results["bayes_ci_lower"] = results.get(
                "rating_q05", results["bayes_prediction"] - 1.645 * results["bayes_uncertainty"]
            )
            results["bayes_ci_upper"] = results.get(
                "rating_q95", results["bayes_prediction"] + 1.645 * results["bayes_uncertainty"]
            )

        # Calculate errors
        results["error"] = results["bayes_prediction"] - results["actual_yards"]
        results["abs_error"] = np.abs(results["error"])
        results["squared_error"] = results["error"] ** 2
        results["pct_error"] = np.abs(results["error"] / results["actual_yards"].clip(lower=1))

        # Check credible interval coverage
        results["in_ci_90"] = (
            (results["actual_yards"] >= results["bayes_ci_lower"])
            & (results["actual_yards"] <= results["bayes_ci_upper"])
        ).astype(int)

        results["in_ci_68"] = (
            (results["actual_yards"] >= results["bayes_prediction"] - results["bayes_uncertainty"])
            & (
                results["actual_yards"]
                <= results["bayes_prediction"] + results["bayes_uncertainty"]
            )
        ).astype(int)

        # Calculate metrics for each season
        season_metrics = {}
        for season in seasons:
            season_metrics[season] = self.calculate_season_metrics(results, season)

        return results, season_metrics

    def _generate_baseline_predictions(self, actuals: pd.DataFrame, stat_type: str) -> pd.DataFrame:
        """Generate baseline predictions using historical averages when Bayesian models unavailable."""

        logger.info("Generating baseline predictions from historical data")

        # Calculate rolling averages for each player
        actuals_sorted = actuals.sort_values(["player_id", "season", "week"])

        baseline_preds = []
        for player_id in actuals_sorted["player_id"].unique():
            player_data = actuals_sorted[actuals_sorted["player_id"] == player_id].copy()

            # Use expanding mean (all previous weeks)
            player_data["bayes_prediction"] = (
                player_data["actual_yards"].expanding(min_periods=1).mean().shift(1)
            )

            # For first game, use position average
            first_game_mask = player_data["bayes_prediction"].isna()
            if first_game_mask.any() and "player_position" in player_data.columns:
                pos = player_data["player_position"].iloc[0]
                pos_avg = actuals[actuals["player_position"] == pos]["actual_yards"].mean()
                player_data.loc[first_game_mask, "bayes_prediction"] = pos_avg

            # Estimate uncertainty as standard deviation of recent games
            player_data["bayes_uncertainty"] = (
                player_data["actual_yards"].expanding(min_periods=2).std().shift(1)
            )
            player_data["bayes_uncertainty"] = player_data["bayes_uncertainty"].fillna(
                50
            )  # Default uncertainty

            baseline_preds.append(player_data)

        baseline_df = pd.concat(baseline_preds, ignore_index=True)

        return baseline_df[["player_id", "season", "bayes_prediction", "bayes_uncertainty"]]

    def generate_comparative_report(
        self,
        results: pd.DataFrame,
        season_metrics: dict,
        stat_type: str,
        output_dir: str = "reports/bayesian_backtest_multiyear",
    ):
        """Generate comprehensive multi-year comparative report."""

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create visualizations
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. MAE by Season
        ax1 = fig.add_subplot(gs[0, 0])
        seasons = sorted(season_metrics.keys())
        maes = [season_metrics[s].get("mae", 0) for s in seasons]
        ax1.bar(seasons, maes, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(seasons)])
        ax1.set_xlabel("Season")
        ax1.set_ylabel("MAE (yards)")
        ax1.set_title("MAE by Season")
        ax1.set_xticks(seasons)
        for i, (s, mae) in enumerate(zip(seasons, maes)):
            ax1.text(s, mae + 1, f"{mae:.1f}", ha="center", va="bottom")

        # 2. Correlation by Season
        ax2 = fig.add_subplot(gs[0, 1])
        correlations = [season_metrics[s].get("correlation", 0) for s in seasons]
        ax2.bar(seasons, correlations, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(seasons)])
        ax2.set_xlabel("Season")
        ax2.set_ylabel("Correlation")
        ax2.set_title("Prediction Correlation by Season")
        ax2.set_xticks(seasons)
        ax2.set_ylim([0, 1])
        for i, (s, corr) in enumerate(zip(seasons, correlations)):
            ax2.text(s, corr + 0.02, f"{corr:.3f}", ha="center", va="bottom")

        # 3. Coverage by Season
        ax3 = fig.add_subplot(gs[0, 2])
        coverage_90 = [season_metrics[s].get("ci_90_coverage", 0) for s in seasons]
        x = np.arange(len(seasons))
        width = 0.35
        ax3.bar(x - width / 2, coverage_90, width, label="90% CI", color="steelblue")
        ax3.axhline(y=0.90, color="r", linestyle="--", label="Expected 90%")
        ax3.set_xlabel("Season")
        ax3.set_ylabel("Coverage Rate")
        ax3.set_title("Credible Interval Coverage by Season")
        ax3.set_xticks(x)
        ax3.set_xticklabels(seasons)
        ax3.legend()
        ax3.set_ylim([0, 1])

        # 4. Sample Size by Season
        ax4 = fig.add_subplot(gs[0, 3])
        sample_sizes = [season_metrics[s].get("n_predictions", 0) for s in seasons]
        ax4.bar(seasons, sample_sizes, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(seasons)])
        ax4.set_xlabel("Season")
        ax4.set_ylabel("Number of Predictions")
        ax4.set_title("Sample Size by Season")
        ax4.set_xticks(seasons)
        for i, (s, n) in enumerate(zip(seasons, sample_sizes)):
            ax4.text(s, n + 50, f"{n:,}", ha="center", va="bottom")

        # 5. Error Distribution by Season
        ax5 = fig.add_subplot(gs[1, :2])
        for season in seasons:
            season_data = results[results["season"] == season]
            if not season_data.empty:
                ax5.hist(season_data["error"], bins=50, alpha=0.5, label=f"{season}", density=True)
        ax5.axvline(0, color="black", linestyle="--", linewidth=2)
        ax5.set_xlabel("Prediction Error (yards)")
        ax5.set_ylabel("Density")
        ax5.set_title("Error Distribution Comparison")
        ax5.legend()

        # 6. Actual vs Predicted by Season
        ax6 = fig.add_subplot(gs[1, 2:])
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, season in enumerate(seasons):
            season_data = results[results["season"] == season]
            if not season_data.empty:
                ax6.scatter(
                    season_data["actual_yards"],
                    season_data["bayes_prediction"],
                    alpha=0.3,
                    s=10,
                    label=f"{season}",
                    color=colors[i],
                )

        max_val = results[["actual_yards", "bayes_prediction"]].max().max()
        ax6.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect Prediction")
        ax6.set_xlabel("Actual Yards")
        ax6.set_ylabel("Predicted Yards")
        ax6.set_title("Predictions vs Actuals (All Seasons)")
        ax6.legend()

        # 7. Weekly MAE Trend Across Seasons
        ax7 = fig.add_subplot(gs[2, :2])
        for season in seasons:
            season_data = results[results["season"] == season]
            if not season_data.empty and "week" in season_data.columns:
                weekly_mae = season_data.groupby("week").apply(
                    lambda x: mean_absolute_error(x["actual_yards"], x["bayes_prediction"])
                )
                ax7.plot(weekly_mae.index, weekly_mae.values, "o-", label=f"{season}", alpha=0.7)

        ax7.set_xlabel("Week")
        ax7.set_ylabel("MAE (yards)")
        ax7.set_title("Weekly MAE Trend (All Seasons)")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Position-specific Performance
        ax8 = fig.add_subplot(gs[2, 2:])
        if "player_position" in results.columns:
            position_performance = []
            for season in seasons:
                season_data = results[results["season"] == season]
                for pos in season_data["player_position"].unique():
                    pos_data = season_data[season_data["player_position"] == pos]
                    if len(pos_data) >= 10:
                        position_performance.append(
                            {
                                "season": season,
                                "position": pos,
                                "mae": mean_absolute_error(
                                    pos_data["actual_yards"], pos_data["bayes_prediction"]
                                ),
                            }
                        )

            if position_performance:
                perf_df = pd.DataFrame(position_performance)
                pivot = perf_df.pivot(index="position", columns="season", values="mae")
                pivot.plot(kind="bar", ax=ax8)
                ax8.set_xlabel("Position")
                ax8.set_ylabel("MAE (yards)")
                ax8.set_title("Position-Specific MAE by Season")
                ax8.legend(title="Season")
                ax8.tick_params(axis="x", rotation=45)

        plt.suptitle(
            f'{stat_type.replace("_", " ").title()} - Multi-Year Backtest Report',
            fontsize=16,
            fontweight="bold",
        )

        plt.savefig(f"{output_dir}/{stat_type}_multiyear_report.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save detailed results
        results.to_csv(f"{output_dir}/{stat_type}_detailed_results.csv", index=False)

        # Save season metrics
        with open(f"{output_dir}/{stat_type}_season_metrics.json", "w") as f:
            json.dump(season_metrics, f, indent=2, default=str)

        # Generate text summary
        with open(f"{output_dir}/{stat_type}_summary.txt", "w") as f:
            f.write(f"Multi-Year Backtest Summary: {stat_type}\n")
            f.write("=" * 80 + "\n\n")

            for season in sorted(season_metrics.keys()):
                metrics = season_metrics[season]
                f.write(f"{season} Season:\n")
                f.write(f"  Predictions: {metrics.get('n_predictions', 0):,}\n")
                f.write(f"  MAE: {metrics.get('mae', 0):.2f} yards\n")
                f.write(f"  RMSE: {metrics.get('rmse', 0):.2f} yards\n")
                f.write(f"  MAPE: {metrics.get('mape', 0):.1f}%\n")
                f.write(f"  Correlation: {metrics.get('correlation', 0):.3f}\n")
                f.write(f"  90% CI Coverage: {metrics.get('ci_90_coverage', 0):.1%}\n")
                f.write("\n")

            # Overall statistics
            f.write("\nOverall Statistics:\n")
            f.write(f"  Total Predictions: {len(results):,}\n")
            f.write(
                f"  Overall MAE: {mean_absolute_error(results['actual_yards'], results['bayes_prediction']):.2f} yards\n"
            )
            f.write(
                f"  Overall RMSE: {np.sqrt(mean_squared_error(results['actual_yards'], results['bayes_prediction'])):.2f} yards\n"
            )
            f.write(
                f"  Overall Correlation: {results[['actual_yards', 'bayes_prediction']].corr().iloc[0, 1]:.3f}\n"
            )

        logger.info(f"Multi-year report saved to {output_dir}")

    def run_full_multiyear_backtest(
        self,
        stat_types: list[str] = ["passing_yards", "rushing_yards", "receiving_yards"],
        seasons: list[int] = [2022, 2023, 2024],
    ):
        """Run complete multi-year backtest for all stat types."""

        print("\n" + "=" * 80)
        print("MULTI-YEAR BAYESIAN HIERARCHICAL BACKTEST")
        print(f"Seasons: {seasons}")
        print("=" * 80)

        all_results = {}
        all_metrics = {}

        for stat_type in stat_types:
            try:
                results, season_metrics = self.run_multiyear_backtest(
                    stat_type=stat_type, seasons=seasons, start_week=1, end_week=17
                )

                if not results.empty:
                    all_results[stat_type] = results
                    all_metrics[stat_type] = season_metrics

                    # Generate report
                    self.generate_comparative_report(
                        results,
                        season_metrics,
                        stat_type,
                        output_dir=f"reports/bayesian_backtest_multiyear/{stat_type}",
                    )

                    # Print summary
                    print(f"\n{stat_type.upper()} - Season Comparison:")
                    print("-" * 60)
                    for season in sorted(season_metrics.keys()):
                        metrics = season_metrics[season]
                        print(
                            f"  {season}: MAE={metrics.get('mae', 0):.2f}, "
                            f"Corr={metrics.get('correlation', 0):.3f}, "
                            f"Coverage={metrics.get('ci_90_coverage', 0):.1%}"
                        )

            except Exception as e:
                logger.error(f"Error processing {stat_type}: {e}", exc_info=True)
                continue

        # Save combined summary
        summary_dir = "reports/bayesian_backtest_multiyear"
        Path(summary_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{summary_dir}/combined_summary.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("MULTI-YEAR BACKTEST COMPLETE")
        print(f"Reports saved to {summary_dir}")
        print("=" * 80)

        return all_results, all_metrics


def main():
    """Run multi-year backtesting suite."""

    backtest = MultiYearBacktest()

    try:
        # Run full backtest for 2022-2024
        results, metrics = backtest.run_full_multiyear_backtest(
            stat_types=["passing_yards", "rushing_yards", "receiving_yards"],
            seasons=[2022, 2023, 2024],
        )

        print("\n✅ Multi-year backtest complete!")
        print("\nKey Findings:")
        for stat_type, stat_metrics in metrics.items():
            print(f"\n{stat_type}:")
            for season, season_data in stat_metrics.items():
                print(f"  {season}: MAE = {season_data.get('mae', 0):.2f} yards")

    finally:
        backtest.conn.close()


if __name__ == "__main__":
    main()
