#!/usr/bin/env python3
"""
Backtest Bayesian Hierarchical Player Props Models on 2024 Season

This script performs comprehensive backtesting of the Bayesian hierarchical
models for player props predictions on the 2024 NFL season.

Metrics evaluated:
- MAE, RMSE, MAPE for point predictions
- Coverage of credible intervals
- Calibration of uncertainty estimates
- Comparison with XGBoost baseline
- Kelly Criterion betting performance
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from features.bayesian_player_features import BayesianPlayerFeatures

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


class BayesianPropsBacktest:
    """Backtest Bayesian hierarchical models for player props."""

    def __init__(self, db_config: dict = None):
        """Initialize backtester with database configuration."""
        self.db_config = db_config or DB_CONFIG
        self.conn = None
        self.feature_extractor = BayesianPlayerFeatures(db_config)
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to database for backtesting")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def fetch_actual_outcomes(
        self, stat_type: str, season: int, weeks: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Fetch actual player outcomes for backtesting.

        Args:
            stat_type: Type of stat ('passing_yards', 'rushing_yards', 'receiving_yards')
            season: Season to fetch
            weeks: Optional list of weeks to include

        Returns:
            DataFrame with actual outcomes
        """
        stat_column_map = {
            "passing_yards": "pass_yards",
            "rushing_yards": "rush_yards",
            "receiving_yards": "receiving_yards",
        }

        stat_col = stat_column_map.get(stat_type)
        if not stat_col:
            raise ValueError(f"Unknown stat type: {stat_type}")

        # Determine table based on stat type
        if stat_type == "passing_yards":
            table = "nextgen_passing"
        elif stat_type == "rushing_yards":
            table = "nextgen_rushing"
        else:
            table = "nextgen_receiving"

        query = f"""
        SELECT
            player_id,
            player_display_name,
            player_position,
            season,
            week,
            team_abbr,
            {stat_col} as actual_yards,
            -- Additional context
            CASE
                WHEN {table} = 'nextgen_passing' THEN attempts
                WHEN {table} = 'nextgen_rushing' THEN carries
                ELSE targets
            END as volume
        FROM {table}
        WHERE season = %s
            AND {stat_col} IS NOT NULL
            {' AND week IN %s' if weeks else ''}
        ORDER BY week, player_id
        """

        params = (season,) if not weeks else (season, tuple(weeks))

        try:
            df = pd.read_sql_query(query, self.conn, params=params)
            logger.info(f"Fetched {len(df)} actual outcomes for {stat_type} in {season}")
            return df
        except Exception as e:
            logger.error(f"Error fetching actual outcomes: {e}")
            return pd.DataFrame()

    def run_weekly_backtest(
        self, stat_type: str, season: int = 2024, start_week: int = 1, end_week: int = 17
    ) -> pd.DataFrame:
        """
        Run week-by-week backtest simulating real-time predictions.

        Args:
            stat_type: Type of stat to backtest
            season: Season to backtest
            start_week: First week to include
            end_week: Last week to include

        Returns:
            DataFrame with predictions and actuals for analysis
        """
        results = []

        for week in range(start_week, end_week + 1):
            logger.info(f"Backtesting {stat_type} for Week {week}, {season}")

            # Get actual outcomes for this week
            actuals = self.fetch_actual_outcomes(stat_type=stat_type, season=season, weeks=[week])

            if actuals.empty:
                logger.warning(f"No actuals found for Week {week}")
                continue

            # Get Bayesian predictions for these players
            player_ids = actuals["player_id"].unique().tolist()
            predictions = self.feature_extractor.get_player_props_features(
                players=player_ids, stat_type=stat_type, season=season, week=week
            )

            # Merge predictions with actuals
            week_results = actuals.merge(predictions, on="player_id", how="inner")

            # Add week identifier
            week_results["week"] = week

            # Calculate errors
            week_results["error"] = week_results["bayes_prediction"] - week_results["actual_yards"]
            week_results["abs_error"] = np.abs(week_results["error"])
            week_results["squared_error"] = week_results["error"] ** 2
            week_results["pct_error"] = np.abs(
                week_results["error"] / week_results["actual_yards"].clip(lower=1)
            )

            # Check if actual falls within credible intervals
            week_results["in_ci_90"] = (
                (week_results["actual_yards"] >= week_results["bayes_ci_lower"])
                & (week_results["actual_yards"] <= week_results["bayes_ci_upper"])
            ).astype(int)

            week_results["in_ci_68"] = (
                (week_results["actual_yards"] >= week_results["bayes_conservative"])
                & (week_results["actual_yards"] <= week_results["bayes_aggressive"])
            ).astype(int)

            results.append(week_results)

        # Combine all weeks
        if results:
            all_results = pd.concat(results, ignore_index=True)
            logger.info(f"Completed backtest: {len(all_results)} predictions evaluated")
            return all_results
        else:
            logger.warning("No results generated from backtest")
            return pd.DataFrame()

    def calculate_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            results: DataFrame with predictions and actuals

        Returns:
            Dictionary of metrics
        """
        if results.empty:
            return {}

        metrics = {}

        # Point estimate accuracy
        metrics["mae"] = mean_absolute_error(results["actual_yards"], results["bayes_prediction"])
        metrics["rmse"] = np.sqrt(
            mean_squared_error(results["actual_yards"], results["bayes_prediction"])
        )
        metrics["mape"] = np.mean(results["pct_error"]) * 100

        # Correlation
        metrics["correlation"] = results[["actual_yards", "bayes_prediction"]].corr().iloc[0, 1]

        # Uncertainty calibration
        metrics["ci_90_coverage"] = results["in_ci_90"].mean()
        metrics["ci_68_coverage"] = results["in_ci_68"].mean()

        # Expected vs actual coverage
        metrics["ci_90_calibration_error"] = abs(0.90 - metrics["ci_90_coverage"])
        metrics["ci_68_calibration_error"] = abs(0.68 - metrics["ci_68_coverage"])

        # Breakdown by data strength (players with more/less data)
        high_data = results[results["bayes_data_strength"] > 0.5]
        low_data = results[results["bayes_data_strength"] <= 0.5]

        if not high_data.empty:
            metrics["mae_high_data"] = mean_absolute_error(
                high_data["actual_yards"], high_data["bayes_prediction"]
            )

        if not low_data.empty:
            metrics["mae_low_data"] = mean_absolute_error(
                low_data["actual_yards"], low_data["bayes_prediction"]
            )

        # Breakdown by position (if available)
        if "player_position" in results.columns:
            for pos in results["player_position"].unique():
                pos_data = results[results["player_position"] == pos]
                if len(pos_data) > 10:  # Minimum sample size
                    metrics[f"mae_{pos}"] = mean_absolute_error(
                        pos_data["actual_yards"], pos_data["bayes_prediction"]
                    )

        # Reliability score validation
        # Higher reliability should correlate with lower errors
        if "bayes_reliability_score" in results.columns:
            reliability_correlation = (
                results[["abs_error", "bayes_reliability_score"]].corr().iloc[0, 1]
            )
            metrics["reliability_validity"] = (
                -reliability_correlation
            )  # Should be positive if valid

        return metrics

    def compare_with_baseline(
        self, results: pd.DataFrame, baseline_predictions: pd.DataFrame | None = None
    ) -> dict:
        """
        Compare Bayesian model with baseline (e.g., XGBoost).

        Args:
            results: DataFrame with Bayesian predictions
            baseline_predictions: Optional baseline predictions to compare

        Returns:
            Comparison metrics
        """
        comparison = {}

        # If no baseline provided, use simple historical average
        if baseline_predictions is None:
            # Simple baseline: player's average from previous weeks
            baseline = results.groupby("player_id")["actual_yards"].shift(1)
            results["baseline_prediction"] = baseline.fillna(results["actual_yards"].mean())
        else:
            results = results.merge(
                baseline_predictions[["player_id", "week", "baseline_prediction"]],
                on=["player_id", "week"],
                how="left",
            )

        # Calculate metrics for both models
        valid_mask = results["baseline_prediction"].notna()
        valid_results = results[valid_mask]

        if not valid_results.empty:
            comparison["bayes_mae"] = mean_absolute_error(
                valid_results["actual_yards"], valid_results["bayes_prediction"]
            )
            comparison["baseline_mae"] = mean_absolute_error(
                valid_results["actual_yards"], valid_results["baseline_prediction"]
            )
            comparison["mae_improvement"] = (
                (comparison["baseline_mae"] - comparison["bayes_mae"])
                / comparison["baseline_mae"]
                * 100
            )

            # Win rate (which model was closer)
            bayes_closer = np.abs(
                valid_results["bayes_prediction"] - valid_results["actual_yards"]
            ) < np.abs(valid_results["baseline_prediction"] - valid_results["actual_yards"])
            comparison["bayes_win_rate"] = bayes_closer.mean()

        return comparison

    def simulate_betting_performance(
        self,
        results: pd.DataFrame,
        props_lines: pd.DataFrame | None = None,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
    ) -> dict:
        """
        Simulate betting performance using Kelly Criterion.

        Args:
            results: DataFrame with predictions
            props_lines: Optional DataFrame with actual prop lines
            bankroll: Starting bankroll
            kelly_fraction: Fraction of Kelly to use (for safety)

        Returns:
            Betting performance metrics
        """
        if props_lines is None:
            # Simulate prop lines as small variations from predictions
            # In practice, you'd use actual sportsbook lines
            results["prop_line"] = results["bayes_prediction"] * np.random.uniform(
                0.95, 1.05, len(results)
            )
        else:
            results = results.merge(
                props_lines[["player_id", "week", "prop_line"]],
                on=["player_id", "week"],
                how="left",
            )

        current_bankroll = bankroll
        bets_placed = 0
        bets_won = 0
        total_wagered = 0
        bankroll_history = [bankroll]

        for _, row in results.iterrows():
            if pd.isna(row.get("prop_line")):
                continue

            # Calculate edge using Bayesian prediction
            # Assume over/under with -110 odds
            implied_prob = 0.5238  # -110 odds
            actual_prob_over = stats.norm.cdf(
                row["prop_line"], loc=row["bayes_prediction"], scale=row["bayes_uncertainty"]
            )
            actual_prob_under = 1 - actual_prob_over

            # Determine which side to bet
            if actual_prob_over > implied_prob:
                # Bet over
                edge = actual_prob_over - implied_prob
                bet_size = kelly_fraction * edge * current_bankroll / (1 / implied_prob - 1)
                bet_size = min(bet_size, current_bankroll * 0.05)  # Max 5% per bet

                if bet_size > 0:
                    won = row["actual_yards"] > row["prop_line"]
                    payout = bet_size * (1 / implied_prob - 1) if won else -bet_size
                    current_bankroll += payout
                    bets_placed += 1
                    bets_won += int(won)
                    total_wagered += bet_size

            elif actual_prob_under > implied_prob:
                # Bet under
                edge = actual_prob_under - implied_prob
                bet_size = kelly_fraction * edge * current_bankroll / (1 / implied_prob - 1)
                bet_size = min(bet_size, current_bankroll * 0.05)

                if bet_size > 0:
                    won = row["actual_yards"] < row["prop_line"]
                    payout = bet_size * (1 / implied_prob - 1) if won else -bet_size
                    current_bankroll += payout
                    bets_placed += 1
                    bets_won += int(won)
                    total_wagered += bet_size

            bankroll_history.append(current_bankroll)

        return {
            "final_bankroll": current_bankroll,
            "total_return": (current_bankroll - bankroll) / bankroll * 100,
            "bets_placed": bets_placed,
            "win_rate": bets_won / bets_placed if bets_placed > 0 else 0,
            "avg_bet_size": total_wagered / bets_placed if bets_placed > 0 else 0,
            "max_bankroll": max(bankroll_history),
            "min_bankroll": min(bankroll_history),
            "max_drawdown": (max(bankroll_history) - min(bankroll_history))
            / max(bankroll_history)
            * 100,
        }

    def generate_report(
        self, results: pd.DataFrame, metrics: dict, output_dir: str = "reports/bayesian_backtest"
    ):
        """Generate comprehensive backtest report with visualizations."""

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle("Bayesian Hierarchical Model Backtest Report - 2024 Season", fontsize=16)

        # 1. Actual vs Predicted scatter
        ax = axes[0, 0]
        ax.scatter(results["actual_yards"], results["bayes_prediction"], alpha=0.5)
        ax.plot([0, results["actual_yards"].max()], [0, results["actual_yards"].max()], "r--")
        ax.set_xlabel("Actual Yards")
        ax.set_ylabel("Predicted Yards")
        ax.set_title(f'Predictions vs Actuals (r={metrics.get("correlation", 0):.3f})')

        # 2. Error distribution
        ax = axes[0, 1]
        ax.hist(results["error"], bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="Zero Error")
        ax.set_xlabel("Prediction Error (yards)")
        ax.set_ylabel("Frequency")
        ax.set_title(f'Error Distribution (MAE={metrics.get("mae", 0):.1f})')
        ax.legend()

        # 3. Calibration plot
        ax = axes[0, 2]
        coverage_levels = [0.5, 0.68, 0.9, 0.95]
        actual_coverage = []
        for level in coverage_levels:
            lower = (
                results["bayes_prediction"]
                - stats.norm.ppf((1 + level) / 2) * results["bayes_uncertainty"]
            )
            upper = (
                results["bayes_prediction"]
                + stats.norm.ppf((1 + level) / 2) * results["bayes_uncertainty"]
            )
            coverage = (
                (results["actual_yards"] >= lower) & (results["actual_yards"] <= upper)
            ).mean()
            actual_coverage.append(coverage)

        ax.plot(coverage_levels, coverage_levels, "k--", label="Perfect Calibration")
        ax.plot(coverage_levels, actual_coverage, "bo-", label="Actual Coverage")
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Actual Coverage")
        ax.set_title("Uncertainty Calibration")
        ax.legend()

        # 4. Weekly MAE trend
        ax = axes[1, 0]
        weekly_mae = results.groupby("week").apply(
            lambda x: mean_absolute_error(x["actual_yards"], x["bayes_prediction"])
        )
        ax.plot(weekly_mae.index, weekly_mae.values, "o-")
        ax.set_xlabel("Week")
        ax.set_ylabel("MAE")
        ax.set_title("Weekly MAE Trend")
        ax.grid(True)

        # 5. Error by data strength
        ax = axes[1, 1]
        bins = pd.qcut(
            results["bayes_data_strength"],
            q=5,
            labels=["Very Low", "Low", "Medium", "High", "Very High"],
        )
        error_by_strength = results.groupby(bins)["abs_error"].mean()
        ax.bar(range(len(error_by_strength)), error_by_strength.values)
        ax.set_xticks(range(len(error_by_strength)))
        ax.set_xticklabels(error_by_strength.index, rotation=45)
        ax.set_xlabel("Data Strength")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Error by Data Availability")

        # 6. Reliability score validation
        ax = axes[1, 2]
        ax.scatter(results["bayes_reliability_score"], results["abs_error"], alpha=0.3)
        z = np.polyfit(results["bayes_reliability_score"], results["abs_error"], 1)
        p = np.poly1d(z)
        ax.plot(
            results["bayes_reliability_score"].sort_values(),
            p(results["bayes_reliability_score"].sort_values()),
            "r-",
            alpha=0.8,
        )
        ax.set_xlabel("Reliability Score")
        ax.set_ylabel("Absolute Error")
        ax.set_title("Reliability Score Validation")

        # 7. Error by position (if available)
        ax = axes[2, 0]
        if "player_position" in results.columns:
            position_errors = results.groupby("player_position")["abs_error"].mean().sort_values()
            ax.barh(range(len(position_errors)), position_errors.values)
            ax.set_yticks(range(len(position_errors)))
            ax.set_yticklabels(position_errors.index)
            ax.set_xlabel("Mean Absolute Error")
            ax.set_title("Error by Position")
        else:
            ax.text(0.5, 0.5, "Position data not available", ha="center", va="center")

        # 8. Prediction intervals
        ax = axes[2, 1]
        sample_players = results.nlargest(20, "volume")  # Top 20 by volume
        x_pos = range(len(sample_players))
        ax.errorbar(
            x_pos,
            sample_players["bayes_prediction"],
            yerr=sample_players["bayes_uncertainty"] * 1.96,
            fmt="o",
            alpha=0.6,
            label="95% CI",
        )
        ax.scatter(
            x_pos, sample_players["actual_yards"], color="red", marker="x", s=50, label="Actual"
        )
        ax.set_xticks([])
        ax.set_xlabel("Players (top 20 by volume)")
        ax.set_ylabel("Yards")
        ax.set_title("Sample Predictions with Uncertainty")
        ax.legend()

        # 9. Summary metrics table
        ax = axes[2, 2]
        ax.axis("off")
        summary_text = f"""
        Summary Metrics:
        ----------------
        MAE: {metrics.get('mae', 0):.2f} yards
        RMSE: {metrics.get('rmse', 0):.2f} yards
        MAPE: {metrics.get('mape', 0):.1f}%
        Correlation: {metrics.get('correlation', 0):.3f}

        Coverage:
        90% CI: {metrics.get('ci_90_coverage', 0):.1%}
        68% CI: {metrics.get('ci_68_coverage', 0):.1%}

        Data Splits:
        High Data MAE: {metrics.get('mae_high_data', 0):.2f}
        Low Data MAE: {metrics.get('mae_low_data', 0):.2f}
        """
        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plt.savefig(f"{output_dir}/backtest_report.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save detailed results
        results.to_csv(f"{output_dir}/detailed_results.csv", index=False)

        # Save metrics
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Report saved to {output_dir}")

    def run_full_backtest(
        self,
        stat_types: list[str] = ["passing_yards", "rushing_yards", "receiving_yards"],
        season: int = 2024,
    ):
        """Run complete backtest for all stat types."""

        all_metrics = {}
        all_results = {}

        for stat_type in stat_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Backtesting {stat_type}")
            logger.info("=" * 60)

            # Run weekly backtest
            results = self.run_weekly_backtest(
                stat_type=stat_type, season=season, start_week=1, end_week=17
            )

            if results.empty:
                logger.warning(f"No results for {stat_type}")
                continue

            # Calculate metrics
            metrics = self.calculate_metrics(results)

            # Compare with baseline
            comparison = self.compare_with_baseline(results)
            metrics.update(comparison)

            # Simulate betting
            betting_performance = self.simulate_betting_performance(results)
            metrics["betting"] = betting_performance

            # Store results
            all_metrics[stat_type] = metrics
            all_results[stat_type] = results

            # Generate report
            self.generate_report(
                results, metrics, output_dir=f"reports/bayesian_backtest/{stat_type}"
            )

            # Print summary
            print(f"\n{stat_type.upper()} Summary:")
            print(f"  MAE: {metrics.get('mae', 0):.2f} yards")
            print(f"  RMSE: {metrics.get('rmse', 0):.2f} yards")
            print(f"  90% CI Coverage: {metrics.get('ci_90_coverage', 0):.1%}")
            print(f"  Correlation: {metrics.get('correlation', 0):.3f}")

            if "baseline_mae" in metrics:
                print(f"  vs Baseline: {metrics.get('mae_improvement', 0):+.1f}% improvement")
                print(f"  Win Rate: {metrics.get('bayes_win_rate', 0):.1%}")

            if "betting" in metrics:
                betting = metrics["betting"]
                print(f"  Betting ROI: {betting.get('total_return', 0):+.1f}%")
                print(f"  Win Rate: {betting.get('win_rate', 0):.1%}")

        # Save combined report
        with open("reports/bayesian_backtest/combined_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        return all_metrics, all_results


def main():
    """Run full backtesting suite."""

    print("\n" + "=" * 80)
    print("BAYESIAN HIERARCHICAL MODELS - 2024 SEASON BACKTEST")
    print("=" * 80)

    # Initialize backtester
    backtester = BayesianPropsBacktest()

    try:
        # Run full backtest
        metrics, results = backtester.run_full_backtest(
            stat_types=["passing_yards", "rushing_yards", "receiving_yards"], season=2024
        )

        # Print final summary
        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE")
        print("=" * 80)

        for stat_type, stat_metrics in metrics.items():
            print(f"\n{stat_type}:")
            print(f"  Final MAE: {stat_metrics.get('mae', 0):.2f}")
            print(f"  Coverage: {stat_metrics.get('ci_90_coverage', 0):.1%}")

            if "betting" in stat_metrics:
                print(f"  Betting ROI: {stat_metrics['betting'].get('total_return', 0):+.1f}%")

        print("\nReports saved to reports/bayesian_backtest/")

    finally:
        backtester.conn.close()
        backtester.feature_extractor.close()

    print("\n" + "=" * 80)
    print("Backtest complete! Ready for user review.")
    print("=" * 80)


if __name__ == "__main__":
    main()
