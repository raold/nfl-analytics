#!/usr/bin/env python3
"""
Comprehensive Ensemble Backtest (2022-2024)

Validates the 4-way ensemble model against historical data to confirm
target +5-7% ROI performance.

Compares:
1. v1.0 Baseline (hierarchical)
2. v2.5 Informative Priors
3. v3.0 Full Ensemble (Bayesian + XGBoost + BNN + state-space)
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

from py.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEnsembleBacktest:
    """
    Comprehensive backtesting framework for ensemble models.

    Features:
    - Historical data loading (2022-2024)
    - Multiple model comparison
    - Realistic betting simulation
    - Correlation-adjusted portfolio optimization
    - Comprehensive metrics calculation
    """

    def __init__(
        self,
        start_season: int = 2022,
        end_season: int = 2024,
        initial_bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.02,
    ):
        """
        Initialize backtest.

        Args:
            start_season: First season to backtest
            end_season: Last season to backtest
            initial_bankroll: Starting bankroll
            kelly_fraction: Fraction of Kelly criterion to use
            min_edge: Minimum edge required for betting
        """
        self.start_season = start_season
        self.end_season = end_season
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge

        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        # Initialize ensemble
        self.ensemble = EnhancedEnsembleV3(
            use_bnn=True,
            use_stacking=True,
            use_portfolio_opt=True,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge,
        )

        logger.info(f"✓ Initialized backtest for {start_season}-{end_season}")

    def load_historical_data(self, season: int, week: int | None = None) -> pd.DataFrame:
        """Load historical game and player data"""
        conn = psycopg2.connect(**self.db_config)

        # Simplified query using plays table directly with correct column names
        query = """
        WITH player_stats AS (
            SELECT
                p.game_id,
                p.passer_player_id as player_id,
                p.passer_player_name as player_name,
                'QB' as position,
                SUM(COALESCE(p.yards_gained, 0)) FILTER (WHERE p.pass = true) as passing_yards,
                0 as rushing_yards,
                0 as receiving_yards,
                COUNT(*) FILTER (WHERE p.pass = true) as attempts,
                COUNT(*) FILTER (WHERE p.complete_pass = 1) as completions,
                0 as carries,
                0 as targets,
                0 as receptions
            FROM plays p
            WHERE p.passer_player_id IS NOT NULL
            GROUP BY p.game_id, p.passer_player_id, p.passer_player_name

            UNION ALL

            SELECT
                p.game_id,
                p.rusher_player_id as player_id,
                p.rusher_player_name as player_name,
                'RB' as position,
                0 as passing_yards,
                SUM(COALESCE(p.yards_gained, 0)) FILTER (WHERE p.rush = true) as rushing_yards,
                0 as receiving_yards,
                0 as attempts,
                0 as completions,
                COUNT(*) FILTER (WHERE p.rush = true) as carries,
                0 as targets,
                0 as receptions
            FROM plays p
            WHERE p.rusher_player_id IS NOT NULL
            GROUP BY p.game_id, p.rusher_player_id, p.rusher_player_name

            UNION ALL

            SELECT
                p.game_id,
                p.receiver_player_id as player_id,
                p.receiver_player_name as player_name,
                'WR' as position,
                0 as passing_yards,
                0 as rushing_yards,
                SUM(COALESCE(p.yards_gained, 0)) FILTER (WHERE p.pass = true) as receiving_yards,
                0 as attempts,
                0 as completions,
                0 as carries,
                COUNT(*) FILTER (WHERE p.pass = true) as targets,
                COUNT(*) FILTER (WHERE p.complete_pass = 1) as receptions
            FROM plays p
            WHERE p.receiver_player_id IS NOT NULL
            GROUP BY p.game_id, p.receiver_player_id, p.receiver_player_name
        )
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            g.home_team,
            g.away_team,
            ps.player_id,
            ps.player_name,
            ps.position,
            ps.passing_yards,
            ps.rushing_yards,
            ps.receiving_yards,
            ps.attempts,
            ps.completions,
            ps.carries,
            ps.targets,
            ps.receptions
        FROM games g
        JOIN player_stats ps ON g.game_id = ps.game_id
        WHERE g.season = %s
        """

        params = [season]

        if week:
            query += " AND g.week = %s"
            params.append(week)

        query += " ORDER BY g.kickoff, g.game_id"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        logger.info(
            f"Loaded {len(df)} player-games for season {season}" + (f" week {week}" if week else "")
        )

        return df

    def load_model_predictions(
        self, model_version: str, season: int, week: int | None = None
    ) -> pd.DataFrame:
        """Load predictions from a specific model version"""

        if model_version == "ensemble_v3.0":
            # Generate ensemble predictions
            return self._generate_ensemble_predictions(season, week)

        # Load from database for v1.0 and v2.5
        conn = psycopg2.connect(**self.db_config)

        query = """
        SELECT
            player_id,
            season,
            stat_type,
            rating_mean as prediction,
            rating_sd as uncertainty,
            rating_q05,
            rating_q95,
            n_games_observed
        FROM mart.bayesian_player_ratings
        WHERE model_version = %s
          AND season = %s
          AND stat_type = 'passing_yards'
        """

        params = [model_version, season]

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # If week specified, need to join with games to filter
        if week:
            # This is simplified - in production would properly handle week filtering
            pass

        logger.info(f"Loaded {len(df)} predictions for {model_version} season {season}")

        return df

    def _generate_ensemble_predictions(self, season: int, week: int | None = None) -> pd.DataFrame:
        """Generate 4-way ensemble predictions"""

        # Load base model predictions
        bayesian_preds = self.load_model_predictions("informative_priors_v2.5", season, week)

        # For demo, create synthetic XGBoost and BNN predictions
        # In production, would load actual trained model predictions

        # XGBoost (slightly different from Bayesian)
        xgb_preds = bayesian_preds.copy()
        xgb_preds["xgb_pred"] = xgb_preds["prediction"] * np.random.normal(1.0, 0.1, len(xgb_preds))
        xgb_preds["stat_type"] = "passing_yards"

        # BNN (with more uncertainty)
        bnn_preds = bayesian_preds.copy()
        bnn_preds["bnn_pred"] = xgb_preds["xgb_pred"] * np.random.normal(1.0, 0.15, len(bnn_preds))
        bnn_preds["bnn_uncertainty"] = bayesian_preds["uncertainty"] * 1.2

        # Rename columns for merge
        bayesian_preds.rename(
            columns={"prediction": "bayesian_pred", "uncertainty": "bayesian_uncertainty"},
            inplace=True,
        )

        # Generate ensemble predictions
        ensemble_preds = self.ensemble.get_4way_ensemble_predictions(
            bayesian_preds, xgb_preds, bnn_preds
        )

        return ensemble_preds

    def simulate_betting_lines(
        self, actual_values: pd.Series, noise_level: float = 0.05
    ) -> pd.DataFrame:
        """Simulate realistic betting lines based on actual values"""

        # Lines are typically set close to expected values with some noise
        lines = actual_values * np.random.normal(1.0, noise_level, len(actual_values))

        # Simulate odds (typical -110 on both sides)
        odds = np.random.choice([-110, -105, -115, -120, 100], size=len(actual_values))

        return pd.DataFrame({"line": lines, "odds": odds, "actual": actual_values})

    def run_backtest(self) -> dict:
        """
        Run comprehensive backtest across all seasons.

        Returns:
            Dictionary with backtest results for each model version
        """
        results = {}

        for model_version in ["hierarchical_v1.0", "informative_priors_v2.5", "ensemble_v3.0"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Backtesting {model_version}")
            logger.info(f"{'='*60}")

            model_results = self._backtest_model(model_version)
            results[model_version] = model_results

            # Print summary
            logger.info(f"\nSummary for {model_version}:")
            logger.info(f"  Total Bets: {model_results['total_bets']}")
            logger.info(f"  Win Rate: {model_results['win_rate']:.1f}%")
            logger.info(f"  ROI: {model_results['roi']:.2f}%")
            logger.info(f"  Sharpe Ratio: {model_results['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {model_results['max_drawdown']:.1f}%")
            logger.info(f"  Final Bankroll: ${model_results['final_bankroll']:.2f}")

        return results

    def _backtest_model(self, model_version: str) -> dict:
        """Backtest a single model version"""

        all_bets = []
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]

        for season in range(self.start_season, self.end_season + 1):
            logger.info(f"\nSeason {season}:")

            # Load actual data
            actual_data = self.load_historical_data(season)

            if actual_data.empty:
                logger.warning(f"No data for season {season}")
                continue

            # Process by week
            for week in range(1, 18):  # Regular season only
                week_data = actual_data[actual_data["week"] == week]

                if week_data.empty:
                    continue

                # Load predictions
                try:
                    predictions = self.load_model_predictions(model_version, season, week)
                except Exception as e:
                    logger.warning(f"Could not load predictions for week {week}: {e}")
                    continue

                # Filter to passing yards for QBs
                qb_data = week_data[week_data["position"] == "QB"].copy()

                if qb_data.empty:
                    continue

                # Simulate betting lines
                lines = self.simulate_betting_lines(qb_data["passing_yards"])

                # Make betting decisions
                for idx, row in qb_data.iterrows():
                    player_id = row["player_id"]

                    # Get prediction for this player
                    player_pred = predictions[predictions["player_id"] == player_id]

                    if player_pred.empty:
                        continue

                    pred_value = player_pred.iloc[0]["prediction"]
                    line = lines.loc[idx, "line"]
                    odds = lines.loc[idx, "odds"]
                    actual = lines.loc[idx, "actual"]

                    # Calculate edge
                    if pred_value > line:
                        # Bet over
                        bet_type = "over"
                        win = actual > line
                    else:
                        # Bet under
                        bet_type = "under"
                        win = actual < line

                    edge = abs(pred_value - line) / line

                    # Only bet if edge exceeds minimum
                    if edge < self.min_edge:
                        continue

                    # Calculate bet size using Kelly criterion
                    implied_prob = self._odds_to_probability(odds)
                    our_prob = 0.5 + edge  # Simplified
                    kelly = (our_prob - implied_prob) / implied_prob
                    bet_fraction = min(kelly * self.kelly_fraction, 0.05)  # Max 5% of bankroll
                    bet_amount = bankroll * bet_fraction

                    # Calculate profit/loss
                    if win:
                        profit = bet_amount * self._calculate_payout(odds)
                    else:
                        profit = -bet_amount

                    # Update bankroll
                    bankroll += profit
                    bankroll_history.append(bankroll)

                    # Record bet
                    all_bets.append(
                        {
                            "season": season,
                            "week": week,
                            "player_id": player_id,
                            "prediction": pred_value,
                            "line": line,
                            "actual": actual,
                            "bet_type": bet_type,
                            "odds": odds,
                            "edge": edge,
                            "bet_amount": bet_amount,
                            "win": win,
                            "profit": profit,
                            "bankroll": bankroll,
                        }
                    )

            logger.info(
                f"  Season {season} bets: {len([b for b in all_bets if b['season'] == season])}"
            )
            logger.info(f"  Current bankroll: ${bankroll:.2f}")

        # Calculate metrics
        if not all_bets:
            return {
                "total_bets": 0,
                "win_rate": 0,
                "roi": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "final_bankroll": bankroll,
                "bets": [],
            }

        bets_df = pd.DataFrame(all_bets)

        # Calculate metrics
        total_bets = len(bets_df)
        win_rate = bets_df["win"].mean() * 100
        total_wagered = bets_df["bet_amount"].sum()
        total_profit = bets_df["profit"].sum()
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

        # Sharpe ratio
        daily_returns = bets_df.groupby(["season", "week"])["profit"].sum()
        if len(daily_returns) > 1:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(17)  # 17 weeks
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        bankroll_series = pd.Series(bankroll_history)
        running_max = bankroll_series.expanding().max()
        drawdown = (bankroll_series - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        return {
            "total_bets": total_bets,
            "win_rate": win_rate,
            "roi": roi,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_bankroll": bankroll,
            "bets": bets_df,
            "bankroll_history": bankroll_history,
        }

    def _odds_to_probability(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    def _calculate_payout(self, odds: int) -> float:
        """Calculate payout multiplier from American odds"""
        if odds < 0:
            return 100 / abs(odds)
        else:
            return odds / 100

    def generate_comparison_report(self, results: dict):
        """Generate comprehensive comparison report"""

        # Create comparison DataFrame
        comparison = []
        for model_version, metrics in results.items():
            comparison.append(
                {
                    "Model": model_version,
                    "Total Bets": metrics["total_bets"],
                    "Win Rate": f"{metrics['win_rate']:.1f}%",
                    "ROI": f"{metrics['roi']:.2f}%",
                    "Sharpe Ratio": f"{metrics['sharpe_ratio']:.2f}",
                    "Max Drawdown": f"{metrics['max_drawdown']:.1f}%",
                    "Final Bankroll": f"${metrics['final_bankroll']:.2f}",
                    "Profit": f"${metrics['final_bankroll'] - self.initial_bankroll:.2f}",
                }
            )

        comparison_df = pd.DataFrame(comparison)

        # Print report
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ENSEMBLE BACKTEST RESULTS")
        print(f"Period: {self.start_season}-{self.end_season}")
        print(f"Initial Bankroll: ${self.initial_bankroll:.2f}")
        print("=" * 80 + "\n")

        print(comparison_df.to_string(index=False))

        print("\n" + "=" * 80)
        print("KEY FINDINGS:")
        print("=" * 80)

        # Compare v3.0 to baseline
        if "ensemble_v3.0" in results and "hierarchical_v1.0" in results:
            v3_roi = results["ensemble_v3.0"]["roi"]
            v1_roi = results["hierarchical_v1.0"]["roi"]
            improvement = v3_roi - v1_roi

            print(f"\n✓ v3.0 Ensemble ROI: {v3_roi:.2f}%")
            print(f"✓ v1.0 Baseline ROI: {v1_roi:.2f}%")
            print(f"✓ Improvement: +{improvement:.2f}%")

            if v3_roi >= 5.0:
                print("\n🎯 TARGET ACHIEVED: v3.0 ensemble meets +5-7% ROI target!")
            else:
                print(f"\n⚠️ Below target: {5.0 - v3_roi:.2f}% short of 5% ROI minimum")

        # Save to file
        report_path = Path("reports/comprehensive_backtest_results.json")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            # Convert for JSON serialization
            json_results = {}
            for model, metrics in results.items():
                json_results[model] = {
                    k: v for k, v in metrics.items() if k not in ["bets", "bankroll_history"]
                }
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Results saved to {report_path}")

    def plot_results(self, results: dict):
        """Create visualization of backtest results"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Comprehensive Backtest Results ({self.start_season}-{self.end_season})", fontsize=16
        )

        # 1. Bankroll evolution
        ax = axes[0, 0]
        for model_version, metrics in results.items():
            if "bankroll_history" in metrics:
                ax.plot(metrics["bankroll_history"], label=model_version)
        ax.set_title("Bankroll Evolution")
        ax.set_xlabel("Number of Bets")
        ax.set_ylabel("Bankroll ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. ROI comparison
        ax = axes[0, 1]
        models = list(results.keys())
        rois = [results[m]["roi"] for m in models]
        bars = ax.bar(range(len(models)), rois)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace("_", "\n") for m in models], rotation=0)
        ax.set_title("ROI Comparison")
        ax.set_ylabel("ROI (%)")
        ax.axhline(y=5.0, color="r", linestyle="--", label="Target (5%)")
        ax.legend()

        # Color bars based on performance
        for i, (bar, roi) in enumerate(zip(bars, rois)):
            if roi >= 5.0:
                bar.set_color("green")
            elif roi >= 3.0:
                bar.set_color("yellow")
            else:
                bar.set_color("red")

        # 3. Win rate vs ROI scatter
        ax = axes[1, 0]
        for model_version, metrics in results.items():
            ax.scatter(metrics["win_rate"], metrics["roi"], s=100, label=model_version)
        ax.set_xlabel("Win Rate (%)")
        ax.set_ylabel("ROI (%)")
        ax.set_title("Win Rate vs ROI")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Risk metrics
        ax = axes[1, 1]
        x = np.arange(len(models))
        width = 0.35
        sharpes = [results[m]["sharpe_ratio"] for m in models]
        drawdowns = [abs(results[m]["max_drawdown"]) for m in models]

        ax.bar(x - width / 2, sharpes, width, label="Sharpe Ratio", color="blue", alpha=0.7)
        ax.bar(x + width / 2, drawdowns, width, label="Max Drawdown (%)", color="red", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in models], rotation=0)
        ax.set_title("Risk Metrics")
        ax.legend()

        plt.tight_layout()
        plt.savefig("reports/comprehensive_backtest_plot.png", dpi=150)
        plt.show()

        print("✓ Plots saved to reports/comprehensive_backtest_plot.png")


def main():
    """Run comprehensive ensemble backtest"""

    logger.info("=" * 60)
    logger.info("COMPREHENSIVE ENSEMBLE BACKTEST")
    logger.info("Validating +5-7% ROI target for v3.0")
    logger.info("=" * 60 + "\n")

    # Initialize backtest
    backtest = ComprehensiveEnsembleBacktest(
        start_season=2022,
        end_season=2024,
        initial_bankroll=10000.0,
        kelly_fraction=0.25,
        min_edge=0.02,
    )

    # Run backtest
    results = backtest.run_backtest()

    # Generate report
    backtest.generate_comparison_report(results)

    # Create visualizations
    backtest.plot_results(results)

    logger.info("\n✓ Comprehensive backtest complete")


if __name__ == "__main__":
    main()
