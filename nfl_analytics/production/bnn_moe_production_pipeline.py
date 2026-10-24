#!/usr/bin/env python3
"""
Phase 2.2 Mixture-of-Experts BNN Production Pipeline
Rushing Yards Prediction & Betting System

Based on Phase 2.2 MoE results:
- 92.2% calibration (90% CI coverage)
- MAE: 18.5 yards (excellent point estimates)
- Balanced expert usage (32%/32%/36%)
- Heterogeneous uncertainty modeling

This pipeline:
1. Loads the trained Phase 2.2 MoE BNN model
2. Generates player rushing features from recent games
3. Makes predictions with uncertainty quantification
4. Compares to prop lines (over/under rushing yards)
5. Selects +EV bets using Kelly criterion
6. Outputs recommendations with confidence intervals

Usage:
    # Full pipeline for upcoming week
    python py/production/bnn_moe_production_pipeline.py \
        --week 11 \
        --season 2025 \
        --bankroll 10000

    # Backtest on historical data
    python py/production/bnn_moe_production_pipeline.py \
        --backtest \
        --season 2024 \
        --weeks 7-17

    # Paper trading mode (no real money)
    python py/production/bnn_moe_production_pipeline.py \
        --paper-trade \
        --week 11
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from models.bnn_mixture_experts_v2 import MixtureExpertsBNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Model path
DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "bayesian" / "bnn_mixture_experts_v2.pkl"
)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "bnn_moe_recommendations"


# ============================================================================
# Feature Engineering
# ============================================================================


def load_rushing_features(
    conn, season: int, week: int = None, player_ids: list[str] = None
) -> pd.DataFrame:
    """
    Load rushing features for prediction.

    Args:
        conn: Database connection
        season: NFL season
        week: Optional week number (None = all weeks)
        player_ids: Optional list of player IDs to filter

    Returns:
        DataFrame with rushing features
    """
    # Query matches training data structure
    query = """
    SELECT
        pgs.player_id,
        pgs.player_display_name as player_name,
        pgs.season,
        pgs.week,
        pgs.current_team as team,
        pgs.stat_yards,
        pgs.stat_attempts as carries,
        -- Recent form (3-game average)
        AVG(pgs.stat_yards) OVER (
            PARTITION BY pgs.player_id
            ORDER BY pgs.season, pgs.week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as avg_rushing_l3,
        -- Season average
        AVG(pgs.stat_yards) OVER (
            PARTITION BY pgs.player_id, pgs.season
            ORDER BY pgs.week
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) as season_avg
    FROM mart.player_game_stats pgs
    WHERE pgs.season = %s
      AND pgs.stat_category = 'rushing'
      AND pgs.position_group IN ('RB', 'FB', 'HB')
      AND pgs.stat_attempts >= 5
      AND pgs.stat_yards IS NOT NULL
    """

    params = [season]

    if week is not None:
        query += " AND pgs.week = %s"
        params.append(week)

    if player_ids:
        placeholders = ",".join(["%s"] * len(player_ids))
        query += f" AND pgs.player_id IN ({placeholders})"
        params.extend(player_ids)

    query += " ORDER BY pgs.season, pgs.week, pgs.stat_yards DESC"

    df = pd.read_sql(query, conn, params=params)

    # Handle missing values (same as training)
    df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
        df.groupby("season")["stat_yards"].transform("median")
    )
    df["season_avg"] = df["season_avg"].fillna(
        df.groupby("season")["stat_yards"].transform("median")
    )

    logger.info(f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players")

    return df


# ============================================================================
# Production Pipeline
# ============================================================================


class BNNMoEProductionPipeline:
    """
    Production pipeline for Phase 2.2 Mixture-of-Experts BNN.

    Integrates Bayesian uncertainty quantification into betting workflow.
    """

    def __init__(
        self,
        model_path: str = None,
        db_config: dict = None,
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.15,
        max_bet_fraction: float = 0.03,
        min_edge: float = 0.03,
        min_confidence: float = 0.80,
    ):
        """
        Initialize production pipeline.

        Args:
            model_path: Path to trained MoE BNN model
            db_config: Database configuration
            bankroll: Current bankroll ($)
            kelly_fraction: Fraction of Kelly to bet (0.15 = conservative)
            max_bet_fraction: Maximum bet as fraction of bankroll (0.03 = 3%)
            min_edge: Minimum edge required to bet (0.03 = 3%)
            min_confidence: Minimum confidence (90% CI doesn't overlap line)
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.db_config = db_config or DB_CONFIG
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
        self.min_confidence = min_confidence

        # Load model
        logger.info(f"Loading Phase 2.2 MoE BNN model from {self.model_path}")
        self.model = MixtureExpertsBNN.load(self.model_path)

        # Feature columns (same as training)
        self.feature_cols = ["carries", "avg_rushing_l3", "season_avg", "week"]

        # Use model's scaler (critical for correct predictions!)
        self.scaler = self.model.scaler

        # Tracking
        self.recommendations = []
        self.performance = {
            "total_predictions": 0,
            "total_bets": 0,
            "total_won": 0,
            "total_return": 0.0,
        }

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def predict_with_uncertainty(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with Bayesian uncertainty.

        Args:
            players_df: DataFrame with player features

        Returns:
            DataFrame with predictions and uncertainty intervals
        """
        logger.info(f"Generating predictions for {len(players_df)} players...")

        # Prepare features
        X = players_df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)  # Use transform, NOT fit_transform!

        # Get predictions with uncertainty
        predictions = self.model.predict(X_scaled)

        # Add to dataframe
        results = players_df.copy()
        results["pred_mean"] = predictions["mean"]
        results["pred_std"] = predictions["std"]
        results["pred_lower_90"] = predictions["lower_90"]
        results["pred_upper_90"] = predictions["upper_90"]

        # Confidence metrics
        results["pred_interval_width"] = results["pred_upper_90"] - results["pred_lower_90"]
        results["pred_cv"] = results["pred_std"] / results["pred_mean"]  # Coefficient of variation

        self.performance["total_predictions"] = len(results)

        logger.info(f"✓ Generated {len(results)} predictions")
        logger.info(f"  Mean prediction: {results['pred_mean'].mean():.1f} yards")
        logger.info(f"  Mean 90% CI width: {results['pred_interval_width'].mean():.1f} yards")

        return results

    def get_prop_lines(self, conn, player_ids: list[str] = None) -> pd.DataFrame:
        """
        Get rushing yards prop lines from database.

        Args:
            conn: Database connection
            player_ids: Optional list of player IDs to filter

        Returns:
            DataFrame with prop lines
        """
        query = """
        SELECT
            player_id,
            player_name,
            player_position,
            player_team,
            line_value as rushing_line,
            over_odds,
            under_odds,
            bookmaker_key as bookmaker,
            over_implied_prob,
            under_implied_prob,
            book_hold,
            commence_time as game_time
        FROM best_prop_lines
        WHERE prop_type = 'rushing_yards'
          AND commence_time >= NOW()
        """

        params = []
        if player_ids:
            placeholders = ",".join(["%s"] * len(player_ids))
            query += f" AND player_id IN ({placeholders})"
            params = player_ids

        query += " ORDER BY commence_time, player_name"

        try:
            prop_lines = pd.read_sql(query, conn, params=params if params else None)
            logger.info(f"✓ Retrieved {len(prop_lines)} rushing yards prop lines")
            return prop_lines
        except Exception as e:
            logger.warning(f"Could not load prop lines: {e}")
            logger.warning("Continuing without prop lines (predictions only)")
            return pd.DataFrame()

    def kelly_criterion(
        self,
        win_prob: float,
        odds_american: int,
    ) -> float:
        """
        Calculate Kelly criterion bet size.

        Args:
            win_prob: Probability of winning (0-1)
            odds_american: American odds (e.g., -110, +150)

        Returns:
            Bet fraction (0-1)
        """
        # Convert American odds to decimal
        if odds_american > 0:
            b = odds_american / 100.0
        else:
            b = 100.0 / abs(odds_american)

        # Kelly formula
        p = win_prob
        q = 1 - p
        kelly_full = (b * p - q) / b

        # Apply fraction and clip
        kelly_bet = kelly_full * self.kelly_fraction
        return np.clip(kelly_bet, 0.0, self.max_bet_fraction)

    def select_bets(
        self, predictions_df: pd.DataFrame, prop_lines_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Select profitable bets using Bayesian uncertainty.

        Args:
            predictions_df: Predictions with uncertainty
            prop_lines_df: Prop lines from sportsbooks

        Returns:
            DataFrame of recommended bets
        """
        if prop_lines_df.empty:
            logger.warning("No prop lines available - cannot select bets")
            return pd.DataFrame()

        logger.info("Analyzing betting opportunities...")

        # Merge predictions with prop lines
        merged = predictions_df.merge(prop_lines_df, on="player_id", suffixes=("", "_prop"))

        bets = []

        for _, row in merged.iterrows():
            rushing_line = row["rushing_line"]
            pred_mean = row["pred_mean"]
            pred_lower = row["pred_lower_90"]
            pred_upper = row["pred_upper_90"]

            # Calculate probabilities using empirical CDF approximation
            # Assume normal distribution for simplicity
            from scipy.stats import norm

            pred_std = row["pred_std"]
            # P(yards > line)
            prob_over = 1 - norm.cdf(rushing_line, loc=pred_mean, scale=pred_std)
            # P(yards < line)
            prob_under = norm.cdf(rushing_line, loc=pred_mean, scale=pred_std)

            # Market probabilities
            market_prob_over = row["over_implied_prob"]
            market_prob_under = row["under_implied_prob"]

            # Edges
            edge_over = prob_over - market_prob_over
            edge_under = prob_under - market_prob_under

            # Confidence check: 90% CI should not overlap line significantly
            # For over bet: lower bound should be close to or above line
            # For under bet: upper bound should be close to or below line
            confidence_over = (pred_lower - rushing_line) / pred_std if pred_std > 0 else 0
            confidence_under = (rushing_line - pred_upper) / pred_std if pred_std > 0 else 0

            # Select best bet direction
            if edge_over > self.min_edge and confidence_over > -1.0:  # At least within 1 std
                # Bet OVER
                bet_side = "over"
                edge = edge_over
                prob_win = prob_over
                odds = row["over_odds"]
                confidence = min(1.0, max(0.0, (confidence_over + 1) / 2))  # Normalize to 0-1

            elif edge_under > self.min_edge and confidence_under > -1.0:
                # Bet UNDER
                bet_side = "under"
                edge = edge_under
                prob_win = prob_under
                odds = row["under_odds"]
                confidence = min(1.0, max(0.0, (confidence_under + 1) / 2))

            else:
                # No bet
                continue

            # Skip if confidence too low
            if confidence < self.min_confidence:
                continue

            # Calculate Kelly bet size
            bet_fraction = self.kelly_criterion(prob_win, odds)
            bet_amount = bet_fraction * self.bankroll

            # Skip if bet too small
            if bet_amount < 10:  # $10 minimum
                continue

            # Create recommendation
            bet = {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "team": row["team"],
                "rushing_line": rushing_line,
                "bet_side": bet_side,
                "prediction_mean": pred_mean,
                "prediction_std": pred_std,
                "prediction_lower_90": pred_lower,
                "prediction_upper_90": pred_upper,
                "prob_win": prob_win,
                "market_prob": market_prob_over if bet_side == "over" else market_prob_under,
                "edge": edge,
                "confidence": confidence,
                "odds": odds,
                "bookmaker": row["bookmaker"],
                "bet_fraction": bet_fraction,
                "bet_amount": bet_amount,
                "game_time": row["game_time"],
            }

            bets.append(bet)

        bets_df = pd.DataFrame(bets)

        if not bets_df.empty:
            # Sort by edge (descending)
            bets_df = bets_df.sort_values("edge", ascending=False)

            logger.info(f"✓ Selected {len(bets_df)} profitable bets")
            logger.info(f"  Total to wager: ${bets_df['bet_amount'].sum():,.0f}")
            logger.info(f"  Average edge: {bets_df['edge'].mean()*100:.2f}%")
            logger.info(f"  Average confidence: {bets_df['confidence'].mean()*100:.1f}%")

            self.performance["total_bets"] = len(bets_df)
        else:
            logger.info("No bets meet selection criteria")

        return bets_df

    def save_recommendations(
        self, predictions_df: pd.DataFrame, bets_df: pd.DataFrame, output_prefix: str = None
    ) -> dict[str, Path]:
        """
        Save predictions and betting recommendations.

        Args:
            predictions_df: All predictions
            bets_df: Selected bets
            output_prefix: Optional prefix for output files

        Returns:
            Dict of output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = output_prefix or f"week_{timestamp}"
        output_files = {}

        # Save predictions
        preds_file = OUTPUT_DIR / f"{prefix}_predictions.csv"
        predictions_df.to_csv(preds_file, index=False)
        output_files["predictions"] = preds_file
        logger.info(f"✓ Saved {len(predictions_df)} predictions to {preds_file}")

        # Save betting recommendations
        if not bets_df.empty:
            bets_file = OUTPUT_DIR / f"{prefix}_bets.csv"
            bets_df.to_csv(bets_file, index=False)
            output_files["bets"] = bets_file
            logger.info(f"✓ Saved {len(bets_df)} bet recommendations to {bets_file}")

            # Latest
            latest_file = OUTPUT_DIR / "latest_bets.csv"
            bets_df.to_csv(latest_file, index=False)

            # Human-readable summary
            summary_file = OUTPUT_DIR / f"{prefix}_summary.txt"
            with open(summary_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("PHASE 2.2 MoE BNN BETTING RECOMMENDATIONS\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Bankroll: ${self.bankroll:,.2f}\n")
                f.write(f"Total bets: {len(bets_df)}\n")
                f.write(f"Total to wager: ${bets_df['bet_amount'].sum():,.2f}\n")
                f.write(f"Capital at risk: {bets_df['bet_amount'].sum()/self.bankroll*100:.1f}%\n")
                f.write(f"Average edge: {bets_df['edge'].mean()*100:.2f}%\n")
                f.write(f"Average confidence: {bets_df['confidence'].mean()*100:.1f}%\n\n")

                f.write("TOP BETS BY EDGE:\n")
                f.write("-" * 80 + "\n\n")

                for idx, row in bets_df.head(10).iterrows():
                    f.write(f"{row['player_name']} ({row['team']}) - {row['bet_side'].upper()}\n")
                    f.write(f"  Line: {row['rushing_line']} yards\n")
                    f.write(f"  Prediction: {row['prediction_mean']:.1f} yards ")
                    f.write(
                        f"(90% CI: {row['prediction_lower_90']:.1f} - {row['prediction_upper_90']:.1f})\n"
                    )
                    f.write(
                        f"  Edge: {row['edge']*100:.2f}% | Confidence: {row['confidence']*100:.1f}%\n"
                    )
                    f.write(f"  Odds: {row['odds']:+d} ({row['bookmaker']})\n")
                    f.write(
                        f"  Bet: ${row['bet_amount']:.2f} ({row['bet_fraction']*100:.2f}% of bankroll)\n"
                    )
                    f.write(f"  Game: {row['game_time']}\n\n")

            output_files["summary"] = summary_file
            logger.info(f"✓ Saved summary to {summary_file}")

        # Save statistics
        stats_file = OUTPUT_DIR / f"{prefix}_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.performance, f, indent=2)
        output_files["stats"] = stats_file

        return output_files

    def run_pipeline(self, season: int, week: int = None, player_ids: list[str] = None) -> dict:
        """
        Run complete production pipeline.

        Args:
            season: NFL season
            week: Week number (optional)
            player_ids: Optional list of player IDs

        Returns:
            Dict with results
        """
        logger.info("=" * 80)
        logger.info("PHASE 2.2 MoE BNN PRODUCTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Season: {season}, Week: {week or 'All'}")
        logger.info(f"Bankroll: ${self.bankroll:,.2f}")
        logger.info(f"Min edge: {self.min_edge*100:.1f}%")
        logger.info(f"Min confidence: {self.min_confidence*100:.1f}%")
        logger.info("=" * 80 + "\n")

        # Connect to database
        conn = psycopg2.connect(**self.db_config)

        # Load features
        players_df = load_rushing_features(conn, season=season, week=week, player_ids=player_ids)

        if players_df.empty:
            logger.error("No player data available")
            conn.close()
            return {"success": False, "error": "No data"}

        # Generate predictions
        predictions_df = self.predict_with_uncertainty(players_df)

        # Get prop lines
        prop_lines_df = self.get_prop_lines(
            conn, player_ids=players_df["player_id"].unique().tolist()
        )

        conn.close()

        # Select bets
        bets_df = self.select_bets(predictions_df, prop_lines_df)

        # Save outputs
        output_files = self.save_recommendations(
            predictions_df, bets_df, output_prefix=f"s{season}_w{week}" if week else f"s{season}"
        )

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

        return {
            "success": True,
            "predictions": len(predictions_df),
            "bets": len(bets_df),
            "output_files": output_files,
        }


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Phase 2.2 MoE BNN Production Pipeline")

    # Data
    parser.add_argument("--season", type=int, default=2025, help="NFL season")
    parser.add_argument("--week", type=int, help="Week number (optional)")

    # Model
    parser.add_argument(
        "--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to MoE BNN model"
    )

    # Bankroll management
    parser.add_argument("--bankroll", type=float, default=10000.0, help="Current bankroll")
    parser.add_argument(
        "--kelly-fraction", type=float, default=0.15, help="Kelly fraction (0.15 = conservative)"
    )
    parser.add_argument(
        "--max-bet-fraction", type=float, default=0.03, help="Max bet fraction (0.03 = 3%)"
    )
    parser.add_argument("--min-edge", type=float, default=0.03, help="Minimum edge (0.03 = 3%)")
    parser.add_argument(
        "--min-confidence", type=float, default=0.80, help="Minimum confidence (0.80 = 80%)"
    )

    # Mode
    parser.add_argument("--paper-trade", action="store_true", help="Paper trading mode")
    parser.add_argument("--backtest", action="store_true", help="Backtest mode")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PHASE 2.2 MIXTURE-OF-EXPERTS BNN")
    logger.info("=" * 80)
    logger.info(f"Mode: {'PAPER TRADING' if args.paper_trade else 'LIVE'}")
    logger.info(f"Bankroll: ${args.bankroll:,.2f}")
    logger.info(f"Kelly fraction: {args.kelly_fraction} (conservative)")
    logger.info(f"Season: {args.season}, Week: {args.week or 'All'}")

    # Initialize pipeline
    pipeline = BNNMoEProductionPipeline(
        model_path=args.model_path,
        db_config=DB_CONFIG,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        max_bet_fraction=args.max_bet_fraction,
        min_edge=args.min_edge,
        min_confidence=args.min_confidence,
    )

    # Run pipeline
    result = pipeline.run_pipeline(season=args.season, week=args.week)

    if result["success"]:
        logger.info("\n✅ Pipeline completed successfully")
        logger.info(f"  Predictions: {result['predictions']}")
        logger.info(f"  Bets recommended: {result['bets']}")
        return 0
    else:
        logger.error(f"\n❌ Pipeline failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
