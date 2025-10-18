"""
Test script for Phase 2.2 MoE BNN Production Pipeline

Runs a demo prediction to validate the pipeline works end-to-end.

Author: Richard Oldham
Date: October 2025
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import the production pipeline
from production.bnn_moe_production_pipeline import BNNMoEProductionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pipeline_with_historical_data():
    """
    Test the pipeline using historical data (Week 7, 2024)
    This validates the pipeline works before deploying to live betting
    """
    logger.info("=" * 80)
    logger.info("PHASE 2.2 MoE BNN PRODUCTION PIPELINE TEST")
    logger.info("=" * 80)

    # Initialize pipeline with conservative settings
    model_path = "/Users/dro/rice/nfl-analytics/models/bayesian/bnn_mixture_experts_v2.pkl"

    pipeline = BNNMoEProductionPipeline(
        model_path=model_path,
        bankroll=10000.0,  # Simulated $10k bankroll
        kelly_fraction=0.15,  # Conservative 15% Kelly
        min_edge=0.03,  # Minimum 3% edge required
        min_confidence=0.80,  # High confidence threshold
    )

    logger.info("\n✓ Pipeline initialized")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Bankroll: ${pipeline.bankroll:,.0f}")
    logger.info(f"  Kelly Fraction: {pipeline.kelly_fraction:.1%}")
    logger.info(f"  Min Edge: {pipeline.min_edge:.1%}")
    logger.info(f"  Min Confidence: {pipeline.min_confidence:.1%}")

    # Test with Week 7, 2024 (historical data for validation)
    test_season = 2024
    test_week = 7

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running pipeline for {test_season} Week {test_week} (historical test)")
    logger.info(f"{'=' * 80}")

    try:
        # Run the pipeline
        recommendations = pipeline.run_pipeline(season=test_season, week=test_week)

        if recommendations is None or len(recommendations) == 0:
            logger.warning("⚠️  No bet recommendations generated")
            logger.info("\nPossible reasons:")
            logger.info("  - No prop lines available for this week")
            logger.info("  - No bets met minimum edge/confidence criteria")
            logger.info("  - Data quality issues")
            return

        # Display results
        logger.info(f"\n{'=' * 80}")
        logger.info("BET RECOMMENDATIONS SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"\nTotal Recommendations: {len(recommendations)}")

        total_bet = recommendations["bet_amount"].sum()
        logger.info(
            f"Total Capital Allocated: ${total_bet:.2f} ({100*total_bet/pipeline.bankroll:.1f}% of bankroll)"
        )

        # Show top 5 recommendations
        logger.info(f"\n{'=' * 80}")
        logger.info("TOP 5 RECOMMENDATIONS (by expected value)")
        logger.info(f"{'=' * 80}")

        top_5 = recommendations.nlargest(5, "expected_value")

        for idx, (_, bet) in enumerate(top_5.iterrows(), 1):
            logger.info(f"\n#{idx}: {bet['player_name']} ({bet['team']})")
            logger.info(
                f"  Market: {bet['bet_type'].upper()} {bet['rushing_line']:.1f} yards @ {bet['odds']:+d}"
            )
            logger.info(f"  Prediction: {bet['pred_mean']:.1f} ± {bet['pred_std']:.1f} yards")
            logger.info(f"  90% CI: [{bet['pred_lower_90']:.1f}, {bet['pred_upper_90']:.1f}]")
            logger.info(f"  Edge: {bet['edge']:.2%} | Confidence: {bet['confidence']:.2f}σ")
            logger.info(f"  Win Prob: {bet['prob_win']:.1%} | Implied: {bet['implied_prob']:.1%}")
            logger.info(f"  💰 Bet: ${bet['bet_amount']:.2f} | EV: ${bet['expected_value']:.2f}")

        # Risk analysis
        logger.info(f"\n{'=' * 80}")
        logger.info("RISK ANALYSIS")
        logger.info(f"{'=' * 80}")

        logger.info("\nEdge Distribution:")
        logger.info(f"  Mean Edge: {recommendations['edge'].mean():.2%}")
        logger.info(f"  Median Edge: {recommendations['edge'].median():.2%}")
        logger.info(f"  Max Edge: {recommendations['edge'].max():.2%}")

        logger.info("\nConfidence Distribution:")
        logger.info(f"  Mean Confidence: {recommendations['confidence'].mean():.2f}σ")
        logger.info(f"  Median Confidence: {recommendations['confidence'].median():.2f}σ")
        logger.info(f"  Min Confidence: {recommendations['confidence'].min():.2f}σ")

        logger.info("\nBet Type Breakdown:")
        bet_counts = recommendations["bet_type"].value_counts()
        for bet_type, count in bet_counts.items():
            total_amount = recommendations[recommendations["bet_type"] == bet_type][
                "bet_amount"
            ].sum()
            logger.info(f"  {bet_type.upper()}: {count} bets, ${total_amount:.2f}")

        # Expected P&L
        total_ev = recommendations["expected_value"].sum()
        logger.info(f"\n{'=' * 80}")
        logger.info("EXPECTED PROFIT & LOSS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total Expected Value: ${total_ev:.2f}")
        logger.info(f"Expected ROI: {100*total_ev/total_bet:.2f}%")

        # Output file location
        output_dir = Path("/Users/dro/rice/nfl-analytics/data/betting_recommendations")
        output_file = output_dir / f"recommendations_{test_season}_week{test_week}.csv"

        logger.info("\n✓ Recommendations saved to:")
        logger.info(f"  {output_file}")

        # Validation checks
        logger.info(f"\n{'=' * 80}")
        logger.info("VALIDATION CHECKS")
        logger.info(f"{'=' * 80}")

        checks_passed = True

        # Check 1: All bets meet minimum edge
        if (recommendations["edge"] < pipeline.min_edge).any():
            logger.warning("⚠️  Some bets below minimum edge threshold")
            checks_passed = False
        else:
            logger.info(f"✓ All bets meet minimum edge requirement ({pipeline.min_edge:.1%})")

        # Check 2: All bets meet minimum confidence
        if (recommendations["confidence"] < pipeline.min_confidence).any():
            logger.warning("⚠️  Some bets below minimum confidence threshold")
            checks_passed = False
        else:
            logger.info(
                f"✓ All bets meet minimum confidence requirement ({pipeline.min_confidence:.1f}σ)"
            )

        # Check 3: Kelly sizing doesn't over-allocate bankroll
        max_allocation = 0.25  # Never bet more than 25% of bankroll
        if total_bet > max_allocation * pipeline.bankroll:
            logger.warning(f"⚠️  Total allocation exceeds {100*max_allocation:.0f}% of bankroll")
            checks_passed = False
        else:
            logger.info(
                f"✓ Total allocation within safe limits (<{100*max_allocation:.0f}% bankroll)"
            )

        # Check 4: Predictions are reasonable (within typical RB ranges)
        if recommendations["pred_mean"].max() > 200:
            logger.warning(f"⚠️  Some predictions seem unrealistically high (>{200} yards)")
            checks_passed = False
        else:
            logger.info("✓ All predictions within reasonable RB rushing ranges")

        if checks_passed:
            logger.info(f"\n{'=' * 80}")
            logger.info("✅ ALL VALIDATION CHECKS PASSED")
            logger.info(f"{'=' * 80}")
            logger.info("\nPipeline is ready for production deployment!")
        else:
            logger.info(f"\n{'=' * 80}")
            logger.info("⚠️  SOME VALIDATION CHECKS FAILED")
            logger.info(f"{'=' * 80}")
            logger.info("\nReview warnings above before deploying to production")

        return recommendations

    except Exception as e:
        logger.error("\n❌ Pipeline test failed with error:")
        logger.error(f"  {type(e).__name__}: {e}")
        logger.error("\nFull traceback:")
        import traceback

        traceback.print_exc()
        return None


def compare_to_actual_results(recommendations, test_season=2024, test_week=7):
    """
    If we have actual results for this week, compare predictions to outcomes

    This validates the model's prediction accuracy on real betting scenarios
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARING TO ACTUAL RESULTS")
    logger.info(f"{'=' * 80}")

    try:
        import psycopg2

        db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        conn = psycopg2.connect(**db_config)

        query = """
        SELECT
            player_id,
            player_display_name,
            stat_yards as actual_yards
        FROM mart.player_game_stats
        WHERE season = %s
          AND week = %s
          AND stat_category = 'rushing'
          AND position_group IN ('RB', 'FB', 'HB')
        """

        actuals = pd.read_sql(query, conn, params=[test_season, test_week])
        conn.close()

        # Merge with recommendations
        comparison = recommendations.merge(actuals, on="player_id", how="left")

        comparison = comparison[comparison["actual_yards"].notna()]

        if len(comparison) == 0:
            logger.warning("⚠️  No actual results available for comparison")
            return

        logger.info(f"\nFound {len(comparison)} bets with actual outcomes")

        # Calculate results
        comparison["won"] = (
            (comparison["bet_type"] == "over")
            & (comparison["actual_yards"] > comparison["rushing_line"])
        ) | (
            (comparison["bet_type"] == "under")
            & (comparison["actual_yards"] < comparison["rushing_line"])
        )

        comparison["profit"] = comparison.apply(
            lambda row: (
                row["bet_amount"] * (row["odds"] / 100) if row["won"] else -row["bet_amount"]
            ),
            axis=1,
        )

        # Summary statistics
        wins = comparison["won"].sum()
        total_bets = len(comparison)
        win_rate = wins / total_bets

        total_wagered = comparison["bet_amount"].sum()
        total_profit = comparison["profit"].sum()
        roi = total_profit / total_wagered

        logger.info(f"\n{'=' * 80}")
        logger.info("BETTING RESULTS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Record: {wins}-{total_bets - wins} ({100*win_rate:.1f}% win rate)")
        logger.info(f"Total Wagered: ${total_wagered:.2f}")
        logger.info(f"Total Profit: ${total_profit:.2f}")
        logger.info(f"ROI: {100*roi:.2f}%")

        # Compare to expected
        expected_ev = comparison["expected_value"].sum()
        logger.info(f"\nExpected EV: ${expected_ev:.2f}")
        logger.info(f"Actual Profit: ${total_profit:.2f}")
        logger.info(f"Variance: ${total_profit - expected_ev:.2f}")

        # Prediction accuracy
        mae = np.abs(comparison["pred_mean"] - comparison["actual_yards"]).mean()
        logger.info(f"\nPrediction MAE: {mae:.2f} yards")

        # Show individual results
        logger.info(f"\n{'=' * 80}")
        logger.info("INDIVIDUAL BET RESULTS")
        logger.info(f"{'=' * 80}")

        for _, bet in comparison.iterrows():
            result = "✓ WIN" if bet["won"] else "✗ LOSS"
            logger.info(f"\n{result}: {bet['player_name']}")
            logger.info(f"  {bet['bet_type'].upper()} {bet['rushing_line']:.1f} yards")
            logger.info(f"  Predicted: {bet['pred_mean']:.1f} | Actual: {bet['actual_yards']:.1f}")
            logger.info(f"  Bet: ${bet['bet_amount']:.2f} | Profit: ${bet['profit']:.2f}")

        return comparison

    except Exception as e:
        logger.warning(f"⚠️  Could not compare to actual results: {e}")
        return None


if __name__ == "__main__":
    logger.info("Starting Phase 2.2 MoE BNN Production Pipeline Test\n")

    # Test the pipeline
    recommendations = test_pipeline_with_historical_data()

    if recommendations is not None:
        # Compare to actual results if available
        compare_to_actual_results(recommendations)

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
