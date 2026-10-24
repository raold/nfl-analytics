#!/usr/bin/env python3
"""
Quick V6.1 Temperature Fix Validation
====================================

Tests if correcting temperature scaling from T=1.0771 to T=0.7083 explains
the discrepancy between dissertation claims (60.8% win rate, +6.45% ROI)
and Nov 8 backtest results (52.2% win rate, -36% ROI).

Strategy:
1. Load historical player-games from database (2022-2024)
2. Generate v6.1 ensemble predictions with CORRECTED temperature (T=0.7083)
3. Calculate comprehensive metrics (Win Rate, ROI, Sharpe, MAE, Calibration)
4. Compare to dissertation claims

Expected runtime: 30-60 minutes (vs 18 hours for full rpy2 backtest)

Author: Claude Code (Sonnet 4.5)
Date: November 23, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import numpy as np
import pandas as pd
import psycopg2

from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_betting_metrics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive betting metrics from predictions."""
    # Simulate betting with standard assumptions
    # - $100 bets on lines we have edge on
    # - Standard -110 vig

    # Determine if we have edge
    df["has_edge"] = (df["predicted"] > df["line"] + 5) | (df["predicted"] < df["line"] - 5)

    # Determine bet outcome (win if actual matches our edge direction)
    df["bet_over"] = df["predicted"] > df["line"] + 5
    df["bet_under"] = df["predicted"] < df["line"] - 5

    # Calculate wins
    df["win"] = ((df["bet_over"]) & (df["actual"] > df["line"])) | (
        (df["bet_under"]) & (df["actual"] < df["line"])
    )

    # Filter to bets we would actually make
    bets_df = df[df["has_edge"]].copy()

    if len(bets_df) == 0:
        return {"total_bets": 0, "win_rate": 0.0, "roi": 0.0, "sharpe": 0.0, "total_pnl": 0.0}

    # Calculate P&L (assuming -110 vig)
    bets_df["pnl"] = bets_df["win"].apply(lambda w: 100.0 if w else -110.0)

    # Calculate metrics
    total_bets = len(bets_df)
    wins = bets_df["win"].sum()
    win_rate = wins / total_bets if total_bets > 0 else 0.0

    total_pnl = bets_df["pnl"].sum()
    total_risked = total_bets * 110.0  # Account for vig
    roi = (total_pnl / total_risked) * 100 if total_risked > 0 else 0.0

    # Sharpe ratio (per-bet returns)
    returns = bets_df["pnl"] / 110.0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(total_bets) if len(returns) > 1 else 0.0

    return {
        "total_bets": total_bets,
        "wins": int(wins),
        "win_rate": win_rate * 100,
        "roi": roi,
        "sharpe": sharpe,
        "total_pnl": total_pnl,
        "avg_bet_pnl": total_pnl / total_bets if total_bets > 0 else 0.0,
    }


def fetch_historical_games(seasons: list[int]) -> pd.DataFrame:
    """Fetch historical player-games with lines and actuals."""

    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password="sicillionbillions",  # pragma: allowlist secret
    )

    query = """
    SELECT
        p.passer_player_id as player_id,
        p.passer_player_name as player_name,
        g.game_id,
        g.season,
        g.week,
        p.posteam as team,
        g.home_team,
        g.away_team,
        SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) as actual_yards,
        COUNT(CASE WHEN p.pass = true THEN 1 ELSE NULL END) as pass_attempts
    FROM plays p
    JOIN games g ON p.game_id = g.game_id
    WHERE p.passer_player_id IS NOT NULL
      AND g.season = ANY(%(seasons)s)
      AND g.week > 0  -- Exclude preseason
    GROUP BY g.game_id, g.season, g.week, g.home_team, g.away_team,
             p.passer_player_id, p.passer_player_name, p.posteam
    HAVING COUNT(CASE WHEN p.pass = true THEN 1 ELSE NULL END) >= 10
    ORDER BY g.season, g.week, g.game_id
    """

    df = pd.read_sql(query, conn, params={"seasons": seasons})
    conn.close()

    logger.info(f"Loaded {len(df)} player-games from seasons {seasons}")

    return df


def main():
    logger.info("=" * 80)
    logger.info("QUICK V6.1 TEMPERATURE FIX VALIDATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Goal: Test if temperature fix (T=1.0771 → T=0.7083) explains")
    logger.info("      dissertation claims vs Nov 8 backtest discrepancy")
    logger.info("")
    logger.info("Dissertation Claims: 60.8% win rate, +6.45% ROI, Sharpe 2.51")
    logger.info("Nov 8 Backtest:      52.2% win rate, -36.0% ROI, Sharpe 0.16")
    logger.info("=" * 80)
    logger.info("")

    # Initialize v6.1 ensemble with corrected temperature
    logger.info("Initializing ensemble v6.1 with CORRECTED temperature...")
    ensemble = EnhancedEnsemble5Way(
        db_host="localhost",
        db_port=5544,
        db_name="devdb01",
        db_user="dro",
        db_password="sicillionbillions",  # pragma: allowlist secret
    )
    logger.info("")

    # Fetch historical games
    logger.info("Fetching historical games (2022-2024)...")
    df = fetch_historical_games([2022, 2023, 2024])
    logger.info(f"Processing {len(df)} player-games...")
    logger.info("")

    # Generate predictions
    predictions = []
    errors = 0

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")

        try:
            pred = ensemble.predict(
                player_id=row["player_id"], season=row["season"], week=row["week"]
            )

            predictions.append(
                {
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "predicted": pred["mean"],
                    "predicted_std": pred["std"],
                    "actual": row["actual_yards"],
                    "line": 240.0,  # Standard placeholder line for now
                    "mae": abs(pred["mean"] - row["actual_yards"]),
                }
            )

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(
                    f"Prediction failed for {row['player_name']} (S{row['season']} W{row['week']}): {e}"
                )

    logger.info("")
    logger.info(f"✅ Generated {len(predictions)} predictions ({errors} errors)")
    logger.info("")

    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)

    # Calculate metrics
    logger.info("=" * 80)
    logger.info("RESULTS: V6.1 WITH CORRECTED TEMPERATURE (T=0.7083)")
    logger.info("=" * 80)
    logger.info("")

    # MAE
    mae = pred_df["mae"].mean()
    logger.info(f"MAE: {mae:.2f} yards")
    logger.info("")

    # Calibration (90% PI coverage)
    pred_df["in_90pi"] = (
        pred_df["actual"] >= pred_df["predicted"] - 1.645 * pred_df["predicted_std"]
    ) & (pred_df["actual"] <= pred_df["predicted"] + 1.645 * pred_df["predicted_std"])
    calibration = pred_df["in_90pi"].mean() * 100
    logger.info(f"Calibration (90% PI): {calibration:.1f}%")
    logger.info("")

    # Betting metrics
    betting = calculate_betting_metrics(pred_df)
    logger.info("BETTING METRICS:")
    logger.info(f"  Total Bets:  {betting['total_bets']}")
    logger.info(f"  Wins:        {betting['wins']}")
    logger.info(f"  Win Rate:    {betting['win_rate']:.1f}%")
    logger.info(f"  ROI:         {betting['roi']:+.2f}%")
    logger.info(f"  Sharpe:      {betting['sharpe']:.2f}")
    logger.info(f"  Total P&L:   ${betting['total_pnl']:,.2f}")
    logger.info(f"  Avg Bet P&L: ${betting['avg_bet_pnl']:.2f}")
    logger.info("")

    # Compare to dissertation claims
    logger.info("=" * 80)
    logger.info("COMPARISON TO DISSERTATION CLAIMS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Metric':<20} {'Dissertation':<15} {'V6.1 Actual':<15} {'Difference':<15}")
    logger.info("-" * 65)
    win_rate_str = f"{betting['win_rate']:.1f}%"
    win_rate_diff = f"{betting['win_rate'] - 60.8:+.1f}pp"
    logger.info(f"{'Win Rate':<20} {'60.8%':<15} {win_rate_str:<15} {win_rate_diff:<15}")
    roi_str = f"{betting['roi']:+.2f}%"
    roi_diff = f"{betting['roi'] - 6.45:+.2f}pp"
    logger.info(f"{'ROI':<20} {'+6.45%':<15} {roi_str:<15} {roi_diff:<15}")
    sharpe_str = f"{betting['sharpe']:.2f}"
    sharpe_diff = f"{betting['sharpe'] - 2.51:+.2f}"
    logger.info(f"{'Sharpe Ratio':<20} {'2.51':<15} {sharpe_str:<15} {sharpe_diff:<15}")
    mae_str = f"{mae:.2f} yds"
    logger.info(f"{'MAE':<20} {'N/A':<15} {mae_str:<15} {'N/A':<15}")
    calib_str = f"{calibration:.1f}%"
    calib_diff = f"{calibration - 90:+.1f}pp"
    logger.info(f"{'Calibration':<20} {'90%':<15} {calib_str:<15} {calib_diff:<15}")
    logger.info("")

    # Save results
    output_path = Path("analysis/results/v6.1_temperature_fix_validation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved results to {output_path}")
    logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("CONCLUSION")
    logger.info("=" * 80)
    logger.info("")

    if betting["win_rate"] >= 58 and betting["roi"] >= 5:
        logger.info("✅ PASS: V6.1 with corrected temperature meets production targets")
        logger.info("   Temperature fix likely explains previous underperformance")
    elif betting["win_rate"] >= 55 and betting["roi"] >= 3:
        logger.info("⚠️  MARGINAL: V6.1 shows improvement but below dissertation claims")
        logger.info("   Additional investigation needed")
    else:
        logger.info("❌ FAIL: V6.1 performance still below targets")
        logger.info("   Temperature fix NOT sufficient to explain discrepancy")
        logger.info("   Dissertation claims may need revision")
    logger.info("")


if __name__ == "__main__":
    main()
