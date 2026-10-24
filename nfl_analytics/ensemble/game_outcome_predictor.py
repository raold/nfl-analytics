#!/usr/bin/env python3
"""
Game Outcome Predictor - Real Ensemble v6.1 Integration
========================================================

Predicts game outcomes (scores, spreads, totals, win probabilities) by:
1. Finding starting QBs for both teams
2. Using Ensemble v6.1 to predict QB passing yards
3. Aggregating QB performance → team scores using defensive adjustments
4. Propagating uncertainty from ensemble predictions

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
Version: 1.0.0
"""

from typing import Any

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
}

# Default values
DEFAULT_QB_YARDS = 240.0  # Average QB passing yards
DEFAULT_DEF_EPA = 0.0  # Neutral defensive EPA
DEFENSIVE_UNCERTAINTY_STD = 3.5  # Additional uncertainty from defense (points)


class GameOutcomePredictor:
    """
    Predicts game outcomes using Ensemble v6.1 QB predictions.

    Converts QB passing yards predictions into team scores by:
    - Base score from QB performance: 10 + (qb_yards / 20)
    - Defensive adjustment: -5 * opponent_def_epa_per_play
    - Propagates uncertainty from ensemble + defensive variability
    """

    def __init__(self):
        """Initialize predictor with Ensemble v6.1."""
        logger.info("Initializing GameOutcomePredictor with Ensemble v6.1")
        self.ensemble = EnhancedEnsemble5Way()
        logger.info("Ensemble v6.1 loaded successfully")

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(**DB_CONFIG)

    def _get_starting_qb(self, team: str, season: int, week: int, conn) -> dict[str, Any] | None:
        """
        Get starting QB for a team in a given week.

        Falls back to most recent starter from previous season if current data unavailable.
        This allows the ensemble to use hierarchical priors and canonical window
        to predict future games based on historical performance.

        Args:
            team: Team abbreviation (e.g., 'BUF', 'HOU')
            season: NFL season
            week: NFL week
            conn: Database connection

        Returns:
            Dictionary with player_id, player_display_name, or None if not found
        """
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # First try current season/week
        query_current = """
        SELECT
            player_id,
            player_display_name,
            current_team,
            stat_attempts
        FROM mart.player_game_stats
        WHERE season = %s
            AND week = %s
            AND current_team = %s
            AND stat_category = 'passing'
        ORDER BY stat_attempts DESC
        LIMIT 1;
        """

        cursor.execute(query_current, (season, week, team))
        qb = cursor.fetchone()

        if qb:
            logger.info(
                f"Found starting QB for {team} ({season} Week {week}): {qb['player_display_name']} "
                f"(id={qb['player_id']}, attempts={qb['stat_attempts']})"
            )
            cursor.close()
            return qb

        # Fallback: Get most recent starter from previous season
        # This lets ensemble use hierarchical priors and canonical window for prediction
        logger.warning(f"No QB found for {team} in {season} Week {week}, checking previous season")

        query_fallback = """
        WITH recent_qbs AS (
            SELECT
                player_id,
                player_display_name,
                current_team,
                season,
                week,
                SUM(stat_attempts) OVER (PARTITION BY player_id ORDER BY season DESC, week DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_attempts
            FROM mart.player_game_stats
            WHERE season = %s
                AND current_team = %s
                AND stat_category = 'passing'
        )
        SELECT DISTINCT ON (current_team)
            player_id,
            player_display_name,
            current_team,
            total_attempts as stat_attempts
        FROM recent_qbs
        ORDER BY current_team, season DESC, week DESC, total_attempts DESC
        LIMIT 1;
        """

        cursor.execute(query_fallback, (season - 1, team))
        qb_fallback = cursor.fetchone()
        cursor.close()

        if qb_fallback:
            logger.info(
                f"Using most recent starter for {team}: {qb_fallback['player_display_name']} "
                f"(id={qb_fallback['player_id']}) - ensemble will use hierarchical priors for {season} Week {week}"
            )
            return qb_fallback
        else:
            logger.error(f"No QB data found for {team} in current or previous season")
            return None

    def _get_defensive_epa(self, team: str, season: int, week: int, conn) -> float:
        """
        Get defensive EPA per play for a team (from previous games).

        Falls back to previous season if current season data unavailable.

        Args:
            team: Team abbreviation
            season: NFL season
            week: NFL week
            conn: Database connection

        Returns:
            Average defensive EPA per play (negative = good defense)
        """
        cursor = conn.cursor()

        # Try current season first
        query_current = """
        WITH recent_defense AS (
            SELECT
                g.game_id,
                g.season,
                g.week,
                CASE
                    WHEN g.home_team = %s THEN g.away_team
                    ELSE g.home_team
                END as opponent,
                CASE
                    WHEN g.home_team = %s THEN te_away.epa_mean
                    ELSE te_home.epa_mean
                END as opponent_epa
            FROM games g
            LEFT JOIN mart.team_epa te_home ON g.game_id = te_home.game_id AND g.home_team = te_home.posteam
            LEFT JOIN mart.team_epa te_away ON g.game_id = te_away.game_id AND g.away_team = te_away.posteam
            WHERE (g.home_team = %s OR g.away_team = %s)
                AND g.season = %s
                AND g.week < %s
            ORDER BY g.week DESC
            LIMIT 5
        )
        SELECT AVG(opponent_epa) as avg_def_epa
        FROM recent_defense
        WHERE opponent_epa IS NOT NULL;
        """

        cursor.execute(query_current, (team, team, team, team, season, week))
        result = cursor.fetchone()

        if result and result[0] is not None:
            def_epa = float(result[0])
            logger.info(f"Defensive EPA for {team} ({season}): {def_epa:.4f}")
            cursor.close()
            return def_epa

        # Fallback to previous season
        logger.warning(f"No defensive EPA for {team} in {season}, checking {season-1}")

        query_fallback = """
        WITH recent_defense AS (
            SELECT
                g.game_id,
                g.season,
                g.week,
                CASE
                    WHEN g.home_team = %s THEN te_away.epa_mean
                    ELSE te_home.epa_mean
                END as opponent_epa
            FROM games g
            LEFT JOIN mart.team_epa te_home ON g.game_id = te_home.game_id AND g.home_team = te_home.posteam
            LEFT JOIN mart.team_epa te_away ON g.game_id = te_away.game_id AND g.away_team = te_away.posteam
            WHERE (g.home_team = %s OR g.away_team = %s)
                AND g.season = %s
            ORDER BY g.week DESC
            LIMIT 8
        )
        SELECT AVG(opponent_epa) as avg_def_epa
        FROM recent_defense
        WHERE opponent_epa IS NOT NULL;
        """

        cursor.execute(query_fallback, (team, team, team, season - 1))
        result_fallback = cursor.fetchone()
        cursor.close()

        if result_fallback and result_fallback[0] is not None:
            def_epa = float(result_fallback[0])
            logger.info(f"Using {season-1} defensive EPA for {team}: {def_epa:.4f}")
            return def_epa
        else:
            logger.warning(f"No defensive EPA data for {team}, using default {DEFAULT_DEF_EPA}")
            return DEFAULT_DEF_EPA

    def _predict_qb_yards(self, player_id: str, season: int, week: int) -> tuple[float, float]:
        """
        Predict QB passing yards using Ensemble v6.1.

        Args:
            player_id: Player ID
            season: NFL season
            week: NFL week

        Returns:
            Tuple of (predicted_yards, uncertainty_std)
        """
        try:
            start = time.time()
            result = self.ensemble.predict(player_id=player_id, season=season, week=week)
            elapsed = time.time() - start

            pred_yards = result["ensemble_mean"]
            pred_std = result["ensemble_std"]

            logger.info(
                f"Ensemble prediction for {player_id}: "
                f"{pred_yards:.1f} ± {pred_std:.1f} yards ({elapsed*1000:.1f}ms)"
            )

            return pred_yards, pred_std

        except Exception as e:
            logger.error(f"Ensemble prediction failed for {player_id}: {e}")
            # Fallback to default
            logger.warning(f"Using default QB yards: {DEFAULT_QB_YARDS}")
            return DEFAULT_QB_YARDS, 50.0  # Default uncertainty

    def _yards_to_score(self, qb_yards: float, opponent_def_epa: float) -> float:
        """
        Convert QB passing yards to team score.

        Formula:
            base_score = 10 + (qb_yards / 20)
            defensive_adjustment = -5 * opponent_def_epa_per_play
            team_score = base_score + defensive_adjustment

        Args:
            qb_yards: Predicted QB passing yards
            opponent_def_epa: Opponent's defensive EPA per play

        Returns:
            Predicted team score
        """
        base_score = 10 + (qb_yards / 20)
        defensive_adjustment = -5 * opponent_def_epa
        team_score = base_score + defensive_adjustment

        logger.debug(
            f"Score conversion: {qb_yards:.1f} yards → "
            f"base={base_score:.1f}, def_adj={defensive_adjustment:.1f}, "
            f"final={team_score:.1f}"
        )

        return team_score

    def _propagate_uncertainty(self, qb_std: float) -> float:
        """
        Propagate uncertainty from QB prediction to score prediction.

        Args:
            qb_std: QB yards prediction standard deviation

        Returns:
            Score prediction standard deviation
        """
        # Convert yards std to points: score = 10 + yards/20 → d(score)/d(yards) = 1/20
        qb_score_std = qb_std / 20

        # Add defensive uncertainty (independent)
        total_std = np.sqrt(qb_score_std**2 + DEFENSIVE_UNCERTAINTY_STD**2)

        return total_std

    def predict_game(
        self, home_team: str, away_team: str, season: int, week: int
    ) -> dict[str, Any]:
        """
        Predict full game outcome.

        Args:
            home_team: Home team abbreviation (e.g., 'HOU')
            away_team: Away team abbreviation (e.g., 'BUF')
            season: NFL season
            week: NFL week

        Returns:
            Dictionary with:
                - predicted_home_score
                - predicted_away_score
                - predicted_total
                - predicted_spread (positive = home favored)
                - predicted_home_win_prob
                - spread_std
                - total_std
                - home_qb_id, home_qb_name
                - away_qb_id, away_qb_name
        """
        logger.info(f"Predicting game: {away_team} @ {home_team} ({season} Week {week})")
        start = time.time()

        conn = self._get_db_connection()

        try:
            # 1. Get starting QBs
            home_qb = self._get_starting_qb(home_team, season, week, conn)
            away_qb = self._get_starting_qb(away_team, season, week, conn)

            # 2. Get defensive EPA
            away_def_epa = self._get_defensive_epa(away_team, season, week, conn)
            home_def_epa = self._get_defensive_epa(home_team, season, week, conn)

            # 3. Predict QB passing yards
            if home_qb:
                home_qb_yards, home_qb_std = self._predict_qb_yards(
                    home_qb["player_id"], season, week
                )
            else:
                logger.warning(f"No home QB found, using default {DEFAULT_QB_YARDS} yards")
                home_qb_yards, home_qb_std = DEFAULT_QB_YARDS, 50.0

            if away_qb:
                away_qb_yards, away_qb_std = self._predict_qb_yards(
                    away_qb["player_id"], season, week
                )
            else:
                logger.warning(f"No away QB found, using default {DEFAULT_QB_YARDS} yards")
                away_qb_yards, away_qb_std = DEFAULT_QB_YARDS, 50.0

            # 4. Convert yards to scores
            home_score = self._yards_to_score(home_qb_yards, away_def_epa)
            away_score = self._yards_to_score(away_qb_yards, home_def_epa)

            # 5. Calculate derived metrics
            predicted_total = home_score + away_score
            predicted_spread = home_score - away_score

            # Win probability (logistic function)
            # P(home wins) = 1 / (1 + exp(-spread / 7))
            predicted_home_win_prob = 1 / (1 + np.exp(-predicted_spread / 7))

            # 6. Propagate uncertainty
            home_score_std = self._propagate_uncertainty(home_qb_std)
            away_score_std = self._propagate_uncertainty(away_qb_std)

            # Combined uncertainty (independent)
            total_std = np.sqrt(home_score_std**2 + away_score_std**2)
            spread_std = total_std  # Spread uncertainty same as total

            elapsed = time.time() - start

            result = {
                "predicted_home_score": float(round(home_score, 1)),
                "predicted_away_score": float(round(away_score, 1)),
                "predicted_total": float(round(predicted_total, 1)),
                "predicted_spread": float(round(predicted_spread, 1)),
                "predicted_home_win_prob": float(round(predicted_home_win_prob, 4)),
                "spread_std": float(round(spread_std, 1)),
                "total_std": float(round(total_std, 1)),
                "home_qb_id": home_qb["player_id"] if home_qb else None,
                "home_qb_name": home_qb["player_display_name"] if home_qb else None,
                "away_qb_id": away_qb["player_id"] if away_qb else None,
                "away_qb_name": away_qb["player_display_name"] if away_qb else None,
                "processing_time_ms": float(round(elapsed * 1000, 1)),
            }

            logger.info(
                f"Game prediction complete: {away_team} @ {home_team} → "
                f"Total: {result['predicted_total']:.1f}, "
                f"Spread: {result['predicted_spread']:.1f}, "
                f"Win%: {result['predicted_home_win_prob']:.1%} "
                f"({elapsed*1000:.1f}ms)"
            )

            return result

        finally:
            conn.close()


if __name__ == "__main__":
    # Test prediction
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    predictor = GameOutcomePredictor()

    # Test with 2025 Week 12: BUF @ HOU
    result = predictor.predict_game(home_team="HOU", away_team="BUF", season=2025, week=12)

    print("\n" + "=" * 60)
    print("GAME PREDICTION TEST: BUF @ HOU (2025 Week 12)")
    print("=" * 60)
    print(f"Home Score (HOU): {result['predicted_home_score']:.1f}")
    print(f"Away Score (BUF): {result['predicted_away_score']:.1f}")
    print(f"Total: {result['predicted_total']:.1f} ± {result['total_std']:.1f}")
    print(f"Spread: {result['predicted_spread']:.1f} ± {result['spread_std']:.1f} (HOU favored)")
    print(f"Home Win Prob: {result['predicted_home_win_prob']:.1%}")
    print(f"\nHome QB: {result['home_qb_name']} (id={result['home_qb_id']})")
    print(f"Away QB: {result['away_qb_name']} (id={result['away_qb_id']})")
    print(f"\nProcessing Time: {result['processing_time_ms']:.1f}ms")
    print("=" * 60)
