"""
Weekly Outcome Updater.

Updates ml.game_predictions with actual outcomes from completed games.
Triggers auto-calculation of metrics (MAE, Brier, spread cover, etc.)

Usage:
    updater = WeeklyOutcomeUpdater()
    result = updater.update_week(season=2025, week=16)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of outcome update operation."""
    season: int
    games_with_scores: int = 0
    predictions_found: int = 0
    predictions_updated: int = 0
    already_complete: int = 0
    weeks_processed: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


class WeeklyOutcomeUpdater:
    """
    Updates ml.game_predictions with actual game outcomes.

    After games complete on Sunday/Monday/Thursday:
    1. Query games table for final scores
    2. Calculate actual spread and total
    3. Update ml.game_predictions with actual values
    4. Auto-triggers calculate spread_error, total_error, etc.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
    ):
        """Initialize with database connection parameters."""
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)

    def validate_completeness(self, season: int, week: int) -> dict:
        """
        Check if all games in a week have final scores.

        Returns:
            Dict with 'complete', 'total_games', 'games_with_scores'
        """
        query = """
        SELECT
            COUNT(*) as total_games,
            COUNT(home_score) as games_with_scores
        FROM games
        WHERE season = %s AND week = %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, week))
                row = cur.fetchone()

        return {
            "complete": row["total_games"] == row["games_with_scores"] and row["total_games"] > 0,
            "total_games": row["total_games"],
            "games_with_scores": row["games_with_scores"],
        }

    def get_completed_games(self, season: int, week: int) -> list[dict]:
        """
        Get all completed games for a week with scores.

        Returns:
            List of game dicts with scores and calculated spread/total
        """
        query = """
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score,
            home_score + away_score AS actual_total,
            home_score - away_score AS actual_spread,
            CASE WHEN home_score > away_score THEN TRUE
                 WHEN home_score < away_score THEN FALSE
                 ELSE NULL END AS home_won
        FROM games
        WHERE season = %s AND week = %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY game_id
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, week))
                return [dict(row) for row in cur.fetchall()]

    def get_predictions_needing_update(
        self,
        season: int,
        week: int,
        force: bool = False
    ) -> list[dict]:
        """
        Get predictions that need outcome updates.

        Args:
            season: Season year
            week: Week number
            force: If True, include predictions that already have outcomes

        Returns:
            List of prediction dicts needing updates
        """
        if force:
            # Get all predictions for the week
            query = """
            SELECT id, game_id, model_id, predicted_spread, predicted_total,
                   actual_home_score, actual_spread
            FROM ml.game_predictions
            WHERE season = %s AND week = %s
            """
        else:
            # Only get predictions missing outcomes
            query = """
            SELECT id, game_id, model_id, predicted_spread, predicted_total,
                   actual_home_score, actual_spread
            FROM ml.game_predictions
            WHERE season = %s AND week = %s
              AND actual_home_score IS NULL
            """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, week))
                return [dict(row) for row in cur.fetchall()]

    def update_week(
        self,
        season: int,
        week: int,
        dry_run: bool = False,
        force: bool = False
    ) -> UpdateResult:
        """
        Update all predictions for a week with actual outcomes.

        Args:
            season: Season year
            week: Week number
            dry_run: If True, don't actually update
            force: If True, update even if outcomes already exist

        Returns:
            UpdateResult with statistics
        """
        result = UpdateResult(season=season, dry_run=dry_run)
        result.weeks_processed = [week]

        # Get completed games
        games = self.get_completed_games(season, week)
        result.games_with_scores = len(games)

        if not games:
            logger.info(f"No completed games found for {season} Week {week}")
            return result

        # Build game lookup
        game_outcomes = {g["game_id"]: g for g in games}
        logger.info(f"Found {len(games)} completed games for {season} Week {week}")

        # Get predictions needing update
        predictions = self.get_predictions_needing_update(season, week, force)
        result.predictions_found = len(predictions)

        if not predictions:
            # Check if predictions exist at all
            all_preds = self.get_predictions_needing_update(season, week, force=True)
            if all_preds:
                result.already_complete = len(all_preds)
                logger.info(f"All {len(all_preds)} predictions already have outcomes")
            else:
                logger.info(f"No predictions found for {season} Week {week}")
            return result

        # Update predictions
        update_query = """
        UPDATE ml.game_predictions
        SET actual_home_score = %s,
            actual_away_score = %s,
            actual_total = %s,
            actual_spread = %s,
            home_won = %s,
            spread_error = predicted_spread - %s,
            total_error = predicted_total - %s,
            home_score_error = predicted_home_score - %s,
            away_score_error = predicted_away_score - %s,
            abs_spread_error = ABS(predicted_spread - %s),
            abs_total_error = ABS(predicted_total - %s),
            status = 'completed',
            updated_at = NOW()
        WHERE id = %s
        """

        updated = 0
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for pred in predictions:
                    game_id = pred["game_id"]
                    outcome = game_outcomes.get(game_id)

                    if not outcome:
                        result.errors.append(f"No outcome for game {game_id}")
                        continue

                    if dry_run:
                        logger.debug(f"Would update {game_id}: spread={outcome['actual_spread']}")
                        updated += 1
                    else:
                        try:
                            cur.execute(update_query, (
                                outcome["home_score"],
                                outcome["away_score"],
                                outcome["actual_total"],
                                outcome["actual_spread"],
                                outcome["home_won"],
                                outcome["actual_spread"],  # For spread_error
                                outcome["actual_total"],   # For total_error
                                outcome["home_score"],     # For home_score_error
                                outcome["away_score"],     # For away_score_error
                                outcome["actual_spread"],  # For abs_spread_error
                                outcome["actual_total"],   # For abs_total_error
                                pred["id"]
                            ))
                            updated += 1
                        except Exception as e:
                            result.errors.append(f"Failed to update {game_id}: {e}")

            if not dry_run:
                conn.commit()

        result.predictions_updated = updated
        logger.info(f"Updated {updated} predictions with outcomes")

        return result

    def update_season(
        self,
        season: int,
        dry_run: bool = False,
        force: bool = False
    ) -> UpdateResult:
        """
        Update all completed weeks in a season.

        Args:
            season: Season year
            dry_run: If True, don't actually update
            force: If True, update even if outcomes already exist

        Returns:
            Aggregated UpdateResult
        """
        result = UpdateResult(season=season, dry_run=dry_run)

        # Get weeks with completed games
        query = """
        SELECT DISTINCT week
        FROM games
        WHERE season = %s
          AND home_score IS NOT NULL
        ORDER BY week
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (season,))
                weeks = [row[0] for row in cur.fetchall()]

        logger.info(f"Found {len(weeks)} weeks with completed games")

        for week in weeks:
            week_result = self.update_week(season, week, dry_run, force)
            result.games_with_scores += week_result.games_with_scores
            result.predictions_found += week_result.predictions_found
            result.predictions_updated += week_result.predictions_updated
            result.already_complete += week_result.already_complete
            result.weeks_processed.append(week)
            result.errors.extend(week_result.errors)

        return result

    def get_update_summary(self, season: int, week: int | None = None) -> dict:
        """
        Get summary of prediction vs outcome status.

        Returns:
            Dict with counts of predictions by status
        """
        if week:
            where_clause = "WHERE season = %s AND week = %s"
            params = (season, week)
        else:
            where_clause = "WHERE season = %s"
            params = (season,)

        query = f"""
        SELECT
            COUNT(*) as total_predictions,
            COUNT(actual_home_score) as with_outcomes,
            COUNT(*) - COUNT(actual_home_score) as pending_outcomes,
            AVG(abs_spread_error) as avg_spread_error,
            AVG(abs_total_error) as avg_total_error,
            SUM(CASE WHEN (predicted_spread > 0) = (actual_spread > 0) THEN 1 ELSE 0 END)::FLOAT
                / NULLIF(COUNT(actual_spread), 0) as spread_accuracy
        FROM ml.game_predictions
        {where_clause}
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return dict(cur.fetchone())


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    updater = WeeklyOutcomeUpdater()

    print("\n=== Checking Week 13 Completeness ===")
    status = updater.validate_completeness(2025, 13)
    print(f"Complete: {status['complete']}")
    print(f"Games with scores: {status['games_with_scores']}/{status['total_games']}")

    print("\n=== Getting Completed Games ===")
    games = updater.get_completed_games(2025, 13)
    for g in games[:3]:
        print(f"  {g['game_id']}: {g['home_score']}-{g['away_score']} (spread={g['actual_spread']})")

    print("\n=== Update Summary ===")
    summary = updater.get_update_summary(2025)
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"With outcomes: {summary['with_outcomes']}")
