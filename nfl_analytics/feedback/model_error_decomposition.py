"""
Model Error Decomposition for Ensemble Predictions.

Analyzes prediction errors from `ml.game_outcome_predictions` by comparing
the ensemble model against Vegas lines and categorizing errors by magnitude
and game context.

Since individual component model predictions (state_space, hierarchical, etc.)
are NOT stored in the database, this module focuses on:
  1. Ensemble error: actual_spread - pred_home_spread
  2. Vegas error: actual_spread - vegas_spread
  3. Model vs Vegas comparison per game
  4. Error categorization by magnitude and game context
  5. Season-level error analysis

Usage:
    decomposer = ModelErrorDecomposer()
    result = decomposer.decompose_game_error(game_id='2025_16_KC_PIT')
    season_report = decomposer.decompose_season(season=2025)
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

logger = logging.getLogger(__name__)


# Error magnitude thresholds (points)
ERROR_SMALL = 3.0  # Within a field goal
ERROR_MEDIUM = 7.0  # Within a touchdown
ERROR_LARGE = 14.0  # Within two touchdowns
# Above LARGE = blowout miss


@dataclass
class ErrorDecomposition:
    """Result of decomposing a single game's prediction error."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str

    # Actuals
    actual_home_score: float
    actual_away_score: float
    actual_spread: float  # home - away (negative = away favored by result)
    actual_total: float

    # Model predictions
    pred_home_spread: float
    pred_total: float
    pred_home_win_prob: float
    spread_std: float | None = None
    total_std: float | None = None

    # Vegas lines
    vegas_spread: float | None = None
    vegas_total: float | None = None

    # Computed errors (signed: actual - predicted)
    model_spread_error: float = 0.0
    model_total_error: float = 0.0
    vegas_spread_error: float | None = None
    vegas_total_error: float | None = None

    # Comparison
    model_beat_vegas_spread: bool | None = None
    model_beat_vegas_total: bool | None = None
    model_correct_side: bool = False  # Did model pick the right side?
    vegas_correct_side: bool | None = None  # Did Vegas pick the right side?

    # Categorization
    error_magnitude: str = ""  # small / medium / large / blowout
    game_context: str = ""  # close_game / moderate / blowout

    # Metadata
    model_version: str = ""
    inference_time_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "season": self.season,
            "week": self.week,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "actual_home_score": self.actual_home_score,
            "actual_away_score": self.actual_away_score,
            "actual_spread": self.actual_spread,
            "actual_total": self.actual_total,
            "pred_home_spread": self.pred_home_spread,
            "pred_total": self.pred_total,
            "pred_home_win_prob": self.pred_home_win_prob,
            "spread_std": self.spread_std,
            "total_std": self.total_std,
            "vegas_spread": self.vegas_spread,
            "vegas_total": self.vegas_total,
            "model_spread_error": self.model_spread_error,
            "model_total_error": self.model_total_error,
            "vegas_spread_error": self.vegas_spread_error,
            "vegas_total_error": self.vegas_total_error,
            "model_beat_vegas_spread": self.model_beat_vegas_spread,
            "model_beat_vegas_total": self.model_beat_vegas_total,
            "model_correct_side": self.model_correct_side,
            "vegas_correct_side": self.vegas_correct_side,
            "error_magnitude": self.error_magnitude,
            "game_context": self.game_context,
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
        }


@dataclass
class SeasonErrorReport:
    """Aggregated error analysis for an entire season."""

    season: int
    total_games: int = 0

    # Model performance
    model_spread_mae: float = 0.0
    model_spread_rmse: float = 0.0
    model_total_mae: float = 0.0
    model_total_rmse: float = 0.0
    model_spread_bias: float = 0.0  # Mean signed error (positive = model underestimates)
    model_total_bias: float = 0.0

    # Vegas performance
    vegas_spread_mae: float | None = None
    vegas_total_mae: float | None = None

    # Head-to-head
    model_beat_vegas_spread_pct: float | None = None
    model_beat_vegas_total_pct: float | None = None
    games_with_vegas: int = 0

    # Side accuracy
    model_correct_side_pct: float = 0.0
    vegas_correct_side_pct: float | None = None

    # Error distribution
    error_magnitude_counts: dict[str, int] = field(default_factory=dict)

    # Weekly breakdown
    weekly_mae: dict[int, float] = field(default_factory=dict)

    # Worst misses
    worst_games: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "total_games": self.total_games,
            "model_spread_mae": round(self.model_spread_mae, 4),
            "model_spread_rmse": round(self.model_spread_rmse, 4),
            "model_total_mae": round(self.model_total_mae, 4),
            "model_total_rmse": round(self.model_total_rmse, 4),
            "model_spread_bias": round(self.model_spread_bias, 4),
            "model_total_bias": round(self.model_total_bias, 4),
            "vegas_spread_mae": round(self.vegas_spread_mae, 4) if self.vegas_spread_mae else None,
            "vegas_total_mae": round(self.vegas_total_mae, 4) if self.vegas_total_mae else None,
            "model_beat_vegas_spread_pct": (
                round(self.model_beat_vegas_spread_pct, 4)
                if self.model_beat_vegas_spread_pct
                else None
            ),
            "model_beat_vegas_total_pct": (
                round(self.model_beat_vegas_total_pct, 4)
                if self.model_beat_vegas_total_pct
                else None
            ),
            "games_with_vegas": self.games_with_vegas,
            "model_correct_side_pct": round(self.model_correct_side_pct, 4),
            "vegas_correct_side_pct": (
                round(self.vegas_correct_side_pct, 4) if self.vegas_correct_side_pct else None
            ),
            "error_magnitude_counts": self.error_magnitude_counts,
            "weekly_mae": {k: round(v, 4) for k, v in self.weekly_mae.items()},
            "worst_games": self.worst_games,
        }


def _classify_error_magnitude(abs_error: float) -> str:
    """Classify absolute error into magnitude bucket."""
    if abs_error <= ERROR_SMALL:
        return "small"
    elif abs_error <= ERROR_MEDIUM:
        return "medium"
    elif abs_error <= ERROR_LARGE:
        return "large"
    else:
        return "blowout"


def _classify_game_context(actual_spread: float) -> str:
    """Classify game by how close the actual result was."""
    margin = abs(actual_spread)
    if margin <= 3:
        return "close_game"
    elif margin <= 10:
        return "moderate"
    else:
        return "blowout"


class ModelErrorDecomposer:
    """
    Decompose ensemble prediction errors using data from ml.game_outcome_predictions.

    Since individual component model predictions are not stored, this class
    compares the ensemble prediction against:
      - Actual outcomes (spread, total)
      - Vegas lines (spread, total)
    and categorizes errors by magnitude and game context.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str | None = None,
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password or os.environ.get("NFL_DB_PASSWORD", "sicillionbillions"),
        }

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def _fetch_game(self, game_id: str) -> dict | None:
        """Fetch a single game's prediction and outcome from ml.game_outcome_predictions."""
        query = """
        SELECT
            game_id, season, week, home_team, away_team,
            pred_home_spread, pred_total, pred_home_win_prob,
            spread_std, total_std,
            vegas_spread, vegas_total,
            actual_home_score, actual_away_score,
            actual_spread, actual_total,
            model_version, inference_time_ms
        FROM ml.game_outcome_predictions
        WHERE game_id = %s
          AND actual_spread IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 1
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (game_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def _fetch_games_for_week(self, season: int, week: int) -> list[dict]:
        """Fetch all games with outcomes for a given week."""
        query = """
        SELECT
            game_id, season, week, home_team, away_team,
            pred_home_spread, pred_total, pred_home_win_prob,
            spread_std, total_std,
            vegas_spread, vegas_total,
            actual_home_score, actual_away_score,
            actual_spread, actual_total,
            model_version, inference_time_ms
        FROM ml.game_outcome_predictions
        WHERE season = %s AND week = %s
          AND actual_spread IS NOT NULL
        ORDER BY game_id
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, week))
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def _fetch_games_for_season(self, season: int) -> list[dict]:
        """Fetch all games with outcomes for an entire season."""
        query = """
        SELECT
            game_id, season, week, home_team, away_team,
            pred_home_spread, pred_total, pred_home_win_prob,
            spread_std, total_std,
            vegas_spread, vegas_total,
            actual_home_score, actual_away_score,
            actual_spread, actual_total,
            model_version, inference_time_ms
        FROM ml.game_outcome_predictions
        WHERE season = %s
          AND actual_spread IS NOT NULL
        ORDER BY week, game_id
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season,))
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def _build_decomposition(self, row: dict) -> ErrorDecomposition:
        """Build an ErrorDecomposition from a database row."""
        actual_spread = float(row["actual_spread"])
        pred_spread = float(row["pred_home_spread"])
        actual_total = float(row["actual_total"])
        pred_total = float(row["pred_total"])

        # Signed errors: positive means model underestimated (actual > predicted)
        model_spread_error = actual_spread - pred_spread
        model_total_error = actual_total - pred_total

        # Vegas comparison
        vegas_spread = row.get("vegas_spread")
        vegas_total = row.get("vegas_total")
        vegas_spread_error = None
        vegas_total_error = None
        model_beat_vegas_spread = None
        model_beat_vegas_total = None
        vegas_correct_side = None

        if vegas_spread is not None:
            vegas_spread = float(vegas_spread)
            vegas_spread_error = actual_spread - vegas_spread
            model_beat_vegas_spread = abs(model_spread_error) < abs(vegas_spread_error)
            # Vegas correct side: did vegas pick the right winner direction?
            if actual_spread != 0:
                vegas_correct_side = (vegas_spread > 0 and actual_spread > 0) or (
                    vegas_spread < 0 and actual_spread < 0
                )

        if vegas_total is not None:
            vegas_total = float(vegas_total)
            vegas_total_error = actual_total - vegas_total
            model_beat_vegas_total = abs(model_total_error) < abs(vegas_total_error)

        # Model correct side: did model pick the right winner direction?
        model_correct_side = False
        if actual_spread != 0:
            model_correct_side = (pred_spread > 0 and actual_spread > 0) or (
                pred_spread < 0 and actual_spread < 0
            )

        # Categorize
        error_magnitude = _classify_error_magnitude(abs(model_spread_error))
        game_context = _classify_game_context(actual_spread)

        return ErrorDecomposition(
            game_id=row["game_id"],
            season=int(row["season"]),
            week=int(row["week"]),
            home_team=row["home_team"],
            away_team=row["away_team"],
            actual_home_score=float(row["actual_home_score"]),
            actual_away_score=float(row["actual_away_score"]),
            actual_spread=actual_spread,
            actual_total=actual_total,
            pred_home_spread=pred_spread,
            pred_total=pred_total,
            pred_home_win_prob=float(row["pred_home_win_prob"]),
            spread_std=float(row["spread_std"]) if row.get("spread_std") else None,
            total_std=float(row["total_std"]) if row.get("total_std") else None,
            vegas_spread=vegas_spread,
            vegas_total=float(row["vegas_total"]) if row.get("vegas_total") else None,
            model_spread_error=round(model_spread_error, 4),
            model_total_error=round(model_total_error, 4),
            vegas_spread_error=(
                round(vegas_spread_error, 4) if vegas_spread_error is not None else None
            ),
            vegas_total_error=(
                round(vegas_total_error, 4) if vegas_total_error is not None else None
            ),
            model_beat_vegas_spread=model_beat_vegas_spread,
            model_beat_vegas_total=model_beat_vegas_total,
            model_correct_side=model_correct_side,
            vegas_correct_side=vegas_correct_side,
            error_magnitude=error_magnitude,
            game_context=game_context,
            model_version=row.get("model_version", ""),
            inference_time_ms=(
                float(row["inference_time_ms"]) if row.get("inference_time_ms") else None
            ),
        )

    def decompose_game_error(self, game_id: str) -> ErrorDecomposition | None:
        """
        Decompose prediction error for a single game.

        Args:
            game_id: Game identifier (e.g., '2025_16_KC_PIT')

        Returns:
            ErrorDecomposition with model vs Vegas comparison, or None if not found.
        """
        row = self._fetch_game(game_id)
        if not row:
            logger.warning(
                f"No prediction/outcome found in ml.game_outcome_predictions for {game_id}"
            )
            return None

        return self._build_decomposition(row)

    def decompose_week(
        self,
        season: int,
        week: int,
        save_to_db: bool = True,
    ) -> list[ErrorDecomposition]:
        """
        Decompose errors for all games in a given week.

        Args:
            season: NFL season year.
            week: Week number.
            save_to_db: If True, persist results to ml.model_error_decomposition.

        Returns:
            List of ErrorDecomposition objects.
        """
        rows = self._fetch_games_for_week(season, week)
        logger.info(f"Decomposing errors for {len(rows)} games in {season} Week {week}")

        results = [self._build_decomposition(row) for row in rows]

        if save_to_db and results:
            self._save_decompositions(results)

        return results

    def decompose_season(self, season: int, save_to_db: bool = False) -> SeasonErrorReport:
        """
        Produce a full-season error analysis.

        Args:
            season: NFL season year.
            save_to_db: If True, persist per-game results to ml.model_error_decomposition.

        Returns:
            SeasonErrorReport with aggregated metrics.
        """
        rows = self._fetch_games_for_season(season)
        if not rows:
            logger.warning(f"No games with outcomes found for season {season}")
            return SeasonErrorReport(season=season)

        decompositions = [self._build_decomposition(row) for row in rows]

        if save_to_db and decompositions:
            self._save_decompositions(decompositions)

        # Aggregate
        spread_errors = np.array([d.model_spread_error for d in decompositions])
        total_errors = np.array([d.model_total_error for d in decompositions])
        abs_spread_errors = np.abs(spread_errors)

        report = SeasonErrorReport(
            season=season,
            total_games=len(decompositions),
            model_spread_mae=float(np.mean(np.abs(spread_errors))),
            model_spread_rmse=float(np.sqrt(np.mean(spread_errors**2))),
            model_total_mae=float(np.mean(np.abs(total_errors))),
            model_total_rmse=float(np.sqrt(np.mean(total_errors**2))),
            model_spread_bias=float(np.mean(spread_errors)),
            model_total_bias=float(np.mean(total_errors)),
        )

        # Side accuracy
        side_correct = [d for d in decompositions if d.model_correct_side]
        # Exclude pushes (actual_spread == 0) from side accuracy denominator
        non_push_games = [d for d in decompositions if d.actual_spread != 0]
        if non_push_games:
            report.model_correct_side_pct = len(side_correct) / len(non_push_games)

        # Vegas comparison (only games with vegas lines)
        vegas_games = [d for d in decompositions if d.vegas_spread is not None]
        report.games_with_vegas = len(vegas_games)

        if vegas_games:
            vegas_spread_errors = np.array([d.vegas_spread_error for d in vegas_games])
            report.vegas_spread_mae = float(np.mean(np.abs(vegas_spread_errors)))

            vegas_total_games = [d for d in vegas_games if d.vegas_total_error is not None]
            if vegas_total_games:
                vegas_total_errors = np.array([d.vegas_total_error for d in vegas_total_games])
                report.vegas_total_mae = float(np.mean(np.abs(vegas_total_errors)))

            beat_spread = [d for d in vegas_games if d.model_beat_vegas_spread]
            report.model_beat_vegas_spread_pct = len(beat_spread) / len(vegas_games)

            beat_total = [d for d in vegas_games if d.model_beat_vegas_total is True]
            if beat_total or vegas_total_games:
                total_comparable = [d for d in vegas_games if d.model_beat_vegas_total is not None]
                if total_comparable:
                    report.model_beat_vegas_total_pct = len(beat_total) / len(total_comparable)

            vegas_side_correct = [d for d in vegas_games if d.vegas_correct_side]
            vegas_non_push = [d for d in vegas_games if d.actual_spread != 0]
            if vegas_non_push:
                report.vegas_correct_side_pct = len(vegas_side_correct) / len(vegas_non_push)

        # Error magnitude distribution
        magnitude_counts: dict[str, int] = {"small": 0, "medium": 0, "large": 0, "blowout": 0}
        for d in decompositions:
            magnitude_counts[d.error_magnitude] = magnitude_counts.get(d.error_magnitude, 0) + 1
        report.error_magnitude_counts = magnitude_counts

        # Weekly MAE breakdown
        weekly_errors: dict[int, list[float]] = {}
        for d in decompositions:
            weekly_errors.setdefault(d.week, []).append(abs(d.model_spread_error))
        report.weekly_mae = {wk: float(np.mean(errs)) for wk, errs in sorted(weekly_errors.items())}

        # Top 5 worst misses
        sorted_by_error = sorted(
            decompositions, key=lambda d: abs(d.model_spread_error), reverse=True
        )
        report.worst_games = [
            {
                "game_id": d.game_id,
                "week": d.week,
                "matchup": f"{d.away_team} @ {d.home_team}",
                "pred_spread": d.pred_home_spread,
                "actual_spread": d.actual_spread,
                "error": d.model_spread_error,
                "vegas_spread": d.vegas_spread,
                "vegas_error": d.vegas_spread_error,
            }
            for d in sorted_by_error[:5]
        ]

        return report

    def _save_decompositions(self, decompositions: list[ErrorDecomposition]) -> int:
        """Save decomposition results to database."""
        create_table = """
        CREATE TABLE IF NOT EXISTS ml.model_error_decomposition (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(50) NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            home_team VARCHAR(10),
            away_team VARCHAR(10),
            actual_spread NUMERIC(5,2),
            actual_total NUMERIC(5,2),
            pred_home_spread NUMERIC(5,2),
            pred_total NUMERIC(5,2),
            pred_home_win_prob NUMERIC(5,4),
            vegas_spread NUMERIC(5,2),
            vegas_total NUMERIC(5,2),
            model_spread_error NUMERIC(6,4),
            model_total_error NUMERIC(6,4),
            vegas_spread_error NUMERIC(6,4),
            vegas_total_error NUMERIC(6,4),
            model_beat_vegas_spread BOOLEAN,
            model_beat_vegas_total BOOLEAN,
            model_correct_side BOOLEAN,
            vegas_correct_side BOOLEAN,
            error_magnitude VARCHAR(20),
            game_context VARCHAR(20),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (game_id)
        )
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table)
            conn.commit()

        insert_query = """
        INSERT INTO ml.model_error_decomposition
            (game_id, season, week, home_team, away_team,
             actual_spread, actual_total, pred_home_spread, pred_total,
             pred_home_win_prob, vegas_spread, vegas_total,
             model_spread_error, model_total_error,
             vegas_spread_error, vegas_total_error,
             model_beat_vegas_spread, model_beat_vegas_total,
             model_correct_side, vegas_correct_side,
             error_magnitude, game_context, model_version)
        VALUES %s
        ON CONFLICT (game_id) DO UPDATE SET
            model_spread_error = EXCLUDED.model_spread_error,
            model_total_error = EXCLUDED.model_total_error,
            vegas_spread_error = EXCLUDED.vegas_spread_error,
            vegas_total_error = EXCLUDED.vegas_total_error,
            model_beat_vegas_spread = EXCLUDED.model_beat_vegas_spread,
            model_beat_vegas_total = EXCLUDED.model_beat_vegas_total,
            model_correct_side = EXCLUDED.model_correct_side,
            vegas_correct_side = EXCLUDED.vegas_correct_side,
            error_magnitude = EXCLUDED.error_magnitude,
            game_context = EXCLUDED.game_context,
            model_version = EXCLUDED.model_version,
            created_at = CURRENT_TIMESTAMP
        """

        values = [
            (
                d.game_id,
                d.season,
                d.week,
                d.home_team,
                d.away_team,
                d.actual_spread,
                d.actual_total,
                d.pred_home_spread,
                d.pred_total,
                d.pred_home_win_prob,
                d.vegas_spread,
                d.vegas_total,
                d.model_spread_error,
                d.model_total_error,
                d.vegas_spread_error,
                d.vegas_total_error,
                d.model_beat_vegas_spread,
                d.model_beat_vegas_total,
                d.model_correct_side,
                d.vegas_correct_side,
                d.error_magnitude,
                d.game_context,
                d.model_version,
            )
            for d in decompositions
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, insert_query, values)
                inserted = cur.rowcount
            conn.commit()

        logger.info(f"Saved {inserted} error decompositions")
        return inserted

    def get_error_by_context(self, season: int) -> pd.DataFrame:
        """
        Break down model MAE by game context (close/moderate/blowout).

        Returns DataFrame with error stats grouped by game_context.
        """
        query = """
        SELECT
            game_context,
            COUNT(*) as games,
            AVG(ABS(model_spread_error)) as model_mae,
            AVG(ABS(vegas_spread_error)) as vegas_mae,
            AVG(CASE WHEN model_beat_vegas_spread THEN 1 ELSE 0 END) as beat_vegas_pct
        FROM ml.model_error_decomposition
        WHERE season = %s
        GROUP BY game_context
        ORDER BY games DESC
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(season,))

    def get_error_by_magnitude(self, season: int) -> pd.DataFrame:
        """
        Distribution of error magnitudes across a season.

        Returns DataFrame with counts and percentages by error_magnitude bucket.
        """
        query = """
        SELECT
            error_magnitude,
            COUNT(*) as games,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct,
            AVG(model_spread_error) as avg_signed_error,
            AVG(ABS(model_spread_error)) as avg_abs_error
        FROM ml.model_error_decomposition
        WHERE season = %s
        GROUP BY error_magnitude
        ORDER BY avg_abs_error
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(season,))

    def get_weekly_trend(self, season: int) -> pd.DataFrame:
        """
        Week-by-week model vs Vegas MAE trend.

        Returns DataFrame with weekly MAE for model and Vegas.
        """
        query = """
        SELECT
            week,
            COUNT(*) as games,
            AVG(ABS(model_spread_error)) as model_spread_mae,
            AVG(ABS(model_total_error)) as model_total_mae,
            AVG(ABS(vegas_spread_error)) as vegas_spread_mae,
            AVG(ABS(vegas_total_error)) as vegas_total_mae,
            AVG(CASE WHEN model_beat_vegas_spread THEN 1 ELSE 0 END) as beat_vegas_spread_pct,
            AVG(CASE WHEN model_correct_side THEN 1 ELSE 0 END) as model_side_accuracy
        FROM ml.model_error_decomposition
        WHERE season = %s
        GROUP BY week
        ORDER BY week
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(season,))


def test_decomposition():
    """Test error decomposition against ml.game_outcome_predictions."""
    print("\n" + "=" * 60)
    print("MODEL ERROR DECOMPOSITION TEST")
    print("=" * 60)

    decomposer = ModelErrorDecomposer()

    # Test on a specific game
    test_game = "2025_16_KC_PIT"
    print(f"\nDecomposing error for game: {test_game}")
    result = decomposer.decompose_game_error(test_game)

    if result:
        print(f"\n  {result.away_team} @ {result.home_team} (Week {result.week})")
        print(f"  Score: {result.actual_away_score:.0f}-{result.actual_home_score:.0f}")
        print(f"  Actual spread (home): {result.actual_spread:+.1f}")
        print(
            f"  Model prediction:     {result.pred_home_spread:+.1f}  (error: {result.model_spread_error:+.1f})"
        )
        if result.vegas_spread is not None:
            print(
                f"  Vegas spread:         {result.vegas_spread:+.1f}  (error: {result.vegas_spread_error:+.1f})"
            )
            beat = "YES" if result.model_beat_vegas_spread else "NO"
            print(f"  Model beat Vegas:     {beat}")
        print(f"  Model correct side:   {result.model_correct_side}")
        print(f"  Error magnitude:      {result.error_magnitude}")
        print(f"  Game context:         {result.game_context}")
    else:
        print("  No data found for this game.")

    # Season-level analysis
    print("\n" + "-" * 60)
    print("SEASON 2025 ERROR ANALYSIS")
    print("-" * 60)

    report = decomposer.decompose_season(season=2025, save_to_db=False)

    if report.total_games == 0:
        print("  No games found for 2025 season.")
        return

    print(f"\n  Total games analyzed: {report.total_games}")
    print(f"\n  Model Performance:")
    print(f"    Spread MAE:  {report.model_spread_mae:.2f}")
    print(f"    Spread RMSE: {report.model_spread_rmse:.2f}")
    print(f"    Spread Bias: {report.model_spread_bias:+.2f}")
    print(f"    Total MAE:   {report.model_total_mae:.2f}")
    print(f"    Side Accuracy: {report.model_correct_side_pct:.1%}")

    if report.games_with_vegas > 0:
        print(f"\n  Vegas Comparison ({report.games_with_vegas} games):")
        if report.vegas_spread_mae is not None:
            print(f"    Vegas Spread MAE:       {report.vegas_spread_mae:.2f}")
        if report.model_beat_vegas_spread_pct is not None:
            print(f"    Model Beat Vegas (spread): {report.model_beat_vegas_spread_pct:.1%}")
        if report.vegas_correct_side_pct is not None:
            print(f"    Vegas Side Accuracy:    {report.vegas_correct_side_pct:.1%}")

    print(f"\n  Error Distribution:")
    for mag, count in sorted(report.error_magnitude_counts.items()):
        pct = count / report.total_games * 100
        print(f"    {mag:>8s}: {count:3d} ({pct:.1f}%)")

    if report.worst_games:
        print(f"\n  Top 5 Worst Misses:")
        for g in report.worst_games:
            vegas_str = ""
            if g.get("vegas_spread") is not None:
                vegas_str = f"  [Vegas: {g['vegas_spread']:+.1f}, err: {g['vegas_error']:+.1f}]"
            print(
                f"    Wk{g['week']:>2d} {g['matchup']:>15s}: "
                f"pred {g['pred_spread']:+.1f}, actual {g['actual_spread']:+.1f}, "
                f"error {g['error']:+.1f}{vegas_str}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_decomposition()
