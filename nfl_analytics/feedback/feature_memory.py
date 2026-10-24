"""
Feature Performance Memory.

Rolling memory of how each feature contributes to prediction accuracy.
Stores per-feature metrics and provides feature ranking.

Usage:
    memory = FeaturePerformanceMemory()
    memory.record_game(game_id, features, prediction_error, correct)
    stats = memory.get_feature_stats('sem_injury_advantage', weeks=8)
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for a single feature."""

    feature_name: str
    mean_contribution: float
    contribution_when_correct: float
    contribution_when_incorrect: float
    correlation_with_error: float
    stability: float  # 1.0 = very stable, 0.0 = highly variable
    sample_size: int
    recommendation: str  # 'keep', 'drop', 'investigate', 'boost'


class FeaturePerformanceMemory:
    """
    Rolling memory of how each feature contributes to accuracy.

    Stores per-feature metrics:
    - Contribution when correct vs incorrect predictions
    - Correlation with prediction error
    - Stability over time (variance of contribution)
    """

    # Thresholds for recommendations
    MIN_SAMPLES = 20
    DROP_THRESHOLD = -0.03  # Mean contribution below this = drop
    BOOST_THRESHOLD = 0.03  # Mean contribution above this = boost
    STABILITY_THRESHOLD = 0.5  # Stability below this = investigate

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

    def record_game(
        self,
        game_id: str,
        season: int,
        week: int,
        features: dict[str, float],
        prediction_error: float,
        correct: bool,
        contributions: dict[str, float] | None = None,
    ) -> None:
        """
        Record feature values and outcome for one game.

        Args:
            game_id: Game identifier
            season: Season year
            week: Week number
            features: Dict of feature_name → value
            prediction_error: Absolute prediction error
            correct: Whether prediction was correct
            contributions: Optional dict of feature contributions (SHAP-like)
        """
        query = """
        INSERT INTO ml.feature_performance
            (game_id, season, week, feature_name, feature_value,
             prediction_error, correct_prediction, marginal_contribution)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (game_id, feature_name) DO UPDATE SET
            feature_value = EXCLUDED.feature_value,
            prediction_error = EXCLUDED.prediction_error,
            correct_prediction = EXCLUDED.correct_prediction,
            marginal_contribution = EXCLUDED.marginal_contribution
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for name, value in features.items():
                    contribution = contributions.get(name, 0.0) if contributions else 0.0
                    cur.execute(
                        query,
                        (
                            game_id,
                            season,
                            week,
                            name,
                            value,
                            prediction_error,
                            correct,
                            contribution,
                        ),
                    )
            conn.commit()

        logger.debug(f"Recorded {len(features)} features for game {game_id}")

    def record_batch(self, records: list[dict]) -> int:
        """
        Record multiple games efficiently.

        Args:
            records: List of dicts with keys:
                game_id, season, week, features, prediction_error, correct, contributions

        Returns:
            Number of records inserted
        """
        query = """
        INSERT INTO ml.feature_performance
            (game_id, season, week, feature_name, feature_value,
             prediction_error, correct_prediction, marginal_contribution)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (game_id, feature_name) DO UPDATE SET
            feature_value = EXCLUDED.feature_value,
            prediction_error = EXCLUDED.prediction_error,
            correct_prediction = EXCLUDED.correct_prediction,
            marginal_contribution = EXCLUDED.marginal_contribution
        """

        count = 0
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    features = rec["features"]
                    contributions = rec.get("contributions", {})

                    for name, value in features.items():
                        contribution = contributions.get(name, 0.0)
                        cur.execute(
                            query,
                            (
                                rec["game_id"],
                                rec["season"],
                                rec["week"],
                                name,
                                value,
                                rec["prediction_error"],
                                rec["correct"],
                                contribution,
                            ),
                        )
                        count += 1
            conn.commit()

        logger.info(f"Recorded {count} feature records from {len(records)} games")
        return count

    def get_feature_stats(
        self, feature_name: str, weeks: int = 8, season: int | None = None
    ) -> FeatureStats:
        """
        Get statistics for a specific feature.

        Args:
            feature_name: Name of the feature
            weeks: Lookback window in weeks
            season: Optional season filter

        Returns:
            FeatureStats with contribution metrics
        """
        query = """
        SELECT
            feature_value,
            prediction_error,
            correct_prediction,
            marginal_contribution
        FROM ml.feature_performance
        WHERE feature_name = %s
          AND created_at >= NOW() - MAKE_INTERVAL(weeks => %s)
        """

        weeks_int = int(weeks)
        if weeks_int <= 0 or weeks_int > 52:
            raise ValueError(f"weeks must be between 1 and 52, got {weeks_int}")
        params = [feature_name, weeks_int]

        if season:
            query += " AND season = %s"
            params.append(season)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        if len(rows) < self.MIN_SAMPLES:
            return FeatureStats(
                feature_name=feature_name,
                mean_contribution=0.0,
                contribution_when_correct=0.0,
                contribution_when_incorrect=0.0,
                correlation_with_error=0.0,
                stability=1.0,
                sample_size=len(rows),
                recommendation="insufficient_data",
            )

        # Compute metrics
        values = np.array([r["feature_value"] for r in rows])
        errors = np.array([r["prediction_error"] for r in rows])
        correct = np.array([r["correct_prediction"] for r in rows])
        contribs = np.array([r["marginal_contribution"] or 0.0 for r in rows])

        mean_contrib = np.mean(contribs)
        contrib_correct = np.mean(contribs[correct]) if correct.sum() > 0 else 0.0
        contrib_incorrect = np.mean(contribs[~correct]) if (~correct).sum() > 0 else 0.0

        # Correlation with error (negative = feature helps reduce error)
        # Check variance before computing correlation to avoid NaN
        if len(values) > 1 and np.var(values) > 0 and np.var(errors) > 0:
            corr = np.corrcoef(values, errors)[0, 1]
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.0

        # Stability (1 - normalized std of contributions)
        std_contrib = np.std(contribs)
        stability = 1.0 / (1.0 + std_contrib) if std_contrib > 0 else 1.0

        # Determine recommendation
        recommendation = self._compute_recommendation(
            mean_contrib, corr, stability, contrib_correct, contrib_incorrect
        )

        return FeatureStats(
            feature_name=feature_name,
            mean_contribution=float(mean_contrib),
            contribution_when_correct=float(contrib_correct),
            contribution_when_incorrect=float(contrib_incorrect),
            correlation_with_error=float(corr),
            stability=float(stability),
            sample_size=len(rows),
            recommendation=recommendation,
        )

    def _compute_recommendation(
        self,
        mean_contrib: float,
        corr: float,
        stability: float,
        contrib_correct: float,
        contrib_incorrect: float,
    ) -> str:
        """Compute recommendation based on metrics."""
        # Negative contribution = bad feature
        if mean_contrib < self.DROP_THRESHOLD:
            return "drop"

        # Positive contribution = good feature
        if mean_contrib > self.BOOST_THRESHOLD:
            return "boost"

        # Low stability = investigate
        if stability < self.STABILITY_THRESHOLD:
            return "investigate"

        # High correlation with error = negative signal
        if corr > 0.2:  # Higher feature values correlate with higher errors
            return "investigate"

        # Different behavior correct vs incorrect
        if abs(contrib_correct - contrib_incorrect) > 0.05:
            return "investigate"

        return "keep"

    def get_feature_rankings(
        self, weeks: int = 8, season: int | None = None, top_n: int | None = None
    ) -> pd.DataFrame:
        """
        Rank all features by contribution to accuracy.

        Args:
            weeks: Lookback window
            season: Optional season filter
            top_n: Limit to top N features (by absolute contribution)

        Returns:
            DataFrame with feature rankings
        """
        query = """
        SELECT
            feature_name,
            COUNT(*) as sample_size,
            AVG(marginal_contribution) as mean_contribution,
            STDDEV(marginal_contribution) as std_contribution,
            AVG(CASE WHEN correct_prediction THEN marginal_contribution END) as contrib_correct,
            AVG(CASE WHEN NOT correct_prediction THEN marginal_contribution END) as contrib_incorrect,
            CORR(feature_value, prediction_error) as error_correlation
        FROM ml.feature_performance
        WHERE created_at >= NOW() - INTERVAL '%s weeks'
        """

        params = [weeks]

        if season:
            query += " AND season = %s"
            params.append(season)

        query += """
        GROUP BY feature_name
        HAVING COUNT(*) >= %s
        ORDER BY ABS(AVG(marginal_contribution)) DESC
        """
        params.append(self.MIN_SAMPLES)

        if top_n:
            query += " LIMIT %s"
            params.append(top_n)

        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        # Compute stability and recommendations
        df["stability"] = 1.0 / (1.0 + df["std_contribution"].fillna(0))
        df["recommendation"] = df.apply(
            lambda row: self._compute_recommendation(
                row["mean_contribution"] or 0,
                row["error_correlation"] or 0,
                row["stability"] or 1.0,
                row["contrib_correct"] or 0,
                row["contrib_incorrect"] or 0,
            ),
            axis=1,
        )

        return df

    def get_features_to_drop(self, weeks: int = 8) -> list[str]:
        """Get features recommended for dropping."""
        rankings = self.get_feature_rankings(weeks)
        if rankings.empty:
            return []
        return rankings[rankings["recommendation"] == "drop"]["feature_name"].tolist()

    def get_features_to_boost(self, weeks: int = 8) -> list[str]:
        """Get features recommended for boosting."""
        rankings = self.get_feature_rankings(weeks)
        if rankings.empty:
            return []
        return rankings[rankings["recommendation"] == "boost"]["feature_name"].tolist()

    def get_weekly_summary(self, season: int, week: int) -> dict:
        """
        Get summary of feature performance for a specific week.

        Returns dict with:
        - total_games: Number of games analyzed
        - top_features: Best performing features
        - weak_features: Worst performing features
        - recommendations: Dict of feature → recommendation
        """
        query = """
        SELECT
            feature_name,
            COUNT(DISTINCT game_id) as games,
            AVG(marginal_contribution) as mean_contribution,
            SUM(CASE WHEN correct_prediction THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as accuracy
        FROM ml.feature_performance
        WHERE season = %s AND week = %s
        GROUP BY feature_name
        ORDER BY AVG(marginal_contribution) DESC
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, week))
                rows = cur.fetchall()

        if not rows:
            return {
                "total_games": 0,
                "top_features": [],
                "weak_features": [],
                "recommendations": {},
            }

        total_games = rows[0]["games"] if rows else 0

        top = [
            {"name": r["feature_name"], "contribution": r["mean_contribution"]}
            for r in rows[:5]
            if r["mean_contribution"] and r["mean_contribution"] > 0
        ]

        weak = [
            {"name": r["feature_name"], "contribution": r["mean_contribution"]}
            for r in reversed(rows[-5:])
            if r["mean_contribution"] and r["mean_contribution"] < 0
        ]

        recommendations = {
            r["feature_name"]: (
                "boost"
                if (r["mean_contribution"] or 0) > 0.03
                else "drop" if (r["mean_contribution"] or 0) < -0.03 else "keep"
            )
            for r in rows
        }

        return {
            "total_games": total_games,
            "top_features": top,
            "weak_features": weak,
            "recommendations": recommendations,
        }

    def export_to_dataframe(
        self, season: int | None = None, weeks: int | None = None
    ) -> pd.DataFrame:
        """Export all feature performance data to DataFrame."""
        query = """
        SELECT
            game_id, season, week, feature_name, feature_value,
            prediction_error, correct_prediction, marginal_contribution,
            created_at
        FROM ml.feature_performance
        WHERE 1=1
        """

        params = []
        if season:
            query += " AND season = %s"
            params.append(season)

        if weeks:
            query += " AND created_at >= NOW() - MAKE_INTERVAL(weeks => %s)"
            params.append(int(weeks))

        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            return pd.read_sql(query, conn, params=params if params else None)


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    memory = FeaturePerformanceMemory()

    print("\n=== Testing Feature Performance Memory ===")

    # Test recording a game
    test_features = {
        "sem_home_sentiment": 0.5,
        "sem_injury_advantage": 0.3,
        "sem_media_buzz": 0.7,
    }
    test_contributions = {
        "sem_home_sentiment": 0.02,
        "sem_injury_advantage": 0.01,
        "sem_media_buzz": -0.01,
    }

    # Record test game
    memory.record_game(
        game_id="test_game_001",
        season=2025,
        week=16,
        features=test_features,
        prediction_error=3.5,
        correct=True,
        contributions=test_contributions,
    )
    print("Recorded test game")

    # Get rankings
    rankings = memory.get_feature_rankings(weeks=52)  # Look back 1 year
    print(f"\nFeature rankings ({len(rankings)} features):")
    if not rankings.empty:
        print(rankings.head())

    # Get weekly summary
    summary = memory.get_weekly_summary(2025, 16)
    print(f"\nWeek 16 summary:")
    print(f"  Total games: {summary['total_games']}")
    print(f"  Top features: {summary['top_features']}")
    print(f"  Weak features: {summary['weak_features']}")
