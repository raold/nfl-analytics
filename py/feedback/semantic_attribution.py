"""
Semantic Feature Attribution Tracker.

Tracks which semantic features contributed to correct/incorrect predictions
using SHAP-like attribution analysis.

Usage:
    tracker = SemanticAttributionTracker()
    df = tracker.compute_week_attribution(season=2025, week=13)
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Result of attribution analysis."""
    season: int
    week: int
    games_analyzed: int = 0
    games_with_semantic: int = 0
    attributions_saved: int = 0
    top_features: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class SemanticAttributionTracker:
    """
    Analyze semantic feature contribution to prediction accuracy.

    Uses simplified attribution (correlation with prediction error) when
    SHAP is not available or model not accessible.

    For each game:
    1. Get predictions with semantic features
    2. Compare to actual outcome
    3. Compute feature attribution scores
    4. Store in ml.semantic_attribution
    """

    # Semantic feature prefixes
    SEMANTIC_PREFIX = "sem_"

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

    def get_predictions_with_outcomes(
        self,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Get predictions that have actual outcomes.

        Returns DataFrame with prediction details and outcomes.
        """
        query = """
        SELECT
            id,
            game_id,
            season,
            week,
            home_team,
            away_team,
            model_id,
            predicted_spread,
            predicted_total,
            actual_spread,
            actual_total,
            abs_spread_error,
            CASE
                WHEN (predicted_spread > 0) = (actual_spread > 0) THEN TRUE
                ELSE FALSE
            END AS spread_correct
        FROM ml.game_predictions
        WHERE season = %s AND week = %s
          AND actual_spread IS NOT NULL
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(season, week))

        return df

    def extract_semantic_features_for_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        as_of: datetime | None = None
    ) -> dict[str, float]:
        """
        Extract semantic features for a specific game.

        Uses SemanticFeatureExtractor to get the 26 semantic features.
        """
        try:
            from py.semantic.features.semantic_features import SemanticFeatureExtractor

            extractor = SemanticFeatureExtractor()
            features = extractor.extract_features(
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                as_of=as_of
            )

            # Add prefix for consistency
            return {f"{self.SEMANTIC_PREFIX}{k}": v for k, v in features.items()}

        except Exception as e:
            logger.warning(f"Failed to extract semantic features: {e}")
            return {}

    def compute_game_attribution(
        self,
        semantic_features: dict[str, float],
        prediction_error: float,
        spread_correct: bool
    ) -> dict[str, float]:
        """
        Compute attribution scores for semantic features for a single game.

        Uses a simplified attribution based on:
        - Feature magnitude (larger values → stronger signal)
        - Direction alignment with prediction accuracy

        For true SHAP, you'd need the model and background data.
        This is a heuristic that works well for ranking features.

        Args:
            semantic_features: Dict of feature name → value
            prediction_error: Absolute spread error
            spread_correct: Whether spread prediction was correct

        Returns:
            Dict of feature name → attribution score
        """
        if not semantic_features:
            return {}

        attributions = {}

        # Baseline: average feature magnitude
        magnitudes = [abs(v) for v in semantic_features.values() if v != 0]
        avg_magnitude = np.mean(magnitudes) if magnitudes else 1.0

        for name, value in semantic_features.items():
            if value == 0:
                attributions[name] = 0.0
                continue

            # Normalized magnitude
            norm_magnitude = abs(value) / (avg_magnitude + 1e-6)

            # Attribution: positive if feature helped, negative if hurt
            # Approximation: features with high magnitude that correlated
            # with correct predictions get positive attribution
            if spread_correct:
                # Correct prediction: feature helped
                attributions[name] = norm_magnitude * 0.1  # Scale down
            else:
                # Wrong prediction: feature may have hurt
                # Higher error + high magnitude = negative attribution
                error_weight = min(prediction_error / 10.0, 1.0)
                attributions[name] = -norm_magnitude * error_weight * 0.1

        return attributions

    def compute_week_attribution(
        self,
        season: int,
        week: int,
        save_to_db: bool = True
    ) -> AttributionResult:
        """
        Compute attribution for all games in a week.

        Args:
            season: Season year
            week: Week number
            save_to_db: Whether to save results to ml.semantic_attribution

        Returns:
            AttributionResult with statistics and top features
        """
        result = AttributionResult(season=season, week=week)

        # Get predictions with outcomes
        predictions = self.get_predictions_with_outcomes(season, week)
        result.games_analyzed = len(predictions)

        if predictions.empty:
            logger.info(f"No predictions with outcomes for {season} Week {week}")
            return result

        logger.info(f"Analyzing {len(predictions)} predictions for {season} Week {week}")

        # Collect all attributions
        all_attributions = []
        feature_totals = {}

        for _, row in predictions.iterrows():
            try:
                # Extract semantic features
                sem_features = self.extract_semantic_features_for_game(
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    season=season,
                    week=week
                )

                if not sem_features:
                    continue

                result.games_with_semantic += 1

                # Compute attribution
                attributions = self.compute_game_attribution(
                    semantic_features=sem_features,
                    prediction_error=row["abs_spread_error"],
                    spread_correct=row["spread_correct"]
                )

                # Track totals for ranking
                for name, value in attributions.items():
                    feature_totals[name] = feature_totals.get(name, [])
                    feature_totals[name].append(value)

                # Prepare for database
                all_attributions.append({
                    "game_id": row["game_id"],
                    "season": season,
                    "week": week,
                    "prediction_correct": row["spread_correct"],
                    "prediction_error": row["abs_spread_error"],
                    "predicted_spread": row["predicted_spread"],
                    "actual_spread": row["actual_spread"],
                    "semantic_features": sem_features,
                    "feature_attributions": attributions,
                    "total_semantic_contribution": sum(attributions.values()),
                })

            except Exception as e:
                result.errors.append(f"Error for {row['game_id']}: {e}")
                logger.warning(f"Attribution failed for {row['game_id']}: {e}")

        # Compute top features by average attribution
        if feature_totals:
            result.top_features = {
                name: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values)
                }
                for name, values in feature_totals.items()
            }

        # Save to database
        if save_to_db and all_attributions:
            result.attributions_saved = self._save_attributions(all_attributions)

        return result

    def _save_attributions(self, attributions: list[dict]) -> int:
        """Save attribution records to database."""
        query = """
        INSERT INTO ml.semantic_attribution
            (game_id, season, week, prediction_correct, prediction_error,
             predicted_spread, actual_spread, semantic_features,
             feature_attributions, total_semantic_contribution)
        VALUES %s
        ON CONFLICT DO NOTHING
        """

        values = [
            (
                a["game_id"],
                a["season"],
                a["week"],
                a["prediction_correct"],
                a["prediction_error"],
                a["predicted_spread"],
                a["actual_spread"],
                json.dumps(a["semantic_features"]),
                json.dumps(a["feature_attributions"]),
                a["total_semantic_contribution"],
            )
            for a in attributions
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
                inserted = cur.rowcount
            conn.commit()

        logger.info(f"Saved {inserted} attribution records")
        return inserted

    def compute_rolling_attribution(
        self,
        season: int,
        end_week: int,
        lookback_weeks: int = 4
    ) -> pd.DataFrame:
        """
        Compute rolling average attribution by semantic feature.

        Args:
            season: Season year
            end_week: End week (inclusive)
            lookback_weeks: Number of weeks to look back

        Returns:
            DataFrame with feature rankings
        """
        start_week = max(1, end_week - lookback_weeks + 1)

        query = """
        SELECT
            feature_attributions,
            prediction_correct,
            prediction_error
        FROM ml.semantic_attribution
        WHERE season = %s AND week >= %s AND week <= %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, start_week, end_week))
                rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        # Aggregate by feature
        feature_stats = {}
        for row in rows:
            attrs = row["feature_attributions"] if row["feature_attributions"] else {}

            for name, value in attrs.items():
                if name not in feature_stats:
                    feature_stats[name] = {
                        "values": [],
                        "correct_values": [],
                        "incorrect_values": []
                    }

                feature_stats[name]["values"].append(value)
                if row["prediction_correct"]:
                    feature_stats[name]["correct_values"].append(value)
                else:
                    feature_stats[name]["incorrect_values"].append(value)

        # Build DataFrame
        records = []
        for name, stats in feature_stats.items():
            records.append({
                "feature": name,
                "mean_attribution": np.mean(stats["values"]),
                "std_attribution": np.std(stats["values"]),
                "mean_when_correct": np.mean(stats["correct_values"]) if stats["correct_values"] else 0,
                "mean_when_incorrect": np.mean(stats["incorrect_values"]) if stats["incorrect_values"] else 0,
                "games_count": len(stats["values"]),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("mean_attribution", ascending=False)

        return df

    def get_feature_impact_summary(self, season: int) -> dict:
        """
        Get summary of semantic feature impact for a season.

        Returns dict with:
        - total_games: Games with attribution data
        - avg_semantic_contribution: Average total semantic contribution
        - feature_rankings: Top 10 features by impact
        """
        query = """
        SELECT
            COUNT(*) as total_games,
            AVG(total_semantic_contribution) as avg_contribution,
            AVG(prediction_error) as avg_error,
            SUM(CASE WHEN prediction_correct THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as accuracy
        FROM ml.semantic_attribution
        WHERE season = %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season,))
                summary = dict(cur.fetchone())

        # Get feature rankings
        rankings = self.compute_rolling_attribution(season, 18, lookback_weeks=18)

        if not rankings.empty:
            summary["feature_rankings"] = rankings.head(10).to_dict("records")
        else:
            summary["feature_rankings"] = []

        return summary


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tracker = SemanticAttributionTracker()

    print("\n=== Testing Semantic Attribution ===")

    # Check if we have any predictions
    df = tracker.get_predictions_with_outcomes(2025, 13)
    print(f"Predictions with outcomes in Week 13: {len(df)}")

    if len(df) > 0:
        result = tracker.compute_week_attribution(2025, 13, save_to_db=False)
        print(f"\nGames analyzed: {result.games_analyzed}")
        print(f"Games with semantic: {result.games_with_semantic}")

        if result.top_features:
            print("\nTop features by attribution:")
            sorted_features = sorted(
                result.top_features.items(),
                key=lambda x: x[1]["mean"],
                reverse=True
            )
            for name, stats in sorted_features[:5]:
                print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    else:
        print("No predictions with outcomes found - testing feature extraction...")

        # Test feature extraction
        features = tracker.extract_semantic_features_for_game("KC", "BAL", 2025, 16)
        print(f"\nExtracted {len(features)} semantic features")
        for k, v in list(features.items())[:5]:
            print(f"  {k}: {v:.4f}")
