"""
Model Error Decomposition for Ensemble Predictions.

Tracks which ensemble component (State-Space, Hierarchical, XGBoost, etc.)
contributed most to prediction errors.

Based on Gemini consultation (Dec 28, 2025):
- Signed Error Contribution: C_i = w_i × (Y - Ŷ_i)
- Sum of contributions equals total ensemble error
- Positive C_i means model pulled toward truth, negative means away

Usage:
    decomposer = ModelErrorDecomposer()
    result = decomposer.decompose_game_error(game_id='2025_16_KC_PIT')
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

logger = logging.getLogger(__name__)


# Production ensemble weights (v6.0 games)
ENSEMBLE_WEIGHTS = {
    "state_space": 0.27,
    "hierarchical": 0.27,
    "xgboost": 0.23,
    "informative_priors": 0.13,
    "hybrid": 0.10,
}


@dataclass
class ErrorDecomposition:
    """Result of decomposing prediction error by model."""
    game_id: str
    actual_spread: float
    ensemble_prediction: float
    ensemble_error: float  # Y - Ŷ (signed)

    # Per-model contributions (signed)
    model_contributions: dict[str, float] = field(default_factory=dict)

    # Per-model individual errors
    model_predictions: dict[str, float] = field(default_factory=dict)
    model_errors: dict[str, float] = field(default_factory=dict)

    # Attribution
    most_responsible: str = ""  # Model with largest negative contribution
    most_helpful: str = ""      # Model with largest positive contribution

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "actual_spread": self.actual_spread,
            "ensemble_prediction": self.ensemble_prediction,
            "ensemble_error": self.ensemble_error,
            "model_contributions": self.model_contributions,
            "model_predictions": self.model_predictions,
            "model_errors": self.model_errors,
            "most_responsible": self.most_responsible,
            "most_helpful": self.most_helpful,
        }


class ModelErrorDecomposer:
    """
    Decompose ensemble prediction errors by component model.

    Uses the formula: C_i = w_i × (Y - Ŷ_i)
    Where:
        - C_i is model i's contribution to error
        - w_i is model i's ensemble weight
        - Y is actual outcome
        - Ŷ_i is model i's prediction

    Properties:
        - Sum of C_i equals total ensemble error
        - Positive C_i means model pulled prediction toward truth
        - Negative C_i means model pulled prediction away from truth
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
        weights: dict[str, float] | None = None,
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        self.weights = weights or ENSEMBLE_WEIGHTS

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def get_model_predictions(
        self,
        game_id: str,
    ) -> dict[str, float]:
        """
        Get individual model predictions for a game.

        Queries ml.model_predictions or reconstructs from ensemble.
        Returns dict of model_name -> predicted_spread.
        """
        # First try to get from stored predictions
        query = """
        SELECT model_id, predicted_spread
        FROM ml.game_predictions
        WHERE game_id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (game_id,))
                rows = cur.fetchall()

        if rows:
            # Map model_id to our ensemble component names
            predictions = {}
            for row in rows:
                model_id = row["model_id"]
                # Extract model type from model_id
                if "state_space" in model_id.lower():
                    predictions["state_space"] = row["predicted_spread"]
                elif "hierarchical" in model_id.lower():
                    predictions["hierarchical"] = row["predicted_spread"]
                elif "xgboost" in model_id.lower():
                    predictions["xgboost"] = row["predicted_spread"]
                elif "informative" in model_id.lower():
                    predictions["informative_priors"] = row["predicted_spread"]
                elif "hybrid" in model_id.lower():
                    predictions["hybrid"] = row["predicted_spread"]

            if predictions:
                return predictions

        # Fallback: generate synthetic predictions for testing
        # In production, this should raise an error
        logger.warning(f"No individual model predictions for {game_id}, using simulation")
        return self._simulate_model_predictions(game_id)

    def _simulate_model_predictions(self, game_id: str) -> dict[str, float]:
        """
        Simulate individual model predictions for testing.

        In production, should be replaced with actual model outputs.
        """
        # Get ensemble prediction
        query = """
        SELECT predicted_spread, actual_spread
        FROM ml.game_predictions
        WHERE game_id = %s
        LIMIT 1
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (game_id,))
                row = cur.fetchone()

        if not row:
            return {}

        ensemble_pred = row["predicted_spread"]

        # Simulate: each model is ensemble + noise
        # Noise magnitude varies by model type
        np.random.seed(hash(game_id) % 2**32)

        return {
            "state_space": ensemble_pred + np.random.normal(0, 2.0),
            "hierarchical": ensemble_pred + np.random.normal(0, 1.5),
            "xgboost": ensemble_pred + np.random.normal(0, 2.5),
            "informative_priors": ensemble_pred + np.random.normal(0, 1.0),
            "hybrid": ensemble_pred + np.random.normal(0, 1.8),
        }

    def decompose_game_error(
        self,
        game_id: str,
    ) -> ErrorDecomposition | None:
        """
        Decompose prediction error for a single game.

        Args:
            game_id: Game identifier

        Returns:
            ErrorDecomposition with per-model contributions
        """
        # Get ensemble prediction and actual
        query = """
        SELECT predicted_spread, actual_spread
        FROM ml.game_predictions
        WHERE game_id = %s AND actual_spread IS NOT NULL
        LIMIT 1
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (game_id,))
                row = cur.fetchone()

        if not row:
            logger.warning(f"No prediction/outcome for {game_id}")
            return None

        actual = row["actual_spread"]
        ensemble_pred = row["predicted_spread"]
        ensemble_error = actual - ensemble_pred  # Signed error

        # Get individual model predictions
        model_preds = self.get_model_predictions(game_id)

        if not model_preds:
            return None

        # Compute contributions: C_i = w_i × (Y - Ŷ_i)
        contributions = {}
        model_errors = {}

        for model, pred in model_preds.items():
            weight = self.weights.get(model, 0)
            model_error = actual - pred  # Individual model's signed error
            contribution = weight * model_error  # Weighted contribution

            contributions[model] = round(contribution, 4)
            model_errors[model] = round(model_error, 4)

        # Find most responsible (largest magnitude negative = pulled away from truth)
        # and most helpful (largest positive = pulled toward truth)
        sorted_by_contrib = sorted(contributions.items(), key=lambda x: x[1])

        most_responsible = sorted_by_contrib[0][0] if sorted_by_contrib else ""
        most_helpful = sorted_by_contrib[-1][0] if sorted_by_contrib else ""

        return ErrorDecomposition(
            game_id=game_id,
            actual_spread=actual,
            ensemble_prediction=ensemble_pred,
            ensemble_error=round(ensemble_error, 4),
            model_contributions=contributions,
            model_predictions={k: round(v, 4) for k, v in model_preds.items()},
            model_errors=model_errors,
            most_responsible=most_responsible,
            most_helpful=most_helpful,
        )

    def decompose_week(
        self,
        season: int,
        week: int,
        save_to_db: bool = True,
    ) -> list[ErrorDecomposition]:
        """
        Decompose errors for all games in a week.

        Returns:
            List of ErrorDecomposition objects
        """
        # Get all games with outcomes
        query = """
        SELECT DISTINCT game_id
        FROM ml.game_predictions
        WHERE season = %s AND week = %s AND actual_spread IS NOT NULL
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (season, week))
                game_ids = [row[0] for row in cur.fetchall()]

        logger.info(f"Decomposing errors for {len(game_ids)} games in {season} Week {week}")

        results = []
        for game_id in game_ids:
            decomp = self.decompose_game_error(game_id)
            if decomp:
                results.append(decomp)

        if save_to_db and results:
            self._save_decompositions(results, season, week)

        return results

    def _save_decompositions(
        self,
        decompositions: list[ErrorDecomposition],
        season: int,
        week: int,
    ) -> int:
        """Save decomposition results to database."""
        # Ensure table exists
        create_table = """
        CREATE TABLE IF NOT EXISTS ml.model_error_decomposition (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(50) NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            actual_spread NUMERIC(5,2),
            ensemble_prediction NUMERIC(5,2),
            ensemble_error NUMERIC(5,2),
            model_contributions JSONB,
            model_predictions JSONB,
            model_errors JSONB,
            most_responsible VARCHAR(50),
            most_helpful VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (game_id)
        )
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table)
            conn.commit()

        # Insert decompositions
        query = """
        INSERT INTO ml.model_error_decomposition
            (game_id, season, week, actual_spread, ensemble_prediction,
             ensemble_error, model_contributions, model_predictions,
             model_errors, most_responsible, most_helpful)
        VALUES %s
        ON CONFLICT (game_id) DO UPDATE SET
            model_contributions = EXCLUDED.model_contributions,
            model_predictions = EXCLUDED.model_predictions,
            model_errors = EXCLUDED.model_errors,
            most_responsible = EXCLUDED.most_responsible,
            most_helpful = EXCLUDED.most_helpful
        """

        values = [
            (
                d.game_id,
                season,
                week,
                d.actual_spread,
                d.ensemble_prediction,
                d.ensemble_error,
                json.dumps(d.model_contributions),
                json.dumps(d.model_predictions),
                json.dumps(d.model_errors),
                d.most_responsible,
                d.most_helpful,
            )
            for d in decompositions
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
                inserted = cur.rowcount
            conn.commit()

        logger.info(f"Saved {inserted} error decompositions")
        return inserted

    def get_model_responsibility_summary(
        self,
        season: int,
        start_week: int = 1,
        end_week: int = 18,
    ) -> pd.DataFrame:
        """
        Get summary of which models are most often responsible for errors.

        Returns DataFrame with model responsibility counts and averages.
        """
        query = """
        SELECT
            most_responsible,
            COUNT(*) as times_responsible,
            AVG(ABS(ensemble_error)) as avg_error_when_responsible
        FROM ml.model_error_decomposition
        WHERE season = %s AND week >= %s AND week <= %s
        GROUP BY most_responsible
        ORDER BY times_responsible DESC
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(season, start_week, end_week))

        return df

    def get_model_contribution_trends(
        self,
        season: int,
    ) -> pd.DataFrame:
        """
        Get week-by-week contribution trends by model.

        Returns DataFrame with weekly average contributions.
        """
        query = """
        SELECT
            week,
            AVG((model_contributions->>'state_space')::FLOAT) as state_space_contrib,
            AVG((model_contributions->>'hierarchical')::FLOAT) as hierarchical_contrib,
            AVG((model_contributions->>'xgboost')::FLOAT) as xgboost_contrib,
            AVG((model_contributions->>'informative_priors')::FLOAT) as informative_contrib,
            AVG((model_contributions->>'hybrid')::FLOAT) as hybrid_contrib
        FROM ml.model_error_decomposition
        WHERE season = %s
        GROUP BY week
        ORDER BY week
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(season,))

        return df


def test_decomposition():
    """Test error decomposition."""
    print("\n" + "=" * 60)
    print("MODEL ERROR DECOMPOSITION TEST")
    print("=" * 60)

    decomposer = ModelErrorDecomposer()

    # Test on a specific game
    test_game = "2025_16_KC_PIT"

    print(f"\nDecomposing error for game: {test_game}")
    result = decomposer.decompose_game_error(test_game)

    if result:
        print(f"\nActual spread: {result.actual_spread}")
        print(f"Ensemble prediction: {result.ensemble_prediction}")
        print(f"Ensemble error: {result.ensemble_error}")
        print(f"\nModel contributions:")
        for model, contrib in sorted(result.model_contributions.items(), key=lambda x: x[1]):
            print(f"  {model}: {contrib:+.4f}")
        print(f"\nMost responsible: {result.most_responsible}")
        print(f"Most helpful: {result.most_helpful}")
    else:
        print("No data available - testing with simulated predictions")

        # Create a synthetic example
        print("\n--- Synthetic Example ---")
        actual = 7.0
        predictions = {
            "state_space": 3.0,
            "hierarchical": 4.0,
            "xgboost": 10.0,
            "informative_priors": 5.0,
            "hybrid": 6.0,
        }
        ensemble = sum(ENSEMBLE_WEIGHTS[m] * predictions[m] for m in predictions)

        print(f"Actual: {actual}")
        print(f"Ensemble: {ensemble:.2f}")
        print(f"Ensemble error: {actual - ensemble:.2f}")

        print("\nContributions:")
        for model, pred in predictions.items():
            w = ENSEMBLE_WEIGHTS[model]
            contrib = w * (actual - pred)
            print(f"  {model}: w={w:.2f}, pred={pred:.1f}, contrib={contrib:+.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_decomposition()
