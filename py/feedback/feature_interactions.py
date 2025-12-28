"""
Feature Interaction Analysis using SHAP.

Detects synergies between semantic features (sem_*) and base features.
Identifies which feature combinations provide strongest predictive signal.

Based on Gemini consultation (Dec 28, 2025):
- Use shap.TreeExplainer.shap_interaction_values()
- Subsample to ~2000 samples for tractability
- Complexity is O(M² × N × T) - expensive but necessary

Usage:
    analyzer = FeatureInteractionAnalyzer()
    top_interactions = analyzer.find_top_interactions(n_samples=1000)
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


# Check for SHAP availability
try:
    import shap
    import xgboost as xgb
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP or XGBoost not available - interaction analysis disabled")


@dataclass
class FeatureInteraction:
    """A detected feature interaction."""
    feature1: str          # First feature (typically semantic)
    feature2: str          # Second feature (typically base)
    mean_abs_interaction: float  # Mean absolute SHAP interaction value
    direction: str         # "synergistic" or "antagonistic"
    samples_analyzed: int


@dataclass
class InteractionAnalysisResult:
    """Result of interaction analysis."""
    total_features: int
    semantic_features: int
    base_features: int
    samples_analyzed: int
    compute_time_seconds: float

    # Top interactions
    top_interactions: list[FeatureInteraction] = field(default_factory=list)

    # Aggregated by semantic feature
    semantic_feature_scores: dict[str, float] = field(default_factory=dict)


class FeatureInteractionAnalyzer:
    """
    Analyze feature interactions using SHAP.

    Focus on semantic × base feature interactions to identify
    which semantic features have the strongest synergies.
    """

    PRODUCTION_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "production" / "game_outcome_v2"
    SEMANTIC_PREFIX = "sem_"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

        self._model = None
        self._explainer = None
        self._feature_names = None
        self._loaded = False

        if SHAP_AVAILABLE:
            self._load_model()

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def _load_model(self) -> bool:
        """Load production XGBoost model."""
        try:
            spread_path = self.PRODUCTION_MODEL_DIR / "spread_model.json"
            feature_names_path = self.PRODUCTION_MODEL_DIR / "feature_names.json"

            if not spread_path.exists():
                logger.warning(f"Model not found: {spread_path}")
                return False

            self._model = xgb.XGBRegressor()
            self._model.load_model(str(spread_path))

            if feature_names_path.exists():
                with open(feature_names_path) as f:
                    self._feature_names = json.load(f)

            self._explainer = shap.TreeExplainer(self._model)
            self._loaded = True

            logger.info(f"Loaded model with {len(self._feature_names or [])} features")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_feature_matrix(
        self,
        season: int = 2024,
        n_samples: int = 2000,
    ) -> pd.DataFrame | None:
        """
        Get feature matrix for analysis.

        Samples games from the specified season.
        """
        try:
            from py.features.game_feature_extractor import GameFeatureExtractor

            # Get game list
            query = """
            SELECT game_id, home_team, away_team, season, week
            FROM games
            WHERE season = %s AND home_score IS NOT NULL
            ORDER BY RANDOM()
            LIMIT %s
            """

            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (season, n_samples))
                    games = cur.fetchall()

            if not games:
                logger.warning(f"No games found for season {season}")
                return None

            extractor = GameFeatureExtractor()
            feature_dicts = []

            for game in games:
                try:
                    features = extractor.extract_game_features(
                        home_team=game["home_team"],
                        away_team=game["away_team"],
                        season=game["season"],
                        week=game["week"],
                    )
                    feature_dicts.append(features)
                except Exception as e:
                    logger.debug(f"Failed to extract features for {game['game_id']}: {e}")

            if not feature_dicts:
                return None

            return pd.DataFrame(feature_dicts)

        except Exception as e:
            logger.error(f"Failed to get feature matrix: {e}")
            return None

    def compute_interaction_values(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray | None:
        """
        Compute SHAP interaction values for feature matrix.

        Args:
            X: Feature matrix (N samples × M features)

        Returns:
            Interaction values array (N × M × M)
        """
        if not self._loaded:
            logger.error("Model not loaded")
            return None

        if self._feature_names is None:
            logger.error("Feature names not loaded")
            return None

        # Ensure columns match model expectations
        X_aligned = pd.DataFrame(0.0, index=X.index, columns=self._feature_names)
        for col in X.columns:
            if col in X_aligned.columns:
                X_aligned[col] = X[col].fillna(0)

        logger.info(f"Computing SHAP interaction values for {len(X_aligned)} samples...")
        logger.info(f"This may take a while (~{len(X_aligned) * 0.1:.0f} seconds estimated)")

        try:
            interaction_values = self._explainer.shap_interaction_values(X_aligned)
            logger.info(f"Computed interaction values: shape {interaction_values.shape}")
            return interaction_values
        except Exception as e:
            logger.error(f"Failed to compute interaction values: {e}")
            return None

    def extract_semantic_base_interactions(
        self,
        interaction_values: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureInteraction]:
        """
        Extract interactions between semantic and base features.

        Args:
            interaction_values: SHAP interaction array (N × M × M)
            feature_names: List of feature names

        Returns:
            List of FeatureInteraction objects sorted by strength
        """
        n_samples = interaction_values.shape[0]

        # Identify feature groups
        sem_indices = [i for i, name in enumerate(feature_names) if name.startswith(self.SEMANTIC_PREFIX)]
        base_indices = [i for i, name in enumerate(feature_names) if not name.startswith(self.SEMANTIC_PREFIX)]

        logger.info(f"Analyzing {len(sem_indices)} semantic × {len(base_indices)} base interactions")

        interactions = []

        for sem_idx in sem_indices:
            sem_name = feature_names[sem_idx]

            for base_idx in base_indices:
                base_name = feature_names[base_idx]

                # Extract interaction values for this pair
                # Average both directions for symmetric measure
                int_sem_base = interaction_values[:, sem_idx, base_idx]
                int_base_sem = interaction_values[:, base_idx, sem_idx]

                mean_abs = (np.mean(np.abs(int_sem_base)) + np.mean(np.abs(int_base_sem))) / 2

                # Determine direction
                mean_signed = (np.mean(int_sem_base) + np.mean(int_base_sem)) / 2
                direction = "synergistic" if mean_signed > 0 else "antagonistic"

                if mean_abs > 0.001:  # Filter out negligible interactions
                    interactions.append(FeatureInteraction(
                        feature1=sem_name,
                        feature2=base_name,
                        mean_abs_interaction=float(mean_abs),
                        direction=direction,
                        samples_analyzed=n_samples,
                    ))

        # Sort by interaction strength
        interactions.sort(key=lambda x: x.mean_abs_interaction, reverse=True)

        return interactions

    def find_top_interactions(
        self,
        season: int = 2024,
        n_samples: int = 1000,
        top_n: int = 50,
        save_to_db: bool = True,
    ) -> InteractionAnalysisResult:
        """
        Find top feature interactions.

        Args:
            season: Season for training data
            n_samples: Number of samples for SHAP computation
            top_n: Number of top interactions to return
            save_to_db: Whether to save results to database

        Returns:
            InteractionAnalysisResult with top interactions
        """
        import time

        start_time = time.time()

        result = InteractionAnalysisResult(
            total_features=len(self._feature_names or []),
            semantic_features=0,
            base_features=0,
            samples_analyzed=0,
            compute_time_seconds=0,
        )

        if not SHAP_AVAILABLE or not self._loaded:
            logger.error("SHAP not available or model not loaded")
            return result

        # Get feature matrix
        logger.info(f"Getting feature matrix for {n_samples} samples from season {season}")
        X = self.get_feature_matrix(season=season, n_samples=n_samples)

        if X is None or X.empty:
            logger.error("Failed to get feature matrix")
            return result

        result.samples_analyzed = len(X)

        # Count feature types
        if self._feature_names:
            result.semantic_features = len([f for f in self._feature_names if f.startswith(self.SEMANTIC_PREFIX)])
            result.base_features = len(self._feature_names) - result.semantic_features

        # Compute interaction values
        interaction_values = self.compute_interaction_values(X)

        if interaction_values is None:
            return result

        # Extract semantic × base interactions
        all_interactions = self.extract_semantic_base_interactions(
            interaction_values,
            self._feature_names,
        )

        result.top_interactions = all_interactions[:top_n]

        # Aggregate by semantic feature
        sem_scores = {}
        for interaction in all_interactions:
            sem_name = interaction.feature1
            if sem_name not in sem_scores:
                sem_scores[sem_name] = []
            sem_scores[sem_name].append(interaction.mean_abs_interaction)

        result.semantic_feature_scores = {
            name: float(np.mean(scores))
            for name, scores in sem_scores.items()
        }

        result.compute_time_seconds = time.time() - start_time

        if save_to_db:
            self._save_results(result, season)

        return result

    def _save_results(self, result: InteractionAnalysisResult, season: int):
        """Save interaction analysis results to database."""
        # Create table if needed
        create_table = """
        CREATE TABLE IF NOT EXISTS ml.feature_interactions (
            id SERIAL PRIMARY KEY,
            season INTEGER NOT NULL,
            feature1 VARCHAR(100) NOT NULL,
            feature2 VARCHAR(100) NOT NULL,
            mean_abs_interaction NUMERIC(10,6),
            direction VARCHAR(20),
            samples_analyzed INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (season, feature1, feature2)
        )
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table)
            conn.commit()

        # Insert interactions
        query = """
        INSERT INTO ml.feature_interactions
            (season, feature1, feature2, mean_abs_interaction, direction, samples_analyzed)
        VALUES %s
        ON CONFLICT (season, feature1, feature2) DO UPDATE SET
            mean_abs_interaction = EXCLUDED.mean_abs_interaction,
            direction = EXCLUDED.direction,
            samples_analyzed = EXCLUDED.samples_analyzed
        """

        values = [
            (
                season,
                i.feature1,
                i.feature2,
                i.mean_abs_interaction,
                i.direction,
                i.samples_analyzed,
            )
            for i in result.top_interactions
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
            conn.commit()

        logger.info(f"Saved {len(values)} interactions to database")

    def get_interaction_heatmap_data(
        self,
        season: int,
        top_n_semantic: int = 10,
        top_n_base: int = 20,
    ) -> pd.DataFrame:
        """
        Get data for interaction heatmap visualization.

        Returns pivot table of semantic × base interaction strengths.
        """
        query = """
        WITH ranked_semantic AS (
            SELECT feature1,
                   AVG(mean_abs_interaction) as avg_interaction
            FROM ml.feature_interactions
            WHERE season = %s
            GROUP BY feature1
            ORDER BY avg_interaction DESC
            LIMIT %s
        ),
        ranked_base AS (
            SELECT feature2,
                   AVG(mean_abs_interaction) as avg_interaction
            FROM ml.feature_interactions
            WHERE season = %s
            GROUP BY feature2
            ORDER BY avg_interaction DESC
            LIMIT %s
        )
        SELECT fi.feature1, fi.feature2, fi.mean_abs_interaction
        FROM ml.feature_interactions fi
        JOIN ranked_semantic rs ON fi.feature1 = rs.feature1
        JOIN ranked_base rb ON fi.feature2 = rb.feature2
        WHERE fi.season = %s
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(season, top_n_semantic, season, top_n_base, season)
            )

        if df.empty:
            return df

        # Pivot to heatmap format
        heatmap = df.pivot(index="feature1", columns="feature2", values="mean_abs_interaction")

        return heatmap

    def quick_interaction_check(
        self,
        feature1: str,
        feature2: str,
        n_samples: int = 500,
    ) -> float | None:
        """
        Quick check of interaction between two specific features.

        Returns mean absolute interaction value.
        """
        if not SHAP_AVAILABLE or not self._loaded:
            return None

        X = self.get_feature_matrix(n_samples=n_samples)
        if X is None or X.empty:
            return None

        interaction_values = self.compute_interaction_values(X)
        if interaction_values is None:
            return None

        # Find feature indices
        try:
            idx1 = self._feature_names.index(feature1)
            idx2 = self._feature_names.index(feature2)
        except ValueError:
            logger.error(f"Feature not found: {feature1} or {feature2}")
            return None

        # Extract interaction
        int_12 = interaction_values[:, idx1, idx2]
        int_21 = interaction_values[:, idx2, idx1]

        return float((np.mean(np.abs(int_12)) + np.mean(np.abs(int_21))) / 2)


def test_interactions():
    """Test feature interaction analysis."""
    print("\n" + "=" * 60)
    print("FEATURE INTERACTION ANALYSIS TEST")
    print("=" * 60)

    if not SHAP_AVAILABLE:
        print("SHAP not available - skipping test")
        return

    analyzer = FeatureInteractionAnalyzer()

    if not analyzer._loaded:
        print("Model not loaded - skipping full test")
        print("\nSimulating interaction analysis...")

        # Create synthetic example
        print("\n--- Synthetic Example ---")
        interactions = [
            FeatureInteraction("sem_injury_severity", "days_rest_differential", 0.15, "synergistic", 1000),
            FeatureInteraction("sem_sentiment_24h", "home_epa_pass", 0.12, "synergistic", 1000),
            FeatureInteraction("sem_media_buzz", "primetime_flag", 0.08, "antagonistic", 1000),
        ]

        print("\nTop Interactions (synthetic):")
        for i, inter in enumerate(interactions, 1):
            print(f"  {i}. {inter.feature1} × {inter.feature2}")
            print(f"     Strength: {inter.mean_abs_interaction:.4f}, Direction: {inter.direction}")

        return

    # Run actual analysis (small sample for testing)
    print("\nRunning interaction analysis with 100 samples (test mode)...")
    result = analyzer.find_top_interactions(n_samples=100, top_n=10, save_to_db=False)

    print(f"\nAnalysis complete in {result.compute_time_seconds:.1f}s")
    print(f"Semantic features: {result.semantic_features}")
    print(f"Base features: {result.base_features}")

    if result.top_interactions:
        print("\nTop 5 Interactions:")
        for i, inter in enumerate(result.top_interactions[:5], 1):
            print(f"  {i}. {inter.feature1} × {inter.feature2}")
            print(f"     Strength: {inter.mean_abs_interaction:.6f}, Direction: {inter.direction}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_interactions()
