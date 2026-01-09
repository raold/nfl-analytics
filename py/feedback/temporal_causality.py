"""
Temporal Causality Graph for Prediction Pipeline.

Traces the causal chain: News → Sentiment → Feature → Prediction → Outcome
When predictions are wrong, helps identify root cause.

Based on Gemini consultation (Dec 28, 2025):
- Store timestamps at each stage
- Track model versions for reproducibility
- Enable retrospective re-analysis

Usage:
    tracker = TemporalCausalityTracker()
    chain = tracker.trace_prediction(game_id='2025_16_KC_PIT')
    diagnosis = tracker.diagnose_error(chain)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class CausalNode:
    """A node in the causal graph."""

    stage: str  # news, sentiment, feature, prediction, outcome
    timestamp: datetime | None
    value: Any
    model_version: str | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class CausalChain:
    """Complete causal chain for a prediction."""

    game_id: str
    nodes: dict[str, CausalNode] = field(default_factory=dict)

    # Error attribution
    error: float | None = None
    error_direction: str = ""  # "over" or "under"
    root_cause: str = ""  # Which stage likely caused the error
    root_cause_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "nodes": {
                k: {
                    "stage": v.stage,
                    "timestamp": str(v.timestamp) if v.timestamp else None,
                    "value": (
                        v.value if not isinstance(v.value, (dict, list)) else json.dumps(v.value)
                    ),
                    "model_version": v.model_version,
                    "confidence": v.confidence,
                }
                for k, v in self.nodes.items()
            },
            "error": self.error,
            "error_direction": self.error_direction,
            "root_cause": self.root_cause,
            "root_cause_confidence": self.root_cause_confidence,
        }


@dataclass
class ErrorDiagnosis:
    """Diagnosis of what went wrong."""

    game_id: str
    error: float
    error_direction: str

    # Probability each stage caused the error
    stage_probabilities: dict[str, float] = field(default_factory=dict)

    # Most likely root cause
    primary_cause: str = ""
    primary_cause_explanation: str = ""

    # Recommendations
    recommendations: list[str] = field(default_factory=list)


class TemporalCausalityTracker:
    """
    Track and trace the causal chain of predictions.

    Stages:
    1. News: Raw news ingestion (injuries, roster moves, etc.)
    2. Sentiment: LLM-processed sentiment scores
    3. Feature: Extracted features (semantic + base)
    4. Prediction: Model predictions
    5. Outcome: Actual game results
    """

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
        self._ensure_tables()

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def _ensure_tables(self):
        """Create required tables if they don't exist."""
        create_tables = """
        -- Causal chain tracking table
        CREATE TABLE IF NOT EXISTS ml.causal_chains (
            id SERIAL PRIMARY KEY,
            game_id VARCHAR(50) NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,

            -- Stage timestamps
            news_timestamp TIMESTAMP,
            sentiment_timestamp TIMESTAMP,
            feature_timestamp TIMESTAMP,
            prediction_timestamp TIMESTAMP,
            outcome_timestamp TIMESTAMP,

            -- Stage values (JSON for flexibility)
            news_data JSONB,
            sentiment_data JSONB,
            feature_data JSONB,
            prediction_data JSONB,
            outcome_data JSONB,

            -- Model versions
            sentiment_model_version VARCHAR(50),
            feature_extractor_version VARCHAR(50),
            prediction_model_version VARCHAR(50),

            -- Error attribution
            prediction_error NUMERIC(5,2),
            error_direction VARCHAR(10),
            root_cause VARCHAR(50),
            root_cause_confidence NUMERIC(5,4),
            diagnosis JSONB,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (game_id)
        );

        -- Index for efficient lookups
        CREATE INDEX IF NOT EXISTS idx_causal_chains_season_week
        ON ml.causal_chains(season, week);
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_tables)
            conn.commit()

    def record_news_event(
        self,
        game_id: str,
        news_text: str,
        source: str,
        publication_time: datetime,
        entities: list[dict] | None = None,
    ):
        """Record a news event for tracing."""
        query = """
        INSERT INTO ml.causal_chains (game_id, season, week, news_timestamp, news_data)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (game_id) DO UPDATE SET
            news_timestamp = EXCLUDED.news_timestamp,
            news_data = ml.causal_chains.news_data || EXCLUDED.news_data
        """

        # Extract season/week from game_id (format: 2025_16_KC_PIT)
        parts = game_id.split("_")
        season = int(parts[0]) if len(parts) >= 2 else 2025
        week = int(parts[1]) if len(parts) >= 2 else 1

        news_data = {
            "texts": [news_text],
            "sources": [source],
            "entities": entities or [],
        }

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (game_id, season, week, publication_time, json.dumps(news_data)))
            conn.commit()

    def record_sentiment(
        self,
        game_id: str,
        sentiment_scores: dict[str, float],
        model_version: str = "ollama-qwen3:8b",
        timestamp: datetime | None = None,
    ):
        """Record sentiment analysis results."""
        query = """
        UPDATE ml.causal_chains SET
            sentiment_timestamp = %s,
            sentiment_data = %s,
            sentiment_model_version = %s
        WHERE game_id = %s
        """

        timestamp = timestamp or datetime.now()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query, (timestamp, json.dumps(sentiment_scores), model_version, game_id)
                )
            conn.commit()

    def record_features(
        self,
        game_id: str,
        features: dict[str, float],
        extractor_version: str = "v6.7",
        timestamp: datetime | None = None,
    ):
        """Record extracted features."""
        query = """
        UPDATE ml.causal_chains SET
            feature_timestamp = %s,
            feature_data = %s,
            feature_extractor_version = %s
        WHERE game_id = %s
        """

        timestamp = timestamp or datetime.now()

        # Store summary of features, not full vector
        feature_summary = {
            "count": len(features),
            "semantic_count": len([k for k in features if k.startswith("sem_")]),
            "top_10": dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]),
        }

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query, (timestamp, json.dumps(feature_summary), extractor_version, game_id)
                )
            conn.commit()

    def record_prediction(
        self,
        game_id: str,
        predicted_spread: float,
        predicted_total: float | None = None,
        model_version: str = "ensemble-v6.2",
        confidence: float | None = None,
        timestamp: datetime | None = None,
    ):
        """Record model prediction."""
        query = """
        UPDATE ml.causal_chains SET
            prediction_timestamp = %s,
            prediction_data = %s,
            prediction_model_version = %s
        WHERE game_id = %s
        """

        timestamp = timestamp or datetime.now()

        prediction_data = {
            "spread": predicted_spread,
            "total": predicted_total,
            "confidence": confidence,
        }

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (timestamp, json.dumps(prediction_data), model_version, game_id))
            conn.commit()

    def record_outcome(
        self,
        game_id: str,
        actual_spread: float,
        actual_total: float | None = None,
        timestamp: datetime | None = None,
    ):
        """Record actual game outcome and compute error."""
        # First get prediction
        query = """
        SELECT prediction_data FROM ml.causal_chains WHERE game_id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (game_id,))
                row = cur.fetchone()

        prediction = json.loads(row[0]) if row and row[0] else {}
        predicted_spread = prediction.get("spread")

        error = None
        error_direction = ""
        if predicted_spread is not None:
            error = actual_spread - predicted_spread
            error_direction = "over" if error > 0 else "under" if error < 0 else "exact"

        timestamp = timestamp or datetime.now()

        outcome_data = {
            "spread": actual_spread,
            "total": actual_total,
        }

        update_query = """
        UPDATE ml.causal_chains SET
            outcome_timestamp = %s,
            outcome_data = %s,
            prediction_error = %s,
            error_direction = %s
        WHERE game_id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    update_query,
                    (timestamp, json.dumps(outcome_data), error, error_direction, game_id),
                )
            conn.commit()

    def trace_prediction(self, game_id: str) -> CausalChain | None:
        """
        Retrieve the full causal chain for a game.

        Returns CausalChain with all stages and timestamps.
        """
        query = """
        SELECT *
        FROM ml.causal_chains
        WHERE game_id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (game_id,))
                row = cur.fetchone()

        if not row:
            return None

        chain = CausalChain(game_id=game_id)

        # Build nodes
        if row["news_data"]:
            chain.nodes["news"] = CausalNode(
                stage="news",
                timestamp=row["news_timestamp"],
                value=row["news_data"],
            )

        if row["sentiment_data"]:
            chain.nodes["sentiment"] = CausalNode(
                stage="sentiment",
                timestamp=row["sentiment_timestamp"],
                value=row["sentiment_data"],
                model_version=row["sentiment_model_version"],
            )

        if row["feature_data"]:
            chain.nodes["feature"] = CausalNode(
                stage="feature",
                timestamp=row["feature_timestamp"],
                value=row["feature_data"],
                model_version=row["feature_extractor_version"],
            )

        if row["prediction_data"]:
            chain.nodes["prediction"] = CausalNode(
                stage="prediction",
                timestamp=row["prediction_timestamp"],
                value=row["prediction_data"],
                model_version=row["prediction_model_version"],
            )

        if row["outcome_data"]:
            chain.nodes["outcome"] = CausalNode(
                stage="outcome",
                timestamp=row["outcome_timestamp"],
                value=row["outcome_data"],
            )

        chain.error = float(row["prediction_error"]) if row["prediction_error"] else None
        chain.error_direction = row["error_direction"] or ""
        chain.root_cause = row["root_cause"] or ""
        chain.root_cause_confidence = (
            float(row["root_cause_confidence"]) if row["root_cause_confidence"] else 0.0
        )

        return chain

    def diagnose_error(self, chain: CausalChain) -> ErrorDiagnosis:
        """
        Diagnose what caused a prediction error.

        Uses heuristics based on:
        - Sentiment confidence
        - Feature completeness
        - Prediction confidence
        - Error magnitude
        """
        if chain.error is None:
            return ErrorDiagnosis(
                game_id=chain.game_id,
                error=0,
                error_direction="none",
                stage_probabilities={},
                primary_cause="unknown",
                primary_cause_explanation="No error to diagnose",
            )

        error = abs(chain.error)
        diagnosis = ErrorDiagnosis(
            game_id=chain.game_id,
            error=chain.error,
            error_direction=chain.error_direction,
        )

        # Initialize probabilities
        probs = {
            "news": 0.1,  # Missing/wrong news
            "sentiment": 0.2,  # Wrong sentiment extraction
            "feature": 0.2,  # Feature extraction error
            "prediction": 0.3,  # Model weights wrong
            "random": 0.2,  # Irreducible noise
        }

        # Adjust based on available data
        if "sentiment" in chain.nodes:
            sentiment = chain.nodes["sentiment"].value
            # Low sentiment values suggest weak signal
            if isinstance(sentiment, dict):
                avg_sentiment = np.mean(
                    [abs(v) for v in sentiment.values() if isinstance(v, (int, float))]
                )
                if avg_sentiment < 0.1:
                    probs["sentiment"] += 0.15
                    probs["prediction"] -= 0.1

        if "feature" in chain.nodes:
            feature = chain.nodes["feature"].value
            if isinstance(feature, dict):
                semantic_count = feature.get("semantic_count", 0)
                if semantic_count < 10:
                    probs["feature"] += 0.1
                    probs["sentiment"] += 0.05

        if "prediction" in chain.nodes:
            pred = chain.nodes["prediction"].value
            if isinstance(pred, dict):
                confidence = pred.get("confidence")
                if confidence and confidence < 0.5:
                    probs["prediction"] += 0.15

        # Large errors more likely model fault
        if error > 10:
            probs["prediction"] += 0.1
            probs["random"] -= 0.05
        elif error < 3:
            probs["random"] += 0.1
            probs["prediction"] -= 0.05

        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        diagnosis.stage_probabilities = {k: round(v, 3) for k, v in probs.items()}

        # Find primary cause
        primary = max(probs.items(), key=lambda x: x[1])
        diagnosis.primary_cause = primary[0]

        # Generate explanation
        explanations = {
            "news": "Missing or incomplete news coverage for this matchup",
            "sentiment": "Sentiment analysis may have misinterpreted news context",
            "feature": "Feature extraction failed to capture relevant signals",
            "prediction": "Model ensemble weights may be suboptimal for this matchup type",
            "random": "Error within expected variance (irreducible noise)",
        }
        diagnosis.primary_cause_explanation = explanations.get(primary[0], "Unknown cause")

        # Generate recommendations
        recommendations = []
        if probs["sentiment"] > 0.25:
            recommendations.append("Review sentiment model output for this game's news")
        if probs["feature"] > 0.25:
            recommendations.append("Check if semantic features were properly extracted")
        if probs["prediction"] > 0.35:
            recommendations.append("Consider adjusting ensemble weights for this game type")
        if probs["news"] > 0.15:
            recommendations.append("Expand news sources for better coverage")

        diagnosis.recommendations = recommendations

        return diagnosis

    def save_diagnosis(self, game_id: str, diagnosis: ErrorDiagnosis):
        """Save diagnosis to database."""
        query = """
        UPDATE ml.causal_chains SET
            root_cause = %s,
            root_cause_confidence = %s,
            diagnosis = %s
        WHERE game_id = %s
        """

        diagnosis_json = {
            "stage_probabilities": diagnosis.stage_probabilities,
            "primary_cause": diagnosis.primary_cause,
            "explanation": diagnosis.primary_cause_explanation,
            "recommendations": diagnosis.recommendations,
        }

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        diagnosis.primary_cause,
                        diagnosis.stage_probabilities.get(diagnosis.primary_cause, 0),
                        json.dumps(diagnosis_json),
                        game_id,
                    ),
                )
            conn.commit()

    def get_root_cause_summary(
        self,
        season: int,
        start_week: int = 1,
        end_week: int = 18,
    ) -> pd.DataFrame:
        """
        Get summary of root causes across games.

        Returns DataFrame with root cause frequencies.
        """
        query = """
        SELECT
            root_cause,
            COUNT(*) as count,
            AVG(ABS(prediction_error)) as avg_error,
            AVG(root_cause_confidence) as avg_confidence
        FROM ml.causal_chains
        WHERE season = %s AND week >= %s AND week <= %s
          AND root_cause IS NOT NULL
        GROUP BY root_cause
        ORDER BY count DESC
        """

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(season, start_week, end_week))

        return df


def test_causality():
    """Test temporal causality tracking."""
    print("\n" + "=" * 60)
    print("TEMPORAL CAUSALITY TEST")
    print("=" * 60)

    tracker = TemporalCausalityTracker()

    # Create a synthetic example
    test_game = "2025_16_KC_PIT"

    print(f"\nRecording causal chain for: {test_game}")

    # Record each stage
    tracker.record_news_event(
        game_id=test_game,
        news_text="Patrick Mahomes questionable with ankle injury",
        source="ESPN",
        publication_time=datetime.now(),
        entities=[{"text": "Mahomes", "type": "PLAYER"}, {"text": "ankle", "type": "INJURY"}],
    )

    tracker.record_sentiment(
        game_id=test_game,
        sentiment_scores={"home_sentiment": -0.3, "away_sentiment": 0.1},
    )

    tracker.record_features(
        game_id=test_game,
        features={"sem_home_sentiment": -0.3, "sem_injury_severity": 3, "base_feature_1": 0.5},
    )

    tracker.record_prediction(
        game_id=test_game,
        predicted_spread=-3.5,
        confidence=0.65,
    )

    tracker.record_outcome(
        game_id=test_game,
        actual_spread=7.0,
    )

    # Trace and diagnose
    chain = tracker.trace_prediction(test_game)

    if chain:
        print(f"\nCausal chain retrieved:")
        print(f"  Error: {chain.error}")
        print(f"  Direction: {chain.error_direction}")
        print(f"  Stages: {list(chain.nodes.keys())}")

        diagnosis = tracker.diagnose_error(chain)
        print(f"\nDiagnosis:")
        print(f"  Primary cause: {diagnosis.primary_cause}")
        print(f"  Explanation: {diagnosis.primary_cause_explanation}")
        print(f"  Stage probabilities: {diagnosis.stage_probabilities}")
        print(f"  Recommendations: {diagnosis.recommendations}")

        # Save diagnosis
        tracker.save_diagnosis(test_game, diagnosis)
        print("\nDiagnosis saved to database")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_causality()
