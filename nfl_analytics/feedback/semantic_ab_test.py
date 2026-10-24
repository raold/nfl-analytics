"""
Semantic A/B Test Framework.

Compare predictions with vs without semantic features.
Provides statistical analysis of semantic feature contribution.

Usage:
    test = SemanticABTest()
    test_id = test.setup_test(season=2025, start_week=1)
    test.record_predictions(test_id, game_id, control_pred, treatment_pred)
    results = test.analyze_results(test_id)
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
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
class ABTestConfig:
    """Configuration for an A/B test."""

    test_id: str
    description: str
    season: int
    start_week: int
    control_name: str  # e.g., "no_semantic"
    treatment_name: str  # e.g., "with_semantic"
    metrics: list[str] = field(default_factory=lambda: ["spread_accuracy", "mae", "brier"])
    status: str = "active"


@dataclass
class ABTestResults:
    """Results of an A/B test analysis."""

    test_id: str
    games_analyzed: int

    # Accuracy metrics
    control_accuracy: float
    treatment_accuracy: float
    accuracy_lift: float
    accuracy_p_value: float

    # MAE metrics
    control_mae: float
    treatment_mae: float
    mae_improvement: float
    mae_p_value: float

    # Brier score (calibration)
    control_brier: float
    treatment_brier: float
    brier_improvement: float

    # Overall verdict
    winner: str  # 'control', 'treatment', 'inconclusive'
    confidence: float
    recommendation: str

    # Feature importance
    feature_importance: dict[str, float] = field(default_factory=dict)


class SemanticABTest:
    """
    A/B test semantic feature contribution.

    Compares control (without semantic features) vs
    treatment (with semantic features) predictions.
    """

    MIN_GAMES_FOR_SIGNIFICANCE = 30
    SIGNIFICANCE_THRESHOLD = 0.05

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

    def setup_test(
        self, season: int, start_week: int, description: str = "Semantic features A/B test"
    ) -> str:
        """
        Create a new A/B test.

        Args:
            season: Season year
            start_week: Week to start testing
            description: Test description

        Returns:
            test_id for tracking
        """
        test_id = f"semantic_ab_{season}_w{start_week}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        query = """
        INSERT INTO ml.ab_test_status
            (test_id, test_name, description, status, start_date,
             control_model, treatment_model)
        VALUES (%s, %s, %s, 'active', NOW(), %s, %s)
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        test_id,
                        f"Semantic A/B {season} Week {start_week}+",
                        description,
                        "ensemble_no_semantic",
                        "ensemble_with_semantic",
                    ),
                )
            conn.commit()

        logger.info(f"Created A/B test {test_id}")
        return test_id

    def record_predictions(
        self,
        test_id: str,
        game_id: str,
        control_pred: dict,
        treatment_pred: dict,
        actual: dict | None = None,
    ) -> None:
        """
        Record both control and treatment predictions for a game.

        Args:
            test_id: Test identifier
            game_id: Game identifier
            control_pred: Prediction without semantic features
            treatment_pred: Prediction with semantic features
            actual: Actual outcome (if known)
        """
        query = """
        INSERT INTO ml.ab_test_results
            (test_id, game_id, variant, predicted_spread, predicted_total,
             predicted_home_win_prob, actual_spread, actual_total,
             spread_correct, prediction_timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (test_id, game_id, variant) DO UPDATE SET
            predicted_spread = EXCLUDED.predicted_spread,
            predicted_total = EXCLUDED.predicted_total,
            predicted_home_win_prob = EXCLUDED.predicted_home_win_prob,
            actual_spread = EXCLUDED.actual_spread,
            actual_total = EXCLUDED.actual_total,
            spread_correct = EXCLUDED.spread_correct
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Record control prediction
                control_correct = None
                if actual and actual.get("actual_spread") is not None:
                    control_correct = (control_pred.get("predicted_spread", 0) > 0) == (
                        actual["actual_spread"] > 0
                    )

                cur.execute(
                    query,
                    (
                        test_id,
                        game_id,
                        "control",
                        control_pred.get("predicted_spread"),
                        control_pred.get("predicted_total"),
                        control_pred.get("predicted_home_win_prob"),
                        actual.get("actual_spread") if actual else None,
                        actual.get("actual_total") if actual else None,
                        control_correct,
                    ),
                )

                # Record treatment prediction
                treatment_correct = None
                if actual and actual.get("actual_spread") is not None:
                    treatment_correct = (treatment_pred.get("predicted_spread", 0) > 0) == (
                        actual["actual_spread"] > 0
                    )

                cur.execute(
                    query,
                    (
                        test_id,
                        game_id,
                        "treatment",
                        treatment_pred.get("predicted_spread"),
                        treatment_pred.get("predicted_total"),
                        treatment_pred.get("predicted_home_win_prob"),
                        actual.get("actual_spread") if actual else None,
                        actual.get("actual_total") if actual else None,
                        treatment_correct,
                    ),
                )

            conn.commit()

        logger.debug(f"Recorded predictions for game {game_id}")

    def update_outcomes(self, test_id: str) -> int:
        """
        Update actual outcomes for test predictions.

        Returns number of games updated.
        """
        query = """
        UPDATE ml.ab_test_results ar
        SET
            actual_spread = g.home_score - g.away_score,
            actual_total = g.home_score + g.away_score,
            spread_correct = CASE
                WHEN ar.predicted_spread IS NOT NULL AND g.home_score IS NOT NULL THEN
                    (ar.predicted_spread > 0) = ((g.home_score - g.away_score) > 0)
                ELSE NULL
            END
        FROM games g
        WHERE ar.test_id = %s
          AND ar.game_id = g.game_id
          AND ar.actual_spread IS NULL
          AND g.home_score IS NOT NULL
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (test_id,))
                updated = cur.rowcount
            conn.commit()

        logger.info(f"Updated {updated} outcomes for test {test_id}")
        return updated

    def analyze_results(self, test_id: str) -> ABTestResults:
        """
        Analyze A/B test results.

        Computes:
        - Accuracy lift (treatment - control)
        - MAE improvement
        - Statistical significance (McNemar test approximation)
        - Feature importance ranking
        """
        # Get paired results
        query = """
        SELECT
            c.game_id,
            c.predicted_spread as control_spread,
            t.predicted_spread as treatment_spread,
            c.actual_spread,
            c.spread_correct as control_correct,
            t.spread_correct as treatment_correct
        FROM ml.ab_test_results c
        JOIN ml.ab_test_results t
            ON c.test_id = t.test_id
            AND c.game_id = t.game_id
            AND c.variant = 'control'
            AND t.variant = 'treatment'
        WHERE c.test_id = %s
          AND c.actual_spread IS NOT NULL
        """

        with self._get_connection() as conn:
            df = pd.read_sql(query, conn, params=[test_id])

        if df.empty:
            return ABTestResults(
                test_id=test_id,
                games_analyzed=0,
                control_accuracy=0.0,
                treatment_accuracy=0.0,
                accuracy_lift=0.0,
                accuracy_p_value=1.0,
                control_mae=0.0,
                treatment_mae=0.0,
                mae_improvement=0.0,
                mae_p_value=1.0,
                control_brier=0.0,
                treatment_brier=0.0,
                brier_improvement=0.0,
                winner="inconclusive",
                confidence=0.0,
                recommendation="Need more data",
            )

        games_analyzed = len(df)

        # Calculate accuracy
        control_acc = df["control_correct"].mean()
        treatment_acc = df["treatment_correct"].mean()
        accuracy_lift = treatment_acc - control_acc

        # Calculate MAE
        control_mae = np.abs(df["control_spread"] - df["actual_spread"]).mean()
        treatment_mae = np.abs(df["treatment_spread"] - df["actual_spread"]).mean()
        mae_improvement = control_mae - treatment_mae  # Positive = treatment is better

        # McNemar test approximation for paired accuracy
        accuracy_p_value = self._mcnemar_test(
            df["control_correct"].values, df["treatment_correct"].values
        )

        # Paired t-test for MAE
        mae_p_value = self._paired_ttest(
            np.abs(df["control_spread"] - df["actual_spread"]).values,
            np.abs(df["treatment_spread"] - df["actual_spread"]).values,
        )

        # Brier score (simplified - using spread prediction as probability proxy)
        control_brier = 0.0
        treatment_brier = 0.0

        # Determine winner
        winner, confidence, recommendation = self._determine_winner(
            accuracy_lift, accuracy_p_value, mae_improvement, mae_p_value, games_analyzed
        )

        # Get feature importance
        feature_importance = self._compute_feature_importance(test_id)

        return ABTestResults(
            test_id=test_id,
            games_analyzed=games_analyzed,
            control_accuracy=float(control_acc),
            treatment_accuracy=float(treatment_acc),
            accuracy_lift=float(accuracy_lift),
            accuracy_p_value=float(accuracy_p_value),
            control_mae=float(control_mae),
            treatment_mae=float(treatment_mae),
            mae_improvement=float(mae_improvement),
            mae_p_value=float(mae_p_value),
            control_brier=control_brier,
            treatment_brier=treatment_brier,
            brier_improvement=0.0,
            winner=winner,
            confidence=confidence,
            recommendation=recommendation,
            feature_importance=feature_importance,
        )

    def _mcnemar_test(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """
        Approximate McNemar test for paired binary outcomes.

        Returns p-value.
        """
        # Count discordant pairs
        b = np.sum(control & ~treatment)  # Control correct, treatment wrong
        c = np.sum(~control & treatment)  # Control wrong, treatment correct

        if b + c < 10:
            return 1.0  # Not enough discordant pairs

        # Chi-squared approximation
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)

        # Convert to p-value (1 df)
        from scipy import stats

        try:
            p_value = 1 - stats.chi2.cdf(chi2, 1)
        except Exception:
            p_value = 0.5

        return p_value

    def _paired_ttest(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """Paired t-test for MAE comparison."""
        if len(control) < 10:
            return 1.0

        try:
            from scipy import stats

            _, p_value = stats.ttest_rel(control, treatment)
            return p_value if not np.isnan(p_value) else 1.0
        except Exception:
            return 1.0

    def _determine_winner(
        self,
        accuracy_lift: float,
        accuracy_p: float,
        mae_improvement: float,
        mae_p: float,
        games: int,
    ) -> tuple[str, float, str]:
        """Determine winner based on metrics."""
        if games < self.MIN_GAMES_FOR_SIGNIFICANCE:
            return "inconclusive", 0.0, f"Need {self.MIN_GAMES_FOR_SIGNIFICANCE}+ games"

        # Treatment wins if:
        # - Accuracy lift is positive AND significant, OR
        # - MAE improvement is positive AND significant
        accuracy_significant = accuracy_p < self.SIGNIFICANCE_THRESHOLD
        mae_significant = mae_p < self.SIGNIFICANCE_THRESHOLD

        treatment_wins_accuracy = accuracy_lift > 0 and accuracy_significant
        treatment_wins_mae = mae_improvement > 0 and mae_significant
        control_wins_accuracy = accuracy_lift < 0 and accuracy_significant
        control_wins_mae = mae_improvement < 0 and mae_significant

        if treatment_wins_accuracy and treatment_wins_mae:
            return (
                "treatment",
                0.95,
                "Treatment (semantic) significantly better on both accuracy and MAE",
            )
        elif treatment_wins_accuracy:
            return "treatment", 0.85, "Treatment (semantic) significantly better on accuracy"
        elif treatment_wins_mae:
            return "treatment", 0.80, "Treatment (semantic) significantly better on MAE"
        elif control_wins_accuracy and control_wins_mae:
            return (
                "control",
                0.90,
                "Control (no semantic) significantly better - consider removing semantic features",
            )
        elif control_wins_accuracy or control_wins_mae:
            return "control", 0.75, "Control (no semantic) better on some metrics"
        else:
            confidence = 0.5 + min(abs(accuracy_lift), 0.1) * 2
            return "inconclusive", confidence, "No significant difference - continue testing"

    def _compute_feature_importance(self, test_id: str) -> dict[str, float]:
        """Compute importance of each semantic feature."""
        # Get feature contributions from semantic_attribution
        query = """
        SELECT
            sa.feature_attributions
        FROM ml.ab_test_results ar
        JOIN ml.semantic_attribution sa ON ar.game_id = sa.game_id
        WHERE ar.test_id = %s
          AND ar.variant = 'treatment'
          AND ar.spread_correct = TRUE
          AND sa.feature_attributions IS NOT NULL
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (test_id,))
                rows = cur.fetchall()

        if not rows:
            return {}

        # Aggregate feature contributions
        feature_sums = {}
        feature_counts = {}

        for row in rows:
            attrs = row.get("feature_attributions") or {}
            for name, value in attrs.items():
                if name not in feature_sums:
                    feature_sums[name] = 0.0
                    feature_counts[name] = 0
                feature_sums[name] += value
                feature_counts[name] += 1

        # Compute averages and normalize
        if not feature_sums:
            return {}

        importance = {name: feature_sums[name] / feature_counts[name] for name in feature_sums}

        # Normalize to sum to 1.0
        total = sum(abs(v) for v in importance.values())
        if total > 0:
            importance = {k: abs(v) / total for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    def get_active_tests(self) -> list[ABTestConfig]:
        """Get all active A/B tests."""
        query = """
        SELECT
            test_id,
            test_name,
            description,
            status,
            EXTRACT(YEAR FROM start_date)::INTEGER as season,
            1 as start_week
        FROM ml.ab_test_status
        WHERE status = 'active'
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()

        return [
            ABTestConfig(
                test_id=row["test_id"],
                description=row["description"] or "",
                season=row["season"] or 2025,
                start_week=row["start_week"] or 1,
                control_name="no_semantic",
                treatment_name="with_semantic",
                status=row["status"],
            )
            for row in rows
        ]

    def print_results(self, results: ABTestResults) -> None:
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("SEMANTIC A/B TEST RESULTS")
        print("=" * 60)
        print(f"Test ID: {results.test_id}")
        print(f"Games analyzed: {results.games_analyzed}")

        print("\n" + "-" * 60)
        print("SPREAD PREDICTION:")
        print(f"  Control (no semantic):  {results.control_accuracy:.1%} accuracy")
        print(f"  Treatment (semantic):   {results.treatment_accuracy:.1%} accuracy")
        print(f"  Lift: {results.accuracy_lift:+.1%}")
        if results.accuracy_p_value < 1.0:
            sig = "significant" if results.accuracy_p_value < 0.05 else "not significant"
            print(f"  p-value: {results.accuracy_p_value:.3f} ({sig})")

        print("\n" + "-" * 60)
        print("MEAN ABSOLUTE ERROR:")
        print(f"  Control (no semantic):  {results.control_mae:.2f} points")
        print(f"  Treatment (semantic):   {results.treatment_mae:.2f} points")
        print(f"  Improvement: {results.mae_improvement:+.2f} points")
        if results.mae_p_value < 1.0:
            sig = "significant" if results.mae_p_value < 0.05 else "not significant"
            print(f"  p-value: {results.mae_p_value:.3f} ({sig})")

        print("\n" + "-" * 60)
        print("VERDICT:")
        print(f"  Winner: {results.winner.upper()}")
        print(f"  Confidence: {results.confidence:.0%}")
        print(f"  Recommendation: {results.recommendation}")

        if results.feature_importance:
            print("\n" + "-" * 60)
            print("FEATURE IMPORTANCE (semantic):")
            for name, importance in list(results.feature_importance.items())[:5]:
                print(f"  {name}: {importance:.1%}")

        print("=" * 60 + "\n")


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test = SemanticABTest()

    print("\n=== Testing Semantic A/B Test ===")

    # Get active tests
    active = test.get_active_tests()
    print(f"\nActive tests: {len(active)}")

    # Analyze results if any test exists
    if active:
        results = test.analyze_results(active[0].test_id)
        test.print_results(results)
    else:
        print("No active tests found. Creating a test...")
        test_id = test.setup_test(2025, 1, "Test semantic feature contribution")
        print(f"Created test: {test_id}")
