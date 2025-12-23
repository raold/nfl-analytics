"""
LLM Prompt Evolver.

Evolves sentiment/NER prompts based on prediction performance.
Creates A/B tests for prompt variations and tracks results.

Usage:
    evolver = PromptEvolver()
    analysis = evolver.analyze_failures('sem_sentiment_24h', game_ids)
    suggestions = evolver.suggest_prompt_improvements(analysis)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysis:
    """Analysis of why a semantic feature led to wrong predictions."""
    feature: str
    total_failures: int
    failure_patterns: list[str]
    common_issues: dict[str, int]
    sample_texts: list[str]
    severity: str  # 'critical', 'moderate', 'minor'


@dataclass
class PromptSuggestion:
    """Suggested improvement to a prompt."""
    feature: str
    issue_addressed: str
    current_prompt_summary: str
    suggested_change: str
    expected_improvement: str
    confidence: float


@dataclass
class PromptABTest:
    """A/B test configuration for prompts."""
    test_id: str
    feature: str
    baseline_version: str
    treatment_version: str
    baseline_prompt: str
    treatment_prompt: str
    start_date: datetime
    games_tested: int = 0
    baseline_accuracy: float = 0.0
    treatment_accuracy: float = 0.0


class PromptEvolver:
    """
    Evolve semantic prompts based on prediction outcomes.

    When a semantic feature contributes to wrong predictions:
    1. Analyze the misclassified cases
    2. Identify prompt weaknesses
    3. Generate improved prompts
    4. A/B test new vs old
    """

    # Default prompts for semantic features
    DEFAULT_PROMPTS = {
        "sentiment": """Analyze the sentiment of this NFL news article about {team}.
Rate the overall sentiment on a scale from -1 (very negative) to +1 (very positive).
Focus on factors that could affect the team's performance.
Consider: injuries, roster changes, coaching decisions, team morale, and momentum.""",

        "injury_severity": """Assess the impact of this injury on {team}'s performance.
Player: {player}
Injury: {injury_type}
Status: {status}

Rate the impact from 1 (minimal) to 5 (severe).
Consider: player importance, position depth, matchup implications.""",

        "ner": """Extract injury-related entities from this NFL news text.
Identify:
- PLAYER: Player names mentioned with injury context
- INJURY: Injury types (ankle, knee, hamstring, concussion, etc.)
- STATUS: Game status (out, questionable, doubtful, probable)

Return as JSON: {"entities": [{"text": "...", "type": "...", "start": N, "end": N}]}"""
    }

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

    def analyze_failures(
        self,
        feature: str,
        game_ids: list[str] | None = None,
        season: int | None = None,
        weeks: int = 4
    ) -> FailureAnalysis:
        """
        Analyze why a semantic feature led to wrong predictions.

        Examines patterns in failure cases to identify prompt weaknesses.
        """
        # Get failure cases
        if game_ids:
            failures = self._get_failures_by_ids(feature, game_ids)
        else:
            failures = self._get_failures_by_period(feature, season or 2025, weeks)

        if not failures:
            return FailureAnalysis(
                feature=feature,
                total_failures=0,
                failure_patterns=[],
                common_issues={},
                sample_texts=[],
                severity="minor"
            )

        # Analyze patterns
        patterns = self._identify_failure_patterns(failures)
        common_issues = self._identify_common_issues(failures)
        sample_texts = [f.get("source_text", "")[:200] for f in failures[:5] if f.get("source_text")]

        # Determine severity
        severity = self._determine_severity(len(failures), patterns)

        return FailureAnalysis(
            feature=feature,
            total_failures=len(failures),
            failure_patterns=patterns,
            common_issues=common_issues,
            sample_texts=sample_texts,
            severity=severity
        )

    def _get_failures_by_ids(self, feature: str, game_ids: list[str]) -> list[dict]:
        """Get failure cases for specific games."""
        query = """
        SELECT
            fp.game_id,
            fp.feature_value,
            fp.prediction_error,
            n.title || ' ' || COALESCE(n.description, '') as source_text,
            n.category
        FROM ml.feature_performance fp
        LEFT JOIN semantic.news_snapshots n ON n.teams @> ARRAY[
            COALESCE(
                (SELECT home_team FROM games WHERE game_id = fp.game_id),
                ''
            )
        ]
        WHERE fp.feature_name = %s
          AND fp.game_id = ANY(%s)
          AND fp.correct_prediction = FALSE
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (feature, game_ids))
                return [dict(row) for row in cur.fetchall()]

    def _get_failures_by_period(
        self,
        feature: str,
        season: int,
        weeks: int
    ) -> list[dict]:
        """Get failure cases for a time period."""
        query = """
        SELECT
            fp.game_id,
            fp.feature_value,
            fp.prediction_error,
            fp.season,
            fp.week
        FROM ml.feature_performance fp
        WHERE fp.feature_name = %s
          AND fp.correct_prediction = FALSE
          AND fp.created_at >= NOW() - INTERVAL '%s weeks'
        ORDER BY fp.prediction_error DESC
        LIMIT 100
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (feature, weeks))
                return [dict(row) for row in cur.fetchall()]

    def _identify_failure_patterns(self, failures: list[dict]) -> list[str]:
        """Identify patterns in failure cases."""
        patterns = []

        if not failures:
            return patterns

        # Check for high-confidence failures
        values = [f.get("feature_value", 0) for f in failures]
        avg_value = np.mean([abs(v) for v in values if v])

        if avg_value > 0.6:
            patterns.append("Overconfident extraction - failures when feature value is extreme")

        # Check for specific categories
        categories = [f.get("category") for f in failures if f.get("category")]
        if categories:
            from collections import Counter
            top_cat = Counter(categories).most_common(1)
            if top_cat and top_cat[0][1] > len(failures) * 0.5:
                patterns.append(f"Category bias - {top_cat[0][0]} articles overrepresented in failures")

        # Check for high-error cases
        errors = [f.get("prediction_error", 0) for f in failures]
        high_errors = [e for e in errors if e and e > 10]
        if len(high_errors) > len(failures) * 0.3:
            patterns.append("Large errors - feature contributing to blowout mispredictions")

        # Check for sign errors (positive value but negative outcome)
        sign_mismatches = [
            f for f in failures
            if f.get("feature_value", 0) > 0  # Predicted positive
        ]
        if len(sign_mismatches) > len(failures) * 0.6:
            patterns.append("Polarity confusion - positive sentiment with negative outcomes")

        return patterns or ["No clear pattern identified"]

    def _identify_common_issues(self, failures: list[dict]) -> dict[str, int]:
        """Identify common issues in failure cases."""
        issues = {
            "overconfidence": 0,
            "noise_sensitivity": 0,
            "temporal_mismatch": 0,
            "sarcasm_detection": 0,
            "injury_impact": 0,
        }

        for f in failures:
            value = f.get("feature_value", 0)
            error = f.get("prediction_error", 0)
            text = f.get("source_text", "").lower()

            # Overconfidence: extreme value but wrong
            if abs(value) > 0.7:
                issues["overconfidence"] += 1

            # Noise: low value but high error
            if abs(value) < 0.2 and error and error > 7:
                issues["noise_sensitivity"] += 1

            # Sarcasm indicators
            sarcasm_words = ["lol", "lmao", "yeah right", "sure", "/s"]
            if any(w in text for w in sarcasm_words):
                issues["sarcasm_detection"] += 1

            # Injury-related
            if any(w in text for w in ["injury", "injured", "out", "questionable"]):
                issues["injury_impact"] += 1

        return {k: v for k, v in issues.items() if v > 0}

    def _determine_severity(self, failure_count: int, patterns: list[str]) -> str:
        """Determine severity of the issues found."""
        if failure_count > 50 or "Overconfident" in str(patterns):
            return "critical"
        elif failure_count > 20 or len(patterns) > 2:
            return "moderate"
        return "minor"

    def suggest_prompt_improvements(
        self,
        analysis: FailureAnalysis
    ) -> list[PromptSuggestion]:
        """Generate improved prompts addressing identified weaknesses."""
        suggestions = []

        feature_type = self._get_feature_type(analysis.feature)
        current_prompt = self.DEFAULT_PROMPTS.get(feature_type, "")

        # Generate suggestions based on patterns
        for pattern in analysis.failure_patterns:
            if "Overconfident" in pattern:
                suggestions.append(PromptSuggestion(
                    feature=analysis.feature,
                    issue_addressed=pattern,
                    current_prompt_summary=current_prompt[:100] + "...",
                    suggested_change="Add uncertainty acknowledgment: 'Express confidence level and note when signal is ambiguous'",
                    expected_improvement="Reduce extreme predictions by 20-30%",
                    confidence=0.7
                ))

            elif "Category bias" in pattern:
                cat = pattern.split("-")[1].strip().split()[0]
                suggestions.append(PromptSuggestion(
                    feature=analysis.feature,
                    issue_addressed=pattern,
                    current_prompt_summary=current_prompt[:100] + "...",
                    suggested_change=f"Add category weighting: 'Weight {cat} articles less heavily'",
                    expected_improvement="Better category balance",
                    confidence=0.6
                ))

            elif "Polarity confusion" in pattern:
                suggestions.append(PromptSuggestion(
                    feature=analysis.feature,
                    issue_addressed=pattern,
                    current_prompt_summary=current_prompt[:100] + "...",
                    suggested_change="Improve polarity detection: 'Carefully distinguish between positive news about opponent vs team'",
                    expected_improvement="10-15% accuracy improvement on sign",
                    confidence=0.65
                ))

        # Add suggestions based on common issues
        for issue, count in analysis.common_issues.items():
            if count < 5:
                continue

            if issue == "sarcasm_detection":
                suggestions.append(PromptSuggestion(
                    feature=analysis.feature,
                    issue_addressed=f"Sarcasm missed ({count} cases)",
                    current_prompt_summary=current_prompt[:100] + "...",
                    suggested_change="Add sarcasm handling: 'Detect sarcasm/irony and invert sentiment when detected'",
                    expected_improvement="Better social media analysis",
                    confidence=0.5
                ))

            elif issue == "injury_impact":
                suggestions.append(PromptSuggestion(
                    feature=analysis.feature,
                    issue_addressed=f"Injury impact misjudged ({count} cases)",
                    current_prompt_summary=current_prompt[:100] + "...",
                    suggested_change="Refine injury weighting: 'Consider backup quality and positional depth when assessing injury impact'",
                    expected_improvement="5-10% MAE reduction on injury games",
                    confidence=0.6
                ))

        return suggestions

    def _get_feature_type(self, feature: str) -> str:
        """Map feature name to prompt type."""
        if "sentiment" in feature.lower():
            return "sentiment"
        elif "injury" in feature.lower():
            return "injury_severity"
        elif "ner" in feature.lower():
            return "ner"
        return "sentiment"

    def create_prompt_ab_test(
        self,
        feature: str,
        current_prompt: str,
        new_prompt: str,
        description: str = ""
    ) -> str:
        """
        Set up A/B test for prompt comparison.

        Returns test_id for tracking.
        """
        test_id = f"prompt_ab_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store baseline prompt
        self._store_prompt_version(feature, "baseline", current_prompt, is_baseline=True)

        # Store treatment prompt
        version = datetime.now().strftime("v%Y%m%d")
        self._store_prompt_version(feature, version, new_prompt, is_active=False)

        # Create A/B test entry
        query = """
        INSERT INTO ml.prompt_ab_tests
            (test_id, feature_name, baseline_version, treatment_version,
             description, status, created_at)
        VALUES (%s, %s, %s, %s, %s, 'active', NOW())
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (test_id, feature, "baseline", version, description))
            conn.commit()

        logger.info(f"Created A/B test {test_id} for {feature}")
        return test_id

    def _store_prompt_version(
        self,
        feature: str,
        version: str,
        prompt: str,
        is_active: bool = False,
        is_baseline: bool = False
    ) -> None:
        """Store a prompt version."""
        query = """
        INSERT INTO ml.prompt_versions
            (feature_name, prompt_version, prompt_text, is_active, is_baseline, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (feature_name, prompt_version) DO UPDATE SET
            prompt_text = EXCLUDED.prompt_text,
            is_active = EXCLUDED.is_active
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (feature, version, prompt, is_active, is_baseline))
            conn.commit()

    def get_active_tests(self) -> list[PromptABTest]:
        """Get all active A/B tests."""
        query = """
        SELECT
            pt.test_id,
            pt.feature_name,
            pt.baseline_version,
            pt.treatment_version,
            pb.prompt_text as baseline_prompt,
            pv.prompt_text as treatment_prompt,
            pt.created_at as start_date,
            pt.games_tested,
            pt.baseline_accuracy,
            pt.treatment_accuracy
        FROM ml.prompt_ab_tests pt
        LEFT JOIN ml.prompt_versions pb
            ON pb.feature_name = pt.feature_name AND pb.prompt_version = pt.baseline_version
        LEFT JOIN ml.prompt_versions pv
            ON pv.feature_name = pt.feature_name AND pv.prompt_version = pt.treatment_version
        WHERE pt.status = 'active'
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()

        return [
            PromptABTest(
                test_id=row["test_id"],
                feature=row["feature_name"],
                baseline_version=row["baseline_version"],
                treatment_version=row["treatment_version"],
                baseline_prompt=row["baseline_prompt"] or "",
                treatment_prompt=row["treatment_prompt"] or "",
                start_date=row["start_date"],
                games_tested=row["games_tested"] or 0,
                baseline_accuracy=row["baseline_accuracy"] or 0.0,
                treatment_accuracy=row["treatment_accuracy"] or 0.0
            )
            for row in rows
        ]

    def record_ab_result(
        self,
        test_id: str,
        game_id: str,
        is_treatment: bool,
        correct: bool
    ) -> None:
        """Record a result in an A/B test."""
        query = """
        UPDATE ml.prompt_ab_tests
        SET games_tested = games_tested + 1,
            baseline_accuracy = CASE
                WHEN NOT %s THEN
                    (baseline_accuracy * games_tested +
                     CASE WHEN %s THEN 1.0 ELSE 0.0 END) / (games_tested + 1)
                ELSE baseline_accuracy
            END,
            treatment_accuracy = CASE
                WHEN %s THEN
                    (treatment_accuracy * games_tested +
                     CASE WHEN %s THEN 1.0 ELSE 0.0 END) / (games_tested + 1)
                ELSE treatment_accuracy
            END
        WHERE test_id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (is_treatment, correct, is_treatment, correct, test_id))
            conn.commit()

    def evaluate_ab_test(self, test_id: str) -> dict:
        """
        Evaluate an A/B test and determine winner.

        Returns dict with:
        - winner: 'baseline', 'treatment', or 'inconclusive'
        - p_value: Statistical significance
        - recommendation: Action to take
        """
        query = """
        SELECT
            games_tested,
            baseline_accuracy,
            treatment_accuracy
        FROM ml.prompt_ab_tests
        WHERE test_id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (test_id,))
                row = cur.fetchone()

        if not row or row["games_tested"] < 30:
            return {
                "winner": "inconclusive",
                "p_value": 1.0,
                "recommendation": "Need more data (min 30 games)",
                "games": row["games_tested"] if row else 0
            }

        baseline = row["baseline_accuracy"]
        treatment = row["treatment_accuracy"]
        games = row["games_tested"]

        # Simple comparison (could use proper statistical test)
        lift = treatment - baseline

        if lift > 0.05 and games >= 50:
            return {
                "winner": "treatment",
                "p_value": 0.05,  # Placeholder
                "recommendation": "Adopt treatment prompt",
                "lift": lift,
                "games": games
            }
        elif lift < -0.05 and games >= 50:
            return {
                "winner": "baseline",
                "p_value": 0.05,
                "recommendation": "Keep baseline prompt",
                "lift": lift,
                "games": games
            }
        else:
            return {
                "winner": "inconclusive",
                "p_value": 0.5,
                "recommendation": "Continue testing" if games < 100 else "No significant difference",
                "lift": lift,
                "games": games
            }

    def apply_winning_prompt(self, test_id: str) -> bool:
        """Apply the winning prompt from an A/B test."""
        evaluation = self.evaluate_ab_test(test_id)

        if evaluation["winner"] != "treatment":
            logger.info(f"Cannot apply - winner is {evaluation['winner']}")
            return False

        # Get the treatment version
        query = """
        SELECT feature_name, treatment_version
        FROM ml.prompt_ab_tests
        WHERE test_id = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (test_id,))
                row = cur.fetchone()

        if not row:
            return False

        feature, version = row

        # Activate treatment, deactivate baseline
        update_query = """
        UPDATE ml.prompt_versions
        SET is_active = (prompt_version = %s)
        WHERE feature_name = %s
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(update_query, (version, feature))

                # Mark test as complete
                cur.execute("""
                    UPDATE ml.prompt_ab_tests
                    SET status = 'completed', winner = 'treatment'
                    WHERE test_id = %s
                """, (test_id,))

            conn.commit()

        logger.info(f"Applied winning prompt {version} for {feature}")
        return True


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    evolver = PromptEvolver()

    print("\n=== Testing Prompt Evolver ===")

    # Analyze failures for a feature
    analysis = evolver.analyze_failures("sem_home_sentiment", weeks=8)
    print(f"\nFailure analysis for sem_home_sentiment:")
    print(f"  Total failures: {analysis.total_failures}")
    print(f"  Patterns: {analysis.failure_patterns}")
    print(f"  Common issues: {analysis.common_issues}")
    print(f"  Severity: {analysis.severity}")

    # Get suggestions
    suggestions = evolver.suggest_prompt_improvements(analysis)
    print(f"\nSuggestions ({len(suggestions)}):")
    for s in suggestions[:3]:
        print(f"  - {s.issue_addressed}")
        print(f"    Change: {s.suggested_change}")
        print(f"    Expected: {s.expected_improvement}")

    # Get active tests
    tests = evolver.get_active_tests()
    print(f"\nActive A/B tests: {len(tests)}")
