"""
Semantic Meta-Learner.

Bidirectional learning system that improves both:
- Direction 1: Model → Semantic (use outcomes to refine LLM/features)
- Direction 2: Semantic → Model (use patterns to suggest new features)

Usage:
    learner = SemanticMetaLearner()
    insights = learner.update_from_outcomes(season=2025, week=16)
    suggestions = learner.suggest_llm_improvements()
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
class FeatureInsight:
    """Insight about a semantic feature's performance."""
    feature_name: str
    mean_contribution: float
    std_contribution: float
    games_evaluated: int
    accuracy_when_high: float  # Accuracy when feature value is high
    accuracy_when_low: float   # Accuracy when feature value is low
    recommendation: str  # 'keep', 'boost', 'demote', 'drop', 'investigate'
    reason: str


@dataclass
class LLMSuggestion:
    """Suggestion for LLM improvement."""
    action: str  # 'refine_prompt', 'add_training_data', 'adjust_threshold'
    feature: str
    issue: str
    suggestion: str
    priority: str  # 'high', 'medium', 'low'
    evidence: dict = field(default_factory=dict)


@dataclass
class PatternDiscovery:
    """Discovered pattern from semantic data."""
    pattern_name: str
    description: str
    correlation_with_outcome: float
    suggested_feature: str
    sample_size: int
    confidence: float


@dataclass
class MetaLearnerResult:
    """Complete result of meta-learning analysis."""
    season: int
    week: int
    games_analyzed: int = 0
    feature_insights: list[FeatureInsight] = field(default_factory=list)
    llm_suggestions: list[LLMSuggestion] = field(default_factory=list)
    pattern_discoveries: list[PatternDiscovery] = field(default_factory=list)
    weight_suggestions: dict[str, float] = field(default_factory=dict)

    @property
    def features_tracked(self) -> int:
        """Number of features analyzed."""
        return len(self.feature_insights)

    @property
    def top_performers(self) -> list[FeatureInsight]:
        """Top performing features by contribution."""
        return sorted(
            [f for f in self.feature_insights if f.mean_contribution > 0],
            key=lambda x: x.mean_contribution,
            reverse=True
        )

    @property
    def underperformers(self) -> list[FeatureInsight]:
        """Underperforming features (negative contribution)."""
        return sorted(
            [f for f in self.feature_insights if f.mean_contribution < 0],
            key=lambda x: x.mean_contribution
        )


class SemanticMetaLearner:
    """
    Meta-learning system for bidirectional improvement.

    Direction 1: Model → Semantic
    - Track which semantic features improved predictions
    - Use outcomes to refine LLM prompts/fine-tuning
    - Drop features with zero or negative contribution

    Direction 2: Semantic → Model
    - Discover new patterns in semantic data
    - Suggest new features based on correlations
    - Refine injury severity weights based on outcomes
    """

    # Thresholds for feature recommendations
    POSITIVE_THRESHOLD = 0.02  # Mean contribution above this = positive feature
    NEGATIVE_THRESHOLD = -0.02  # Mean contribution below this = negative feature
    HIGH_VALUE_THRESHOLD = 0.5  # Feature value considered "high"
    MIN_GAMES_FOR_ANALYSIS = 20

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
        lookback_weeks: int = 4
    ):
        """Initialize meta-learner."""
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        self.lookback_weeks = lookback_weeks

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)

    def update_from_outcomes(
        self,
        season: int,
        week: int
    ) -> MetaLearnerResult:
        """
        Update meta-learner based on game outcomes.

        After game outcomes are known:
        1. Compute SHAP-like values for each semantic feature
        2. Track rolling hit rate per feature
        3. Identify features to promote/demote

        Args:
            season: Season year
            week: Week number

        Returns:
            MetaLearnerResult with insights and suggestions
        """
        result = MetaLearnerResult(season=season, week=week)

        # Get attribution data
        start_week = max(1, week - self.lookback_weeks + 1)
        attributions = self._get_attribution_data(season, start_week, week)

        if not attributions:
            logger.info("No attribution data available for meta-learning")
            return result

        result.games_analyzed = len(attributions)

        # Analyze feature performance
        feature_stats = self._compute_feature_stats(attributions)
        result.feature_insights = self._generate_feature_insights(feature_stats)

        # Generate LLM suggestions based on insights
        result.llm_suggestions = self._generate_llm_suggestions(
            result.feature_insights, attributions
        )

        # Discover new patterns
        result.pattern_discoveries = self._discover_patterns(season, week)

        # Suggest feature weights
        result.weight_suggestions = self._suggest_feature_weights(feature_stats)

        return result

    def _get_attribution_data(
        self,
        season: int,
        start_week: int,
        end_week: int
    ) -> list[dict]:
        """Get attribution data for analysis."""
        query = """
        SELECT
            game_id,
            season,
            week,
            prediction_correct,
            prediction_error,
            semantic_features,
            feature_attributions,
            total_semantic_contribution
        FROM ml.semantic_attribution
        WHERE season = %s AND week >= %s AND week <= %s
        ORDER BY week, game_id
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (season, start_week, end_week))
                return [dict(row) for row in cur.fetchall()]

    def _compute_feature_stats(
        self,
        attributions: list[dict]
    ) -> dict[str, dict]:
        """Compute statistics for each feature."""
        feature_data = {}

        for attr in attributions:
            features = attr.get("semantic_features") or {}
            contribs = attr.get("feature_attributions") or {}
            correct = attr.get("prediction_correct")

            for name, value in features.items():
                if name not in feature_data:
                    feature_data[name] = {
                        "values": [],
                        "contributions": [],
                        "correct_high": 0,
                        "total_high": 0,
                        "correct_low": 0,
                        "total_low": 0,
                    }

                feature_data[name]["values"].append(value)
                feature_data[name]["contributions"].append(
                    contribs.get(name, 0.0)
                )

                # Track accuracy by value level
                is_high = abs(value) > self.HIGH_VALUE_THRESHOLD
                if is_high:
                    feature_data[name]["total_high"] += 1
                    if correct:
                        feature_data[name]["correct_high"] += 1
                else:
                    feature_data[name]["total_low"] += 1
                    if correct:
                        feature_data[name]["correct_low"] += 1

        # Compute summary stats
        stats = {}
        for name, data in feature_data.items():
            total_high = data["total_high"] or 1
            total_low = data["total_low"] or 1

            stats[name] = {
                "mean_contribution": np.mean(data["contributions"]),
                "std_contribution": np.std(data["contributions"]),
                "mean_value": np.mean(data["values"]),
                "std_value": np.std(data["values"]),
                "games": len(data["values"]),
                "accuracy_when_high": data["correct_high"] / total_high,
                "accuracy_when_low": data["correct_low"] / total_low,
            }

        return stats

    def _generate_feature_insights(
        self,
        stats: dict[str, dict]
    ) -> list[FeatureInsight]:
        """Generate insights for each feature."""
        insights = []

        for name, s in stats.items():
            if s["games"] < self.MIN_GAMES_FOR_ANALYSIS:
                continue

            mean_c = s["mean_contribution"]
            acc_high = s["accuracy_when_high"]
            acc_low = s["accuracy_when_low"]

            # Determine recommendation
            if mean_c > self.POSITIVE_THRESHOLD:
                recommendation = "boost" if mean_c > 0.05 else "keep"
                reason = f"Positive contribution ({mean_c:.4f})"
            elif mean_c < self.NEGATIVE_THRESHOLD:
                recommendation = "drop" if mean_c < -0.05 else "demote"
                reason = f"Negative contribution ({mean_c:.4f})"
            elif abs(acc_high - acc_low) > 0.1:
                recommendation = "investigate"
                reason = f"Accuracy varies by value level ({acc_high:.1%} high vs {acc_low:.1%} low)"
            else:
                recommendation = "keep"
                reason = "Neutral contribution, stable performance"

            insights.append(FeatureInsight(
                feature_name=name,
                mean_contribution=mean_c,
                std_contribution=s["std_contribution"],
                games_evaluated=s["games"],
                accuracy_when_high=acc_high,
                accuracy_when_low=acc_low,
                recommendation=recommendation,
                reason=reason
            ))

        # Sort by absolute contribution
        insights.sort(key=lambda x: abs(x.mean_contribution), reverse=True)
        return insights

    def _generate_llm_suggestions(
        self,
        insights: list[FeatureInsight],
        attributions: list[dict]
    ) -> list[LLMSuggestion]:
        """Generate suggestions for LLM improvement."""
        suggestions = []

        for insight in insights:
            if insight.recommendation in ["drop", "demote"]:
                # Feature hurts predictions - analyze why
                issue = self._analyze_negative_feature(
                    insight.feature_name, attributions
                )
                suggestions.append(LLMSuggestion(
                    action="refine_prompt",
                    feature=insight.feature_name,
                    issue=issue,
                    suggestion=f"Reduce weight or refine extraction for {insight.feature_name}",
                    priority="high" if insight.recommendation == "drop" else "medium",
                    evidence={
                        "mean_contribution": insight.mean_contribution,
                        "games": insight.games_evaluated
                    }
                ))

            elif insight.recommendation == "investigate":
                suggestions.append(LLMSuggestion(
                    action="add_training_data",
                    feature=insight.feature_name,
                    issue=f"Inconsistent performance across value ranges",
                    suggestion=f"Add more training examples for edge cases",
                    priority="medium",
                    evidence={
                        "accuracy_high": insight.accuracy_when_high,
                        "accuracy_low": insight.accuracy_when_low
                    }
                ))

            elif insight.recommendation == "boost":
                suggestions.append(LLMSuggestion(
                    action="adjust_threshold",
                    feature=insight.feature_name,
                    issue="Strong positive contributor",
                    suggestion=f"Increase weight for {insight.feature_name}",
                    priority="medium",
                    evidence={
                        "mean_contribution": insight.mean_contribution,
                        "games": insight.games_evaluated
                    }
                ))

        return suggestions

    def _analyze_negative_feature(
        self,
        feature_name: str,
        attributions: list[dict]
    ) -> str:
        """Analyze why a feature has negative contribution."""
        # Check if feature is worse when confident (high value)
        high_value_errors = []
        low_value_errors = []

        for attr in attributions:
            features = attr.get("semantic_features") or {}
            value = features.get(feature_name, 0)
            error = attr.get("prediction_error", 0)

            if abs(value) > self.HIGH_VALUE_THRESHOLD:
                high_value_errors.append(error)
            else:
                low_value_errors.append(error)

        if high_value_errors and low_value_errors:
            high_mean = np.mean(high_value_errors)
            low_mean = np.mean(low_value_errors)

            if high_mean > low_mean * 1.2:
                return "Overconfident extraction - higher errors when feature is strong"

        return "Feature not predictive of outcomes"

    def _discover_patterns(
        self,
        season: int,
        week: int
    ) -> list[PatternDiscovery]:
        """
        Discover new patterns in semantic data.

        Direction 2: Semantic → Model
        Look for patterns not yet captured by features.
        """
        patterns = []

        # Pattern 1: Sentiment momentum (3-day trend)
        momentum_pattern = self._check_momentum_pattern(season, week)
        if momentum_pattern:
            patterns.append(momentum_pattern)

        # Pattern 2: Injury clustering (multiple injuries same week)
        injury_pattern = self._check_injury_clustering(season, week)
        if injury_pattern:
            patterns.append(injury_pattern)

        # Pattern 3: Media buzz spikes before upsets
        buzz_pattern = self._check_buzz_upset_correlation(season, week)
        if buzz_pattern:
            patterns.append(buzz_pattern)

        return patterns

    def _check_momentum_pattern(
        self,
        season: int,
        week: int
    ) -> PatternDiscovery | None:
        """Check if sentiment momentum predicts outcomes."""
        # This would require historical sentiment data
        # For now, return placeholder if we have the data
        query = """
        SELECT COUNT(*) as count
        FROM semantic.sentiment_scores
        WHERE processed_at >= NOW() - INTERVAL '7 days'
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                count = cur.fetchone()[0]

        if count > 100:
            return PatternDiscovery(
                pattern_name="sentiment_momentum",
                description="3-day sentiment trend correlates with game outcomes",
                correlation_with_outcome=0.0,  # Would compute from data
                suggested_feature="sem_sentiment_3d_momentum",
                sample_size=count,
                confidence=0.5
            )

        return None

    def _check_injury_clustering(
        self,
        season: int,
        week: int
    ) -> PatternDiscovery | None:
        """Check if multiple injuries in same week predict underperformance."""
        query = """
        SELECT team, COUNT(*) as injury_count
        FROM semantic.injury_snapshots
        WHERE updated_at >= NOW() - INTERVAL '7 days'
          AND status IN ('Out', 'Doubtful', 'IR')
        GROUP BY team
        HAVING COUNT(*) >= 3
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                teams = cur.fetchall()

        if teams:
            return PatternDiscovery(
                pattern_name="injury_clustering",
                description="Teams with 3+ serious injuries underperform",
                correlation_with_outcome=0.0,
                suggested_feature="sem_injury_cluster_indicator",
                sample_size=len(teams),
                confidence=0.6
            )

        return None

    def _check_buzz_upset_correlation(
        self,
        season: int,
        week: int
    ) -> PatternDiscovery | None:
        """Check if media buzz spikes before upsets."""
        # Would need to correlate buzz with upset outcomes
        # Placeholder for now
        return None

    def _suggest_feature_weights(
        self,
        stats: dict[str, dict]
    ) -> dict[str, float]:
        """
        Suggest feature weights based on rolling performance.

        Features with positive contribution → weight up
        Features with negative contribution → weight down or drop
        """
        weights = {}

        for name, s in stats.items():
            if s["games"] < self.MIN_GAMES_FOR_ANALYSIS:
                weights[name] = 1.0  # Default weight
                continue

            mean_c = s["mean_contribution"]

            if mean_c > 0.05:
                weights[name] = 1.5  # Boost
            elif mean_c > self.POSITIVE_THRESHOLD:
                weights[name] = 1.2  # Slight boost
            elif mean_c < -0.05:
                weights[name] = 0.0  # Drop
            elif mean_c < self.NEGATIVE_THRESHOLD:
                weights[name] = 0.5  # Demote
            else:
                weights[name] = 1.0  # Keep default

        return weights

    def compute_feature_contribution(
        self,
        game_id: str,
        prediction: dict,
        actual: dict
    ) -> dict[str, float]:
        """
        Compute marginal contribution of each semantic feature for a single game.

        Uses simplified attribution based on:
        - Feature magnitude
        - Prediction correctness

        For true SHAP, would need model and background data.
        """
        features = {k: v for k, v in prediction.items() if k.startswith("sem_")}
        if not features:
            return {}

        correct = (prediction.get("predicted_spread", 0) > 0) == (actual.get("actual_spread", 0) > 0)
        error = abs(prediction.get("predicted_spread", 0) - actual.get("actual_spread", 0))

        # Compute contributions
        contributions = {}
        magnitudes = [abs(v) for v in features.values() if v != 0]
        avg_mag = np.mean(magnitudes) if magnitudes else 1.0

        for name, value in features.items():
            if value == 0:
                contributions[name] = 0.0
                continue

            norm_mag = abs(value) / (avg_mag + 1e-6)

            if correct:
                contributions[name] = norm_mag * 0.1
            else:
                error_weight = min(error / 10.0, 1.0)
                contributions[name] = -norm_mag * error_weight * 0.1

        return contributions

    def suggest_llm_improvements(self) -> list[LLMSuggestion]:
        """
        Get LLM improvement suggestions based on recent performance.

        Convenience method that runs update_from_outcomes on recent data.
        """
        # Get current week
        query = """
        SELECT MAX(week) as week FROM games
        WHERE season = 2025 AND home_score IS NOT NULL
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                week = cur.fetchone()[0] or 1

        result = self.update_from_outcomes(2025, week)
        return result.llm_suggestions

    def discover_new_patterns(
        self,
        season: int,
        weeks: list[int]
    ) -> list[PatternDiscovery]:
        """
        Discover patterns across multiple weeks.

        Direction 2: Semantic → Model suggestions
        """
        all_patterns = []

        for week in weeks:
            patterns = self._discover_patterns(season, week)
            all_patterns.extend(patterns)

        # Deduplicate by pattern name
        unique = {}
        for p in all_patterns:
            if p.pattern_name not in unique or p.sample_size > unique[p.pattern_name].sample_size:
                unique[p.pattern_name] = p

        return list(unique.values())


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    learner = SemanticMetaLearner()

    print("\n=== Testing Semantic Meta-Learner ===")

    # Test update from outcomes
    result = learner.update_from_outcomes(2025, 13)

    print(f"\nGames analyzed: {result.games_analyzed}")

    if result.feature_insights:
        print("\nFeature Insights:")
        for insight in result.feature_insights[:5]:
            print(f"  {insight.feature_name}:")
            print(f"    Contribution: {insight.mean_contribution:.4f}")
            print(f"    Recommendation: {insight.recommendation}")
            print(f"    Reason: {insight.reason}")

    if result.llm_suggestions:
        print("\nLLM Suggestions:")
        for sug in result.llm_suggestions[:3]:
            print(f"  [{sug.priority}] {sug.action}: {sug.feature}")
            print(f"    Issue: {sug.issue}")
            print(f"    Suggestion: {sug.suggestion}")

    if result.pattern_discoveries:
        print("\nPattern Discoveries:")
        for pat in result.pattern_discoveries:
            print(f"  {pat.pattern_name}: {pat.description}")
            print(f"    Suggested feature: {pat.suggested_feature}")

    if result.weight_suggestions:
        print("\nWeight Suggestions:")
        for feat, weight in sorted(result.weight_suggestions.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat}: {weight}")
