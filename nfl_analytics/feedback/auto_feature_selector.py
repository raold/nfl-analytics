"""
Automatic Feature Selector.

Automatically adjusts which semantic features are active based on performance.
Features flow through states: probation → active → disabled.

Usage:
    selector = AutoFeatureSelector()
    selection = selector.evaluate_features(season=2025, week=16)
    selector.apply_selection(selection)
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nfl_analytics.feedback.feature_memory import FeaturePerformanceMemory

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection evaluation."""

    season: int
    week: int
    active: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    probation: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)  # Just moved to active
    demoted: list[str] = field(default_factory=list)  # Just moved to disabled/probation
    suggested_new: list[str] = field(default_factory=list)  # New features from meta-learner
    changes_made: int = 0


class AutoFeatureSelector:
    """
    Automatically adjust which semantic features are active.

    Feature States:
    - probation: New feature, recorded but not used in predictions
    - active: Full weight in predictions
    - disabled: Zero weight (negative contribution detected)

    Rules:
    - Features with 4+ weeks negative contribution → disable
    - Features with 2+ weeks strong positive → promote from probation
    - New features start in probation
    """

    # Thresholds
    NEGATIVE_WEEKS_TO_DISABLE = 4  # Consecutive weeks with negative contribution
    POSITIVE_WEEKS_TO_PROMOTE = 2  # Consecutive weeks with positive contribution
    CONTRIBUTION_POSITIVE = 0.02  # Threshold for "positive" contribution
    CONTRIBUTION_NEGATIVE = -0.02  # Threshold for "negative" contribution
    MIN_SAMPLES_TO_EVALUATE = 20  # Minimum games to evaluate a feature

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
        self.memory = FeaturePerformanceMemory(**self.conn_params)

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)

    def get_current_states(self) -> dict[str, dict]:
        """
        Get current feature selection states.

        Returns dict of feature_name → {status, weight_multiplier, weeks_active, ...}
        """
        query = """
        SELECT
            feature_name,
            status,
            weight_multiplier,
            rolling_contribution,
            weeks_active,
            weeks_since_change,
            last_status_change,
            change_reason
        FROM ml.feature_selection_state
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()

        return {row["feature_name"]: dict(row) for row in rows}

    def evaluate_features(
        self, season: int, week: int, lookback_weeks: int = 4
    ) -> FeatureSelectionResult:
        """
        Evaluate all features and determine which should be active/disabled.

        Args:
            season: Current season
            week: Current week
            lookback_weeks: Weeks to look back for performance

        Returns:
            FeatureSelectionResult with updated selections
        """
        result = FeatureSelectionResult(season=season, week=week)

        # Get current states
        current_states = self.get_current_states()

        # Get feature performance from memory
        rankings = self.memory.get_feature_rankings(weeks=lookback_weeks)

        if rankings.empty:
            # No data - return current states
            for name, state in current_states.items():
                if state["status"] == "active":
                    result.active.append(name)
                elif state["status"] == "disabled":
                    result.disabled.append(name)
                else:
                    result.probation.append(name)
            return result

        # Evaluate each feature
        for _, row in rankings.iterrows():
            name = row["feature_name"]
            mean_contrib = row["mean_contribution"] or 0.0
            sample_size = row["sample_size"]

            current = current_states.get(name, {})
            current_status = current.get("status", "probation")
            weeks_active = current.get("weeks_active", 0)

            # Skip if insufficient data
            if sample_size < self.MIN_SAMPLES_TO_EVALUATE:
                if name not in current_states:
                    result.probation.append(name)
                else:
                    # Keep current status
                    if current_status == "active":
                        result.active.append(name)
                    elif current_status == "disabled":
                        result.disabled.append(name)
                    else:
                        result.probation.append(name)
                continue

            # Apply rules
            new_status = self._determine_status(mean_contrib, current_status, weeks_active)

            # Track changes
            if new_status != current_status:
                result.changes_made += 1
                if new_status == "active" and current_status == "probation":
                    result.promoted.append(name)
                elif new_status == "disabled" and current_status in ["active", "probation"]:
                    result.demoted.append(name)
                elif new_status == "probation" and current_status == "active":
                    result.demoted.append(name)

            # Add to appropriate list
            if new_status == "active":
                result.active.append(name)
            elif new_status == "disabled":
                result.disabled.append(name)
            else:
                result.probation.append(name)

        # Check for new features suggested by meta-learner
        result.suggested_new = self._get_suggested_new_features()

        return result

    def _determine_status(
        self, mean_contribution: float, current_status: str, weeks_active: int
    ) -> str:
        """
        Determine what status a feature should have.

        Rules:
        - Strong negative → disable
        - Strong positive + probation → promote to active
        - Moderate negative + long active → move to probation
        - Otherwise → keep current
        """
        # Strong negative contribution → disable
        if mean_contribution < self.CONTRIBUTION_NEGATIVE * 2:
            return "disabled"

        # Strong positive contribution
        if mean_contribution > self.CONTRIBUTION_POSITIVE * 2:
            # Promote probation to active
            if current_status == "probation":
                return "active"
            return "active"

        # Moderate positive contribution
        if mean_contribution > self.CONTRIBUTION_POSITIVE:
            if current_status == "probation" and weeks_active >= self.POSITIVE_WEEKS_TO_PROMOTE:
                return "active"
            return current_status if current_status != "disabled" else "probation"

        # Moderate negative contribution
        if mean_contribution < self.CONTRIBUTION_NEGATIVE:
            if current_status == "active" and weeks_active >= self.NEGATIVE_WEEKS_TO_DISABLE:
                return "probation"
            elif current_status == "probation":
                return "disabled"
            return current_status

        # Neutral contribution - keep current or default to probation
        if current_status == "disabled":
            return "probation"  # Give it another chance
        return current_status

    def _get_suggested_new_features(self) -> list[str]:
        """Get new features suggested by meta-learner pattern discovery."""
        query = """
        SELECT DISTINCT suggested_feature
        FROM ml.prompt_versions
        WHERE is_active = FALSE
          AND created_at >= NOW() - INTERVAL '7 days'
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return [row[0] for row in cur.fetchall()]
        except Exception:
            return []

    def apply_selection(self, selection: FeatureSelectionResult, dry_run: bool = False) -> None:
        """
        Apply feature selection to database.

        Updates ml.feature_selection_state with new statuses.
        """
        if dry_run:
            logger.info("DRY RUN - not applying changes")
            self._log_selection(selection)
            return

        # Prepare updates
        updates = []

        for name in selection.active:
            updates.append((name, "active", 1.0, selection.season, selection.week))

        for name in selection.disabled:
            updates.append((name, "disabled", 0.0, selection.season, selection.week))

        for name in selection.probation:
            updates.append((name, "probation", 0.5, selection.season, selection.week))

        # Apply updates
        query = """
        INSERT INTO ml.feature_selection_state
            (feature_name, status, weight_multiplier, weeks_active, updated_at)
        VALUES (%s, %s, %s,
            COALESCE((SELECT weeks_active FROM ml.feature_selection_state WHERE feature_name = %s), 0) + 1,
            NOW())
        ON CONFLICT (feature_name) DO UPDATE SET
            status = EXCLUDED.status,
            weight_multiplier = EXCLUDED.weight_multiplier,
            weeks_active = CASE
                WHEN ml.feature_selection_state.status = EXCLUDED.status
                THEN ml.feature_selection_state.weeks_active + 1
                ELSE 1
            END,
            weeks_since_change = CASE
                WHEN ml.feature_selection_state.status = EXCLUDED.status
                THEN ml.feature_selection_state.weeks_since_change + 1
                ELSE 0
            END,
            last_status_change = CASE
                WHEN ml.feature_selection_state.status != EXCLUDED.status
                THEN NOW()
                ELSE ml.feature_selection_state.last_status_change
            END,
            change_reason = CASE
                WHEN ml.feature_selection_state.status != EXCLUDED.status
                THEN 'Auto-selected based on Week ' || %s::TEXT || ' performance'
                ELSE ml.feature_selection_state.change_reason
            END,
            updated_at = NOW()
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for name, status, weight, season, week in updates:
                    cur.execute(query, (name, status, weight, name, week))
            conn.commit()

        # Also update rolling contributions
        self._update_rolling_contributions()

        logger.info(f"Applied {len(updates)} feature selection updates")
        self._log_selection(selection)

    def _update_rolling_contributions(self) -> None:
        """Update rolling contribution values in feature_selection_state."""
        query = """
        UPDATE ml.feature_selection_state fs
        SET rolling_contribution = sub.avg_contrib
        FROM (
            SELECT
                feature_name,
                AVG(marginal_contribution) as avg_contrib
            FROM ml.feature_performance
            WHERE created_at >= NOW() - INTERVAL '4 weeks'
            GROUP BY feature_name
        ) sub
        WHERE fs.feature_name = sub.feature_name
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()

    def _log_selection(self, selection: FeatureSelectionResult) -> None:
        """Log selection summary."""
        logger.info(f"Feature Selection for {selection.season} Week {selection.week}:")
        logger.info(f"  Active: {len(selection.active)} features")
        logger.info(f"  Disabled: {len(selection.disabled)} features")
        logger.info(f"  Probation: {len(selection.probation)} features")

        if selection.promoted:
            logger.info(f"  Promoted: {selection.promoted}")
        if selection.demoted:
            logger.info(f"  Demoted: {selection.demoted}")
        if selection.suggested_new:
            logger.info(f"  Suggested new: {selection.suggested_new}")

    def get_active_features(self) -> list[str]:
        """Get list of currently active features."""
        query = """
        SELECT feature_name
        FROM ml.feature_selection_state
        WHERE status = 'active'
        ORDER BY weight_multiplier DESC, feature_name
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [row[0] for row in cur.fetchall()]

    def get_feature_weights(self) -> dict[str, float]:
        """Get weight multipliers for all features."""
        query = """
        SELECT feature_name, weight_multiplier
        FROM ml.feature_selection_state
        """

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return {row["feature_name"]: row["weight_multiplier"] for row in cur.fetchall()}

    def manually_set_status(self, feature_name: str, status: str, reason: str) -> None:
        """
        Manually set a feature's status.

        For overriding automatic selection when needed.
        """
        if status not in ["active", "disabled", "probation"]:
            raise ValueError(f"Invalid status: {status}")

        weight = 1.0 if status == "active" else 0.0 if status == "disabled" else 0.5

        query = """
        INSERT INTO ml.feature_selection_state
            (feature_name, status, weight_multiplier, last_status_change, change_reason, updated_at)
        VALUES (%s, %s, %s, NOW(), %s, NOW())
        ON CONFLICT (feature_name) DO UPDATE SET
            status = EXCLUDED.status,
            weight_multiplier = EXCLUDED.weight_multiplier,
            last_status_change = NOW(),
            change_reason = EXCLUDED.change_reason,
            weeks_since_change = 0,
            updated_at = NOW()
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (feature_name, status, weight, f"Manual: {reason}"))
            conn.commit()

        logger.info(f"Manually set {feature_name} to {status}: {reason}")

    def generate_report(self) -> dict:
        """Generate a comprehensive report of feature selection state."""
        states = self.get_current_states()

        active = [name for name, s in states.items() if s["status"] == "active"]
        disabled = [name for name, s in states.items() if s["status"] == "disabled"]
        probation = [name for name, s in states.items() if s["status"] == "probation"]

        # Get performance stats for each
        performance = {}
        for name in states:
            stats = self.memory.get_feature_stats(name, weeks=4)
            performance[name] = {
                "mean_contribution": stats.mean_contribution,
                "sample_size": stats.sample_size,
                "recommendation": stats.recommendation,
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "active_count": len(active),
                "disabled_count": len(disabled),
                "probation_count": len(probation),
            },
            "active_features": active,
            "disabled_features": disabled,
            "probation_features": probation,
            "performance": performance,
        }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    selector = AutoFeatureSelector()

    print("\n=== Testing Auto Feature Selector ===")

    # Get current states
    states = selector.get_current_states()
    print(f"\nCurrent feature states: {len(states)} features tracked")

    # Evaluate features
    result = selector.evaluate_features(2025, 16)
    print(f"\nEvaluation for Week 16:")
    print(f"  Active: {len(result.active)}")
    print(f"  Disabled: {len(result.disabled)}")
    print(f"  Probation: {len(result.probation)}")
    print(f"  Promoted: {result.promoted}")
    print(f"  Demoted: {result.demoted}")
    print(f"  Changes: {result.changes_made}")

    # Generate report
    report = selector.generate_report()
    print(f"\nReport summary: {report['summary']}")
