#!/usr/bin/env python3
"""
Performance Tracking System for NFL Analytics Compute.

Tracks model performance over time, detects regressions,
calculates ROI, and provides data for adaptive scheduling.
Enhanced with statistical significance testing and formal hypothesis testing.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# Import our statistical testing framework
try:
    from .statistics.effect_size import EffectSizeCalculator
    from .statistics.multiple_comparisons import MultipleComparisonCorrection
    from .statistics.statistical_tests import BootstrapTest, PermutationTest
except ImportError:
    # Fallback if statistics module not available
    PermutationTest = None
    BootstrapTest = None
    EffectSizeCalculator = None
    MultipleComparisonCorrection = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    IMPROVING = "improving"
    REGRESSING = "regressing"
    PLATEAU = "plateau"
    VOLATILE = "volatile"


@dataclass
class PerformanceMetric:
    model_id: str
    metric_name: str
    value: float
    compute_hours: float
    timestamp: datetime
    task_id: str

    @property
    def efficiency(self) -> float:
        """Performance per compute hour."""
        return self.value / max(self.compute_hours, 0.01)


@dataclass
class StatisticalComparisonResult:
    """Results of statistical comparison between model performances."""

    comparison_name: str
    baseline_model: str
    treatment_model: str
    baseline_performance: float
    treatment_performance: float
    effect_size: float
    p_value: float
    confidence_interval: tuple[float, float]
    statistical_method: str
    is_significant: bool
    practical_significance: str
    interpretation: str


class PerformanceTracker:
    """Track and analyze model performance over time with statistical testing."""

    def __init__(self, db_path: str = "compute_queue.db"):
        self.db_path = Path(db_path)

        # Initialize statistical testing components
        self.perm_test = PermutationTest(n_permutations=5000) if PermutationTest else None
        self.boot_test = BootstrapTest(n_bootstrap=5000) if BootstrapTest else None
        self.effect_calc = EffectSizeCalculator() if EffectSizeCalculator else None
        self.mc_corrector = MultipleComparisonCorrection() if MultipleComparisonCorrection else None
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrent access during Google Drive sync
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=memory")
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
        self.conn.commit()

        self._init_performance_tables()

    def _init_performance_tables(self):
        """Initialize performance tracking tables."""
        self.conn.executescript(
            """
            -- Model performance history
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                task_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metrics TEXT NOT NULL,  -- JSON
                compute_hours_invested REAL NOT NULL,
                performance_delta REAL,  -- Change from baseline
                is_baseline BOOLEAN DEFAULT 0,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_model_timestamp
            ON model_performance(model_id, timestamp);

            -- Performance trends analysis
            CREATE TABLE IF NOT EXISTS performance_trends (
                model_type TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                trend_direction TEXT NOT NULL,
                compute_efficiency REAL,  -- Improvement per hour
                diminishing_returns_point REAL,  -- Hours where improvement plateaus
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trend_data TEXT  -- JSON with detailed trend info
            );

            -- Expected value estimates for tasks
            CREATE TABLE IF NOT EXISTS task_value_estimates (
                task_type TEXT NOT NULL,
                config_hash TEXT NOT NULL,  -- Hash of config for uniqueness
                estimated_improvement REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                compute_cost_estimate REAL,
                expected_roi REAL,
                based_on_samples INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (task_type, config_hash)
            );

            -- Statistical comparisons between models
            CREATE TABLE IF NOT EXISTS statistical_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_name TEXT NOT NULL,
                baseline_model TEXT NOT NULL,
                treatment_model TEXT NOT NULL,
                baseline_performance REAL NOT NULL,
                treatment_performance REAL NOT NULL,
                effect_size REAL NOT NULL,
                p_value REAL NOT NULL,
                confidence_interval_lower REAL NOT NULL,
                confidence_interval_upper REAL NOT NULL,
                statistical_method TEXT NOT NULL,
                is_significant BOOLEAN NOT NULL,
                practical_significance TEXT NOT NULL,
                interpretation TEXT NOT NULL,
                performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_statistical_comparisons
            ON statistical_comparisons(comparison_name, performed_at);

            -- Regression detection tests
            CREATE TABLE IF NOT EXISTS regression_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                test_statistic REAL NOT NULL,
                p_value REAL NOT NULL,
                is_regression BOOLEAN NOT NULL,
                baseline_period_start TIMESTAMP NOT NULL,
                baseline_period_end TIMESTAMP NOT NULL,
                test_period_start TIMESTAMP NOT NULL,
                test_period_end TIMESTAMP NOT NULL,
                performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_regression_tests
            ON regression_tests(model_id, performed_at);

            -- Performance milestones
            CREATE TABLE IF NOT EXISTS performance_milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                milestone_type TEXT NOT NULL,  -- 'breakthrough', 'regression', 'plateau'
                description TEXT,
                metric_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Model comparison matrix
            CREATE TABLE IF NOT EXISTS model_comparisons (
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                winner TEXT,
                margin REAL,
                significance REAL,  -- p-value
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_a, model_b, metric_name)
            );
        """
        )
        self.conn.commit()

    def record_performance(
        self, task_id: str, model_id: str, metrics: dict[str, float], compute_hours: float
    ) -> dict[str, Any]:
        """Record performance metrics for a completed task."""

        # Get baseline for this model if exists
        baseline = self._get_baseline(model_id)

        # Calculate performance delta
        performance_delta = None
        primary_metric = self._get_primary_metric(model_id, metrics)

        if baseline and primary_metric:
            baseline_value = baseline.get("metrics", {}).get(primary_metric, 0)
            # Extract scalar value from metric (handle both dict and scalar)
            current_value = metrics[primary_metric]
            if isinstance(current_value, dict):
                current_value = current_value.get("mean", current_value.get("value", 0))
            if isinstance(baseline_value, dict):
                baseline_value = baseline_value.get("mean", baseline_value.get("value", 0))

            if baseline_value:
                performance_delta = (current_value - baseline_value) / baseline_value

        # Insert performance record
        is_baseline = baseline is None

        def _record_operation():
            self.conn.execute(
                """
                INSERT INTO model_performance
                (model_id, task_id, metrics, compute_hours_invested, performance_delta, is_baseline)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    model_id,
                    task_id,
                    json.dumps(self._make_json_serializable(metrics)),
                    compute_hours,
                    performance_delta,
                    is_baseline,
                ),
            )
            self.conn.commit()

        self._execute_with_retry(_record_operation)

        # Update trends
        self._update_trends(model_id, metrics, compute_hours)

        # Check for milestones
        milestones = self._check_milestones(model_id, metrics, performance_delta)

        # Update task value estimates
        self._update_value_estimates(task_id, model_id, performance_delta, compute_hours)

        self.conn.commit()

        return {
            "performance_delta": performance_delta,
            "is_improvement": performance_delta and performance_delta > 0,
            "milestones": milestones,
            "trend": self._get_current_trend(model_id),
        }

    def _get_baseline(self, model_id: str) -> dict | None:
        """Get baseline performance for a model."""
        cursor = self.conn.execute(
            """
            SELECT * FROM model_performance
            WHERE model_id = ? AND is_baseline = 1
            ORDER BY timestamp DESC
            LIMIT 1
        """,
            (model_id,),
        )

        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["metrics"] = json.loads(result["metrics"])
            return result
        return None

    def _get_primary_metric(self, model_id: str, metrics: dict[str, float]) -> str | None:
        """Determine primary metric for a model type."""
        # Model-specific primary metrics
        metric_priority = {
            "dqn": ["final_loss", "best_loss", "avg_reward"],
            "ppo": ["avg_reward", "final_reward", "best_reward"],
            "state_space": ["log_likelihood", "accuracy", "brier_score"],
            "monte_carlo": ["sharpe", "cvar", "var"],
            "glm": ["accuracy", "brier_score", "ece"],
            "copula": ["p_value", "gof_statistic"],
        }

        # Find matching metric
        for model_type, priority_list in metric_priority.items():
            if model_type in model_id.lower():
                for metric in priority_list:
                    if metric in metrics:
                        return metric

        # Default to first metric
        return list(metrics.keys())[0] if metrics else None

    def _update_trends(self, model_id: str, metrics: dict[str, float], compute_hours: float):
        """Update performance trend analysis."""
        # Get recent performance history
        cursor = self.conn.execute(
            """
            SELECT * FROM model_performance
            WHERE model_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        """,
            (model_id,),
        )

        history = []
        for row in cursor:
            hist_metrics = json.loads(row["metrics"])
            history.append(
                {
                    "metrics": hist_metrics,
                    "compute_hours": row["compute_hours_invested"],
                    "timestamp": row["timestamp"],
                }
            )

        if len(history) < 3:
            return  # Not enough data for trend

        # Analyze trend
        primary_metric = self._get_primary_metric(model_id, metrics)
        if not primary_metric:
            return

        values = [h["metrics"].get(primary_metric, 0) for h in history[::-1]]
        hours = [h["compute_hours"] for h in history[::-1]]

        # Calculate trend direction
        if len(values) >= 5:
            recent = values[-5:]
            if all(recent[i] <= recent[i + 1] for i in range(4)):
                direction = TrendDirection.IMPROVING
            elif all(recent[i] >= recent[i + 1] for i in range(4)):
                direction = TrendDirection.REGRESSING
            elif np.std(recent) < 0.01 * np.mean(recent):
                direction = TrendDirection.PLATEAU
            else:
                direction = TrendDirection.VOLATILE
        else:
            direction = TrendDirection.VOLATILE

        # Calculate efficiency (improvement per hour)
        if len(values) >= 2:
            total_improvement = values[-1] - values[0]
            total_hours = sum(hours)
            efficiency = total_improvement / max(total_hours, 1)
        else:
            efficiency = 0

        # Find diminishing returns point
        diminishing_point = self._find_diminishing_returns(values, hours)

        # Store trend
        model_type = model_id.split("_")[0]
        trend_data = {
            "values": values[-10:],
            "hours": hours[-10:],
            "regression_slope": self._calculate_slope(hours, values),
        }

        self.conn.execute(
            """
            INSERT OR REPLACE INTO performance_trends
            (model_type, metric_name, trend_direction, compute_efficiency,
             diminishing_returns_point, trend_data, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                model_type,
                primary_metric,
                direction.value,
                efficiency,
                diminishing_point,
                json.dumps(trend_data),
            ),
        )

    def _find_diminishing_returns(self, values: list[float], hours: list[float]) -> float | None:
        """Find point where improvements start diminishing."""
        if len(values) < 5:
            return None

        # Calculate rolling improvement rate
        improvements = []
        cumulative_hours = np.cumsum(hours)

        for i in range(1, len(values)):
            if values[i] != values[i - 1] and hours[i] > 0:
                improvement_rate = (values[i] - values[i - 1]) / hours[i]
                improvements.append((cumulative_hours[i], improvement_rate))

        if len(improvements) < 3:
            return None

        # Find where improvement rate drops below 50% of peak
        peak_rate = max(imp[1] for imp in improvements)
        for cum_hours, rate in improvements:
            if rate < 0.5 * peak_rate:
                return cum_hours

        return None

    def _calculate_slope(self, x: list[float], y: list[float]) -> float:
        """Calculate linear regression slope."""
        if len(x) < 2:
            return 0
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    def compare_models_statistically(
        self,
        baseline_model: str,
        treatment_model: str,
        metric_name: str = None,
        min_observations: int = 10,
    ) -> StatisticalComparisonResult:
        """
        Perform statistical comparison between two models.

        Args:
            baseline_model: Baseline model identifier
            treatment_model: Treatment model identifier
            metric_name: Specific metric to compare (None for primary metric)
            min_observations: Minimum observations required for comparison

        Returns:
            Statistical comparison results
        """
        if not self.perm_test or not self.effect_calc:
            logger.warning("Statistical testing framework not available")
            return self._fallback_comparison(baseline_model, treatment_model, metric_name)

        # Get performance data for both models
        baseline_data = self._get_model_performance_data(baseline_model, metric_name)
        treatment_data = self._get_model_performance_data(treatment_model, metric_name)

        if len(baseline_data) < min_observations or len(treatment_data) < min_observations:
            logger.warning(
                f"Insufficient data for statistical comparison: "
                f"{len(baseline_data)} vs {len(treatment_data)} observations"
            )
            return self._fallback_comparison(baseline_model, treatment_model, metric_name)

        # Perform permutation test
        perm_result = self.perm_test.two_sample_test(
            np.array(baseline_data), np.array(treatment_data), alternative="two-sided"
        )

        # Calculate effect size
        effect_result = self.effect_calc.cohens_d(np.array(treatment_data), np.array(baseline_data))

        # Bootstrap confidence interval
        boot_result = (
            self.boot_test.two_sample_test(np.array(baseline_data), np.array(treatment_data))
            if self.boot_test
            else None
        )

        # Determine practical significance
        practical_sig = self._assess_practical_significance(effect_result.value)

        # Create result
        comparison_result = StatisticalComparisonResult(
            comparison_name=f"{treatment_model}_vs_{baseline_model}",
            baseline_model=baseline_model,
            treatment_model=treatment_model,
            baseline_performance=np.mean(baseline_data),
            treatment_performance=np.mean(treatment_data),
            effect_size=effect_result.value,
            p_value=perm_result.p_value,
            confidence_interval=(
                boot_result.confidence_interval if boot_result else (np.nan, np.nan)
            ),
            statistical_method="Permutation Test + Bootstrap CI",
            is_significant=perm_result.is_significant(),
            practical_significance=practical_sig,
            interpretation=self._interpret_comparison(perm_result, effect_result),
        )

        # Store in database
        self._store_statistical_comparison(comparison_result)

        return comparison_result

    def detect_performance_regression(
        self, model_id: str, baseline_period_days: int = 7, test_period_days: int = 7
    ) -> dict[str, Any]:
        """
        Detect statistical regression in model performance.

        Args:
            model_id: Model identifier
            baseline_period_days: Days to use for baseline period
            test_period_days: Days to use for test period

        Returns:
            Regression test results
        """
        if not self.perm_test:
            logger.warning("Statistical testing framework not available")
            return {"error": "Statistical testing not available"}

        # Define time periods
        now = datetime.now()
        test_end = now
        test_start = now - timedelta(days=test_period_days)
        baseline_end = test_start
        baseline_start = baseline_end - timedelta(days=baseline_period_days)

        # Get performance data for each period
        baseline_data = self._get_model_performance_data_period(
            model_id, baseline_start, baseline_end
        )
        test_data = self._get_model_performance_data_period(model_id, test_start, test_end)

        if len(baseline_data) < 5 or len(test_data) < 5:
            return {
                "error": "Insufficient data for regression detection",
                "baseline_count": len(baseline_data),
                "test_count": len(test_data),
            }

        # Perform one-sided permutation test (testing for decrease)
        regression_test = self.perm_test.two_sample_test(
            np.array(test_data),
            np.array(baseline_data),
            alternative="less",  # Test if test_data < baseline_data
        )

        # Calculate effect size
        effect_result = (
            self.effect_calc.cohens_d(np.array(test_data), np.array(baseline_data))
            if self.effect_calc
            else None
        )

        is_regression = regression_test.p_value < 0.05 and np.mean(test_data) < np.mean(
            baseline_data
        )

        result = {
            "model_id": model_id,
            "is_regression": is_regression,
            "p_value": regression_test.p_value,
            "test_statistic": regression_test.statistic,
            "effect_size": effect_result.value if effect_result else None,
            "baseline_mean": np.mean(baseline_data),
            "test_mean": np.mean(test_data),
            "baseline_period": (baseline_start, baseline_end),
            "test_period": (test_start, test_end),
            "confidence": 1 - regression_test.p_value if is_regression else None,
        }

        # Store regression test result
        self._store_regression_test(result)

        return result

    def compare_multiple_models(
        self,
        model_ids: list[str],
        metric_name: str = None,
        correction_method: str = "benjamini_hochberg",
    ) -> dict[str, Any]:
        """
        Compare multiple models with appropriate multiple testing correction.

        Args:
            model_ids: List of model identifiers
            metric_name: Specific metric to compare
            correction_method: Multiple comparison correction method

        Returns:
            Multiple comparison results with corrections
        """
        if not self.mc_corrector or len(model_ids) < 2:
            return {"error": "Insufficient models or correction framework unavailable"}

        # Perform pairwise comparisons
        comparisons = []
        p_values = []
        comparison_names = []

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                baseline = model_ids[i]
                treatment = model_ids[j]

                comparison = self.compare_models_statistically(
                    baseline, treatment, metric_name, min_observations=5
                )

                comparisons.append(comparison)
                p_values.append(comparison.p_value)
                comparison_names.append(comparison.comparison_name)

        # Apply multiple comparison correction
        if len(p_values) > 1:
            from .statistics.multiple_comparisons import CorrectionMethod

            method_map = {
                "benjamini_hochberg": CorrectionMethod.BENJAMINI_HOCHBERG,
                "bonferroni": CorrectionMethod.BONFERRONI,
                "holm_bonferroni": CorrectionMethod.HOLM_BONFERRONI,
            }

            correction_result = self.mc_corrector.correct(
                p_values, method_map.get(correction_method, CorrectionMethod.BENJAMINI_HOCHBERG)
            )

            # Update significance based on correction
            for i, comparison in enumerate(comparisons):
                comparison.is_significant = correction_result.rejected[i]

        return {
            "comparisons": comparisons,
            "correction_method": correction_method,
            "num_comparisons": len(comparisons),
            "num_significant": sum(1 for c in comparisons if c.is_significant),
            "family_wise_error_controlled": True,
        }

    def _get_model_performance_data(self, model_id: str, metric_name: str = None) -> list[float]:
        """Get performance data for a model."""
        cursor = self.conn.execute(
            """
            SELECT metrics FROM model_performance
            WHERE model_id = ?
            ORDER BY timestamp DESC
            LIMIT 50
        """,
            (model_id,),
        )

        data = []
        for row in cursor:
            metrics = json.loads(row["metrics"])
            if metric_name and metric_name in metrics:
                data.append(metrics[metric_name])
            else:
                # Use primary metric
                primary_metric = self._get_primary_metric(model_id, metrics)
                if primary_metric and primary_metric in metrics:
                    data.append(metrics[primary_metric])

        return data

    def _get_model_performance_data_period(
        self, model_id: str, start_time: datetime, end_time: datetime
    ) -> list[float]:
        """Get performance data for a model within a time period."""
        cursor = self.conn.execute(
            """
            SELECT metrics FROM model_performance
            WHERE model_id = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """,
            (model_id, start_time.isoformat(), end_time.isoformat()),
        )

        data = []
        for row in cursor:
            metrics = json.loads(row["metrics"])
            primary_metric = self._get_primary_metric(model_id, metrics)
            if primary_metric and primary_metric in metrics:
                data.append(metrics[primary_metric])

        return data

    def _assess_practical_significance(self, effect_size: float) -> str:
        """Assess practical significance based on effect size."""
        abs_effect = abs(effect_size)

        if abs_effect < 0.1:
            return "negligible"
        elif abs_effect < 0.3:
            return "small"
        elif abs_effect < 0.5:
            return "medium"
        elif abs_effect < 0.8:
            return "large"
        else:
            return "very_large"

    def _interpret_comparison(self, perm_result, effect_result) -> str:
        """Create interpretation of statistical comparison."""
        significance = "significant" if perm_result.is_significant() else "not significant"
        direction = "better" if effect_result.value > 0 else "worse"
        magnitude = self._assess_practical_significance(effect_result.value)

        return (
            f"Treatment model performs {direction} than baseline "
            f"({significance}, {magnitude} effect size)"
        )

    def _store_statistical_comparison(self, result: StatisticalComparisonResult):
        """Store statistical comparison result in database."""
        self.conn.execute(
            """
            INSERT INTO statistical_comparisons
            (comparison_name, baseline_model, treatment_model, baseline_performance,
             treatment_performance, effect_size, p_value, confidence_interval_lower,
             confidence_interval_upper, statistical_method, is_significant,
             practical_significance, interpretation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result.comparison_name,
                result.baseline_model,
                result.treatment_model,
                result.baseline_performance,
                result.treatment_performance,
                result.effect_size,
                result.p_value,
                result.confidence_interval[0],
                result.confidence_interval[1],
                result.statistical_method,
                result.is_significant,
                result.practical_significance,
                result.interpretation,
            ),
        )
        self.conn.commit()

    def _store_regression_test(self, result: dict[str, Any]):
        """Store regression test result in database."""
        baseline_start, baseline_end = result["baseline_period"]
        test_start, test_end = result["test_period"]

        self.conn.execute(
            """
            INSERT INTO regression_tests
            (model_id, test_type, test_statistic, p_value, is_regression,
             baseline_period_start, baseline_period_end,
             test_period_start, test_period_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result["model_id"],
                "permutation_test",
                result["test_statistic"],
                result["p_value"],
                result["is_regression"],
                baseline_start.isoformat(),
                baseline_end.isoformat(),
                test_start.isoformat(),
                test_end.isoformat(),
            ),
        )
        self.conn.commit()

    def _fallback_comparison(
        self, baseline_model: str, treatment_model: str, metric_name: str = None
    ) -> StatisticalComparisonResult:
        """Fallback comparison when statistical framework unavailable."""
        baseline_data = self._get_model_performance_data(baseline_model, metric_name)
        treatment_data = self._get_model_performance_data(treatment_model, metric_name)

        if not baseline_data or not treatment_data:
            baseline_mean = treatment_mean = 0.0
            effect_size = 0.0
            p_value = 1.0
        else:
            baseline_mean = np.mean(baseline_data)
            treatment_mean = np.mean(treatment_data)

            # Simple t-test fallback
            if len(baseline_data) > 1 and len(treatment_data) > 1:
                try:
                    t_stat, p_value = stats.ttest_ind(treatment_data, baseline_data)
                    pooled_std = np.sqrt(
                        (np.var(baseline_data, ddof=1) + np.var(treatment_data, ddof=1)) / 2
                    )
                    effect_size = (
                        (treatment_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
                    )
                except Exception:
                    p_value = 1.0
                    effect_size = 0.0
            else:
                p_value = 1.0
                effect_size = 0.0

        return StatisticalComparisonResult(
            comparison_name=f"{treatment_model}_vs_{baseline_model}",
            baseline_model=baseline_model,
            treatment_model=treatment_model,
            baseline_performance=baseline_mean,
            treatment_performance=treatment_mean,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=(np.nan, np.nan),
            statistical_method="Classical t-test (fallback)",
            is_significant=p_value < 0.05,
            practical_significance=self._assess_practical_significance(effect_size),
            interpretation=f"Fallback comparison: {'significant' if p_value < 0.05 else 'not significant'}",
        )

    def _check_milestones(
        self, model_id: str, metrics: dict[str, float], performance_delta: float | None
    ) -> list[str]:
        """Check for performance milestones."""
        milestones = []

        # Check for breakthrough (>10% improvement)
        if performance_delta and performance_delta > 0.1:
            self._record_milestone(
                model_id, "breakthrough", f"Performance improved by {performance_delta*100:.1f}%"
            )
            milestones.append("🏆 BREAKTHROUGH")

        # Check for regression (>5% drop)
        if performance_delta and performance_delta < -0.05:
            self._record_milestone(
                model_id, "regression", f"Performance dropped by {abs(performance_delta)*100:.1f}%"
            )
            milestones.append("⚠️ REGRESSION")

        # Check for new best
        primary_metric = self._get_primary_metric(model_id, metrics)
        if primary_metric:
            current_best = self._get_best_performance(model_id, primary_metric)
            # Extract scalar value from metric (handle both dict and scalar)
            metric_value = metrics[primary_metric]
            if isinstance(metric_value, dict):
                metric_value = metric_value.get("mean", metric_value.get("value", 0))

            if not current_best or metric_value > current_best:
                self._record_milestone(
                    model_id,
                    "new_best",
                    f"New best {primary_metric}: {metric_value:.4f}",
                )
                milestones.append("⭐ NEW BEST")

        return milestones

    def _record_milestone(self, model_id: str, milestone_type: str, description: str):
        """Record a performance milestone."""
        self.conn.execute(
            """
            INSERT INTO performance_milestones
            (model_id, milestone_type, description)
            VALUES (?, ?, ?)
        """,
            (model_id, milestone_type, description),
        )

    def _get_best_performance(self, model_id: str, metric_name: str) -> float | None:
        """Get best historical performance for a metric."""
        cursor = self.conn.execute(
            """
            SELECT MAX(CAST(json_extract(metrics, '$.' || ?) AS REAL)) as best
            FROM model_performance
            WHERE model_id = ?
        """,
            (metric_name, model_id),
        )

        row = cursor.fetchone()
        return row["best"] if row and row["best"] is not None else None

    def _update_value_estimates(
        self, task_id: str, model_id: str, performance_delta: float | None, compute_hours: float
    ):
        """Update expected value estimates based on results."""
        # Get task info
        cursor = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        task = cursor.fetchone()
        if not task:
            return

        task_type = task["type"]
        config = json.loads(task["config"])
        config_hash = str(hash(json.dumps(config, sort_keys=True)))

        # Calculate ROI
        if performance_delta and compute_hours > 0:
            roi = performance_delta / compute_hours
        else:
            roi = 0

        # Get historical estimates for this type
        cursor = self.conn.execute(
            """
            SELECT * FROM task_value_estimates
            WHERE task_type = ? AND config_hash = ?
        """,
            (task_type, config_hash),
        )

        existing = cursor.fetchone()

        if existing:
            # Update with exponential moving average
            alpha = 0.3
            new_roi = alpha * roi + (1 - alpha) * existing["expected_roi"]
            samples = existing["based_on_samples"] + 1
        else:
            new_roi = roi
            samples = 1

        # Insert/update estimate
        self.conn.execute(
            """
            INSERT OR REPLACE INTO task_value_estimates
            (task_type, config_hash, estimated_improvement, confidence_lower,
             confidence_upper, compute_cost_estimate, expected_roi, based_on_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_type,
                config_hash,
                performance_delta or 0,
                new_roi * 0.8,
                new_roi * 1.2,
                compute_hours,
                new_roi,
                samples,
            ),
        )

    def _get_current_trend(self, model_id: str) -> str:
        """Get current trend for a model."""
        model_type = model_id.split("_")[0]
        cursor = self.conn.execute(
            """
            SELECT trend_direction FROM performance_trends
            WHERE model_type = ?
        """,
            (model_type,),
        )

        row = cursor.fetchone()
        return row["trend_direction"] if row else "unknown"

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall performance summary with statistical insights."""
        summary = {}

        # Get best performers
        cursor = self.conn.execute(
            """
            SELECT
                model_id,
                json_extract(metrics, '$.accuracy') as accuracy,
                json_extract(metrics, '$.final_loss') as loss,
                json_extract(metrics, '$.sharpe') as sharpe,
                compute_hours_invested
            FROM model_performance
            WHERE is_baseline = 0
            ORDER BY timestamp DESC
            LIMIT 50
        """
        )

        models = {}
        for row in cursor:
            model_id = row["model_id"]
            if model_id not in models:
                models[model_id] = {
                    "accuracy": row["accuracy"],
                    "loss": row["loss"],
                    "sharpe": row["sharpe"],
                    "compute_hours": row["compute_hours_invested"],
                }

        summary["top_models"] = models

        # Get trends
        cursor = self.conn.execute("SELECT * FROM performance_trends")
        trends = {}
        for row in cursor:
            trends[row["model_type"]] = {
                "direction": row["trend_direction"],
                "efficiency": row["compute_efficiency"],
                "diminishing_point": row["diminishing_returns_point"],
            }
        summary["trends"] = trends

        # Get recent milestones
        cursor = self.conn.execute(
            """
            SELECT * FROM performance_milestones
            ORDER BY timestamp DESC
            LIMIT 10
        """
        )

        milestones = []
        for row in cursor:
            milestones.append(
                {
                    "model": row["model_id"],
                    "type": row["milestone_type"],
                    "description": row["description"],
                }
            )
        summary["recent_milestones"] = milestones

        # Get recent statistical comparisons
        cursor = self.conn.execute(
            """
            SELECT comparison_name, baseline_model, treatment_model,
                   effect_size, p_value, is_significant, practical_significance,
                   interpretation, performed_at
            FROM statistical_comparisons
            ORDER BY performed_at DESC
            LIMIT 10
        """
        )

        comparisons = []
        for row in cursor:
            comparisons.append(
                {
                    "name": row["comparison_name"],
                    "baseline": row["baseline_model"],
                    "treatment": row["treatment_model"],
                    "effect_size": row["effect_size"],
                    "p_value": row["p_value"],
                    "significant": row["is_significant"],
                    "practical": row["practical_significance"],
                    "interpretation": row["interpretation"],
                    "date": row["performed_at"],
                }
            )
        summary["recent_comparisons"] = comparisons

        # Get regression alerts
        cursor = self.conn.execute(
            """
            SELECT model_id, p_value, is_regression, performed_at
            FROM regression_tests
            WHERE is_regression = 1
            ORDER BY performed_at DESC
            LIMIT 5
        """
        )

        regressions = []
        for row in cursor:
            regressions.append(
                {"model": row["model_id"], "p_value": row["p_value"], "date": row["performed_at"]}
            )
        summary["regression_alerts"] = regressions

        return summary

    def run_automated_analysis(self) -> dict[str, Any]:
        """
        Run automated statistical analysis across all models.

        Returns:
            Comprehensive analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "regression_tests": [],
            "model_comparisons": [],
            "recommendations": [],
        }

        # Get all active models
        cursor = self.conn.execute(
            """
            SELECT DISTINCT model_id FROM model_performance
            WHERE timestamp > datetime('now', '-7 days')
        """
        )

        active_models = [row["model_id"] for row in cursor]

        # Run regression tests for each model
        for model_id in active_models:
            regression_result = self.detect_performance_regression(model_id)
            if "error" not in regression_result:
                results["regression_tests"].append(regression_result)

                if regression_result["is_regression"]:
                    results["recommendations"].append(
                        {
                            "type": "regression_alert",
                            "model": model_id,
                            "action": "investigate performance drop",
                            "priority": "high",
                        }
                    )

        # Run pairwise comparisons for models of similar types
        model_types = {}
        for model_id in active_models:
            model_type = model_id.split("_")[0]
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(model_id)

        for model_type, models in model_types.items():
            if len(models) > 1:
                comparison_result = self.compare_multiple_models(models)
                if "error" not in comparison_result:
                    results["model_comparisons"].append(
                        {"model_type": model_type, "results": comparison_result}
                    )

        # Generate recommendations based on analysis
        if not results["regression_tests"] and not results["model_comparisons"]:
            results["recommendations"].append(
                {
                    "type": "info",
                    "action": "insufficient data for statistical analysis",
                    "priority": "low",
                }
            )

        return results

    def _execute_with_retry(self, operation, *args, max_retries=3, **kwargs):
        """Execute database operation with retry logic for handling locks."""
        import time

        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    wait_time = (2**attempt) * 0.1  # Exponential backoff
                    logger.warning(
                        f"Database locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)

    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Test the enhanced tracker with statistical capabilities
    tracker = PerformanceTracker()

    # Simulate recording performance for multiple models
    models_data = [
        ("dqn_baseline", {"final_loss": 0.234, "accuracy": 0.823}, 2.5),
        ("dqn_improved", {"final_loss": 0.198, "accuracy": 0.856}, 3.2),
        ("ppo_baseline", {"avg_reward": 150.3, "final_reward": 180.2}, 4.1),
        ("ppo_improved", {"avg_reward": 165.8, "final_reward": 195.6}, 4.8),
    ]

    for model_id, metrics, compute_hours in models_data:
        result = tracker.record_performance(
            task_id=f"test-{model_id}",
            model_id=model_id,
            metrics=metrics,
            compute_hours=compute_hours,
        )
        print(f"Performance recorded for {model_id}: {result}")

    # Test statistical comparison
    if tracker.perm_test:
        print("\n=== Statistical Comparison ===")
        comparison = tracker.compare_models_statistically(
            "dqn_baseline", "dqn_improved", "accuracy"
        )
        print(f"Comparison: {comparison.interpretation}")
        print(f"Effect size: {comparison.effect_size:.3f} ({comparison.practical_significance})")
        print(f"P-value: {comparison.p_value:.3f}")

        # Test multiple model comparison
        print("\n=== Multiple Model Comparison ===")
        multi_comparison = tracker.compare_multiple_models(
            ["dqn_baseline", "dqn_improved"], "accuracy"
        )
        print(f"Multiple comparison results: {multi_comparison}")

    # Run automated analysis
    print("\n=== Automated Analysis ===")
    analysis = tracker.run_automated_analysis()
    print(f"Analysis results: {json.dumps(analysis, indent=2, default=str)}")

    # Get enhanced summary
    summary = tracker.get_performance_summary()
    print("\n=== Enhanced Performance Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    tracker.close()
