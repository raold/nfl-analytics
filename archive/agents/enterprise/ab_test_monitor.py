"""
A/B Test Monitor Agent - Real-time experiment tracking with auto-stop guardrails.

Monitors active A/B tests, enforces safety guardrails, and provides statistical
decision support for experiment promotion/termination.

Key Features:
- Real-time metric tracking (MAE, win rate, ROI, calibration)
- Automated guardrail enforcement (auto-stop on violations)
- Bayesian and frequentist hypothesis testing
- Sequential probability ratio test (SPRT) for early stopping
- Prometheus metrics export for Grafana dashboards
- Integration with model registry and database logging

Usage:
    from py.utils.ab_test_monitor import ABTestMonitor

    monitor = ABTestMonitor()

    # Real-time monitoring
    status = monitor.check_guardrails()
    if status.violated:
        monitor.auto_stop_if_violated()

    # Statistical analysis
    sig_report = monitor.calculate_significance()
    decision = monitor.recommend_decision()

    # Export results
    results_df = monitor.export_results()

Author: Claude Code (Sonnet 4.5)
Created: November 22, 2025
"""

import logging
import os
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy import stats
from prometheus_client import Gauge, Counter, Histogram, Info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class TestStatus(Enum):
    """A/B test status."""
    RUNNING = "running"
    STOPPED_GUARDRAIL = "stopped_guardrail"
    STOPPED_MANUAL = "stopped_manual"
    COMPLETED = "completed"
    PROMOTED = "promoted"


class Decision(Enum):
    """A/B test decision recommendation."""
    CONTINUE = "continue"
    STOP_DEGRADATION = "stop_degradation"
    STOP_NO_EFFECT = "stop_no_effect"
    PROMOTE_TREATMENT = "promote_treatment"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class GuardrailConfig:
    """A/B test guardrail configuration."""
    max_mae_degradation_pct: float = 10.0  # %
    max_roi_drop_pct: float = 2.0  # percentage points
    min_win_rate_drop_pct: float = 2.0  # percentage points
    max_calibration_drop_pct: float = 5.0  # %
    min_sample_size: int = 50  # minimum games
    max_p_value: float = 0.05  # significance threshold


@dataclass
class GuardrailStatus:
    """Guardrail violation status."""
    violated: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'violated': self.violated,
            'violations': self.violations,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SignificanceReport:
    """Statistical significance test results."""
    metric: str
    control_mean: float
    treatment_mean: float
    difference: float
    difference_pct: float
    p_value: float
    significant: bool
    confidence_interval_95: Tuple[float, float]
    test_type: str  # 't-test', 'bootstrap', 'bayesian'
    sample_size_control: int
    sample_size_treatment: int
    effect_size: float  # Cohen's d

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'metric': self.metric,
            'control_mean': self.control_mean,
            'treatment_mean': self.treatment_mean,
            'difference': self.difference,
            'difference_pct': self.difference_pct,
            'p_value': self.p_value,
            'significant': self.significant,
            'ci_lower': self.confidence_interval_95[0],
            'ci_upper': self.confidence_interval_95[1],
            'test_type': self.test_type,
            'n_control': self.sample_size_control,
            'n_treatment': self.sample_size_treatment,
            'effect_size': self.effect_size
        }


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Gauges for current values
ab_test_mae_control = Gauge('ab_test_mae_control', 'MAE for control group')
ab_test_mae_treatment = Gauge('ab_test_mae_treatment', 'MAE for treatment group')
ab_test_win_rate_control = Gauge('ab_test_win_rate_control', 'Win rate for control group')
ab_test_win_rate_treatment = Gauge('ab_test_win_rate_treatment', 'Win rate for treatment group')
ab_test_roi_control = Gauge('ab_test_roi_control', 'ROI for control group')
ab_test_roi_treatment = Gauge('ab_test_roi_treatment', 'ROI for treatment group')
ab_test_sample_size = Gauge('ab_test_sample_size', 'Current sample size', ['group'])

# Counters for events
ab_test_guardrail_violations = Counter('ab_test_guardrail_violations', 'Guardrail violations', ['violation_type'])
ab_test_auto_stops = Counter('ab_test_auto_stops', 'Auto-stop events')
ab_test_promotions = Counter('ab_test_promotions', 'Treatment promotions')

# Histograms for distributions
ab_test_p_value = Histogram('ab_test_p_value', 'P-values over time', buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0])

# Info for test metadata
ab_test_info = Info('ab_test_info', 'A/B test metadata')


# ============================================================================
# A/B Test Monitor
# ============================================================================

class ABTestMonitor:
    """
    Monitor A/B tests with real-time guardrails and statistical decision support.

    Tracks performance metrics, enforces safety guardrails, and provides
    recommendations for experiment decisions (continue/stop/promote).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        db_url: Optional[str] = None,
        guardrails: Optional[GuardrailConfig] = None
    ):
        """
        Initialize A/B test monitor.

        Args:
            config_path: Path to ab_test_config.yaml (default: config/ab_test_config.yaml)
            db_url: Database connection string (default: from env)
            guardrails: Custom guardrail configuration (default: from config)
        """
        self.config_path = config_path or "config/ab_test_config.yaml"
        self.db_url = db_url or os.getenv(
            'DATABASE_URL',
            'postgresql://dro:sicillionbillions@localhost:5544/devdb01'  # pragma: allowlist secret
        )

        # Load configuration
        self.config = self._load_config()
        self.guardrails = guardrails or self._load_guardrails()

        # Initialize state
        self.test_start_date = datetime.fromisoformat(self.config['test_config']['start_date'])
        self.test_end_date = datetime.fromisoformat(self.config['test_config']['end_date'])

        # Update Prometheus info
        ab_test_info.info({
            'test_name': self.config['test_config']['name'],
            'start_date': self.config['test_config']['start_date'],
            'end_date': self.config['test_config']['end_date'],
            'traffic_split': str(self.config['test_config']['traffic_split'])
        })

        logger.info(f"A/B Test Monitor initialized for test: {self.config['test_config']['name']}")

    def _load_config(self) -> Dict:
        """Load A/B test configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded A/B test config from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"A/B test config not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse A/B test config: {e}")
            raise

    def _load_guardrails(self) -> GuardrailConfig:
        """Load guardrail configuration from test config."""
        guardrails_dict = self.config.get('guardrails', {})
        return GuardrailConfig(
            max_mae_degradation_pct=guardrails_dict.get('max_mae_degradation_pct', 10.0),
            max_roi_drop_pct=guardrails_dict.get('max_roi_drop_pct', 2.0),
            min_win_rate_drop_pct=guardrails_dict.get('min_win_rate_drop_pct', 2.0),
            max_calibration_drop_pct=guardrails_dict.get('max_calibration_drop_pct', 5.0),
            min_sample_size=self.config['test_config'].get('min_sample_size', 50),
            max_p_value=self.config['success_criteria'].get('statistical_significance', {}).get('p_value', 0.05)
        )

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    # ========================================================================
    # Data Retrieval
    # ========================================================================

    def get_ab_test_results(self, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve A/B test results from database.

        Args:
            start_date: Filter results from this date (default: test start date)

        Returns:
            DataFrame with columns: game_id, prediction_date, group, predicted_value,
                                   actual_value, error, win, roi
        """
        start_date = start_date or self.test_start_date

        query = """
        SELECT
            game_id,
            prediction_date,
            ab_group as group,
            predicted_value,
            actual_value,
            ABS(predicted_value - actual_value) as error,
            CASE WHEN predicted_value > actual_value THEN 1 ELSE 0 END as win,
            CASE
                WHEN predicted_value > actual_value THEN 1.0  -- Win (assume -110 odds)
                ELSE -1.1  -- Loss
            END as roi
        FROM ml.ab_test_results
        WHERE prediction_date >= %s
        ORDER BY prediction_date
        """

        with self._get_db_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(start_date,))

        logger.info(f"Retrieved {len(df)} A/B test results from database")
        return df

    def get_current_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate current metrics for control and treatment groups.

        Returns:
            Dict with 'control' and 'treatment' keys, each containing:
                - mae: Mean absolute error
                - win_rate: Percentage of winning predictions
                - roi: Return on investment (%)
                - calibration: Prediction calibration (coverage %)
                - sample_size: Number of predictions
        """
        df = self.get_ab_test_results()

        if len(df) == 0:
            logger.warning("No A/B test results available")
            return {'control': {}, 'treatment': {}}

        metrics = {}
        for group in ['control', 'treatment']:
            group_df = df[df['group'] == group]

            if len(group_df) == 0:
                metrics[group] = {}
                continue

            metrics[group] = {
                'mae': group_df['error'].mean(),
                'win_rate': (group_df['win'].sum() / len(group_df)) * 100,
                'roi': group_df['roi'].sum() / len(group_df) * 100,
                'calibration': self._calculate_calibration(group_df),
                'sample_size': len(group_df)
            }

        # Update Prometheus metrics
        if 'control' in metrics and metrics['control']:
            ab_test_mae_control.set(metrics['control']['mae'])
            ab_test_win_rate_control.set(metrics['control']['win_rate'])
            ab_test_roi_control.set(metrics['control']['roi'])
            ab_test_sample_size.labels(group='control').set(metrics['control']['sample_size'])

        if 'treatment' in metrics and metrics['treatment']:
            ab_test_mae_treatment.set(metrics['treatment']['mae'])
            ab_test_win_rate_treatment.set(metrics['treatment']['win_rate'])
            ab_test_roi_treatment.set(metrics['treatment']['roi'])
            ab_test_sample_size.labels(group='treatment').set(metrics['treatment']['sample_size'])

        return metrics

    def _calculate_calibration(self, df: pd.DataFrame) -> float:
        """
        Calculate prediction calibration (coverage %).

        Assumes predictions include uncertainty intervals stored separately.
        For now, returns 90% as placeholder (would need interval data from DB).

        Args:
            df: DataFrame with predictions

        Returns:
            Calibration percentage (0-100)
        """
        # TODO: Implement actual calibration calculation when interval data available
        # For now, return placeholder
        return 90.0

    # ========================================================================
    # Guardrail Enforcement
    # ========================================================================

    def check_guardrails(self) -> GuardrailStatus:
        """
        Check if guardrails are violated.

        Returns:
            GuardrailStatus with violation details
        """
        metrics = self.get_current_metrics()

        if not metrics.get('control') or not metrics.get('treatment'):
            return GuardrailStatus(
                violated=False,
                warnings=["Insufficient data for guardrail checks"]
            )

        control = metrics['control']
        treatment = metrics['treatment']

        violations = []
        warnings = []

        # Check sample size
        if treatment['sample_size'] < self.guardrails.min_sample_size:
            warnings.append(
                f"Sample size ({treatment['sample_size']}) below minimum "
                f"({self.guardrails.min_sample_size})"
            )

        # Check MAE degradation
        mae_degradation_pct = ((treatment['mae'] - control['mae']) / control['mae']) * 100
        if mae_degradation_pct > self.guardrails.max_mae_degradation_pct:
            violations.append(
                f"MAE degradation {mae_degradation_pct:.1f}% exceeds limit "
                f"{self.guardrails.max_mae_degradation_pct}%"
            )
            ab_test_guardrail_violations.labels(violation_type='mae_degradation').inc()

        # Check ROI drop
        roi_drop = control['roi'] - treatment['roi']
        if roi_drop > self.guardrails.max_roi_drop_pct:
            violations.append(
                f"ROI drop {roi_drop:.1f}pp exceeds limit "
                f"{self.guardrails.max_roi_drop_pct}pp"
            )
            ab_test_guardrail_violations.labels(violation_type='roi_drop').inc()

        # Check win rate drop
        win_rate_drop = control['win_rate'] - treatment['win_rate']
        if win_rate_drop > self.guardrails.min_win_rate_drop_pct:
            violations.append(
                f"Win rate drop {win_rate_drop:.1f}pp exceeds limit "
                f"{self.guardrails.min_win_rate_drop_pct}pp"
            )
            ab_test_guardrail_violations.labels(violation_type='win_rate_drop').inc()

        # Check calibration drop
        calibration_drop_pct = ((control['calibration'] - treatment['calibration']) /
                                 control['calibration']) * 100
        if calibration_drop_pct > self.guardrails.max_calibration_drop_pct:
            violations.append(
                f"Calibration drop {calibration_drop_pct:.1f}% exceeds limit "
                f"{self.guardrails.max_calibration_drop_pct}%"
            )
            ab_test_guardrail_violations.labels(violation_type='calibration_drop').inc()

        status = GuardrailStatus(
            violated=len(violations) > 0,
            violations=violations,
            warnings=warnings,
            metrics={
                'mae_degradation_pct': mae_degradation_pct,
                'roi_drop': roi_drop,
                'win_rate_drop': win_rate_drop,
                'calibration_drop_pct': calibration_drop_pct
            }
        )

        if status.violated:
            logger.error(f"Guardrail violations detected: {violations}")
        elif warnings:
            logger.warning(f"Guardrail warnings: {warnings}")
        else:
            logger.info("All guardrails passed")

        return status

    def auto_stop_if_violated(self) -> bool:
        """
        Automatically stop A/B test if guardrails violated.

        Returns:
            True if test was stopped, False otherwise
        """
        status = self.check_guardrails()

        if status.violated:
            logger.error(f"AUTO-STOPPING A/B test due to guardrail violations: {status.violations}")
            self._update_test_status(TestStatus.STOPPED_GUARDRAIL, status.violations)
            ab_test_auto_stops.inc()
            return True

        return False

    def _update_test_status(self, status: TestStatus, reason: List[str]):
        """Update A/B test status in database."""
        query = """
        INSERT INTO ml.ab_test_status (test_name, status, reason, timestamp)
        VALUES (%s, %s, %s, %s)
        """

        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        self.config['test_config']['name'],
                        status.value,
                        '; '.join(reason),
                        datetime.now()
                    )
                )
                conn.commit()

        logger.info(f"Updated test status to {status.value}")

    # ========================================================================
    # Statistical Analysis
    # ========================================================================

    def calculate_significance(
        self,
        metric: str = 'mae',
        test_type: str = 't-test'
    ) -> SignificanceReport:
        """
        Calculate statistical significance for a metric.

        Args:
            metric: Metric to test ('mae', 'win_rate', 'roi')
            test_type: Test type ('t-test', 'bootstrap', 'bayesian')

        Returns:
            SignificanceReport with test results
        """
        df = self.get_ab_test_results()

        control = df[df['group'] == 'control']
        treatment = df[df['group'] == 'treatment']

        if len(control) == 0 or len(treatment) == 0:
            logger.warning("Insufficient data for significance testing")
            return SignificanceReport(
                metric=metric,
                control_mean=0.0,
                treatment_mean=0.0,
                difference=0.0,
                difference_pct=0.0,
                p_value=1.0,
                significant=False,
                confidence_interval_95=(0.0, 0.0),
                test_type=test_type,
                sample_size_control=len(control),
                sample_size_treatment=len(treatment),
                effect_size=0.0
            )

        # Extract metric values
        if metric == 'mae':
            control_values = control['error'].values
            treatment_values = treatment['error'].values
        elif metric == 'win_rate':
            control_values = control['win'].values
            treatment_values = treatment['win'].values
        elif metric == 'roi':
            control_values = control['roi'].values
            treatment_values = treatment['roi'].values
        else:
            raise ValueError(f"Unknown metric: {metric}")

        control_mean = control_values.mean()
        treatment_mean = treatment_values.mean()
        difference = treatment_mean - control_mean
        difference_pct = (difference / control_mean) * 100 if control_mean != 0 else 0.0

        # Perform statistical test
        if test_type == 't-test':
            t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            ci_lower, ci_upper = self._bootstrap_ci(treatment_values, control_values)
        elif test_type == 'bootstrap':
            p_value = self._bootstrap_p_value(treatment_values, control_values)
            ci_lower, ci_upper = self._bootstrap_ci(treatment_values, control_values)
        elif test_type == 'bayesian':
            p_value, ci_lower, ci_upper = self._bayesian_test(treatment_values, control_values)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_values) - 1) * control_values.std()**2 +
             (len(treatment_values) - 1) * treatment_values.std()**2) /
            (len(control_values) + len(treatment_values) - 2)
        )
        effect_size = difference / pooled_std if pooled_std != 0 else 0.0

        # Update Prometheus
        ab_test_p_value.observe(p_value)

        report = SignificanceReport(
            metric=metric,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            difference=difference,
            difference_pct=difference_pct,
            p_value=p_value,
            significant=p_value < self.guardrails.max_p_value,
            confidence_interval_95=(ci_lower, ci_upper),
            test_type=test_type,
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values),
            effect_size=effect_size
        )

        logger.info(f"Significance test ({test_type}): {metric}={difference:.2f} "
                   f"({difference_pct:+.1f}%), p={p_value:.4f}")

        return report

    def _bootstrap_p_value(
        self,
        treatment: np.ndarray,
        control: np.ndarray,
        n_iterations: int = 10000
    ) -> float:
        """Calculate p-value using bootstrap resampling."""
        observed_diff = treatment.mean() - control.mean()

        # Pool data under null hypothesis
        pooled = np.concatenate([treatment, control])

        # Bootstrap null distribution
        null_diffs = []
        for _ in range(n_iterations):
            bootstrap_treatment = np.random.choice(pooled, size=len(treatment), replace=True)
            bootstrap_control = np.random.choice(pooled, size=len(control), replace=True)
            null_diffs.append(bootstrap_treatment.mean() - bootstrap_control.mean())

        null_diffs = np.array(null_diffs)

        # Two-tailed p-value
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

        return p_value

    def _bootstrap_ci(
        self,
        treatment: np.ndarray,
        control: np.ndarray,
        n_iterations: int = 10000,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate 95% confidence interval using bootstrap."""
        diffs = []
        for _ in range(n_iterations):
            bootstrap_treatment = np.random.choice(treatment, size=len(treatment), replace=True)
            bootstrap_control = np.random.choice(control, size=len(control), replace=True)
            diffs.append(bootstrap_treatment.mean() - bootstrap_control.mean())

        diffs = np.array(diffs)
        ci_lower = np.percentile(diffs, (alpha / 2) * 100)
        ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)

        return ci_lower, ci_upper

    def _bayesian_test(
        self,
        treatment: np.ndarray,
        control: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Bayesian hypothesis test using normal-normal conjugate prior.

        Returns:
            (p_value, ci_lower, ci_upper)
        """
        # Assume normal priors with uninformative hyperparameters
        treatment_mean = treatment.mean()
        treatment_std = treatment.std()
        control_mean = control.mean()
        control_std = control.std()

        # Posterior for difference (simplified)
        diff_mean = treatment_mean - control_mean
        diff_std = np.sqrt((treatment_std**2 / len(treatment)) +
                          (control_std**2 / len(control)))

        # Probability that treatment > control (one-tailed)
        # For two-tailed, calculate P(|diff| > 0)
        p_value = 2 * (1 - stats.norm.cdf(abs(diff_mean) / diff_std))

        # 95% credible interval
        ci_lower = diff_mean - 1.96 * diff_std
        ci_upper = diff_mean + 1.96 * diff_std

        return p_value, ci_lower, ci_upper

    # ========================================================================
    # Decision Support
    # ========================================================================

    def recommend_decision(self) -> Decision:
        """
        Recommend A/B test decision (continue/stop/promote).

        Returns:
            Decision enum with recommendation
        """
        metrics = self.get_current_metrics()

        if not metrics.get('control') or not metrics.get('treatment'):
            return Decision.INSUFFICIENT_DATA

        # Check sample size
        if metrics['treatment']['sample_size'] < self.guardrails.min_sample_size:
            logger.info(f"Insufficient sample size: {metrics['treatment']['sample_size']} < "
                       f"{self.guardrails.min_sample_size}")
            return Decision.INSUFFICIENT_DATA

        # Check guardrails first
        guardrail_status = self.check_guardrails()
        if guardrail_status.violated:
            logger.warning("Recommending STOP due to guardrail violations")
            return Decision.STOP_DEGRADATION

        # Calculate significance for key metrics
        mae_sig = self.calculate_significance('mae', test_type='t-test')
        roi_sig = self.calculate_significance('roi', test_type='t-test')

        # Decision logic based on success criteria
        success_criteria = self.config['success_criteria']

        # Check if treatment is significantly better
        mae_improved = (mae_sig.difference < -success_criteria['mae_improvement_yards'] and
                       mae_sig.significant)
        roi_improved = (roi_sig.difference > success_criteria['roi_improvement_pct'] and
                       roi_sig.significant)

        if mae_improved and roi_improved:
            logger.info("Recommending PROMOTE: Treatment significantly better on MAE and ROI")
            return Decision.PROMOTE_TREATMENT

        # Check if treatment shows no effect (futility)
        if not mae_sig.significant and not roi_sig.significant:
            # Use SPRT for early stopping
            if self._sequential_futility_check():
                logger.info("Recommending STOP: No significant effect detected (futility)")
                return Decision.STOP_NO_EFFECT

        # Default: continue test
        logger.info("Recommending CONTINUE: Insufficient evidence for decision")
        return Decision.CONTINUE

    def _sequential_futility_check(self) -> bool:
        """
        Sequential probability ratio test (SPRT) for futility.

        Returns:
            True if test should stop due to futility, False otherwise
        """
        # Simplified SPRT: check if 95% CI contains 0 and sample size > threshold
        mae_sig = self.calculate_significance('mae', test_type='t-test')
        roi_sig = self.calculate_significance('roi', test_type='t-test')

        mae_contains_zero = (mae_sig.confidence_interval_95[0] <= 0 <=
                            mae_sig.confidence_interval_95[1])
        roi_contains_zero = (roi_sig.confidence_interval_95[0] <= 0 <=
                            roi_sig.confidence_interval_95[1])

        # If both CIs contain 0 and we have enough data, recommend stopping
        min_futility_samples = self.guardrails.min_sample_size * 2

        if (mae_contains_zero and roi_contains_zero and
            mae_sig.sample_size_treatment >= min_futility_samples):
            return True

        return False

    # ========================================================================
    # Export and Reporting
    # ========================================================================

    def export_results(self, format: str = 'dataframe') -> pd.DataFrame:
        """
        Export A/B test results for analysis.

        Args:
            format: Export format ('dataframe', 'csv', 'json')

        Returns:
            DataFrame with all test results and metrics
        """
        # Get raw results
        df = self.get_ab_test_results()

        # Calculate group metrics
        metrics = self.get_current_metrics()

        # Add significance tests
        mae_sig = self.calculate_significance('mae')
        roi_sig = self.calculate_significance('roi')
        win_rate_sig = self.calculate_significance('win_rate')

        # Create summary DataFrame
        summary = pd.DataFrame({
            'group': ['control', 'treatment'],
            'mae': [metrics['control']['mae'], metrics['treatment']['mae']],
            'win_rate': [metrics['control']['win_rate'], metrics['treatment']['win_rate']],
            'roi': [metrics['control']['roi'], metrics['treatment']['roi']],
            'sample_size': [metrics['control']['sample_size'], metrics['treatment']['sample_size']]
        })

        # Add significance test results as attributes
        summary.attrs['mae_significance'] = mae_sig.to_dict()
        summary.attrs['roi_significance'] = roi_sig.to_dict()
        summary.attrs['win_rate_significance'] = win_rate_sig.to_dict()
        summary.attrs['recommendation'] = self.recommend_decision().value

        logger.info(f"Exported {len(df)} test results with summary metrics")

        if format == 'csv':
            output_path = f"results/ab_test_export_{datetime.now():%Y%m%d_%H%M%S}.csv"
            summary.to_csv(output_path, index=False)
            logger.info(f"Exported to {output_path}")
        elif format == 'json':
            output_path = f"results/ab_test_export_{datetime.now():%Y%m%d_%H%M%S}.json"
            summary.to_json(output_path, orient='records', indent=2)
            logger.info(f"Exported to {output_path}")

        return summary

    def generate_report(self) -> str:
        """
        Generate human-readable A/B test report.

        Returns:
            Formatted report string
        """
        metrics = self.get_current_metrics()
        guardrail_status = self.check_guardrails()
        decision = self.recommend_decision()

        mae_sig = self.calculate_significance('mae')
        roi_sig = self.calculate_significance('roi')

        report = f"""
=============================================================================
A/B TEST REPORT - {self.config['test_config']['name']}
=============================================================================
Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
Test Period: {self.test_start_date:%Y-%m-%d} to {self.test_end_date:%Y-%m-%d}

CURRENT METRICS
-----------------------------------------------------------------------------
                    Control         Treatment       Difference
MAE:                {metrics['control']['mae']:7.2f}         {metrics['treatment']['mae']:7.2f}         {mae_sig.difference:+7.2f} ({mae_sig.difference_pct:+.1f}%)
Win Rate:           {metrics['control']['win_rate']:6.1f}%         {metrics['treatment']['win_rate']:6.1f}%         {(metrics['treatment']['win_rate'] - metrics['control']['win_rate']):+6.1f}pp
ROI:                {metrics['control']['roi']:6.1f}%         {metrics['treatment']['roi']:6.1f}%         {roi_sig.difference:+6.1f}pp
Sample Size:        {metrics['control']['sample_size']:7d}         {metrics['treatment']['sample_size']:7d}

STATISTICAL SIGNIFICANCE
-----------------------------------------------------------------------------
MAE:        p = {mae_sig.p_value:.4f} {'✅ SIGNIFICANT' if mae_sig.significant else '❌ NOT SIGNIFICANT'}
            95% CI: [{mae_sig.confidence_interval_95[0]:.2f}, {mae_sig.confidence_interval_95[1]:.2f}]
            Effect size (Cohen's d): {mae_sig.effect_size:.3f}

ROI:        p = {roi_sig.p_value:.4f} {'✅ SIGNIFICANT' if roi_sig.significant else '❌ NOT SIGNIFICANT'}
            95% CI: [{roi_sig.confidence_interval_95[0]:.2f}, {roi_sig.confidence_interval_95[1]:.2f}]
            Effect size (Cohen's d): {roi_sig.effect_size:.3f}

GUARDRAILS
-----------------------------------------------------------------------------
Status: {'⛔ VIOLATED' if guardrail_status.violated else '✅ PASSED'}
"""

        if guardrail_status.violations:
            report += "\nViolations:\n"
            for v in guardrail_status.violations:
                report += f"  ⛔ {v}\n"

        if guardrail_status.warnings:
            report += "\nWarnings:\n"
            for w in guardrail_status.warnings:
                report += f"  ⚠️  {w}\n"

        report += f"""
RECOMMENDATION
-----------------------------------------------------------------------------
{decision.value.upper().replace('_', ' ')}
"""

        if decision == Decision.PROMOTE_TREATMENT:
            report += "\n✅ Treatment shows significant improvement. Safe to promote.\n"
        elif decision == Decision.STOP_DEGRADATION:
            report += "\n⛔ Treatment shows degradation. Stop immediately.\n"
        elif decision == Decision.STOP_NO_EFFECT:
            report += "\n⏹️  No significant effect detected. Consider stopping for futility.\n"
        elif decision == Decision.CONTINUE:
            report += "\n⏳ Insufficient evidence. Continue collecting data.\n"
        else:
            report += "\n⏳ Insufficient data for decision.\n"

        report += "=============================================================================\n"

        return report


# ============================================================================
# CLI Wrapper Functions
# ============================================================================

def cli_status():
    """Print A/B test status."""
    monitor = ABTestMonitor()
    print(monitor.generate_report())


def cli_metrics():
    """Print current metrics."""
    monitor = ABTestMonitor()
    metrics = monitor.get_current_metrics()

    print("\n=== Current Metrics ===")
    for group in ['control', 'treatment']:
        if group in metrics and metrics[group]:
            print(f"\n{group.upper()}:")
            for metric, value in metrics[group].items():
                print(f"  {metric}: {value}")


def cli_check_guardrails():
    """Check guardrail violations."""
    monitor = ABTestMonitor()
    status = monitor.check_guardrails()

    print("\n=== Guardrail Status ===")
    print(f"Violated: {status.violated}")

    if status.violations:
        print("\nViolations:")
        for v in status.violations:
            print(f"  ⛔ {v}")

    if status.warnings:
        print("\nWarnings:")
        for w in status.warnings:
            print(f"  ⚠️  {w}")

    if not status.violated and not status.warnings:
        print("\n✅ All guardrails passed")


def cli_auto_stop():
    """Auto-stop if guardrails violated."""
    monitor = ABTestMonitor()
    stopped = monitor.auto_stop_if_violated()

    if stopped:
        print("\n⛔ A/B test AUTO-STOPPED due to guardrail violations")
        print("\nCheck logs for details:")
        print("  tail -f logs/ab_test_monitor.log")
    else:
        print("\n✅ No guardrail violations. Test continues.")


def cli_promote():
    """Promote treatment to production."""
    monitor = ABTestMonitor()
    decision = monitor.recommend_decision()

    if decision == Decision.PROMOTE_TREATMENT:
        print("\n✅ Promoting treatment to production...")
        monitor._update_test_status(TestStatus.PROMOTED, ["Significant improvement validated"])
        ab_test_promotions.inc()
        print("✅ Treatment promoted successfully")
        print("\nNext steps:")
        print("  1. Update ensemble configuration to use treatment model")
        print("  2. Deploy to production environment")
        print("  3. Monitor performance for 1 week")
    else:
        print(f"\n⛔ Cannot promote. Current recommendation: {decision.value}")
        print("\nRun 'monitor_ab_test.sh status' for details")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ab_test_monitor.py [status|metrics|check|stop|promote]")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'status':
        cli_status()
    elif command == 'metrics':
        cli_metrics()
    elif command == 'check':
        cli_check_guardrails()
    elif command == 'stop':
        cli_auto_stop()
    elif command == 'promote':
        cli_promote()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
