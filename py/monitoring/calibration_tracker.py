#!/usr/bin/env python3
"""
Calibration Tracking System for NFL Props Models

Monitors prediction calibration and reliability diagrams.
Tracks whether confidence intervals are properly calibrated.
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics"""
    expected_confidence: float
    observed_coverage: float
    calibration_error: float
    n_predictions: int
    confidence_level: str


class CalibrationTracker:
    """
    Track and monitor model calibration over time.

    Features:
    - Confidence interval coverage analysis
    - Reliability diagrams
    - Calibration drift detection
    - Model-specific tracking
    """

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'
        }

    def load_predictions_with_outcomes(
        self,
        start_date: datetime,
        end_date: datetime,
        model_version: Optional[str] = None,
        prop_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Load predictions with actual outcomes"""
        conn = psycopg2.connect(**self.db_config)

        query = """
        WITH predictions_with_outcomes AS (
            SELECT
                pr.player_id,
                pr.season,
                pr.week,
                pr.model_version,
                pr.stat_type as prop_type,
                pr.rating_mean as prediction,
                pr.rating_sd as uncertainty,
                pr.rating_q05 as q05,
                pr.rating_q25 as q25,
                pr.rating_q50 as q50,
                pr.rating_q75 as q75,
                pr.rating_q95 as q95,
                pgs.stat_yards as actual,
                pr.updated_at as prediction_date
            FROM mart.bayesian_player_ratings pr
            LEFT JOIN mart.player_game_stats pgs
                ON pr.player_id = pgs.player_id
                AND pr.season = pgs.season
                AND pr.week = pgs.week
                AND pr.stat_type = pgs.stat_type
            WHERE pr.updated_at BETWEEN %s AND %s
              AND pgs.stat_yards IS NOT NULL
        )
        SELECT *
        FROM predictions_with_outcomes
        WHERE actual IS NOT NULL
        """

        params = [start_date, end_date]

        if model_version:
            query = query.replace("WHERE pr.updated_at",
                                 f"AND pr.model_version = %s\nWHERE pr.updated_at")
            params.insert(0, model_version)

        if prop_type:
            query = query.replace("WHERE actual",
                                 f"AND prop_type = %s\nWHERE actual")
            params.append(prop_type)

        query += " ORDER BY prediction_date DESC"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        return df

    def calculate_coverage(
        self,
        df: pd.DataFrame,
        confidence_levels: List[float] = [0.50, 0.68, 0.90, 0.95]
    ) -> Dict[float, CalibrationMetrics]:
        """Calculate coverage for different confidence levels"""

        metrics = {}

        for conf_level in confidence_levels:
            # Calculate quantile columns based on confidence level
            lower_q = (1 - conf_level) / 2
            upper_q = 1 - lower_q

            # Map confidence levels to quantile columns
            if conf_level == 0.50:
                lower_col, upper_col = 'q25', 'q75'
            elif conf_level == 0.90:
                lower_col, upper_col = 'q05', 'q95'
            else:
                # For other levels, use normal approximation
                z_score = stats.norm.ppf(upper_q)
                df[f'lower_{conf_level}'] = df['prediction'] - z_score * df['uncertainty']
                df[f'upper_{conf_level}'] = df['prediction'] + z_score * df['uncertainty']
                lower_col = f'lower_{conf_level}'
                upper_col = f'upper_{conf_level}'

            # Check coverage
            if lower_col in df.columns and upper_col in df.columns:
                in_interval = (df['actual'] >= df[lower_col]) & (df['actual'] <= df[upper_col])
                observed_coverage = in_interval.mean()

                calibration_error = abs(observed_coverage - conf_level)

                metrics[conf_level] = CalibrationMetrics(
                    expected_confidence=conf_level,
                    observed_coverage=observed_coverage,
                    calibration_error=calibration_error,
                    n_predictions=len(df),
                    confidence_level=f"{int(conf_level * 100)}%"
                )

        return metrics

    def calculate_quantile_calibration(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate calibration at each quantile"""

        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantile_cols = ['q05', 'q25', 'q50', 'q75', 'q95']

        calibration_data = []

        for q, col in zip(quantiles, quantile_cols):
            if col in df.columns:
                # Proportion of actuals below this quantile prediction
                observed_proportion = (df['actual'] <= df[col]).mean()

                calibration_data.append({
                    'quantile': q,
                    'expected': q,
                    'observed': observed_proportion,
                    'error': observed_proportion - q,
                    'n_predictions': len(df)
                })

        return pd.DataFrame(calibration_data)

    def create_reliability_diagram(
        self,
        df: pd.DataFrame,
        n_bins: int = 10
    ) -> go.Figure:
        """Create reliability diagram"""

        # Calculate predicted probabilities for different thresholds
        # For props, we'll use the probability of going over certain values

        fig = go.Figure()

        # Create bins of predicted probabilities
        df['predicted_prob'] = stats.norm.cdf(
            (df['actual'].median() - df['prediction']) / df['uncertainty']
        )

        df['prob_bin'] = pd.cut(df['predicted_prob'], bins=n_bins)

        # Calculate observed frequency in each bin
        reliability = df.groupby('prob_bin').agg({
            'actual': lambda x: (x <= x.median()).mean(),  # Observed frequency
            'predicted_prob': 'mean'  # Mean predicted probability
        }).reset_index()

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))

        # Actual calibration
        fig.add_trace(go.Scatter(
            x=reliability['predicted_prob'],
            y=reliability['actual'],
            mode='markers+lines',
            name='Observed',
            marker=dict(size=10)
        ))

        fig.update_layout(
            title='Reliability Diagram',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Observed Frequency',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=500,
            showlegend=True
        )

        return fig

    def create_calibration_plot(
        self,
        df: pd.DataFrame
    ) -> go.Figure:
        """Create comprehensive calibration plot"""

        quantile_cal = self.calculate_quantile_calibration(df)

        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray', width=2)
        ))

        # Observed calibration
        fig.add_trace(go.Scatter(
            x=quantile_cal['expected'],
            y=quantile_cal['observed'],
            mode='markers+lines',
            name='Observed',
            marker=dict(size=12, color='blue'),
            line=dict(width=2)
        ))

        # Add confidence bands (binomial confidence intervals)
        n = quantile_cal['n_predictions'].iloc[0]
        for _, row in quantile_cal.iterrows():
            p = row['expected']
            se = np.sqrt(p * (1-p) / n)
            lower = p - 1.96 * se
            upper = p + 1.96 * se

            fig.add_shape(
                type="line",
                x0=p, y0=lower,
                x1=p, y1=upper,
                line=dict(color="rgba(0,0,255,0.2)", width=8)
            )

        fig.update_layout(
            title='Quantile Calibration Plot',
            xaxis_title='Expected Quantile',
            yaxis_title='Observed Quantile',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=500,
            showlegend=True
        )

        return fig

    def detect_calibration_drift(
        self,
        df: pd.DataFrame,
        window_size: int = 50
    ) -> pd.DataFrame:
        """Detect calibration drift over time"""

        df = df.sort_values('prediction_date')

        drift_data = []

        for i in range(window_size, len(df), 10):  # Sliding window with step
            window = df.iloc[i-window_size:i]

            # Calculate 90% CI coverage for this window
            in_90ci = (window['actual'] >= window['q05']) & (window['actual'] <= window['q95'])
            coverage_90 = in_90ci.mean()

            # Calculate calibration error
            calibration_error = abs(coverage_90 - 0.90)

            drift_data.append({
                'date': window['prediction_date'].max(),
                'coverage_90': coverage_90,
                'calibration_error': calibration_error,
                'n_predictions': len(window)
            })

        return pd.DataFrame(drift_data)

    def save_calibration_metrics(
        self,
        metrics: Dict[float, CalibrationMetrics],
        model_version: str
    ):
        """Save calibration metrics to database"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        for conf_level, metric in metrics.items():
            query = """
            INSERT INTO predictions.calibration_metrics (
                model_version,
                confidence_level,
                expected_coverage,
                observed_coverage,
                calibration_error,
                n_predictions,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """

            try:
                cur.execute(query, (
                    model_version,
                    metric.confidence_level,
                    metric.expected_confidence,
                    metric.observed_coverage,
                    metric.calibration_error,
                    metric.n_predictions,
                    datetime.now()
                ))
            except Exception as e:
                logger.warning(f"Could not save calibration metrics: {e}")

        conn.commit()
        conn.close()

    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        model_version: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive calibration report"""

        # Load data
        df = self.load_predictions_with_outcomes(start_date, end_date, model_version)

        if df.empty:
            return {
                'status': 'error',
                'message': 'No predictions with outcomes found'
            }

        # Calculate coverage metrics
        coverage_metrics = self.calculate_coverage(df)

        # Calculate quantile calibration
        quantile_cal = self.calculate_quantile_calibration(df)

        # Detect drift
        drift = self.detect_calibration_drift(df)

        # Create visualizations
        reliability_fig = self.create_reliability_diagram(df)
        calibration_fig = self.create_calibration_plot(df)

        # Summary statistics
        summary = {
            'n_predictions': len(df),
            'date_range': f"{start_date.date()} to {end_date.date()}",
            'model_version': model_version or 'All',
            'coverage_50': coverage_metrics.get(0.50, None),
            'coverage_68': coverage_metrics.get(0.68, None),
            'coverage_90': coverage_metrics.get(0.90, None),
            'coverage_95': coverage_metrics.get(0.95, None),
            'mean_calibration_error': np.mean([m.calibration_error for m in coverage_metrics.values()]),
            'max_calibration_error': max([m.calibration_error for m in coverage_metrics.values()]),
            'calibration_status': 'GOOD' if max([m.calibration_error for m in coverage_metrics.values()]) < 0.05 else 'NEEDS ATTENTION'
        }

        return {
            'status': 'success',
            'summary': summary,
            'coverage_metrics': coverage_metrics,
            'quantile_calibration': quantile_cal,
            'drift_analysis': drift,
            'reliability_diagram': reliability_fig,
            'calibration_plot': calibration_fig
        }


def main():
    """Demo calibration tracking"""

    logger.info("="*60)
    logger.info("CALIBRATION TRACKING SYSTEM")
    logger.info("="*60)

    tracker = CalibrationTracker()

    # Generate report for last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Generate report
    report = tracker.generate_report(start_date, end_date)

    if report['status'] == 'success':
        summary = report['summary']

        logger.info("\nCalibration Summary:")
        logger.info(f"  Predictions analyzed: {summary['n_predictions']}")
        logger.info(f"  Date range: {summary['date_range']}")
        logger.info(f"  Model: {summary['model_version']}")

        if summary['coverage_50']:
            logger.info("\nCoverage Analysis:")
            for level in [50, 68, 90, 95]:
                key = f'coverage_{level}'
                if summary[key]:
                    metric = summary[key]
                    logger.info(f"  {level}% CI: {metric.observed_coverage:.1%} "
                              f"(expected: {metric.expected_confidence:.1%}, "
                              f"error: {metric.calibration_error:.1%})")

        logger.info(f"\nCalibration Status: {summary['calibration_status']}")
        logger.info(f"  Mean calibration error: {summary['mean_calibration_error']:.1%}")
        logger.info(f"  Max calibration error: {summary['max_calibration_error']:.1%}")

        # Check for specific models
        for model_version in ['hierarchical_v1.0', 'informative_priors_v2.5']:
            logger.info(f"\n{model_version}:")
            model_report = tracker.generate_report(start_date, end_date, model_version)

            if model_report['status'] == 'success' and model_report['summary']['n_predictions'] > 0:
                model_summary = model_report['summary']
                logger.info(f"  Predictions: {model_summary['n_predictions']}")
                logger.info(f"  Status: {model_summary['calibration_status']}")
                logger.info(f"  Mean error: {model_summary['mean_calibration_error']:.1%}")
            else:
                logger.info(f"  No predictions found")

    else:
        logger.warning(f"Report generation failed: {report['message']}")

    logger.info("\n" + "="*60)
    logger.info("✓ Calibration tracking complete")
    logger.info("="*60)


if __name__ == "__main__":
    main()