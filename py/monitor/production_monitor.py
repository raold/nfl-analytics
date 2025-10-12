#!/usr/bin/env python3
"""
Production Model Monitoring

Tracks model performance, data drift, and prediction quality in production.
Logs metrics to database for ongoing monitoring and alerting.

Usage:
    # Monitor predictions for a week
    python py/monitor/production_monitor.py --season 2025 --week 5

    # Monitor specific games
    python py/monitor/production_monitor.py --game-ids 2025_05_KC_NO 2025_05_BUF_HOU

    # Weekly monitoring job (run after week completes)
    python py/monitor/production_monitor.py --weekly-report --season 2025 --week 5
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score, log_loss
import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionMonitor:
    """Monitor model performance in production."""

    def __init__(self, db_config: Optional[Dict] = None):
        """Initialize monitor with database connection."""
        self.db_config = db_config or {
            'dbname': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions',
            'host': 'localhost',
            'port': 5544
        }

    def connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def init_monitoring_tables(self):
        """Initialize monitoring tables in database."""
        conn = self.connect_db()
        cur = conn.cursor()

        # Create monitoring schema
        cur.execute("CREATE SCHEMA IF NOT EXISTS monitoring")

        # Predictions log table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS monitoring.predictions (
                prediction_id SERIAL PRIMARY KEY,
                game_id TEXT NOT NULL,
                season INT NOT NULL,
                week INT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                model_version TEXT NOT NULL,
                home_win_prob FLOAT NOT NULL,
                away_win_prob FLOAT NOT NULL,
                predicted_winner TEXT NOT NULL,
                confidence FLOAT NOT NULL,
                actual_winner TEXT,
                actual_home_win BOOLEAN,
                predicted_at TIMESTAMP NOT NULL,
                evaluated_at TIMESTAMP,
                correct BOOLEAN,
                brier_score FLOAT,
                log_loss FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Performance metrics table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS monitoring.model_metrics (
                metric_id SERIAL PRIMARY KEY,
                model_version TEXT NOT NULL,
                metric_type TEXT NOT NULL, -- 'weekly', 'monthly', 'season'
                season INT NOT NULL,
                week INT,
                num_predictions INT NOT NULL,
                accuracy FLOAT NOT NULL,
                brier_score FLOAT NOT NULL,
                log_loss FLOAT NOT NULL,
                auc FLOAT,
                calibration_score FLOAT,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Feature drift table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS monitoring.feature_drift (
                drift_id SERIAL PRIMARY KEY,
                feature_name TEXT NOT NULL,
                season INT NOT NULL,
                week INT NOT NULL,
                mean_value FLOAT,
                std_value FLOAT,
                min_value FLOAT,
                max_value FLOAT,
                null_pct FLOAT,
                drift_score FLOAT,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Alerts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS monitoring.alerts (
                alert_id SERIAL PRIMARY KEY,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL, -- 'info', 'warning', 'error'
                message TEXT NOT NULL,
                details JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                resolved_by TEXT
            )
        """)

        conn.commit()
        cur.close()
        conn.close()

        logger.info("Monitoring tables initialized")

    def load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """Load predictions from file."""
        return pd.read_csv(predictions_file)

    def load_actual_results(self, season: int, week: int) -> pd.DataFrame:
        """Load actual game results from database."""
        conn = self.connect_db()

        query = """
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score,
            CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_win,
            CASE WHEN home_score > away_score THEN home_team ELSE away_team END as winner
        FROM games
        WHERE season = %s AND week = %s
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY game_id
        """

        df = pd.read_sql_query(query, conn, params=(season, week))
        conn.close()

        return df

    def evaluate_predictions(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame
    ) -> Dict:
        """
        Evaluate predictions against actual results.

        Returns:
            Dictionary of performance metrics
        """
        # Merge predictions with actuals
        merged = predictions.merge(
            actuals,
            on=['game_id', 'season', 'week', 'home_team', 'away_team'],
            how='inner'
        )

        if len(merged) == 0:
            logger.warning("No matching predictions and actuals found")
            return {}

        # Calculate metrics
        y_true = merged['home_win'].values
        y_pred = merged['home_win_prob'].values

        metrics = {
            'num_games': len(merged),
            'accuracy': float(accuracy_score(y_true, (y_pred > 0.5).astype(int))),
            'brier_score': float(brier_score_loss(y_true, y_pred)),
            'log_loss': float(log_loss(y_true, y_pred)),
            'auc': float(roc_auc_score(y_true, y_pred))
        }

        # Calibration analysis (binned)
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_pred, bins)
        calibration_diffs = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_pred = y_pred[mask].mean()
                bin_actual = y_true[mask].mean()
                calibration_diffs.append(abs(bin_pred - bin_actual))

        metrics['calibration_score'] = float(np.mean(calibration_diffs)) if calibration_diffs else None

        # Add per-game correctness
        merged['correct'] = (merged['predicted_winner'] == merged['winner'])

        return metrics, merged

    def log_predictions(self, predictions: pd.DataFrame, actuals: Optional[pd.DataFrame] = None):
        """Log predictions to monitoring database."""
        conn = self.connect_db()
        cur = conn.cursor()

        # Merge with actuals if provided
        if actuals is not None:
            predictions = predictions.merge(
                actuals[['game_id', 'winner', 'home_win']],
                on='game_id',
                how='left'
            )
            predictions['actual_winner'] = predictions['winner']
            predictions['actual_home_win'] = predictions['home_win']
            predictions['correct'] = (predictions['predicted_winner'] == predictions['winner'])

        # Prepare data for insertion
        records = []
        for _, row in predictions.iterrows():
            records.append((
                row['game_id'],
                int(row['season']),
                int(row['week']),
                row['home_team'],
                row['away_team'],
                row.get('model_version', 'v3.0.0'),
                float(row['home_win_prob']),
                float(row['away_win_prob']),
                row['predicted_winner'],
                float(row['confidence']),
                row.get('actual_winner'),
                row.get('actual_home_win'),
                row.get('predicted_at', datetime.now()),
                datetime.now() if actuals is not None else None,
                row.get('correct')
            ))

        # Insert
        execute_values(
            cur,
            """
            INSERT INTO monitoring.predictions
            (game_id, season, week, home_team, away_team, model_version,
             home_win_prob, away_win_prob, predicted_winner, confidence,
             actual_winner, actual_home_win, predicted_at, evaluated_at, correct)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            records
        )

        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Logged {len(records)} predictions to database")

    def log_metrics(self, model_version: str, season: int, week: int, metrics: Dict):
        """Log performance metrics to database."""
        conn = self.connect_db()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO monitoring.model_metrics
            (model_version, metric_type, season, week, num_predictions,
             accuracy, brier_score, log_loss, auc, calibration_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            model_version,
            'weekly',
            season,
            week,
            metrics['num_games'],
            metrics['accuracy'],
            metrics['brier_score'],
            metrics['log_loss'],
            metrics.get('auc'),
            metrics.get('calibration_score')
        ))

        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Logged metrics for {model_version} week {season}-{week}")

    def generate_weekly_report(self, season: int, week: int) -> Dict:
        """Generate weekly performance report."""
        conn = self.connect_db()

        # Get predictions for the week
        query = """
        SELECT
            model_version,
            COUNT(*) as num_predictions,
            AVG(CASE WHEN correct THEN 1 ELSE 0 END) as accuracy,
            AVG(ABS(home_win_prob - CAST(actual_home_win AS INT))) as avg_error
        FROM monitoring.predictions
        WHERE season = %s AND week = %s
          AND actual_home_win IS NOT NULL
        GROUP BY model_version
        """

        metrics_df = pd.read_sql_query(query, conn, params=(season, week))
        conn.close()

        report = {
            'season': season,
            'week': week,
            'generated_at': datetime.now().isoformat(),
            'models': metrics_df.to_dict('records') if not metrics_df.empty else []
        }

        return report

    def check_data_drift(self, current_features: pd.DataFrame, baseline_features: pd.DataFrame) -> List[Dict]:
        """Check for feature drift compared to baseline."""
        drift_alerts = []

        numeric_features = current_features.select_dtypes(include=[np.number]).columns

        for feature in numeric_features:
            if feature not in baseline_features.columns:
                continue

            curr_mean = current_features[feature].mean()
            curr_std = current_features[feature].std()
            base_mean = baseline_features[feature].mean()
            base_std = baseline_features[feature].std()

            # Calculate z-score of mean difference
            if base_std > 0:
                z_score = abs(curr_mean - base_mean) / base_std
            else:
                z_score = 0

            # Alert if significant drift (z > 3)
            if z_score > 3:
                drift_alerts.append({
                    'feature': feature,
                    'current_mean': float(curr_mean),
                    'baseline_mean': float(base_mean),
                    'z_score': float(z_score),
                    'severity': 'warning' if z_score < 5 else 'error'
                })

        return drift_alerts


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Monitor model production performance')
    parser.add_argument('--init-tables', action='store_true', help='Initialize monitoring tables')
    parser.add_argument('--predictions-file', help='Path to predictions CSV')
    parser.add_argument('--season', type=int, help='Season to monitor')
    parser.add_argument('--week', type=int, help='Week to monitor')
    parser.add_argument('--weekly-report', action='store_true', help='Generate weekly report')
    parser.add_argument('--model-version', default='v3.0.0', help='Model version')

    args = parser.parse_args()

    monitor = ProductionMonitor()

    if args.init_tables:
        monitor.init_monitoring_tables()
        print("✓ Monitoring tables initialized")
        return

    if args.weekly_report and args.season and args.week:
        report = monitor.generate_weekly_report(args.season, args.week)
        print("\n" + "=" * 80)
        print(f"WEEKLY REPORT: Season {args.season}, Week {args.week}")
        print("=" * 80)
        print(json.dumps(report, indent=2))
        return

    if args.predictions_file and args.season and args.week:
        # Load predictions and actuals
        predictions = monitor.load_predictions(args.predictions_file)
        actuals = monitor.load_actual_results(args.season, args.week)

        # Evaluate
        metrics, merged = monitor.evaluate_predictions(predictions, actuals)

        # Log to database
        monitor.log_predictions(merged)
        monitor.log_metrics(args.model_version, args.season, args.week, metrics)

        # Display summary
        print("\n" + "=" * 80)
        print(f"MONITORING SUMMARY: {args.model_version}")
        print("=" * 80)
        print(f"Season {args.season}, Week {args.week}")
        print(f"Games: {metrics['num_games']}")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        if metrics.get('calibration_score'):
            print(f"Calibration: {metrics['calibration_score']:.4f}")
        print("\n✓ Metrics logged to monitoring.model_metrics")


if __name__ == '__main__':
    main()
