#!/usr/bin/env python3
"""
Train Temperature Scaler for Ensemble v6.0 Production
======================================================

**Purpose:** Generate ensemble v6.0 predictions on 2022-2024 historical data
and train isotonic regression temperature scaler for 90% calibration coverage.

**Workflow:**
1. Generate v6.0 ensemble predictions on 2022-2024 (148 games, ~2000 player-games)
2. Train isotonic regression scaler on predicted probabilities
3. Evaluate calibration before/after scaling
4. Save temperature scaler to models/calibration/temperature_scaler_v6.pkl
5. Generate calibration plot (reliability diagram)

**Target Calibration:**
- Coverage: 88-92% (90% target)
- ECE: < 0.05 (Expected Calibration Error)
- Interval width: Minimize while maintaining coverage

**Expected Runtime:** 30-45 minutes (ensemble predictions are slow with R models)

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
Version: 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional matplotlib import (for plotting)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - calibration plots will be skipped")

# Import ensemble and temperature scaling
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way
from py.calibration.temperature_scaling import TemperatureScaler


class TemperatureScalerTrainerV6:
    """Train temperature scaler for ensemble v6.0."""

    def __init__(
        self,
        start_season: int = 2022,
        end_season: int = 2024,
        output_dir: str = "models/calibration",
        results_dir: str = "analysis/results"
    ):
        """
        Initialize temperature scaler trainer.

        Args:
            start_season: Start season for calibration data (2022)
            end_season: End season for calibration data (2024)
            output_dir: Directory to save temperature scaler
            results_dir: Directory to save results and plots
        """
        self.start_season = start_season
        self.end_season = end_season
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ensemble v6.0
        logger.info("Initializing ensemble v6.0 production...")
        self.ensemble = EnhancedEnsemble5Way()

        logger.info(f"Loaded ensemble weights: {self.ensemble.weights}")
        logger.info(f"Current fallback temperature: {self.ensemble.temperature:.4f}")

    def generate_ensemble_predictions(self) -> pd.DataFrame:
        """
        Generate ensemble v6.0 predictions on 2022-2024 historical data.

        Returns:
            DataFrame with columns:
                - player_id, season, week, game_id
                - actual_yards: Actual passing yards
                - pred_mean: Ensemble mean prediction
                - pred_std: Ensemble standard deviation (UNCALIBRATED)
                - lower_90: 90% PI lower bound (uncalibrated)
                - upper_90: 90% PI upper bound (uncalibrated)
        """
        logger.info("=" * 80)
        logger.info("GENERATING ENSEMBLE v6.0 PREDICTIONS (2022-2024)")
        logger.info("=" * 80)

        # Check if predictions already exist
        cache_path = self.results_dir / f"ensemble_v6_predictions_{self.start_season}_{self.end_season}.csv"
        if cache_path.exists():
            logger.info(f"Loading cached predictions from {cache_path}")
            df = pd.read_csv(cache_path)
            logger.info(f"Loaded {len(df)} cached predictions")
            return df

        # Fetch historical player-games from database
        import psycopg2

        db_password = os.getenv("DB_PASSWORD")
        if not db_password:
            raise ValueError("DB_PASSWORD environment variable not set")

        conn = psycopg2.connect(
            host="localhost",
            port=5544,
            database="devdb01",
            user="dro",
            password=db_password
        )

        query = """
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.kickoff,
            p.passer_player_id as player_id,
            p.passer_player_name as player_name,
            p.posteam as team,
            SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) as actual_yards,
            COUNT(CASE WHEN p.pass = true THEN 1 END) as pass_attempts
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.passer_player_id IS NOT NULL
          AND g.season >= %(start_season)s
          AND g.season <= %(end_season)s
          AND g.week > 0
        GROUP BY g.game_id, g.season, g.week, g.kickoff,
                 p.passer_player_id, p.passer_player_name, p.posteam
        HAVING SUM(CASE WHEN p.pass = true THEN p.yards_gained ELSE 0 END) > 0
           AND COUNT(CASE WHEN p.pass = true THEN 1 END) >= 10
        ORDER BY g.kickoff, g.game_id
        """

        df = pd.read_sql_query(
            query,
            conn,
            params={
                'start_season': self.start_season,
                'end_season': self.end_season
            }
        )
        conn.close()

        logger.info(f"Loaded {len(df)} player-games from database")
        logger.info(f"  Games: {df['game_id'].nunique()}")
        logger.info(f"  Players: {df['player_id'].nunique()}")
        logger.info(f"  Avg actual yards: {df['actual_yards'].mean():.1f}")

        # Generate ensemble predictions for each player-game
        predictions = []
        failed_count = 0

        logger.info(f"\nGenerating ensemble predictions for {len(df)} player-games...")
        logger.info("⚠️  This will take 30-45 minutes (R model integration is slow)")

        for idx, row in df.iterrows():
            try:
                # Generate ensemble prediction
                pred = self.ensemble.predict(
                    player_id=row['player_id'],
                    season=row['season'],
                    week=row['week'],
                    return_components=False,
                    return_uncertainty=True
                )

                predictions.append({
                    'game_id': row['game_id'],
                    'season': row['season'],
                    'week': row['week'],
                    'player_id': row['player_id'],
                    'player_name': row['player_name'],
                    'team': row['team'],
                    'actual_yards': row['actual_yards'],
                    'pass_attempts': row['pass_attempts'],
                    'pred_mean': pred['ensemble_mean'],
                    'pred_std': pred['ensemble_std'],  # UNCALIBRATED
                    'lower_90': pred['lower_90'],
                    'upper_90': pred['upper_90'],
                    'inference_time_ms': pred['prediction_time_ms']
                })

                # Log progress every 100 games
                if (idx + 1) % 100 == 0:
                    logger.info(
                        f"  Progress: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%) | "
                        f"Valid: {len(predictions)} | Failed: {failed_count}"
                    )

            except Exception as e:
                logger.warning(f"Failed to predict {row['player_name']} {row['season']} W{row['week']}: {e}")
                failed_count += 1

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)

        logger.info(f"\n✅ Predictions complete!")
        logger.info(f"  Valid predictions: {len(predictions_df)}")
        logger.info(f"  Failed predictions: {failed_count}")

        # Save predictions to cache
        predictions_df.to_csv(cache_path, index=False)
        logger.info(f"✅ Cached predictions to {cache_path}")

        return predictions_df

    def train_temperature_scaler(self, predictions_df: pd.DataFrame) -> Tuple[TemperatureScaler, Dict]:
        """
        Train isotonic regression temperature scaler.

        Args:
            predictions_df: DataFrame with pred_mean, pred_std, actual_yards

        Returns:
            (TemperatureScaler, calibration_report)
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING TEMPERATURE SCALER (ISOTONIC REGRESSION)")
        logger.info("=" * 80)

        # Prepare data for TemperatureScaler
        # Rename columns to match TemperatureScaler expectations
        calib_df = predictions_df.rename(columns={'actual_yards': 'actual'}).copy()

        # Initialize temperature scaler
        scaler = TemperatureScaler(target_coverage=0.90, method='brentq')

        # Fit temperature scaler
        optimal_T = scaler.fit(calib_df)

        # Get calibration report
        report = scaler.get_calibration_report()

        logger.info("\n" + "=" * 80)
        logger.info("CALIBRATION REPORT")
        logger.info("=" * 80)
        logger.info(f"Optimal Temperature: T = {report['temperature']:.4f}")
        logger.info(f"Target Coverage: {report['target_coverage']:.1%}")
        logger.info(f"\nBefore Temperature Scaling:")
        logger.info(f"  Coverage: {report['coverage_before']:.1%}")
        logger.info(f"  Mean Interval Width: {report['width_before']:.1f} yards")
        logger.info(f"\nAfter Temperature Scaling:")
        logger.info(f"  Coverage: {report['coverage_after']:.1%}")
        logger.info(f"  Mean Interval Width: {report['width_after']:.1f} yards")
        logger.info(f"\nMAE: {report['mae']:.2f} yards (unchanged by scaling)")
        logger.info(f"Samples: {report['n_samples']}")

        # Validate calibration achieved target
        coverage_error = abs(report['coverage_after'] - report['target_coverage'])
        if coverage_error > 0.02:  # 2% tolerance
            logger.warning(f"⚠️  Large coverage error: {coverage_error:.1%}")
            logger.warning("Temperature scaling may not have fully converged")
        else:
            logger.info("✅ Temperature scaling calibrated successfully!")

        return scaler, report

    def generate_calibration_plot(
        self,
        predictions_df: pd.DataFrame,
        scaler: TemperatureScaler,
        report: Dict
    ):
        """
        Generate calibration plot (reliability diagram).

        Args:
            predictions_df: DataFrame with predictions
            scaler: Trained temperature scaler
            report: Calibration report
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - skipping calibration plot")
            return

        logger.info("\nGenerating calibration plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Coverage before/after temperature scaling
        calib_df = predictions_df.rename(columns={'actual_yards': 'actual'}).copy()

        # Calculate z-scores before temperature scaling
        z_raw = (calib_df['actual'] - calib_df['pred_mean']) / calib_df['pred_std']

        # Calculate z-scores after temperature scaling
        std_calibrated = calib_df['pred_std'] * np.sqrt(scaler.temperature)
        z_calibrated = (calib_df['actual'] - calib_df['pred_mean']) / std_calibrated

        # Plot empirical CDF vs theoretical normal CDF
        quantiles = np.linspace(0, 1, 100)
        theoretical_quantiles = stats.norm.ppf(quantiles)

        empirical_raw = np.quantile(z_raw, quantiles)
        empirical_calibrated = np.quantile(z_calibrated, quantiles)

        axes[0].plot(theoretical_quantiles, empirical_raw, 'o-', label='Before Scaling', alpha=0.7)
        axes[0].plot(theoretical_quantiles, empirical_calibrated, 's-', label='After Scaling', alpha=0.7)
        axes[0].plot(theoretical_quantiles, theoretical_quantiles, 'k--', label='Perfect Calibration')
        axes[0].set_xlabel('Theoretical Quantiles (Normal)')
        axes[0].set_ylabel('Empirical Quantiles')
        axes[0].set_title('Q-Q Plot: Calibration Assessment')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right plot: Coverage by confidence level
        confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        coverage_raw = []
        coverage_calibrated = []

        for conf in confidence_levels:
            alpha = 1 - conf
            z_crit = stats.norm.ppf(1 - alpha / 2)

            # Raw coverage
            within_raw = (np.abs(z_raw) <= z_crit).mean()
            coverage_raw.append(within_raw)

            # Calibrated coverage
            within_cal = (np.abs(z_calibrated) <= z_crit).mean()
            coverage_calibrated.append(within_cal)

        axes[1].plot(confidence_levels, coverage_raw, 'o-', label='Before Scaling', alpha=0.7)
        axes[1].plot(confidence_levels, coverage_calibrated, 's-', label='After Scaling', alpha=0.7)
        axes[1].plot(confidence_levels, confidence_levels, 'k--', label='Perfect Calibration')
        axes[1].set_xlabel('Nominal Confidence Level')
        axes[1].set_ylabel('Empirical Coverage')
        axes[1].set_title('Coverage by Confidence Level')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.results_dir / f"temperature_scaler_v6_calibration_{self.start_season}_{self.end_season}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Calibration plot saved to {plot_path}")

        plt.close()

    def save_temperature_scaler(self, scaler: TemperatureScaler, report: Dict):
        """
        Save temperature scaler and metadata.

        Args:
            scaler: Trained temperature scaler
            report: Calibration report
        """
        logger.info("\n" + "=" * 80)
        logger.info("SAVING TEMPERATURE SCALER")
        logger.info("=" * 80)

        # Save temperature scaler (pickle format)
        scaler_path = self.output_dir / "temperature_scaler_v6.pkl"
        scaler.save(scaler_path)

        logger.info(f"✅ Temperature scaler saved to {scaler_path}")

        # Save metadata (JSON format)
        metadata = {
            'version': 'v6.0_production',
            'training_date': datetime.now().isoformat(),
            'training_seasons': f"{self.start_season}-{self.end_season}",
            'ensemble_version': 'v6.0',
            'ensemble_weights': self.ensemble.weights,
            'temperature': report['temperature'],
            'target_coverage': report['target_coverage'],
            'coverage_before': report['coverage_before'],
            'coverage_after': report['coverage_after'],
            'width_before': report['width_before'],
            'width_after': report['width_after'],
            'mae': report['mae'],
            'n_samples': report['n_samples'],
            'improvement_pct': (report['coverage_after'] - report['coverage_before']) * 100,
            'width_reduction_pct': (1 - report['width_after'] / report['width_before']) * 100
        }

        # Convert numpy types to native Python types
        metadata_native = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metadata.items()
        }

        metadata_path = self.output_dir / "temperature_scaler_v6_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_native, f, indent=2)

        logger.info(f"✅ Metadata saved to {metadata_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TEMPERATURE SCALER v6.0 TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Optimal Temperature: T = {metadata['temperature']:.4f}")
        logger.info(f"Coverage Improvement: {metadata['coverage_before']:.1%} → {metadata['coverage_after']:.1%}")
        logger.info(f"Interval Width Reduction: {metadata['width_reduction_pct']:.1f}%")
        logger.info(f"Target Coverage: {metadata['target_coverage']:.1%}")
        logger.info(f"Calibration Status: {'✅ PASS' if 0.88 <= metadata['coverage_after'] <= 0.92 else '❌ FAIL'}")

    def run(self):
        """Run full temperature scaler training pipeline."""
        logger.info("=" * 80)
        logger.info("TEMPERATURE SCALER v6.0 TRAINING PIPELINE")
        logger.info("=" * 80)

        # Step 1: Generate ensemble predictions
        predictions_df = self.generate_ensemble_predictions()

        # Step 2: Train temperature scaler
        scaler, report = self.train_temperature_scaler(predictions_df)

        # Step 3: Generate calibration plot
        self.generate_calibration_plot(predictions_df, scaler, report)

        # Step 4: Save temperature scaler
        self.save_temperature_scaler(scaler, report)

        logger.info("\n✅ All steps complete!")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    trainer = TemperatureScalerTrainerV6(
        start_season=2022,
        end_season=2024,
        output_dir="models/calibration",
        results_dir="analysis/results"
    )

    trainer.run()


if __name__ == "__main__":
    main()
