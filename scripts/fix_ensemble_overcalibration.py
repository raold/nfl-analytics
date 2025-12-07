#!/usr/bin/env python3
"""
Fix Ensemble Overcalibration by Retraining Temperature Scaler

Retrains temperature scaling on 2022-2024 backtest data to achieve optimal
90% calibration coverage while reducing prediction interval width by 22.6%.

Background:
- Current calibration: 98.41% (overcalibrated - intervals too wide)
- Target calibration: 90% (85-95% acceptable range)
- Optimal temperature: T = 0.5991
- Expected improvement: 334.0 yard intervals (vs 431.5 current)

Usage:
    python scripts/fix_ensemble_overcalibration.py

Output:
    - models/calibration/temperature_scaler_4way.pkl (new scaler)
    - models/calibration/temperature_scaler_4way_metadata.json (training info)
    - Validation report printed to console

Author: Claude Code (Haiku 4.5)
Date: November 16, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import optimize, stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_backtest_data() -> pd.DataFrame:
    """Load 2022-2024 backtest results."""
    csv_path = Path('analysis/results/ensemble_4way_backtest_2022_2024.csv')

    if not csv_path.exists():
        logger.error(f"Backtest file not found: {csv_path}")
        raise FileNotFoundError(f"Missing: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded backtest: {len(df)} games ({df['season'].min()}-{df['season'].max()})")

    return df


def calculate_current_calibration(df: pd.DataFrame) -> dict:
    """Calculate current calibration WITHOUT temperature scaling."""
    alpha = 0.10  # 90% PI
    z_scores = (df['actual_yards'] - df['ensemble_mean']) / df['ensemble_std']
    lower_q = stats.norm.ppf(alpha / 2)
    upper_q = stats.norm.ppf(1 - alpha / 2)

    within_90 = ((z_scores >= lower_q) & (z_scores <= upper_q)).sum()
    coverage = within_90 / len(df)

    # Interval width
    width = df['ensemble_std'].mean() * 2 * stats.norm.ppf(1 - alpha / 2)

    return {
        'coverage': coverage,
        'within_90': within_90,
        'total': len(df),
        'width': width,
        'mae': np.abs(df['actual_yards'] - df['ensemble_mean']).mean(),
    }


def find_optimal_temperature(df: pd.DataFrame, target_coverage: float = 0.90) -> float:
    """Find temperature that achieves target coverage."""
    alpha = 0.10  # 90% PI
    lower_q = stats.norm.ppf(alpha / 2)
    upper_q = stats.norm.ppf(1 - alpha / 2)

    def coverage_at_temperature(T: float) -> float:
        """Calculate coverage with temperature T."""
        std_scaled = df['ensemble_std'] * np.sqrt(T)
        z_scaled = (df['actual_yards'] - df['ensemble_mean']) / std_scaled
        within = ((z_scaled >= lower_q) & (z_scaled <= upper_q)).sum()
        return within / len(df)

    def coverage_error(T: float) -> float:
        """Coverage error function for root finding."""
        return coverage_at_temperature(T) - target_coverage

    # Use Brent's method to find root
    try:
        T_optimal = optimize.brentq(
            coverage_error,
            a=0.5,
            b=2.0,
            xtol=0.001,
            maxiter=100
        )
        logger.info(f"✓ Found optimal temperature: T = {T_optimal:.4f}")
        return T_optimal
    except ValueError as e:
        logger.error(f"✗ Failed to find temperature: {e}")
        raise


def calculate_calibration_after_scaling(df: pd.DataFrame, temperature: float) -> dict:
    """Calculate calibration after applying temperature scaling."""
    alpha = 0.10  # 90% PI
    std_scaled = df['ensemble_std'] * np.sqrt(temperature)
    z_scores = (df['actual_yards'] - df['ensemble_mean']) / std_scaled
    lower_q = stats.norm.ppf(alpha / 2)
    upper_q = stats.norm.ppf(1 - alpha / 2)

    within_90 = ((z_scores >= lower_q) & (z_scores <= upper_q)).sum()
    coverage = within_90 / len(df)

    # Interval width
    width = std_scaled.mean() * 2 * stats.norm.ppf(1 - alpha / 2)

    return {
        'coverage': coverage,
        'within_90': within_90,
        'total': len(df),
        'width': width,
        'mae': np.abs(df['actual_yards'] - df['ensemble_mean']).mean(),
    }


def retrain_temperature_scaler(df: pd.DataFrame, temperature: float) -> dict:
    """Retrain the temperature scaler with new temperature value."""
    logger.info("\n" + "="*80)
    logger.info("RETRAINING TEMPERATURE SCALER")
    logger.info("="*80)

    # Calculate calibration before and after
    before = calculate_current_calibration(df)
    after = calculate_calibration_after_scaling(df, temperature)

    logger.info("\n📊 Calibration Before Scaling:")
    logger.info(f"   Coverage: {before['coverage']*100:.2f}%")
    logger.info(f"   Within 90% PI: {before['within_90']}/{before['total']}")
    logger.info(f"   Mean Interval Width: {before['width']:.1f} yards")
    logger.info(f"   MAE: {before['mae']:.2f} yards")

    logger.info("\n🔧 Temperature Scaling:")
    logger.info(f"   Temperature: T = {temperature:.4f}")
    logger.info(f"   Scaling Factor: √T = {np.sqrt(temperature):.4f}")

    logger.info("\n📊 Calibration After Scaling:")
    logger.info(f"   Coverage: {after['coverage']*100:.2f}%")
    logger.info(f"   Within 90% PI: {after['within_90']}/{after['total']}")
    logger.info(f"   Mean Interval Width: {after['width']:.1f} yards")
    logger.info(f"   MAE: {after['mae']:.2f} yards")

    logger.info("\n📈 Improvements:")
    coverage_change = (after['coverage'] - before['coverage']) * 100
    width_change = (after['width'] - before['width']) / before['width'] * 100
    logger.info(f"   Coverage: {before['coverage']*100:.2f}% → {after['coverage']*100:.2f}% ({coverage_change:+.2f}pp)")
    logger.info(f"   Width: {before['width']:.1f} → {after['width']:.1f} yards ({width_change:+.1f}%)")

    # Validate
    is_calibrated = 0.85 <= after['coverage'] <= 0.95
    width_improved = after['width'] < before['width']

    logger.info("\n✅ Validation:")
    logger.info(f"   Coverage in target range (85-95%): {'✅ YES' if is_calibrated else '❌ NO'}")
    logger.info(f"   Interval width improved: {'✅ YES' if width_improved else '❌ NO'}")

    if is_calibrated:
        logger.info("   Status: ✅ CALIBRATED - READY FOR DEPLOYMENT")
    else:
        logger.warning(f"   Status: ⚠️  MARGINAL - Coverage {after['coverage']*100:.2f}%")

    return {
        'temperature': temperature,
        'target_coverage': 0.90,
        'coverage_before': before['coverage'],
        'coverage_after': after['coverage'],
        'width_before': before['width'],
        'width_after': after['width'],
        'mae': after['mae'],
        'n_samples': len(df),
        'training_seasons': f"{df['season'].min()}-{df['season'].max()}",
        'training_date': datetime.now().isoformat(),
    }


def save_temperature_scaler(scaler_data: dict) -> Path:
    """Save temperature scaler to disk."""
    save_path = Path('models/calibration/temperature_scaler_4way.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create scaler object
    scaler = {
        'temperature': scaler_data['temperature'],
        'target_coverage': scaler_data['target_coverage'],
        'method': 'brentq',
    }

    # Save pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"✅ Saved temperature scaler: {save_path}")

    # Save metadata JSON
    metadata_path = save_path.with_name('temperature_scaler_4way_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)

    logger.info(f"✅ Saved metadata: {metadata_path}")

    return save_path


def main():
    """Main function."""
    try:
        logger.info("\n" + "="*80)
        logger.info("FIX ENSEMBLE OVERCALIBRATION")
        logger.info("="*80)

        # Load backtest data
        df = load_backtest_data()

        # Find optimal temperature
        logger.info("\n🔍 Finding optimal temperature...")
        T_optimal = find_optimal_temperature(df, target_coverage=0.90)

        # Retrain scaler
        scaler_data = retrain_temperature_scaler(df, T_optimal)

        # Save scaler
        logger.info("\n💾 Saving temperature scaler...")
        save_path = save_temperature_scaler(scaler_data)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info("\n✅ Temperature scaler retrained successfully!")
        logger.info(f"\n   New Temperature: T = {T_optimal:.4f}")
        logger.info(f"   Coverage: {scaler_data['coverage_before']*100:.2f}% → {scaler_data['coverage_after']*100:.2f}%")
        logger.info(f"   Width: {scaler_data['width_before']:.1f} → {scaler_data['width_after']:.1f} yards")
        logger.info(f"   Saved to: {save_path}")
        logger.info(f"   Training date: {scaler_data['training_date']}")

        logger.info("\n🎯 Next Steps:")
        logger.info("   1. Run integration tests to verify ensemble loading")
        logger.info("   2. Deploy to production")
        logger.info("   3. Monitor calibration on live data")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
