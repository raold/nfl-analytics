#!/usr/bin/env python3
"""
Optimize 3-Way Ensemble Weights - XGBoost Excluded Due to Leakage

XGBoost v2 excluded due to critical target leakage discovered Nov 16, 2025:
- 91.4% feature importance on `residual_feature` containing actual_test_yards
- MAE of 2.72 yards (impossibly good - indicates model copying target)
- See docs/fixes/XGBOOST_LEAKAGE_AUDIT_NOV16.md for details

Implements 3 weighting strategies for the 3-way ensemble:
1. Inverse Variance Weighting: w_i = (1/σ²_i) / Σ(1/σ²_j)
2. Equal Weighting: w_i = 1/3 (baseline)
3. Performance-Based Weighting: w_i = (1/MAE_i) / Σ(1/MAE_j)

Current weights (arbitrary, for 4-way including XGBoost):
- State-Space v2: 30%
- Hierarchical v2: 30%
- XGBoost v2: 25% (EXCLUDED - leakage)
- Informative Priors v2: 15%

Usage:
    python scripts/optimize_3way_weights.py \\
        --predictions analysis/results/ensemble_4way_backtest_2022_2024_CORRECTED.csv \\
        --output models/ensemble/weights_optimized_3way_2022_2024.json

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


MODEL_NAMES = [
    'state_space',
    'hierarchical',
    'informative_priors'
]

MODEL_DISPLAY_NAMES = {
    'state_space': 'State-Space v2',
    'hierarchical': 'Hierarchical v2',
    'informative_priors': 'Informative Priors v2'
}


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load predictions from backtest CSV."""
    logger.info(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"  Loaded {len(df)} predictions")
    logger.info(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    logger.info(f"  Players: {df['player_id'].nunique()}")

    return df


def calculate_model_performance(df: pd.DataFrame, model_name: str) -> Dict:
    """Calculate performance metrics for a single model."""
    mean_col = f"{model_name}_mean"
    std_col = f"{model_name}_std"

    # Filter to predictions where this model contributed
    model_df = df[df[mean_col].notna()].copy()

    if len(model_df) == 0:
        logger.warning(f"No predictions found for {model_name}")
        return None

    # Calculate errors
    errors = model_df['actual_yards'] - model_df[mean_col]
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    bias = errors.mean()

    # Calculate average uncertainty
    avg_std = model_df[std_col].mean()
    avg_variance = (model_df[std_col] ** 2).mean()

    # Calculate calibration (90% CI)
    quantile_10 = model_df[mean_col] - 1.645 * model_df[std_col]
    quantile_90 = model_df[mean_col] + 1.645 * model_df[std_col]
    within_ci = (
        (model_df['actual_yards'] >= quantile_10) &
        (model_df['actual_yards'] <= quantile_90)
    ).sum()
    calibration_90 = within_ci / len(model_df) * 100

    return {
        'name': model_name,
        'display_name': MODEL_DISPLAY_NAMES[model_name],
        'n_predictions': len(model_df),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'avg_std': avg_std,
        'avg_variance': avg_variance,
        'calibration_90': calibration_90
    }


def inverse_variance_weighting(performances: List[Dict]) -> Dict[str, float]:
    """Calculate weights via inverse variance: w_i = (1/σ²_i) / Σ(1/σ²_j)."""
    inverse_variances = np.array([1.0 / p['avg_variance'] for p in performances])
    weights = inverse_variances / inverse_variances.sum()

    return {p['name']: w for p, w in zip(performances, weights)}


def equal_weighting(performances: List[Dict]) -> Dict[str, float]:
    """Equal weights: w_i = 1/N."""
    n = len(performances)
    return {p['name']: 1.0 / n for p in performances}


def performance_weighting(performances: List[Dict]) -> Dict[str, float]:
    """Performance-based weights: w_i = (1/MAE_i) / Σ(1/MAE_j)."""
    inverse_maes = np.array([1.0 / p['mae'] for p in performances])
    weights = inverse_maes / inverse_maes.sum()

    return {p['name']: w for p, w in zip(performances, weights)}


def evaluate_ensemble(df: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """Evaluate ensemble performance with given weights."""
    # Filter to predictions where all 3 models contributed
    valid_df = df.copy()
    for model in MODEL_NAMES:
        valid_df = valid_df[valid_df[f"{model}_mean"].notna()]

    if len(valid_df) == 0:
        logger.error("No predictions with all 3 models!")
        return None

    # Calculate weighted ensemble mean
    ensemble_mean = np.zeros(len(valid_df))
    for model, weight in weights.items():
        ensemble_mean += weight * valid_df[f"{model}_mean"].values

    # Calculate ensemble variance (within + between)
    within_var = np.zeros(len(valid_df))
    for model, weight in weights.items():
        within_var += weight * (valid_df[f"{model}_std"].values ** 2)

    between_var = np.zeros(len(valid_df))
    for model, weight in weights.items():
        between_var += weight * ((valid_df[f"{model}_mean"].values - ensemble_mean) ** 2)

    ensemble_var = within_var + between_var
    ensemble_std = np.sqrt(ensemble_var)

    # Calculate errors
    errors = valid_df['actual_yards'].values - ensemble_mean
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    bias = errors.mean()

    # Calculate calibration
    quantile_10 = ensemble_mean - 1.645 * ensemble_std
    quantile_90 = ensemble_mean + 1.645 * ensemble_std
    within_ci = (
        (valid_df['actual_yards'].values >= quantile_10) &
        (valid_df['actual_yards'].values <= quantile_90)
    ).sum()
    calibration_90 = within_ci / len(valid_df) * 100

    return {
        'n_predictions': len(valid_df),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'avg_std': ensemble_std.mean(),
        'calibration_90': calibration_90
    }


def optimize_weights(df: pd.DataFrame) -> Dict:
    """Run all 3 weighting methods and compare results."""
    logger.info("\n" + "="*80)
    logger.info("3-WAY ENSEMBLE WEIGHT OPTIMIZATION (XGBoost excluded due to leakage)")
    logger.info("="*80)

    # Step 1: Calculate individual model performance
    logger.info("\n📊 Individual Model Performance:")
    performances = []
    for model in MODEL_NAMES:
        perf = calculate_model_performance(df, model)
        if perf:
            performances.append(perf)
            logger.info(
                f"  {perf['display_name']:25s} | "
                f"{perf['n_predictions']:4d} preds | "
                f"MAE: {perf['mae']:5.2f} | "
                f"RMSE: {perf['rmse']:5.2f} | "
                f"Bias: {perf['bias']:6.2f} | "
                f"σ: {perf['avg_std']:5.2f} | "
                f"Cal: {perf['calibration_90']:5.1f}%"
            )

    if len(performances) < 3:
        logger.error("Need all 3 models to optimize!")
        return None

    # Step 2: Calculate weights using 3 methods
    methods = {
        'inverse_variance': inverse_variance_weighting(performances),
        'equal': equal_weighting(performances),
        'performance': performance_weighting(performances)
    }

    # Step 3: Evaluate each method
    logger.info("\n🔍 Evaluating Weight Methods:")
    results = {}
    for method_name, weights in methods.items():
        logger.info(f"\n  Method: {method_name}")
        logger.info(f"  Weights:")
        for model, weight in weights.items():
            logger.info(f"    {MODEL_DISPLAY_NAMES[model]:25s}: {weight:.3f} ({weight*100:.1f}%)")

        ensemble_perf = evaluate_ensemble(df, weights)
        logger.info(f"  Ensemble MAE: {ensemble_perf['mae']:.2f} yards")
        logger.info(f"  Ensemble RMSE: {ensemble_perf['rmse']:.2f} yards")
        logger.info(f"  Ensemble Bias: {ensemble_perf['bias']:.2f} yards")
        logger.info(f"  Calibration: {ensemble_perf['calibration_90']:.1f}%")

        results[method_name] = {
            'weights': weights,
            'performance': ensemble_perf
        }

    # Step 4: Find best method (lowest MAE)
    best_method = min(results.keys(), key=lambda m: results[m]['performance']['mae'])
    logger.info(f"\n✅ Best Method: {best_method}")
    logger.info(f"   MAE: {results[best_method]['performance']['mae']:.2f} yards")
    logger.info(f"   Bias: {results[best_method]['performance']['bias']:.2f} yards")

    # Step 5: Compare to old 4-way arbitrary weights (renormalized for 3-way)
    # Old: SS 30%, Hier 30%, XGB 25%, IP 15%
    # Renormalized (exclude XGB 25%): SS 40%, Hier 40%, IP 20%
    renormalized_weights = {
        'state_space': 0.40,
        'hierarchical': 0.40,
        'informative_priors': 0.20
    }
    renormalized_perf = evaluate_ensemble(df, renormalized_weights)
    logger.info(f"\n📊 Old Arbitrary Weights (renormalized, XGBoost excluded):")
    logger.info(f"   SS: 40%, Hier: 40%, IP: 20%")
    logger.info(f"   MAE: {renormalized_perf['mae']:.2f} yards")
    logger.info(f"   Bias: {renormalized_perf['bias']:.2f} yards")
    logger.info(f"   Calibration: {renormalized_perf['calibration_90']:.1f}%")

    improvement = renormalized_perf['mae'] - results[best_method]['performance']['mae']
    improvement_pct = improvement / renormalized_perf['mae'] * 100
    logger.info(f"\n💡 Improvement: {improvement:.2f} yards ({improvement_pct:.1f}%)")

    return {
        'best_method': best_method,
        'best_weights': results[best_method]['weights'],
        'best_performance': results[best_method]['performance'],
        'all_methods': results,
        'renormalized_weights': renormalized_weights,
        'renormalized_performance': renormalized_perf,
        'improvement_yards': improvement,
        'improvement_pct': improvement_pct,
        'individual_models': performances
    }


def main():
    parser = argparse.ArgumentParser(
        description='Optimize 3-way ensemble weights (XGBoost excluded)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to backtest CSV file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/ensemble/weights_optimized_3way.json'),
        help='Output JSON file for optimized weights'
    )

    args = parser.parse_args()

    # Load data
    df = load_predictions(args.predictions)

    # Optimize weights
    result = optimize_weights(df)

    if result is None:
        logger.error("Optimization failed!")
        return 1

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    output_data = {
        'best_method': result['best_method'],
        'recommended_weights': {
            MODEL_DISPLAY_NAMES[k]: float(v)
            for k, v in result['best_weights'].items()
        },
        'performance': {
            'mae': float(result['best_performance']['mae']),
            'rmse': float(result['best_performance']['rmse']),
            'bias': float(result['best_performance']['bias']),
            'calibration_90': float(result['best_performance']['calibration_90']),
            'n_predictions': int(result['best_performance']['n_predictions'])
        },
        'improvement_vs_renormalized': {
            'mae_yards': float(result['improvement_yards']),
            'mae_pct': float(result['improvement_pct'])
        },
        'all_methods': {
            method: {
                'weights': {MODEL_DISPLAY_NAMES[k]: float(v) for k, v in data['weights'].items()},
                'mae': float(data['performance']['mae']),
                'rmse': float(data['performance']['rmse']),
                'bias': float(data['performance']['bias']),
                'calibration_90': float(data['performance']['calibration_90'])
            }
            for method, data in result['all_methods'].items()
        },
        'notes': {
            'xgboost_excluded_reason': 'Critical target leakage - 91.4% feature importance on residual containing actual_test_yards',
            'xgboost_mae_before_exclusion': 2.72,
            'discovery_date': '2025-11-16',
            'audit_report': 'docs/fixes/XGBOOST_LEAKAGE_AUDIT_NOV16.md'
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✅ Results saved to: {args.output}")
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
