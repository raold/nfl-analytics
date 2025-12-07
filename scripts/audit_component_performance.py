#!/usr/bin/env python3
"""
Component Performance Audit - Ensemble v6.0 with Recovered XGBoost v4

Analyzes validation results to understand why ensemble MAE is 55.16 yards
instead of expected ~35 yards.

This audit:
1. Loads validation results from all 3 seasons (2022-2024)
2. Calculates individual MAE for each component
3. Compares against expected performance benchmarks
4. Identifies underperforming components
5. Explains the 20-yard gap between theoretical and actual MAE

Created: November 19, 2025
Purpose: Diagnose ensemble performance gap after XGBoost v4 recovery
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def calculate_component_mae(df: pd.DataFrame, component_col: str) -> dict:
    """Calculate MAE for a single component."""
    # Filter to only games where component made a prediction
    valid = df[df[component_col].notna()].copy()

    if len(valid) == 0:
        return {
            'mae': None,
            'rmse': None,
            'n_games': 0,
            'coverage': 0.0
        }

    # Calculate MAE
    errors = np.abs(valid['actual'] - valid[component_col])
    mae = errors.mean()
    rmse = np.sqrt(((valid['actual'] - valid[component_col]) ** 2).mean())

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'n_games': len(valid),
        'coverage': len(valid) / len(df) * 100
    }


def calculate_theoretical_ensemble_mae(component_metrics: dict, weights: dict) -> float:
    """
    Calculate theoretical ensemble MAE using weighted average.

    This assumes components are independent and errors combine linearly.
    """
    weighted_mae = 0.0
    total_weight = 0.0

    for component, weight in weights.items():
        if component_metrics[component]['mae'] is not None:
            weighted_mae += component_metrics[component]['mae'] * weight
            total_weight += weight

    return weighted_mae / total_weight if total_weight > 0 else None


def main():
    """Run component performance audit."""
    print("\n" + "=" * 80)
    print("ENSEMBLE v6.0 COMPONENT PERFORMANCE AUDIT")
    print("=" * 80)
    print()
    print("Purpose: Investigate why ensemble MAE is 55.16 instead of expected ~35 yards")
    print()

    # Load validation results
    results_dir = project_root / "results" / "validation_v6_recovered_xgb"

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1

    # Load all season results
    all_results = []
    for season in [2022, 2023, 2024]:
        csv_file = results_dir / f"v6_recovered_xgb_{season}_predictions.csv"
        if not csv_file.exists():
            print(f"⚠️  Missing results for {season}: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        all_results.append(df)
        print(f"✅ Loaded {len(df)} games from {season}")

    if not all_results:
        print("❌ No validation results found")
        return 1

    # Combine all seasons
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\n✅ Total games: {len(combined_df)}")
    print()

    # Define component columns and expected performance
    components = {
        'state_space_pred': {
            'name': 'State-Space v2',
            'expected_mae': 47.77,
            'weight': 0.27
        },
        'hierarchical_pred': {
            'name': 'Hierarchical v2',
            'expected_mae': 45.0,  # Approximate
            'weight': 0.27
        },
        'xgboost_v4_pred': {
            'name': 'XGBoost v4 Ridge',
            'expected_mae': 4.27,
            'weight': 0.23
        },
        'informative_priors_pred': {
            'name': 'Informative Priors v2',
            'expected_mae': 40.0,  # Approximate
            'weight': 0.13
        },
        'hybrid_pred': {
            'name': 'Hybrid Static-Incremental',
            'expected_mae': 38.03,
            'weight': 0.10
        }
    }

    # Calculate actual performance for each component
    print("=" * 80)
    print("COMPONENT PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    component_metrics = {}

    for col, info in components.items():
        metrics = calculate_component_mae(combined_df, col)
        component_metrics[col] = metrics

        print(f"{info['name']}:")
        print(f"  Expected MAE: {info['expected_mae']:.2f} yards")
        print(f"  Actual MAE:   {metrics['mae']:.2f} yards" if metrics['mae'] else "  Actual MAE:   N/A")

        if metrics['mae'] is not None and info['expected_mae'] is not None:
            diff = metrics['mae'] - info['expected_mae']
            diff_pct = (diff / info['expected_mae']) * 100
            print(f"  Difference:   {diff:+.2f} yards ({diff_pct:+.1f}%)")

        print(f"  Coverage:     {metrics['coverage']:.1f}%")
        print(f"  Games:        {metrics['n_games']}/{len(combined_df)}")
        print()

    # Calculate theoretical ensemble MAE
    weights = {col: info['weight'] for col, info in components.items()}

    print("=" * 80)
    print("ENSEMBLE WEIGHTED AVERAGE ANALYSIS")
    print("=" * 80)
    print()

    # Theoretical MAE (using expected component performance)
    theoretical_mae = sum(
        info['expected_mae'] * info['weight']
        for info in components.values()
    )
    print("Theoretical MAE (expected component performance):")
    print(f"  {theoretical_mae:.2f} yards")
    print()

    # Theoretical MAE (using actual component performance)
    actual_theoretical_mae = calculate_theoretical_ensemble_mae(component_metrics, weights)
    print("Theoretical MAE (actual component performance):")
    print(f"  {actual_theoretical_mae:.2f} yards" if actual_theoretical_mae else "  N/A")
    print()

    # Actual ensemble MAE
    ensemble_mae = np.mean(np.abs(combined_df['actual'] - combined_df['predicted']))
    print("Actual Ensemble MAE:")
    print(f"  {ensemble_mae:.2f} yards")
    print()

    # Calculate gaps
    print("=" * 80)
    print("GAP ANALYSIS")
    print("=" * 80)
    print()

    gap1 = ensemble_mae - theoretical_mae
    gap1_pct = (gap1 / theoretical_mae) * 100
    print("Gap 1 (Actual Ensemble vs Expected Components):")
    print(f"  {gap1:+.2f} yards ({gap1_pct:+.1f}%)")
    print()

    if actual_theoretical_mae:
        gap2 = ensemble_mae - actual_theoretical_mae
        gap2_pct = (gap2 / actual_theoretical_mae) * 100
        print("Gap 2 (Actual Ensemble vs Theoretical Weighted Average):")
        print(f"  {gap2:+.2f} yards ({gap2_pct:+.1f}%)")
        print()

    # Identify underperforming components
    print("=" * 80)
    print("COMPONENT STATUS")
    print("=" * 80)
    print()

    for col, info in components.items():
        metrics = component_metrics[col]
        if metrics['mae'] is None:
            status = "❌ NOT WORKING"
        elif metrics['mae'] > info['expected_mae'] * 1.1:  # More than 10% worse
            status = f"⚠️  UNDERPERFORMING ({metrics['mae'] - info['expected_mae']:+.2f} yards)"
        elif metrics['mae'] < info['expected_mae'] * 0.9:  # More than 10% better
            status = f"✅ OUTPERFORMING ({metrics['mae'] - info['expected_mae']:+.2f} yards)"
        else:
            status = "✅ MEETING EXPECTATIONS"

        print(f"{info['name']}: {status}")

    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(f"Expected Ensemble MAE: {theoretical_mae:.2f} yards")
    print(f"Actual Ensemble MAE:   {ensemble_mae:.2f} yards")
    print(f"Gap:                   {gap1:+.2f} yards ({gap1_pct:+.1f}%)")
    print()

    # Save detailed report
    report = {
        'audit_date': pd.Timestamp.now().isoformat(),
        'total_games': len(combined_df),
        'ensemble_mae': float(ensemble_mae),
        'theoretical_mae_expected': float(theoretical_mae),
        'theoretical_mae_actual': float(actual_theoretical_mae) if actual_theoretical_mae else None,
        'gap_yards': float(gap1),
        'gap_pct': float(gap1_pct),
        'component_performance': {
            info['name']: {
                'expected_mae': info['expected_mae'],
                'actual_mae': component_metrics[col]['mae'],
                'difference': component_metrics[col]['mae'] - info['expected_mae'] if component_metrics[col]['mae'] else None,
                'weight': info['weight'],
                'coverage': component_metrics[col]['coverage'],
                'n_games': component_metrics[col]['n_games']
            }
            for col, info in components.items()
        }
    }

    report_file = results_dir / "component_performance_audit.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Detailed report saved: {report_file}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
