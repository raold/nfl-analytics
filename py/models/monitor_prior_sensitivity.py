#!/usr/bin/env python3
"""
Monitor Phase 2 prior sensitivity training and analyze results.

Checks for completed training runs and extracts calibration metrics.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

def check_training_complete(sigma_values):
    """Check which training jobs have completed"""
    results_dir = Path('experiments/calibration')
    completed = []
    pending = []

    for sigma in sigma_values:
        results_file = results_dir / f"prior_sensitivity_sigma{sigma:.1f}.json"
        if results_file.exists():
            completed.append(sigma)
        else:
            pending.append(sigma)

    return completed, pending

def load_results(sigma_values):
    """Load results for completed training runs"""
    results_dir = Path('experiments/calibration')
    results = []

    for sigma in sigma_values:
        results_file = results_dir / f"prior_sensitivity_sigma{sigma:.1f}.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'sigma': sigma,
                    'coverage_90': data['calibration']['90%_coverage'],
                    'coverage_68': data['calibration']['68%_coverage'],
                    'mae': data['accuracy']['mae'],
                    'rmse': data['accuracy']['rmse'],
                    'timestamp': data['timestamp']
                })

    return pd.DataFrame(results) if results else None

def analyze_results(df):
    """Analyze completed results and make recommendations"""
    if df is None or len(df) == 0:
        return None

    # Sort by sigma
    df = df.sort_values('sigma')

    # Find optimal sigma (closest to 90% coverage target)
    df['coverage_error'] = abs(df['coverage_90'] - 0.90)
    optimal_idx = df['coverage_error'].idxmin()
    optimal = df.loc[optimal_idx]

    # Check if optimal is good enough (85-95% range)
    is_optimal = 0.85 <= optimal['coverage_90'] <= 0.95

    analysis = {
        'optimal_sigma': float(optimal['sigma']),
        'optimal_coverage_90': float(optimal['coverage_90']),
        'optimal_coverage_68': float(optimal['coverage_68']),
        'optimal_mae': float(optimal['mae']),
        'is_well_calibrated': is_optimal,
        'all_results': df.to_dict('records')
    }

    return analysis

def print_status():
    """Print current status of training jobs"""
    sigma_values = [0.5, 0.7, 1.0, 1.5]
    completed, pending = check_training_complete(sigma_values)

    print("="*80)
    print("PHASE 2 PRIOR SENSITIVITY - TRAINING STATUS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print(f"\nCompleted: {len(completed)}/4")
    for sigma in completed:
        print(f"  ✅ sigma={sigma}")

    print(f"\nPending: {len(pending)}/4")
    for sigma in pending:
        print(f"  ⏳ sigma={sigma}")

    # Load and display results
    if completed:
        print("\n" + "="*80)
        print("PRELIMINARY RESULTS")
        print("="*80)

        df = load_results(completed)
        if df is not None:
            print(f"\n{'Sigma':<8} {'90% Cov':<12} {'68% Cov':<12} {'MAE':<12} {'Status':<20}")
            print("-" * 64)

            for _, row in df.iterrows():
                coverage_90 = row['coverage_90']
                status = ""
                if 0.85 <= coverage_90 <= 0.95:
                    status = "✅ WELL CALIBRATED"
                elif coverage_90 > 0.40:
                    status = "↗ IMPROVED"
                else:
                    status = "❌ UNDER-CALIBRATED"

                print(f"{row['sigma']:<8.1f} {coverage_90:<12.1%} {row['coverage_68']:<12.1%} "
                      f"{row['mae']:<12.2f} {status:<20}")

            # Show optimal if all complete
            if len(completed) == 4:
                print("\n" + "="*80)
                print("FINAL ANALYSIS")
                print("="*80)

                analysis = analyze_results(df)
                if analysis:
                    print(f"\n🎯 Optimal sigma: {analysis['optimal_sigma']:.1f}")
                    print(f"   90% Coverage: {analysis['optimal_coverage_90']:.1%} (target: 85-95%)")
                    print(f"   68% Coverage: {analysis['optimal_coverage_68']:.1%} (target: 60-75%)")
                    print(f"   MAE: {analysis['optimal_mae']:.2f} yards")

                    if analysis['is_well_calibrated']:
                        print(f"\n✅ SUCCESS: Optimal calibration achieved!")
                        print(f"   Recommendation: Use sigma={analysis['optimal_sigma']:.1f} for production")
                    else:
                        print(f"\n⚠️  Suboptimal: Further tuning may be needed")

                        # Suggest next steps
                        if analysis['optimal_coverage_90'] < 0.85:
                            print(f"   Coverage still low - consider testing sigma > {max(completed):.1f}")
                        elif analysis['optimal_coverage_90'] > 0.95:
                            print(f"   Coverage too high - consider interpolating between tested values")

    print("\n" + "="*80)

def main():
    """Main monitoring loop"""
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        # Single check
        print_status()
    else:
        # Monitor until complete
        sigma_values = [0.5, 0.7, 1.0, 1.5]

        while True:
            completed, pending = check_training_complete(sigma_values)

            print_status()

            if len(pending) == 0:
                print("\n✅ All training jobs complete!")
                break

            print(f"\nWaiting for {len(pending)} jobs to complete...")
            print("Checking again in 2 minutes...")
            time.sleep(120)  # Check every 2 minutes

if __name__ == "__main__":
    main()
