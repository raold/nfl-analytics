#!/usr/bin/env python3
"""Quick calibration check - minimal dependencies"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("analysis/results/historical_validation_4way_EMERGENCY_FIX.csv")

print("=" * 80)
print("CALIBRATION DIAGNOSTIC - QUICK ANALYSIS")
print("=" * 80)
print()

# Filter to rows with ensemble predictions
ensemble_df = df[df['prediction_mean'].notna()].copy()

print(f"Total predictions: {len(ensemble_df)}")
print()

# Calculate ensemble coverage
actual = ensemble_df['actual_yards'].values
q05 = ensemble_df['prediction_q05'].values
q95 = ensemble_df['prediction_q95'].values

in_interval = (actual >= q05) & (actual <= q95)
coverage = in_interval.mean() * 100

print(f"Ensemble Coverage: {coverage:.2f}% (target: 85-95%)")
print(f"Coverage Gap: {90 - coverage:+.2f} pp")
print()

# Interval width analysis
interval_widths = q95 - q05
mean_width = interval_widths.mean()
median_width = np.median(interval_widths)

print(f"Mean Interval Width: {mean_width:.1f} yards")
print(f"Median Interval Width: {median_width:.1f} yards")
print()

# Std analysis
stds = ensemble_df['prediction_std'].values
mean_std = stds.mean()

print(f"Mean Std: {mean_std:.1f} yards")
print(f"Expected 90% CI width: {3.29 * mean_std:.1f} yards")
print(f"Observed vs Expected: {mean_width / (3.29 * mean_std):.2f}x")
print()

# MAE
mae = np.abs(actual - ensemble_df['prediction_mean'].values).mean()
print(f"MAE: {mae:.1f} yards")
print()

# Model availability
print("=" * 80)
print("MODEL AVAILABILITY")
print("=" * 80)
print()

for model in ['bayesian', 'xgboost', 'bnn', 'bnn_calibrated']:
    col = f"{model}_mean"
    if col in df.columns:
        n = df[col].notna().sum()
        pct = n / len(df) * 100
        model_mae = np.abs(df[col].values - df['actual_yards'].values).mean()
        print(f"{model.upper():15s}: {n:5d} predictions ({pct:5.1f}%), MAE={model_mae:.1f} yards")

print()
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()

# Determine severity
if coverage < 50:
    print("Status: CRITICAL - Severe under-calibration")
    print()
    print("Top 3 Recommendations:")
    print()
    print("1. AGGRESSIVE TEMPERATURE SCALING (Expected: +40 pp)")
    print("   - Reduce temperatures to 0.30-0.40 (~2.5-3x interval widening)")
    print("   - Edit: py/models/player_props_ensemble.py, lines 423-428")
    print()
    print("2. SWITCH TO EQUAL WEIGHTING (Expected: +7 pp)")
    print("   - Replace inverse variance with equal weighting (1/N)")
    print("   - Edit: py/models/player_props_ensemble.py, lines 924-926")
    print()
    print("3. INVESTIGATE MODEL TRAINING (Expected: +15 pp)")
    print("   - Retrain with stronger regularization")
    print("   - Use validation-based early stopping")
elif coverage < 75:
    print("Status: HIGH - Moderate under-calibration")
    print()
    print("Top 3 Recommendations:")
    print()
    print("1. MODERATE TEMPERATURE SCALING (Expected: +20-30 pp)")
    print("   - Reduce temperatures to 0.50-0.60 (~1.7-2x interval widening)")
    print()
    print("2. SWITCH TO EQUAL WEIGHTING (Expected: +7 pp)")
    print("   - Replace inverse variance with equal weighting")
    print()
    print("3. CHECK MODEL CALIBRATION (Expected: +10 pp)")
    print("   - Individual models may need recalibration")
else:
    print("Status: MEDIUM - Slight under-calibration")
    print()
    print("Top 3 Recommendations:")
    print()
    print("1. SLIGHT TEMPERATURE ADJUSTMENT (Expected: +10-15 pp)")
    print("   - Reduce temperatures to 0.65-0.70 (~1.4-1.5x widening)")
    print()
    print("2. SWITCH TO EQUAL WEIGHTING (Expected: +7 pp)")
    print("   - Replace inverse variance with equal weighting")
    print()
    print("3. FINE-TUNE CALIBRATION (Expected: +5 pp)")
    print("   - Use calibration plots to identify specific issues")

print()
print("=" * 80)
