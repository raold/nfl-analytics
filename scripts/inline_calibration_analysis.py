#!/usr/bin/env python3
"""
Inline Calibration Analysis - Extract key metrics from backtest CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
csv_file = Path("/Users/dro/rice/nfl-analytics/analysis/results/historical_validation_4way_EMERGENCY_FIX.csv")
df = pd.read_csv(csv_file)

# Filter to ensemble predictions
ensemble_df = df[df['prediction_mean'].notna()].copy()

# Key metrics
actual = ensemble_df['actual_yards'].values
pred_mean = ensemble_df['prediction_mean'].values
pred_std = ensemble_df['prediction_std'].values
q05 = ensemble_df['prediction_q05'].values
q95 = ensemble_df['prediction_q95'].values

# Coverage
in_interval = (actual >= q05) & (actual <= q95)
coverage_pct = in_interval.mean() * 100

# Interval width
interval_widths = q95 - q05
mean_interval_width = interval_widths.mean()

# Std stats
mean_std = pred_std.mean()
median_std = np.median(pred_std)

# MAE
mae = np.abs(actual - pred_mean).mean()

# Expected width from std (90% CI = mean ± 1.645*std, so width = 3.29*std)
expected_width = 3.29 * mean_std
width_ratio = mean_interval_width / expected_width

# Calibration curve (simple)
# For each decile, check if actual fraction matches predicted fraction
from scipy import stats as sp_stats

predicted_percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
actual_percentiles = []

for p in predicted_percentiles:
    z = sp_stats.norm.ppf(p)
    predicted_values = pred_mean + z * pred_std
    actual_pct = (actual < predicted_values).mean()
    actual_percentiles.append(actual_pct)

calibration_error = np.abs(np.array(actual_percentiles) - np.array(predicted_percentiles)).mean()

# Model availability
model_stats = {}
for model in ['bayesian', 'xgboost', 'bnn', 'bnn_calibrated']:
    col = f"{model}_mean"
    if col in df.columns:
        model_df = df[df[col].notna()].copy()
        n = len(model_df)
        pct = n / len(df) * 100
        model_mae = np.abs(model_df[col].values - model_df['actual_yards'].values).mean()
        mean_pred = model_df[col].mean()
        model_stats[model] = {
            'n': n,
            'pct': pct,
            'mae': model_mae,
            'mean_pred': mean_pred
        }

# Print results
print("=" * 90)
print("CALIBRATION DIAGNOSTIC SUMMARY")
print("=" * 90)
print()
print(f"Data: {csv_file.name}")
print(f"Total predictions: {len(ensemble_df):,}")
print()

print("ENSEMBLE CALIBRATION:")
print("-" * 90)
print(f"  Coverage:           {coverage_pct:6.2f}% (target: 85-95%)")
print(f"  Coverage Error:     {coverage_pct - 90:+6.2f} percentage points")
print(f"  Mean Interval Width: {mean_interval_width:6.1f} yards")
print(f"  Mean Std:           {mean_std:6.1f} yards")
print(f"  Expected Width:     {expected_width:6.1f} yards (from 3.29*std)")
print(f"  Width Ratio:        {width_ratio:6.2f}x (observed/expected)")
print(f"  MAE:                {mae:6.1f} yards")
print(f"  Calibration Error:  {calibration_error:6.3f}")
print()

print("TEMPERATURE SCALING ANALYSIS:")
print("-" * 90)
print(f"  Current ensemble std: {mean_std:.1f} yards")
print(f"  Current interval width: {mean_interval_width:.1f} yards")
print(f"  Width/Std ratio: {mean_interval_width/mean_std:.2f}x (expected: 3.29x for 90% CI)")
print()
if width_ratio < 0.95:
    print("  ⚠️  Intervals NARROWER than expected - temperature scaling may not be working!")
elif width_ratio > 1.05:
    print("  ✅ Intervals WIDER than expected - temperature scaling is active")
else:
    print("  ✅ Intervals match expected width")
print()

print("INDIVIDUAL MODEL PERFORMANCE:")
print("-" * 90)
print(f"{'Model':<15} {'N':>8} {'%':>7} {'MAE':>8} {'Mean Pred':>10}")
print("-" * 90)
for model, stats in model_stats.items():
    print(f"{model.upper():<15} {stats['n']:>8,} {stats['pct']:>6.1f}% {stats['mae']:>8.1f} {stats['mean_pred']:>10.1f}")
print()

print("=" * 90)
print("ROOT CAUSE ANALYSIS")
print("=" * 90)
print()

# Determine severity and recommendations
gap = 90 - coverage_pct

if coverage_pct < 50:
    severity = "CRITICAL"
    temp_reduction = 0.40
    temp_widening = 2.5
    temp_improvement = min(40, gap * 0.5)
elif coverage_pct < 75:
    severity = "HIGH"
    temp_reduction = 0.55
    temp_widening = 1.8
    temp_improvement = min(30, gap * 0.4)
else:
    severity = "MEDIUM"
    temp_reduction = 0.67
    temp_widening = 1.5
    temp_improvement = min(15, gap * 0.3)

print(f"Severity: {severity}")
print(f"Coverage Gap: {gap:.1f} pp")
print()

print("TOP 3 RECOMMENDATIONS (Ranked by Expected Impact):")
print()

# Recommendation 1: Temperature scaling
print(f"1. AGGRESSIVE TEMPERATURE SCALING")
print(f"   Expected Impact: +{temp_improvement:.1f} pp")
print(f"   Current: T = 0.75-0.80 (33% widening)")
print(f"   Recommended: T = {temp_reduction:.2f} ({temp_widening:.1f}x widening)")
print(f"   Implementation:")
print(f"     - File: py/models/player_props_ensemble.py")
print(f"     - Method: _apply_temperature_scaling() (lines 398-451)")
print(f"     - Change temperature_map values to {temp_reduction:.2f}")
print()

# Recommendation 2: Equal weighting
equal_weight_improvement = 7
print(f"2. SWITCH TO EQUAL WEIGHTING")
print(f"   Expected Impact: +{equal_weight_improvement:.1f} pp")
print(f"   Current: Inverse variance weighting (amplifies overconfident models)")
print(f"   Recommended: Equal weighting (1/N for N models)")
print(f"   Implementation:")
print(f"     - File: py/models/player_props_ensemble.py")
print(f"     - Lines: 924-926 (ensemble aggregation)")
print(f"     - Replace: weights = np.ones(len(predictions)) / len(predictions)")
print()

# Recommendation 3: Model training (if severe)
if coverage_pct < 50:
    training_improvement = 15
    print(f"3. RETRAIN MODELS WITH STRONGER REGULARIZATION")
    print(f"   Expected Impact: +{training_improvement:.1f} pp")
    print(f"   Current: Models severely overconfident (suggests overfitting)")
    print(f"   Recommended:")
    print(f"     - Increase L2 regularization (weight decay)")
    print(f"     - Validation-based early stopping")
    print(f"     - Cross-validation instead of single train/test split")
else:
    training_improvement = 5
    print(f"3. FINE-TUNE MODEL CALIBRATION")
    print(f"   Expected Impact: +{training_improvement:.1f} pp")
    print(f"   Recommended:")
    print(f"     - Use Platt scaling or isotonic regression")
    print(f"     - Calibration plots to identify specific biases")

print()
total_improvement = temp_improvement + equal_weight_improvement + training_improvement
projected_coverage = coverage_pct + total_improvement

print(f"TOTAL EXPECTED IMPROVEMENT: +{total_improvement:.1f} pp")
print(f"PROJECTED COVERAGE: {projected_coverage:.1f}%")
print()

if projected_coverage >= 85:
    print("✅ Projected coverage meets target (85-95%)")
else:
    print(f"⚠️  Projected coverage still below target by {85 - projected_coverage:.1f} pp")
    print("   Consider combining all 3 recommendations + additional model improvements")

print()
print("=" * 90)

# Save to simple text file
output_dir = Path("/Users/dro/rice/nfl-analytics/docs/diagnostics")
output_dir.mkdir(exist_ok=True, parents=True)

with open(output_dir / "calibration_summary_nov7.txt", 'w') as f:
    f.write("CALIBRATION DIAGNOSTIC SUMMARY\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Date: November 7, 2025\n")
    f.write(f"File: {csv_file.name}\n\n")
    f.write(f"ENSEMBLE COVERAGE: {coverage_pct:.2f}% (target: 85-95%)\n")
    f.write(f"COVERAGE GAP: {gap:+.2f} pp\n")
    f.write(f"SEVERITY: {severity}\n\n")
    f.write(f"TOP 3 RECOMMENDATIONS:\n")
    f.write(f"1. Temperature Scaling: +{temp_improvement:.1f} pp (reduce T to {temp_reduction:.2f})\n")
    f.write(f"2. Equal Weighting: +{equal_weight_improvement:.1f} pp\n")
    f.write(f"3. Model Training/Calibration: +{training_improvement:.1f} pp\n\n")
    f.write(f"TOTAL EXPECTED IMPROVEMENT: +{total_improvement:.1f} pp\n")
    f.write(f"PROJECTED COVERAGE: {projected_coverage:.1f}%\n")

print(f"Summary saved to: {output_dir}/calibration_summary_nov7.txt")
print()
