#!/usr/bin/env python3
"""
Test script to reproduce and diagnose State-Space v2 prediction bug

Expected: ~243 yards (verified by manual calculation)
Actual: 639 yards (140% error)

Root cause hypothesis: Stochastic noise sampling creating extreme outliers
"""

import numpy as np
from pathlib import Path

# Manual calculation (verified correct)
print("=" * 80)
print("MANUAL CALCULATION (VERIFIED CORRECT)")
print("=" * 80)

skill_mean = 2.6
beta_attempts = 0.810
beta_home = 0.0186
attempts = 35
is_home = 1

log_pred = skill_mean + beta_attempts * np.log(attempts) + beta_home * is_home
yards_expected = np.exp(log_pred) - 1

print(f"skill_mean: {skill_mean}")
print(f"beta_attempts: {beta_attempts}")
print(f"beta_home: {beta_home}")
print(f"log(attempts): {np.log(attempts):.4f}")
print(f"log_pred: {log_pred:.4f}")
print(f"yards_expected: {yards_expected:.1f} yards ✅")
print()

# Simulate predictor behavior with noise
print("=" * 80)
print("SIMULATING PREDICTOR WITH NOISE")
print("=" * 80)

R_mean = 0.214  # From training log
n_samples = 1000

# Simulate what the predictor does
theta_samp = np.random.normal(skill_mean, 0.1, size=n_samples)
beta_attempts_samp = np.random.normal(beta_attempts, 0.01, size=n_samples)
beta_home_samp = np.random.normal(beta_home, 0.005, size=n_samples)
R_samp = np.random.normal(R_mean, 0.02, size=n_samples)

log_attempts = np.log(attempts)

# This is what lines 224-235 do
log_yards_pred = (
    theta_samp
    + beta_attempts_samp * log_attempts
    + beta_home_samp * is_home
)

print(f"log_yards_pred before noise: mean={log_yards_pred.mean():.4f}, std={log_yards_pred.std():.4f}")
print(f"  -> yards before noise: mean={np.exp(log_yards_pred.mean()) - 1:.1f}")

# Add measurement noise (LINE 231 - POTENTIAL BUG SOURCE)
log_yards_pred += np.random.normal(0, R_samp)

print(f"log_yards_pred after noise: mean={log_yards_pred.mean():.4f}, std={log_yards_pred.std():.4f}")
print(f"  -> max log value: {log_yards_pred.max():.4f}")

# Transform back to yards (LINE 235)
yards_pred = np.exp(log_yards_pred) - 1

print(f"\nYards predictions:")
print(f"  Mean: {yards_pred.mean():.1f}")
print(f"  Std: {yards_pred.std():.1f}")
print(f"  Min: {yards_pred.min():.1f}")
print(f"  Max: {yards_pred.max():.1f}")
print(f"  95th percentile: {np.percentile(yards_pred, 95):.1f}")
print(f"  99th percentile: {np.percentile(yards_pred, 99):.1f}")

# Check for outliers
outliers = yards_pred > 500
if outliers.any():
    print(f"\n⚠️ FOUND {outliers.sum()} OUTLIERS > 500 yards:")
    outlier_indices = np.where(outliers)[0][:5]  # Show first 5
    for idx in outlier_indices:
        print(f"  Sample {idx}: {yards_pred[idx]:.1f} yards (log_pred={log_yards_pred[idx]:.4f})")

# Clip to reasonable range (LINE 238)
yards_pred_clipped = np.clip(yards_pred, 0, 600)

print(f"\nAfter clipping to [0, 600]:")
print(f"  Mean: {yards_pred_clipped.mean():.1f}")
print(f"  Max: {yards_pred_clipped.max():.1f}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if yards_pred_clipped.mean() > 300:
    print("❌ BUG CONFIRMED: Mean prediction > 300 yards (expected ~243)")
    print("   Root cause: Exponential transformation amplifies noise outliers")
    print("   Solution: Use mean R instead of sampling, or remove outliers before averaging")
elif yards_pred.max() > 600:
    print("⚠️ OUTLIERS DETECTED: Max prediction > 600 yards")
    print("   Clipping helps but doesn't fix root cause")
    print("   Solution: Use mean R or clip in log-space before exponentiating")
else:
    print("✅ No obvious bug in this simulation")
    print("   Bug may be in actual parameter loading or different sampling behavior")

print("\n" + "=" * 80)
print("RECOMMENDED FIX")
print("=" * 80)
print("Option 1: Use mean R instead of sampling from posterior")
print("  log_yards_pred += np.random.normal(0, self.params['R'])")
print("\nOption 2: Remove outliers before averaging")
print("  yards_pred = yards_pred[yards_pred < 500]  # Filter outliers")
print("\nOption 3: Clip in log-space before exponentiating")
print("  log_yards_pred = np.clip(log_yards_pred, 0, 7)  # max ~1000 yards")
