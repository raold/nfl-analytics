#!/usr/bin/env python3
"""
Diagnostic Test 1: Spread-Only Baseline
Tests model with ONLY spread_close feature to detect leakage.
Expected: 52-53% win rate (random + market info, minus vig)
If >55%: Confirms data leakage in feature pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" DIAGNOSTIC TEST 1: SPREAD-ONLY BASELINE ".center(70))
print("=" * 70)

# Load clean pre-game features
features_path = Path("data/processed/features/asof_clean_backtest.csv")
df = pd.read_csv(features_path)

# Filter to games with betting lines and outcomes
df = df[df['spread_close'].notna() & df['home_cover'].notna()].copy()
print(f"\n[OK] Loaded {len(df):,} games with spreads")
print(f"[OK] Years: {df.season.min()}-{df.season.max()}")

# SPREAD ONLY - no other features
X_spread = df[['spread_close']].fillna(0).values
y_cover = df['home_cover'].astype(int).values

# Train/test split
test_start_season = 2020
train_mask = df['season'] < test_start_season
test_mask = df['season'] >= test_start_season

print(f"\n[OK] Train: {train_mask.sum():,} games (pre-{test_start_season})")
print(f"[OK] Test:  {test_mask.sum():,} games ({test_start_season}+)")

# Train model
model = LogisticRegression(max_iter=2000, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_spread[train_mask])
X_test_scaled = scaler.transform(X_spread[test_mask])

model.fit(X_train_scaled, y_cover[train_mask])

# Evaluate
train_acc = model.score(X_train_scaled, y_cover[train_mask])
test_acc = model.score(X_test_scaled, y_cover[test_mask])

print("\n" + "=" * 70)
print(" RESULTS ".center(70))
print("=" * 70)

print(f"\nTrain Accuracy: {train_acc:.1%}")
print(f"Test Accuracy:  {test_acc:.1%}")

# Validation checks
print("\n" + "=" * 70)
print(" VALIDATION ".center(70))
print("=" * 70)

if test_acc > 0.55:
    print(f"[FAIL] Test accuracy {test_acc:.1%} > 55% - LEAKAGE CONFIRMED!")
    print("       Spread-only should achieve ~52-53% max.")
elif test_acc > 0.53:
    print(f"[WARNING] Test accuracy {test_acc:.1%} is higher than expected")
    print("          Expected: 52-53% for spread-only baseline")
elif test_acc >= 0.50:
    print(f"[OK] Test accuracy {test_acc:.1%} is reasonable")
    print("     Within expected range for spread-only (50-53%)")
else:
    print(f"[UNEXPECTED] Test accuracy {test_acc:.1%} < 50%")
    print("             Model performs worse than random - check implementation")

# Additional diagnostics
print(f"\n[INFO] Coefficient: {model.coef_[0][0]:.4f}")
print(f"[INFO] Intercept:   {model.intercept_[0]:.4f}")

# Baseline comparison
baseline_acc = y_cover[test_mask].mean()  # Home team cover rate
print(f"\n[BASELINE] Home team covers spread: {baseline_acc:.1%}")
print(f"[BASELINE] Random guessing: 50.0%")

print("\n" + "=" * 70)
