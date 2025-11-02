#!/usr/bin/env python3
"""
FIXED Historical Backtest - With NULL kickoff ordering fix
Uses asof_FIXED_v2.csv (na_position='last' applied)
Expected ROI: 1-3% (not 51,000%)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" FIXED HISTORICAL BACKTEST (NULL KICKOFF FIX APPLIED) ".center(70))
print("=" * 70)

# Load FIXED pre-game features
print("\n1. Loading FIXED pre-game features...")
features_path = Path("data/processed/features/asof_FIXED_v2.csv")

if not features_path.exists():
    print(f"[FAIL] Features not found at {features_path}")
    print("Run: bash -c 'source .venv/Scripts/activate && export PYTHONPATH=. && python py/features/asof_features.py --output data/processed/features/asof_FIXED_v2.csv --season-start 1999 --season-end 2024 --validate'")
    exit(1)

df = pd.read_csv(features_path)
print(f"[OK] Loaded {len(df):,} games with FIXED features")
print(f"[OK] Years: {df.season.min()}-{df.season.max()}")

# Filter to games with betting lines and outcomes
df = df[df['spread_close'].notna() & df['home_cover'].notna()].copy()
print(f"[OK] Games with spreads: {len(df):,}")

# Select CLEAN pre-game features only
clean_feature_cols = [
    'spread_close',
    'total_close',
    'prior_epa_mean_diff',
    'epa_pp_last3_diff',
    'margin_last3_diff',
    'prior_win_pct_diff',
    'rest_diff',
    'qb_change_diff',
    'coach_change_diff',
]

available_cols = [c for c in clean_feature_cols if c in df.columns]
print(f"\n[OK] Using {len(available_cols)} clean pre-game features")
print(f"[OK] Features: {', '.join(available_cols)}")

# Prepare data
X = df[available_cols].fillna(0).values
y_cover = df['home_cover'].astype(int).values

# Train/test split
test_start_season = 2020
train_mask = df['season'] < test_start_season
test_mask = df['season'] >= test_start_season

print(f"\n[OK] Train: {train_mask.sum():,} games (pre-{test_start_season})")
print(f"[OK] Test:  {test_mask.sum():,} games ({test_start_season}+)")

# Train logistic regression
print("\n2. Training Logistic Regression...")
model = LogisticRegression(max_iter=2000, random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X[train_mask])
X_test_scaled = scaler.transform(X[test_mask])

model.fit(X_train_scaled, y_cover[train_mask])

# Evaluate
train_acc = model.score(X_train_scaled, y_cover[train_mask])
test_acc = model.score(X_test_scaled, y_cover[test_mask])

print("\n" + "=" * 70)
print(" RESULTS ".center(70))
print("=" * 70)

print(f"\nTrain Accuracy: {train_acc:.1%}")
print(f"Test Accuracy:  {test_acc:.1%}")

# Comparison with original
print("\n" + "=" * 70)
print(" COMPARISON WITH ORIGINAL ".center(70))
print("=" * 70)

print("\nOriginal (with NULL kickoff leakage):")
print("  Test Accuracy: 62.4%")
print("  ROI: +51,106%")

print(f"\nFixed (na_position='last'):")
print(f"  Test Accuracy: {test_acc:.1%}")
print(f"  Improvement: {test_acc - 0.624:.1%} pp")

# Validation
print("\n" + "=" * 70)
print(" VALIDATION ".center(70))
print("=" * 70)

if test_acc > 0.55:
    print(f"[WARNING] Test accuracy {test_acc:.1%} > 55% - still higher than expected")
    print("          Expected: 52-54% for realistic models")
    print("          Additional leakage sources may exist")
elif test_acc >= 0.52:
    print(f"[OK] Test accuracy {test_acc:.1%} is in realistic range (52-54%)")
    print("     Within expected bounds for good NFL betting model")
else:
    print(f"[OK] Test accuracy {test_acc:.1%} is baseline range")

# Save results
results = {
    'test_accuracy': float(test_acc),
    'train_accuracy': float(train_acc),
    'original_accuracy': 0.624,
    'improvement_pp': float(test_acc - 0.624),
    'test_games': int(test_mask.sum()),
    'features_used': available_cols,
}

output_dir = Path("analysis/backtest")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "fixed_backtest_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Results saved to {output_dir}/fixed_backtest_results.json")
print("=" * 70)
