#!/usr/bin/env python3
"""
Phase 5: Recent Seasons Test - Train 2022-2023, Test 2024
Alternative to 2025 test (no 2025 data available yet)

Tests model on most recent complete season with minimal lookback window.
Expected: 52-54% win rate if leakage is fully resolved
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 70)
print(" RECENT SEASONS TEST: Train 2022-2023, Test 2024 ".center(70))
print("=" * 70)

# Load FIXED pre-game features
print("\n1. Loading FIXED pre-game features...")
features_path = Path("data/processed/features/asof_FIXED_v2.csv")

if not features_path.exists():
    print(f"[FAIL] Features not found at {features_path}")
    print(
        "Run: bash -c 'source .venv/Scripts/activate && export PYTHONPATH=. && python nfl_analytics/features/asof_features.py --output data/processed/features/asof_FIXED_v2.csv --season-start 1999 --season-end 2024 --validate'"
    )
    exit(1)

df = pd.read_csv(features_path)
print(f"[OK] Loaded {len(df):,} games with FIXED features")
print(f"[OK] Years: {df.season.min()}-{df.season.max()}")

# Filter to games with betting lines and outcomes
df = df[df["spread_close"].notna() & df["home_cover"].notna()].copy()
print(f"[OK] Games with spreads: {len(df):,}")

# Select CLEAN pre-game features only
clean_feature_cols = [
    "spread_close",
    "total_close",
    "prior_epa_mean_diff",
    "epa_pp_last3_diff",
    "margin_last3_diff",
    "prior_win_pct_diff",
    "rest_diff",
    "qb_change_diff",
    "coach_change_diff",
]

available_cols = [c for c in clean_feature_cols if c in df.columns]
print(f"\n[OK] Using {len(available_cols)} clean pre-game features")
print(f"[OK] Features: {', '.join(available_cols)}")

# Prepare data
X = df[available_cols].fillna(0).values
y_cover = df["home_cover"].astype(int).values

# Train/test split: Train 2022-2023, Test 2024
train_start_season = 2022
test_season = 2024
train_mask = (df["season"] >= train_start_season) & (df["season"] < test_season)
test_mask = df["season"] == test_season

print(f"\n[OK] Train: {train_mask.sum():,} games ({train_start_season}-{test_season-1})")
print(f"[OK] Test:  {test_mask.sum():,} games ({test_season})")

if train_mask.sum() == 0 or test_mask.sum() == 0:
    print("[FAIL] No games in train or test set")
    exit(1)

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

print(f"\nTrain Accuracy ({train_start_season}-{test_season-1}): {train_acc:.1%}")
print(f"Test Accuracy ({test_season}):  {test_acc:.1%}")

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": available_cols, "coefficient": model.coef_[0]}
).sort_values("coefficient", key=abs, ascending=False)

print("\n" + "=" * 70)
print(" FEATURE IMPORTANCE ".center(70))
print("=" * 70)
print("\nTop 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['coefficient']:.4f}")

# Comparison
print("\n" + "=" * 70)
print(" COMPARISON WITH OTHER TESTS ".center(70))
print("=" * 70)

print("\nPrevious Tests:")
print("  Original (severe leakage): 4,469,906% ROI")
print("  Clean (51K% ROI): 62.4% win rate")
print("  FIXED (NULL kickoff): 60.2% win rate")
print(f"\nRecent Seasons (2022-2023 to 2024): {test_acc:.1%}")

# Validation
print("\n" + "=" * 70)
print(" VALIDATION ".center(70))
print("=" * 70)

if test_acc > 0.55:
    print(f"[WARNING] Test accuracy {test_acc:.1%} > 55% - still higher than expected")
    print("          Expected: 52-54% for realistic models")
    print("          Additional leakage sources likely exist")
    print("\n[RECOMMENDATION] Remove season-level features (season_win_pct, etc.)")
elif test_acc >= 0.52:
    print(f"[OK] Test accuracy {test_acc:.1%} is in realistic range (52-54%)")
    print("     Within expected bounds for good NFL betting model")
else:
    print(f"[OK] Test accuracy {test_acc:.1%} is baseline range")

# Save results
results = {
    "test_accuracy": float(test_acc),
    "train_accuracy": float(train_acc),
    "test_season": test_season,
    "train_seasons": f"{train_start_season}-{test_season-1}",
    "test_games": int(test_mask.sum()),
    "train_games": int(train_mask.sum()),
    "features_used": available_cols,
    "feature_importance": feature_importance.to_dict("records"),
}

output_dir = Path("analysis/backtest")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "recent_seasons_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Results saved to {output_dir}/recent_seasons_test_results.json")
print("=" * 70)

# Note about 2025 data
print("\n" + "=" * 70)
print(" NOTE: 2025 DATA NOT AVAILABLE ".center(70))
print("=" * 70)
print("\nOriginal Phase 5 requested: Train 2022-2024, Test 2025 Week 1-9")
print("Issue: No 2025 data in database (season hasn't started yet)")
print("Alternative: Train 2022-2023, Test 2024 (most recent complete season)")
print("=" * 70)
