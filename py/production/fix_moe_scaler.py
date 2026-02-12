"""
Fix Phase 2.2 MoE model by adding missing scaler parameters

The model was trained on standardized features but the scaler wasn't saved.
This script reconstructs the scaler from the training data and updates the model.

Author: Richard Oldham
Date: October 2025
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import pickle
from pathlib import Path

import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("FIXING PHASE 2.2 MoE MODEL SCALER")
print("=" * 80)

# Database config
db_config = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Load the exact training data used in Phase 2.2
print("\nLoading training data (2020-2024)...")
conn = psycopg2.connect(**db_config)

query = """
SELECT
    pgs.player_id,
    pgs.player_display_name as player_name,
    pgs.season,
    pgs.week,
    pgs.current_team as team,
    pgs.stat_yards,
    pgs.stat_attempts as carries,
    AVG(pgs.stat_yards) OVER (
        PARTITION BY pgs.player_id
        ORDER BY pgs.season, pgs.week
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as avg_rushing_l3,
    AVG(pgs.stat_yards) OVER (
        PARTITION BY pgs.player_id, pgs.season
        ORDER BY pgs.week
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) as season_avg
FROM mart.player_game_stats pgs
WHERE pgs.season BETWEEN 2020 AND 2024
  AND pgs.stat_category = 'rushing'
  AND pgs.position_group IN ('RB', 'FB', 'HB')
  AND pgs.stat_attempts >= 5
  AND pgs.stat_yards IS NOT NULL
ORDER BY pgs.season, pgs.week, pgs.stat_yards DESC
"""

df = pd.read_sql(query, conn, params=[])
conn.close()

df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
    df.groupby("season")["stat_yards"].transform("median")
)
df["season_avg"] = df["season_avg"].fillna(df.groupby("season")["stat_yards"].transform("median"))

print(f"✓ Loaded {len(df)} rushing performances")

# Split into train (same as Phase 2.2)
train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 6))
df_train = df[train_mask].copy()

print(f"✓ Training set: {len(df_train)} samples")

# Fit scaler on training features
feature_cols = ["carries", "avg_rushing_l3", "season_avg", "week"]
X_train = df_train[feature_cols].fillna(0).values

scaler = StandardScaler()
scaler.fit(X_train)

print("\nScaler parameters:")
print(f"  Feature means: {scaler.mean_}")
print(f"  Feature stds: {scaler.scale_}")

# Load the saved model
model_path = Path("/Users/dro/rice/nfl-analytics/models/bayesian/bnn_mixture_experts_v2.pkl")
print(f"\nLoading model from {model_path}...")

with open(model_path, "rb") as f:
    save_obj = pickle.load(f)

print("✓ Model loaded")
print(f"  Keys before update: {list(save_obj.keys())}")

# Add scaler parameters and training shapes
save_obj["scaler_mean"] = scaler.mean_
save_obj["scaler_scale"] = scaler.scale_
save_obj["X_train_shape"] = X_train.shape
save_obj["y_train_shape"] = (len(df_train),)

print(f"\n  Keys after update: {list(save_obj.keys())}")

# Save updated model
print("\nSaving updated model...")
with open(model_path, "wb") as f:
    pickle.dump(save_obj, f)

print("✓ Model updated successfully!")
print("\nThe model now includes:")
print("  ✓ Scaler mean and scale parameters")
print("  ✓ Training data shapes")
print("\nProduction pipeline should now work correctly.")
