#!/usr/bin/env python3
"""
Diagnose BNN Receiving data quality issues causing -inf log probability.
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import numpy as np
import pandas as pd
import psycopg2

# Database connection
db_config = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

print("=" * 60)
print("BNN RECEIVING DATA QUALITY DIAGNOSTIC")
print("=" * 60)

# Load data
conn = psycopg2.connect(**db_config)

query = """
WITH receiving_data AS (
    SELECT
        p.game_id,
        p.receiver_player_id as player_id,
        p.receiver_player_name as player_name,
        g.season,
        g.week,
        p.posteam as team,
        p.defteam as opponent,
        -- Aggregate receiving stats
        COUNT(CASE WHEN p.pass_attempt THEN 1 END) as targets,
        COUNT(CASE WHEN p.complete_pass = 1 THEN 1 END) as receptions,
        SUM(CASE WHEN p.complete_pass = 1 THEN p.receiving_yards ELSE 0 END) as receiving_yards,
        AVG(CASE WHEN p.complete_pass = 1 THEN p.receiving_yards END) as yards_per_reception,
        AVG(p.air_yards) as avg_air_yards,
        AVG(p.yards_after_catch) as avg_yac
    FROM plays p
    JOIN games g ON p.game_id = g.game_id
    WHERE g.season BETWEEN 2020 AND 2024
      AND p.season_type = 'REG'
      AND p.receiver_player_id IS NOT NULL
      AND p.pass_attempt
    GROUP BY
        p.game_id, p.receiver_player_id, p.receiver_player_name,
        g.season, g.week, p.posteam, p.defteam
)
SELECT
    rd.*,
    LAG(rd.receiving_yards, 1) OVER (PARTITION BY rd.player_id ORDER BY rd.season, rd.week) as prev_game_yards,
    LAG(rd.receiving_yards, 2) OVER (PARTITION BY rd.player_id ORDER BY rd.season, rd.week) as prev2_game_yards,
    LAG(rd.receiving_yards, 3) OVER (PARTITION BY rd.player_id ORDER BY rd.season, rd.week) as prev3_game_yards
FROM receiving_data rd
WHERE rd.targets >= 3
ORDER BY rd.season, rd.week, rd.receiving_yards DESC
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"\n✓ Loaded {len(df)} receiving performances")
print(f"  Players: {df['player_id'].nunique()}")
print(f"  Date range: {df['season'].min()}-{df['season'].max()}")

# Calculate rolling averages
df["avg_receiving_l3"] = df[
    ["prev_game_yards", "prev2_game_yards", "prev3_game_yards"]
].mean(axis=1)

# Check for NaN and inf
print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

feature_cols = [
    "targets",
    "receptions",
    "avg_receiving_l3",
    "avg_air_yards",
    "avg_yac",
    "week",
    "receiving_yards",
]

for col in feature_cols:
    if col not in df.columns:
        print(f"⚠️  Missing column: {col}")
        continue

    nan_count = df[col].isna().sum()
    inf_count = np.isinf(df[col]).sum()

    print(f"\n{col}:")
    print(f"  NaN: {nan_count} ({nan_count/len(df)*100:.1f}%)")
    print(f"  Inf: {inf_count}")
    print(f"  Min: {df[col].min():.2f}")
    print(f"  Max: {df[col].max():.2f}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Std: {df[col].std():.2f}")

    # Check for extreme outliers (> 5 std)
    if not df[col].isna().all():
        mean = df[col].mean()
        std = df[col].std()
        outliers = np.abs(df[col] - mean) > 5 * std
        outlier_count = outliers.sum()

        if outlier_count > 0:
            print(f"  ⚠️  Outliers (>5σ): {outlier_count}")
            print(f"     Values: {df.loc[outliers, col].values[:5]}")

# Check target variable (receiving_yards) specifically
print("\n" + "=" * 60)
print("TARGET VARIABLE ANALYSIS (receiving_yards)")
print("=" * 60)

y = df["receiving_yards"].values
print(f"Shape: {y.shape}")
print(f"Min: {y.min()}")
print(f"Max: {y.max()}")
print(f"Mean: {y.mean():.2f}")
print(f"Std: {y.std():.2f}")
print(f"Zeros: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"Negatives: {(y < 0).sum()}")

# Log transform check
print("\nLog transform (log1p):")
y_log = np.log1p(y)
print(f"  Min: {y_log.min():.4f}")
print(f"  Max: {y_log.max():.4f}")
print(f"  Mean: {y_log.mean():.4f}")
print(f"  Std: {y_log.std():.4f}")
print(f"  -inf count: {np.isinf(y_log).sum()}")
print(f"  NaN count: {np.isnan(y_log).sum()}")

# Check train/test split
print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)

train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["week"] <= 6))
test_mask = (df["season"] == 2024) & (df["week"] > 6)

print(f"Training samples: {train_mask.sum()}")
print(f"Test samples: {test_mask.sum()}")
print(f"Training yards: min={df.loc[train_mask, 'receiving_yards'].min()}, max={df.loc[train_mask, 'receiving_yards'].max()}")
print(f"Test yards: min={df.loc[test_mask, 'receiving_yards'].min()}, max={df.loc[test_mask, 'receiving_yards'].max()}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
