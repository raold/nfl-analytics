#!/usr/bin/env python3
"""
Verify bias calculation - why are models under-predicting when
validation data has LOWER average than training?
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Load ensemble predictions
df = pd.read_csv("analysis/results/ensemble_4way_backtest_2022_2024_FIXED.csv")

print("=" * 80)
print("BIAS CALCULATION VERIFICATION")
print("=" * 80)
print()

print("1. ENSEMBLE PREDICTIONS (2022-2024)")
print("-" * 80)
print(f"Mean actual yards: {df['actual_yards'].mean():.2f}")
print(f"Mean predicted yards: {df['ensemble_mean'].mean():.2f}")
print(f"Bias (actual - predicted): {df['actual_yards'].mean() - df['ensemble_mean'].mean():.2f}")
print()

print("2. INDIVIDUAL MODEL PREDICTIONS")
print("-" * 80)
for model in ["state_space", "hierarchical", "xgboost", "informative_priors"]:
    mean_col = f"{model}_mean"
    if mean_col in df.columns:
        valid = df[df[mean_col].notna()]
        if len(valid) > 0:
            print(f"\n{model.upper()}:")
            print(f"  Mean actual: {valid['actual_yards'].mean():.2f}")
            print(f"  Mean predicted: {valid[mean_col].mean():.2f}")
            print(f"  Bias: {valid['actual_yards'].mean() - valid[mean_col].mean():.2f}")

print()
print("\n3. CROSS-CHECK WITH DATABASE")
print("-" * 80)
conn = psycopg2.connect(
    host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
)
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Get actual QB passing yards from database (2022-2024)
query = """
SELECT
    season,
    AVG(stat_yards) as avg_yards,
    COUNT(*) as n_games
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
WHERE stat_category = 'passing'
    AND stat_yards IS NOT NULL
    AND stat_attempts >= 10
    AND season BETWEEN 2022 AND 2024
    AND ph.position_group = 'QB'
GROUP BY season
ORDER BY season
"""

cursor.execute(query)
db_data = pd.DataFrame(cursor.fetchall())

print("Database QB passing yards (2022-2024):")
for _, row in db_data.iterrows():
    print(f"  Season {row['season']}: {row['avg_yards']:.2f} yards ({row['n_games']} games)")

overall_avg = db_data["avg_yards"].mean()
print(f"\nOverall average: {overall_avg:.2f} yards")

# Compare to ensemble predictions
print()
print("4. COMPARISON")
print("-" * 80)
print(f"Database average (2022-2024): {overall_avg:.2f} yards")
print(f"Validation data average: {df['actual_yards'].mean():.2f} yards")
print(f"Ensemble predictions average: {df['ensemble_mean'].mean():.2f} yards")
print()
print(f"Ensemble bias vs database: {overall_avg - df['ensemble_mean'].mean():.2f} yards")
print(
    f"Ensemble bias vs validation: {df['actual_yards'].mean() - df['ensemble_mean'].mean():.2f} yards"
)

print()
print("\n5. HYPOTHESIS: SAMPLE SELECTION BIAS?")
print("-" * 80)
print("If validation data only includes specific players/games, it may not")
print("represent the full population of QB games in 2022-2024.")
print()

# Check player coverage
total_qb_games_query = """
SELECT COUNT(*) as total_games
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
WHERE stat_category = 'passing'
    AND stat_yards IS NOT NULL
    AND stat_attempts >= 10
    AND season BETWEEN 2022 AND 2024
    AND ph.position_group = 'QB'
"""
cursor.execute(total_qb_games_query)
total_games = cursor.fetchone()["total_games"]

print(f"Total QB games in database (2022-2024): {total_games:,}")
print(f"Games in validation data: {len(df):,}")
print(f"Coverage: {len(df) / total_games * 100:.1f}%")

if len(df) / total_games < 0.95:
    print()
    print("*** WARNING: Validation data does not cover all QB games ***")
    print("    This could introduce sample selection bias")

cursor.close()
conn.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
