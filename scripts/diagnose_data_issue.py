#!/usr/bin/env python3
"""
Diagnose data issue: Why is validation data showing ~214 yards average
when database shows ~406 yards average?
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Load validation data
df = pd.read_csv('analysis/results/ensemble_4way_backtest_2022_2024_FIXED.csv')

print("="*80)
print("DATA ISSUE DIAGNOSIS")
print("="*80)
print()

print("1. VALIDATION DATA INSPECTION")
print("-" * 80)
print(f"Columns: {df.columns.tolist()}")
print()
print("First 5 rows:")
print(df[['game_id', 'player_name', 'actual_yards', 'ensemble_mean']].head())
print()
print("actual_yards statistics:")
print(df['actual_yards'].describe())
print()

print("\n2. CHECK FOR TEAM TOTALS VS PLAYER YARDS")
print("-" * 80)
# Database query: Get both team totals and player yards
conn = psycopg2.connect(
    host="localhost",
    port=5544,
    dbname="devdb01",
    user="dro",
    password="sicillionbillions"
)
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Sample game from validation data
sample_game_id = df.iloc[0]['game_id']
sample_player_id = df.iloc[0]['player_id']
sample_season = df.iloc[0]['season']
sample_week = df.iloc[0]['week']

print(f"Sample game: {sample_game_id}")
print(f"Sample player: {sample_player_id} ({df.iloc[0]['player_name']})")
print()

# Get player stats from database
player_query = """
SELECT
    pgs.player_id,
    ph.player_name,
    pgs.season,
    pgs.week,
    pgs.stat_yards,
    pgs.stat_attempts
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
WHERE pgs.player_id = %s
    AND pgs.season = %s
    AND pgs.week = %s
    AND pgs.stat_category = 'passing'
"""

cursor.execute(player_query, (sample_player_id, sample_season, sample_week))
db_result = cursor.fetchone()

if db_result:
    print("Database result:")
    print(f"  Player: {db_result['player_name']}")
    print(f"  Season/Week: {db_result['season']}/{db_result['week']}")
    print(f"  Passing yards: {db_result['stat_yards']}")
    print(f"  Attempts: {db_result['stat_attempts']}")
    print()

    # Compare to validation data
    val_yards = df.iloc[0]['actual_yards']
    print("Validation data:")
    print(f"  actual_yards: {val_yards}")
    print()

    if db_result['stat_yards'] != val_yards:
        print("*** DATA MISMATCH ***")
        print(f"  Database: {db_result['stat_yards']} yards")
        print(f"  Validation: {val_yards} yards")
        print(f"  Difference: {db_result['stat_yards'] - val_yards} yards")
    else:
        print("✓ Data matches")

print()
print("\n3. CHECK MULTIPLE SAMPLES")
print("-" * 80)
matches = 0
mismatches = 0
for i in range(min(20, len(df))):
    row = df.iloc[i]
    cursor.execute(player_query, (row['player_id'], row['season'], row['week']))
    db_result = cursor.fetchone()

    if db_result:
        if abs(db_result['stat_yards'] - row['actual_yards']) < 0.1:
            matches += 1
        else:
            mismatches += 1
            if mismatches <= 5:  # Show first 5 mismatches
                print(f"MISMATCH: {row['player_name']} Week {row['week']} {row['season']}")
                print(f"  DB: {db_result['stat_yards']}, Validation: {row['actual_yards']}")

print()
print(f"Sample check (n=20): {matches} matches, {mismatches} mismatches")

print()
print("\n4. HYPOTHESIS: ARE WE LOOKING AT DIFFERENT STAT CATEGORIES?")
print("-" * 80)

# Check all stat categories for sample player
all_stats_query = """
SELECT
    pgs.stat_category,
    pgs.stat_yards,
    pgs.stat_attempts
FROM mart.player_game_stats pgs
WHERE pgs.player_id = %s
    AND pgs.season = %s
    AND pgs.week = %s
ORDER BY pgs.stat_category
"""

cursor.execute(all_stats_query, (sample_player_id, sample_season, sample_week))
all_stats = cursor.fetchall()

print(f"All stats for {df.iloc[0]['player_name']} Week {sample_week} {sample_season}:")
for stat in all_stats:
    print(f"  {stat['stat_category']}: {stat['stat_yards']} yards, {stat['stat_attempts']} attempts")

cursor.close()
conn.close()

print()
print("="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
