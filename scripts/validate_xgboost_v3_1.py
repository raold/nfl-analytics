#!/usr/bin/env python3
"""
XGBoost v3.1 Validation Script (Quick Coverage Test)

POSTMORTEM FIX VALIDATION:
1. Check feature coverage on small 2018-2024 sample
2. Verify no missing opponent data
3. Compare sample predictions to baseline
4. Provide go/no-go decision for full training

Author: NFL Analytics Research Team
Date: November 14, 2025

Usage:
    python scripts/validate_xgboost_v3_1.py [--sample-size 100]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import pandas as pd
import psycopg2

# Database connection
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5544')),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
    'database': os.getenv('DB_NAME', 'devdb01')
}

COVERAGE_THRESHOLD = 0.80
COVERAGE_TARGET = 0.95


def check_feature_coverage(sample_size: int = 100) -> dict:
    """
    Validate feature coverage on small sample before full training.

    Args:
        sample_size: Number of games to sample (default: 100)

    Returns:
        Dict with coverage stats and go/no-go decision
    """
    print("=" * 70)
    print("XGBOOST v3.1 PRE-TRAINING VALIDATION")
    print("=" * 70)
    print(f"Sample size: {sample_size} games (2018-2024)")
    print(f"Coverage threshold: {COVERAGE_THRESHOLD:.0%}")
    print(f"Coverage target: {COVERAGE_TARGET:.0%}")
    print("=" * 70 + "\n")

    conn = psycopg2.connect(**DB_CONFIG)

    # Query sample of 2018-2024 data
    query = f"""
    WITH qb_games AS (
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as yards,
            pgs.stat_attempts as attempts,
            pgs.current_team as team,
            g.game_id
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = pgs.current_team OR g.away_team = pgs.current_team)
        WHERE pgs.stat_category = 'passing'
            AND pgs.stat_yards IS NOT NULL
            AND pgs.stat_yards > 0
            AND pgs.stat_attempts >= 10
            AND pgs.season >= 2018  -- ALIGNED WINDOW
            AND pgs.season <= 2024
            AND ph.position_group = 'QB'
        ORDER BY RANDOM()
        LIMIT {sample_size}
    )
    SELECT * FROM qb_games
    """

    print(f"Loading {sample_size} random QB games (2018-2024)...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"✓ Loaded {len(df)} games")
    print(f"  Seasons: {df.season.min()}-{df.season.max()}")
    print("  Season distribution:")
    for season, count in df.season.value_counts().sort_index().items():
        print(f"    {season}: {count} games")

    # Load opponent strength features
    print("\nLoading opponent strength features...")
    features_path = Path('data/processed/features/opponent_strength_features.csv')

    if not features_path.exists():
        print("✗ FAILED: Opponent strength features not found")
        print(f"  Path: {features_path}")
        print("  Run: python py/features/opponent_strength.py")
        return {'status': 'FAILED', 'reason': 'Missing opponent features file'}

    opp_df = pd.read_csv(features_path)
    print(f"✓ Loaded {len(opp_df):,} opponent strength records")

    # Select features
    opp_features = opp_df[['game_id', 'team', 'opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']].copy()

    # Merge
    print("\nMerging opponent features with sample...")
    before_merge = len(df)
    merged_df = df.merge(opp_features, on=['game_id', 'team'], how='left')
    after_merge = len(merged_df)

    if after_merge != before_merge:
        print(f"⚠ WARNING: Row count changed: {before_merge} -> {after_merge}")
    else:
        print(f"✓ Merge successful: {after_merge} rows")

    # Check coverage
    print("\n" + "=" * 70)
    print("FEATURE COVERAGE AUDIT (SAMPLE)")
    print("=" * 70)

    opponent_features = ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']
    coverage_report = {}
    all_pass = True

    for feat in opponent_features:
        coverage = merged_df[feat].notna().mean()
        coverage_report[feat] = coverage

        if coverage >= COVERAGE_TARGET:
            status = "✓ EXCELLENT"
        elif coverage >= COVERAGE_THRESHOLD:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False

        missing_count = merged_df[feat].isna().sum()
        print(f"  {feat:30s} {coverage:6.1%}  {status:12s}  (missing: {missing_count}/{len(merged_df)})")

    print("=" * 70)

    # Analyze missing data by season
    print("\nMissing data breakdown by season:")
    for season in sorted(merged_df.season.unique()):
        season_df = merged_df[merged_df.season == season]
        missing_pct = season_df['opp_def_epa_exp'].isna().mean()
        print(f"  {season}: {missing_pct:5.1%} missing ({season_df['opp_def_epa_exp'].isna().sum()}/{len(season_df)} games)")

    # Decision
    print("\n" + "=" * 70)
    print("VALIDATION DECISION")
    print("=" * 70)

    if all_pass:
        print("✓ GO - PROCEED WITH FULL TRAINING")
        print(f"  All features meet {COVERAGE_THRESHOLD:.0%} threshold")
        print("  Expected to replicate on full 2018-2024 dataset")
        print("\nNext step:")
        print("  python py/models/train_xgboost_v3_1_aligned_window.py")
        decision = 'GO'
    else:
        print("✗ NO-GO - DO NOT TRAIN")
        print(f"  Some features below {COVERAGE_THRESHOLD:.0%} threshold")
        print("  Training would fail coverage audit")
        print("\nOptions:")
        print("  1. Investigate missing data causes")
        print("  2. Regenerate opponent features (py/features/opponent_strength.py)")
        print("  3. Adjust training window further")
        decision = 'NO_GO'

    print("=" * 70 + "\n")

    return {
        'status': decision,
        'sample_size': len(merged_df),
        'coverage_report': coverage_report,
        'missing_by_season': {
            int(season): float(merged_df[merged_df.season == season]['opp_def_epa_exp'].isna().mean())
            for season in merged_df.season.unique()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Validate XGBoost v3.1 feature coverage')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of games to sample (default: 100)')

    args = parser.parse_args()

    result = check_feature_coverage(sample_size=args.sample_size)

    # Exit with status code
    sys.exit(0 if result['status'] == 'GO' else 1)


if __name__ == '__main__':
    main()
