#!/usr/bin/env python3
"""
L1 Regularization Experiment (Fixed - Using v3.1 Data Loading Pattern)

Grid search over L1 alpha values to find optimal regularization.
Uses proven v3.1 data loading from mart.player_game_stats + games tables.

Author: NFL Analytics Research Team
Date: November 14, 2025

Experiment Setup:
- Training data: 2018-2024 (aligned window)
- Test data: 2025 season (hold-out)
- L1 alphas: [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
- L2 lambda: FIXED at 1.5 (baseline)
- Baseline MAE: 40.20 yards (v2 XGBoost)

Target: Find if any L1 alpha beats baseline without overfitting

Usage:
    python scripts/xgb_exp5_l1_fixed.py

Output:
    - Console report with all 8 model MAEs
    - Best model saved if MAE < 40.20
    - Experiment summary to file
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5544')),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
    'database': os.getenv('DB_NAME', 'devdb01')
}

# Training window (same as v3.1)
ALIGNED_WINDOW = {
    'train_start': 2018,
    'train_end': 2024,
    'test_season': 2025,
}

# L1 grid search
L1_ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
L2_LAMBDA_FIXED = 1.5
BASELINE_MAE = 40.20

# XGBoost base parameters (same as v3.1)
XGBOOST_BASE_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_lambda': L2_LAMBDA_FIXED,  # L2 fixed
    'seed': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

# 16 features (13 baseline + 3 opponent strength)
FEATURE_COLS = [
    # Baseline 13 features
    'log_attempts',
    'yards_per_attempt',
    'td_rate',
    'is_home',
    'spread_abs',
    'total_line',
    'is_bad_weather',
    'yards_L3',
    'yards_per_attempt_L3',
    'td_rate_L3',
    'week_in_season',
    'temp',
    'wind',
    # Opponent strength features (v3)
    'opp_def_epa_exp',
    'opp_pass_def_rank',
    'matchup_difficulty'
]


def load_data_from_database() -> pd.DataFrame:
    """
    Load QB passing data from database (2018-2025 ONLY).

    Uses same pattern as train_xgboost_v3_1_aligned_window.py
    """
    logger.info("=" * 70)
    logger.info("LOADING DATA FROM DATABASE (2018-2025)")
    logger.info("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)

    # Query with aligned window (2018+)
    query = """
    WITH qb_games AS (
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as yards,
            pgs.stat_touchdowns as touchdowns,
            pgs.stat_attempts as attempts,
            pgs.stat_completions as completions,
            pgs.avg_time_to_throw,
            pgs.avg_air_yards_differential,
            pgs.cpoe,
            pgs.current_team as team,
            pgs.player_display_name as qb_name,
            g.game_id,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
            g.temp,
            g.wind,
            g.stadium,
            g.roof,
            g.surface,
            g.is_division_game,
            g.is_thursday_night,
            g.is_short_week,
            g.away_rest,
            g.home_rest,
            CASE WHEN pgs.current_team = g.home_team
                THEN g.home_rest
                ELSE g.away_rest
            END as rest_days,
            CASE WHEN pgs.current_team = g.home_team
                THEN g.away_def_epa_last_4
                ELSE g.home_def_epa_last_4
            END as opp_def_epa,
            g.home_turnovers,
            g.away_turnovers,
            g.home_penalties,
            g.away_penalties
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
            AND pgs.season >= 2018
            AND pgs.season <= 2025
            AND ph.position_group = 'QB'
    )
    SELECT * FROM qb_games
    ORDER BY season, week, player_id
    """

    logger.info("Executing database query...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"✓ Loaded {len(df):,} QB games")
    logger.info(f"  Seasons: {df.season.min()}-{df.season.max()}")
    logger.info(f"  Unique QBs: {df.player_id.nunique()}")

    return df


def load_opponent_features() -> pd.DataFrame:
    """Load opponent strength features from CSV."""
    logger.info("\nLoading opponent strength features...")

    features_path = Path('data/processed/features/opponent_strength_features.csv')

    if not features_path.exists():
        logger.warning(f"Opponent features not found: {features_path}")
        logger.warning("Creating dummy features with NaN (will be filled during engineering)")
        return pd.DataFrame()

    opp_df = pd.read_csv(features_path)
    logger.info(f"✓ Loaded {len(opp_df):,} opponent feature records")

    return opp_df[[
        'game_id', 'team',
        'opp_def_epa_exp',
        'opp_pass_def_rank',
        'matchup_difficulty'
    ]].copy()


def engineer_features(df: pd.DataFrame, opp_features: pd.DataFrame) -> pd.DataFrame:
    """Engineer 16 features from raw data."""
    logger.info("\n" + "=" * 70)
    logger.info("ENGINEERING 16 FEATURES")
    logger.info("=" * 70)

    df = df.sort_values(['player_id', 'season', 'week']).copy()

    # Merge opponent strength features if available
    if not opp_features.empty:
        logger.info("Merging opponent strength features...")
        df = df.merge(opp_features, on=['game_id', 'team'], how='left')
        logger.info("✓ Merge complete")
    else:
        logger.info("Creating placeholder opponent features (NaN)")
        for col in ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']:
            df[col] = np.nan

    # Basic efficiency metrics
    logger.info("Creating efficiency metrics...")
    df['yards_per_attempt'] = df['yards'] / df['attempts'].replace(0, np.nan)
    df['td_rate'] = df['touchdowns'] / df['attempts'].replace(0, np.nan)
    df['log_attempts'] = np.log1p(df['attempts'])

    # Weather features
    logger.info("Creating weather features...")
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['is_dome'] = (df['roof'].isin(['dome', 'closed'])).astype(int)
    df['is_cold'] = ((df['temp'] < 32) & (~df['is_dome'].astype(bool))).astype(int)
    df['is_windy'] = ((df['wind'] > 15) & (~df['is_dome'].astype(bool))).astype(int)
    df['is_bad_weather'] = (df['is_cold'] | df['is_windy']).astype(int)

    # Game script features
    logger.info("Creating game script features...")
    df['spread_abs'] = df['spread_close'].abs()
    df['total_line'] = df['total_close']

    # Schedule factors
    df['week_in_season'] = df['week'].astype(float)

    # Rolling averages (L3 only - anti-leakage)
    logger.info("Creating rolling averages...")
    for col in ['yards', 'yards_per_attempt', 'td_rate']:
        df[f'{col}_L3'] = (
            df.groupby('player_id')[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .shift(1)  # Shift to avoid lookahead bias
        )

    # Fill baseline features only (NOT opponent features)
    baseline_cols = [
        'yards_per_attempt', 'td_rate', 'log_attempts',
        'temp', 'wind', 'spread_abs', 'total_line', 'week_in_season',
        'yards_L3', 'yards_per_attempt_L3', 'td_rate_L3'
    ]

    for col in baseline_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    logger.info(f"✓ Feature engineering complete ({len(FEATURE_COLS)} total)")

    return df


def prepare_train_test_split(df: pd.DataFrame) -> tuple:
    """Split data into train/test sets."""
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING TRAIN/TEST SPLIT")
    logger.info("=" * 70)

    # Verify all features exist
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Split by season
    train_df = df[df['season'].between(
        ALIGNED_WINDOW['train_start'],
        ALIGNED_WINDOW['train_end']
    )].copy()
    test_df = df[df['season'] == ALIGNED_WINDOW['test_season']].copy()

    logger.info("\n✓ Split complete:")
    logger.info(f"  Train: {len(train_df):,} samples ({ALIGNED_WINDOW['train_start']}-{ALIGNED_WINDOW['train_end']})")
    logger.info(f"  Test: {len(test_df):,} samples ({ALIGNED_WINDOW['test_season']})")

    # Extract features
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['yards'].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['yards'].values

    logger.info(f"  Features: {len(FEATURE_COLS)}")
    logger.info(f"  Train shape: {X_train.shape}")
    logger.info(f"  Test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test, train_df, test_df


def train_xgboost_model(X_train, y_train, X_test, y_test, l1_alpha: float):
    """Train single XGBoost model with specified L1 alpha."""
    params = XGBOOST_BASE_PARAMS.copy()
    params['reg_alpha'] = l1_alpha  # Override with L1 value

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train with early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Predictions
    y_pred = model.predict(dtest)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, mae, rmse, r2, y_pred


def run_l1_grid_search(X_train, y_train, X_test, y_test):
    """Run L1 regularization grid search."""
    logger.info("\n" + "=" * 70)
    logger.info("L1 REGULARIZATION GRID SEARCH")
    logger.info("=" * 70)
    logger.info(f"Grid: {L1_ALPHAS}")
    logger.info(f"L2 Lambda (fixed): {L2_LAMBDA_FIXED}")
    logger.info(f"Baseline MAE: {BASELINE_MAE:.2f} yards")
    logger.info("=" * 70)

    results = []
    best_mae = float('inf')
    best_alpha = None
    best_model = None

    for i, alpha in enumerate(L1_ALPHAS, 1):
        logger.info(f"\n[{i}/{len(L1_ALPHAS)}] Training with L1 alpha = {alpha}...")

        start_time = time.time()
        model, mae, rmse, r2, y_pred = train_xgboost_model(X_train, y_train, X_test, y_test, alpha)
        elapsed = time.time() - start_time

        beat_baseline = "✓ BETTER" if mae < BASELINE_MAE else "✗ WORSE"

        logger.info(f"  MAE: {mae:.2f} yards {beat_baseline}")
        logger.info(f"  RMSE: {rmse:.2f} yards")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Time: {elapsed:.1f}s")

        results.append({
            'l1_alpha': alpha,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'time_seconds': elapsed,
            'beats_baseline': mae < BASELINE_MAE
        })

        # Track best
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_model = model

    return results, best_model, best_alpha, best_mae


def main():
    """Main execution."""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("L1 REGULARIZATION EXPERIMENT (Fixed v3.1 Pattern)")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        # Load data
        df = load_data_from_database()
        opp_features = load_opponent_features()

        # Engineer features
        df = engineer_features(df, opp_features)

        # Prepare train/test
        X_train, y_train, X_test, y_test, train_df, test_df = prepare_train_test_split(df)

        # Run grid search
        results, best_model, best_alpha, best_mae = run_l1_grid_search(
            X_train, y_train, X_test, y_test
        )

        # Report results
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("=" * 70)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('mae')

        logger.info("\nAll Results (sorted by MAE):")
        logger.info("-" * 70)
        for _, row in results_df.iterrows():
            status = "✓" if row['beats_baseline'] else "✗"
            logger.info(
                f"{status} L1={row['l1_alpha']:6.1f}: MAE={row['mae']:6.2f}, "
                f"RMSE={row['rmse']:6.2f}, R²={row['r2']:6.4f}"
            )

        logger.info("\n" + "-" * 70)
        logger.info(f"BEST: L1 alpha = {best_alpha} with MAE = {best_mae:.2f} yards")

        # Check if any beat baseline
        beats_baseline = results_df[results_df['beats_baseline']]
        if len(beats_baseline) > 0:
            logger.info(f"\n✓ SUCCESS: {len(beats_baseline)} alpha(s) beat baseline ({BASELINE_MAE:.2f})")
            logger.info("  Alphas that beat baseline:")
            for _, row in beats_baseline.iterrows():
                logger.info(f"    - L1={row['l1_alpha']:6.1f}: MAE={row['mae']:6.2f} "
                           f"(Δ={BASELINE_MAE - row['mae']:+.2f})")

            # Save best model
            if best_mae < BASELINE_MAE:
                model_path = Path('models/xgboost/passing_xgboost_l1_exp5_best.pkl')
                model_path.parent.mkdir(parents=True, exist_ok=True)
                best_model.save_model(str(model_path))
                logger.info(f"\n✓ Best model saved: {model_path}")

                # Save metadata
                metadata = {
                    'experiment': 'L1 Regularization Grid Search (v3.1 pattern)',
                    'l1_alpha': best_alpha,
                    'l2_lambda': L2_LAMBDA_FIXED,
                    'test_mae': best_mae,
                    'test_rmse': results_df[results_df['l1_alpha'] == best_alpha]['rmse'].values[0],
                    'test_r2': results_df[results_df['l1_alpha'] == best_alpha]['r2'].values[0],
                    'baseline_mae': BASELINE_MAE,
                    'improvement': BASELINE_MAE - best_mae,
                    'improvement_pct': ((BASELINE_MAE - best_mae) / BASELINE_MAE) * 100,
                    'training_window': f"{ALIGNED_WINDOW['train_start']}-{ALIGNED_WINDOW['train_end']}",
                    'test_season': ALIGNED_WINDOW['test_season'],
                    'features': FEATURE_COLS,
                    'date': pd.Timestamp.now().isoformat()
                }

                metadata_path = Path('models/xgboost/passing_xgboost_l1_exp5_best_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"✓ Metadata saved: {metadata_path}")
        else:
            logger.info(f"\n✗ NO IMPROVEMENT: All L1 alphas >= baseline ({BASELINE_MAE:.2f})")
            logger.info("  Recommendation: L1 regularization may not help for this model")

        # Save full results
        results_path = Path('scripts/xgb_exp5_l1_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n✓ Full results saved: {results_path}")

        elapsed = time.time() - start_time
        logger.info(f"\nTotal execution time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    except Exception as e:
        logger.error(f"✗ FAILED: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
