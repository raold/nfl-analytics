#!/usr/bin/env python3
"""
River Optimization - Multi-Approach Validation Script

Tests 6 different approaches to make River competitive for NFL betting.

BASELINE (November 12, 2025):
- MAE: 173.59 yards (target: <60, static: 55.16)
- Sample count: 373
- Drift events: 0 (target: 5-10)
- Online is 110% worse than static baseline

APPROACHES:
1. Hyperparameter tuning (NFL-optimized ARF config)
2. Ensemble of online learners (ARF+AMT+PA+KNN)
3. Warm-start with static model (XGBoost priors)
4. Feature engineering (15-20 features vs current 5-8)
5. Drift detector ensemble (weighted voting vs 2/3 consensus)
6. Custom incremental XGBoost (sliding window retraining)

SUCCESS CRITERIA:
- ✅ MAE < 65 yards (within 15% of static baseline)
- ⚠️ MAE 65-80 yards (partial success, document)
- ⛔ MAE > 80 yards (failure, abandon River)

Usage:
    python scripts/validate_river_approaches.py --approach 1 --season 2024
    python scripts/validate_river_approaches.py --approach all --season 2024

Author: Claude Code (River Optimization - November 12, 2025)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from py.online.online_learning_adapter import (
    OnlinePredictionPipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading (Reused from fixed validation script)
# =============================================================================

def load_player_stats_by_week(
    season: int,
    db_params: dict[str, str],
    min_passing_yards: int = 100,
) -> pd.DataFrame:
    """Load player passing stats by week for validation"""
    logger.info(f"Loading player stats for {season} season...")

    conn = psycopg2.connect(**db_params)

    query = """
        WITH weekly_stats AS (
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.kickoff,
                p.passer_player_id AS player_id,
                p.passer_player_name AS player_name,
                SUM(p.passing_yards) AS passing_yards,
                COUNT(*) AS num_plays,
                SUM(CASE WHEN p.pass_attempt THEN 1 ELSE 0 END) AS attempts,
                SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) AS completions,
                SUM(CASE WHEN p.interception = 1 THEN 1 ELSE 0 END) AS interceptions,
                SUM(CASE WHEN p.touchdown = 1 THEN 1 ELSE 0 END) AS touchdowns
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season = %s
                AND p.passer_player_id IS NOT NULL
                AND p.season_type = 'REG'
            GROUP BY p.game_id, g.season, g.week, g.kickoff, p.passer_player_id, p.passer_player_name
            HAVING SUM(p.passing_yards) >= %s
        )
        SELECT
            ws.*,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            COALESCE(g.spread_close, 0.0) as spread_close,
            COALESCE(g.total_close, 47.0) as total_close
        FROM weekly_stats ws
        JOIN games g ON ws.game_id = g.game_id
        ORDER BY ws.kickoff, ws.player_id
    """

    df = pd.read_sql(query, conn, params=(season, min_passing_yards))
    conn.close()

    logger.info(f"Loaded {len(df)} player-game records")
    return df


def engineer_features(df: pd.DataFrame, enhanced: bool = False) -> pd.DataFrame:
    """
    Engineer features for online learning

    Args:
        df: Raw player stats DataFrame
        enhanced: If True, add 15-20 features (Approach 4), else use 5-8 features
    """
    logger.info(f"Engineering features (enhanced={enhanced})...")
    logger.info(f"Starting with {len(df)} records")

    df = df.copy()
    df = df.sort_values(['player_id', 'kickoff'])

    # Rolling averages (L3 window only to preserve data)
    for window in [3]:
        df[f'passing_yards_L{window}'] = df.groupby('player_id')['passing_yards'].shift(1).rolling(window).mean()
        df[f'attempts_L{window}'] = df.groupby('player_id')['attempts'].shift(1).rolling(window).mean()
        df[f'completions_L{window}'] = df.groupby('player_id')['completions'].shift(1).rolling(window).mean()

    # Lagged completion percentage (no target leakage)
    df['completion_pct_L3'] = (
        df.groupby('player_id')['completions'].shift(1).rolling(3).sum() /
        df.groupby('player_id')['attempts'].shift(1).rolling(3).sum()
    )

    # Lagged yards per attempt (no target leakage)
    df['yards_per_attempt_L3'] = (
        df.groupby('player_id')['passing_yards'].shift(1).rolling(3).sum() /
        df.groupby('player_id')['attempts'].shift(1).rolling(3).sum()
    )

    # Game context features
    df['home_favorite'] = (df['spread_close'] < 0).astype(int)
    df['high_total'] = (df['total_close'] > 47).astype(int)

    if enhanced:
        # APPROACH 4: Enhanced feature engineering (15-20 features)
        logger.info("Adding enhanced features...")

        # 1. Momentum features (trend)
        df['yards_trend_L3'] = (
            df.groupby('player_id')['passing_yards'].shift(1).rolling(3).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / 3 if len(x) == 3 else 0, raw=False
            )
        )

        # 2. Volatility features
        df['yards_std_L3'] = df.groupby('player_id')['passing_yards'].shift(1).rolling(3).std()

        # 3. Days rest (temporal feature) - simulated
        df['days_since_last_game'] = df.groupby('player_id')['kickoff'].diff().dt.days.fillna(7)

        # 4. Week of season (temporal)
        df['week_of_season'] = df['week']

        # 5. Interaction features
        df['rest_x_trend'] = df['days_since_last_game'] * df['yards_trend_L3']
        df['week_x_volatility'] = df['week_of_season'] * df['yards_std_L3']

        # 6. Spread interaction
        df['spread_x_total'] = df['spread_close'] * df['total_close']

        # 7. Touchdown rate (lagged)
        df['touchdown_rate_L3'] = (
            df.groupby('player_id')['touchdowns'].shift(1).rolling(3).sum() /
            df.groupby('player_id')['attempts'].shift(1).rolling(3).sum().replace(0, 1)
        )

        # 8. Interception rate (lagged)
        df['interception_rate_L3'] = (
            df.groupby('player_id')['interceptions'].shift(1).rolling(3).sum() /
            df.groupby('player_id')['attempts'].shift(1).rolling(3).sum().replace(0, 1)
        )

        # Fill NaN in enhanced features
        enhanced_cols = ['yards_trend_L3', 'yards_std_L3', 'rest_x_trend', 'week_x_volatility',
                        'spread_x_total', 'touchdown_rate_L3', 'interception_rate_L3']
        for col in enhanced_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        logger.info(f"Enhanced features added: {len(enhanced_cols)} new columns")

    # Only drop rows with NaN in critical features
    critical_features = ['passing_yards_L3', 'attempts_L3', 'completions_L3']
    df = df.dropna(subset=critical_features)

    # Fill remaining NaN with median
    df['completion_pct_L3'] = df['completion_pct_L3'].fillna(df['completion_pct_L3'].median())
    df['yards_per_attempt_L3'] = df['yards_per_attempt_L3'].fillna(df['yards_per_attempt_L3'].median())

    logger.info(f"After feature engineering: {len(df)} records")
    return df


# =============================================================================
# Approach 1: Hyperparameter Tuning
# =============================================================================

def approach_1_hyperparameter_tuning(df: pd.DataFrame, db_params: dict) -> dict[str, Any]:
    """
    Approach 1: NFL-optimized hyperparameter configuration

    CHANGES:
    - n_models: 10 → 3 (fewer models, more samples each)
    - grace_period: 50 → 10 (faster splits)
    - lambda_value: 6 → 2 (faster adaptation)
    - split_confidence: 0.01 → 0.10 (less conservative)
    - drift_detector: ADWIN(delta=0.002) → ADWIN(delta=0.01) (more sensitive)

    Expected: MAE < 80 yards (50% better than baseline 173)
    """
    logger.info("=" * 60)
    logger.info("APPROACH 1: HYPERPARAMETER TUNING (NFL-OPTIMIZED)")
    logger.info("=" * 60)

    # Initialize with NFL-optimized model
    pipeline = OnlinePredictionPipeline(
        online_model_type='adaptive_random_forest_nfl',  # NEW NFL-tuned config
        drift_sensitivity='high',  # More sensitive drift detection
    )

    # Run prequential evaluation
    results = pipeline.evaluate_prequential(
        df=df,
        target_col='passing_yards',
        date_col='kickoff',
        window_size=50,
    )

    results['approach'] = 'hyperparameter_tuning'
    results['config'] = {
        'n_models': 3,
        'grace_period': 10,
        'lambda_value': 2,
        'split_confidence': 0.10,
        'drift_sensitivity': 'high',
    }

    logger.info(f"Results: MAE={results['mean_mae']:.2f} yards, Drifts={results['num_drift_events']}")
    return results


# =============================================================================
# Approach 2: Ensemble of Online Learners
# =============================================================================

def approach_2_multi_ensemble(df: pd.DataFrame, db_params: dict) -> dict[str, Any]:
    """
    Approach 2: Ensemble of diverse online learners

    MODELS:
    - Adaptive Random Forest (tree-based, drift-aware)
    - Adaptive Model Trees (rule-based)
    - Passive-Aggressive Regressor (linear, fast)
    - KNN Regressor (lazy learning)

    Expected: MAE < 70 yards (diverse models reduce variance)
    """
    logger.info("=" * 60)
    logger.info("APPROACH 2: MULTI-MODEL ENSEMBLE")
    logger.info("=" * 60)

    from py.online.multi_ensemble import MultiOnlineEnsemble

    # Initialize multi-ensemble
    ensemble = MultiOnlineEnsemble()

    # Prequential evaluation
    df_sorted = df.sort_values('kickoff').reset_index(drop=True)

    # Select features
    exclude_cols = ['passing_yards', 'kickoff', 'player_id', 'player_name', 'game_id',
                   'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'num_plays']
    feature_cols = [c for c in df_sorted.columns if c not in exclude_cols and df_sorted[c].dtype in ['int64', 'float64']]
    X = df_sorted[feature_cols]
    y = df_sorted['passing_yards']

    mae_history = []

    for i in range(50, len(df_sorted)):  # Warm-up period
        # Test
        x_test = {k: float(v) for k, v in X.iloc[i].to_dict().items()}
        y_test = float(y.iloc[i])

        pred = ensemble.predict_one(x_test)
        error = abs(y_test - pred)
        mae_history.append(error)

        # Train
        ensemble.learn_one(x_test, y_test)

    results = {
        'approach': 'multi_ensemble',
        'mean_mae': np.mean(mae_history),
        'median_mae': np.median(mae_history),
        'std_mae': np.std(mae_history),
        'num_drift_events': 0,  # No drift detection in multi-ensemble
        'final_model_samples': len(mae_history),
        'model_weights': ensemble.weights,
    }

    logger.info(f"Results: MAE={results['mean_mae']:.2f} yards")
    return results


# =============================================================================
# Approach 3: Warm-Start with Static Model
# =============================================================================

def approach_3_warm_start(df: pd.DataFrame, db_params: dict) -> dict[str, Any]:
    """
    Approach 3: Pre-train on historical data, blend with online model

    STRATEGY:
    - Train XGBoost on 2018-2023 data
    - Use predictions as priors for 2024
    - Exponential decay: 90% static → 10% static over time

    Expected: MAE < 65 yards (cold-start problem solved)
    """
    logger.info("=" * 60)
    logger.info("APPROACH 3: WARM-START WITH STATIC MODEL")
    logger.info("=" * 60)

    # TODO: Load historical data (2018-2023) for static model training
    # For now, use mean as static predictor

    pipeline = OnlinePredictionPipeline(
        online_model_type='adaptive_random_forest_nfl',
        drift_sensitivity='medium',
    )

    df_sorted = df.sort_values('kickoff').reset_index(drop=True)

    # Split: first 50% for static model
    split_idx = len(df_sorted) // 2
    static_mean = df_sorted['passing_yards'].iloc[:split_idx].mean()

    # Prequential evaluation with blending
    exclude_cols = ['passing_yards', 'kickoff', 'player_id', 'player_name', 'game_id',
                   'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'num_plays']
    feature_cols = [c for c in df_sorted.columns if c not in exclude_cols and df_sorted[c].dtype in ['int64', 'float64']]
    X = df_sorted[feature_cols]
    y = df_sorted['passing_yards']

    mae_history = []

    for i in range(50, len(df_sorted)):
        # Exponential decay weight
        decay = 0.9 ** (i / 50)

        # Static prediction
        static_pred = static_mean

        # Online prediction
        x_test = {k: float(v) for k, v in X.iloc[i].to_dict().items()}
        y_test = float(y.iloc[i])
        online_pred = pipeline.online_model.predict(x_test)

        # Blend
        blended_pred = decay * static_pred + (1 - decay) * online_pred
        error = abs(y_test - blended_pred)
        mae_history.append(error)

        # Train
        pipeline.online_model.partial_fit(x_test, y_test)

    results = {
        'approach': 'warm_start',
        'mean_mae': np.mean(mae_history),
        'median_mae': np.median(mae_history),
        'std_mae': np.std(mae_history),
        'num_drift_events': 0,
        'final_model_samples': len(mae_history),
    }

    logger.info(f"Results: MAE={results['mean_mae']:.2f} yards")
    return results


# =============================================================================
# Approach 4: Feature Engineering
# =============================================================================

def approach_4_feature_engineering(df: pd.DataFrame, db_params: dict) -> dict[str, Any]:
    """
    Approach 4: Enhanced feature engineering (15-20 features vs 5-8)

    NEW FEATURES:
    - Momentum (trend_L3, volatility)
    - Temporal (days_rest, week_of_season)
    - Interactions (rest_x_trend, week_x_volatility, spread_x_total)
    - Rates (touchdown_rate_L3, interception_rate_L3)

    Expected: MAE < 75 yards (more information for trees)
    """
    logger.info("=" * 60)
    logger.info("APPROACH 4: FEATURE ENGINEERING (15-20 FEATURES)")
    logger.info("=" * 60)

    # Re-engineer features with enhanced=True
    df_enhanced = load_player_stats_by_week(2024, db_params)
    df_enhanced = engineer_features(df_enhanced, enhanced=True)

    logger.info(f"Features: {len([c for c in df_enhanced.columns if df_enhanced[c].dtype in ['int64', 'float64']])} total")

    # Initialize pipeline with NFL-optimized model
    pipeline = OnlinePredictionPipeline(
        online_model_type='adaptive_random_forest_nfl',
        drift_sensitivity='medium',
    )

    # Run prequential evaluation
    results = pipeline.evaluate_prequential(
        df=df_enhanced,
        target_col='passing_yards',
        date_col='kickoff',
        window_size=50,
    )

    results['approach'] = 'feature_engineering'
    results['num_features'] = len([c for c in df_enhanced.columns if df_enhanced[c].dtype in ['int64', 'float64']])

    logger.info(f"Results: MAE={results['mean_mae']:.2f} yards, Features={results['num_features']}")
    return results


# =============================================================================
# Approach 6: Custom Incremental XGBoost
# =============================================================================

def approach_6_incremental_xgboost(df: pd.DataFrame, db_params: dict) -> dict[str, Any]:
    """
    Approach 6: Incremental XGBoost with sliding window

    STRATEGY:
    - Sliding window of 100 recent samples
    - Retrain every 20 samples
    - XGBoost expressiveness >> Hoeffding trees

    Expected: MAE < 60 yards (match static baseline)
    """
    logger.info("=" * 60)
    logger.info("APPROACH 6: INCREMENTAL XGBOOST")
    logger.info("=" * 60)

    from py.online.incremental_xgboost import IncrementalXGBoost

    # Initialize incremental XGBoost
    model = IncrementalXGBoost(
        window_size=100,
        retrain_every=20,
        n_estimators=50,
        max_depth=5,
    )

    # Prequential evaluation
    df_sorted = df.sort_values('kickoff').reset_index(drop=True)

    # Select features
    exclude_cols = ['passing_yards', 'kickoff', 'player_id', 'player_name', 'game_id',
                   'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'num_plays']
    feature_cols = [c for c in df_sorted.columns if c not in exclude_cols and df_sorted[c].dtype in ['int64', 'float64']]
    X = df_sorted[feature_cols]
    y = df_sorted['passing_yards']

    mae_history = []

    for i in range(50, len(df_sorted)):  # Warm-up period
        # Test
        x_test = {k: float(v) for k, v in X.iloc[i].to_dict().items()}
        y_test = float(y.iloc[i])

        pred = model.predict_one(x_test)
        error = abs(y_test - pred)
        mae_history.append(error)

        # Train
        model.learn_one(x_test, y_test)

    results = {
        'approach': 'incremental_xgboost',
        'mean_mae': np.mean(mae_history),
        'median_mae': np.median(mae_history),
        'std_mae': np.std(mae_history),
        'num_drift_events': 0,  # No drift detection
        'final_model_samples': len(mae_history),
        'retrains': model.get_metrics()['retrains'],
    }

    logger.info(f"Results: MAE={results['mean_mae']:.2f} yards, Retrains={results['retrains']}")
    return results


# =============================================================================
# Main Validation Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate River optimization approaches')
    parser.add_argument('--approach', type=str, default='1',
                       choices=['1', '2', '3', '4', '5', '6', 'all'],
                       help='Approach number to test (or "all")')
    parser.add_argument('--season', type=int, default=2024, help='NFL season')
    parser.add_argument('--output-dir', type=str, default='results/', help='Output directory')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RIVER OPTIMIZATION - MULTI-APPROACH VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Testing approach: {args.approach}")
    logger.info(f"Season: {args.season}")

    # Database connection
    db_params = {
        'host': 'localhost',
        'port': 5544,
        'dbname': 'devdb01',
        'user': 'dro',
        'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
    }

    # Load data
    df = load_player_stats_by_week(args.season, db_params)
    df = engineer_features(df)

    logger.info(f"Data loaded: {len(df)} records")

    # Run approaches
    results_all = {}

    if args.approach in ['1', 'all']:
        results_all['approach_1'] = approach_1_hyperparameter_tuning(df, db_params)

    if args.approach in ['2', 'all']:
        try:
            results_all['approach_2'] = approach_2_multi_ensemble(df, db_params)
        except ImportError:
            logger.warning("Approach 2 skipped - multi_ensemble.py not created yet")

    if args.approach in ['3', 'all']:
        results_all['approach_3'] = approach_3_warm_start(df, db_params)

    if args.approach in ['4', 'all']:
        results_all['approach_4'] = approach_4_feature_engineering(df, db_params)

    if args.approach in ['5', 'all']:
        logger.warning("Approach 5 (drift ensemble) not implemented yet - skipping")

    if args.approach in ['6', 'all']:
        results_all['approach_6'] = approach_6_incremental_xgboost(df, db_params)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"online_approach_{args.approach}_validation.json"
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("RIVER OPTIMIZATION SUMMARY")
    print("=" * 80)
    print("Baseline: MAE=173.59 yards, Static=55.16 yards")
    print("\nResults:")
    for approach_name, result in results_all.items():
        mae = result.get('mean_mae', 0)
        drifts = result.get('num_drift_events', 0)
        status = "✅" if mae < 65 else ("⚠️" if mae < 80 else "⛔")
        print(f"  {status} {approach_name}: MAE={mae:.2f} yards, Drifts={drifts}")
    print("=" * 80)


if __name__ == "__main__":
    main()
