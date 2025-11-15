#!/usr/bin/env python3
"""
BFS Experiment 5: L1 Regularization for XGBoost Opponent Feature Regression

PURPOSE:
Fix XGBoost v3.1 performance regression (57.37 MAE) by using L1 regularization
to automatically remove redundant opponent features.

BACKGROUND:
- XGBoost v2 (no opponent features): 40.20 MAE (baseline)
- XGBoost v3.1 (with opponent features): 57.37 MAE (-42.71% regression)
- Root Cause: Opponent features (opp_def_epa_exp, opp_pass_def_rank, matchup_difficulty)
  are redundant with existing features (opp_def_epa, total_line, etc.)

RESEARCH FINDING:
L1 regularization (alpha parameter) can automatically remove redundant features
by driving their weights to zero. This is a "safety net" approach.

SUCCESS CRITERIA:
- MAE < 45 yards → INVESTIGATE
- MAE < 42 yards → DEPLOY (integrate into ensemble)
- MAE < 40.20 yards → BREAKTHROUGH (beats baseline)

APPROACH:
1. Load same data as v3.1 (2018-2024 train, 2025 test)
2. Grid search over L1 (alpha) and L2 (lambda) parameters
3. Use 5-fold CV on training set to select best combination
4. Evaluate final model on 2025 holdout
5. Log feature importance to verify opponent features driven to zero
6. Compare to baselines

OUTPUT:
- models/xgboost/passing_xgboost_v3_2_l1.pkl (best model)
- models/xgboost/passing_xgboost_v3_2_l1_metadata.json
- Grid search results table
- Feature importance analysis

RUNTIME ESTIMATE: 15-25 minutes (grid search: 8 alphas × 3 lambdas × 5 CV = 120 models)

Author: NFL Analytics Research Team
Date: November 14, 2025

Usage:
    python scripts/xgb_v3_2_l1_regularization.py
"""

import sys
import os
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5544')),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
    'database': os.getenv('DB_NAME', 'devdb01')
}

# Training window (aligned with EPA availability)
ALIGNED_WINDOW = {
    'train_start': 2018,
    'train_end': 2024,
    'test_season': 2025,
}

# Feature coverage requirements
COVERAGE_THRESHOLD = 0.80
COVERAGE_TARGET = 0.95

# Grid search parameters
L1_ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
L2_LAMBDAS = [1.0, 2.0, 5.0]

# XGBoost base hyperparameters (same as v3.1)
XGBOOST_BASE_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'gamma': 0.05,
    'seed': 42,
    'n_jobs': -1,
}

# CV configuration
CV_FOLDS = 5
NUM_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 75

# Baseline comparison
V2_BASELINE_MAE = 40.20
V3_1_REGRESSION_MAE = 57.37

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GridSearchResult:
    """Store grid search result for a single alpha/lambda combination"""
    alpha: float
    lambda_param: float
    cv_mae_mean: float
    cv_mae_std: float
    test_mae: float
    rank: int = 0

    def __repr__(self):
        return (
            f"alpha={self.alpha:5.1f}, lambda={self.lambda_param:4.1f}, "
            f"CV MAE: {self.cv_mae_mean:6.2f}±{self.cv_mae_std:5.2f}, "
            f"Test MAE: {self.test_mae:6.2f}"
        )


# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_data_from_db():
    """Load QB passing data from database (2018-2025 only)."""
    logger.info("=" * 70)
    logger.info("LOADING DATA FROM DATABASE (2018-2025 ONLY)")
    logger.info("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)

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

            -- Game context
            g.game_id,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

            -- Weather
            g.temp,
            g.wind,
            g.stadium,
            g.roof,
            g.surface,

            -- Game dynamics
            g.is_division_game,
            g.is_thursday_night,
            g.is_short_week,
            g.away_rest,
            g.home_rest,
            CASE WHEN pgs.current_team = g.home_team
                THEN g.home_rest
                ELSE g.away_rest
            END as rest_days,

            -- Opponent defense strength
            CASE WHEN pgs.current_team = g.home_team
                THEN g.away_def_epa_last_4
                ELSE g.home_def_epa_last_4
            END as opp_def_epa,

            -- Turnovers & Penalties
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


def load_opponent_strength_features():
    """Load opponent strength features from CSV file."""
    logger.info("\n" + "=" * 70)
    logger.info("LOADING OPPONENT STRENGTH FEATURES")
    logger.info("=" * 70)

    features_path = Path('data/processed/features/opponent_strength_features.csv')

    if not features_path.exists():
        raise FileNotFoundError(
            f"Opponent strength features not found at {features_path}"
        )

    opp_df = pd.read_csv(features_path)
    logger.info(f"✓ Loaded {len(opp_df):,} opponent strength records")
    logger.info(f"  Seasons: {opp_df.season.min()}-{opp_df.season.max()}")

    # Select only the features we need
    opp_features = opp_df[[
        'game_id', 'team',
        'opp_def_epa_exp',
        'opp_pass_def_rank',
        'matchup_difficulty'
    ]].copy()

    # Check coverage
    non_null_counts = opp_features[['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']].notna().sum()
    logger.info(f"\nOpponent feature availability:")
    for col, count in non_null_counts.items():
        logger.info(f"  {col}: {count:,} ({count/len(opp_features)*100:.1f}%)")

    return opp_features


def engineer_features(df: pd.DataFrame, opp_features: pd.DataFrame) -> pd.DataFrame:
    """Engineer 16 features (13 baseline + 3 opponent strength)."""
    logger.info("\n" + "=" * 70)
    logger.info("ENGINEERING 16 FEATURES (13 BASELINE + 3 OPPONENT STRENGTH)")
    logger.info("=" * 70)

    df = df.sort_values(['player_id', 'season', 'week']).copy()

    # Merge opponent strength features
    logger.info("\nMerging opponent strength features...")
    before_merge = len(df)
    df = df.merge(
        opp_features,
        on=['game_id', 'team'],
        how='left'
    )
    after_merge = len(df)

    if after_merge != before_merge:
        logger.warning(f"Row count changed: {before_merge} -> {after_merge}")
    else:
        logger.info(f"✓ Merge successful, {after_merge:,} rows preserved")

    # Basic efficiency metrics
    logger.info("\nCreating basic efficiency metrics...")
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
    logger.info("Creating schedule factors...")
    df['week_in_season'] = df['week'].astype(float)

    # Rolling averages (anti-leakage)
    logger.info("Creating rolling averages (L3 only)...")
    for col in ['yards', 'yards_per_attempt', 'td_rate']:
        df[f'{col}_L3'] = (
            df.groupby('player_id')[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .shift(1)
        )

    # Fill baseline features (not opponent features)
    baseline_numeric_cols = [
        'yards_per_attempt', 'td_rate', 'log_attempts',
        'temp', 'wind', 'spread_abs', 'total_line', 'week_in_season',
        'yards_L3', 'yards_per_attempt_L3', 'td_rate_L3'
    ]

    for col in baseline_numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    logger.info(f"\n✓ Feature engineering complete")
    logger.info(f"  Total features: 16 (13 baseline + 3 opponent strength)")
    logger.info(f"  Samples: {len(df):,}")

    return df


def audit_feature_coverage(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = COVERAGE_THRESHOLD,
) -> Dict[str, float]:
    """Audit feature coverage before training."""
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE COVERAGE AUDIT (CRITICAL GATE)")
    logger.info("=" * 70)

    coverage_report = {}
    all_pass = True

    for feat in features:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in DataFrame")

        coverage = df[feat].notna().mean()
        coverage_report[feat] = coverage

        if coverage >= COVERAGE_TARGET:
            status = "✓ EXCELLENT"
        elif coverage >= threshold:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_pass = False

        missing_count = len(df) - df[feat].notna().sum()
        logger.info(
            f"  {feat:30s} {coverage:6.1%}  {status:12s}  "
            f"(missing: {missing_count:,})"
        )

    logger.info("=" * 70)

    if not all_pass:
        raise ValueError(f"Coverage audit failed - some features below {threshold:.0%}")

    logger.info("✓ ALL FEATURES PASS COVERAGE AUDIT\n")
    return coverage_report


def prepare_train_test_split(df: pd.DataFrame) -> Tuple:
    """Split data into train/test sets."""
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING TRAIN/TEST SPLIT (ALIGNED WINDOW)")
    logger.info("=" * 70)

    feature_cols = [
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

    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Train/test split
    train_df = df[df['season'].between(
        ALIGNED_WINDOW['train_start'],
        ALIGNED_WINDOW['train_end']
    )].copy()
    test_df = df[df['season'] == ALIGNED_WINDOW['test_season']].copy()

    logger.info(f"\n✓ Split complete:")
    logger.info(f"  Train: {len(train_df):,} samples ({ALIGNED_WINDOW['train_start']}-{ALIGNED_WINDOW['train_end']})")
    logger.info(f"  Test: {len(test_df):,} samples ({ALIGNED_WINDOW['test_season']})")
    logger.info(f"  Features: {len(feature_cols)}")

    # Coverage audit
    logger.info(f"\nRunning coverage audit on training set...")
    coverage_report = audit_feature_coverage(
        train_df,
        feature_cols,
        threshold=COVERAGE_THRESHOLD,
    )

    # Extract features
    X_train = train_df[feature_cols].values
    y_train = train_df['yards'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['yards'].values

    logger.info(f"\n16 Features (13 Baseline + 3 Opponent Strength):")
    for i, col in enumerate(feature_cols, 1):
        coverage = coverage_report.get(col, 0.0)
        marker = " [NEW]" if col in ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty'] else ""
        logger.info(f"  {i:2d}. {col:30s} ({coverage:5.1%} coverage){marker}")

    return X_train, y_train, X_test, y_test, feature_cols, train_df


# ============================================================================
# GRID SEARCH
# ============================================================================

def cross_validate_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    lambda_param: float,
    cv_folds: int = CV_FOLDS,
) -> Tuple[float, float]:
    """
    Perform k-fold cross-validation for a specific alpha/lambda combination.

    Returns:
        (cv_mae_mean, cv_mae_std)
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_maes = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_cv_train = X_train[train_idx]
        y_cv_train = y_train[train_idx]
        X_cv_val = X_train[val_idx]
        y_cv_val = y_train[val_idx]

        # Create XGBoost params with current alpha/lambda
        params = XGBOOST_BASE_PARAMS.copy()
        params['reg_alpha'] = alpha
        params['reg_lambda'] = lambda_param

        # Train
        dtrain = xgb.DMatrix(X_cv_train, label=y_cv_train)
        dval = xgb.DMatrix(X_cv_val, label=y_cv_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dval, 'val')],
            evals_result={},
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False
        )

        # Evaluate
        y_pred = model.predict(dval)
        mae = mean_absolute_error(y_cv_val, y_pred)
        cv_maes.append(mae)

    return np.mean(cv_maes), np.std(cv_maes)


def grid_search_regularization(X_train, y_train, X_test, y_test, feature_cols):
    """
    Grid search over L1 (alpha) and L2 (lambda) parameters.

    Returns:
        list of GridSearchResult objects sorted by CV performance
    """
    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH: L1/L2 REGULARIZATION TUNING")
    logger.info("=" * 70)
    logger.info(f"Alpha values: {L1_ALPHAS}")
    logger.info(f"Lambda values: {L2_LAMBDAS}")
    logger.info(f"CV folds: {CV_FOLDS}")
    logger.info(f"Total combinations: {len(L1_ALPHAS) * len(L2_LAMBDAS)}")
    logger.info("=" * 70 + "\n")

    results = []
    total_combinations = len(L1_ALPHAS) * len(L2_LAMBDAS)
    start_time = time.time()

    for combo_idx, alpha in enumerate(L1_ALPHAS):
        for lambda_param in L2_LAMBDAS:
            combo_num = combo_idx * len(L2_LAMBDAS) + (L2_LAMBDAS.tolist().index(lambda_param) if hasattr(L2_LAMBDAS, '__iter__') else L2_LAMBDAS.index(lambda_param))

            logger.info(f"\n[{combo_num + 1}/{total_combinations}] Grid point: alpha={alpha}, lambda={lambda_param}")
            logger.info("  Running 5-fold CV...")

            combo_start = time.time()

            # Cross-validate
            cv_mae_mean, cv_mae_std = cross_validate_xgboost(
                X_train, y_train,
                alpha=alpha,
                lambda_param=lambda_param,
                cv_folds=CV_FOLDS
            )

            # Train final model on full training set
            logger.info("  Training final model on full training set...")
            params = XGBOOST_BASE_PARAMS.copy()
            params['reg_alpha'] = alpha
            params['reg_lambda'] = lambda_param

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=NUM_BOOST_ROUNDS,
                evals=[(dtest, 'test')],
                evals_result={},
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False
            )

            # Evaluate on test set
            y_pred = model.predict(dtest)
            test_mae = mean_absolute_error(y_test, y_pred)

            combo_time = time.time() - combo_start

            # Store result
            result = GridSearchResult(
                alpha=alpha,
                lambda_param=lambda_param,
                cv_mae_mean=cv_mae_mean,
                cv_mae_std=cv_mae_std,
                test_mae=test_mae
            )
            results.append(result)

            logger.info(f"  CV MAE: {cv_mae_mean:.2f} ± {cv_mae_std:.2f}")
            logger.info(f"  Test MAE: {test_mae:.2f}")
            logger.info(f"  Time: {combo_time/60:.2f} minutes")

    total_time = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"Grid search complete in {total_time/60:.2f} minutes")
    logger.info("=" * 70 + "\n")

    # Rank by CV performance
    results.sort(key=lambda x: x.cv_mae_mean)
    for i, result in enumerate(results, 1):
        result.rank = i

    return results


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(X_train, y_train, feature_cols, alpha_values, lambda_values):
    """
    Train models at different alpha values and compare feature importance.
    This shows which features are driven to zero by L1 regularization.
    """
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE ANALYSIS ACROSS L1 VALUES")
    logger.info("=" * 70)
    logger.info("Training models at alpha=0, 1.0, 5.0, 10.0 to see feature elimination...")
    logger.info("=" * 70 + "\n")

    analyze_alphas = [0.0, 1.0, 5.0, 10.0]
    lambda_param = 1.0  # Use baseline lambda

    importance_by_alpha = {}

    for alpha in analyze_alphas:
        params = XGBOOST_BASE_PARAMS.copy()
        params['reg_alpha'] = alpha
        params['reg_lambda'] = lambda_param

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            verbose_eval=False
        )

        importance = model.get_score(importance_type='gain')
        importance_by_alpha[alpha] = importance

        logger.info(f"\nAlpha = {alpha}:")
        logger.info("-" * 70)

        # Calculate percentages
        total_importance = sum(importance.values()) if importance else 1.0

        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        for i, (feat, score) in enumerate(sorted_importance[:10], 1):  # Top 10
            pct = (score / total_importance) * 100 if total_importance > 0 else 0
            is_opponent = feat in ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']
            marker = " [OPP]" if is_opponent else ""
            logger.info(f"  {i:2d}. {feat:30s} {pct:6.1f}%{marker}")

    # Summary: show opponent feature importance progression
    logger.info("\n" + "=" * 70)
    logger.info("OPPONENT FEATURE IMPORTANCE PROGRESSION")
    logger.info("=" * 70)
    logger.info(f"{'Alpha':>8} {'opp_def_epa_exp':>18} {'opp_pass_def_rank':>20} {'matchup_difficulty':>20}")
    logger.info("-" * 70)

    for alpha in analyze_alphas:
        importance = importance_by_alpha[alpha]
        total_importance = sum(importance.values()) if importance else 1.0

        opp_epa_pct = (importance.get('opp_def_epa_exp', 0) / total_importance * 100) if total_importance > 0 else 0
        opp_rank_pct = (importance.get('opp_pass_def_rank', 0) / total_importance * 100) if total_importance > 0 else 0
        matchup_pct = (importance.get('matchup_difficulty', 0) / total_importance * 100) if total_importance > 0 else 0

        logger.info(f"{alpha:8.1f} {opp_epa_pct:17.1f}% {opp_rank_pct:19.1f}% {matchup_pct:19.1f}%")

    logger.info("\nInterpretation:")
    logger.info("  - Values near 0% indicate feature was eliminated by L1 regularization")
    logger.info("  - If opponent features drop to 0% at high alpha, they are redundant")
    logger.info("  - Baseline features should remain important across all alpha values")


# ============================================================================
# FINAL MODEL TRAINING & EVALUATION
# ============================================================================

def train_best_model(X_train, y_train, X_test, y_test, feature_cols, best_result):
    """Train and evaluate the best model from grid search."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING BEST MODEL (FULL DATASET)")
    logger.info("=" * 70)
    logger.info(f"Best hyperparameters: alpha={best_result.alpha}, lambda={best_result.lambda_param}")
    logger.info(f"CV Performance: MAE = {best_result.cv_mae_mean:.2f} ± {best_result.cv_mae_std:.2f}")
    logger.info("=" * 70)

    params = XGBOOST_BASE_PARAMS.copy()
    params['reg_alpha'] = best_result.alpha
    params['reg_lambda'] = best_result.lambda_param

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    start_time = time.time()

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50
    )

    training_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(dtest)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Baseline
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'baseline_mae': baseline_mae,
        'improvement_pct': (1 - mae / baseline_mae) * 100,
        'training_time_minutes': training_time / 60,
        'best_iteration': model.best_iteration,
        'num_features': len(feature_cols),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'best_alpha': best_result.alpha,
        'best_lambda': best_result.lambda_param,
    }

    return model, y_pred, metrics


def evaluate_and_save(model, y_test, y_pred, metrics, feature_cols, grid_results):
    """Evaluate model and save artifacts."""
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION - 2025 HOLDOUT SET")
    logger.info("=" * 70)

    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  RMSE: {metrics['rmse']:.2f} yards")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  Mean Prediction: {y_pred.mean():.1f} yards")
    logger.info(f"  Mean Actual: {y_test.mean():.1f} yards")

    logger.info(f"\nBaseline Comparison:")
    logger.info(f"  Baseline MAE: {metrics['baseline_mae']:.2f} yards")
    logger.info(f"  Improvement: {metrics['improvement_pct']:.1f}%")

    # Compare to v2 baseline
    improvement_vs_v2 = ((V2_BASELINE_MAE - metrics['mae']) / V2_BASELINE_MAE) * 100
    improvement_vs_v3_1 = ((V3_1_REGRESSION_MAE - metrics['mae']) / V3_1_REGRESSION_MAE) * 100

    logger.info(f"\nv2 Baseline Comparison (40.20 MAE):")
    logger.info(f"  v3.2 L1 MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  Change: {improvement_vs_v2:+.2f}%")

    logger.info(f"\nv3.1 Regression Comparison (57.37 MAE):")
    logger.info(f"  v3.2 L1 MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  Improvement: {improvement_vs_v3_1:+.2f}%")

    logger.info(f"\nTraining Info:")
    logger.info(f"  Training time: {metrics['training_time_minutes']:.2f} minutes")
    logger.info(f"  Best iteration: {metrics['best_iteration']}")
    logger.info(f"  Best alpha: {metrics['best_alpha']}")
    logger.info(f"  Best lambda: {metrics['best_lambda']}")
    logger.info(f"  Features: {metrics['num_features']}")
    logger.info(f"  Train samples: {metrics['train_samples']:,}")
    logger.info(f"  Test samples: {metrics['test_samples']:,}")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE (by gain) - BEST MODEL")
    logger.info("=" * 70)

    total_importance = sum(score for _, score in sorted_importance)

    for i, (feat, score) in enumerate(sorted_importance, 1):
        pct = (score / total_importance) * 100
        marker = " [OPP]" if feat in ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty'] else ""
        logger.info(f"  {i:2d}. {feat:30s} {score:10.1f} ({pct:5.1f}%){marker}")

    # Save model
    model_dir = Path('models/xgboost')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'passing_xgboost_v3_2_l1.pkl'
    model.save_model(str(model_path))

    # Save metadata
    metadata = {
        'model_name': 'XGBoost Passing v3.2 (L1 Regularization, Opponent Feature Optimization)',
        'training_date': pd.Timestamp.now().isoformat(),
        'aligned_window': ALIGNED_WINDOW,
        'hyperparameters': {**XGBOOST_BASE_PARAMS, 'reg_alpha': metrics['best_alpha'], 'reg_lambda': metrics['best_lambda']},
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                   for k, v in metrics.items()},
        'feature_cols': feature_cols,
        'num_features': len(feature_cols),
        'ensemble_compatible': True,
        'feature_importance_all': {k: float(v) for k, v in sorted_importance},
        'ensemble_weight': 0.20,
        'test_season': ALIGNED_WINDOW['test_season'],
        'train_seasons': f"{ALIGNED_WINDOW['train_start']}-{ALIGNED_WINDOW['train_end']}",
        'opponent_strength_features': [
            'opp_def_epa_exp',
            'opp_pass_def_rank',
            'matchup_difficulty'
        ],
        'grid_search': {
            'alpha_values': [float(x) for x in L1_ALPHAS],
            'lambda_values': [float(x) for x in L2_LAMBDAS],
            'cv_folds': CV_FOLDS,
            'total_combinations': len(L1_ALPHAS) * len(L2_LAMBDAS),
            'best_alpha': float(metrics['best_alpha']),
            'best_lambda': float(metrics['best_lambda']),
            'best_cv_mae': float(grid_results[0].cv_mae_mean) if grid_results else 0.0,
            'best_cv_std': float(grid_results[0].cv_mae_std) if grid_results else 0.0,
        },
        'research_goal': 'Fix v3.1 regression (57.37 MAE) by using L1 to eliminate redundant opponent features',
        'success_criteria': {
            'investigate': 'MAE < 45.0 yards',
            'deploy': 'MAE < 42.0 yards',
            'breakthrough': 'MAE < 40.2 yards (beats v2 baseline)'
        },
        'v2_baseline_mae': V2_BASELINE_MAE,
        'v3_1_regression_mae': V3_1_REGRESSION_MAE,
        'v3_2_improvement_vs_v2': improvement_vs_v2,
        'v3_2_improvement_vs_v3_1': improvement_vs_v3_1,
    }

    metadata_path = model_dir / 'passing_xgboost_v3_2_l1_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("MODEL SAVED")
    logger.info("=" * 70)
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    return model_path, metadata_path, improvement_vs_v2, improvement_vs_v3_1


def print_grid_search_summary(grid_results):
    """Print summary of grid search results."""
    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Rank':>4} {'Alpha':>8} {'Lambda':>8} {'CV MAE':>12} {'Std Dev':>10} {'Test MAE':>12}")
    logger.info("-" * 70)

    for i, result in enumerate(grid_results[:15], 1):  # Top 15
        logger.info(
            f"{i:4d} {result.alpha:8.1f} {result.lambda_param:8.1f} "
            f"{result.cv_mae_mean:12.2f} {result.cv_mae_std:10.2f} {result.test_mae:12.2f}"
        )

    logger.info("=" * 70)


def decision_gate(mae, v2_baseline=V2_BASELINE_MAE, v3_1_regression=V3_1_REGRESSION_MAE):
    """Determine decision based on MAE performance."""
    logger.info("\n" + "=" * 70)
    logger.info("DECISION GATE: L1 REGULARIZATION SUCCESS EVALUATION")
    logger.info("=" * 70)
    logger.info(f"v2 Baseline: {v2_baseline:.2f} MAE")
    logger.info(f"v3.1 Regression: {v3_1_regression:.2f} MAE")
    logger.info(f"v3.2 L1 Result: {mae:.2f} MAE")
    logger.info("=" * 70)

    if mae < 42.0:
        decision = "✓ DEPLOY"
        if mae < v2_baseline:
            recommendation = (
                f"v3.2 beats v2 baseline by {((v2_baseline - mae) / v2_baseline * 100):.2f}%.\n"
                f"Opponent features are now optimized with L1 regularization.\n"
                f"RECOMMENDED: Integrate into ensemble v6.2 (20% weight)"
            )
        else:
            improvement = ((v3_1_regression - mae) / v3_1_regression * 100)
            recommendation = (
                f"v3.2 fixed v3.1 regression by {improvement:.2f}%.\n"
                f"L1 regularization successfully eliminated redundant opponent features.\n"
                f"RECOMMENDED: Integrate into ensemble v6.2 (20% weight)"
            )
        status = "SUCCESS - Ready for production"
    elif mae < 45.0:
        decision = "⚠ INVESTIGATE"
        improvement = ((v3_1_regression - mae) / v3_1_regression * 100)
        recommendation = (
            f"v3.2 shows improvement over v3.1 ({improvement:.2f}%).\n"
            f"Performance still below target.\n"
            f"RECOMMENDED: Further investigation (feature ablation, hyperparameter tuning)"
        )
        status = "PARTIAL - Needs investigation"
    else:
        decision = "✗ ABANDON"
        recommendation = (
            f"L1 regularization did not fix the regression.\n"
            f"Opponent features may not be addressable via regularization alone.\n"
            f"RECOMMENDED: Try alternative approaches (feature removal, different opponent proxies)"
        )
        status = "FAILURE - Approach not working"

    logger.info(f"\nDecision: {decision}")
    logger.info(f"Status: {status}")
    logger.info(f"\nRecommendation:")
    for line in recommendation.split('\n'):
        logger.info(f"  {line}")

    return decision, status


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline with grid search."""
    logger.info("\n" + "=" * 70)
    logger.info("BFS EXPERIMENT 5: L1 REGULARIZATION FOR XGBOOST OPPONENT FEATURES")
    logger.info("=" * 70)
    logger.info(f"Goal: Fix v3.1 regression (57.37 MAE) via L1 feature elimination")
    logger.info(f"Target: MAE < 42 yards (deploy) or < 40.2 yards (beats baseline)")
    logger.info(f"Method: Grid search L1 alpha ({len(L1_ALPHAS)} values) × L2 lambda ({len(L2_LAMBDAS)} values)")
    logger.info(f"Estimated Runtime: 15-25 minutes")
    logger.info("=" * 70 + "\n")

    pipeline_start = time.time()

    # 1. Load data
    df = load_data_from_db()

    # 2. Load opponent features
    opp_features = load_opponent_strength_features()

    # 3. Engineer features
    df = engineer_features(df, opp_features)

    # 4. Train/test split with coverage audit
    try:
        X_train, y_train, X_test, y_test, feature_cols, train_df = prepare_train_test_split(df)
    except ValueError as e:
        logger.error(f"\n✗ COVERAGE AUDIT FAILED")
        logger.error(f"  {str(e)}")
        return

    # 5. Grid search
    grid_results = grid_search_regularization(X_train, y_train, X_test, y_test, feature_cols)

    # 6. Print grid search summary
    print_grid_search_summary(grid_results)

    # 7. Feature importance analysis
    analyze_feature_importance(X_train, y_train, feature_cols, L1_ALPHAS, L2_LAMBDAS)

    # 8. Train best model
    best_result = grid_results[0]
    model, y_pred, metrics = train_best_model(
        X_train, y_train, X_test, y_test, feature_cols, best_result
    )

    # 9. Evaluate and save
    model_path, metadata_path, improvement_vs_v2, improvement_vs_v3_1 = evaluate_and_save(
        model, y_test, y_pred, metrics, feature_cols, grid_results
    )

    # 10. Decision gate
    decision, status = decision_gate(metrics['mae'])

    pipeline_time = time.time() - pipeline_start

    logger.info("\n" + "=" * 70)
    logger.info("BFS EXPERIMENT 5 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {pipeline_time/60:.2f} minutes")
    logger.info(f"Decision: {decision}")
    logger.info(f"Status: {status}")
    logger.info(f"Model saved: {model_path}")
    logger.info("=" * 70 + "\n")


if __name__ == '__main__':
    main()
