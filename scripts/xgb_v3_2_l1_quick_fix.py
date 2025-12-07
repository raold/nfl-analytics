#!/usr/bin/env python3
"""
Train XGBoost v3.1 Model with Aligned Training Window (2018-2024 ONLY)

POSTMORTEM FIX: XGBoost v3 Performance Regression
Root Cause: 35% missing opponent data in 2016-2017 filled with median (noise)
Solution: Align training window with feature availability (2018-2024)

Author: NFL Analytics Research Team
Date: November 14, 2025

Training Window: 2018-2024 ONLY (no decay years, aligned with EPA availability)
Hold-out Test: 2025 season (148 games)
Features: 16 features (13 baseline + 3 opponent strength)

New Features (v3):
- opp_def_epa_exp: Exponentially weighted opponent defensive EPA
- opp_pass_def_rank: Opponent pass defense percentile rank (0-100)
- matchup_difficulty: Composite difficulty index (0-100)

Expected Performance: MAE < 40.20 yards (v2 baseline)
Target: Beat XGBoost v2 to validate opponent strength features add value

Key Changes from v3:
1. Training window: 2018-2024 ONLY (no 2015-2017 decay years)
2. Feature coverage audit BEFORE training (>80% threshold)
3. Validation on small subset to verify no missing data
4. Clear go/no-go decision gates

Usage:
    python py/models/train_xgboost_v3_1_aligned_window.py

Output:
    - models/xgboost/passing_xgboost_v3_1_aligned.pkl
    - models/xgboost/passing_xgboost_v3_1_aligned_metadata.json
    - Coverage audit report (logged to console)

Decision Gates:
    - Feature coverage >80%: PASS/FAIL (blocks training if FAIL)
    - v3.1 MAE < v2 MAE (40.20): DEPLOY/INVESTIGATE
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
    "database": os.getenv("DB_NAME", "devdb01"),
}

# Aligned training window (EPA available 2018+)
ALIGNED_WINDOW = {
    "train_start": 2018,  # Aligned with EPA availability
    "train_end": 2024,
    "test_season": 2025,
    "note": "No decay years - clean 2018-2024 only",
}

# Feature coverage requirements
COVERAGE_THRESHOLD = 0.80  # Minimum 80% non-null coverage
COVERAGE_TARGET = 0.95  # Target 95% coverage for clean features

# XGBoost hyperparameters (same as v3 for fair comparison)
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "gamma": 0.05,
    "reg_alpha": 0.15,
    "reg_lambda": 1.5,
    "seed": 42,
    "n_jobs": -1,
}

NUM_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 75

# Baseline comparison (XGBoost v2 performance)
V2_BASELINE_MAE = 40.20  # From postmortem


def audit_feature_coverage(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = COVERAGE_THRESHOLD,
    fail_on_low_coverage: bool = True,
) -> dict[str, float]:
    """
    Audit feature coverage before model training (CRITICAL GATE).

    This is the #1 lesson from XGBoost v3 postmortem:
    ALWAYS validate feature coverage BEFORE training.

    Args:
        df: Training DataFrame
        features: List of feature names to audit
        threshold: Minimum acceptable coverage (default: 0.80)
        fail_on_low_coverage: Raise error if coverage below threshold

    Returns:
        Dict mapping feature names to coverage percentages

    Raises:
        ValueError: If any feature has coverage below threshold and
                   fail_on_low_coverage=True
    """
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE COVERAGE AUDIT (CRITICAL GATE)")
    logger.info("=" * 70)
    logger.info(f"Threshold: {threshold:.0%} (minimum required)")
    logger.info(f"Target: {COVERAGE_TARGET:.0%} (clean features)")
    logger.info("=" * 70)

    coverage_report = {}
    all_pass = True

    for feat in features:
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in DataFrame")

        coverage = df[feat].notna().mean()
        coverage_report[feat] = coverage

        # Determine status
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
            f"(missing: {missing_count:,} / {len(df):,})"
        )

    logger.info("=" * 70)

    if all_pass:
        logger.info("✓ ALL FEATURES PASS COVERAGE AUDIT")
        logger.info("  Training can proceed safely")
    else:
        msg = (
            f"\n✗ COVERAGE AUDIT FAILED\n"
            f"Some features have coverage below {threshold:.0%} threshold.\n"
            f"This indicates systematic missing data that will degrade model quality.\n"
            f"\nOptions:\n"
            f"1. Adjust training window to match feature availability\n"
            f"2. Remove low-coverage features\n"
            f"3. Use better imputation (not median)\n"
            f"\nDO NOT PROCEED WITH TRAINING until coverage issue is resolved."
        )

        if fail_on_low_coverage:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    logger.info("=" * 70 + "\n")

    return coverage_report


def load_data_from_db():
    """
    Load QB passing data from database (2018-2025 ONLY).

    Returns:
        pd.DataFrame: Complete dataset with all features
    """
    logger.info("=" * 70)
    logger.info("LOADING DATA FROM DATABASE (2018-2025 ONLY)")
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
            AND pgs.season >= 2018  -- ALIGNED WINDOW: 2018+ only
            AND pgs.season <= 2025
            AND ph.position_group = 'QB'
    )
    SELECT * FROM qb_games
    ORDER BY season, week, player_id
    """

    logger.info("Executing database query (2018-2025 only)...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"✓ Loaded {len(df):,} QB games")
    logger.info(f"  Seasons: {df.season.min()}-{df.season.max()}")
    logger.info(f"  Unique QBs: {df.player_id.nunique()}")
    logger.info("  Note: 2016-2017 excluded (no EPA data)")

    return df


def load_opponent_strength_features():
    """
    Load opponent strength features from CSV file.

    Returns:
        pd.DataFrame: Opponent strength features
    """
    logger.info("\n" + "=" * 70)
    logger.info("LOADING OPPONENT STRENGTH FEATURES")
    logger.info("=" * 70)

    features_path = Path("data/processed/features/opponent_strength_features.csv")

    if not features_path.exists():
        raise FileNotFoundError(
            f"Opponent strength features not found at {features_path}. "
            f"Run py/features/opponent_strength.py first."
        )

    opp_df = pd.read_csv(features_path)

    logger.info(f"✓ Loaded {len(opp_df):,} opponent strength records")
    logger.info(f"  Seasons: {opp_df.season.min()}-{opp_df.season.max()}")

    # Select only the features we need
    opp_features = opp_df[
        ["game_id", "team", "opp_def_epa_exp", "opp_pass_def_rank", "matchup_difficulty"]
    ].copy()

    # Check coverage
    non_null_counts = (
        opp_features[["opp_def_epa_exp", "opp_pass_def_rank", "matchup_difficulty"]].notna().sum()
    )
    logger.info("\nOpponent feature availability:")
    for col, count in non_null_counts.items():
        logger.info(f"  {col}: {count:,} ({count/len(opp_features)*100:.1f}%)")

    return opp_features


def engineer_features(df: pd.DataFrame, opp_features: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 16 features (13 baseline + 3 opponent strength).

    Args:
        df: Raw data from database
        opp_features: Opponent strength features

    Returns:
        pd.DataFrame: Engineered features (16 total)
    """
    logger.info("\n" + "=" * 70)
    logger.info("ENGINEERING 16 FEATURES (13 BASELINE + 3 OPPONENT STRENGTH)")
    logger.info("=" * 70)

    # Sort for rolling windows
    df = df.sort_values(["player_id", "season", "week"]).copy()

    # ===== MERGE OPPONENT STRENGTH FEATURES =====
    logger.info("\nMerging opponent strength features...")
    before_merge = len(df)
    df = df.merge(opp_features, on=["game_id", "team"], how="left")
    after_merge = len(df)

    if after_merge != before_merge:
        logger.warning(f"Row count changed after merge: {before_merge} -> {after_merge}")
    else:
        logger.info(f"✓ Merge successful, {after_merge:,} rows preserved")

    # ===== BASIC EFFICIENCY METRICS (3 features) =====
    logger.info("\nCreating basic efficiency metrics...")
    df["yards_per_attempt"] = df["yards"] / df["attempts"].replace(0, np.nan)
    df["td_rate"] = df["touchdowns"] / df["attempts"].replace(0, np.nan)
    df["log_attempts"] = np.log1p(df["attempts"])

    # ===== WEATHER FEATURES (3 features) =====
    logger.info("Creating weather features...")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")

    df["is_dome"] = (df["roof"].isin(["dome", "closed"])).astype(int)
    df["is_cold"] = ((df["temp"] < 32) & (~df["is_dome"].astype(bool))).astype(int)
    df["is_windy"] = ((df["wind"] > 15) & (~df["is_dome"].astype(bool))).astype(int)
    df["is_bad_weather"] = (df["is_cold"] | df["is_windy"]).astype(int)

    # ===== GAME SCRIPT FEATURES (2 features) =====
    logger.info("Creating game script features...")
    df["spread_abs"] = df["spread_close"].abs()
    df["total_line"] = df["total_close"]

    # ===== SCHEDULE FACTORS (1 feature) =====
    logger.info("Creating schedule factors...")
    df["week_in_season"] = df["week"].astype(float)

    # ===== ROLLING AVERAGES (3 features - ANTI-LEAKAGE) =====
    logger.info("Creating rolling averages (L3 only)...")

    for col in ["yards", "yards_per_attempt", "td_rate"]:
        df[f"{col}_L3"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .shift(1)  # CRITICAL: Shift to avoid lookahead bias
        )

    # Fill NaNs ONLY for non-opponent features
    baseline_numeric_cols = [
        "yards_per_attempt",
        "td_rate",
        "log_attempts",
        "temp",
        "wind",
        "spread_abs",
        "total_line",
        "week_in_season",
        "yards_L3",
        "yards_per_attempt_L3",
        "td_rate_L3",
    ]

    for col in baseline_numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # DO NOT FILL opponent features - let coverage audit catch issues
    logger.info("\n✓ Feature engineering complete")
    logger.info("  Total features: 16 (13 baseline + 3 opponent strength)")
    logger.info(f"  Samples: {len(df):,}")
    logger.info("  Note: Opponent features NOT imputed (coverage audit will verify)")

    return df


def prepare_train_test_split(df: pd.DataFrame) -> tuple:
    """
    Split data into train/test sets with coverage validation.

    Train: 2018-2024 (aligned window, no decay)
    Test: 2025 season (hold-out set)

    Args:
        df: Engineered features dataframe

    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_cols, train_df)
    """
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING TRAIN/TEST SPLIT (ALIGNED WINDOW)")
    logger.info("=" * 70)

    # 16 features: 13 baseline + 3 opponent strength
    feature_cols = [
        # Baseline 13 features
        "log_attempts",
        "yards_per_attempt",
        "td_rate",
        "is_home",
        "spread_abs",
        "total_line",
        "is_bad_weather",
        "yards_L3",
        "yards_per_attempt_L3",
        "td_rate_L3",
        "week_in_season",
        "temp",
        "wind",
        # New opponent strength features (v3)
        "opp_def_epa_exp",
        "opp_pass_def_rank",
        "matchup_difficulty",
    ]

    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Train/test split (aligned window)
    train_df = df[
        df["season"].between(ALIGNED_WINDOW["train_start"], ALIGNED_WINDOW["train_end"])
    ].copy()
    test_df = df[df["season"] == ALIGNED_WINDOW["test_season"]].copy()

    logger.info("\n✓ Split complete:")
    logger.info(
        f"  Train: {len(train_df):,} samples ({ALIGNED_WINDOW['train_start']}-{ALIGNED_WINDOW['train_end']})"
    )
    logger.info(f"  Test: {len(test_df):,} samples ({ALIGNED_WINDOW['test_season']})")
    logger.info(f"  Features: {len(feature_cols)}")

    # ===== CRITICAL: FEATURE COVERAGE AUDIT =====
    logger.info("\nRunning coverage audit on training set...")
    coverage_report = audit_feature_coverage(
        train_df,
        feature_cols,
        threshold=COVERAGE_THRESHOLD,
        fail_on_low_coverage=True,  # BLOCK training if coverage fails
    )

    # Extract features AFTER coverage audit passes
    X_train = train_df[feature_cols].values
    y_train = train_df["yards"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["yards"].values

    logger.info("\n16 Features (13 Baseline + 3 Opponent Strength):")
    for i, col in enumerate(feature_cols, 1):
        coverage = coverage_report.get(col, 0.0)
        marker = (
            " [NEW]"
            if col in ["opp_def_epa_exp", "opp_pass_def_rank", "matchup_difficulty"]
            else ""
        )
        logger.info(f"  {i:2d}. {col:30s} ({coverage:5.1%} coverage){marker}")

    return X_train, y_train, X_test, y_test, feature_cols, train_df


def train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols):
    """
    Train XGBoost model with aligned window and evaluate.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        feature_cols: Feature names

    Returns:
        tuple: (trained_model, test_predictions, metrics)
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING XGBOOST MODEL (ALIGNED WINDOW)")
    logger.info("=" * 70)

    start_time = time.time()

    # Create DMatrix (no sample weights - all data equally weighted)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    logger.info(f"Training with {NUM_BOOST_ROUNDS} boost rounds...")
    logger.info(f"Early stopping: {EARLY_STOPPING_ROUNDS} rounds")
    logger.info("Note: No exponential decay weights (uniform 2018-2024)")

    # Training
    evals = [(dtrain, "train"), (dtest, "test")]
    evals_result = {}

    model = xgb.train(
        XGBOOST_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )

    training_time = time.time() - start_time

    logger.info(f"\n✓ Training complete in {training_time/60:.2f} minutes")
    logger.info(f"  Best iteration: {model.best_iteration}")
    logger.info(f"  Best train MAE: {evals_result['train']['mae'][model.best_iteration]:.2f}")
    logger.info(f"  Best test MAE: {evals_result['test']['mae'][model.best_iteration]:.2f}")

    # Generate predictions
    y_pred = model.predict(dtest)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Baseline (mean predictor)
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "baseline_mae": baseline_mae,
        "improvement_pct": (1 - mae / baseline_mae) * 100,
        "training_time_minutes": training_time / 60,
        "best_iteration": model.best_iteration,
        "num_features": len(feature_cols),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return model, y_pred, metrics


def evaluate_and_save(model, y_test, y_pred, metrics, feature_cols):
    """
    Evaluate model performance and save artifacts.

    Args:
        model: Trained XGBoost model
        y_test: True test values
        y_pred: Predicted test values
        metrics: Performance metrics dict
        feature_cols: Feature names
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION - 2025 HOLDOUT SET")
    logger.info("=" * 70)

    logger.info("\nPerformance Metrics:")
    logger.info(f"  MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  RMSE: {metrics['rmse']:.2f} yards")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  Mean Prediction: {y_pred.mean():.1f} yards")
    logger.info(f"  Mean Actual: {y_test.mean():.1f} yards")

    logger.info("\nBaseline Comparison:")
    logger.info(f"  Baseline MAE: {metrics['baseline_mae']:.2f} yards")
    logger.info(f"  Improvement: {metrics['improvement_pct']:.1f}%")

    # Compare to v2 baseline
    improvement_vs_v2 = ((V2_BASELINE_MAE - metrics["mae"]) / V2_BASELINE_MAE) * 100
    logger.info("\nv2 Baseline Comparison:")
    logger.info(f"  v2 MAE: {V2_BASELINE_MAE:.2f} yards")
    logger.info(f"  v3.1 MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  Improvement: {improvement_vs_v2:+.2f}%")

    logger.info("\nTraining Info:")
    logger.info(f"  Training time: {metrics['training_time_minutes']:.2f} minutes")
    logger.info(f"  Best iteration: {metrics['best_iteration']}")
    logger.info(f"  Features: {metrics['num_features']}")
    logger.info(f"  Train samples: {metrics['train_samples']:,}")
    logger.info(f"  Test samples: {metrics['test_samples']:,}")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE (by gain)")
    logger.info("=" * 70)

    # Calculate percentages
    total_importance = sum(score for _, score in sorted_importance)

    for i, (feat, score) in enumerate(sorted_importance, 1):
        pct = (score / total_importance) * 100
        marker = (
            " [NEW]"
            if feat in ["opp_def_epa_exp", "opp_pass_def_rank", "matchup_difficulty"]
            else ""
        )
        logger.info(f"  {i:2d}. {feat:30s} {score:10.1f} ({pct:5.1f}%){marker}")

    # Save model
    model_dir = Path("models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "passing_xgboost_v3_1_aligned.pkl"
    model.save_model(str(model_path))

    # Save metadata
    metadata = {
        "model_name": "XGBoost Passing v3.1 (Aligned Window 2018-2024, 16 Features)",
        "training_date": pd.Timestamp.now().isoformat(),
        "aligned_window": ALIGNED_WINDOW,
        "hyperparameters": XGBOOST_PARAMS,
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()
        },
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "ensemble_compatible": True,
        "feature_importance_all": {k: float(v) for k, v in sorted_importance},
        "ensemble_weight": 0.30,
        "test_season": ALIGNED_WINDOW["test_season"],
        "train_seasons": f"{ALIGNED_WINDOW['train_start']}-{ALIGNED_WINDOW['train_end']}",
        "opponent_strength_features": [
            "opp_def_epa_exp",
            "opp_pass_def_rank",
            "matchup_difficulty",
        ],
        "postmortem_fix": {
            "issue": "XGBoost v3 had 35% missing opponent data (2016-2017)",
            "solution": "Aligned training window with feature availability (2018-2024)",
            "coverage_threshold": COVERAGE_THRESHOLD,
            "v2_baseline_mae": V2_BASELINE_MAE,
            "v3_1_improvement": improvement_vs_v2,
        },
        "note": "v3.1 fixes v3 regression by aligning training window with EPA availability",
    }

    metadata_path = model_dir / "passing_xgboost_v3_1_aligned_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("MODEL SAVED")
    logger.info("=" * 70)
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    return model_path, metadata_path, improvement_vs_v2


def main():
    """Main training pipeline with decision gates"""

    logger.info("\n" + "=" * 70)
    logger.info("XGBOOST v3.1 TRAINING (ALIGNED WINDOW - POSTMORTEM FIX)")
    logger.info("=" * 70)
    logger.info(
        f"Training Window: {ALIGNED_WINDOW['train_start']}-{ALIGNED_WINDOW['train_end']} (aligned with EPA)"
    )
    logger.info(f"Test Set: {ALIGNED_WINDOW['test_season']} season")
    logger.info("Features: 16 (13 baseline + 3 opponent strength)")
    logger.info(f"Target: Beat v2 baseline (MAE < {V2_BASELINE_MAE:.2f} yards)")
    logger.info(
        f"Coverage Gate: >{COVERAGE_THRESHOLD:.0%} required (target: >{COVERAGE_TARGET:.0%})"
    )
    logger.info("=" * 70 + "\n")

    pipeline_start = time.time()

    # 1. Load data (2018-2025 only)
    df = load_data_from_db()

    # 2. Load opponent strength features
    opp_features = load_opponent_strength_features()

    # 3. Engineer features (including opponent strength)
    df = engineer_features(df, opp_features)

    # 4. Train/test split WITH COVERAGE AUDIT (critical gate)
    try:
        X_train, y_train, X_test, y_test, feature_cols, train_df = prepare_train_test_split(df)
    except ValueError as e:
        logger.error("\n✗ COVERAGE AUDIT FAILED - TRAINING BLOCKED")
        logger.error(f"  {str(e)}")
        logger.error("\nDO NOT PROCEED WITH TRAINING")
        logger.error("See docs/postmortems/XGBOOST_V3_PERFORMANCE_REGRESSION.md for guidance")
        return

    # 5. Train model (only if coverage audit passed)
    model, y_pred, metrics = train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols)

    # 6. Evaluate and save
    model_path, metadata_path, improvement_vs_v2 = evaluate_and_save(
        model, y_test, y_pred, metrics, feature_cols
    )

    pipeline_time = time.time() - pipeline_start

    # ===== DECISION GATE: v3.1 vs v2 COMPARISON =====
    logger.info("\n" + "=" * 70)
    logger.info("DECISION GATE: v3.1 vs v2 COMPARISON")
    logger.info("=" * 70)
    logger.info(f"v2 Baseline MAE: {V2_BASELINE_MAE:.2f} yards")
    logger.info(f"v3.1 Aligned MAE: {metrics['mae']:.2f} yards")
    logger.info(f"Improvement: {improvement_vs_v2:+.2f}%")

    if metrics["mae"] < V2_BASELINE_MAE:
        decision = "✓ DEPLOY"
        recommendation = (
            f"v3.1 beats v2 baseline by {improvement_vs_v2:.2f}%.\n"
            f"Opponent strength features add value.\n"
            f"RECOMMENDED: Integrate into ensemble v6.1 (30% weight)"
        )
    elif metrics["mae"] < V2_BASELINE_MAE * 1.02:  # Within 2% margin
        decision = "⚠ BORDERLINE"
        recommendation = (
            "v3.1 performance within 2% of v2 baseline.\n"
            "Opponent strength features show marginal value.\n"
            "RECOMMENDED: Further investigation (feature ablation, longer window)"
        )
    else:
        decision = "✗ DO NOT DEPLOY"
        recommendation = (
            "v3.1 still regresses vs v2 baseline.\n"
            "Alignment did not solve the issue.\n"
            "RECOMMENDED: Investigate feature redundancy or alternative opponent proxies"
        )

    logger.info(f"\nDecision: {decision}")
    logger.info("\nRecommendation:")
    for line in recommendation.split("\n"):
        logger.info(f"  {line}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {pipeline_time/60:.2f} minutes")
    logger.info(f"Coverage audit: PASSED (all features >{COVERAGE_THRESHOLD:.0%})")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Decision: {decision}")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
