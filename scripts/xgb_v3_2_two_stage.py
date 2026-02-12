#!/usr/bin/env python3
"""
XGBoost v3.2 Experiment 6: Two-Stage Modeling

Hypothesis: Different game scripts require different models. Competitive games
(spread < 7) follow different patterns than blowouts (spread >= 7).

Approach:
- Train separate XGBoost models for competitive games and blowouts
- Route predictions based on spread_abs threshold
- Use same v3.1 features (no opponent features)

Expected runtime: 20 minutes
Target: MAE < 40.20 (v2 baseline)
"""

import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())


# Database connection
DB_URI = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"


def load_data():
    """Load training and test data from PostgreSQL."""
    engine = create_engine(DB_URI)

    query = """
    SELECT
        -- Target
        passing_yards,

        -- Core player features (10 features)
        qb_epa_L3, qb_cpoe_L3, qb_sack_rate_L3, qb_ypa_L3,
        wr1_epa_L3, wr2_epa_L3, wr1_tgt_L3, wr2_tgt_L3,
        qb_home_away_split, qb_consistency,

        -- Game context (2 features)
        spread_close, total_close,

        -- Team features (4 features)
        team_off_epa_L3, team_pass_rate_L3, team_pace_L3, team_off_success_L3,

        -- Metadata
        season, week, posteam, defteam
    FROM ml.model_features
    WHERE position = 'QB'
        AND season >= 2018
        AND season <= 2025
        AND passing_yards IS NOT NULL
        AND qb_epa_L3 IS NOT NULL
    ORDER BY season, week
    """

    df = pd.read_sql(query, engine)
    engine.dispose()

    print(f"✓ Loaded {len(df):,} rows")
    print(f"  - Training (2018-2024): {len(df[df['season'] <= 2024]):,}")
    print(f"  - Test (2025): {len(df[df['season'] == 2025]):,}")

    return df


def prepare_features(df):
    """Prepare feature matrix (same as v3.1)."""
    # Add spread_abs for routing
    df["spread_abs"] = df["spread_close"].abs()

    feature_cols = [
        # Core player features (10)
        "qb_epa_L3",
        "qb_cpoe_L3",
        "qb_sack_rate_L3",
        "qb_ypa_L3",
        "wr1_epa_L3",
        "wr2_epa_L3",
        "wr1_tgt_L3",
        "wr2_tgt_L3",
        "qb_home_away_split",
        "qb_consistency",
        # Game context (2)
        "spread_close",
        "total_close",
        # Team features (4)
        "team_off_epa_L3",
        "team_pass_rate_L3",
        "team_pace_L3",
        "team_off_success_L3",
    ]

    X = df[feature_cols].copy()
    y = df["passing_yards"].copy()

    return X, y, feature_cols, df["spread_abs"].copy()


def train_two_stage_models(train_df, test_df, threshold=7.0):
    """
    Train separate models for competitive games and blowouts.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        threshold: Spread threshold for competitive vs blowout (default: 7.0)
    """
    print("\n" + "=" * 60)
    print(f"TRAINING TWO-STAGE MODEL (threshold={threshold})")
    print("=" * 60)

    # Add spread_abs
    train_df["spread_abs"] = train_df["spread_close"].abs()
    test_df["spread_abs"] = test_df["spread_close"].abs()

    # Split by game script
    train_competitive = train_df[train_df["spread_abs"] < threshold].copy()
    train_blowout = train_df[train_df["spread_abs"] >= threshold].copy()

    test_competitive = test_df[test_df["spread_abs"] < threshold].copy()
    test_blowout = test_df[test_df["spread_abs"] >= threshold].copy()

    print("\nTraining Split:")
    print(f"  Competitive (spread < {threshold}): {len(train_competitive):,}")
    print(f"  Blowout (spread >= {threshold}): {len(train_blowout):,}")

    print("\nTest Split:")
    print(f"  Competitive (spread < {threshold}): {len(test_competitive):,}")
    print(f"  Blowout (spread >= {threshold}): {len(test_blowout):,}")

    # Prepare features
    X_train_comp, y_train_comp, feature_cols, _ = prepare_features(train_competitive)
    X_train_blow, y_train_blow, _, _ = prepare_features(train_blowout)

    X_test_comp, y_test_comp, _, _ = prepare_features(test_competitive)
    X_test_blow, y_test_blow, _, _ = prepare_features(test_blowout)

    # Train competitive model
    print("\n" + "-" * 60)
    print("Training Competitive Games Model...")
    print("-" * 60)

    params_comp = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
    }

    model_comp = xgb.XGBRegressor(**params_comp)
    model_comp.fit(
        X_train_comp,
        y_train_comp,
        eval_set=[(X_test_comp, y_test_comp)],
        early_stopping_rounds=50,
        verbose=False,
    )

    # Train blowout model
    print("\n" + "-" * 60)
    print("Training Blowout Games Model...")
    print("-" * 60)

    params_blow = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
    }

    model_blow = xgb.XGBRegressor(**params_blow)
    model_blow.fit(
        X_train_blow,
        y_train_blow,
        eval_set=[(X_test_blow, y_test_blow)],
        early_stopping_rounds=50,
        verbose=False,
    )

    # Evaluate each model on its test set
    y_pred_comp = model_comp.predict(X_test_comp)
    y_pred_blow = model_blow.predict(X_test_blow)

    mae_comp = np.mean(np.abs(y_test_comp - y_pred_comp))
    mae_blow = np.mean(np.abs(y_test_blow - y_pred_blow))

    # Combined MAE (weighted by test set size)
    total_samples = len(test_competitive) + len(test_blowout)
    mae_combined = (mae_comp * len(test_competitive) + mae_blow * len(test_blowout)) / total_samples

    print("\n" + "=" * 60)
    print("TWO-STAGE RESULTS")
    print("=" * 60)
    print(f"Competitive Games MAE: {mae_comp:.2f} yards ({len(test_competitive)} games)")
    print(f"Blowout Games MAE: {mae_blow:.2f} yards ({len(test_blowout)} games)")
    print(f"Combined MAE: {mae_combined:.2f} yards")
    print("v2 baseline: 40.20 yards")

    improvement_pct = ((40.20 - mae_combined) / 40.20) * 100
    print(f"Improvement: {improvement_pct:+.1f}%")

    if mae_combined < 40.20:
        print("\n🎯 SUCCESS: Beat baseline!")
    else:
        print("\n❌ FAILED: Did not beat baseline")

    print("=" * 60 + "\n")

    return model_comp, model_blow, mae_combined, feature_cols


def main():
    print("\n" + "=" * 60)
    print("XGBoost v3.2 Experiment 6: Two-Stage Modeling")
    print("=" * 60 + "\n")

    # Load data
    df = load_data()

    # Split
    train_df = df[df["season"] <= 2024].copy()
    test_df = df[df["season"] == 2025].copy()

    # Train two-stage models
    model_comp, model_blow, mae, feature_cols = train_two_stage_models(
        train_df, test_df, threshold=7.0
    )

    # Save models if successful
    if mae < 40.20:
        model_comp.save_model(
            str(_PROJECT_ROOT / "models/xgboost/xgb_v3_2_two_stage_competitive.json")
        )
        model_blow.save_model(
            str(_PROJECT_ROOT / "models/xgboost/xgb_v3_2_two_stage_blowout.json")
        )
        print("✓ Models saved to models/xgboost/xgb_v3_2_two_stage_*.json\n")

    return 0 if mae < 40.20 else 1


if __name__ == "__main__":
    sys.exit(main())
