#!/usr/bin/env python3
"""
XGBoost v3.2 Experiment 3: Temporal Opponent Features

Hypothesis: Recent opponent performance (L3 games) is more predictive than
season-long opponent EPA.

Approach:
- Replace opp_def_epa_exp with opp_def_epa_L3
- Add opp_sacks_L3 (recent pass rush pressure)
- Add opp_def_epa_trend (slope of last 5 games)

Expected runtime: 25 minutes
Target: MAE < 40.20 (v2 baseline)
"""

import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine

# Database connection
DB_URI = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"


def load_data():
    """Load training and test data with temporal opponent features."""
    engine = create_engine(DB_URI)

    # First, get base features
    query = """
    WITH defensive_rolling AS (
        SELECT
            defteam,
            season,
            week,
            -- L3 defensive EPA
            AVG(def_epa_play) OVER (
                PARTITION BY defteam, season
                ORDER BY week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) AS def_epa_L3,
            -- L3 sacks per game
            AVG(sacks) OVER (
                PARTITION BY defteam, season
                ORDER BY week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) AS sacks_L3,
            -- L5 defensive EPA for trend calculation
            REGR_SLOPE(def_epa_play, week) OVER (
                PARTITION BY defteam, season
                ORDER BY week
                ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
            ) AS def_epa_trend
        FROM mart.game_summary
        WHERE season >= 2018
    )
    SELECT
        mf.passing_yards,

        -- Core player features (10 features)
        mf.qb_epa_L3, mf.qb_cpoe_L3, mf.qb_sack_rate_L3, mf.qb_ypa_L3,
        mf.wr1_epa_L3, mf.wr2_epa_L3, mf.wr1_tgt_L3, mf.wr2_tgt_L3,
        mf.qb_home_away_split, mf.qb_consistency,

        -- Game context (2 features)
        mf.spread_close, mf.total_close,

        -- Temporal opponent features (3 features)
        dr.def_epa_L3 AS opp_def_epa_L3,
        dr.sacks_L3 AS opp_sacks_L3,
        dr.def_epa_trend AS opp_def_epa_trend,

        -- Team features (4 features)
        mf.team_off_epa_L3, mf.team_pass_rate_L3, mf.team_pace_L3, mf.team_off_success_L3,

        -- Metadata
        mf.season, mf.week, mf.posteam, mf.defteam
    FROM ml.model_features mf
    LEFT JOIN defensive_rolling dr
        ON mf.defteam = dr.defteam
        AND mf.season = dr.season
        AND mf.week = dr.week
    WHERE mf.position = 'QB'
        AND mf.season >= 2018
        AND mf.season <= 2025
        AND mf.passing_yards IS NOT NULL
        AND mf.qb_epa_L3 IS NOT NULL
        AND dr.def_epa_L3 IS NOT NULL  -- Require temporal features
    ORDER BY mf.season, mf.week
    """

    df = pd.read_sql(query, engine)
    engine.dispose()

    print(f"✓ Loaded {len(df):,} rows")
    print(f"  - Training (2018-2024): {len(df[df['season'] <= 2024]):,}")
    print(f"  - Test (2025): {len(df[df['season'] == 2025]):,}")

    return df


def prepare_features(df):
    """Prepare feature matrix."""
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
        # Temporal opponent features (3)
        "opp_def_epa_L3",
        "opp_sacks_L3",
        "opp_def_epa_trend",
        # Team features (4)
        "team_off_epa_L3",
        "team_pass_rate_L3",
        "team_pace_L3",
        "team_off_success_L3",
    ]

    X = df[feature_cols].copy()
    y = df["passing_yards"].copy()

    return X, y, feature_cols


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost and evaluate."""
    print("\n" + "=" * 60)
    print("TRAINING XGBoost v3.2 (Temporal Opponent Features)")
    print("=" * 60)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(feature_names)}")
    print("=" * 60 + "\n")

    # Train model with same hyperparameters as v3.1
    params = {
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

    model = xgb.XGBRegressor(**params)

    start_time = time.time()
    model.fit(
        X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False
    )
    train_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # Feature importance
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"MAE: {mae:.2f} yards")
    print(f"RMSE: {rmse:.2f} yards")
    print(f"Training time: {train_time:.1f}s")
    print("v2 baseline: 40.20 yards")

    improvement_pct = ((40.20 - mae) / 40.20) * 100
    print(f"Improvement: {improvement_pct:+.1f}%")

    if mae < 40.20:
        print("\n🎯 SUCCESS: Beat baseline!")
    else:
        print("\n❌ FAILED: Did not beat baseline")

    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    print("=" * 60 + "\n")

    return model, mae, importance_df


def main():
    print("\n" + "=" * 60)
    print("XGBoost v3.2 Experiment 3: Temporal Opponent Features")
    print("=" * 60 + "\n")

    # Load data
    df = load_data()

    # Split
    train_df = df[df["season"] <= 2024].copy()
    test_df = df[df["season"] == 2025].copy()

    # Prepare features
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    # Train and evaluate
    model, mae, importance_df = train_and_evaluate(X_train, y_train, X_test, y_test, feature_names)

    # Save model if successful
    if mae < 40.20:
        model.save_model("/Users/dro/rice/nfl-analytics/models/xgboost/xgb_v3_2_temporal_opp.json")
        importance_df.to_csv(
            "/Users/dro/rice/nfl-analytics/models/xgboost/xgb_v3_2_temporal_opp_importance.csv",
            index=False,
        )
        print("✓ Model saved to models/xgboost/xgb_v3_2_temporal_opp.json\n")

    return 0 if mae < 40.20 else 1


if __name__ == "__main__":
    sys.exit(main())
