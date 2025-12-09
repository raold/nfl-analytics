#!/usr/bin/env python3
"""
XGBoost v3.2 Experiment 7: Ridge Residuals

Hypothesis: Use Ridge regression to predict yards from opponent features,
then use the residuals (yards beyond opponent expectation) as features
in XGBoost. This isolates player skill from opponent strength.

Approach:
- Train RidgeCV to predict yards from opponent features only
- Calculate residuals: yards_beyond_opponent = yards - ridge.predict(opp_features)
- Use residuals as features in XGBoost alongside baseline features

Expected runtime: 30 minutes (most promising from research)
Target: MAE < 40.20 (v2 baseline)
"""

import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sqlalchemy import create_engine

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

        -- Opponent features for Ridge regression
        opp_def_epa_exp, opp_pass_def_rank,

        -- Metadata
        season, week, posteam, defteam
    FROM ml.model_features
    WHERE position = 'QB'
        AND season >= 2018
        AND season <= 2025
        AND passing_yards IS NOT NULL
        AND qb_epa_L3 IS NOT NULL
        AND opp_def_epa_exp IS NOT NULL
    ORDER BY season, week
    """

    df = pd.read_sql(query, engine)
    engine.dispose()

    print(f"✓ Loaded {len(df):,} rows")
    print(f"  - Training (2018-2024): {len(df[df['season'] <= 2024]):,}")
    print(f"  - Test (2025): {len(df[df['season'] == 2025]):,}")

    return df


def train_ridge_model(train_df):
    """
    Train Ridge regression to predict yards from opponent features.

    Returns:
        ridge: Fitted RidgeCV model
        alpha: Best alpha selected by CV
    """
    print("\n" + "=" * 60)
    print("STEP 1: TRAIN RIDGE REGRESSION (OPPONENT FEATURES)")
    print("=" * 60)

    opp_features = ["opp_def_epa_exp", "opp_pass_def_rank"]

    X_opp = train_df[opp_features].copy()
    y = train_df["passing_yards"].copy()

    # RidgeCV with automatic alpha selection
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_opp, y)

    # Predictions
    y_pred_ridge = ridge.predict(X_opp)
    mae_ridge = np.mean(np.abs(y - y_pred_ridge))

    print("\nRidge Regression Results:")
    print(f"  Best alpha: {ridge.alpha_:.2f}")
    print(f"  Train MAE: {mae_ridge:.2f} yards")
    print(f"  R²: {ridge.score(X_opp, y):.4f}")

    # Coefficient interpretation
    for feat, coef in zip(opp_features, ridge.coef_):
        print(f"  {feat}: {coef:.2f}")

    print("=" * 60 + "\n")

    return ridge


def calculate_residuals(df, ridge_model):
    """
    Calculate residuals: yards_beyond_opponent = yards - ridge.predict(opp_features)

    This represents how many yards the QB exceeded (or fell short of) the
    opponent-based expectation.
    """
    opp_features = ["opp_def_epa_exp", "opp_pass_def_rank"]

    X_opp = df[opp_features].copy()
    y_pred_ridge = ridge_model.predict(X_opp)

    df["yards_beyond_opponent"] = df["passing_yards"] - y_pred_ridge

    print("=" * 60)
    print("STEP 2: CALCULATE RESIDUALS (YARDS BEYOND OPPONENT EXPECTATION)")
    print("=" * 60)
    print("\nResiduals Statistics:")
    print(f"  Mean: {df['yards_beyond_opponent'].mean():.2f} yards")
    print(f"  Std: {df['yards_beyond_opponent'].std():.2f} yards")
    print(f"  Min: {df['yards_beyond_opponent'].min():.2f} yards")
    print(f"  Max: {df['yards_beyond_opponent'].max():.2f} yards")
    print("\nInterpretation:")
    print("  Positive residual: QB exceeded opponent-based expectation")
    print("  Negative residual: QB fell short of opponent-based expectation")
    print("=" * 60 + "\n")

    return df


def prepare_features(df):
    """Prepare feature matrix with residuals."""
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
        # Ridge residuals (1) - NEW
        "yards_beyond_opponent",
    ]

    X = df[feature_cols].copy()
    y = df["passing_yards"].copy()

    return X, y, feature_cols


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost with Ridge residuals and evaluate."""
    print("\n" + "=" * 60)
    print("STEP 3: TRAIN XGBOOST WITH RIDGE RESIDUALS")
    print("=" * 60)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(feature_names)} (16 baseline + 1 residual)")
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

    # Check residual importance
    residual_importance = importance_df[importance_df["feature"] == "yards_beyond_opponent"][
        "importance"
    ].values[0]
    residual_rank = importance_df[importance_df["feature"] == "yards_beyond_opponent"].index[0] + 1

    print("\nRidge Residual Feature:")
    print(f"  yards_beyond_opponent rank: {residual_rank}/17")
    print(f"  yards_beyond_opponent importance: {residual_importance:.4f}")

    print("=" * 60 + "\n")

    return model, mae, importance_df


def main():
    print("\n" + "=" * 60)
    print("XGBoost v3.2 Experiment 7: Ridge Residuals")
    print("=" * 60 + "\n")

    # Load data
    df = load_data()

    # Split
    train_df = df[df["season"] <= 2024].copy()
    test_df = df[df["season"] == 2025].copy()

    # Step 1: Train Ridge regression on training data
    ridge_model = train_ridge_model(train_df)

    # Step 2: Calculate residuals for both train and test
    train_df = calculate_residuals(train_df, ridge_model)
    test_df = calculate_residuals(test_df, ridge_model)

    # Step 3: Prepare features with residuals
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    # Step 4: Train XGBoost and evaluate
    model, mae, importance_df = train_and_evaluate(X_train, y_train, X_test, y_test, feature_names)

    # Save model if successful
    if mae < 40.20:
        model.save_model(
            "/Users/dro/rice/nfl-analytics/models/xgboost/xgb_v3_2_ridge_residuals.json"
        )
        importance_df.to_csv(
            "/Users/dro/rice/nfl-analytics/models/xgboost/xgb_v3_2_ridge_residuals_importance.csv",
            index=False,
        )
        print("✓ Model saved to models/xgboost/xgb_v3_2_ridge_residuals.json\n")

    return 0 if mae < 40.20 else 1


if __name__ == "__main__":
    sys.exit(main())
