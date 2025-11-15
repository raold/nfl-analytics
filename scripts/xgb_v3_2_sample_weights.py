#!/usr/bin/env python3
"""
XGBoost v3.2 Experiment 4: Sample Weighting by Matchup Difficulty

Hypothesis: Weighting samples by opponent difficulty helps model learn
matchup-specific patterns without direct opponent features.

Approach:
- Calculate sample weights based on opponent defensive strength
- weights = 1 + (opp_def_epa_exp - mean) / std
- Clip weights to [0.5, 2.0] to avoid outlier dominance
- Use v3.1 features (no opponent features in model)

Expected runtime: 15 minutes
Target: MAE < 40.20 (v2 baseline)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import time
import sys

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

        -- Opponent feature for weighting (not in model)
        opp_def_epa_exp,

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

def calculate_sample_weights(df):
    """
    Calculate sample weights based on opponent defensive strength.

    Higher weights for tougher matchups (lower opp_def_epa_exp = better defense).
    """
    # Standardize opponent EPA (training set only for mean/std)
    train_mask = df['season'] <= 2024
    mean_epa = df.loc[train_mask, 'opp_def_epa_exp'].mean()
    std_epa = df.loc[train_mask, 'opp_def_epa_exp'].std()

    # Calculate weights: 1 + standardized opponent strength
    # Better defense (lower EPA) → higher weight
    weights = 1 + (df['opp_def_epa_exp'] - mean_epa) / std_epa

    # Clip to [0.5, 2.0] to avoid outlier dominance
    weights = np.clip(weights, 0.5, 2.0)

    print("\n" + "="*60)
    print("SAMPLE WEIGHTS DISTRIBUTION")
    print("="*60)
    print(f"Mean: {weights.mean():.3f}")
    print(f"Std: {weights.std():.3f}")
    print(f"Min: {weights.min():.3f}")
    print(f"Max: {weights.max():.3f}")
    print(f"Median: {weights.median():.3f}")
    print("="*60 + "\n")

    return weights

def prepare_features(df):
    """Prepare feature matrix (same as v3.1 - no opponent features)."""
    feature_cols = [
        # Core player features (10)
        'qb_epa_L3', 'qb_cpoe_L3', 'qb_sack_rate_L3', 'qb_ypa_L3',
        'wr1_epa_L3', 'wr2_epa_L3', 'wr1_tgt_L3', 'wr2_tgt_L3',
        'qb_home_away_split', 'qb_consistency',

        # Game context (2)
        'spread_close', 'total_close',

        # Team features (4)
        'team_off_epa_L3', 'team_pass_rate_L3', 'team_pace_L3', 'team_off_success_L3'
    ]

    X = df[feature_cols].copy()
    y = df['passing_yards'].copy()

    return X, y, feature_cols

def train_and_evaluate(X_train, y_train, X_test, y_test, sample_weights_train, feature_names):
    """Train XGBoost with sample weights and evaluate."""
    print("\n" + "="*60)
    print("TRAINING XGBoost v3.2 (Sample Weighting)")
    print("="*60)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Using sample weights: Yes (based on opponent strength)")
    print("="*60 + "\n")

    # Train model with same hyperparameters as v3.1
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)

    start_time = time.time()
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,  # Apply weights during training
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    train_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"MAE: {mae:.2f} yards")
    print(f"RMSE: {rmse:.2f} yards")
    print(f"Training time: {train_time:.1f}s")
    print(f"v2 baseline: 40.20 yards")
    print(f"v3.1 baseline: 41.36 yards (no opponent features)")

    improvement_pct = ((40.20 - mae) / 40.20) * 100
    print(f"Improvement vs v2: {improvement_pct:+.1f}%")

    if mae < 40.20:
        print("\n🎯 SUCCESS: Beat v2 baseline!")
    elif mae < 41.36:
        print("\n⚠️ MARGINAL: Beat v3.1 but not v2")
    else:
        print("\n❌ FAILED: Did not beat v3.1")

    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    print("="*60 + "\n")

    return model, mae, importance_df

def main():
    print("\n" + "="*60)
    print("XGBoost v3.2 Experiment 4: Sample Weighting")
    print("="*60 + "\n")

    # Load data
    df = load_data()

    # Calculate sample weights
    sample_weights = calculate_sample_weights(df)

    # Split
    train_mask = df['season'] <= 2024
    test_mask = df['season'] == 2025

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    sample_weights_train = sample_weights[train_mask]
    sample_weights_test = sample_weights[test_mask]

    # Prepare features
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    # Train and evaluate
    model, mae, importance_df = train_and_evaluate(
        X_train, y_train, X_test, y_test, sample_weights_train, feature_names
    )

    # Save model if successful
    if mae < 40.20:
        model.save_model('/Users/dro/rice/nfl-analytics/models/xgboost/xgb_v3_2_sample_weights.json')
        importance_df.to_csv('/Users/dro/rice/nfl-analytics/models/xgboost/xgb_v3_2_sample_weights_importance.csv', index=False)
        print("✓ Model saved to models/xgboost/xgb_v3_2_sample_weights.json\n")

    return 0 if mae < 40.20 else 1

if __name__ == "__main__":
    sys.exit(main())
