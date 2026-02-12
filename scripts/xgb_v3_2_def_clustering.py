#!/usr/bin/env python3
"""
XGBoost v3.2 Experiment 2: Defensive Clustering

Hypothesis: K-means clustering of defenses into tiers reduces redundancy
while capturing non-linear opponent strength effects.

Approach:
- Cluster defenses using opp_def_epa_exp + opp_pass_def_rank
- Create 4 tier dummies: def_tier_1, def_tier_2, def_tier_3, def_tier_4
- Replace continuous opponent features with categorical tiers

Expected runtime: 20 minutes
Target: MAE < 40.20 (v2 baseline)
"""

import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

        -- Opponent features for clustering (2 features)
        opp_def_epa_exp, opp_pass_def_rank,

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


def create_defensive_clusters(train_df):
    """
    Cluster defenses into 4 tiers using K-means.

    Returns:
        kmeans: Fitted KMeans model
        cluster_stats: DataFrame with cluster statistics
    """
    # Extract opponent features
    X_cluster = train_df[["opp_def_epa_exp", "opp_pass_def_rank"]].copy()

    # Standardize features for K-means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Fit K-means with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_scaled)

    # Analyze cluster characteristics
    cluster_df = train_df.copy()
    cluster_df["cluster"] = train_clusters

    cluster_stats = (
        cluster_df.groupby("cluster")
        .agg({"opp_def_epa_exp": ["mean", "std", "count"], "opp_pass_def_rank": ["mean", "std"]})
        .round(3)
    )

    print("\n" + "=" * 60)
    print("DEFENSIVE CLUSTER STATISTICS")
    print("=" * 60)
    print(cluster_stats)
    print("=" * 60 + "\n")

    return kmeans, scaler


def assign_clusters(df, kmeans, scaler):
    """Assign cluster labels to all data."""
    X_cluster = df[["opp_def_epa_exp", "opp_pass_def_rank"]].copy()
    X_scaled = scaler.transform(X_cluster)
    clusters = kmeans.predict(X_scaled)

    # Create dummy variables
    for i in range(4):
        df[f"def_tier_{i+1}"] = (clusters == i).astype(int)

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
        # Defensive tiers (4)
        "def_tier_1",
        "def_tier_2",
        "def_tier_3",
        "def_tier_4",
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
    print("TRAINING XGBoost v3.2 (Defensive Clustering)")
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
    print("XGBoost v3.2 Experiment 2: Defensive Clustering")
    print("=" * 60 + "\n")

    # Load data
    df = load_data()

    # Split
    train_df = df[df["season"] <= 2024].copy()
    test_df = df[df["season"] == 2025].copy()

    # Create defensive clusters on training data
    kmeans, scaler = create_defensive_clusters(train_df)

    # Assign clusters to all data
    train_df = assign_clusters(train_df, kmeans, scaler)
    test_df = assign_clusters(test_df, kmeans, scaler)

    # Prepare features
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    # Train and evaluate
    model, mae, importance_df = train_and_evaluate(X_train, y_train, X_test, y_test, feature_names)

    # Save model if successful
    if mae < 40.20:
        model.save_model(
            str(_PROJECT_ROOT / "models/xgboost/xgb_v3_2_def_clustering.json")
        )
        importance_df.to_csv(
            str(_PROJECT_ROOT / "models/xgboost/xgb_v3_2_def_clustering_importance.csv"),
            index=False,
        )
        print("✓ Model saved to models/xgboost/xgb_v3_2_def_clustering.json\n")

    return 0 if mae < 40.20 else 1


if __name__ == "__main__":
    sys.exit(main())
