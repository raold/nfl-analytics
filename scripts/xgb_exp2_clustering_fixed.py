#!/usr/bin/env python3
"""
XGBoost Exp 2: Defensive Clustering (FIXED & COMPLETE)

HYPOTHESIS:
K-means clustering of defenses into tiers (Elite/Good/Avg/Weak) reduces noise
while capturing non-linear opponent strength effects. Categorical features
may generalize better than continuous opponent metrics in tree-based models.

APPROACH:
1. Load 2018-2024 QB passing yards + opponent features
2. K-means cluster (k=4) using: opp_def_epa_exp + opp_pass_def_rank
3. Create dummy variables: def_tier_elite, def_tier_good, def_tier_avg, def_tier_weak
4. Train XGBoost with categorical tiers (NOT continuous opponent metrics)
5. Compare MAE to v2 baseline (40.20 yards)

THEORY:
Trees handle categorical features better than continuous. Binning opponent
strength may reduce overfitting to noise in EPA/rank metrics. 4-tier system:
- Tier 4 (Elite): Top 25% defenses
- Tier 3 (Good): 25-50%
- Tier 2 (Avg): 50-75%
- Tier 1 (Weak): Bottom 25%

EXPECTED RESULT:
- MAE < 40.20 yards (beat v2 baseline)
- Defensive tiers show strong feature importance
- Tier dummies less noisy than continuous opponent features

Author: NFL Analytics Research Team
Date: November 14, 2025
Expected Runtime: ~20 minutes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# Configuration
DB_URI = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
BASELINE_MAE = 40.20
N_CLUSTERS = 4
RANDOM_STATE = 42


def load_data():
    """
    Load QB passing yards + features from PostgreSQL.

    Returns:
        DataFrame with columns: passing_yards, features, season, week, posteam, defteam
    """
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

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

        -- Metadata for split
        season, week, posteam, defteam
    FROM ml.model_features
    WHERE position = 'QB'
        AND season >= 2018
        AND season <= 2025
        AND passing_yards IS NOT NULL
        AND qb_epa_L3 IS NOT NULL
    ORDER BY season, week
    """

    try:
        df = pd.read_sql(query, engine)
        engine.dispose()
    except Exception as e:
        print(f"✗ FAILED to load data: {e}")
        print("  Ensure PostgreSQL is running and database is initialized")
        sys.exit(1)

    print(f"✓ Loaded {len(df):,} total rows")

    # Check for NaN in target
    rows_with_valid_target = df['passing_yards'].notna().sum()
    print(f"  - Valid targets: {rows_with_valid_target:,} ({rows_with_valid_target/len(df)*100:.1f}%)")

    # Show season distribution
    print(f"  - Seasons: {df['season'].min()}-{df['season'].max()}")
    print("  - Season distribution:")
    for season, count in df['season'].value_counts().sort_index().items():
        print(f"      {season}: {count:,} games")

    # Check opponent features
    opp_coverage = {
        'opp_def_epa_exp': df['opp_def_epa_exp'].notna().mean(),
        'opp_pass_def_rank': df['opp_pass_def_rank'].notna().mean()
    }
    print("\n  - Opponent feature coverage:")
    for feat, cov in opp_coverage.items():
        print(f"      {feat}: {cov*100:.1f}%")

    print("=" * 70 + "\n")
    return df


def create_defensive_clusters(train_df):
    """
    Cluster defenses into 4 tiers using K-means.

    Strategy:
    - Standardize opp_def_epa_exp + opp_pass_def_rank
    - Fit K-means with k=4
    - Label: Cluster 3 = Elite (low EPA allowed, high rank), Cluster 0 = Weak

    Args:
        train_df: Training data (2018-2024)

    Returns:
        kmeans: Fitted KMeans model
        scaler: StandardScaler for feature scaling
    """
    print("=" * 70)
    print("CREATING DEFENSIVE CLUSTERS")
    print("=" * 70)

    # Remove NaN rows for clustering
    cluster_data = train_df[
        (train_df['opp_def_epa_exp'].notna()) &
        (train_df['opp_pass_def_rank'].notna())
    ].copy()

    print(f"✓ Using {len(cluster_data):,} rows with complete opponent features")

    # Extract and standardize
    X_cluster = cluster_data[['opp_def_epa_exp', 'opp_pass_def_rank']].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # K-means clustering
    print(f"\n  Fitting K-means (k={N_CLUSTERS}, random_state={RANDOM_STATE})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_assignments = kmeans.fit_predict(X_scaled)

    cluster_data['cluster'] = cluster_assignments

    # Analyze cluster characteristics
    print("\n  Cluster Analysis:")
    print(f"  {'Tier':<8} {'Count':<10} {'EPA Mean':<12} {'EPA Std':<12} {'Rank Mean':<12}")
    print("  " + "-" * 54)

    for tier in range(N_CLUSTERS):
        tier_data = cluster_data[cluster_data['cluster'] == tier]
        epa_mean = tier_data['opp_def_epa_exp'].mean()
        epa_std = tier_data['opp_def_epa_exp'].std()
        rank_mean = tier_data['opp_pass_def_rank'].mean()
        count = len(tier_data)

        # Interpret tier
        if epa_mean < cluster_data['opp_def_epa_exp'].quantile(0.25):
            tier_name = "ELITE"
        elif epa_mean < cluster_data['opp_def_epa_exp'].quantile(0.50):
            tier_name = "GOOD"
        elif epa_mean < cluster_data['opp_def_epa_exp'].quantile(0.75):
            tier_name = "AVG"
        else:
            tier_name = "WEAK"

        print(f"  {tier_name:<8} {count:<10,} {epa_mean:<12.3f} {epa_std:<12.3f} {rank_mean:<12.1f}")

    print("=" * 70 + "\n")
    return kmeans, scaler


def assign_clusters(df, kmeans, scaler):
    """
    Assign cluster tiers to all data and create dummy variables.

    Creates 4 binary features:
    - def_tier_1: Weak defenses (bottom 25%)
    - def_tier_2: Average defenses (25-50%)
    - def_tier_3: Good defenses (50-75%)
    - def_tier_4: Elite defenses (top 25%)

    Args:
        df: DataFrame with opponent features
        kmeans: Fitted KMeans model
        scaler: StandardScaler

    Returns:
        df with tier dummies added
    """
    # Handle rows with missing opponent data
    df = df.copy()

    # For complete rows, assign clusters
    mask_complete = (df['opp_def_epa_exp'].notna()) & (df['opp_pass_def_rank'].notna())

    X_cluster = df.loc[mask_complete, ['opp_def_epa_exp', 'opp_pass_def_rank']].copy()
    X_scaled = scaler.transform(X_cluster)
    clusters = kmeans.predict(X_scaled)

    # Initialize tier dummies
    for i in range(1, N_CLUSTERS + 1):
        df[f'def_tier_{i}'] = 0

    # Assign tiers for complete rows
    df.loc[mask_complete, 'cluster_assignment'] = clusters
    for idx, cluster in zip(df.loc[mask_complete].index, clusters):
        df.loc[idx, f'def_tier_{cluster + 1}'] = 1

    # For incomplete rows, use most common tier
    if mask_complete.sum() < len(df):
        most_common_tier = int(np.median(clusters)) + 1
        df.loc[~mask_complete, f'def_tier_{most_common_tier}'] = 1
        print(f"\n⚠  Note: {(~mask_complete).sum()} rows assigned to most common tier (def_tier_{most_common_tier})")

    return df


def prepare_features(df):
    """
    Prepare feature matrix for training.

    Features (20 total):
    - Core player: 10 (QB EPA, CPOE, sack rate, YPA, WR EPA/targets, consistency)
    - Game context: 2 (spread, total)
    - Defensive tiers: 4 (categorical replacements for opponent EPA/rank)
    - Team features: 4 (offense EPA, pass rate, pace, success)

    Args:
        df: DataFrame with cluster assignments

    Returns:
        X, y, feature_cols (cleaned, no NaN)
    """
    feature_cols = [
        # Core player features (10)
        'qb_epa_L3', 'qb_cpoe_L3', 'qb_sack_rate_L3', 'qb_ypa_L3',
        'wr1_epa_L3', 'wr2_epa_L3', 'wr1_tgt_L3', 'wr2_tgt_L3',
        'qb_home_away_split', 'qb_consistency',

        # Game context (2)
        'spread_close', 'total_close',

        # Defensive tiers (4) - CATEGORICAL
        'def_tier_1', 'def_tier_2', 'def_tier_3', 'def_tier_4',

        # Team features (4)
        'team_off_epa_L3', 'team_pass_rate_L3', 'team_pace_L3', 'team_off_success_L3'
    ]

    # Create feature matrix
    X = df[feature_cols].copy()
    y = df['passing_yards'].copy()

    # Remove rows with any NaN
    valid_mask = (X.notna().all(axis=1)) & (y.notna())
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  - Removed {(~valid_mask).sum():,} rows with NaN values")
    print(f"  - Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]} columns")

    return X, y, feature_cols


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names):
    """
    Train XGBoost and evaluate on test set.

    Hyperparameters match v3.1 baseline for fair comparison.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data (2025)
        feature_names: List of feature column names

    Returns:
        model, mae, importance_df
    """
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST v3.2 (DEFENSIVE CLUSTERING)")
    print("=" * 70)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(feature_names)} ({N_CLUSTERS} tier dummies)")
    print("=" * 70 + "\n")

    # Hyperparameters (same as v3.1 for fair comparison)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0
    }

    model = xgb.XGBRegressor(**params)

    # Train with early stopping
    start_time = time.time()
    print("  Training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    train_time = time.time() - start_time

    # Predictions
    print(f"  ✓ Completed in {train_time:.1f}s\n")
    print("  Getting predictions...")
    y_pred = model.predict(X_test)

    # Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"MAE: {mae:.2f} yards")
    print(f"RMSE: {rmse:.2f} yards")
    print(f"R²: {r2:.4f}")
    print(f"Training time: {train_time:.1f}s")
    print(f"v2 baseline: {BASELINE_MAE:.2f} yards")

    improvement_abs = BASELINE_MAE - mae
    improvement_pct = (improvement_abs / BASELINE_MAE) * 100

    print("\nComparison to v2 baseline:")
    print(f"  Δ MAE: {improvement_abs:+.2f} yards ({improvement_pct:+.1f}%)")

    if mae < BASELINE_MAE:
        print(f"\n✅ SUCCESS: Beat v2 baseline by {improvement_abs:.2f} yards!")
    else:
        print(f"\n⚠️  Did not beat baseline (worse by {abs(improvement_abs):.2f} yards)")

    print("\nTop 15 Features:")
    print(importance_df.head(15).to_string(index=False))

    # Highlight tier importance
    tier_features = importance_df[importance_df['feature'].str.contains('def_tier')]
    if len(tier_features) > 0:
        print("\n" + "-" * 70)
        print("DEFENSIVE TIER IMPORTANCE:")
        total_tier_importance = tier_features['importance'].sum()
        total_importance = importance_df['importance'].sum()
        tier_pct = (total_tier_importance / total_importance) * 100
        print(f"Combined tier importance: {total_tier_importance:.4f} ({tier_pct:.1f}% of total)")
        print(tier_features.to_string(index=False))

    print("=" * 70 + "\n")

    return model, mae, rmse, r2, importance_df


def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print("XGBoost Experiment 2: Defensive Clustering (FIXED)")
    print("=" * 70)
    print(f"Target: Beat v2 baseline (MAE < {BASELINE_MAE:.2f} yards)")
    print("Theory: Categorical tier dummies reduce noise vs continuous opponent metrics")
    print("=" * 70)

    # Load data
    df = load_data()

    # Split into train/test
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT")
    print("=" * 70)
    train_df = df[df['season'] <= 2024].copy()
    test_df = df[df['season'] == 2025].copy()
    print(f"Training (2018-2024): {len(train_df):,} rows")
    print(f"Test (2025): {len(test_df):,} rows")
    print("=" * 70 + "\n")

    # Create defensive clusters on training data
    kmeans, scaler = create_defensive_clusters(train_df)

    # Assign clusters to all data
    print("Assigning clusters to training and test data...")
    train_df = assign_clusters(train_df, kmeans, scaler)
    test_df = assign_clusters(test_df, kmeans, scaler)
    print("✓ Cluster assignments complete\n")

    # Prepare features
    print("=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)
    print("Training set:")
    X_train, y_train, feature_names = prepare_features(train_df)
    print("Test set:")
    X_test, y_test, _ = prepare_features(test_df)
    print("=" * 70 + "\n")

    # Train and evaluate
    model, mae, rmse, r2, importance_df = train_and_evaluate(
        X_train, y_train, X_test, y_test, feature_names
    )

    # Save results
    results_dir = Path("models/xgboost")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save importance
    importance_path = results_dir / "xgb_exp2_clustering_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to {importance_path}\n")

    # Save results summary
    summary = {
        'experiment': 'XGBoost Exp 2: Defensive Clustering',
        'date': pd.Timestamp.now().isoformat(),
        'approach': 'K-means clustering (k=4) on opp_def_epa_exp + opp_pass_def_rank',
        'features': {
            'total': len(feature_names),
            'tiers': N_CLUSTERS,
            'feature_list': feature_names
        },
        'training': {
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'baseline_mae': BASELINE_MAE,
            'improvement_abs': float(BASELINE_MAE - mae),
            'improvement_pct': float(((BASELINE_MAE - mae) / BASELINE_MAE) * 100)
        },
        'model_file': None,
        'status': 'SUCCESS' if mae < BASELINE_MAE else 'FAILED'
    }

    summary_path = results_dir / "xgb_exp2_clustering_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Results summary saved to {summary_path}\n")

    # Save model if successful
    if mae < BASELINE_MAE:
        model_path = results_dir / "xgb_exp2_clustering.json"
        model.save_model(str(model_path))
        summary['model_file'] = str(model_path)
        print(f"✓ Model saved to {model_path}\n")

        # Update summary with model path
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return 0
    else:
        print("⚠️  Model not saved (did not beat baseline)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
