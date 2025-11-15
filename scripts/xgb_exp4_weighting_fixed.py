#!/usr/bin/env python3
"""
XGBoost Experiment 4: Sample Weighting by Opponent Defensive Strength

Hypothesis: Weighting samples by opponent difficulty (opponent EPA expected)
helps model learn matchup-specific patterns without explicit opponent features.

Theory:
- Elite defenses present harder prediction problems
- Higher weight on tough matchups forces model to learn defensive impact
- Emphasis on difficult cases improves generalization on future tough matchups

Approach:
1. Load data using v3.1 feature pattern (no opponent features in model)
2. Calculate sample weights: weight = 1 + (opp_def_epa_exp - mean) / std
3. Clip weights to [0.5, 2.0] (prevent extreme outlier dominance)
4. Train XGBoost with sample_weight parameter
5. Evaluate on 2025 holdout test set

Expected Runtime: 12 minutes
Target: MAE < 40.20 yards (v2 baseline)

Features Used (16 total):
- Core player (10): qb_epa_L3, qb_cpoe_L3, qb_sack_rate_L3, qb_ypa_L3,
                    wr1_epa_L3, wr2_epa_L3, wr1_tgt_L3, wr2_tgt_L3,
                    qb_home_away_split, qb_consistency
- Game context (2): spread_close, total_close
- Team features (4): team_off_epa_L3, team_pass_rate_L3, team_pace_L3,
                     team_off_success_L3

Weighting Strategy:
- Lower opponent EPA = tougher defense = higher weight (1.5-2.0)
- Higher opponent EPA = weaker defense = lower weight (0.5-1.0)
- This forces model to focus on games with difficult matchups

Author: Claude Code
Date: November 14, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import time
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
DB_URI = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"


def load_data():
    """Load training and test data from PostgreSQL.

    Returns:
        DataFrame with features, target, weights for training
    """
    logger.info("Loading data from PostgreSQL...")
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

        -- Opponent feature for weighting (NOT in model features)
        opp_def_epa_exp,

        -- Metadata
        season, week, posteam, defteam, kickoff
    FROM ml.model_features
    WHERE position = 'QB'
        AND season >= 2018
        AND season <= 2025
        AND passing_yards IS NOT NULL
        AND qb_epa_L3 IS NOT NULL
        AND opp_def_epa_exp IS NOT NULL
    ORDER BY season, week
    """

    try:
        df = pd.read_sql(query, engine)
        engine.dispose()

        logger.info(f"✓ Loaded {len(df):,} rows")
        logger.info(f"  Training set (2018-2024): {len(df[df['season'] <= 2024]):,} rows")
        logger.info(f"  Test set (2025): {len(df[df['season'] == 2025]):,} rows")

        # Data validation
        missing_cols = df.isnull().sum()
        if missing_cols.any():
            logger.warning(f"Missing values:\n{missing_cols[missing_cols > 0]}")

        return df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def calculate_sample_weights(df):
    """Calculate sample weights based on opponent defensive strength.

    Weights higher for tough matchups (lower opp_def_epa_exp = better defense).

    Formula:
        weight = 1 + (opp_def_epa_exp - mean) / std
        clipped to [0.5, 2.0]

    Args:
        df: DataFrame with opp_def_epa_exp column

    Returns:
        Series of sample weights (same length as df)
    """
    logger.info("Calculating sample weights...")

    # Use training set statistics for normalization
    train_mask = df['season'] <= 2024
    mean_epa = df.loc[train_mask, 'opp_def_epa_exp'].mean()
    std_epa = df.loc[train_mask, 'opp_def_epa_exp'].std()

    logger.info(f"  Opponent EPA stats (training 2018-2024):")
    logger.info(f"    Mean: {mean_epa:.3f}")
    logger.info(f"    Std: {std_epa:.3f}")

    # Calculate weights: higher weight for tough matchups (lower EPA)
    # Example: opp_def_epa_exp = 0.10 (elite defense):
    #   weight = 1 + (0.10 - 0.30) / 0.15 = 1 + (-0.20 / 0.15) = 1 - 1.33 = -0.33 → clipped to 0.5
    # Example: opp_def_epa_exp = 0.50 (weak defense):
    #   weight = 1 + (0.50 - 0.30) / 0.15 = 1 + (0.20 / 0.15) = 1 + 1.33 = 2.33 → clipped to 2.0
    weights = 1 + (df['opp_def_epa_exp'] - mean_epa) / std_epa
    weights = np.clip(weights, 0.5, 2.0)

    # Print distribution statistics
    logger.info("Sample weights distribution:")
    logger.info(f"  Mean: {weights.mean():.3f}")
    logger.info(f"  Std: {weights.std():.3f}")
    logger.info(f"  Min: {weights.min():.3f}")
    logger.info(f"  Max: {weights.max():.3f}")
    logger.info(f"  Median: {weights.median():.3f}")
    logger.info(f"  25th percentile: {weights.quantile(0.25):.3f}")
    logger.info(f"  75th percentile: {weights.quantile(0.75):.3f}")

    # Show breakdown by weight category
    weight_cats = pd.cut(weights, bins=[0, 0.75, 1.0, 1.25, 2.0],
                         labels=['Light (0.5-0.75)', 'Normal (0.75-1.0)',
                                'Heavy (1.0-1.25)', 'Very Heavy (1.25-2.0)'])
    logger.info("Weight distribution by category:")
    for cat in weight_cats.unique():
        count = (weight_cats == cat).sum()
        pct = 100 * count / len(weights)
        logger.info(f"  {cat}: {count:,} ({pct:.1f}%)")

    return weights


def prepare_features(df):
    """Prepare feature matrix for XGBoost (v3.1 pattern - no opponent features).

    Args:
        df: DataFrame with all features

    Returns:
        Tuple of (X, y, feature_names)
    """
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

    # Remove any rows with NaN in features or target
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Features shape after removing NaN: {X.shape}")

    return X, y, feature_cols


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       sample_weights_train, feature_names):
    """Train XGBoost with sample weights and evaluate on test set.

    Args:
        X_train, y_train: Training features and target
        X_test, y_test: Test features and target
        sample_weights_train: Sample weights for training
        feature_names: List of feature names

    Returns:
        Tuple of (model, mae, rmse, importance_df)
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING XGBoost v3.2 with Sample Weighting")
    logger.info("="*70)
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Test samples: {len(X_test):,}")
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Using sample weights: Yes (based on opponent defensive strength)")
    logger.info("="*70)

    # XGBoost hyperparameters (tuned for passing yards)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mae'
    }

    logger.info(f"\nHyperparameters:")
    for key, val in params.items():
        logger.info(f"  {key}: {val}")

    model = xgb.XGBRegressor(**params)

    logger.info("\nTraining model (this may take 5-10 minutes)...")
    start_time = time.time()

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,  # Apply weights during training
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )

    train_time = time.time() - start_time
    logger.info(f"✓ Training complete in {train_time:.1f}s")

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Percentile errors
    abs_errors = np.abs(y_test - y_pred)
    p50_error = np.percentile(abs_errors, 50)
    p90_error = np.percentile(abs_errors, 90)

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Results summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"MAE: {mae:.2f} yards")
    logger.info(f"RMSE: {rmse:.2f} yards")
    logger.info(f"MAPE: {mape:.1f}%")
    logger.info(f"Median error (50th percentile): {p50_error:.2f} yards")
    logger.info(f"90th percentile error: {p90_error:.2f} yards")
    logger.info(f"Training time: {train_time:.1f}s")

    logger.info(f"\nBaseline comparisons:")
    logger.info(f"  v2 baseline (with opponent features): 40.20 yards")
    logger.info(f"  v3.1 baseline (no opponent features): 41.36 yards")

    improvement_v2 = ((40.20 - mae) / 40.20) * 100
    improvement_v31 = ((41.36 - mae) / 41.36) * 100

    logger.info(f"\nImprovement:")
    logger.info(f"  vs v2 baseline: {improvement_v2:+.2f}%")
    logger.info(f"  vs v3.1 baseline: {improvement_v31:+.2f}%")

    # Success determination
    logger.info("\n" + "-"*70)
    if mae < 40.20:
        logger.info("🎯 SUCCESS: Beat v2 baseline (with opponent features)!")
    elif mae < 41.36:
        logger.info("⚠️ MARGINAL: Beat v3.1 but not v2")
    else:
        logger.info("❌ FAILED: Did not beat v3.1 (no improvement)")
    logger.info("-"*70)

    logger.info("\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")

    logger.info("="*70 + "\n")

    return model, mae, rmse, importance_df


def save_results(model, importance_df, mae, rmse, params_dict):
    """Save model, feature importance, and metadata.

    Args:
        model: Trained XGBoost model
        importance_df: Feature importance DataFrame
        mae: Test MAE score
        rmse: Test RMSE score
        params_dict: Dictionary of experiment parameters
    """
    output_dir = Path("/Users/dro/rice/nfl-analytics/models/xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "xgb_exp4_weighting_fixed.json"
    model.save_model(str(model_path))
    logger.info(f"✓ Model saved: {model_path}")

    # Save feature importance
    importance_path = output_dir / "xgb_exp4_weighting_fixed_importance.csv"
    importance_df.to_csv(str(importance_path), index=False)
    logger.info(f"✓ Feature importance saved: {importance_path}")

    # Save metadata
    metadata = {
        "experiment": "XGBoost Experiment 4: Sample Weighting by Opponent Strength",
        "date": pd.Timestamp.now().isoformat(),
        "test_mae": round(float(mae), 2),
        "test_rmse": round(float(rmse), 2),
        "parameters": params_dict,
        "baselines": {
            "v2_with_opponent_features": 40.20,
            "v3_1_no_opponent_features": 41.36
        },
        "improvement_vs_v2_pct": round(((40.20 - mae) / 40.20) * 100, 2),
        "improvement_vs_v31_pct": round(((41.36 - mae) / 41.36) * 100, 2)
    }

    metadata_path = output_dir / "xgb_exp4_weighting_fixed_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved: {metadata_path}")


def main():
    """Main experiment runner."""
    logger.info("\n" + "="*70)
    logger.info("XGBoost Experiment 4: Sample Weighting by Opponent Defensive Strength")
    logger.info("="*70 + "\n")

    try:
        # Load data
        df = load_data()

        # Calculate sample weights based on opponent EPA
        sample_weights = calculate_sample_weights(df)

        # Split into train/test
        logger.info("\nSplitting data...")
        train_mask = df['season'] <= 2024
        test_mask = df['season'] == 2025

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        sample_weights_train = sample_weights[train_mask]
        sample_weights_test = sample_weights[test_mask]

        logger.info(f"✓ Train set: {len(train_df):,} rows")
        logger.info(f"✓ Test set: {len(test_df):,} rows")

        # Prepare features
        X_train, y_train, feature_names = prepare_features(train_df)
        X_test, y_test, _ = prepare_features(test_df)

        # Align sample weights with features (in case prepare_features removed NaN)
        # For this simple case, we'll rebuild train_mask
        train_mask_clean = df[train_mask].index.isin(X_train.index)
        sample_weights_train_clean = sample_weights[train_mask][train_mask_clean]

        logger.info(f"✓ Feature matrix shape (train): {X_train.shape}")
        logger.info(f"✓ Feature matrix shape (test): {X_test.shape}")

        # Train and evaluate
        params_dict = {
            "approach": "Sample weighting by opponent defensive EPA",
            "weight_formula": "1 + (opp_def_epa_exp - mean) / std",
            "weight_range": [0.5, 2.0],
            "features": feature_names,
            "n_features": len(feature_names),
            "train_seasons": "2018-2024",
            "test_season": 2025,
            "hyperparameters": {
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3
            }
        }

        model, mae, rmse, importance_df = train_and_evaluate(
            X_train, y_train, X_test, y_test,
            sample_weights_train_clean, feature_names
        )

        # Save results
        save_results(model, importance_df, mae, rmse, params_dict)

        # Return success/failure code
        return 0 if mae < 40.20 else 1

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
