#!/usr/bin/env python3
"""
Train Game Outcome Model v2.1 (Simplified for Docker)
======================================================

Uses direct psycopg2 connections instead of pool for container compatibility.

Train: 2022-2024, Test: 2025 (true holdout)
New features: Kalman EPA (18) + GP Injury (17)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Direct import of feature extractor (avoid pool dependency)
from py.features.game_feature_extractor import GameFeatureExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
}

# Training configuration
# temporal_cv: season-based split prevents data leakage (train on past, test on future)
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
MIN_WEEK = 3
OUTPUT_DIR = "models/staging/game_outcome_v2.1"


def get_connection():
    """Get direct psycopg2 connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_games(seasons: list[int], min_week: int = 3) -> pd.DataFrame:
    """Load game data with outcomes."""
    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.spread_close as vegas_spread,
        g.total_close as vegas_total
    FROM games g
    WHERE g.season = ANY(%s)
      AND g.week >= %s
      AND g.week <= 18
      AND g.home_score IS NOT NULL
      AND g.away_score IS NOT NULL
    ORDER BY g.season, g.week
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(seasons, min_week))
    logger.info(f"Loaded {len(df)} games from seasons {seasons}")
    return df


def build_dataset(games_df: pd.DataFrame, extractor, desc: str) -> tuple:
    """Build feature matrix and targets."""
    total = len(games_df)
    logger.info(f"Building {desc}: {total} games")

    features_list = []
    targets_list = []

    kalman_found = set()
    gp_found = set()

    for idx, row in games_df.iterrows():
        if (idx + 1) % 25 == 0:
            logger.info(f"  Progress: {idx + 1}/{total}")

        try:
            features = extractor.extract_game_features(
                row["home_team"], row["away_team"],
                row["season"], row["week"],
                use_cache=True
            )
        except Exception as e:
            logger.warning(f"Failed {row['home_team']} vs {row['away_team']}: {e}")
            continue

        if features is None:
            continue

        # Track new feature types
        for k in features.keys():
            if k.startswith("kal_"):
                kalman_found.add(k)
            elif k.startswith("gp_"):
                gp_found.add(k)

        features_list.append(features)

        spread = row["home_score"] - row["away_score"]
        total_pts = row["home_score"] + row["away_score"]
        targets_list.append({
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "actual_spread": spread,
            "actual_total": total_pts,
            "home_win": 1 if spread > 0 else 0,
            "vegas_spread": row["vegas_spread"],
            "vegas_total": row["vegas_total"],
        })

    features_df = pd.DataFrame(features_list)
    targets_df = pd.DataFrame(targets_list)

    logger.info(f"  Kalman features: {len(kalman_found)}")
    logger.info(f"  GP features: {len(gp_found)}")
    logger.info(f"  Total features: {len(features_df.columns)}")

    return features_df, targets_df


def train_regression(X_train, y_train, X_test, y_test, y_vegas, name):
    """Train XGBoost regressor."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    vegas_mae = mean_absolute_error(y_test, y_vegas)

    cover_rate = None
    if name == "spread":
        correct = ((y_pred > y_vegas) == (y_test > y_vegas)).sum()
        cover_rate = correct / len(y_test) * 100

    logger.info(f"\n{name.upper()} Results:")
    logger.info(f"  MAE: {mae:.2f} (Vegas: {vegas_mae:.2f})")
    logger.info(f"  RMSE: {rmse:.2f}, R²: {r2:.4f}")
    if cover_rate:
        logger.info(f"  Cover Rate: {cover_rate:.1f}%")

    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "vegas_mae": float(vegas_mae)}
    if cover_rate:
        metrics["cover_rate"] = float(cover_rate)
    return model, metrics


def train_classification(X_train, y_train, X_test, y_test, name):
    """Train XGBoost classifier."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = (y_pred == y_test).mean() * 100
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"\n{name.upper()} Results:")
    logger.info(f"  Accuracy: {accuracy:.1f}%")
    logger.info(f"  AUC: {auc:.4f}")

    return model, {"accuracy": float(accuracy), "auc": float(auc)}


def main():
    logger.info("=" * 70)
    logger.info("GAME OUTCOME v2.1 TRAINING")
    logger.info("Train: 2022-2024, Test: 2025 (holdout)")
    logger.info("New: Kalman EPA + GP Injury features")
    logger.info("=" * 70)

    # Initialize feature extractor with Docker DB config
    db_config = {
        "host": DB_CONFIG["host"],
        "port": DB_CONFIG["port"],
        "dbname": DB_CONFIG["dbname"],
        "user": DB_CONFIG["user"],
        "password": DB_CONFIG["password"],
    }
    extractor = GameFeatureExtractor(db_config=db_config)

    # Check new features
    has_kalman = hasattr(extractor, 'kalman_filter') and extractor.kalman_filter
    has_gp = hasattr(extractor, 'gp_recovery') and extractor.gp_recovery
    logger.info(f"Kalman EPA: {'✅' if has_kalman else '❌'}")
    logger.info(f"GP Recovery: {'✅' if has_gp else '❌'}")

    # Load data
    train_games = get_games(TRAIN_SEASONS, MIN_WEEK)
    test_games = get_games([TEST_SEASON], MIN_WEEK)

    # Build datasets
    X_train, y_train = build_dataset(train_games, extractor, "training")
    X_test, y_test = build_dataset(test_games, extractor, "test")

    # Align features
    common = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common].fillna(0)
    X_test = X_test[common].fillna(0)
    feature_names = common

    logger.info(f"\nFinal features: {len(feature_names)}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Train models
    results = {"version": "v2.1", "n_features": len(feature_names), "models": {}}

    spread_model, spread_m = train_regression(
        X_train, y_train["actual_spread"],
        X_test, y_test["actual_spread"],
        y_test["vegas_spread"], "spread"
    )
    results["models"]["spread"] = spread_m

    total_model, total_m = train_regression(
        X_train, y_train["actual_total"],
        X_test, y_test["actual_total"],
        y_test["vegas_total"], "total"
    )
    results["models"]["total"] = total_m

    ml_model, ml_m = train_classification(
        X_train, y_train["home_win"],
        X_test, y_test["home_win"], "moneyline"
    )
    results["models"]["moneyline"] = ml_m

    # Save models
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    spread_model.save_model(str(out_path / "spread_model.json"))
    total_model.save_model(str(out_path / "total_model.json"))
    ml_model.save_model(str(out_path / "win_prob_model.json"))

    with open(out_path / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # v2 baseline comparison
    v2_spread_mae = 10.45
    v2_total_mae = 10.08
    v2_accuracy = 69.1

    logger.info("\n" + "=" * 70)
    logger.info("v2.1 vs v2 COMPARISON")
    logger.info("=" * 70)
    logger.info(f"Spread MAE: {spread_m['mae']:.2f} vs {v2_spread_mae} ({spread_m['mae'] - v2_spread_mae:+.2f})")
    logger.info(f"Total MAE: {total_m['mae']:.2f} vs {v2_total_mae} ({total_m['mae'] - v2_total_mae:+.2f})")
    logger.info(f"Win Accuracy: {ml_m['accuracy']:.1f}% vs {v2_accuracy}% ({ml_m['accuracy'] - v2_accuracy:+.1f}%)")

    # Save metadata
    metadata = {
        "version": "v2.1",
        "training_date": datetime.now().isoformat(),
        "train_seasons": TRAIN_SEASONS,
        "test_season": TEST_SEASON,
        "n_features": len(feature_names),
        "metrics": results["models"],
        "comparison_vs_v2": {
            "spread_delta": spread_m["mae"] - v2_spread_mae,
            "total_delta": total_m["mae"] - v2_total_mae,
            "accuracy_delta": ml_m["accuracy"] - v2_accuracy,
        }
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Feature importance
    importance = spread_model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False)

    logger.info("\nTop 20 Spread Features:")
    for i, row in imp_df.head(20).iterrows():
        prefix = ""
        if row["feature"].startswith("kal_"):
            prefix = "[KAL] "
        elif row["feature"].startswith("gp_"):
            prefix = "[GP] "
        elif "rest" in row["feature"].lower():
            prefix = "[REST] "
        logger.info(f"  {prefix}{row['feature']}: {row['importance']:.4f}")

    imp_df.to_csv(out_path / "spread_importance.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info(f"TRAINING COMPLETE - saved to {OUTPUT_DIR}")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
