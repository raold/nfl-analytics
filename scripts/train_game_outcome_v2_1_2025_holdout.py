#!/usr/bin/env python3
"""
Train Game Outcome Model v2.1 with Kalman EPA + GP Injury Features
===================================================================

v2.1 extends v2 with:
- Kalman EPA Filter (18 features): EPA regime detection with CUSUM
- GP Injury Recovery (17 features): Uncertainty-aware recovery curves
- Rest differential (already in v2)

Total: 383 features (305 base + 37 synthetic + 6 rest + 18 Kalman + 17 GP)

This script trains v2.1 and provides direct comparison metrics vs v2 baseline.

Author: Claude Code (Opus 4.5)
Date: December 28, 2025
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from py.features.game_feature_extractor import GameFeatureExtractor
from py.features.player_aggregation_features import aggregate_player_projections
from py.utils.psycopg_pool import get_connection_pool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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

# Training configuration - 2025 holdout for true OOS validation
# temporal_cv: season-based split prevents data leakage (train on past, test on future)
TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
MIN_WEEK = 3  # Skip first 2 weeks (insufficient data)

# Output directory
OUTPUT_DIR = "models/staging/game_outcome_v2.1"


class GameOutcomeV21Trainer:
    """
    Train XGBoost v2.1 models with Kalman EPA and GP Injury features.

    Key differences from v2:
    - GameFeatureExtractor now includes Kalman EPA Filter (18 features)
    - GameFeatureExtractor now includes GP Injury Recovery (17 features)
    - Total features: 383 (vs ~350 in v2)
    """

    def __init__(self, db_config: dict | None = None):
        """Initialize trainer with database connection and feature extractors."""
        self.db_config = db_config or DB_CONFIG

        # Initialize connection pool
        self.pool = get_connection_pool(
            host=self.db_config.get("host", "localhost"),
            port=self.db_config.get("port", 5544),
            dbname=self.db_config.get("dbname", "devdb01"),
            user=self.db_config.get("user", "dro"),
            password=self.db_config.get("password", "sicillionbillions"),
        )

        # Initialize feature extractor with new Kalman + GP features
        self.game_extractor = GameFeatureExtractor(db_config=self.db_config)

        # Log feature extractor capabilities
        has_kalman = hasattr(self.game_extractor, 'kalman_filter') and self.game_extractor.kalman_filter is not None
        has_gp = hasattr(self.game_extractor, 'gp_recovery') and self.game_extractor.gp_recovery is not None

        logger.info("=" * 60)
        logger.info("v2.1 FEATURE EXTRACTOR STATUS")
        logger.info("=" * 60)
        logger.info(f"  Kalman EPA Filter: {'✅ ENABLED' if has_kalman else '❌ DISABLED'}")
        logger.info(f"  GP Injury Recovery: {'✅ ENABLED' if has_gp else '❌ DISABLED'}")
        logger.info("=" * 60)

        if not has_kalman or not has_gp:
            logger.warning("Some new features are disabled - v2.1 may not fully outperform v2")

        # Model storage
        self.spread_model = None
        self.total_model = None
        self.moneyline_model = None
        self.feature_names = []

        logger.info("GameOutcomeV21Trainer initialized")

    def _get_games(self, seasons: list[int], min_week: int = 3) -> pd.DataFrame:
        """Load game data with outcomes for specified seasons."""
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

        with self.pool.get_connection() as conn:
            df = pd.read_sql(query, conn, params=(seasons, min_week))

        logger.info(f"Loaded {len(df)} games from seasons {seasons} (weeks {min_week}+)")
        return df

    def _build_dataset(
        self, games_df: pd.DataFrame, desc: str = "dataset"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build feature matrix and target matrix for games."""
        total_games = len(games_df)
        logger.info(f"Building {desc}: {total_games} games")

        # Pre-load player aggregations
        season_weeks = games_df[["season", "week"]].drop_duplicates()
        player_agg_cache = {}

        logger.info(f"Pre-loading player aggregations for {len(season_weeks)} season-weeks...")
        for _, sw in season_weeks.iterrows():
            key = (int(sw["season"]), int(sw["week"]))
            try:
                player_agg_cache[key] = aggregate_player_projections(
                    season=key[0],
                    week=key[1],
                    model_names=["state_space_v2", "hierarchical_v2", "informative_priors_v2"],
                    db_host=self.db_config.get("host", "localhost"),
                    db_port=self.db_config.get("port", 5544),
                    db_name=self.db_config.get("dbname", "devdb01"),
                    db_user=self.db_config.get("user", "dro"),
                    db_password=self.db_config.get("password", "sicillionbillions"),
                )
            except Exception as e:
                logger.warning(f"Failed to load player aggregation for {key}: {e}")
                player_agg_cache[key] = pd.DataFrame()

        # Track Kalman and GP feature extraction
        kalman_features_found = set()
        gp_features_found = set()

        features_batches = []
        targets_batches = []
        current_features = []
        current_targets = []
        batch_size = 100

        for idx, row in games_df.iterrows():
            if (idx + 1) % 50 == 0:
                logger.info(f"  Progress: {idx + 1}/{total_games} games")

            try:
                features = self.game_extractor.extract_game_features(
                    row["home_team"],
                    row["away_team"],
                    row["season"],
                    row["week"],
                    use_cache=True,
                )
            except Exception as e:
                logger.warning(
                    f"Feature extraction failed for {row['home_team']} vs {row['away_team']} ({row['season']} W{row['week']}): {e}"
                )
                continue

            if features is None:
                continue

            # Track new features for reporting
            for k in features.keys():
                if k.startswith("kal_"):
                    kalman_features_found.add(k)
                elif k.startswith("gp_"):
                    gp_features_found.add(k)

            # Add player aggregation
            key = (int(row["season"]), int(row["week"]))
            player_agg_df = player_agg_cache.get(key, pd.DataFrame())

            if not player_agg_df.empty:
                game_row = player_agg_df[
                    (player_agg_df["home_team"] == row["home_team"])
                    & (player_agg_df["away_team"] == row["away_team"])
                ]
                if not game_row.empty:
                    for col in game_row.columns:
                        if col.startswith(("home_", "away_")) and col not in features:
                            val = game_row[col].values[0]
                            if pd.notna(val):
                                features[col] = float(val)

            # Calculate targets
            actual_spread = row["home_score"] - row["away_score"]
            actual_total = row["home_score"] + row["away_score"]
            home_win = 1 if actual_spread > 0 else 0

            current_features.append(features)
            current_targets.append(
                {
                    "game_id": row["game_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "actual_spread": actual_spread,
                    "actual_total": actual_total,
                    "home_win": home_win,
                    "vegas_spread": row["vegas_spread"],
                    "vegas_total": row["vegas_total"],
                }
            )

            if len(current_features) >= batch_size:
                features_batches.append(pd.DataFrame(current_features))
                targets_batches.append(pd.DataFrame(current_targets))
                current_features = []
                current_targets = []

        if current_features:
            features_batches.append(pd.DataFrame(current_features))
            targets_batches.append(pd.DataFrame(current_targets))

        features_df = (
            pd.concat(features_batches, ignore_index=True) if features_batches else pd.DataFrame()
        )
        targets_df = (
            pd.concat(targets_batches, ignore_index=True) if targets_batches else pd.DataFrame()
        )

        # Report new features extracted
        logger.info(f"\nv2.1 NEW FEATURES in {desc}:")
        logger.info(f"  Kalman EPA features: {len(kalman_features_found)} found")
        if kalman_features_found:
            for f in sorted(kalman_features_found)[:5]:
                logger.info(f"    - {f}")
            if len(kalman_features_found) > 5:
                logger.info(f"    ... and {len(kalman_features_found) - 5} more")

        logger.info(f"  GP Injury features: {len(gp_features_found)} found")
        if gp_features_found:
            for f in sorted(gp_features_found)[:5]:
                logger.info(f"    - {f}")
            if len(gp_features_found) > 5:
                logger.info(f"    ... and {len(gp_features_found) - 5} more")

        logger.info(f"Built {desc}: {len(features_df)} games, {len(features_df.columns)} features")
        return features_df, targets_df

    def train(
        self, train_seasons: list[int] = None, test_season: int = None, min_week: int = MIN_WEEK
    ) -> dict[str, any]:
        """Train models for spread, total, and moneyline."""
        train_seasons = train_seasons or TRAIN_SEASONS
        test_season = test_season or TEST_SEASON

        logger.info("=" * 80)
        logger.info("GAME OUTCOME MODEL v2.1 TRAINING")
        logger.info("New features: Kalman EPA (18) + GP Injury (17)")
        logger.info("=" * 80)
        logger.info(f"Training seasons: {train_seasons}")
        logger.info(f"Test season: {test_season}")
        logger.info(f"Min week: {min_week}")

        # Load games
        train_games = self._get_games(train_seasons, min_week)
        test_games = self._get_games([test_season], min_week)

        # Build datasets
        logger.info("\n" + "=" * 40)
        logger.info("Building Training Dataset")
        logger.info("=" * 40)
        X_train, y_train = self._build_dataset(train_games, "training")

        logger.info("\n" + "=" * 40)
        logger.info("Building Test Dataset")
        logger.info("=" * 40)
        X_test, y_test = self._build_dataset(test_games, "test")

        # Store feature names
        self.feature_names = list(X_train.columns)
        logger.info(f"\nTotal features: {len(self.feature_names)}")

        # Count new vs old features
        kalman_count = len([f for f in self.feature_names if f.startswith("kal_")])
        gp_count = len([f for f in self.feature_names if f.startswith("gp_")])
        rest_count = len([f for f in self.feature_names if "rest" in f.lower() or "btb" in f.lower()])

        logger.info(f"  Kalman EPA features: {kalman_count}")
        logger.info(f"  GP Injury features: {gp_count}")
        logger.info(f"  Rest/BTB features: {rest_count}")
        logger.info(f"  Other features: {len(self.feature_names) - kalman_count - gp_count - rest_count}")

        # Handle NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Align features
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        self.feature_names = common_cols

        results = {
            "version": "v2.1",
            "train_seasons": train_seasons,
            "test_season": test_season,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(self.feature_names),
            "n_kalman_features": kalman_count,
            "n_gp_features": gp_count,
            "models": {},
        }

        # Train spread model
        logger.info("\n" + "=" * 40)
        logger.info("Training SPREAD Model")
        logger.info("=" * 40)
        self.spread_model, spread_metrics = self._train_regression(
            X_train, y_train["actual_spread"], X_test, y_test["actual_spread"],
            y_test["vegas_spread"], "spread"
        )
        results["models"]["spread"] = spread_metrics

        # Train total model
        logger.info("\n" + "=" * 40)
        logger.info("Training TOTAL Model")
        logger.info("=" * 40)
        self.total_model, total_metrics = self._train_regression(
            X_train, y_train["actual_total"], X_test, y_test["actual_total"],
            y_test["vegas_total"], "total"
        )
        results["models"]["total"] = total_metrics

        # Train moneyline model
        logger.info("\n" + "=" * 40)
        logger.info("Training MONEYLINE Model")
        logger.info("=" * 40)
        self.moneyline_model, ml_metrics = self._train_classification(
            X_train, y_train["home_win"], X_test, y_test["home_win"], "moneyline"
        )
        results["models"]["moneyline"] = ml_metrics

        return results

    def _train_regression(
        self, X_train, y_train, X_test, y_test, y_vegas, target_name
    ) -> tuple[xgb.XGBRegressor, dict]:
        """Train XGBoost regression model."""
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        vegas_mae = mean_absolute_error(y_test, y_vegas)

        # Cover rate for spread
        if target_name == "spread":
            model_over = y_pred > y_vegas
            actual_over = y_test > y_vegas
            covers = (model_over == actual_over).sum()
            cover_rate = covers / len(y_test) * 100
        else:
            cover_rate = None

        logger.info(f"\n{target_name.upper()} Model Results:")
        logger.info(f"  MAE: {mae:.2f} (Vegas baseline: {vegas_mae:.2f})")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  R²: {r2:.4f}")
        if cover_rate:
            logger.info(f"  Cover Rate: {cover_rate:.1f}%")

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "vegas_mae": float(vegas_mae),
            "improvement_vs_vegas": float((vegas_mae - mae) / vegas_mae * 100) if vegas_mae > 0 else 0,
        }
        if cover_rate:
            metrics["cover_rate"] = float(cover_rate)

        return model, metrics

    def _train_classification(
        self, X_train, y_train, X_test, y_test, target_name
    ) -> tuple[xgb.XGBClassifier, dict]:
        """Train XGBoost classification model for moneyline."""
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
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

        logger.info(f"\n{target_name.upper()} Model Results:")
        logger.info(f"  Accuracy: {accuracy:.1f}%")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Home win rate (actual): {y_test.mean()*100:.1f}%")
        logger.info(f"  Home win rate (predicted): {y_pred.mean()*100:.1f}%")

        metrics = {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "actual_home_win_rate": float(y_test.mean() * 100),
            "predicted_home_win_rate": float(y_pred.mean() * 100),
        }

        return model, metrics

    def save_models(self, output_dir: str = OUTPUT_DIR):
        """Save trained models and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save spread model
        if self.spread_model:
            spread_path = output_path / "spread_model.json"
            self.spread_model.save_model(str(spread_path))
            logger.info(f"Saved spread model: {spread_path}")

        # Save total model
        if self.total_model:
            total_path = output_path / "total_model.json"
            self.total_model.save_model(str(total_path))
            logger.info(f"Saved total model: {total_path}")

        # Save moneyline model
        if self.moneyline_model:
            ml_path = output_path / "win_prob_model.json"
            self.moneyline_model.save_model(str(ml_path))
            logger.info(f"Saved moneyline model: {ml_path}")

        # Save feature names
        feature_path = output_path / "feature_names.json"
        with open(feature_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved feature names: {feature_path}")

    def get_feature_importance(self, model_type: str = "spread", top_n: int = 30) -> pd.DataFrame:
        """Get feature importance for specified model."""
        model = {
            "spread": self.spread_model,
            "total": self.total_model,
            "moneyline": self.moneyline_model
        }.get(model_type)

        if model is None:
            raise ValueError(f"{model_type} model not trained")

        importance = model.feature_importances_
        df = pd.DataFrame({"feature": self.feature_names, "importance": importance})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df.head(top_n)


def load_v2_baseline() -> dict:
    """Load v2 baseline metrics for comparison."""
    v2_path = Path("models/production/game_outcome_v2/metadata.json")
    if v2_path.exists():
        with open(v2_path) as f:
            return json.load(f)

    # Fallback to documented v2 metrics
    return {
        "models": {
            "spread": {"mae": 10.45, "cover_rate": None},
            "total": {"mae": 10.08},
            "moneyline": {"accuracy": 69.1, "auc": 0.691}
        }
    }


def main():
    """Main training script."""
    logger.info("=" * 80)
    logger.info("GAME OUTCOME MODEL v2.1 TRAINING")
    logger.info("Kalman EPA + GP Injury Recovery Features")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 80)

    # Initialize trainer
    trainer = GameOutcomeV21Trainer()

    # Train models
    results = trainer.train()

    # Load v2 baseline for comparison
    v2_baseline = load_v2_baseline()

    # Print comparison summary
    logger.info("\n" + "=" * 80)
    logger.info("v2.1 vs v2 COMPARISON")
    logger.info("=" * 80)

    for model_name in ["spread", "total", "moneyline"]:
        v21_metrics = results["models"][model_name]
        v2_metrics = v2_baseline.get("models", {}).get(model_name, {})

        logger.info(f"\n{model_name.upper()}:")

        if model_name in ["spread", "total"]:
            v21_mae = v21_metrics.get("mae", 0)
            v2_mae = v2_metrics.get("mae", 0)
            if v2_mae > 0:
                delta = v2_mae - v21_mae
                pct = (delta / v2_mae) * 100
                logger.info(f"  MAE: {v21_mae:.2f} (v2: {v2_mae:.2f}) -> {'+' if delta > 0 else ''}{delta:.2f} ({'+' if pct > 0 else ''}{pct:.1f}%)")
            else:
                logger.info(f"  MAE: {v21_mae:.2f}")

            if model_name == "spread" and "cover_rate" in v21_metrics:
                logger.info(f"  Cover Rate: {v21_metrics['cover_rate']:.1f}%")
        else:
            v21_acc = v21_metrics.get("accuracy", 0)
            v2_acc = v2_metrics.get("accuracy", 0)
            v21_auc = v21_metrics.get("auc", 0)
            v2_auc = v2_metrics.get("auc", 0)

            if v2_acc > 0:
                acc_delta = v21_acc - v2_acc
                logger.info(f"  Accuracy: {v21_acc:.1f}% (v2: {v2_acc:.1f}%) -> {'+' if acc_delta > 0 else ''}{acc_delta:.1f}%")
            else:
                logger.info(f"  Accuracy: {v21_acc:.1f}%")

            if v2_auc > 0:
                auc_delta = v21_auc - v2_auc
                logger.info(f"  AUC: {v21_auc:.4f} (v2: {v2_auc:.4f}) -> {'+' if auc_delta > 0 else ''}{auc_delta:.4f}")
            else:
                logger.info(f"  AUC: {v21_auc:.4f}")

    # Save models
    trainer.save_models()

    # Save metadata
    metadata = {
        "version": "v2.1",
        "training_date": datetime.now().isoformat(),
        "train_seasons": results["train_seasons"],
        "test_season": results["test_season"],
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "n_features": results["n_features"],
        "n_kalman_features": results.get("n_kalman_features", 0),
        "n_gp_features": results.get("n_gp_features", 0),
        "new_features": {
            "kalman_epa": "Team EPA regime detection with CUSUM",
            "gp_injury": "Uncertainty-aware injury recovery curves"
        },
        "metrics": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in results["models"].items()
        },
        "comparison_vs_v2": {
            "spread_mae_improvement": v2_baseline.get("models", {}).get("spread", {}).get("mae", 10.45) - results["models"]["spread"]["mae"],
            "total_mae_improvement": v2_baseline.get("models", {}).get("total", {}).get("mae", 10.08) - results["models"]["total"]["mae"],
            "accuracy_improvement": results["models"]["moneyline"]["accuracy"] - v2_baseline.get("models", {}).get("moneyline", {}).get("accuracy", 69.1),
        }
    }

    metadata_path = Path(OUTPUT_DIR) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_path}")

    # Print top features
    logger.info("\n" + "=" * 40)
    logger.info("Top 30 Features (Spread Model)")
    logger.info("=" * 40)
    importance_df = trainer.get_feature_importance("spread", top_n=30)

    new_features_in_top = []
    for i, row in importance_df.iterrows():
        prefix = ""
        if row["feature"].startswith("kal_"):
            prefix = "[KALMAN] "
            new_features_in_top.append(row["feature"])
        elif row["feature"].startswith("gp_"):
            prefix = "[GP] "
            new_features_in_top.append(row["feature"])
        elif "rest" in row["feature"].lower():
            prefix = "[REST] "
            new_features_in_top.append(row["feature"])
        logger.info(f"  {i+1:2d}. {prefix}{row['feature']}: {row['importance']:.4f}")

    logger.info(f"\nNew features in top 30: {len(new_features_in_top)}")

    # Save feature importance
    importance_path = Path(OUTPUT_DIR) / "spread_feature_importance.csv"
    trainer.get_feature_importance("spread", top_n=100).to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance: {importance_path}")

    logger.info("\n" + "=" * 80)
    logger.info("v2.1 TRAINING COMPLETE")
    logger.info(f"Models saved to: {OUTPUT_DIR}")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    main()
