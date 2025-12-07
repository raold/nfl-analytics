#!/usr/bin/env python3
"""
Train XGBoost Binary Classifier for Touchdown Prediction

Predicts probability of scoring a touchdown for rushing and receiving props.
Uses player-game level features including red zone attempts, goal-line carries,
and player role indicators.

Target: 75%+ accuracy on 2024 test set
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TouchdownPredictor:
    """
    XGBoost binary classifier for touchdown prediction.

    Features:
    - Red zone attempts (inside 20-yard line)
    - Goal-line carries (inside 5-yard line)
    - Player role (primary rusher/receiver vs backup)
    - Recent TD form
    - Defensive TD rate allowed
    """

    def __init__(
        self,
        max_depth: int = 6,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 3.0,  # Handle class imbalance
    ):
        """
        Initialize TD predictor.

        Args:
            max_depth: Max tree depth
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            scale_pos_weight: Weight for positive class (TDs are rare)
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

        # Database connection
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        logger.info(
            f"Initialized TouchdownPredictor with {n_estimators} trees, "
            f"depth={max_depth}, pos_weight={scale_pos_weight}"
        )

    def load_data(self, start_season: int = 2018, end_season: int = 2024) -> pd.DataFrame:
        """Load touchdown data from database"""
        conn = psycopg2.connect(**self.db_config)

        query = """
        WITH player_game_stats AS (
            SELECT
                p.game_id,
                g.season,
                g.week,
                p.rusher_player_id as player_id,
                p.rusher_player_name as player_name,
                'rushing' as stat_type,
                p.posteam as team,
                p.defteam as opponent,
                -- TD outcome (binary)
                MAX(CASE WHEN p.touchdown::int = 1 AND p.rush_attempt::int = 1 THEN 1 ELSE 0 END) as scored_td,
                -- Volume features
                COUNT(*) FILTER (WHERE p.rush_attempt::int = 1) as attempts,
                COUNT(*) FILTER (WHERE p.yardline_100 <= 20 AND p.rush_attempt::int = 1) as redzone_attempts,
                COUNT(*) FILTER (WHERE p.yardline_100 <= 5 AND p.rush_attempt::int = 1) as goalline_attempts,
                -- Performance features
                SUM(p.rushing_yards) as total_yards,
                AVG(p.rushing_yards) FILTER (WHERE p.rush_attempt::int = 1) as yards_per_carry,
                -- Context features
                AVG(p.score_differential) as avg_score_diff,
                COUNT(*) FILTER (WHERE p.quarter = 4) as fourth_quarter_attempts
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season BETWEEN %s AND %s
              AND p.season_type = 'REG'
              AND p.rusher_player_id IS NOT NULL
              AND p.rush_attempt::int = 1
            GROUP BY 1,2,3,4,5,6,7,8
            HAVING COUNT(*) FILTER (WHERE p.rush_attempt::int = 1) >= 5

            UNION ALL

            SELECT
                p.game_id,
                g.season,
                g.week,
                p.receiver_player_id as player_id,
                p.receiver_player_name as player_name,
                'receiving' as stat_type,
                p.posteam as team,
                p.defteam as opponent,
                -- TD outcome (binary)
                MAX(CASE WHEN p.touchdown::int = 1 AND p.pass_attempt::int = 1 THEN 1 ELSE 0 END) as scored_td,
                -- Volume features
                COUNT(*) FILTER (WHERE p.pass_attempt::int = 1) as attempts,
                COUNT(*) FILTER (WHERE p.yardline_100 <= 20 AND p.pass_attempt::int = 1) as redzone_attempts,
                COUNT(*) FILTER (WHERE p.yardline_100 <= 5 AND p.pass_attempt::int = 1) as goalline_attempts,
                -- Performance features
                SUM(CASE WHEN p.complete_pass::int = 1 THEN p.receiving_yards ELSE 0 END) as total_yards,
                AVG(p.air_yards) FILTER (WHERE p.pass_attempt::int = 1) as yards_per_carry,
                -- Context features
                AVG(p.score_differential) as avg_score_diff,
                COUNT(*) FILTER (WHERE p.quarter = 4) as fourth_quarter_attempts
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season BETWEEN %s AND %s
              AND p.season_type = 'REG'
              AND p.receiver_player_id IS NOT NULL
              AND p.pass_attempt::int = 1
            GROUP BY 1,2,3,4,5,6,7,8
            HAVING COUNT(*) FILTER (WHERE p.pass_attempt::int = 1) >= 3
        ),
        player_recent_form AS (
            SELECT
                player_id,
                season,
                week,
                stat_type,
                -- Rolling TD rate (last 3 games)
                AVG(scored_td::numeric) OVER (
                    PARTITION BY player_id, stat_type
                    ORDER BY season, week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as recent_td_rate_l3,
                -- Season TD rate
                AVG(scored_td::numeric) OVER (
                    PARTITION BY player_id, stat_type, season
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as season_td_rate
            FROM player_game_stats
        ),
        defense_stats AS (
            SELECT
                pgs.game_id,
                pgs.opponent,
                pgs.stat_type,
                -- Defensive TD rate allowed
                AVG(pgs.scored_td::numeric) as def_td_rate_allowed
            FROM player_game_stats pgs
            GROUP BY 1,2,3
        )
        SELECT
            pgs.*,
            prf.recent_td_rate_l3,
            prf.season_td_rate,
            ds.def_td_rate_allowed,
            -- Team strength proxy
            CASE
                WHEN pgs.team IN ('KC', 'BUF', 'PHI', 'SF', 'DAL') THEN 1
                ELSE 0
            END as is_elite_team
        FROM player_game_stats pgs
        LEFT JOIN player_recent_form prf
            ON pgs.player_id = prf.player_id
            AND pgs.season = prf.season
            AND pgs.week = prf.week
            AND pgs.stat_type = prf.stat_type
        LEFT JOIN defense_stats ds
            ON pgs.game_id = ds.game_id
            AND pgs.opponent = ds.opponent
            AND pgs.stat_type = ds.stat_type
        ORDER BY pgs.season, pgs.week, pgs.scored_td DESC
        """

        df = pd.read_sql(
            query, conn, params=[start_season, end_season, start_season, end_season]
        )
        conn.close()

        # Handle missing values
        df["recent_td_rate_l3"] = df.groupby("stat_type")["recent_td_rate_l3"].transform(
            lambda x: x.fillna(x.median())
        )
        df["season_td_rate"] = df.groupby("stat_type")["season_td_rate"].transform(
            lambda x: x.fillna(x.median())
        )
        df["def_td_rate_allowed"] = df.groupby("stat_type")["def_td_rate_allowed"].transform(
            lambda x: x.fillna(x.median())
        )

        # Create engineered features
        df["redzone_pct"] = df["redzone_attempts"] / df["attempts"].clip(lower=1)
        df["goalline_pct"] = df["goalline_attempts"] / df["attempts"].clip(lower=1)
        df["yards_per_attempt"] = df["total_yards"] / df["attempts"].clip(lower=1)

        logger.info(
            f"Loaded {len(df)} player-game records ({df['scored_td'].sum()} TDs, "
            f"{len(df) - df['scored_td'].sum()} non-TDs)"
        )
        logger.info(
            f"TD rate: {df['scored_td'].mean():.1%} (class balance: "
            f"{df['scored_td'].value_counts().to_dict()})"
        )

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target"""

        self.feature_names = [
            # Volume features
            "attempts",
            "redzone_attempts",
            "goalline_attempts",
            "redzone_pct",
            "goalline_pct",
            # Performance features
            "total_yards",
            "yards_per_attempt",
            # Form features
            "recent_td_rate_l3",
            "season_td_rate",
            # Context features
            "avg_score_diff",
            "fourth_quarter_attempts",
            "def_td_rate_allowed",
            "is_elite_team",
            # Stat type
            "stat_type",
        ]

        # Encode stat_type as binary
        df["stat_type"] = (df["stat_type"] == "rushing").astype(int)

        # Handle missing values
        X = df[self.feature_names].fillna(0).values
        y = df["scored_td"].values

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ):
        """Train XGBoost classifier"""

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=self.feature_names)

        # Validation set (optional)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "val"))

        # Training parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "aucpr"],  # Area under precision-recall curve
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "scale_pos_weight": self.scale_pos_weight,  # Handle imbalance
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "tree_method": "hist",
            "seed": 42,
        }

        logger.info("Training XGBoost TD classifier...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=20,
        )

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")

    def predict(self, X_test: np.ndarray) -> dict[str, np.ndarray]:
        """Make predictions"""

        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X_test)
        dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)

        # Predict probabilities
        probs = self.model.predict(dtest)

        # Default threshold of 0.5 for binary classification
        preds_binary = (probs >= 0.5).astype(int)

        return {"probabilities": probs, "predictions": preds_binary}

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
    ) -> dict:
        """Evaluate model performance"""

        predictions = self.predict(X_test)
        probs = predictions["probabilities"]
        preds = (probs >= threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        # Precision, recall, F1
        report = classification_report(
            y_test, preds, target_names=["No TD", "TD"], output_dict=True
        )

        # Find optimal threshold (maximize F1)
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "precision": report["TD"]["precision"],
            "recall": report["TD"]["recall"],
            "f1_score": report["TD"]["f1-score"],
            "confusion_matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "optimal_threshold": float(optimal_threshold),
            "threshold_used": threshold,
        }

        return metrics

    def save_model(self, filepath: str):
        """Save trained model"""

        # Save XGBoost model
        self.model.save_model(filepath)

        # Save metadata
        metadata = {
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "scale_pos_weight": self.scale_pos_weight,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "best_iteration": int(self.model.best_iteration),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath.replace(".json", "_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")


def main():
    """Train and evaluate TD prediction model"""

    logger.info("=" * 60)
    logger.info("TRAINING XGBOOST TOUCHDOWN PREDICTION MODEL")
    logger.info("=" * 60)

    # Initialize model
    predictor = TouchdownPredictor(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.05,
        scale_pos_weight=3.0,  # TDs occur ~25% of the time
    )

    # Load data
    logger.info("\nLoading touchdown data...")
    df = predictor.load_data(start_season=2018, end_season=2024)

    # Prepare features
    X, y = predictor.prepare_features(df)
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {predictor.feature_names}")

    # Train/val/test split
    train_mask = (df["season"] < 2023)
    val_mask = (df["season"] == 2023)
    test_mask = (df["season"] == 2024)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"\nTraining set: {len(X_train)} samples ({y_train.sum()} TDs)")
    logger.info(f"Validation set: {len(X_val)} samples ({y_val.sum()} TDs)")
    logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} TDs)")

    # Train model
    logger.info("\nTraining XGBoost (this may take 3-5 minutes)...")
    predictor.train(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    logger.info("\nEvaluating on 2024 test set...")
    metrics = predictor.evaluate(X_test, y_test, threshold=0.5)

    logger.info("\nTest Set Performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.1%}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.1%}")
    logger.info(f"  Recall: {metrics['recall']:.1%}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
    logger.info(f"  Optimal Threshold: {metrics['optimal_threshold']:.3f}")

    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {metrics['true_negatives']:>5d}")
    logger.info(f"  False Positives: {metrics['false_positives']:>5d}")
    logger.info(f"  False Negatives: {metrics['false_negatives']:>5d}")
    logger.info(f"  True Positives:  {metrics['true_positives']:>5d}")

    # Feature importance
    logger.info("\nTop 10 Feature Importances:")
    importance = predictor.model.get_score(importance_type="gain")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, score in sorted_importance:
        logger.info(f"  {feature:30s}: {score:.1f}")

    # Save model
    model_dir = Path("/Users/dro/rice/nfl-analytics/models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_touchdowns_v1.json"
    predictor.save_model(str(model_path))
    logger.info(f"\n✓ Model saved to {model_path}")

    # Sample predictions
    logger.info("\nSample Predictions (Test Set):")
    test_df = df[test_mask].copy()
    test_predictions = predictor.predict(X_test)
    test_df["td_probability"] = test_predictions["probabilities"]

    # Show high-probability TDs
    high_prob_tds = (
        test_df[test_df["td_probability"] >= 0.4]
        .sort_values("td_probability", ascending=False)
        .head(10)
    )

    for _, row in high_prob_tds.iterrows():
        stat_type_str = "RUSHING" if row['stat_type'] == 0 else "RECEIVING"
        logger.info(
            f"  {row['player_name']:25s} ({stat_type_str:>9s}): "
            f"P(TD)={row['td_probability']:.1%}, Actual={row['scored_td']}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("✓ TOUCHDOWN PREDICTION MODEL TRAINING COMPLETE")
    logger.info("=" * 60)

    # Check if target met
    if metrics["accuracy"] >= 0.75:
        logger.info(f"✅ TARGET MET: Accuracy {metrics['accuracy']:.1%} >= 75%")
    else:
        logger.warning(
            f"⚠️ TARGET MISSED: Accuracy {metrics['accuracy']:.1%} < 75% "
            f"(may need hyperparameter tuning)"
        )


if __name__ == "__main__":
    main()
