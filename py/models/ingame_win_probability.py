#!/usr/bin/env python3
"""
In-Game Win Probability Model

Predicts win probability at any point during a game using play-by-play data.

Features:
- Score differential
- Time remaining (seconds)
- Field position (yards from own endzone)
- Down & distance
- Quarter
- Timeouts remaining (if available)

Target: Did home team win?

Usage:
    # Train model
    python py/models/ingame_win_probability.py \
        --start-season 2006 \
        --end-season 2021 \
        --test-seasons 2024 \
        --output-dir models/ingame_wp/v1

    # Batch inference on a specific game
    python py/models/ingame_win_probability.py \
        --mode inference \
        --game-id 2024_10_CIN_BAL \
        --model-dir models/ingame_wp/v1
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InGameWinProbabilityModel:
    """Train and predict in-game win probability."""

    def __init__(self):
        """Initialize model."""
        self.db_config = {
            "dbname": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
            "host": "localhost",
            "port": 5544,
        }
        self.model = None
        self.feature_names = None

    def connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def extract_features(self, seasons: list[int], min_time_remaining: int = 0) -> pd.DataFrame:
        """
        Extract in-game features from play-by-play data.

        Args:
            seasons: List of seasons to include
            min_time_remaining: Minimum time remaining to include (0 = all plays)

        Returns:
            DataFrame with play-level features
        """
        logger.info(f"Extracting in-game features for seasons {seasons[0]}-{seasons[-1]}...")

        conn = self.connect_db()

        # Build season list for SQL
        season_list = ",".join(str(s) for s in seasons)

        query = f"""
        WITH game_outcomes AS (
            SELECT
                game_id,
                home_team,
                away_team,
                CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_won
            FROM games
            WHERE home_score IS NOT NULL
              AND SUBSTRING(game_id, 1, 4)::int IN ({season_list})
        ),
        play_features AS (
            SELECT
                p.game_id,
                p.play_id,
                p.posteam,
                p.defteam,
                p.quarter,
                p.time_seconds,
                p.down,
                p.ydstogo,
                p.posteam_score,
                p.defteam_score,
                p.score_differential,
                -- Field position (yardline_100 not in schema, use estimated from context)
                -- For now, we'll use score_differential as proxy
                100 - COALESCE(p.ydstogo, 10) as estimated_yardline,
                p.goal_to_go,
                p.qb_kneel,
                p.qb_spike,
                -- Game clock features
                CASE
                    WHEN p.quarter = 1 THEN 3600 + p.time_seconds
                    WHEN p.quarter = 2 THEN 1800 + p.time_seconds
                    WHEN p.quarter = 3 THEN 1800 - p.time_seconds
                    WHEN p.quarter = 4 THEN 0 + (900 - p.time_seconds)
                    ELSE 0
                END as total_time_elapsed,
                3600 - CASE
                    WHEN p.quarter = 1 THEN p.time_seconds
                    WHEN p.quarter = 2 THEN 1800 + p.time_seconds
                    WHEN p.quarter = 3 THEN 1800 + 1800 - p.time_seconds
                    WHEN p.quarter = 4 THEN 1800 + 1800 + (900 - p.time_seconds)
                    ELSE 3600
                END as time_remaining_reg,
                g.home_team,
                g.away_team,
                go.home_won
            FROM plays p
            INNER JOIN games g ON p.game_id = g.game_id
            INNER JOIN game_outcomes go ON p.game_id = go.game_id
            WHERE p.posteam IS NOT NULL
              AND p.quarter IS NOT NULL
              AND p.quarter <= 4  -- Regular time only for now
              AND p.posteam_score IS NOT NULL
              AND p.defteam_score IS NOT NULL
        )
        SELECT
            game_id,
            play_id,
            posteam,
            defteam,
            home_team,
            away_team,
            quarter,
            time_seconds,
            GREATEST(0, time_remaining_reg) as time_remaining,
            down,
            ydstogo,
            posteam_score,
            defteam_score,
            score_differential,
            estimated_yardline,
            COALESCE(goal_to_go, 0) as goal_to_go,
            COALESCE(qb_kneel, 0) as qb_kneel,
            COALESCE(qb_spike, 0) as qb_spike,
            -- Is posteam the home team?
            CASE WHEN posteam = home_team THEN 1 ELSE 0 END as posteam_is_home,
            -- Target: did posteam's perspective team (home) win?
            home_won
        FROM play_features
        WHERE time_remaining_reg >= {min_time_remaining}
        ORDER BY game_id, play_id;
        """

        logger.info("Executing feature extraction query...")
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"Extracted {len(df):,} plays from {df['game_id'].nunique():,} games")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw play data.

        Args:
            df: Raw play features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")

        # Handle missing values
        df["down"] = df["down"].fillna(0)
        df["ydstogo"] = df["ydstogo"].fillna(10)
        df["estimated_yardline"] = df["estimated_yardline"].fillna(50)

        # Score features (from perspective of posteam)
        df["posteam_score_lead"] = df.apply(
            lambda row: (
                row["posteam_score"] - row["defteam_score"]
                if row["posteam_is_home"] == 1
                else row["defteam_score"] - row["posteam_score"]
            ),
            axis=1,
        )

        # For home team win prediction, flip if posteam is away
        df["home_score_lead"] = df.apply(
            lambda row: (
                row["posteam_score"] - row["defteam_score"]
                if row["posteam_is_home"] == 1
                else row["defteam_score"] - row["posteam_score"]
            ),
            axis=1,
        )

        # Time features
        df["time_remaining_minutes"] = df["time_remaining"] / 60.0
        df["quarter_time_ratio"] = df["time_seconds"] / 900.0  # % through quarter

        # Situation features
        df["is_red_zone"] = (df["estimated_yardline"] <= 20).astype(int)
        df["is_midfield"] = (
            (df["estimated_yardline"] >= 40) & (df["estimated_yardline"] <= 60)
        ).astype(int)
        df["is_4th_down"] = (df["down"] == 4).astype(int)
        df["is_3rd_down"] = (df["down"] == 3).astype(int)
        df["long_distance"] = (df["ydstogo"] >= 7).astype(int)

        # Critical game situations
        df["is_two_minute_drill"] = (df["time_remaining"] <= 120).astype(int)
        df["is_final_minute"] = (df["time_remaining"] <= 60).astype(int)

        # Remove plays with missing critical data
        df = df.dropna(subset=["time_remaining", "home_score_lead", "home_won"])

        logger.info(f"Engineered features, {len(df):,} plays remaining")

        return df

    def prepare_training_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """
        Prepare features and target for training.

        Args:
            df: Engineered features DataFrame

        Returns:
            (X, y, feature_names) tuple
        """
        # Feature columns
        feature_cols = [
            "home_score_lead",
            "time_remaining",
            "time_remaining_minutes",
            "quarter",
            "quarter_time_ratio",
            "down",
            "ydstogo",
            "estimated_yardline",
            "goal_to_go",
            "is_red_zone",
            "is_midfield",
            "is_4th_down",
            "is_3rd_down",
            "long_distance",
            "is_two_minute_drill",
            "is_final_minute",
            "qb_kneel",
            "qb_spike",
        ]

        X = df[feature_cols].copy()
        y = df["home_won"].values

        # Fill any remaining NaNs
        X = X.fillna(0)

        logger.info(f"Training data: {len(X):,} samples, {len(feature_cols)} features")
        logger.info(f"Target balance: {y.mean():.1%} home wins")

        return X, y, feature_cols

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> xgb.Booster:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Trained XGBoost booster
        """
        logger.info("Training in-game win probability model...")

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "tree_method": "hist",
            "seed": 42,
        }

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Eval set
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        # Train
        logger.info(f"Training with {len(X_train):,} plays...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        logger.info(f"✓ Training complete (best iteration: {model.best_iteration})")

        return model

    def evaluate(
        self, model: xgb.Booster, X: pd.DataFrame, y: np.ndarray, label: str = "Test"
    ) -> dict:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X: Features
            y: True labels
            label: Dataset label for logging

        Returns:
            Dictionary of metrics
        """
        dtest = xgb.DMatrix(X)
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "brier_score": float(brier_score_loss(y, y_pred_proba)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "auc": float(roc_auc_score(y, y_pred_proba)),
            "home_win_rate": float(y.mean()),
            "n_samples": len(y),
        }

        logger.info(f"\n{label} Metrics:")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.1%}")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Samples: {metrics['n_samples']:,}")

        return metrics

    def save_model(self, model: xgb.Booster, output_dir: Path, metadata: dict):
        """Save model and metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / "model.json"
        model.save_model(str(model_path))
        logger.info(f"✓ Saved model to {model_path}")

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata to {metadata_path}")

    def load_model(self, model_dir: Path) -> tuple[xgb.Booster, dict]:
        """Load model and metadata."""
        model_path = model_dir / "model.json"
        metadata_path = model_dir / "metadata.json"

        model = xgb.Booster()
        model.load_model(str(model_path))

        with open(metadata_path) as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]

        logger.info(f"✓ Loaded model from {model_dir}")
        return model, metadata

    def predict_game(self, game_id: str, model: xgb.Booster | None = None) -> pd.DataFrame:
        """
        Generate play-by-play win probabilities for a specific game.

        Args:
            game_id: Game ID to analyze
            model: Trained model (uses self.model if None)

        Returns:
            DataFrame with play-by-play win probabilities
        """
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model available. Train or load a model first.")

        logger.info(f"Generating in-game win probabilities for {game_id}...")

        # Extract features for this game
        conn = self.connect_db()

        query = """
        SELECT
            p.game_id,
            p.play_id,
            p.quarter,
            p.time_seconds,
            p.down,
            p.ydstogo,
            p.posteam,
            p.posteam_score,
            p.defteam_score,
            g.home_team,
            g.away_team
        FROM plays p
        INNER JOIN games g ON p.game_id = g.game_id
        WHERE p.game_id = %s
          AND p.quarter <= 4
          AND p.posteam IS NOT NULL
        ORDER BY p.play_id;
        """

        df = pd.read_sql(query, conn, params=(game_id,))
        conn.close()

        if len(df) == 0:
            logger.warning(f"No plays found for game {game_id}")
            return pd.DataFrame()

        # Engineer features (simplified for inference)
        df["time_remaining"] = 3600 - ((df["quarter"] - 1) * 900 + (900 - df["time_seconds"]))
        df["time_remaining"] = df["time_remaining"].clip(lower=0)
        df["time_remaining_minutes"] = df["time_remaining"] / 60.0
        df["quarter_time_ratio"] = df["time_seconds"] / 900.0

        df["home_score_lead"] = df.apply(
            lambda row: (
                row["posteam_score"] - row["defteam_score"]
                if row["posteam"] == row["home_team"]
                else row["defteam_score"] - row["posteam_score"]
            ),
            axis=1,
        )

        df["down"] = df["down"].fillna(0)
        df["ydstogo"] = df["ydstogo"].fillna(10)
        df["estimated_yardline"] = 100 - df["ydstogo"]  # Rough estimate
        df["goal_to_go"] = 0
        df["is_red_zone"] = (df["estimated_yardline"] <= 20).astype(int)
        df["is_midfield"] = (
            (df["estimated_yardline"] >= 40) & (df["estimated_yardline"] <= 60)
        ).astype(int)
        df["is_4th_down"] = (df["down"] == 4).astype(int)
        df["is_3rd_down"] = (df["down"] == 3).astype(int)
        df["long_distance"] = (df["ydstogo"] >= 7).astype(int)
        df["is_two_minute_drill"] = (df["time_remaining"] <= 120).astype(int)
        df["is_final_minute"] = (df["time_remaining"] <= 60).astype(int)
        df["qb_kneel"] = 0
        df["qb_spike"] = 0

        # Prepare features
        X = df[self.feature_names].fillna(0)

        # Predict
        dtest = xgb.DMatrix(X)
        df["home_win_prob"] = model.predict(dtest)

        logger.info(f"✓ Generated {len(df)} play-by-play probabilities")

        return df[
            [
                "game_id",
                "play_id",
                "quarter",
                "time_seconds",
                "posteam",
                "posteam_score",
                "defteam_score",
                "home_win_prob",
            ]
        ]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="In-game win probability model")
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--start-season", type=int, default=2006)
    parser.add_argument("--end-season", type=int, default=2021)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2024])
    parser.add_argument("--output-dir", default="models/ingame_wp/v1")
    parser.add_argument("--model-dir", help="Model directory for inference")
    parser.add_argument("--game-id", help="Game ID for inference")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    model_trainer = InGameWinProbabilityModel()

    if args.mode == "train":
        # Training mode
        logger.info("=" * 80)
        logger.info("IN-GAME WIN PROBABILITY MODEL TRAINING")
        logger.info("=" * 80)

        # Train seasons
        train_seasons = list(range(args.start_season, args.end_season + 1))
        test_seasons = args.test_seasons

        # Extract features
        logger.info(f"\nTraining seasons: {train_seasons[0]}-{train_seasons[-1]}")
        logger.info(f"Test seasons: {test_seasons}")

        df_train = model_trainer.extract_features(train_seasons)
        df_train = model_trainer.engineer_features(df_train)

        # Prepare training data
        X_train, y_train, feature_names = model_trainer.prepare_training_data(df_train)
        model_trainer.feature_names = feature_names

        # Train model
        model = model_trainer.train(X_train, y_train)
        model_trainer.model = model

        # Evaluate on train
        train_metrics = model_trainer.evaluate(model, X_train, y_train, "Train")

        # Evaluate on test seasons
        test_results = {}
        for test_season in test_seasons:
            logger.info(f"\nEvaluating on {test_season}...")
            df_test = model_trainer.extract_features([test_season])
            df_test = model_trainer.engineer_features(df_test)
            X_test, y_test, _ = model_trainer.prepare_training_data(df_test)

            test_metrics = model_trainer.evaluate(model, X_test, y_test, f"{test_season} Test")
            test_results[str(test_season)] = test_metrics

        # Save model
        metadata = {
            "model_type": "ingame_win_probability",
            "version": "v1",
            "created_at": datetime.now().isoformat(),
            "train_seasons": f"{train_seasons[0]}-{train_seasons[-1]}",
            "test_seasons": test_seasons,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "train_metrics": train_metrics,
            "test_results": test_results,
            "hyperparameters": {
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        }

        output_dir = Path(args.output_dir)
        model_trainer.save_model(model, output_dir, metadata)

        print("\n" + "=" * 80)
        print("IN-GAME WIN PROBABILITY MODEL TRAINING COMPLETE")
        print("=" * 80)
        print(f"Output: {output_dir}")
        print(f"Train Brier: {train_metrics['brier_score']:.4f}")
        print(
            f"Test Brier ({test_seasons[0]}): {test_results[str(test_seasons[0])]['brier_score']:.4f}"
        )

    else:
        # Inference mode
        if not args.model_dir or not args.game_id:
            parser.error("--model-dir and --game-id required for inference mode")

        logger.info("=" * 80)
        logger.info("IN-GAME WIN PROBABILITY INFERENCE")
        logger.info("=" * 80)

        model_dir = Path(args.model_dir)
        model, metadata = model_trainer.load_model(model_dir)

        # Generate predictions
        predictions = model_trainer.predict_game(args.game_id, model)

        # Print sample predictions
        print("\n" + "=" * 80)
        print(f"IN-GAME WIN PROBABILITIES: {args.game_id}")
        print("=" * 80)
        print("\nSample Predictions (every 20 plays):")
        print(
            predictions.iloc[::20][
                [
                    "quarter",
                    "time_seconds",
                    "posteam",
                    "posteam_score",
                    "defteam_score",
                    "home_win_prob",
                ]
            ]
        )

        # Save to CSV
        output_path = f"data/predictions/ingame_wp_{args.game_id}.csv"
        predictions.to_csv(output_path, index=False)
        print(f"\n✓ Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
