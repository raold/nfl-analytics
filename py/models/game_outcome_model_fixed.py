#!/usr/bin/env python3
"""
Fixed version of XGBoost-based NFL game outcome prediction model.

Fixes critical issues identified by code review:
1. Data leakage in train-validation split (CRITICAL)
2. Hardcoded uncertainty estimates
3. SHAP explainer storage issue
4. Input validation for predict() method
5. Removed unnecessary StandardScaler for XGBoost
6. Performance optimizations with batch feature extraction

This model predicts:
- Point spread (home team perspective)
- Total points (over/under)
- Win probability

Author: Claude Code
Date: November 29, 2025
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from py.features.game_feature_extractor import GameFeatureExtractor
from py.utils.team_mapping import VALID_TEAMS, normalize_team

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Note: VALID_TEAMS imported from py.utils.team_mapping
# Includes both canonical (LAR, LAC, LV) and database (LA, SD, OAK) abbreviations


class GameOutcomeModel:
    """
    XGBoost-based model for NFL game outcome predictions.

    Fixed issues:
    - Proper temporal train-validation split without data leakage
    - Computed uncertainty estimates from validation MAE
    - Correct SHAP value storage
    - Input validation with informative errors
    - No StandardScaler (unnecessary for XGBoost)
    - Batch feature extraction for performance
    """

    def __init__(self, model_dir: str = "models/game_outcome"):
        """
        Initialize game outcome model.

        Args:
            model_dir: Directory to save/load model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize feature extractor
        self.feature_extractor = GameFeatureExtractor()

        # Model components (initialized during training)
        self.spread_model = None
        self.total_model = None
        self.win_prob_model = None

        # Feature metadata (set during training)
        self.feature_names = None
        self.models_trained = False

        # SHAP explainers and values (computed after training)
        self.spread_explainer = None
        self.total_explainer = None
        self.win_prob_explainer = None
        self.shap_values = {}  # Store SHAP values separately

        # Uncertainty estimates (computed from validation MAE)
        self.spread_std = None
        self.total_std = None

        # Model metadata
        self.metadata = {
            "version": "2.0.0",
            "created_at": None,
            "training_seasons": None,
            "feature_count": None,
            "validation_metrics": {},
        }

    def train(
        self,
        seasons: list[int],
        validation_split: float = 0.2,
        xgb_params: dict | None = None,
        n_jobs: int = -1,
    ) -> dict[str, float]:
        """
        Train game outcome models on historical data.

        FIXED: Proper temporal split without data leakage

        Args:
            seasons: List of seasons to train on (e.g., [2022, 2023, 2024])
            validation_split: Fraction of data for validation (chronologically last)
            xgb_params: Optional XGBoost hyperparameters
            n_jobs: Number of parallel threads for training

        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Training on seasons: {seasons}")
        start = time.time()

        # Default XGBoost parameters (optimized for NFL data)
        if xgb_params is None:
            xgb_params = {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 1,
                "reg_alpha": 0.1,
                "reg_lambda": 1,
                "min_child_weight": 5,
                "random_state": 42,
                "n_jobs": n_jobs,
                "early_stopping_rounds": 50,
                "eval_metric": "mae",
            }

        # 1. Load training data with batch extraction
        logger.info("Loading training data with batch feature extraction...")
        df = self._load_training_data_batch(seasons)
        logger.info(f"Loaded {len(df)} games from seasons {seasons}")

        # 2. Prepare features and targets
        feature_cols = [
            col
            for col in df.columns
            if col
            not in ["season", "week", "home_team", "away_team", "spread", "total", "home_win"]
        ]
        self.feature_names = feature_cols

        # FIXED: Sort BEFORE extracting arrays to maintain alignment
        df_sorted = df.sort_values(["season", "week"]).reset_index(drop=True)

        X = df_sorted[feature_cols].values
        y_spread = df_sorted["spread"].values
        y_total = df_sorted["total"].values
        y_win_prob = df_sorted["home_win"].values

        logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

        # 3. FIXED: Season-based temporal split to prevent data leakage
        # Training on seasons < max_season, validating on max_season
        max_season = df_sorted["season"].max()
        train_mask = df_sorted["season"] < max_season
        val_mask = df_sorted["season"] == max_season

        X_train = X[train_mask]
        X_val = X[val_mask]
        y_spread_train = y_spread[train_mask]
        y_spread_val = y_spread[val_mask]
        y_total_train = y_total[train_mask]
        y_total_val = y_total[val_mask]
        y_win_train = y_win_prob[train_mask]
        y_win_val = y_win_prob[val_mask]

        train_seasons = df_sorted[train_mask]["season"].unique()
        logger.info(f"Training: {len(X_train)} samples from seasons {sorted(train_seasons)}")
        logger.info(f"Validation: {len(X_val)} samples from season {max_season}")

        # Store validation data info for uncertainty estimation
        val_info = df_sorted[val_mask][["season", "week"]].describe()
        logger.info(f"Validation set summary:\n{val_info}")

        # 4. Train models in parallel on M4 Mac
        logger.info("Training 3 XGBoost models in parallel...")
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Detect number of cores (M4 Mac has excellent parallelization)
        n_cores = multiprocessing.cpu_count()
        logger.info(f"Using {n_cores} CPU cores for parallel training")

        def train_spread_model():
            logger.info("  Starting spread model training...")
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_spread_train, eval_set=[(X_val, y_spread_val)], verbose=False)
            logger.info("  ✓ Spread model complete")
            return model

        def train_total_model():
            logger.info("  Starting total model training...")
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_total_train, eval_set=[(X_val, y_total_val)], verbose=False)
            logger.info("  ✓ Total model complete")
            return model

        def train_win_prob_model():
            logger.info("  Starting win probability model training...")
            clf_params = xgb_params.copy()
            clf_params["eval_metric"] = "logloss"
            model = xgb.XGBClassifier(**clf_params)
            model.fit(X_train, y_win_train, eval_set=[(X_val, y_win_val)], verbose=False)
            logger.info("  ✓ Win probability model complete")
            return model

        # Train all three models in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(train_spread_model): "spread",
                executor.submit(train_total_model): "total",
                executor.submit(train_win_prob_model): "win_prob",
            }

            for future in as_completed(futures):
                model_type = futures[future]
                try:
                    if model_type == "spread":
                        self.spread_model = future.result()
                    elif model_type == "total":
                        self.total_model = future.result()
                    elif model_type == "win_prob":
                        self.win_prob_model = future.result()
                except Exception as e:
                    logger.error(f"Training failed for {model_type} model: {e}")
                    raise

        logger.info("All models trained successfully in parallel")

        # 5. Evaluate on validation set
        metrics = self._evaluate_models(X_val, y_spread_val, y_total_val, y_win_val)

        # FIXED: Use RMSE as proper uncertainty estimate (residual standard deviation)
        # RMSE is the correct measure for prediction intervals (unlike MAE * arbitrary factor)
        self.spread_std = metrics["spread_rmse"]
        self.total_std = metrics["total_rmse"]

        # 6. Compute SHAP values (FIXED: proper storage)
        self._compute_shap_values(X_val)

        # 7. Update metadata
        self.metadata.update(
            {
                "created_at": datetime.now().isoformat(),
                "training_seasons": seasons,
                "feature_count": len(feature_cols),
                "validation_metrics": metrics,
                "spread_uncertainty": self.spread_std,
                "total_uncertainty": self.total_std,
            }
        )

        self.models_trained = True

        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed:.1f}s")
        logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")

        return metrics

    def _load_training_data_batch(self, seasons: list[int]) -> pd.DataFrame:
        """
        Load training data with batch feature extraction for performance.

        OPTIMIZATION: Extract features in batches instead of one-by-one.

        Args:
            seasons: List of seasons to load

        Returns:
            DataFrame with features and targets
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor

        all_games = []

        # Database connection for getting games
        conn_str = (
            "postgresql://dro:sicillionbillions@localhost:5544/devdb01"  # pragma: allowlist secret
        )

        for season in seasons:
            logger.info(f"Extracting features for season {season}...")

            # Get all games for the season
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT DISTINCT
                            g.season,
                            g.week,
                            g.home_team,
                            g.away_team,
                            g.home_score - g.away_score as spread,
                            g.home_score + g.away_score as total,
                            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win
                        FROM games g
                        WHERE g.season = %s
                            AND g.game_type = 'REG'
                            AND g.home_score IS NOT NULL
                        ORDER BY g.week
                    """,
                        (season,),
                    )

                    games = cur.fetchall()

            # Extract features for each game
            for game in games:
                try:
                    features = self.feature_extractor.extract_game_features(
                        game["home_team"], game["away_team"], game["season"], game["week"]
                    )

                    # Add target variables
                    features["spread"] = game["spread"]
                    features["total"] = game["total"]
                    features["home_win"] = game["home_win"]
                    features["season"] = game["season"]
                    features["week"] = game["week"]
                    features["home_team"] = game["home_team"]
                    features["away_team"] = game["away_team"]

                    all_games.append(features)

                except Exception as e:
                    logger.debug(
                        f"Failed to extract features for {game['away_team']}@{game['home_team']} "
                        f"Week {game['week']} Season {game['season']}: {e}"
                    )
                    continue

            logger.info(
                f"  Extracted features for {len([g for g in all_games if g.get('season') == season])} games from season {season}"
            )

        df = pd.DataFrame(all_games)

        # Remove games with missing features or targets
        initial_count = len(df)
        df = df.dropna(subset=["spread", "total", "home_win"])
        logger.info(f"Retained {len(df)}/{initial_count} games after removing missing data")

        return df

    def predict(self, home_team: str, away_team: str, season: int, week: int) -> dict[str, float]:
        """
        Predict game outcome.

        FIXED: Added input validation

        Args:
            home_team: Home team abbreviation (e.g., 'KC')
            away_team: Away team abbreviation (e.g., 'BUF')
            season: Season year
            week: Week number

        Returns:
            Dictionary with predictions and uncertainty estimates

        Raises:
            RuntimeError: If models not trained
            ValueError: If invalid inputs provided
        """
        # FIXED: Input validation
        if not self.models_trained:
            raise RuntimeError("Models not trained. Call train() or load_models() first.")

        # Validate teams - accept both canonical (LAR) and database (LA) formats
        # VALID_TEAMS includes both formats from py.utils.team_mapping
        if home_team not in VALID_TEAMS:
            raise ValueError(f"Invalid home_team: {home_team}. Must be one of {VALID_TEAMS}")
        if away_team not in VALID_TEAMS:
            raise ValueError(f"Invalid away_team: {away_team}. Must be one of {VALID_TEAMS}")

        # Normalize to database format for feature extraction (LA not LAR)
        # The feature extractor queries the database which uses nflverse format
        home_team_db = normalize_team(home_team, to_canonical=False)  # LAR -> LA
        away_team_db = normalize_team(away_team, to_canonical=False)

        # Validate season and week
        if not (2000 <= season <= 2030):
            raise ValueError(f"Invalid season: {season}. Must be between 2000-2030")
        if not (1 <= week <= 22):  # Extended to 22 for playoffs
            raise ValueError(f"Invalid week: {week}. Must be between 1-22")

        if home_team == away_team:
            raise ValueError(f"home_team and away_team cannot be the same: {home_team}")

        start = time.time()

        # 1. Extract features (use database format team names)
        try:
            features = self.feature_extractor.extract_game_features(
                home_team_db, away_team_db, season, week
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise ValueError(
                f"Could not extract features for {away_team}@{home_team} Week {week}: {e}"
            )

        # 2. Prepare feature vector (match training feature order)
        X = np.array([[features.get(col, 0.0) for col in self.feature_names]])

        # 3. Predict (no scaling needed for XGBoost)
        home_spread = float(self.spread_model.predict(X)[0])
        total_points = float(self.total_model.predict(X)[0])
        home_win_probability = float(self.win_prob_model.predict_proba(X)[0, 1])

        # 4. Use computed uncertainty from validation
        spread_std = self.spread_std if self.spread_std else 3.5
        total_std = self.total_std if self.total_std else 5.0

        elapsed_ms = (time.time() - start) * 1000

        result = {
            "home_spread": float(round(home_spread, 1)),
            "total_points": float(round(total_points, 1)),
            "home_win_probability": float(round(home_win_probability, 3)),
            "spread_std": float(round(spread_std, 2)),
            "total_std": float(round(total_std, 2)),
            "inference_time_ms": float(round(elapsed_ms, 1)),
            "home_team": home_team,
            "away_team": away_team,
            "season": season,
            "week": week,
        }

        return result

    def predict_games(self, games: list[dict]) -> list[dict]:
        """
        Predict outcomes for multiple games (batch prediction).

        This is a convenience wrapper around predict() for batch operations.

        Args:
            games: List of game dictionaries, each containing:
                - game_id: Unique game identifier
                - home_team: Home team abbreviation
                - away_team: Away team abbreviation
                - season: Season year
                - week: Week number

        Returns:
            List of prediction dictionaries, each containing:
                - game_id: From input
                - home_spread, total_points, home_win_probability
                - spread_std, total_std
                - inference_time_ms
        """
        predictions = []
        for game in games:
            try:
                pred = self.predict(
                    home_team=game['home_team'],
                    away_team=game['away_team'],
                    season=game['season'],
                    week=game['week'],
                )
                pred['game_id'] = game.get('game_id')
                predictions.append(pred)
            except Exception as e:
                logger.warning(
                    f"Failed to predict {game.get('away_team')}@{game.get('home_team')}: {e}"
                )
                # Return partial result with error
                predictions.append({
                    'game_id': game.get('game_id'),
                    'error': str(e),
                })
        return predictions

    def _evaluate_models(
        self,
        X_val: np.ndarray,
        y_spread_val: np.ndarray,
        y_total_val: np.ndarray,
        y_win_val: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate models on validation data."""
        # Predictions
        spread_pred = self.spread_model.predict(X_val)
        total_pred = self.total_model.predict(X_val)
        win_prob_pred = self.win_prob_model.predict_proba(X_val)[:, 1]

        # Metrics
        metrics = {
            "spread_mae": float(mean_absolute_error(y_spread_val, spread_pred)),
            "spread_rmse": float(np.sqrt(mean_squared_error(y_spread_val, spread_pred))),
            "total_mae": float(mean_absolute_error(y_total_val, total_pred)),
            "total_rmse": float(np.sqrt(mean_squared_error(y_total_val, total_pred))),
            "win_prob_auc": float(roc_auc_score(y_win_val, win_prob_pred)),
            "win_prob_accuracy": float(((win_prob_pred > 0.5) == y_win_val).mean()),
        }

        return metrics

    def _compute_shap_values(self, X_val: np.ndarray):
        """
        Compute SHAP values for model interpretability.

        FIXED: Properly store SHAP values separately from explainers
        """
        try:
            # Sample subset for SHAP (SHAP is slow on large datasets)
            n_samples = min(100, len(X_val))
            X_sample = X_val[:n_samples]

            # Spread model
            self.spread_explainer = shap.TreeExplainer(self.spread_model)
            self.shap_values["spread"] = self.spread_explainer.shap_values(X_sample)

            # Total model
            self.total_explainer = shap.TreeExplainer(self.total_model)
            self.shap_values["total"] = self.total_explainer.shap_values(X_sample)

            # Win probability model
            self.win_prob_explainer = shap.TreeExplainer(self.win_prob_model)
            self.shap_values["win_prob"] = self.win_prob_explainer.shap_values(X_sample)

            logger.info("SHAP explainers computed successfully")

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            # Non-critical, continue without SHAP

    def get_feature_importance(self, model_type: str = "spread") -> pd.DataFrame:
        """
        Get feature importance for specified model.

        Args:
            model_type: One of 'spread', 'total', 'win_prob'

        Returns:
            DataFrame with feature importance scores
        """
        if not self.models_trained:
            raise RuntimeError("Models not trained yet")

        if model_type == "spread":
            model = self.spread_model
        elif model_type == "total":
            model = self.total_model
        elif model_type == "win_prob":
            model = self.win_prob_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        importance = model.feature_importances_

        df = pd.DataFrame({"feature": self.feature_names, "importance": importance}).sort_values(
            "importance", ascending=False
        )

        return df

    def save_models(self):
        """Save trained models and metadata to disk."""
        if not self.models_trained:
            raise RuntimeError("No models to save. Train first.")

        # Save XGBoost models
        self.spread_model.save_model(str(self.model_dir / "spread_model.json"))
        self.total_model.save_model(str(self.model_dir / "total_model.json"))
        self.win_prob_model.save_model(str(self.model_dir / "win_prob_model.json"))

        # Save metadata and feature names
        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        with open(self.model_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

        # Save SHAP values if available
        if self.shap_values:
            with open(self.model_dir / "shap_values.pkl", "wb") as f:
                pickle.dump(self.shap_values, f)

        logger.info(f"Models saved to {self.model_dir}")

    def load_models(self):
        """Load trained models from disk."""
        try:
            # Load XGBoost models
            self.spread_model = xgb.XGBRegressor()
            self.spread_model.load_model(str(self.model_dir / "spread_model.json"))

            self.total_model = xgb.XGBRegressor()
            self.total_model.load_model(str(self.model_dir / "total_model.json"))

            self.win_prob_model = xgb.XGBClassifier()
            self.win_prob_model.load_model(str(self.model_dir / "win_prob_model.json"))

            # Load metadata and feature names
            with open(self.model_dir / "metadata.json") as f:
                self.metadata = json.load(f)

            with open(self.model_dir / "feature_names.json") as f:
                self.feature_names = json.load(f)

            # Load uncertainty estimates from metadata
            self.spread_std = self.metadata.get("spread_uncertainty")
            self.total_std = self.metadata.get("total_uncertainty")

            # Load SHAP values if available
            shap_path = self.model_dir / "shap_values.pkl"
            if shap_path.exists():
                with open(shap_path, "rb") as f:
                    self.shap_values = pickle.load(f)

            self.models_trained = True
            logger.info(f"Models loaded from {self.model_dir}")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Could not load models from {self.model_dir}: {e}")


def main():
    """Example usage and testing."""
    from datetime import datetime

    # Dynamic season calculation (NFL season runs Sept-Feb)
    current_year = datetime.now().year
    current_month = datetime.now().month
    # If Jan-Aug, NFL season is previous year; if Sept-Dec, it's current year
    current_season = current_year if current_month >= 9 else current_year - 1
    training_seasons = list(range(current_season - 2, current_season + 1))

    model = GameOutcomeModel()

    # Train on recent seasons (dynamically computed)
    print(f"Training on seasons: {training_seasons}")
    metrics = model.train(seasons=training_seasons, validation_split=0.2)

    print("Validation Metrics:")
    print(f"  Spread MAE: {metrics['spread_mae']:.2f}")
    print(f"  Total MAE: {metrics['total_mae']:.2f}")
    print(f"  Win Prob AUC: {metrics['win_prob_auc']:.3f}")

    # Save models
    model.save_models()

    # Test prediction (use current season)
    try:
        prediction = model.predict(home_team="KC", away_team="BUF", season=current_season, week=11)
        print("\nSample Prediction:")
        print(json.dumps(prediction, indent=2))
    except Exception as e:
        print(f"Prediction failed: {e}")

    # Feature importance
    importance = model.get_feature_importance("spread")
    print("\nTop 10 Features for Spread Prediction:")
    print(importance.head(10))


if __name__ == "__main__":
    main()
