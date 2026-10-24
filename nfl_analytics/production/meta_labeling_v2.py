#!/usr/bin/env python3
"""
Enhanced Meta-Labeling Classifier v2 - Bet Quality Prediction
==============================================================

Implements comprehensive meta-labeling approach with 27 features across 6 categories.
Based on López de Prado (2018) "Advances in Financial Machine Learning" with NFL-specific
enhancements.

**Architecture:**
- 27 meta-features across 6 categories (signals, market, historical, uncertainty, temporal, team)
- XGBoost classifier with Platt calibration
- Walk-forward training methodology
- Kelly criterion integration via meta_multiplier

**Feature Categories:**
1. Primary Signals (6): confidence, edge, agreement, variance, sharpe, concentration
2. Market Context (6): primetime, divisional, spread_magnitude, line_movement, public_pct, steam
3. Historical Performance (6): accuracy_last_10, recent_roi, win_streak, calibration_error, ...
4. Uncertainty Quantification (4): prediction_std, epistemic_uncertainty, aleatoric_uncertainty, ...
5. Temporal Factors (3): days_rest_diff, travel_distance, season_phase
6. Team Context (2): injury_load_diff, weather_severity

**Integration Formula:**
```python
final_bet = kelly_fraction × meta_multiplier × bankroll
# meta_prob < 0.4 → multiplier = 0.0 (no bet)
# meta_prob 0.4-0.6 → linear interpolation
# meta_prob > 0.6 → multiplier = 1.0 (full Kelly)
```

**Expected Impact:**
- Reduce bet volume by 40-50% (filter low-quality bets)
- Increase win rate by 4-6% (higher quality bets)
- Improve ROI by 3-5% (fewer losing bets, better sizing)

Author: Claude Code (Opus 4.5)
Date: December 11, 2025
Reference: López de Prado (2018), Chapter 3
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Bumped when the training-label definition changes; checked in load_model()
LABEL_CONVENTION = "nflverse-spread-v1"

warnings.filterwarnings("ignore")


# ============================================================================
# Meta-Feature Extractor
# ============================================================================


class MetaFeatureExtractor:
    """
    Extracts 27 meta-features from ensemble predictions and game context.

    Features are grouped into 6 categories for interpretability and
    systematic feature engineering.
    """

    # Feature names organized by category
    PRIMARY_SIGNALS = [
        "primary_confidence",  # abs(prob - 0.5) * 2
        "primary_edge",  # predicted - vegas
        "ensemble_agreement",  # % components agreeing
        "prediction_variance",  # variance across components
        "ensemble_sharpe",  # mean edge / std edge
        "weight_concentration",  # Herfindahl index of ensemble weights
    ]

    MARKET_CONTEXT = [
        "is_primetime",  # TNF/SNF/MNF flag
        "is_divisional",  # Divisional game flag
        "spread_magnitude",  # abs(spread) - large spreads riskier
        "line_movement",  # close - open (steam indicator)
        "public_betting_pct",  # % of bets on favorite (if available)
        "steam_move",  # Rapid line movement flag
    ]

    HISTORICAL_PERFORMANCE = [
        "model_accuracy_last_10",  # Rolling 10-game accuracy
        "recent_roi",  # ROI over last 20 bets
        "recent_win_streak",  # Current win/loss streak
        "calibration_error",  # Brier score over recent predictions
        "model_sharpe_ratio",  # Risk-adjusted recent performance
        "edge_realization_rate",  # How often predicted edge realizes
    ]

    UNCERTAINTY_QUANTIFICATION = [
        "prediction_std",  # Standard deviation of ensemble
        "epistemic_uncertainty",  # Model uncertainty (knowledge gaps)
        "aleatoric_uncertainty",  # Inherent randomness
        "confidence_interval_width",  # 95% CI width
    ]

    TEMPORAL_FACTORS = [
        "days_rest_diff",  # Home rest - away rest
        "travel_distance",  # Miles traveled by away team
        "season_phase",  # Early (0-5), mid (6-12), late (13+), playoffs
    ]

    TEAM_CONTEXT = [
        "injury_load_diff",  # Home injury load - away
        "weather_severity",  # Composite weather score (wind, precip, temp)
    ]

    # All features combined
    ALL_FEATURES = (
        PRIMARY_SIGNALS
        + MARKET_CONTEXT
        + HISTORICAL_PERFORMANCE
        + UNCERTAINTY_QUANTIFICATION
        + TEMPORAL_FACTORS
        + TEAM_CONTEXT
    )

    def __init__(self):
        """Initialize meta-feature extractor."""
        logger.info(f"MetaFeatureExtractor initialized with {len(self.ALL_FEATURES)} features")

    def extract_from_prediction(
        self,
        prediction: dict[str, Any],
        game_features: dict[str, float],
        historical_metrics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Extract meta-features from ensemble prediction and game context.

        Args:
            prediction: Ensemble prediction dict with keys:
                - predicted_spread: Ensemble mean prediction
                - probability: Win probability
                - vegas_spread: Vegas closing spread
                - component_predictions: Dict of individual model predictions
                - component_weights: Dict of ensemble weights
                - uncertainty: Overall uncertainty estimate
                - spread_open: Opening spread (for line movement)
            game_features: Game features from GameFeatureExtractor
            historical_metrics: Recent model performance metrics

        Returns:
            Dictionary of 27 meta-features
        """
        if historical_metrics is None:
            historical_metrics = {}

        meta_features = {}

        # === PRIMARY SIGNALS ===
        prob = prediction.get("probability", 0.5)
        meta_features["primary_confidence"] = abs(prob - 0.5) * 2  # Scale to [0, 1]

        predicted = prediction.get("predicted_spread", 0)
        vegas = prediction.get("vegas_spread", 0)
        meta_features["primary_edge"] = predicted - vegas

        # Ensemble agreement
        components = prediction.get("component_predictions", {})
        if components:
            predicted_direction = np.sign(predicted + vegas)
            agreeing = sum(
                1 for p in components.values() if np.sign(p + vegas) == predicted_direction
            )
            meta_features["ensemble_agreement"] = agreeing / len(components)

            # Prediction variance
            component_values = list(components.values())
            meta_features["prediction_variance"] = float(np.var(component_values))

            # Ensemble Sharpe (mean edge / std edge)
            edges = [p - vegas for p in component_values]
            edge_std = np.std(edges)
            meta_features["ensemble_sharpe"] = np.mean(edges) / edge_std if edge_std > 0 else 0

            # Weight concentration (Herfindahl index)
            weights = list(prediction.get("component_weights", {}).values())
            if weights:
                meta_features["weight_concentration"] = sum(w**2 for w in weights)
            else:
                meta_features["weight_concentration"] = 0.2  # Default: equal weights
        else:
            # No component info - use defaults
            meta_features["ensemble_agreement"] = 0.5
            meta_features["prediction_variance"] = 0.0
            meta_features["ensemble_sharpe"] = 0.0
            meta_features["weight_concentration"] = 0.2

        # === MARKET CONTEXT ===
        meta_features["is_primetime"] = game_features.get("pt_is_primetime", 0)
        meta_features["is_divisional"] = game_features.get("div_is_divisional", 0)
        meta_features["spread_magnitude"] = abs(vegas)

        # Line movement
        spread_open = prediction.get("spread_open", vegas)
        line_movement = vegas - spread_open
        meta_features["line_movement"] = line_movement

        # Public betting percentage (if available)
        meta_features["public_betting_pct"] = game_features.get("public_betting_pct", 0.5)

        # Steam move detection (line movement > 2 points)
        meta_features["steam_move"] = 1.0 if abs(line_movement) > 2.0 else 0.0

        # === HISTORICAL PERFORMANCE ===
        meta_features["model_accuracy_last_10"] = historical_metrics.get("accuracy_last_10", 0.5)
        meta_features["recent_roi"] = historical_metrics.get("roi_last_20", 0.0)
        meta_features["recent_win_streak"] = historical_metrics.get("win_streak", 0)
        meta_features["calibration_error"] = historical_metrics.get("brier_score", 0.25)
        meta_features["model_sharpe_ratio"] = historical_metrics.get("sharpe_ratio", 0.0)
        meta_features["edge_realization_rate"] = historical_metrics.get("edge_realization", 0.5)

        # === UNCERTAINTY QUANTIFICATION ===
        meta_features["prediction_std"] = prediction.get("uncertainty", 0.0)

        # Epistemic vs aleatoric uncertainty decomposition
        # Epistemic: variance across models (reducible with more data)
        # Aleatoric: inherent randomness (irreducible)
        if components:
            epistemic = float(np.var(component_values))  # Between-model variance
            total_uncertainty = prediction.get("uncertainty", 0.0) ** 2
            aleatoric = max(0, total_uncertainty - epistemic)
            meta_features["epistemic_uncertainty"] = epistemic
            meta_features["aleatoric_uncertainty"] = aleatoric
        else:
            meta_features["epistemic_uncertainty"] = 0.0
            meta_features["aleatoric_uncertainty"] = prediction.get("uncertainty", 0.0) ** 2

        # 95% confidence interval width (approximately 1.96 * std)
        meta_features["confidence_interval_width"] = 1.96 * meta_features["prediction_std"]

        # === TEMPORAL FACTORS ===
        meta_features["days_rest_diff"] = game_features.get("rest_days_diff", 0)
        meta_features["travel_distance"] = game_features.get("travel_distance", 0)

        # Season phase encoding
        week = game_features.get("week", 1)
        if week <= 5:
            season_phase = 0  # Early season
        elif week <= 12:
            season_phase = 1  # Mid season
        elif week <= 18:
            season_phase = 2  # Late season
        else:
            season_phase = 3  # Playoffs
        meta_features["season_phase"] = season_phase

        # === TEAM CONTEXT ===
        meta_features["injury_load_diff"] = game_features.get("injury_load_diff", 0)

        # Weather severity composite
        wind = game_features.get("wind_speed", 0)
        temp = game_features.get("temperature", 70)
        precip = game_features.get("precipitation", 0)
        # Normalized severity: high wind, extreme temp, rain/snow
        weather_severity = (
            (wind / 20.0) + (abs(temp - 70) / 40.0) + (precip / 0.5)
        ) / 3.0  # Average
        meta_features["weather_severity"] = min(1.0, weather_severity)

        # Ensure all features present
        for feat in self.ALL_FEATURES:
            if feat not in meta_features:
                meta_features[feat] = 0.0

        return meta_features

    def extract_from_historical_bet(
        self, bet_record: pd.Series, rolling_metrics: dict[str, float] | None = None
    ) -> dict[str, float]:
        """
        Extract meta-features from historical bet record for training.

        Args:
            bet_record: Row from historical bets DataFrame
            rolling_metrics: Pre-computed rolling metrics

        Returns:
            Dictionary of meta-features
        """
        if rolling_metrics is None:
            rolling_metrics = {}

        # Build prediction dict from bet record
        prediction = {
            "predicted_spread": bet_record.get("predicted_spread", 0),
            "probability": bet_record.get("predicted_probability", 0.5),
            "vegas_spread": bet_record.get("vegas_spread", 0),
            "uncertainty": bet_record.get("prediction_uncertainty", 0),
            "spread_open": bet_record.get("spread_open", bet_record.get("vegas_spread", 0)),
        }

        # Build game features dict from bet record
        game_features = {
            "pt_is_primetime": bet_record.get("is_primetime", 0),
            "div_is_divisional": bet_record.get("is_divisional", 0),
            "week": bet_record.get("week", 1),
            "rest_days_diff": bet_record.get("rest_days_diff", 0),
            "travel_distance": bet_record.get("travel_distance", 0),
            "injury_load_diff": bet_record.get("injury_load_diff", 0),
            "wind_speed": bet_record.get("wind_speed", 0),
            "temperature": bet_record.get("temperature", 70),
            "precipitation": bet_record.get("precipitation", 0),
            "public_betting_pct": bet_record.get("public_betting_pct", 0.5),
        }

        return self.extract_from_prediction(prediction, game_features, rolling_metrics)


# ============================================================================
# Enhanced Meta-Labeling Classifier
# ============================================================================


class MetaLabelingClassifier:
    """
    Production meta-labeling classifier with XGBoost and comprehensive features.

    **Architecture:**
    - Base model: XGBoost (gradient boosting decision trees)
    - Calibration: Platt scaling (sigmoid calibration)
    - Validation: Walk-forward time-series cross-validation
    - Features: 27 meta-features across 6 categories

    **Training Methodology:**
    Walk-forward validation ensures no temporal leakage:
    1. Train on seasons 2020-2021
    2. Validate on season 2022
    3. Train on 2020-2022, validate on 2023
    4. Train on 2020-2023, validate on 2024
    5. Final training on all data for production

    **Integration:**
    ```python
    # At prediction time
    meta_features = extractor.extract_from_prediction(prediction, game_features)
    meta_prob = classifier.predict_probability(meta_features)
    kelly_multiplier = classifier.get_kelly_multiplier(meta_prob)
    final_bet_size = kelly_fraction * kelly_multiplier * bankroll
    ```
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
        model_path: str | None = None,
    ):
        """
        Initialize meta-labeling classifier.

        Args:
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            model_path: Path to saved model (loads if exists)
        """
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

        # Initialize feature extractor
        self.feature_extractor = MetaFeatureExtractor()

        # Initialize XGBoost classifier with NFL-tuned hyperparameters
        self.base_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            scale_pos_weight=1.0,  # Adjust if class imbalance
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )

        # Calibrated wrapper
        self.calibrated_model = None

        # Training metadata
        self.training_metadata = {
            "trained_at": None,
            "n_samples": 0,
            "n_positive": 0,
            "cv_scores": [],
            "feature_importance": {},
            "walk_forward_results": [],
        }

        # Load model if path provided
        self.model_path = model_path
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            logger.info(f"Loaded meta-labeling model from {model_path}")

        logger.info("MetaLabelingClassifier v2 initialized")

    def _connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def train_walk_forward(
        self,
        df: pd.DataFrame,
        start_season: int = 2020,
        end_season: int = 2024,
        calibrate: bool = True,
    ) -> dict[str, Any]:
        """
        Train with walk-forward validation (no temporal leakage).

        Walk-forward methodology:
        - Each fold uses all previous data for training
        - Validates on next season
        - Prevents future information leakage
        - More realistic performance estimates

        Args:
            df: Training DataFrame with meta-features and outcomes
            start_season: First season to use
            end_season: Last season to use
            calibrate: Whether to apply Platt calibration

        Returns:
            Dictionary with validation metrics across folds
        """
        # Sort by season to ensure temporal order
        df = df.sort_values(["season", "week"]).reset_index(drop=True)

        # Prepare features
        available_features = [f for f in self.feature_extractor.ALL_FEATURES if f in df.columns]
        X = df[available_features].values
        y = df["bet_won"].values
        seasons = df["season"].values

        logger.info(
            f"Walk-forward training on {len(X)} samples, {len(available_features)} features"
        )

        # Walk-forward validation
        walk_forward_results = []
        unique_seasons = sorted(df["season"].unique())

        for i, val_season in enumerate(unique_seasons[1:], start=1):
            # Training: all seasons before validation season
            train_mask = seasons < val_season
            val_mask = seasons == val_season

            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]

            if len(y_val) == 0:
                continue

            # Train on this fold
            self.base_model.fit(X_train, y_train)
            y_pred = self.base_model.predict(X_val)
            y_prob = self.base_model.predict_proba(X_val)[:, 1]

            # Compute metrics
            fold_metrics = {
                "val_season": val_season,
                "train_seasons": f"{unique_seasons[0]}-{val_season-1}",
                "n_train": len(y_train),
                "n_val": len(y_val),
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5,
                "brier": brier_score_loss(y_val, y_prob),
                "log_loss": log_loss(y_val, y_prob),
            }
            walk_forward_results.append(fold_metrics)

            logger.info(
                f"Fold {i}: Train {fold_metrics['train_seasons']}, Val {val_season} | "
                f"AUC={fold_metrics['roc_auc']:.3f}, "
                f"Acc={fold_metrics['accuracy']:.3f}, "
                f"Brier={fold_metrics['brier']:.3f}"
            )

        # Final training on all data
        logger.info("Final training on full dataset...")
        self.base_model.fit(X, y)

        # Calibrate probabilities
        if calibrate:
            logger.info("Applying Platt calibration...")
            self.calibrated_model = CalibratedClassifierCV(
                self.base_model, method="sigmoid", cv="prefit"
            )
            self.calibrated_model.fit(X, y)

        # Store metadata
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_positive": int(y.sum()),
            "n_features": len(available_features),
            "features_used": available_features,
            "walk_forward_results": walk_forward_results,
            "feature_importance": dict(
                zip(available_features, self.base_model.feature_importances_.tolist())
            ),
        }

        # Aggregate metrics
        mean_metrics = {}
        if walk_forward_results:
            metric_keys = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier",
                "log_loss",
            ]
            mean_metrics = {
                f"mean_{k}": np.mean([r[k] for r in walk_forward_results]) for k in metric_keys
            }

        # Print summary
        self._print_training_summary()

        return mean_metrics

    def train_time_series_cv(
        self, df: pd.DataFrame, n_splits: int = 5, calibrate: bool = True
    ) -> dict[str, float]:
        """
        Train with standard time-series cross-validation.

        Alternative to walk-forward when you want fixed-size training windows.

        Args:
            df: Training DataFrame
            n_splits: Number of CV folds
            calibrate: Whether to calibrate probabilities

        Returns:
            Mean validation metrics
        """
        available_features = [f for f in self.feature_extractor.ALL_FEATURES if f in df.columns]
        X = df[available_features].values
        y = df["bet_won"].values

        logger.info(f"Time-series CV training: {len(X)} samples, {n_splits} folds")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.base_model.fit(X_train, y_train)
            y_pred = self.base_model.predict(X_val)
            y_prob = self.base_model.predict_proba(X_val)[:, 1]

            fold_metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5,
                "brier": brier_score_loss(y_val, y_prob),
            }
            cv_scores.append(fold_metrics)

            logger.info(
                f"Fold {fold+1}/{n_splits}: "
                f"AUC={fold_metrics['roc_auc']:.3f}, "
                f"Acc={fold_metrics['accuracy']:.3f}"
            )

        # Final training
        self.base_model.fit(X, y)

        if calibrate:
            self.calibrated_model = CalibratedClassifierCV(
                self.base_model, method="sigmoid", cv="prefit"
            )
            self.calibrated_model.fit(X, y)

        # Store metadata
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_positive": int(y.sum()),
            "n_features": len(available_features),
            "features_used": available_features,
            "cv_scores": cv_scores,
            "feature_importance": dict(
                zip(available_features, self.base_model.feature_importances_.tolist())
            ),
        }

        mean_metrics = {k: np.mean([s[k] for s in cv_scores]) for k in cv_scores[0].keys()}

        self._print_training_summary()

        return mean_metrics

    def _print_training_summary(self):
        """Print training summary."""
        logger.info("\n" + "=" * 70)
        logger.info("Meta-Labeling Classifier v2 - Training Complete")
        logger.info("=" * 70)
        logger.info(f"Samples: {self.training_metadata['n_samples']}")
        logger.info(
            f"Positive class: {self.training_metadata['n_positive']} "
            f"({self.training_metadata['n_positive']/self.training_metadata['n_samples']*100:.1f}%)"
        )
        logger.info(f"Features: {self.training_metadata['n_features']}")

        # Top features
        logger.info("\nTop 10 Features by Importance:")
        sorted_features = sorted(
            self.training_metadata["feature_importance"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (feat, imp) in enumerate(sorted_features[:10], 1):
            logger.info(f"  {i:2d}. {feat:30s} {imp:.4f}")

        # Performance metrics
        if "walk_forward_results" in self.training_metadata:
            results = self.training_metadata["walk_forward_results"]
            if results:
                logger.info("\nWalk-Forward Validation Results:")
                logger.info(f"  Mean AUC: {np.mean([r['roc_auc'] for r in results]):.3f}")
                logger.info(f"  Mean Accuracy: {np.mean([r['accuracy'] for r in results]):.3f}")
                logger.info(f"  Mean Brier: {np.mean([r['brier'] for r in results]):.3f}")

    def predict_probability(self, meta_features: dict[str, float]) -> float:
        """
        Predict probability that a bet will be profitable.

        Args:
            meta_features: Dictionary of meta-feature values

        Returns:
            Calibrated probability in [0, 1]
        """
        if self.calibrated_model is None and self.base_model is None:
            logger.warning("Model not trained, returning 0.5")
            return 0.5

        # Prepare feature vector
        features_used = self.training_metadata.get(
            "features_used", self.feature_extractor.ALL_FEATURES
        )
        X = np.array([[meta_features.get(f, 0.0) for f in features_used]])

        # Get calibrated probability
        if self.calibrated_model is not None:
            prob = self.calibrated_model.predict_proba(X)[0, 1]
        else:
            prob = self.base_model.predict_proba(X)[0, 1]

        return float(prob)

    def get_kelly_multiplier(
        self, meta_probability: float, low_threshold: float = 0.4, high_threshold: float = 0.6
    ) -> float:
        """
        Convert meta-probability to Kelly fraction multiplier.

        Integration with Kelly sizing:
        ```python
        kelly_fraction = compute_kelly(edge, odds)
        meta_prob = clf.predict_probability(meta_features)
        kelly_multiplier = clf.get_kelly_multiplier(meta_prob)
        final_bet_size = kelly_fraction * kelly_multiplier * bankroll
        ```

        Args:
            meta_probability: Probability from predict_probability()
            low_threshold: Below this, no bet (return 0)
            high_threshold: Above this, full Kelly (return 1)

        Returns:
            Multiplier in [0, 1] to apply to Kelly fraction
        """
        if meta_probability < low_threshold:
            return 0.0  # No bet - low quality
        elif meta_probability > high_threshold:
            return 1.0  # Full Kelly - high quality
        else:
            # Linear interpolation
            return (meta_probability - low_threshold) / (high_threshold - low_threshold)

    def save_model(self, path: str):
        """Save trained model to disk."""
        model_data = {
            "base_model": self.base_model,
            "calibrated_model": self.calibrated_model,
            "training_metadata": self.training_metadata,
            "label_convention": LABEL_CONVENTION,
            "feature_extractor": self.feature_extractor,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
        logger.info(f"Saved meta-labeling model to {path}")

    def load_model(self, path: str, allow_legacy_labels: bool = False):
        """Load trained model from disk.

        Models saved before the spread sign fix (no ``label_convention``) were trained on
        ``margin + spread`` (sportsbook-rule) labels, which are wrong for nflverse spreads. They are
        refused unless ``allow_legacy_labels=True`` (only safe for total/moneyline models).
        """
        model_data = joblib.load(path)
        if model_data.get("label_convention") != LABEL_CONVENTION and not allow_legacy_labels:
            raise ValueError(
                f"{path} was trained with the pre-fix spread cover labels; retrain it "
                "(py/training/train_meta_classifier.py) or pass allow_legacy_labels=True"
            )
        self.base_model = model_data["base_model"]
        self.calibrated_model = model_data["calibrated_model"]
        self.training_metadata = model_data["training_metadata"]
        if "feature_extractor" in model_data:
            self.feature_extractor = model_data["feature_extractor"]
        logger.info(f"Loaded meta-labeling model from {path}")


# ============================================================================
# Utility Functions
# ============================================================================


def get_feature_names() -> list[str]:
    """Return list of all meta-feature names."""
    return MetaFeatureExtractor.ALL_FEATURES.copy()


def get_feature_categories() -> dict[str, list[str]]:
    """Return dictionary of features grouped by category."""
    return {
        "primary_signals": MetaFeatureExtractor.PRIMARY_SIGNALS,
        "market_context": MetaFeatureExtractor.MARKET_CONTEXT,
        "historical_performance": MetaFeatureExtractor.HISTORICAL_PERFORMANCE,
        "uncertainty": MetaFeatureExtractor.UNCERTAINTY_QUANTIFICATION,
        "temporal": MetaFeatureExtractor.TEMPORAL_FACTORS,
        "team_context": MetaFeatureExtractor.TEAM_CONTEXT,
    }


# ============================================================================
# Demo / Quick Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Meta-Labeling Classifier v2 - Quick Test")
    print("=" * 70 + "\n")

    # Initialize
    extractor = MetaFeatureExtractor()
    classifier = MetaLabelingClassifier()

    # Example prediction
    example_prediction = {
        "predicted_spread": -3.5,
        "probability": 0.65,
        "vegas_spread": -6.0,
        "component_predictions": {
            "hierarchical": -3.2,
            "state_space": -3.8,
            "xgboost": -3.5,
            "informative_priors": -3.6,
            "hybrid": -3.4,
        },
        "component_weights": {
            "hierarchical": 0.27,
            "state_space": 0.27,
            "xgboost": 0.23,
            "informative_priors": 0.13,
            "hybrid": 0.10,
        },
        "uncertainty": 2.5,
        "spread_open": -5.5,
    }

    example_game_features = {
        "pt_is_primetime": 1,
        "div_is_divisional": 0,
        "week": 8,
        "rest_days_diff": 0,
        "travel_distance": 1200,
        "injury_load_diff": -0.05,
        "wind_speed": 5,
        "temperature": 45,
        "precipitation": 0,
        "public_betting_pct": 0.65,
    }

    example_historical = {
        "accuracy_last_10": 0.60,
        "roi_last_20": 0.08,
        "win_streak": 2,
        "brier_score": 0.20,
        "sharpe_ratio": 0.45,
        "edge_realization": 0.55,
    }

    # Extract features
    print("Extracting meta-features...")
    meta_features = extractor.extract_from_prediction(
        example_prediction, example_game_features, example_historical
    )

    print("\nExtracted Features by Category:")
    categories = get_feature_categories()
    for category, features in categories.items():
        print(f"\n{category.upper()}:")
        for feat in features:
            value = meta_features.get(feat, 0.0)
            print(f"  {feat:30s} {value:8.4f}")

    print("\n" + "=" * 70)
    print("Note: Actual prediction requires trained model.")
    print("Run train_meta_classifier.py to train on historical data.")
    print("=" * 70 + "\n")
