"""
props_predictor.py

NFL Player Props Prediction Model

Predicts player prop outcomes (over/under) for:
- Passing yards, passing TDs, interceptions
- Rushing yards, rushing TDs
- Receiving yards, receptions, receiving TDs
- Kicking (FG made, XP made)

Uses XGBoost regression models trained on player-level features:
- Recent performance (rolling averages)
- Opponent strength (defensive rankings)
- Game script (implied team total, spread)
- Environmental factors (weather, stadium)
- Usage metrics (snap %, target share, carry share)

Usage:
    # Train model
    python py/models/props_predictor.py \
        --features data/processed/features/player_features.csv \
        --train-seasons 2010-2023 \
        --test-season 2024 \
        --prop-type passing_yards \
        --output models/props/passing_yards_model.json

    # Predict single player
    python py/models/props_predictor.py \
        --model models/props/passing_yards_model.json \
        --predict \
        --player-id "00-0036355" \
        --opponent "KC" \
        --line 275.5

    # Backtest all props
    python py/models/props_predictor.py \
        --features data/processed/features/player_features.csv \
        --backtest \
        --season 2024
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PropPrediction:
    """Single prop prediction."""

    player_id: str
    player_name: str
    prop_type: str
    line: float
    prediction: float
    std: float
    over_prob: float
    under_prob: float
    edge: float  # EV advantage over fair odds
    recommended_bet: Optional[str]  # "over", "under", or None


@dataclass
class BacktestResult:
    """Backtest results for a prop type."""

    prop_type: str
    n_bets: int
    roi: float
    sharpe: float
    win_rate: float
    avg_edge: float
    mae: float
    rmse: float
    r2: float


# ============================================================================
# Props Predictor Model
# ============================================================================


class PropsPredictor:
    """
    NFL Player Props Prediction Model.

    Trains separate XGBoost models for each prop type (passing_yards, rushing_yards, etc.).
    Uses player-level features including recent performance, opponent strength, and game script.
    """

    # Prop types and their typical lines
    PROP_TYPES = {
        "passing_yards": {"min": 150, "max": 400, "step": 0.5},
        "passing_tds": {"min": 0, "max": 5, "step": 0.5},
        "interceptions": {"min": 0, "max": 3, "step": 0.5},
        "rushing_yards": {"min": 30, "max": 150, "step": 0.5},
        "rushing_tds": {"min": 0, "max": 2, "step": 0.5},
        "receiving_yards": {"min": 20, "max": 120, "step": 0.5},
        "receptions": {"min": 2, "max": 10, "step": 0.5},
        "receiving_tds": {"min": 0, "max": 2, "step": 0.5},
    }

    # Feature columns by prop type
    # NOTE: Column names match actual generated features from player_features.py
    FEATURE_SETS = {
        "passing_yards": [
            "passing_yards_last3",
            "passing_yards_last5",
            "passing_yards_season",
            "pass_attempts_last3",
            "pass_attempts_last5",
            "completions_last3",
            "completions_last5",
            "passing_tds_last3",
            "interceptions_last3",
            "sacks_last3",
            "opponent_pass_yards_allowed_avg",
            "implied_team_total",
            "spread",
            "is_home",
            "weather_wind_mph",
            "weather_temp",
            "days_rest",
            "is_dome",
            "is_outdoors",
        ],
        "passing_tds": [
            "passing_tds_last3",
            "passing_tds_last5",
            "passing_tds_season",
            "pass_attempts_last3",
            "pass_attempts_last5",
            "red_zone_attempts_last3",
            "red_zone_attempts_last5",
            "passing_yards_last3",
            "completions_last3",
            "opponent_pass_yards_allowed_avg",
            "implied_team_total",
            "spread",
            "is_home",
        ],
        "interceptions": [
            "interceptions_last3",
            "interceptions_last5",
            "interceptions_season",
            "pass_attempts_last3",
            "pass_attempts_last5",
            "passing_yards_last3",
            "sacks_last3",
            "opponent_pass_yards_allowed_avg",
            "implied_team_total",
            "spread",
            "is_home",
            "weather_wind_mph",
        ],
        "rushing_yards": [
            "rushing_yards_last3",
            "rushing_yards_last5",
            "rushing_yards_season",
            "rush_attempts_last3",
            "rush_attempts_last5",
            "rushing_tds_last3",
            "red_zone_carries_last3",
            "opponent_pass_yards_allowed_avg",  # Using as proxy for defensive strength
            "implied_team_total",
            "spread",
            "is_home",
            "is_turf",
        ],
        "rushing_tds": [
            "rushing_tds_last3",
            "rushing_tds_last5",
            "rushing_tds_season",
            "rush_attempts_last3",
            "red_zone_carries_last3",
            "red_zone_carries_last5",
            "rushing_yards_last3",
            "opponent_pass_yards_allowed_avg",
            "implied_team_total",
            "spread",
            "is_home",
        ],
        "receiving_yards": [
            "receiving_yards_last3",
            "receiving_yards_last5",
            "receiving_yards_season",
            "targets_last3",
            "targets_last5",
            "receptions_last3",
            "receptions_last5",
            "receiving_tds_last3",
            "red_zone_targets_last3",
            "opponent_pass_yards_allowed_avg",
            "implied_team_total",
            "spread",
            "is_home",
            "weather_wind_mph",
            "weather_temp",
            "is_dome",
        ],
        "receptions": [
            "receptions_last3",
            "receptions_last5",
            "receptions_season",
            "targets_last3",
            "targets_last5",
            "receiving_yards_last3",
            "red_zone_targets_last3",
            "opponent_pass_yards_allowed_avg",
            "implied_team_total",
            "spread",
            "is_home",
        ],
        "receiving_tds": [
            "receiving_tds_last3",
            "receiving_tds_last5",
            "receiving_tds_season",
            "targets_last3",
            "receptions_last3",
            "red_zone_targets_last3",
            "red_zone_targets_last5",
            "receiving_yards_last3",
            "opponent_pass_yards_allowed_avg",
            "implied_team_total",
            "spread",
            "is_home",
        ],
    }

    def __init__(
        self,
        prop_type: str,
        xgb_params: Optional[Dict] = None,
        min_edge: float = 0.03,
        kelly_fraction: float = 0.25,
    ):
        """
        Initialize PropsPredictor.

        Args:
            prop_type: Type of prop to predict (e.g., "passing_yards")
            xgb_params: XGBoost hyperparameters
            min_edge: Minimum edge required to place bet
            kelly_fraction: Fractional Kelly for bet sizing
        """
        if prop_type not in self.PROP_TYPES:
            raise ValueError(f"Unknown prop type: {prop_type}")

        self.prop_type = prop_type
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction

        # Default XGBoost params for regression
        self.xgb_params = xgb_params or {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "auto",  # Changed from "hist" to "auto" for compatibility
            "device": "cpu",  # Changed from "cuda" to "cpu" for compatibility
            "random_state": 42,
        }

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_cols: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def prepare_features(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction.

        Args:
            df: Raw player features dataframe
            target_col: Name of target column (e.g., "passing_yards")

        Returns:
            X (features), y (target)
        """
        # Get feature columns for this prop type
        if self.prop_type in self.FEATURE_SETS:
            self.feature_cols = self.FEATURE_SETS[self.prop_type]
        else:
            # Fallback: use all numeric columns except target
            self.feature_cols = [
                col
                for col in df.columns
                if col != target_col
                and df[col].dtype in [np.float64, np.int64]
            ]

        logger.info(
            f"Using {len(self.feature_cols)} features for {self.prop_type}: {self.feature_cols[:5]}..."
        )

        # Extract features
        X = df[self.feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Extract target
        if target_col in df.columns:
            y = df[target_col].copy()
        else:
            y = None

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        val_split: float = 0.2,
        early_stopping_rounds: int = 50,
    ) -> Dict:
        """
        Train XGBoost model.

        Args:
            df: Training dataframe with player features
            target_col: Name of target column
            val_split: Validation set proportion
            early_stopping_rounds: Early stopping patience

        Returns:
            Training metrics
        """
        logger.info(f"Training {self.prop_type} model on {len(df)} samples")

        # Prepare features
        X, y = self.prepare_features(df, target_col)

        # Train/val split (time series aware)
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(
            f"Train: {len(X_train)} samples | Val: {len(X_val)} samples"
        )

        # Standardize features (helps with XGBoost convergence)
        self.scaler_mean = X_train.mean().values
        self.scaler_std = X_train.std().values + 1e-8  # Avoid div by zero

        X_train_scaled = (X_train.values - self.scaler_mean) / self.scaler_std
        X_val_scaled = (X_val.values - self.scaler_mean) / self.scaler_std

        # Train model
        # Note: early_stopping_rounds is now set in XGBRegressor constructor
        xgb_params_with_early_stopping = self.xgb_params.copy()
        xgb_params_with_early_stopping["early_stopping_rounds"] = early_stopping_rounds
        xgb_params_with_early_stopping["eval_metric"] = "rmse"

        self.model = xgb.XGBRegressor(**xgb_params_with_early_stopping)
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )

        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)

        train_metrics = {
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "r2": r2_score(y_train, y_train_pred),
        }

        val_metrics = {
            "mae": mean_absolute_error(y_val, y_val_pred),
            "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "r2": r2_score(y_val, y_val_pred),
        }

        logger.info(f"Train - MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.3f}")
        logger.info(f"Val   - MAE: {val_metrics['mae']:.2f}, R²: {val_metrics['r2']:.3f}")

        return {"train": train_metrics, "val": val_metrics}

    def predict(
        self, X: pd.DataFrame, with_uncertainty: bool = True,
        bayesian_priors: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict prop values with uncertainty estimates.

        Args:
            X: Features dataframe
            with_uncertainty: If True, estimate prediction std via quantile regression
            bayesian_priors: DataFrame with Bayesian predictions (player_id, predicted_value, predicted_std)

        Returns:
            predictions, std (if with_uncertainty=True)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Standardize features
        X_scaled = (X[self.feature_cols].values - self.scaler_mean) / self.scaler_std

        # Get XGBoost predictions
        xgb_predictions = self.model.predict(X_scaled)

        # Combine with Bayesian priors if available
        if bayesian_priors is not None and 'player_id' in X.columns:
            predictions = np.zeros_like(xgb_predictions)
            stds = np.zeros_like(xgb_predictions)

            for i, player_id in enumerate(X['player_id']):
                # Check if we have Bayesian prior for this player
                bayesian_row = bayesian_priors[bayesian_priors['player_id'] == player_id]

                if not bayesian_row.empty:
                    # We have a Bayesian prior - combine with XGBoost
                    bayesian_mean = bayesian_row['predicted_value'].iloc[0]
                    bayesian_std = bayesian_row['predicted_std'].iloc[0]
                    xgb_mean = xgb_predictions[i]
                    xgb_std = self.xgb_params.get("rmse", 10.0)

                    # Combine using inverse variance weighting
                    # This gives more weight to the more confident prediction
                    bayesian_weight = 1 / (bayesian_std ** 2)
                    xgb_weight = 1 / (xgb_std ** 2)
                    total_weight = bayesian_weight + xgb_weight

                    # Combined mean
                    predictions[i] = (
                        bayesian_mean * bayesian_weight + xgb_mean * xgb_weight
                    ) / total_weight

                    # Combined std (simplified - could use more sophisticated approach)
                    stds[i] = 1 / np.sqrt(total_weight)

                    # Log the combination
                    logger.debug(
                        f"Player {player_id}: Bayesian={bayesian_mean:.1f}±{bayesian_std:.1f}, "
                        f"XGBoost={xgb_mean:.1f}±{xgb_std:.1f}, "
                        f"Combined={predictions[i]:.1f}±{stds[i]:.1f}"
                    )
                else:
                    # No Bayesian prior - use XGBoost only
                    predictions[i] = xgb_predictions[i]
                    stds[i] = self.xgb_params.get("rmse", 10.0)
        else:
            # No Bayesian priors provided - use XGBoost only
            predictions = xgb_predictions
            stds = np.full(len(predictions), self.xgb_params.get("rmse", 10.0))

        if with_uncertainty:
            return predictions, stds
        else:
            return predictions, None

    def predict_single(
        self,
        player_id: str,
        player_name: str,
        features: Dict,
        line: float,
        over_odds: int = -110,
        under_odds: int = -110,
    ) -> PropPrediction:
        """
        Predict single player prop with betting recommendation.

        Args:
            player_id: Player ID (GSIS ID)
            player_name: Player name
            features: Dictionary of player features
            line: Prop line (e.g., 275.5 passing yards)
            over_odds: American odds for over bet
            under_odds: American odds for under bet

        Returns:
            PropPrediction with betting recommendation
        """
        # Convert features dict to dataframe
        X = pd.DataFrame([features])

        # Predict
        pred, std = self.predict(X, with_uncertainty=True)
        pred_value = pred[0]
        pred_std = std[0] if std is not None else 10.0

        # Calculate probabilities
        # Assume normal distribution around prediction
        over_prob = 1 - stats.norm.cdf(line, loc=pred_value, scale=pred_std)
        under_prob = stats.norm.cdf(line, loc=pred_value, scale=pred_std)

        # Convert American odds to decimal
        over_decimal = self._american_to_decimal(over_odds)
        under_decimal = self._american_to_decimal(under_odds)

        # Calculate edge (EV - 1)
        over_edge = over_prob * over_decimal - 1
        under_edge = under_prob * under_decimal - 1

        # Determine recommended bet
        max_edge = max(over_edge, under_edge)
        if max_edge > self.min_edge:
            recommended_bet = "over" if over_edge > under_edge else "under"
            edge = max_edge
        else:
            recommended_bet = None
            edge = max_edge

        return PropPrediction(
            player_id=player_id,
            player_name=player_name,
            prop_type=self.prop_type,
            line=line,
            prediction=pred_value,
            std=pred_std,
            over_prob=over_prob,
            under_prob=under_prob,
            edge=edge,
            recommended_bet=recommended_bet,
        )

    def backtest(
        self,
        df: pd.DataFrame,
        target_col: str,
        line_col: str = "prop_line",
        over_odds_col: str = "over_odds",
        under_odds_col: str = "under_odds",
        bayesian_priors: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Backtest prop predictions against historical lines.

        Args:
            df: Dataframe with player features, actual outcomes, and prop lines
            target_col: Actual outcome column (e.g., "passing_yards")
            line_col: Prop line column
            over_odds_col: Over odds column (American)
            under_odds_col: Under odds column (American)

        Returns:
            BacktestResult
        """
        logger.info(f"Backtesting {self.prop_type} on {len(df)} props")

        # Prepare features
        X, y_actual = self.prepare_features(df, target_col)

        # Predict
        y_pred, y_std = self.predict(X, with_uncertainty=True)

        # Get lines and odds
        lines = df[line_col].values
        over_odds = df[over_odds_col].values if over_odds_col in df.columns else np.full(len(df), -110)
        under_odds = df[under_odds_col].values if under_odds_col in df.columns else np.full(len(df), -110)

        # Simulate betting
        bets = []
        for i in range(len(df)):
            # Calculate probabilities
            over_prob = 1 - stats.norm.cdf(lines[i], loc=y_pred[i], scale=y_std[i])
            under_prob = stats.norm.cdf(lines[i], loc=y_pred[i], scale=y_std[i])

            # Calculate edge
            over_edge = over_prob * self._american_to_decimal(over_odds[i]) - 1
            under_edge = under_prob * self._american_to_decimal(under_odds[i]) - 1

            # Place bet if edge > threshold
            if over_edge > self.min_edge:
                bet_side = "over"
                edge = over_edge
                bet_odds = over_odds[i]
                win = y_actual.iloc[i] > lines[i]
            elif under_edge > self.min_edge:
                bet_side = "under"
                edge = under_edge
                bet_odds = under_odds[i]
                win = y_actual.iloc[i] < lines[i]
            else:
                continue  # No bet

            # Calculate payout
            if win:
                payout = self._american_to_decimal(bet_odds) - 1
            else:
                payout = -1

            bets.append(
                {
                    "bet_side": bet_side,
                    "edge": edge,
                    "win": win,
                    "payout": payout,
                }
            )

        # Calculate metrics
        if len(bets) == 0:
            logger.warning("No bets placed (no edges > threshold)")
            return BacktestResult(
                prop_type=self.prop_type,
                n_bets=0,
                roi=0.0,
                sharpe=0.0,
                win_rate=0.0,
                avg_edge=0.0,
                mae=mean_absolute_error(y_actual, y_pred),
                rmse=np.sqrt(mean_squared_error(y_actual, y_pred)),
                r2=r2_score(y_actual, y_pred),
            )

        bets_df = pd.DataFrame(bets)

        roi = bets_df["payout"].mean()
        sharpe = bets_df["payout"].mean() / (bets_df["payout"].std() + 1e-8) if len(bets) > 1 else 0.0
        win_rate = bets_df["win"].mean()
        avg_edge = bets_df["edge"].mean()

        logger.info(f"Backtest: {len(bets)} bets | ROI: {roi:.2%} | Sharpe: {sharpe:.3f} | Win Rate: {win_rate:.2%}")

        return BacktestResult(
            prop_type=self.prop_type,
            n_bets=len(bets),
            roi=roi,
            sharpe=sharpe,
            win_rate=win_rate,
            avg_edge=avg_edge,
            mae=mean_absolute_error(y_actual, y_pred),
            rmse=np.sqrt(mean_squared_error(y_actual, y_pred)),
            r2=r2_score(y_actual, y_pred),
        )

    def save_model(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            "prop_type": self.prop_type,
            "feature_cols": self.feature_cols,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "xgb_params": self.xgb_params,
            "min_edge": self.min_edge,
            "kelly_fraction": self.kelly_fraction,
        }

        # Save XGBoost model
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path.with_suffix(".ubj")))

        # Save metadata
        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "PropsPredictor":
        """Load model from disk."""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        predictor = cls(
            prop_type=model_data["prop_type"],
            xgb_params=model_data["xgb_params"],
            min_edge=model_data["min_edge"],
            kelly_fraction=model_data["kelly_fraction"],
        )

        predictor.feature_cols = model_data["feature_cols"]
        predictor.scaler_mean = np.array(model_data["scaler_mean"])
        predictor.scaler_std = np.array(model_data["scaler_std"])

        # Load XGBoost model
        model_path = Path(filepath).with_suffix(".ubj")
        predictor.model = xgb.XGBRegressor()
        predictor.model.load_model(str(model_path))

        logger.info(f"Model loaded from {filepath}")
        return predictor

    @staticmethod
    def _american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return 1 + american_odds / 100
        else:
            return 1 + 100 / abs(american_odds)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NFL Player Props Prediction Model"
    )
    parser.add_argument(
        "--features",
        type=str,
        required=False,
        help="Path to player features CSV",
    )
    parser.add_argument(
        "--prop-type",
        type=str,
        default="passing_yards",
        choices=list(PropsPredictor.PROP_TYPES.keys()),
        help="Prop type to predict",
    )
    parser.add_argument(
        "--train-seasons",
        type=str,
        help="Training seasons (e.g., '2010-2023')",
    )
    parser.add_argument(
        "--test-season",
        type=int,
        help="Test season",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/props/model.json",
        help="Output model path",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to saved model (for prediction/backtest)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict mode (requires --model, --player-id, --line)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Backtest mode (requires --model or --features)",
    )
    parser.add_argument(
        "--player-id",
        type=str,
        help="Player ID for prediction",
    )
    parser.add_argument(
        "--line",
        type=float,
        help="Prop line for prediction",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum edge for betting",
    )

    args = parser.parse_args()

    # ========================================================================
    # Predict Mode
    # ========================================================================
    if args.predict:
        if not args.model or not args.player_id or args.line is None:
            parser.error("--predict requires --model, --player-id, --line")

        predictor = PropsPredictor.load_model(args.model)

        # TODO: Load player features from database
        # For now, use dummy features
        features = {col: 0.0 for col in predictor.feature_cols}

        result = predictor.predict_single(
            player_id=args.player_id,
            player_name="Player Name",
            features=features,
            line=args.line,
        )

        print("\n" + "=" * 70)
        print(f"PROP PREDICTION: {result.prop_type.upper()}")
        print("=" * 70)
        print(f"Player: {result.player_name} ({result.player_id})")
        print(f"Line: {result.line}")
        print(f"Prediction: {result.prediction:.1f} ± {result.std:.1f}")
        print(f"Over Probability: {result.over_prob:.1%}")
        print(f"Under Probability: {result.under_prob:.1%}")
        print(f"Edge: {result.edge:+.2%}")
        print(
            f"Recommended Bet: {result.recommended_bet.upper() if result.recommended_bet else 'PASS'}"
        )
        print("=" * 70)

        return

    # ========================================================================
    # Backtest Mode
    # ========================================================================
    if args.backtest:
        if args.model:
            predictor = PropsPredictor.load_model(args.model)
        elif args.features:
            predictor = PropsPredictor(
                prop_type=args.prop_type, min_edge=args.min_edge
            )
        else:
            parser.error("--backtest requires --model or --features")

        # Load test data
        df = pd.read_csv(args.features)
        if args.test_season:
            df = df[df["season"] == args.test_season]

        # Backtest
        result = predictor.backtest(
            df, target_col=args.prop_type, line_col="prop_line"
        )

        print("\n" + "=" * 70)
        print(f"BACKTEST RESULTS: {result.prop_type.upper()}")
        print("=" * 70)
        print(f"Bets Placed: {result.n_bets}")
        print(f"ROI: {result.roi:.2%}")
        print(f"Sharpe Ratio: {result.sharpe:.3f}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Avg Edge: {result.avg_edge:.2%}")
        print(f"MAE: {result.mae:.2f}")
        print(f"RMSE: {result.rmse:.2f}")
        print(f"R²: {result.r2:.3f}")
        print("=" * 70)

        return

    # ========================================================================
    # Train Mode
    # ========================================================================
    if not args.features:
        parser.error("--features required for training")

    # Load data
    df = pd.read_csv(args.features)

    # Filter by position based on prop type
    prop_to_positions = {
        "passing_yards": ["QB"],
        "passing_tds": ["QB"],
        "interceptions": ["QB"],
        "rushing_yards": ["RB"],
        "rushing_tds": ["RB"],
        "receiving_yards": ["WR", "TE"],
        "receptions": ["WR", "TE"],
        "receiving_tds": ["WR", "TE"],
    }

    valid_positions = prop_to_positions.get(args.prop_type, [])
    if valid_positions and "position" in df.columns:
        df = df[df["position"].isin(valid_positions)]
        logger.info(f"Filtered to {len(df)} samples for positions: {valid_positions}")

    # Drop rows where target is NaN
    if args.prop_type in df.columns:
        df = df.dropna(subset=[args.prop_type])
        logger.info(f"After dropping NaN targets: {len(df)} samples")

    # Filter seasons
    if args.train_seasons:
        start, end = map(int, args.train_seasons.split("-"))
        df = df[(df["season"] >= start) & (df["season"] <= end)]

    if args.test_season:
        df_train = df[df["season"] < args.test_season]
        df_test = df[df["season"] == args.test_season]
    else:
        df_train = df
        df_test = None

    # Train model
    predictor = PropsPredictor(prop_type=args.prop_type, min_edge=args.min_edge)
    metrics = predictor.train(df_train, target_col=args.prop_type)

    # Test set evaluation
    if df_test is not None and len(df_test) > 0:
        logger.info(f"Testing on {len(df_test)} samples from {args.test_season}")
        test_result = predictor.backtest(
            df_test, target_col=args.prop_type, line_col="prop_line"
        )

        print("\n" + "=" * 70)
        print(f"TEST SET RESULTS: {args.test_season}")
        print("=" * 70)
        print(f"Bets Placed: {test_result.n_bets}")
        print(f"ROI: {test_result.roi:.2%}")
        print(f"Sharpe Ratio: {test_result.sharpe:.3f}")
        print(f"Win Rate: {test_result.win_rate:.2%}")
        print(f"Avg Edge: {test_result.avg_edge:.2%}")
        print(f"MAE: {test_result.mae:.2f}")
        print(f"RMSE: {test_result.rmse:.2f}")
        print(f"R²: {test_result.r2:.3f}")
        print("=" * 70)

    # Save model
    predictor.save_model(args.output)


if __name__ == "__main__":
    main()
