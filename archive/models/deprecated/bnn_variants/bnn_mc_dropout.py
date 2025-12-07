#!/usr/bin/env python3
"""
MC Dropout BNN - Variational Approximation for Uncertainty Quantification

Method (Gal & Ghahramani 2016):
- Standard NN with dropout layers (p=0.1, 0.2, 0.3)
- Enable dropout at inference time
- Monte Carlo sampling (T=100) gives posterior approximation
- Aleatoric + epistemic uncertainty

Key Advantages:
- Fast training (~5 minutes per architecture)
- No MCMC pathologies
- Proven to work on tabular data
- Dropout = approximate Bayesian inference

Architectures Tested:
1. Deep: [13 → 64 → 32 → 16 → 1] (p=0.1)
2. Wide: [13 → 128 → 64 → 1] (p=0.2)
3. Shallow: [13 → 32 → 1] (p=0.3)

Expected Performance:
- MAE: 330-345 yards
- Calibration: 68-72%
- Training: 5-10 minutes per architecture

Usage:
    python py/models/bnn_mc_dropout.py
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class MCDropoutBNN(nn.Module):
    """
    Bayesian Neural Network using MC Dropout for uncertainty quantification

    Architecture:
    - Multiple hidden layers with ReLU activation
    - Dropout layers (fixed p throughout network)
    - Heteroscedastic: Predict both mean and log(std)

    Inference:
    - Enable dropout during inference
    - Sample T times (default T=100)
    - Aggregate: mean for prediction, std for uncertainty
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int], dropout_p: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout_p

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))
            prev_dim = hidden_dim

        # Output layer: predict mean
        layers.append(nn.Linear(prev_dim, 1))

        self.mean_head = nn.Sequential(*layers)

        # Heteroscedastic: separate head for log(std)
        # Use shallow network to prevent overfitting
        self.log_std_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(prev_dim // 2, 1),
        )

        logger.info(
            f"Initialized MC Dropout BNN: input={input_dim}, hidden={hidden_dims}, "
            f"dropout={dropout_p:.2f}, params={self.count_parameters()}"
        )

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            mean: Predicted mean
            log_std: Predicted log(std) for heteroscedastic noise
        """
        # Get features from last hidden layer
        features = x
        for layer in self.mean_head[:-1]:  # All except output layer
            features = layer(features)

        # Mean prediction
        mean = self.mean_head[-1](features)

        # Log-std prediction (ensure std > 0)
        log_std = self.log_std_head(features)

        return mean.squeeze(), log_std.squeeze()

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        MC Dropout inference for uncertainty quantification

        Args:
            x: Input features [batch_size, input_dim]
            n_samples: Number of MC samples (default: 100)

        Returns:
            pred_mean: Mean predictions [batch_size]
            pred_std: Predictive standard deviation [batch_size]
        """
        self.train()  # Enable dropout at inference

        mean_samples = []
        var_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                mean, log_std = self.forward(x)
                mean_samples.append(mean.cpu().numpy())
                var_samples.append(torch.exp(2 * log_std).cpu().numpy())

        # Aggregate samples
        mean_samples = np.array(mean_samples)  # [n_samples, batch_size]
        var_samples = np.array(var_samples)

        # Predictive mean: average of means
        pred_mean = mean_samples.mean(axis=0)

        # Predictive variance: epistemic + aleatoric
        epistemic_var = mean_samples.var(axis=0)  # Model uncertainty
        aleatoric_var = var_samples.mean(axis=0)  # Data uncertainty
        pred_var = epistemic_var + aleatoric_var
        pred_std = np.sqrt(pred_var)

        return pred_mean, pred_std


def load_passing_data_from_db():
    """Load passing yards data from database"""
    conn = psycopg2.connect(
        host="localhost", port=5544, user="dro", password="sicillionbillions", database="devdb01"
    )

    query = """
    SELECT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.stat_yards as yards,
        pgs.stat_touchdowns as touchdowns,
        pgs.stat_attempts as attempts,
        pgs.current_team as team,
        g.home_score,
        g.away_score,
        g.spread_close as spread_line,
        g.total_close as total_line,
        g.temp,
        g.wind,
        CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    LEFT JOIN games g ON
      g.season = pgs.season
      AND g.week = pgs.week
      AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
    WHERE pgs.stat_category = 'passing'
      AND pgs.stat_yards IS NOT NULL
      AND pgs.stat_yards > 0
      AND pgs.stat_attempts >= 10
      AND pgs.season >= 2015
      AND pgs.season <= 2024
      AND ph.position_group = 'QB'
    ORDER BY pgs.season, pgs.week, pgs.player_id
    """

    logger.info("Loading passing data from database...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} QB games from {df.season.min()}-{df.season.max()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create BNN input features"""
    logger.info("Engineering features...")

    # Derive efficiency metrics from attempts
    df["log_attempts"] = np.log1p(df["attempts"])
    df["yards_per_attempt"] = df["yards"] / df["attempts"].replace(0, np.nan)
    df["td_rate"] = df["touchdowns"] / df["attempts"].replace(0, np.nan)

    # Game context
    df["spread_abs"] = df["spread_line"].abs()
    df["log_yards"] = np.log1p(df["yards"])

    # Convert weather columns to numeric
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")

    df["is_bad_weather"] = ((df["temp"] < 32) | (df["wind"] > 15)).fillna(False).astype(int)

    # Rolling averages (per player)
    df = df.sort_values(["player_id", "season", "week"])

    for col in ["yards", "yards_per_attempt", "td_rate"]:
        df[f"{col}_L3"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .shift(1)
        )

    # Season progress
    df["week_in_season"] = df["week"].astype(float)

    # Fill NaNs
    df = df.fillna(df.median(numeric_only=True))

    logger.info(f"Feature engineering complete: {len(df)} samples")
    return df


def prepare_train_val_test_split(df: pd.DataFrame):
    """Split data by season"""
    train_seasons = [2020, 2021, 2022]  # Training
    val_season = 2023  # Validation
    test_season = 2024  # Test

    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"] == val_season].copy()
    test_df = df[df["season"] == test_season].copy()

    # Feature columns
    feature_cols = [
        "log_attempts",
        "yards_per_attempt",
        "td_rate",
        "is_home",
        "spread_abs",
        "total_line",
        "is_bad_weather",
        "yards_L3",
        "yards_per_attempt_L3",
        "td_rate_L3",
        "week_in_season",
        "temp",
        "wind",
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df["yards"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["yards"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["yards"].values

    logger.info(f"Train: {len(X_train)} samples ({train_seasons})")
    logger.info(f"Val: {len(X_val)} samples ({val_season})")
    logger.info(f"Test: {len(X_test)} samples ({test_season})")
    logger.info(f"Features: {len(feature_cols)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """Compute calibration: fraction within ±1σ (target: ~68%)"""
    z_scores = np.abs((y_true - y_pred) / y_std)
    calibration = np.mean(z_scores < 1.0)
    return calibration


def train_mc_dropout_bnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: tuple[int],
    dropout_p: float = 0.1,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[MCDropoutBNN, StandardScaler, float, float]:
    """
    Train MC Dropout BNN

    Returns:
        model: Trained MC Dropout BNN
        scaler: Fitted StandardScaler
        y_mean: Target mean (for denormalization)
        y_std: Target std (for denormalization)
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Normalize target
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train_norm).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.FloatTensor(y_val_norm).to(device)

    # Initialize model
    model = MCDropoutBNN(
        input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout_p=dropout_p
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    logger.info(f"Training MC Dropout BNN (epochs={epochs}, batch_size={batch_size})...")

    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        indices = np.random.permutation(len(X_train_t))
        train_loss = 0.0

        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i : i + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()

            # Forward pass
            mean, log_std = model(X_batch)
            std = torch.exp(log_std)

            # Negative log-likelihood loss (Gaussian)
            loss = 0.5 * torch.mean(torch.log(2 * np.pi * std**2) + ((y_batch - mean) / std) ** 2)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(X_train_t) // batch_size

        # Validation
        model.eval()
        with torch.no_grad():
            val_mean, val_log_std = model(X_val_t)
            val_std = torch.exp(val_log_std)
            val_loss = (
                0.5
                * torch.mean(
                    torch.log(2 * np.pi * val_std**2) + ((y_val_t - val_mean) / val_std) ** 2
                ).item()
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )

    logger.info(f"✓ Training complete: best_val_loss={best_val_loss:.4f}")

    return model, scaler, y_mean, y_std


def evaluate_model(
    model: MCDropoutBNN,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    y_mean: float,
    y_std: float,
    n_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """Evaluate MC Dropout BNN"""
    # Normalize features
    X_scaled = scaler.transform(X)
    X_t = torch.FloatTensor(X_scaled).to(device)

    # MC Dropout inference
    pred_mean_norm, pred_std_norm = model.predict_with_uncertainty(X_t, n_samples=n_samples)

    # Denormalize predictions
    pred_mean = pred_mean_norm * y_std + y_mean
    pred_std = pred_std_norm * y_std  # Scale uncertainty

    # Metrics
    mae = np.mean(np.abs(pred_mean - y))
    rmse = np.sqrt(np.mean((pred_mean - y) ** 2))
    calibration = compute_calibration(y, pred_mean, pred_std)

    return {"mae": mae, "rmse": rmse, "calibration": calibration, "mean_pred_std": pred_std.mean()}


def main():
    """Main MC Dropout BNN training pipeline"""
    logger.info("=" * 80)
    logger.info("MC DROPOUT BNN - Comprehensive Evaluation")
    logger.info("=" * 80 + "\n")

    # Load and prepare data
    df = load_passing_data_from_db()
    df = engineer_features(df)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_train_val_test_split(df)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}\n")

    # Configurations to test
    configs = [
        {"name": "Deep", "hidden_dims": (64, 32, 16), "dropout_p": 0.1},
        {"name": "Wide", "hidden_dims": (128, 64), "dropout_p": 0.2},
        {"name": "Shallow", "hidden_dims": (32,), "dropout_p": 0.3},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"CONFIG {i}/{len(configs)}: {config['name']}")
        logger.info(f"  Hidden: {config['hidden_dims']}")
        logger.info(f"  Dropout: {config['dropout_p']:.2f}")
        logger.info(f"{'='*80}\n")

        # Train model
        model, scaler, y_mean, y_std = train_mc_dropout_bnn(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_dims=config["hidden_dims"],
            dropout_p=config["dropout_p"],
            epochs=100,
            batch_size=64,
            lr=1e-3,
            device=device,
        )

        # Evaluate on validation set
        logger.info("\nValidation Set (2023):")
        val_metrics = evaluate_model(model, scaler, X_val, y_val, y_mean, y_std, device=device)
        logger.info(f"  MAE: {val_metrics['mae']:.2f} yards")
        logger.info(f"  RMSE: {val_metrics['rmse']:.2f} yards")
        logger.info(f"  Calibration: {val_metrics['calibration']:.1%} (target: 68%)")
        logger.info(f"  Mean Uncertainty: {val_metrics['mean_pred_std']:.2f} yards")

        # Evaluate on test set
        logger.info("\nTest Set (2024):")
        test_metrics = evaluate_model(model, scaler, X_test, y_test, y_mean, y_std, device=device)
        logger.info(f"  MAE: {test_metrics['mae']:.2f} yards")
        logger.info(f"  RMSE: {test_metrics['rmse']:.2f} yards")
        logger.info(f"  Calibration: {test_metrics['calibration']:.1%} (target: 68%)")
        logger.info(f"  Mean Uncertainty: {test_metrics['mean_pred_std']:.2f} yards")

        # Save model
        model_path = Path(f"models/bayesian/mc_dropout_{config['name'].lower()}_v1.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "y_mean": y_mean,
            "y_std": y_std,
            "hidden_dims": config["hidden_dims"],
            "dropout_p": config["dropout_p"],
            "input_dim": X_train.shape[1],
        }

        joblib.dump(save_dict, model_path)
        logger.info(f"\n✓ Model saved to {model_path}")

        # Store results
        results.append(
            {
                "config_name": config["name"],
                "hidden_dims": str(config["hidden_dims"]),
                "dropout_p": config["dropout_p"],
                "val_mae": val_metrics["mae"],
                "val_calibration": val_metrics["calibration"],
                "val_uncertainty": val_metrics["mean_pred_std"],
                "test_mae": test_metrics["mae"],
                "test_calibration": test_metrics["calibration"],
                "test_uncertainty": test_metrics["mean_pred_std"],
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    results_path = Path("analysis/results/mc_dropout_bnn_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    logger.info(f"\n{'='*80}")
    logger.info("MC DROPOUT BNN - SUMMARY")
    logger.info(f"{'='*80}\n")
    logger.info(results_df.to_string(index=False))
    logger.info(f"\n✓ Results saved to {results_path}")

    # Identify best model
    best_idx = results_df["test_mae"].idxmin()
    best_config = results_df.iloc[best_idx]

    logger.info(f"\n{'='*80}")
    logger.info("BEST MODEL")
    logger.info(f"{'='*80}")
    logger.info(f"Config: {best_config['config_name']}")
    logger.info(f"Test MAE: {best_config['test_mae']:.2f} yards")
    logger.info(f"Test Calibration: {best_config['test_calibration']:.1%}")
    logger.info(f"Test Uncertainty: {best_config['test_uncertainty']:.2f} yards")

    return results_df


if __name__ == "__main__":
    main()
