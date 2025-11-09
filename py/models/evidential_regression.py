#!/usr/bin/env python3
"""
Evidential Deep Learning for NFL Player Props Betting

Implements evidential regression using Normal Inverse-Gamma (NIG) distribution
for single forward-pass uncertainty quantification.

Key Features:
- Single forward pass (50× faster than MC dropout)
- Explicit epistemic uncertainty (ν parameter - high for OOD data)
- Explicit aleatoric uncertainty (α, β parameters - heteroscedastic noise)
- No external dependencies (custom implementation)

Theory:
--------
Evidential learning predicts parameters of a Normal Inverse-Gamma distribution:
    p(y|μ,σ²) ~ N(μ, σ²)
    p(σ²|α,β) ~ InvGamma(α, β)

Network outputs 4 values:
    γ (gamma): Mean prediction (μ)
    ν (nu): Epistemic uncertainty - higher for out-of-distribution data
    α (alpha): Shape of inverse-gamma (aleatoric uncertainty)
    β (beta): Scale of inverse-gamma (aleatoric uncertainty)

Expected variance:
    Aleatoric: β / (α - 1)  [data noise, irreducible]
    Epistemic: β / (ν(α-1))  [model uncertainty, reducible with more data]
    Total: β/(α-1) * (1 + 1/ν)

Loss Function (Amini et al. 2020):
    NLL = 0.5*log(π/ν) - α*log(2β(1+ν))
          + (α+0.5)*log(ν(y-γ)² + 2β(1+ν))
          + loggamma(α) - loggamma(α+0.5)

    Regularization: λ * |y-γ| * (2ν + α)

    Total Loss: NLL + λ * Reg

References:
-----------
- Amini et al. (2020): "Deep Evidential Regression" (NeurIPS)
- Sensoy et al. (2018): "Evidential Deep Learning to Quantify Classification Uncertainty"
- Jøsang (2016): "Subjective Logic" (book on belief theory)

Author: Claude (Anthropic)
Date: November 9, 2025
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import psycopg2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidentialNeuralNetwork(nn.Module):
    """
    Evidential Neural Network for regression.

    Architecture:
    - Input layer
    - Hidden layers with ReLU activation
    - Output layer with 4 heads (γ, ν, α, β)
    - Softplus activation for positive constraints on ν, α, β
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.2
    ):
        """
        Initialize evidential neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output heads for NIG parameters
        # γ (gamma): mean prediction (unconstrained)
        self.gamma_head = nn.Linear(prev_dim, 1)

        # ν (nu): epistemic uncertainty (positive)
        # Evidence: high ν = low epistemic uncertainty (confident)
        self.nu_head = nn.Linear(prev_dim, 1)

        # α (alpha): shape parameter for inverse-gamma (positive, >1)
        self.alpha_head = nn.Linear(prev_dim, 1)

        # β (beta): scale parameter for inverse-gamma (positive)
        self.beta_head = nn.Linear(prev_dim, 1)

        # Initialize output heads
        self._init_output_heads()

    def _init_output_heads(self):
        """Initialize output heads with reasonable defaults"""
        # γ (mean): small random initialization
        nn.init.xavier_normal_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        # ν (evidence): initialize to produce moderate evidence (~10)
        nn.init.xavier_normal_(self.nu_head.weight)
        nn.init.constant_(self.nu_head.bias, 2.0)  # softplus(2) ≈ 2.1

        # α (shape): initialize to produce α > 1 (required for valid variance)
        nn.init.xavier_normal_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, 1.5)  # softplus(1.5) ≈ 1.7

        # β (scale): initialize to produce moderate scale
        nn.init.xavier_normal_(self.beta_head.weight)
        nn.init.constant_(self.beta_head.bias, 1.0)  # softplus(1) ≈ 1.3

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict NIG parameters.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            gamma: Mean prediction (batch_size,)
            nu: Epistemic uncertainty evidence (batch_size,)
            alpha: Inverse-gamma shape (batch_size,)
            beta: Inverse-gamma scale (batch_size,)
        """
        # Hidden layers
        h = self.hidden_layers(x)

        # Output heads
        gamma = self.gamma_head(h).squeeze(-1)  # Unconstrained mean

        # Positive constraints using softplus
        # Add small epsilon for numerical stability
        eps = 1e-6
        nu = F.softplus(self.nu_head(h)).squeeze(-1) + eps
        alpha = F.softplus(self.alpha_head(h)).squeeze(-1) + 1.0 + eps  # α > 1
        beta = F.softplus(self.beta_head(h)).squeeze(-1) + eps

        return gamma, nu, alpha, beta

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with aleatoric and epistemic uncertainty.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Dictionary with:
                - mean: Expected prediction γ
                - aleatoric_std: Aleatoric (data) uncertainty
                - epistemic_std: Epistemic (model) uncertainty
                - total_std: Total uncertainty
                - alpha, beta, nu: NIG parameters
        """
        gamma, nu, alpha, beta = self.forward(x)

        # Aleatoric uncertainty: σ²_aleatoric = β/(α-1)
        aleatoric_var = beta / (alpha - 1)
        aleatoric_std = torch.sqrt(aleatoric_var)

        # Epistemic uncertainty: σ²_epistemic = β/(ν(α-1))
        epistemic_var = beta / (nu * (alpha - 1))
        epistemic_std = torch.sqrt(epistemic_var)

        # Total uncertainty: σ²_total = β/(α-1) * (1 + 1/ν)
        total_var = aleatoric_var * (1 + 1/nu)
        total_std = torch.sqrt(total_var)

        return {
            'mean': gamma,
            'aleatoric_std': aleatoric_std,
            'epistemic_std': epistemic_std,
            'total_std': total_std,
            'alpha': alpha,
            'beta': beta,
            'nu': nu
        }


class EvidentialLoss(nn.Module):
    """
    Normal Inverse-Gamma (NIG) loss for evidential regression.

    Implements the loss function from Amini et al. (2020):
        L = NLL + λ * Regularization

    Where:
        NLL: Negative log-likelihood of NIG distribution
        Regularization: Penalty for overconfident predictions
    """

    def __init__(self, lambda_reg: float = 0.01, epsilon: float = 1e-8):
        """
        Initialize NIG loss.

        Args:
            lambda_reg: Regularization strength (default: 0.01)
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon

    def forward(
        self,
        y_true: torch.Tensor,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NIG loss.

        Args:
            y_true: True target values (batch_size,)
            gamma: Predicted means (batch_size,)
            nu: Evidence parameters (batch_size,)
            alpha: Inverse-gamma shape (batch_size,)
            beta: Inverse-gamma scale (batch_size,)

        Returns:
            Loss value (scalar)
        """
        # Compute negative log-likelihood
        nll = self._nig_nll(y_true, gamma, nu, alpha, beta)

        # Compute regularization term
        # Penalizes confident predictions when error is large
        error = torch.abs(y_true - gamma)
        reg = error * (2 * nu + alpha)

        # Total loss
        loss = nll + self.lambda_reg * reg

        return loss.mean()

    def _nig_nll(
        self,
        y_true: torch.Tensor,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood of Normal Inverse-Gamma distribution.

        NLL = 0.5*log(π/ν) - α*log(2β(1+ν))
              + (α+0.5)*log(ν(y-γ)² + 2β(1+ν))
              + loggamma(α) - loggamma(α+0.5)

        Args:
            y_true: True targets
            gamma: Predicted means
            nu: Evidence
            alpha: Shape
            beta: Scale

        Returns:
            NLL values (batch_size,)
        """
        # Add epsilon for numerical stability
        eps = self.epsilon

        # Term 1: 0.5 * log(π/ν)
        term1 = 0.5 * torch.log(torch.tensor(np.pi) / (nu + eps))

        # Term 2: -α * log(2β(1+ν))
        term2 = -alpha * torch.log(2 * beta * (1 + nu) + eps)

        # Term 3: (α+0.5) * log(ν(y-γ)² + 2β(1+ν))
        error_squared = (y_true - gamma) ** 2
        term3 = (alpha + 0.5) * torch.log(
            nu * error_squared + 2 * beta * (1 + nu) + eps
        )

        # Term 4: loggamma(α) - loggamma(α+0.5)
        # Use torch.lgamma for numerical stability
        term4 = torch.lgamma(alpha + eps) - torch.lgamma(alpha + 0.5 + eps)

        nll = term1 + term2 + term3 + term4

        return nll


class EvidentialRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible evidential regressor.

    Wraps EvidentialNeuralNetwork for easy integration with
    existing ML pipelines and backtesting infrastructure.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.2,
        lambda_reg: float = 0.01,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        device: str = 'cpu',
        random_state: Optional[int] = None
    ):
        """
        Initialize evidential regressor.

        Args:
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
            lambda_reg: Regularization strength
            learning_rate: Learning rate for Adam optimizer
            batch_size: Training batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            device: 'cpu' or 'cuda'
            random_state: Random seed
        """
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state

        # Internal state
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.is_fitted_ = False
        self.training_history_ = []

        # Set random seeds
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Fit evidential regressor.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Print training progress

        Returns:
            self
        """
        # Normalize features
        X_scaled = self.scaler_.fit_transform(X)

        # Initialize model
        input_dim = X.shape[1]
        self.model_ = EvidentialNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Loss and optimizer
        criterion = EvidentialLoss(lambda_reg=self.lambda_reg)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=verbose
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.FloatTensor(y)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Validation loader (if provided)
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_.transform(X_val)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training phase
            self.model_.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                gamma, nu, alpha, beta = self.model_(batch_X)

                # Compute loss
                loss = criterion(batch_y, gamma, nu, alpha, beta)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        gamma, nu, alpha, beta = self.model_(batch_X)
                        loss = criterion(batch_y, gamma, nu, alpha, beta)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                # Log progress
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.max_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                self.training_history_.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.max_epochs} - "
                        f"Train Loss: {train_loss:.4f}"
                    )

                self.training_history_.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss
                })

        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)
            return_uncertainty: If True, return dictionary with uncertainty

        Returns:
            predictions: Mean predictions (n_samples,)
                        or dict with uncertainty if return_uncertainty=True
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            if return_uncertainty:
                predictions = self.model_.predict_with_uncertainty(X_tensor)
                # Convert to numpy
                return {
                    k: v.cpu().numpy() for k, v in predictions.items()
                }
            else:
                gamma, _, _, _ = self.model_(X_tensor)
                return gamma.cpu().numpy()

    def save(self, filepath: str):
        """Save model to disk"""
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_data = {
            'model_state': self.model_.state_dict(),
            'scaler': self.scaler_,
            'config': {
                'hidden_dims': self.hidden_dims,
                'dropout_rate': self.dropout_rate,
                'lambda_reg': self.lambda_reg,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'patience': self.patience
            },
            'training_history': self.training_history_,
            'input_dim': self.model_.input_dim,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(model_data, filepath)

        # Save metadata as JSON
        metadata = {
            'architecture': 'evidential_regression',
            'hidden_dims': list(self.hidden_dims),
            'dropout_rate': self.dropout_rate,
            'lambda_reg': self.lambda_reg,
            'n_epochs_trained': len(self.training_history_),
            'final_train_loss': self.training_history_[-1]['train_loss'],
            'timestamp': datetime.now().isoformat()
        }

        if 'val_loss' in self.training_history_[-1]:
            metadata['final_val_loss'] = self.training_history_[-1]['val_loss']

        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")
        logger.info(f"Metadata saved to {metadata_path}")

    def load(self, filepath: str):
        """Load model from disk"""
        filepath = Path(filepath)

        # Load model data
        model_data = torch.load(filepath, map_location=self.device)

        # Restore config
        config = model_data['config']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout_rate']
        self.lambda_reg = config['lambda_reg']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.max_epochs = config['max_epochs']
        self.patience = config['patience']

        # Restore scaler
        self.scaler_ = model_data['scaler']

        # Restore model
        input_dim = model_data['input_dim']
        self.model_ = EvidentialNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        self.model_.load_state_dict(model_data['model_state'])

        # Restore training history
        self.training_history_ = model_data['training_history']

        self.is_fitted_ = True
        logger.info(f"Model loaded from {filepath}")

        return self


def load_rushing_data_from_db(
    start_season: int = 2020,
    end_season: int = 2024
) -> pd.DataFrame:
    """
    Load rushing yards data from database.

    Args:
        start_season: Start season (inclusive)
        end_season: End season (inclusive)

    Returns:
        DataFrame with rushing data
    """
    logger.info(f"Loading rushing data from seasons {start_season}-{end_season}")

    conn = psycopg2.connect(
        host='localhost',
        port=5544,
        dbname='devdb01',
        user='dro',
        password='sicillionbillions'
    )

    query = """
    SELECT
        pgs.player_id,
        pgs.player_display_name as player_name,
        pgs.season,
        pgs.week,
        pgs.current_team as team,
        pgs.position_group as position,
        pgs.stat_yards as rushing_yards,
        pgs.stat_attempts as carries,
        pgs.stat_touchdowns as rushing_tds,
        -- Rolling averages (lag 3 games)
        AVG(pgs.stat_yards) OVER (
            PARTITION BY pgs.player_id
            ORDER BY pgs.season, pgs.week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as avg_rushing_l3,
        AVG(pgs.stat_attempts) OVER (
            PARTITION BY pgs.player_id
            ORDER BY pgs.season, pgs.week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as avg_carries_l3,
        -- Season average
        AVG(pgs.stat_yards) OVER (
            PARTITION BY pgs.player_id, pgs.season
            ORDER BY pgs.week
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) as season_avg
    FROM mart.player_game_stats pgs
    WHERE pgs.season BETWEEN %s AND %s
      AND pgs.stat_category = 'rushing'
      AND pgs.position_group IN ('RB', 'FB', 'HB')
      AND pgs.stat_attempts >= 5
      AND pgs.stat_yards IS NOT NULL
    ORDER BY pgs.season, pgs.week, pgs.stat_yards DESC
    """

    df = pd.read_sql(query, conn, params=[start_season, end_season])
    conn.close()

    # Handle missing values
    df['avg_rushing_l3'] = df['avg_rushing_l3'].fillna(
        df.groupby('season')['rushing_yards'].transform('median')
    )
    df['avg_carries_l3'] = df['avg_carries_l3'].fillna(
        df.groupby('season')['carries'].transform('median')
    )
    df['season_avg'] = df['season_avg'].fillna(
        df.groupby('season')['rushing_yards'].transform('median')
    )

    logger.info(f"Loaded {len(df)} rushing performances from {df['player_id'].nunique()} players")

    return df


if __name__ == "__main__":
    # Quick demo
    logger.info("=" * 70)
    logger.info("EVIDENTIAL DEEP LEARNING - QUICK DEMO")
    logger.info("=" * 70)

    # Load data
    df = load_rushing_data_from_db(start_season=2020, end_season=2024)

    # Prepare features
    feature_cols = ['carries', 'avg_rushing_l3', 'avg_carries_l3', 'season_avg']
    X = df[feature_cols].values
    y = df['rushing_yards'].values

    # Train/test split
    train_mask = df['season'] <= 2023
    test_mask = df['season'] == 2024

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(f"\nTrain: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")

    # Train model
    logger.info("\nTraining evidential regressor...")
    model = EvidentialRegressor(
        hidden_dims=(128, 64),
        dropout_rate=0.2,
        lambda_reg=0.01,
        learning_rate=1e-3,
        batch_size=128,
        max_epochs=50,
        patience=5,
        device='cpu'
    )

    model.fit(X_train, y_train, verbose=True)

    # Predict with uncertainty
    logger.info("\nMaking predictions with uncertainty...")
    predictions = model.predict(X_test, return_uncertainty=True)

    # Evaluate
    mae = np.mean(np.abs(predictions['mean'] - y_test))
    rmse = np.sqrt(np.mean((predictions['mean'] - y_test) ** 2))

    logger.info(f"\nTest MAE: {mae:.2f} yards")
    logger.info(f"Test RMSE: {rmse:.2f} yards")
    logger.info(f"Mean aleatoric uncertainty: {predictions['aleatoric_std'].mean():.2f} yards")
    logger.info(f"Mean epistemic uncertainty: {predictions['epistemic_std'].mean():.2f} yards")
    logger.info(f"Mean total uncertainty: {predictions['total_std'].mean():.2f} yards")

    logger.info("\n✓ Demo complete")
