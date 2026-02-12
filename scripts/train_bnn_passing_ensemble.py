#!/usr/bin/env python3
"""
Train BNN for Passing Yards Ensemble Integration

Quick training script for passing yards BNN compatible with ensemble.
Uses same 19-feature set as XGBoost for consistency.

Training time: ~5 minutes on CPU (Mac M4)
Output: models/bayesian/bnn_passing_ensemble_v1.pt

Author: Claude Code
Date: November 7, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty"""

    def __init__(self, in_features, out_features, prior_var=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in")
        nn.init.constant_(self.weight_rho, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x, sample=True):
        if sample:
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * weight_epsilon

            bias_epsilon = torch.randn_like(self.bias_mu)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """KL divergence from posterior to prior"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        kl_weight = 0.5 * torch.sum(
            weight_sigma**2 / self.prior_var
            + self.weight_mu**2 / self.prior_var
            - torch.log(weight_sigma**2 / self.prior_var)
            - 1
        )

        kl_bias = 0.5 * torch.sum(
            bias_sigma**2 / self.prior_var
            + self.bias_mu**2 / self.prior_var
            - torch.log(bias_sigma**2 / self.prior_var)
            - 1
        )

        return kl_weight + kl_bias


class PassingYardsBNN(nn.Module):
    """Bayesian Neural Network for Passing Yards Prediction"""

    def __init__(self, input_dim=19, hidden_dim=64, dropout_rate=0.2):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Output: yards prediction (mean and log_std)
        self.yards_mean = BayesianLinear(hidden_dim, 1)
        self.yards_log_std = BayesianLinear(hidden_dim, 1)

    def forward(self, x, sample=True):
        h = F.relu(self.fc1(x, sample))
        h = self.dropout(h)
        h = F.relu(self.fc2(h, sample))
        h = self.dropout(h)

        mean = self.yards_mean(h, sample).squeeze(-1)
        log_std = self.yards_log_std(h, sample).squeeze(-1)

        return mean, log_std

    def kl_divergence(self):
        kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence()
        return kl


def load_training_data(db_url, feature_cols):
    """Load passing yards training data with same features as XGBoost"""

    conn = psycopg.connect(db_url)

    # Query player game stats
    query = """
    WITH player_stats AS (
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as yards,
            pgs.stat_attempts as attempts,
            pgs.stat_completions as completions,
            pgs.stat_touchdowns as tds,
            pgs.stat_negative as interceptions,
            pgs.cpoe,
            ph.current_team as team,

            -- Game context
            g.home_team,
            g.away_team,
            CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
            g.spread_close as spread,
            g.total_close / 2.0 as implied_team_total

        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON (
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
        )
        WHERE pgs.stat_category = 'passing'
            AND pgs.stat_yards IS NOT NULL
            AND pgs.stat_attempts >= 10
            AND pgs.season BETWEEN 2020 AND 2024
            AND ph.position_group = 'QB'
        ORDER BY pgs.player_id, pgs.season, pgs.week
    )
    SELECT * FROM player_stats
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} QB games from 2020-2024")

    # Compute rolling features (same as feature extractor)
    features_list = []

    for player_id in df["player_id"].unique():
        player_df = df[df["player_id"] == player_id].sort_values(["season", "week"])

        for idx in range(5, len(player_df)):  # Need at least 5 games for features
            row = player_df.iloc[idx]
            prev_games = player_df.iloc[:idx]

            last3 = prev_games.tail(3)
            last5 = prev_games.tail(5)
            season_games = prev_games[prev_games["season"] == row["season"]]

            feature_dict = {
                "yards_target": row["yards"],  # Target
                "passing_yards_last3": last3["yards"].mean(),
                "passing_yards_last5": last5["yards"].mean(),
                "passing_yards_season": (
                    season_games["yards"].mean() if len(season_games) > 0 else last5["yards"].mean()
                ),
                "pass_attempts_last3": last3["attempts"].mean(),
                "pass_attempts_last5": last5["attempts"].mean(),
                "completions_last3": last3["completions"].mean(),
                "completions_last5": last5["completions"].mean(),
                "passing_tds_last3": last3["tds"].mean(),
                "interceptions_last3": last3["interceptions"].mean(),
                "sacks_last3": 2.5,  # Default (not in database)
                "opponent_passing_yards_allowed_avg": 250.0,  # Default (requires opponent query)
                "implied_team_total": (
                    row["implied_team_total"] if pd.notna(row["implied_team_total"]) else 22.0
                ),
                "spread": row["spread"] if pd.notna(row["spread"]) else 0.0,
                "is_home": row["is_home"],
                "weather_wind_mph": 5.0,  # Default
                "weather_temp": 60.0,  # Default
                "days_rest": 7.0,  # Default
                "is_dome": 0.0,  # Default
                "is_outdoors": 1.0,  # Default
            }

            features_list.append(feature_dict)

    features_df = pd.DataFrame(features_list)
    logger.info(f"Created {len(features_df)} training samples with features")

    return features_df


def train_bnn(args):
    """Train BNN model"""

    logger.info("=" * 80)
    logger.info("TRAINING BNN FOR PASSING YARDS ENSEMBLE")
    logger.info("=" * 80)

    # Feature columns (same as XGBoost)
    feature_cols = [
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
        "opponent_passing_yards_allowed_avg",
        "implied_team_total",
        "spread",
        "is_home",
        "weather_wind_mph",
        "weather_temp",
        "days_rest",
        "is_dome",
        "is_outdoors",
    ]

    # Load data
    db_url = f"postgresql://{args.db_user}:{args.db_password}@{args.db_host}:{args.db_port}/{args.db_name}"
    df = load_training_data(db_url, feature_cols)

    # Prepare features and target
    X = df[feature_cols].values
    y = df["yards_target"].values

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features AND target
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    # Standardize target (CRITICAL FIX!)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    logger.info("\nTarget statistics:")
    logger.info(f"  y_train mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    logger.info(
        f"  y_train_scaled mean: {y_train_scaled.mean():.4f}, std: {y_train_scaled.std():.4f}"
    )

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled)  # Use scaled target!
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val_scaled)  # Use scaled target!

    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = PassingYardsBNN(
        input_dim=len(feature_cols), hidden_dim=args.hidden_dim, dropout_rate=args.dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info("\nModel Architecture:")
    logger.info(f"  Input: {len(feature_cols)} features")
    logger.info(f"  Hidden: {args.hidden_dim}")
    logger.info("  Output: yards_mean, yards_log_std")
    logger.info("\nTraining Settings:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  KL scale: {args.kl_scale}")

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            mean, log_std = model(X_batch, sample=True)

            # Negative log likelihood (Gaussian)
            std = torch.exp(log_std)
            nll = 0.5 * torch.log(2 * np.pi * std**2) + 0.5 * ((y_batch - mean) / std) ** 2
            nll_loss = nll.mean()

            # KL divergence
            kl_loss = model.kl_divergence() / len(X_train)

            # Total loss
            loss = nll_loss + args.kl_scale * kl_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae_scaled = 0.0
        val_mae_yards = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                mean, log_std = model(X_batch, sample=False)
                std = torch.exp(log_std)
                nll = 0.5 * torch.log(2 * np.pi * std**2) + 0.5 * ((y_batch - mean) / std) ** 2
                val_loss += nll.mean().item()
                val_mae_scaled += torch.abs(y_batch - mean).mean().item()

                # Denormalize for yards MAE
                mean_yards = mean.numpy() * scaler_y.scale_[0] + scaler_y.mean_[0]
                y_yards = y_batch.numpy() * scaler_y.scale_[0] + scaler_y.mean_[0]
                val_mae_yards += np.abs(y_yards - mean_yards).mean()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae_scaled /= len(val_loader)
        val_mae_yards /= len(val_loader)

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae_yards:.2f} yards"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae_yards

            # Save checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "scaler_X_mean": scaler_X.mean_,
                "scaler_X_scale": scaler_X.scale_,
                "scaler_y_mean": scaler_y.mean_[0],
                "scaler_y_scale": scaler_y.scale_[0],
                "feature_cols": feature_cols,
                "args": vars(args),
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
            }

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)

    logger.info("\n✅ Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation MAE: {best_val_mae:.2f} yards")
    logger.info(f"Model saved: {args.output}")

    return model, scaler_X, scaler_y, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BNN for passing yards ensemble")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5544, help="Database port")
    parser.add_argument("--db-name", default="devdb01", help="Database name")
    parser.add_argument("--db-user", default="dro", help="Database user")
    parser.add_argument("--db-password", default="sicillionbillions", help="Database password")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--kl-scale", type=float, default=0.01, help="KL divergence scale")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--output", default="models/bayesian/bnn_passing_ensemble_v1.pt", help="Output path"
    )

    args = parser.parse_args()

    model, scaler_X, scaler_y, feature_cols = train_bnn(args)
