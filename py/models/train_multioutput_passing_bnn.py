#!/usr/bin/env python3
"""
Multi-Output Bayesian Neural Network for Passing Yards + TD Probability
Adapted from successful Multi-Output Rushing BNN (92% calibration)

Architecture:
- Mixture of Experts (3 expert networks)
- Joint prediction: passing yards (continuous) + passing TD probability (binary)
- Bayesian variational inference with GPU acceleration (RTX 4090)

Training Window:
- 2018-2024 (train/val split: 2018-2023 train, 2024 test)
- 2015-2017 with H=4 exponential decay weights (canonical window)

Expected Performance:
- Target Calibration: >88% (comparable to temperature-scaled ADVI)
- MAE: <60 yards (competitive with existing models)
- Training Time: 3-4 hours on RTX 4090

Key Insight from Multi-Output Rushing BNN Success:
"Joint modeling of correlated outputs dramatically improved calibration (+65.8pp from
baseline), suggesting that modeling TD probability provides crucial information for
uncertainty quantification of yardage. Touchdown outcomes provide discrete 'regime
indicators' that help the model identify high-variance vs. low-variance game situations."
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty via variational inference"""

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
        """Initialize using He initialization"""
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in")
        nn.init.constant_(self.weight_rho, -3)  # log(exp(0.05) - 1) ≈ -3
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x, sample=True):
        """
        Forward pass with optional stochastic sampling

        Args:
            x: Input tensor
            sample: If True, sample weights; if False, use mean (deterministic)
        """
        if sample:
            # Reparameterization trick: w = μ + σ * ε, ε ~ N(0,1)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * weight_epsilon

            bias_epsilon = torch.randn_like(self.bias_mu)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # Deterministic forward pass (use posterior mean)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """KL divergence from posterior q(w|θ) to prior p(w)"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # KL(q||p) for Gaussian: 0.5 * [σ²/σ₀² + μ²/σ₀² - log(σ²/σ₀²) - 1]
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


class ExpertNetwork(nn.Module):
    """Single expert network for Mixture of Experts architecture"""

    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.2):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Dual output heads
        self.yards_head = BayesianLinear(hidden_dim, 1)  # Passing yards (continuous)
        self.td_head = BayesianLinear(hidden_dim, 1)  # Passing TD (binary)

    def forward(self, x, sample=True):
        h = F.relu(self.fc1(x, sample))
        h = self.dropout(h)
        h = F.relu(self.fc2(h, sample))
        h = self.dropout(h)

        yards = self.yards_head(h, sample)
        td_logits = self.td_head(h, sample)

        return yards.squeeze(-1), td_logits.squeeze(-1)

    def kl_divergence(self):
        """Accumulate KL divergence from all Bayesian layers"""
        kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence()
        return kl


class GatingNetwork(nn.Module):
    """Gating network for Mixture of Experts (softmax weights)"""

    def __init__(self, input_dim, n_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class MultiOutputPassingBNN(nn.Module):
    """
    Multi-Output Mixture of Experts Bayesian Neural Network for Passing Yards

    Key Innovation: Joint modeling of passing yards + TD probability
    - TD outcomes act as "regime indicators" for variance estimation
    - High TD probability games → high variance (explosive plays)
    - Low TD probability games → low variance (conservative gameplan)
    """

    def __init__(self, input_dim, n_experts=3, hidden_dim=32, dropout_rate=0.2):
        super().__init__()
        self.n_experts = n_experts

        # Create expert networks
        self.experts = nn.ModuleList(
            [ExpertNetwork(input_dim, hidden_dim, dropout_rate) for _ in range(n_experts)]
        )

        # Gating network (learns to route inputs to specialists)
        self.gate = GatingNetwork(input_dim, n_experts)

        # Expert-specific heteroscedastic noise parameters
        self.log_noise = nn.Parameter(torch.zeros(n_experts))

    def forward(self, x, sample=True):
        """
        Forward pass with gated mixture of experts

        Returns:
            yards_combined: Weighted average of expert yard predictions
            td_logits_combined: Weighted average of expert TD logits
            weighted_noise: Weighted heteroscedastic uncertainty
        """
        # Get gating weights (batch_size, n_experts)
        gate_weights = self.gate(x)

        # Get predictions from each expert
        yards_preds = []
        td_logits_preds = []

        for expert in self.experts:
            yards, td_logits = expert(x, sample)
            yards_preds.append(yards)
            td_logits_preds.append(td_logits)

        # Stack predictions: (n_experts, batch_size)
        yards_preds = torch.stack(yards_preds, dim=0)
        td_logits_preds = torch.stack(td_logits_preds, dim=0)

        # Weighted combination (expert gating)
        gate_weights_t = gate_weights.t()  # (n_experts, batch_size)
        yards_combined = (gate_weights_t * yards_preds).sum(dim=0)
        td_logits_combined = (gate_weights_t * td_logits_preds).sum(dim=0)

        # Weighted heteroscedastic noise
        noise_std = torch.exp(0.5 * self.log_noise)  # (n_experts,)
        weighted_noise = (gate_weights * noise_std.unsqueeze(0)).sum(dim=-1)  # (batch_size,)

        return yards_combined, td_logits_combined, weighted_noise

    def kl_divergence(self):
        """Total KL divergence from all expert networks"""
        kl = 0
        for expert in self.experts:
            kl += expert.kl_divergence()
        return kl


def load_passing_data_from_db():
    """
    Load passing yards data with canonical training window

    Training Window:
    - 2018-2024: Full data (split 2018-2023 train/val, 2024 test)
    - 2015-2017: Decay weights with H=4 exponential decay

    Returns:
        df: DataFrame with passing stats, window weights, train/val/test flags
    """
    logger.info("Loading passing data from database (canonical window)...")

    conn = psycopg2.connect(
        host="localhost", port=5544, user="dro", password="sicillionbillions", database="devdb01"
    )

    query = """
    WITH qb_passing AS (
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as yards,
            pgs.stat_touchdowns as touchdowns,
            CASE WHEN pgs.stat_touchdowns >= 1 THEN 1 ELSE 0 END as scored_td,
            pgs.stat_attempts as attempts,
            pgs.current_team as team,
            g.home_score,
            g.away_score,
            g.spread_close as spread_line,
            g.total_close as total_line,
            g.temp,
            g.wind,
            CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
            -- Rolling averages (lag 1)
            AVG(pgs.stat_yards) OVER (
                PARTITION BY pgs.player_id
                ORDER BY pgs.season, pgs.week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as yards_L3,
            AVG(pgs.stat_yards / NULLIF(pgs.stat_attempts, 0)) OVER (
                PARTITION BY pgs.player_id
                ORDER BY pgs.season, pgs.week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as yards_per_attempt_L3,
            AVG(CAST(pgs.stat_touchdowns AS FLOAT) / NULLIF(pgs.stat_attempts, 0)) OVER (
                PARTITION BY pgs.player_id
                ORDER BY pgs.season, pgs.week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as td_rate_L3
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
          AND pgs.week >= 1  -- Exclude Week 0 (cumulative stats contamination)
          AND ph.position_group = 'QB'
    )
    SELECT
        *,
        -- Canonical window weights (H=4 exponential decay)
        CASE
            WHEN season >= 2018 THEN 1.0
            WHEN season = 2017 THEN EXP(-1.0/4.0)  -- exp(-1/4) ≈ 0.779
            WHEN season = 2016 THEN EXP(-2.0/4.0)  -- exp(-2/4) ≈ 0.607
            WHEN season = 2015 THEN EXP(-3.0/4.0)  -- exp(-3/4) ≈ 0.472
            ELSE 0.0
        END as weight,
        -- Dataset split
        CASE
            WHEN season BETWEEN 2015 AND 2023 THEN 'train'
            WHEN season = 2024 THEN 'test'
        END as split
    FROM qb_passing
    WHERE yards_L3 IS NOT NULL  -- Filter out first 3 games per player (no rolling avg)
    ORDER BY season, week, yards DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} QB games from 2015-2024")
    logger.info(f"Train: {len(df[df['split']=='train'])} games (2015-2023)")
    logger.info(f"Test: {len(df[df['split']=='test'])} games (2024)")
    logger.info(f"TD rate: {100*df['scored_td'].mean():.1f}%")
    logger.info(
        f"Decay window (2015-2017): {len(df[df['season']<2018])} games (weights: 0.47-0.78)"
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for BNN"""
    logger.info("Engineering features...")

    # Derive efficiency metrics (from raw data)
    df["log_attempts"] = np.log1p(df["attempts"])
    df["yards_per_attempt"] = df["yards"] / df["attempts"].replace(0, np.nan)
    df["td_rate"] = df["touchdowns"] / df["attempts"].replace(0, np.nan)

    # Game context
    df["spread_abs"] = df["spread_line"].abs()

    # Convert weather to numeric
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")

    df["is_bad_weather"] = ((df["temp"] < 32) | (df["wind"] > 15)).fillna(False).astype(int)

    # Week in season (proxy for playoff push, player fatigue)
    df["week_in_season"] = df["week"].astype(float)

    # Rename rolling average columns from SQL query (already computed)
    # SQL query returns: yards_L3, yards_per_attempt_L3, td_rate_L3

    # Fill missing values with median
    numeric_cols = [
        "yards_l3",
        "yards_per_attempt_l3",
        "td_rate_l3",  # Match SQL column case
        "spread_line",
        "total_line",
        "temp",
        "wind",
        "yards_per_attempt",
        "td_rate",  # Derived features
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"Feature engineering complete: {len(df)} samples")
    return df


def prepare_train_test_split(df: pd.DataFrame):
    """
    Split data by canonical window

    Train/Val: 2015-2023 (with decay weights for 2015-2017)
    Test: 2024
    """
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Feature columns (13 features matching existing BNN)
    # Note: SQL query returns lowercase column names for rolling averages
    feature_cols = [
        "log_attempts",
        "yards_per_attempt",
        "td_rate",
        "is_home",
        "spread_abs",
        "total_line",
        "is_bad_weather",
        "yards_l3",  # Lowercase (from SQL query)
        "yards_per_attempt_l3",  # Lowercase (from SQL query)
        "td_rate_l3",  # Lowercase (from SQL query)
        "week_in_season",
        "temp",
        "wind",
    ]

    # Extract features
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_yards_train = train_df["yards"].values.astype(np.float32)
    y_td_train = train_df["scored_td"].values.astype(np.float32)
    weights_train = train_df["weight"].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_yards_test = test_df["yards"].values.astype(np.float32)
    y_td_test = test_df["scored_td"].values.astype(np.float32)

    logger.info(f"Train: {len(X_train)} samples (2015-2023)")
    logger.info(f"  - Full weight (2018-2023): {np.sum(weights_train == 1.0):.0f} samples")
    logger.info(f"  - Decay weight (2015-2017): {np.sum(weights_train < 1.0):.0f} samples")
    logger.info(f"Test: {len(X_test)} samples (2024)")
    logger.info(f"Features: {len(feature_cols)}")

    return (
        X_train,
        y_yards_train,
        y_td_train,
        weights_train,
        X_test,
        y_yards_test,
        y_td_test,
        feature_cols,
    )


def train_model(
    model, train_loader, val_loader, device, epochs=200, lr=0.001, kl_scale=0.01, log_path=None
):
    """
    Train Multi-Output BNN with KL annealing and weighted loss

    Args:
        model: MultiOutputPassingBNN
        train_loader: DataLoader with (X, y_yards, y_td, weights)
        val_loader: DataLoader with (X, y_yards, y_td)
        device: torch.device
        epochs: Training epochs
        lr: Learning rate
        kl_scale: KL divergence scaling factor
        log_path: Optional path to save training log
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.5)

    # KL annealing schedule (linear over 30 epochs)
    def kl_weight(epoch):
        return min(1.0, epoch / 30)

    best_val_loss = float("inf")
    train_history = []

    # Initialize checkpoint (prevents UnboundLocalError)
    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": float("inf"),
        "val_yards_mae": float("inf"),
        "val_td_acc": 0.0,
    }

    logger.info(f"Training on {device} for {epochs} epochs")
    logger.info(f"Learning rate: {lr}, KL scale: {kl_scale}")

    for epoch in range(epochs):
        start_time = time.time()

        # ===== Training =====
        model.train()
        train_loss = 0
        train_yards_loss = 0
        train_td_loss = 0
        train_kl = 0

        for batch_x, batch_y_yards, batch_y_td, batch_weights in train_loader:
            batch_x = batch_x.to(device)
            batch_y_yards = batch_y_yards.to(device)
            batch_y_td = batch_y_td.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()

            # Forward pass (stochastic)
            yards_pred, td_logits, noise_std = model(batch_x, sample=True)

            # Numerical stability: Clamp noise_std to prevent log(0) or division by zero
            # Min value: exp(-5) ≈ 0.0067 (prevents log underflow)
            # Max value: exp(5) ≈ 148 (prevents explosion)
            noise_std_clamped = torch.clamp(noise_std, min=1e-3, max=10.0)

            # Negative log-likelihood for yards (Gaussian, heteroscedastic)
            yards_nll = 0.5 * (
                torch.log(2 * np.pi * noise_std_clamped**2)
                + (batch_y_yards - yards_pred) ** 2 / (noise_std_clamped**2 + 1e-6)
            )
            # Weight by canonical window (H=4 decay)
            yards_nll_weighted = torch.mean(batch_weights * yards_nll)

            # Binary cross-entropy for TDs (weighted)
            td_loss_per_sample = F.binary_cross_entropy_with_logits(
                td_logits, batch_y_td, reduction="none"
            )
            td_loss_weighted = torch.mean(batch_weights * td_loss_per_sample)

            # KL divergence (variational lower bound)
            kl_div = model.kl_divergence() / len(train_loader.dataset)

            # Total loss = NLL (yards) + NLL (TD) + β*KL
            loss = yards_nll_weighted + td_loss_weighted + kl_weight(epoch) * kl_scale * kl_div

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_yards_loss += yards_nll_weighted.item()
            train_td_loss += td_loss_weighted.item()
            train_kl += kl_div.item()

        # ===== Validation =====
        model.eval()
        val_loss = 0
        val_yards_mae = 0
        val_td_acc = 0

        with torch.no_grad():
            for batch_x, batch_y_yards, batch_y_td in val_loader:
                batch_x = batch_x.to(device)
                batch_y_yards = batch_y_yards.to(device)
                batch_y_td = batch_y_td.to(device)

                # Multiple forward passes for uncertainty quantification
                n_samples = 10
                yards_samples = []
                td_probs = []

                for _ in range(n_samples):
                    yards_pred, td_logits, _ = model(batch_x, sample=True)
                    yards_samples.append(yards_pred)
                    td_probs.append(torch.sigmoid(td_logits))

                # Aggregate predictions
                yards_mean = torch.stack(yards_samples).mean(dim=0)
                td_prob_mean = torch.stack(td_probs).mean(dim=0)

                # Metrics
                val_yards_mae += torch.mean(torch.abs(yards_mean - batch_y_yards)).item()
                val_td_acc += ((td_prob_mean > 0.5).float() == batch_y_td).float().mean().item()
                val_loss += F.mse_loss(yards_mean, batch_y_yards).item()

        # Average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_yards_mae = val_yards_mae / len(val_loader)
        avg_val_td_acc = val_td_acc / len(val_loader)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_yards_mae": avg_val_yards_mae,
                "val_td_acc": avg_val_td_acc,
            }

        # Logging
        elapsed = time.time() - start_time
        if epoch % 10 == 0:
            log_msg = (
                f"Epoch {epoch:4d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val MAE: {avg_val_yards_mae:.2f} yards | "
                f"Val TD Acc: {avg_val_td_acc:.2%} | "
                f"Time: {elapsed:.1f}s"
            )
            logger.info(log_msg)

            if log_path:
                with open(log_path, "a") as f:
                    f.write(log_msg + "\n")

            # GPU memory monitoring
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU Memory: {gpu_mem:.2f} GB")

        # Store history
        train_history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_mae": avg_val_yards_mae,
                "val_td_acc": avg_val_td_acc,
            }
        )

    logger.info(f"✓ Training complete: best_val_loss={best_val_loss:.4f}")

    return checkpoint, train_history


def evaluate_model(model, test_loader, scaler, device, y_mean, y_std):
    """
    Evaluate Multi-Output BNN on test set (2024)

    Metrics:
    - MAE (yards)
    - RMSE (yards)
    - Calibration: 90% coverage (target: >88%)
    - TD accuracy and Brier score
    """
    model.eval()

    all_yards_true = []
    all_yards_pred = []
    all_yards_std = []
    all_td_true = []
    all_td_pred = []

    with torch.no_grad():
        for batch_x, batch_y_yards, batch_y_td in test_loader:
            batch_x = batch_x.to(device)

            # Monte Carlo sampling (100 forward passes)
            n_samples = 100
            yards_samples = []
            td_samples = []

            for _ in range(n_samples):
                yards_pred, td_logits, _ = model(batch_x, sample=True)
                yards_samples.append(yards_pred.cpu())
                td_samples.append(torch.sigmoid(td_logits).cpu())

            yards_samples = torch.stack(yards_samples)  # (n_samples, batch_size)
            td_samples = torch.stack(td_samples)

            # Statistics
            yards_mean = yards_samples.mean(dim=0)
            yards_std = yards_samples.std(dim=0)
            td_prob_mean = td_samples.mean(dim=0)

            all_yards_true.append(batch_y_yards.numpy())
            all_yards_pred.append(yards_mean.numpy())
            all_yards_std.append(yards_std.numpy())
            all_td_true.append(batch_y_td.numpy())
            all_td_pred.append(td_prob_mean.numpy())

    # Concatenate all batches
    yards_true_norm = np.concatenate(all_yards_true)
    yards_pred_norm = np.concatenate(all_yards_pred)
    yards_std_norm = np.concatenate(all_yards_std)
    td_true = np.concatenate(all_td_true)
    td_pred = np.concatenate(all_td_pred)

    # Denormalize yards predictions
    yards_true = yards_true_norm * y_std + y_mean
    yards_pred = yards_pred_norm * y_std + y_mean
    yards_std = yards_std_norm * y_std  # Scale uncertainty

    # Yards metrics
    mae = np.mean(np.abs(yards_pred - yards_true))
    rmse = np.sqrt(np.mean((yards_pred - yards_true) ** 2))

    # 90% prediction interval coverage (±1.645σ)
    yards_lower = yards_pred - 1.645 * yards_std
    yards_upper = yards_pred + 1.645 * yards_std
    coverage_90 = np.mean((yards_true >= yards_lower) & (yards_true <= yards_upper))

    # TD metrics
    td_accuracy = np.mean((td_pred > 0.5) == td_true)
    td_brier = np.mean((td_pred - td_true) ** 2)

    results = {
        "yards_mae": float(mae),
        "yards_rmse": float(rmse),
        "yards_coverage_90": float(coverage_90),
        "yards_mean_std": float(yards_std.mean()),
        "td_accuracy": float(td_accuracy),
        "td_brier_score": float(td_brier),
        "td_mean_prob": float(td_pred.mean()),
        "td_actual_rate": float(td_true.mean()),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS (2024 Test Set)")
    logger.info("=" * 80)
    logger.info("Passing Yards:")
    logger.info(f"  MAE: {mae:.2f} yards")
    logger.info(f"  RMSE: {rmse:.2f} yards")
    logger.info(f"  90% Coverage: {coverage_90:.1%} (target: >88%)")
    logger.info(f"  Mean Uncertainty: {yards_std.mean():.2f} yards")
    logger.info("\nPassing TDs:")
    logger.info(f"  Accuracy: {td_accuracy:.1%}")
    logger.info(f"  Brier Score: {td_brier:.4f}")
    logger.info(f"  Mean Pred Prob: {td_pred.mean():.1%}")
    logger.info(f"  Actual TD Rate: {td_true.mean():.1%}")
    logger.info("=" * 80)

    # Decision criteria
    if coverage_90 >= 0.88:
        logger.info("✅ SUCCESS: Calibration >88% - Multi-Output BNN viable for production!")
    elif coverage_90 >= 0.80:
        logger.info("⚠️  PARTIAL: Calibration 80-88% - Further tuning recommended")
    else:
        logger.info("❌ FAILURE: Calibration <80% - Consider demotion to secondary role")

    return results


def main():
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("MULTI-OUTPUT PASSING BNN - Canonical Training Window")
    logger.info("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data
    df = load_passing_data_from_db()
    df = engineer_features(df)
    (
        X_train,
        y_yards_train,
        y_td_train,
        weights_train,
        X_test,
        y_yards_test,
        y_td_test,
        feature_cols,
    ) = prepare_train_test_split(df)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Normalize yards target (z-score normalization)
    # CRITICAL: Must normalize AFTER applying exponential decay weights
    # Weighted mean and std to account for canonical window
    weighted_mean = np.average(y_yards_train, weights=weights_train)
    weighted_var = np.average((y_yards_train - weighted_mean) ** 2, weights=weights_train)
    weighted_std = np.sqrt(weighted_var)

    # Fallback to simple normalization if weights cause issues
    if not np.isfinite(weighted_mean) or not np.isfinite(weighted_std) or weighted_std < 1e-6:
        logger.warning("Weighted normalization failed, using simple mean/std")
        weighted_mean = y_yards_train.mean()
        weighted_std = y_yards_train.std()

    y_yards_train_norm = (y_yards_train - weighted_mean) / weighted_std
    y_yards_test_norm = (y_yards_test - weighted_mean) / weighted_std

    # Store for denormalization
    y_mean = weighted_mean
    y_std = weighted_std

    logger.info("\nTarget normalization (weighted by canonical window):")
    logger.info(f"  Mean: {y_mean:.2f} yards")
    logger.info(f"  Std: {y_std:.2f} yards")
    logger.info(
        f"  Normalized train range: [{y_yards_train_norm.min():.2f}, {y_yards_train_norm.max():.2f}]"
    )

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_yards_train_t = torch.FloatTensor(y_yards_train_norm)
    y_td_train_t = torch.FloatTensor(y_td_train)
    weights_train_t = torch.FloatTensor(weights_train)

    X_test_t = torch.FloatTensor(X_test_scaled)
    y_yards_test_t = torch.FloatTensor(y_yards_test_norm)
    y_td_test_t = torch.FloatTensor(y_td_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_yards_train_t, y_td_train_t, weights_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = TensorDataset(X_test_t, y_yards_test_t, y_td_test_t)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Initialize model
    model = MultiOutputPassingBNN(
        input_dim=X_train.shape[1], n_experts=3, hidden_dim=32, dropout_rate=0.2
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModel: {n_params:,} parameters, 3 experts")

    # Training
    log_path = Path("logs/multioutput_passing_bnn_canonical.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)

    checkpoint, train_history = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=200,
        lr=0.001,
        kl_scale=0.01,
        log_path=log_path,
    )

    # Load best model for evaluation
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    # Evaluation
    results = evaluate_model(model, test_loader, scaler, device, y_mean, y_std)

    # Save model
    model_path = Path("models/bayesian/bnn_multioutput_passing_v1_canonical.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "y_mean": y_mean,
            "y_std": y_std,
            "feature_cols": feature_cols,
            "results": results,
            "train_history": train_history,
            "hyperparameters": {
                "n_experts": 3,
                "hidden_dim": 32,
                "dropout_rate": 0.2,
                "epochs": 200,
                "learning_rate": 0.001,
                "kl_scale": 0.01,
            },
        },
        model_path,
    )

    logger.info(f"\n✅ Model saved to {model_path}")

    # Save results CSV
    results_path = Path("analysis/results/multioutput_passing_bnn_canonical.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([results])
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    # GPU summary
    if device.type == "cuda":
        logger.info("\nGPU Memory Usage:")
        logger.info(f"  Current: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        logger.info(f"  Peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    return results


if __name__ == "__main__":
    main()
