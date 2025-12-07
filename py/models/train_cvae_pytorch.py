#!/usr/bin/env python3
"""
Proper Conditional Variational Autoencoder (CVAE) for NFL game prediction.
Uses PyTorch for real VAE architecture with proper loss function.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

print("=" * 70)
print(" CONDITIONAL VAE TRAINING (PYTORCH) ".center(70))
print("=" * 70)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[*] Using device: {device}")

# Database connection
conn = psycopg2.connect(
    host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
)

# Load aggregated play data
print("\n1. Loading play-level data...")
query = """
WITH game_stats AS (
    SELECT
        SUBSTRING(game_id FROM 1 FOR 4)::int as season,
        game_id,
        posteam as team,
        defteam as opp_team,
        COUNT(*) as plays,
        AVG(epa) as avg_epa,
        STDDEV(epa) as std_epa,
        AVG(yards_gained) as avg_yards,
        STDDEV(yards_gained) as std_yards,
        AVG(wp) as avg_wp,
        STDDEV(wp) as std_wp,
        SUM(CASE WHEN pass = 1 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as pass_rate,
        SUM(CASE WHEN touchdown = 1 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as td_rate,
        SUM(CASE WHEN first_down = 1 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) as first_down_rate,
        MAX(posteam_score_post) as final_score,
        MAX(defteam_score_post) as opp_score
    FROM plays
    WHERE posteam IS NOT NULL
    AND SUBSTRING(game_id FROM 1 FOR 4)::int >= 2015
    GROUP BY game_id, posteam, defteam
    HAVING COUNT(*) > 20
)
SELECT
    *,
    CASE WHEN final_score > opp_score THEN 1.0 ELSE 0.0 END as win
FROM game_stats
ORDER BY season, game_id
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"[+] Loaded {len(df):,} team-game records")
print(f"[+] Years: {df.season.min()}-{df.season.max()}")

# Handle missing values
for col in df.select_dtypes(include=[np.float64, np.int64]).columns:
    df[col] = df[col].fillna(df[col].median())


class ConditionalVAE(nn.Module):
    """
    Proper CVAE architecture:
    - Encoder: (X, C) -> mu, log_var
    - Reparameterization: z ~ N(mu, var)
    - Decoder: (z, C) -> X_reconstructed
    """

    def __init__(self, input_dim, condition_dim, latent_dim=16, hidden_dim=64):
        super(ConditionalVAE, self).__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # Encoder: (input + condition) -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (latent + condition) -> output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),  # Predict final score
        )

    def encode(self, x, c):
        """Encode input + condition to latent distribution."""
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """Decode latent + condition to output."""
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(self, x, c):
        """Full forward pass."""
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)
        return x_recon, mu, log_var

    def loss_function(self, x_recon, x_true, mu, log_var, beta=0.01):
        """
        CVAE loss = Reconstruction Loss + KL Divergence
        beta: weight for KL term (beta-VAE for better disentanglement)
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x_true, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div / x_true.size(0)  # Average over batch

        return recon_loss + beta * kl_div, recon_loss, kl_div


class GameDataset(Dataset):
    """PyTorch dataset for game data."""

    def __init__(self, X, conditions, y):
        self.X = torch.FloatTensor(X)
        self.conditions = torch.FloatTensor(conditions)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.conditions[idx], self.y[idx]


# Prepare features
print("\n2. Preparing features...")

# Features (stats)
feature_cols = [
    "avg_epa",
    "std_epa",
    "avg_yards",
    "std_yards",
    "avg_wp",
    "std_wp",
    "pass_rate",
    "td_rate",
    "first_down_rate",
]

# Conditions (context)
condition_cols = ["season", "plays"]

# Target
target_col = "final_score"

X = df[feature_cols].values
conditions = df[condition_cols].values
y = df[target_col].values

# Split data
train_mask = df["season"] < 2022
X_train = X[train_mask]
X_test = X[~train_mask]
conditions_train = conditions[train_mask]
conditions_test = conditions[~train_mask]
y_train = y[train_mask]
y_test = y[~train_mask]

# Scale data
scaler_input = StandardScaler()
scaler_condition = StandardScaler()
scaler_output = StandardScaler()

X_train_scaled = scaler_input.fit_transform(X_train)
X_test_scaled = scaler_input.transform(X_test)
conditions_train_scaled = scaler_condition.fit_transform(conditions_train)
conditions_test_scaled = scaler_condition.transform(conditions_test)
y_train_scaled = scaler_output.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_output.transform(y_test.reshape(-1, 1)).ravel()

print(f"[+] Train: {len(X_train):,} samples")
print(f"[+] Test: {len(X_test):,} samples")

# Create dataloaders
train_dataset = GameDataset(X_train_scaled, conditions_train_scaled, y_train_scaled)
test_dataset = GameDataset(X_test_scaled, conditions_test_scaled, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
print("\n3. Initializing CVAE...")
model = ConditionalVAE(
    input_dim=len(feature_cols), condition_dim=len(condition_cols), latent_dim=16, hidden_dim=64
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

print(f"[+] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
print("\n4. Training CVAE...")
n_epochs = 200
best_test_loss = float("inf")
patience_counter = 0
max_patience = 30

start_time = time.time()

for epoch in range(n_epochs):
    # Training
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0

    for X_batch, c_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        c_batch = c_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass (reconstruct final_score from features + conditions)
        y_recon, mu, log_var = model(X_batch, c_batch)

        # Compute loss
        loss, recon_loss, kl_loss = model.loss_function(y_recon, y_batch, mu, log_var, beta=0.01)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()

    train_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    train_kl_loss /= len(train_loader)

    # Validation
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for X_batch, c_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            c_batch = c_batch.to(device)
            y_batch = y_batch.to(device)

            y_recon, mu, log_var = model(X_batch, c_batch)
            loss, recon_loss, kl_loss = model.loss_function(
                y_recon, y_batch, mu, log_var, beta=0.01
            )

            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()

    test_loss /= len(test_loader)
    test_recon_loss /= len(test_loader)
    test_kl_loss /= len(test_loader)

    scheduler.step(test_loss)

    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "models/cvae/best_model.pt")
    else:
        patience_counter += 1

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{n_epochs}] "
            f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}) | "
            f"Test Loss: {test_loss:.4f} (Recon: {test_recon_loss:.4f}, KL: {test_kl_loss:.4f})"
        )

    if patience_counter >= max_patience:
        print(f"[*] Early stopping at epoch {epoch+1}")
        break

training_time = time.time() - start_time
print(f"[+] Training completed in {training_time:.1f} seconds")

# Load best model
model.load_state_dict(torch.load("models/cvae/best_model.pt"))
model.eval()

# Evaluate with uncertainty quantification
print("\n5. Evaluating with uncertainty quantification...")


def predict_with_uncertainty(model, X, conditions, n_samples=100):
    """Generate predictions with uncertainty using multiple samples."""
    X_tensor = torch.FloatTensor(X).to(device)
    c_tensor = torch.FloatTensor(conditions).to(device)

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            mu, log_var = model.encode(X_tensor, c_tensor)
            z = model.reparameterize(mu, log_var)
            y_pred = model.decode(z, c_tensor)
            predictions.append(y_pred.cpu().numpy())

    predictions = np.array(predictions).squeeze()

    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    lower_90 = np.percentile(predictions, 5, axis=0)
    upper_90 = np.percentile(predictions, 95, axis=0)

    return {"mean": mean_pred, "std": std_pred, "lower_90": lower_90, "upper_90": upper_90}


# Get predictions
train_preds = predict_with_uncertainty(
    model, X_train_scaled, conditions_train_scaled, n_samples=100
)
test_preds = predict_with_uncertainty(model, X_test_scaled, conditions_test_scaled, n_samples=100)

# Inverse transform predictions
train_preds_original = scaler_output.inverse_transform(train_preds["mean"].reshape(-1, 1)).ravel()
test_preds_original = scaler_output.inverse_transform(test_preds["mean"].reshape(-1, 1)).ravel()

# Calculate metrics
train_mse = mean_squared_error(y_train, train_preds_original)
test_mse = mean_squared_error(y_test, test_preds_original)
train_r2 = r2_score(y_train, train_preds_original)
test_r2 = r2_score(y_test, test_preds_original)

# Calculate coverage
train_lower = scaler_output.inverse_transform(train_preds["lower_90"].reshape(-1, 1)).ravel()
train_upper = scaler_output.inverse_transform(train_preds["upper_90"].reshape(-1, 1)).ravel()
test_lower = scaler_output.inverse_transform(test_preds["lower_90"].reshape(-1, 1)).ravel()
test_upper = scaler_output.inverse_transform(test_preds["upper_90"].reshape(-1, 1)).ravel()

train_coverage = np.mean((y_train >= train_lower) & (y_train <= train_upper))
test_coverage = np.mean((y_test >= test_lower) & (y_test <= test_upper))

# Average uncertainty (in original scale)
train_uncertainty = np.mean(train_upper - train_lower) / 1.645  # Convert 90% CI to std
test_uncertainty = np.mean(test_upper - test_lower) / 1.645

print("\n" + "=" * 70)
print(" RESULTS ".center(70))
print("=" * 70)
print(f"Train MSE: {train_mse:.2f} | R²: {train_r2:.3f}")
print(f"Test MSE:  {test_mse:.2f} | R²: {test_r2:.3f}")
print(f"Train 90% Coverage: {train_coverage:.1%}")
print(f"Test 90% Coverage:  {test_coverage:.1%}")
print(f"Avg Uncertainty - Train: {train_uncertainty:.1f} points")
print(f"Avg Uncertainty - Test:  {test_uncertainty:.1f} points")

# Save everything
output_dir = Path("models/cvae")
output_dir.mkdir(parents=True, exist_ok=True)

# Save scalers
import joblib

joblib.dump(scaler_input, output_dir / "scaler_input_pytorch.pkl")
joblib.dump(scaler_condition, output_dir / "scaler_condition_pytorch.pkl")
joblib.dump(scaler_output, output_dir / "scaler_output_pytorch.pkl")

# Save metrics
metrics = {
    "train_mse": float(train_mse),
    "test_mse": float(test_mse),
    "train_r2": float(train_r2),
    "test_r2": float(test_r2),
    "train_coverage": float(train_coverage),
    "test_coverage": float(test_coverage),
    "train_uncertainty": float(train_uncertainty),
    "test_uncertainty": float(test_uncertainty),
    "training_time": float(training_time),
    "latent_dim": model.latent_dim,
    "hidden_dim": 64,
    "features": feature_cols,
    "conditions": condition_cols,
    "architecture": "proper_pytorch_vae",
}

with open(output_dir / "cvae_pytorch_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[+] Model saved to {output_dir}/best_model.pt")
print(f"[+] Metrics saved to {output_dir}/cvae_pytorch_metrics.json")
