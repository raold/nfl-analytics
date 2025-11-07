#!/usr/bin/env python3
"""
Diagnose BNN Training Issue

Check what the trained BNN is predicting and why.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from scripts.train_bnn_passing_ensemble import PassingYardsBNN

# Load checkpoint
checkpoint_path = "models/bayesian/bnn_passing_ensemble_v1.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*80)
print("BNN DIAGNOSTIC")
print("="*80)

# Check checkpoint contents
print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Check scaler
if 'scaler_mean' in checkpoint:
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    print(f"\nScaler:")
    print(f"  Mean: {scaler_mean[:5]} ... (shape: {scaler_mean.shape})")
    print(f"  Scale: {scaler_scale[:5]} ... (shape: {scaler_scale.shape})")
else:
    print("\n⚠️  No scaler found in checkpoint")

# Check training stats
if 'best_val_loss' in checkpoint:
    print(f"\nTraining stats:")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Best val MAE: {checkpoint['val_mae']:.2f} yards")

# Reconstruct model
args = checkpoint.get('args', {})
hidden_dim = args.get('hidden_dim', 64)
dropout = args.get('dropout', 0.2)
input_dim = len(checkpoint['feature_cols'])

print(f"\nModel architecture:")
print(f"  Input: {input_dim} features")
print(f"  Hidden: {hidden_dim}")
print(f"  Dropout: {dropout}")

model = PassingYardsBNN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    dropout_rate=dropout
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("\n✓ Model loaded successfully")

# Test prediction with dummy data
print("\n" + "="*80)
print("TEST PREDICTIONS")
print("="*80)

# Create test features (around typical Mahomes values)
test_features = np.array([[
    226.33,  # passing_yards_last3
    251.80,  # passing_yards_last5
    239.90,  # passing_yards_season
    27.67,   # pass_attempts_last3
    32.40,   # pass_attempts_last5
    18.33,   # completions_last3
    22.40,   # completions_last5
    1.67,    # passing_tds_last3
    0.67,    # interceptions_last3
    2.5,     # sacks_last3
    270.40,  # opponent_passing_yards_allowed_avg
    22.0,    # implied_team_total
    0.0,     # spread
    1.0,     # is_home
    5.0,     # weather_wind_mph
    60.0,    # weather_temp
    7.0,     # days_rest
    0.0,     # is_dome
    1.0,     # is_outdoors
]])

print("\n1. RAW FEATURES (unnormalized):")
print(f"   Features: {test_features[0, :5]} ... (shape: {test_features.shape})")

# Normalize
X_norm = (test_features - scaler_mean) / scaler_scale
print(f"\n2. NORMALIZED FEATURES:")
print(f"   Features: {X_norm[0, :5]} ... (shape: {X_norm.shape})")

# Convert to tensor
X_tensor = torch.FloatTensor(X_norm)

# Predict
with torch.no_grad():
    mean, log_std = model(X_tensor, sample=False)
    mean_val = mean[0].item()
    std_val = np.exp(log_std[0].item())

print(f"\n3. MODEL OUTPUT:")
print(f"   Mean: {mean_val:.2f} yards")
print(f"   Log std: {log_std[0].item():.4f}")
print(f"   Std: {std_val:.2f} yards")

# Check if we need to denormalize
print(f"\n4. ANALYSIS:")
if abs(mean_val) < 10:
    print(f"   ⚠️  Mean is very small ({mean_val:.2f} yards)")
    print(f"   Possible issues:")
    print(f"   1. Target variable was normalized but not denormalized")
    print(f"   2. Model is predicting on normalized scale")
    print(f"   3. Training bug - model didn't learn properly")
elif abs(mean_val - 240) > 100:
    print(f"   ⚠️  Mean is far from typical values ({mean_val:.2f} yards)")
    print(f"   Expected: ~240 yards for Mahomes")
else:
    print(f"   ✓ Prediction looks reasonable")

# Check model weights
print(f"\n5. MODEL WEIGHTS:")
print(f"   fc1 weight_mu mean: {model.fc1.weight_mu.mean().item():.6f}")
print(f"   fc1 weight_mu std: {model.fc1.weight_mu.std().item():.6f}")
print(f"   yards_mean weight_mu mean: {model.yards_mean.weight_mu.mean().item():.6f}")
print(f"   yards_mean bias_mu: {model.yards_mean.bias_mu.item():.6f}")

if abs(model.yards_mean.bias_mu.item()) > 100:
    print(f"   ⚠️  Large bias in output layer: {model.yards_mean.bias_mu.item():.2f}")

print("\n" + "="*80)
