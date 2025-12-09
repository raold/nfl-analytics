#!/usr/bin/env python3
"""
GPU-Optimized Multi-Output Bayesian Neural Network using PyTorch
Joint prediction of rushing yards + rushing TDs with mixture of experts
Fast variational inference for RTX 4090 GPU acceleration

Key Features:
- 3 expert networks with gating mechanism
- Joint prediction of yards (continuous) and TDs (binary)
- GPU-accelerated training with variational inference
- Checkpoint support for long training runs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import argparse
import json
from pathlib import Path
import logging

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
        # Initialize using He initialization
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in')
        nn.init.constant_(self.weight_rho, -3)  # log(exp(0.05) - 1) ≈ -3
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x, sample=True):
        if sample:
            # Sample weights using reparameterization trick
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # Use mean weights for deterministic forward pass
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """KL divergence from posterior to prior"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # KL divergence for weights
        kl_weight = 0.5 * torch.sum(
            weight_sigma**2 / self.prior_var + self.weight_mu**2 / self.prior_var 
            - torch.log(weight_sigma**2 / self.prior_var) - 1
        )
        
        # KL divergence for bias
        kl_bias = 0.5 * torch.sum(
            bias_sigma**2 / self.prior_var + self.bias_mu**2 / self.prior_var
            - torch.log(bias_sigma**2 / self.prior_var) - 1
        )
        
        return kl_weight + kl_bias


class ExpertNetwork(nn.Module):
    """Single expert network for mixture of experts"""
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.2):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output heads
        self.yards_head = BayesianLinear(hidden_dim, 1)  # Continuous output
        self.td_head = BayesianLinear(hidden_dim, 1)     # Binary output
        
    def forward(self, x, sample=True):
        h = F.relu(self.fc1(x, sample))
        h = self.dropout(h)
        h = F.relu(self.fc2(h, sample))
        h = self.dropout(h)
        
        yards = self.yards_head(h, sample)
        td_logits = self.td_head(h, sample)
        
        return yards.squeeze(-1), td_logits.squeeze(-1)
    
    def kl_divergence(self):
        kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence()
        return kl


class GatingNetwork(nn.Module):
    """Gating network for mixture of experts"""
    def __init__(self, input_dim, n_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_experts)
        
    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class MultiOutputMoEBNN(nn.Module):
    """Multi-Output Mixture of Experts Bayesian Neural Network"""
    def __init__(self, input_dim, n_experts=3, hidden_dim=32, dropout_rate=0.2):
        super().__init__()
        self.n_experts = n_experts
        
        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, dropout_rate)
            for _ in range(n_experts)
        ])
        
        # Gating network
        self.gate = GatingNetwork(input_dim, n_experts)
        
        # Expert-specific noise parameters for yards
        self.log_noise = nn.Parameter(torch.zeros(n_experts))
        
    def forward(self, x, sample=True):
        # Get gating weights
        gate_weights = self.gate(x)  # (batch_size, n_experts)
        
        # Get predictions from each expert
        yards_preds = []
        td_logits_preds = []
        
        for i, expert in enumerate(self.experts):
            yards, td_logits = expert(x, sample)
            yards_preds.append(yards)
            td_logits_preds.append(td_logits)
        
        # Stack predictions: (n_experts, batch_size)
        yards_preds = torch.stack(yards_preds, dim=0)
        td_logits_preds = torch.stack(td_logits_preds, dim=0)
        
        # Weighted combination
        # Transpose gate_weights to (n_experts, batch_size) for broadcasting
        gate_weights_t = gate_weights.t()
        
        # Weighted mean for yards
        yards_combined = (gate_weights_t * yards_preds).sum(dim=0)
        
        # Weighted mean for TD logits
        td_logits_combined = (gate_weights_t * td_logits_preds).sum(dim=0)
        
        # Weighted noise for yards (heteroscedastic)
        noise_std = torch.exp(0.5 * self.log_noise)  # (n_experts,)
        weighted_noise = (gate_weights * noise_std.unsqueeze(0)).sum(dim=-1)  # (batch_size,)
        
        return yards_combined, td_logits_combined, weighted_noise
    
    def kl_divergence(self):
        kl = 0
        for expert in self.experts:
            kl += expert.kl_divergence()
        return kl


def load_data(start_season=2020, end_season=2024):
    """Load rushing data from database"""
    logger.info(f"Loading data from seasons {start_season}-{end_season}")
    
    conn = psycopg2.connect(
        host='localhost',
        port=5544,
        dbname='devdb01',
        user='dro',
        password='sicillionbillions'
    )
    
    query = """
    WITH player_stats AS (
        SELECT
            pgs.player_id,
            pgs.player_name,
            pgs.season,
            pgs.week,
            pgs.team,
            pgs.position,
            pgs.rushing_yards as stat_yards,
            pgs.rushing_tds as stat_touchdowns,
            CASE WHEN pgs.rushing_tds >= 1 THEN 1 ELSE 0 END as scored_td,
            -- Estimate carries (not available in schema, use proxy)
            CASE
                WHEN pgs.rushing_yards > 0 THEN GREATEST(5, pgs.rushing_yards / 4.5)
                ELSE 5
            END as carries,
            -- Rolling averages
            AVG(pgs.rushing_yards) OVER (
                PARTITION BY pgs.player_id
                ORDER BY pgs.season, pgs.week
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as avg_rushing_l3,
            AVG(pgs.rushing_yards) OVER (
                PARTITION BY pgs.player_id, pgs.season
                ORDER BY pgs.week
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) as season_avg,
            -- Opponent strength (simplified)
            AVG(pgs.rushing_yards) OVER (
                PARTITION BY pgs.season, pgs.week
            ) as week_avg_yards
        FROM mart.player_game_stats pgs
        WHERE pgs.rushing_yards > 0
          AND pgs.position IN ('RB', 'FB', 'QB', 'WR')
          AND pgs.season BETWEEN %s AND %s
    )
    SELECT * FROM player_stats
    WHERE avg_rushing_l3 IS NOT NULL
    ORDER BY season, week, stat_yards DESC
    """
    
    df = pd.read_sql(query, conn, params=[start_season, end_season])
    conn.close()
    
    logger.info(f"Loaded {len(df)} rushing performances")
    logger.info(f"TD rate: {100*df['scored_td'].mean():.1f}%")
    
    return df


def prepare_features(df):
    """Prepare features and targets"""
    # Feature columns
    feature_cols = ['carries', 'avg_rushing_l3', 'season_avg', 'week', 'week_avg_yards']
    
    # Fill missing values
    for col in ['avg_rushing_l3', 'season_avg', 'week_avg_yards']:
        df[col] = df[col].fillna(df[col].median())
    
    X = df[feature_cols].values
    
    # Targets - handle negative yards
    y_yards = df['stat_yards'].values.astype(np.float32)
    y_td = df['scored_td'].values.astype(np.float32)
    
    # Apply log transform to yards (shift to positive first)
    yards_min = y_yards.min()
    if yards_min < 0:
        yards_shift = abs(yards_min) + 10
    else:
        yards_shift = 0
    y_yards_transformed = np.log1p(y_yards + yards_shift)
    
    return X, y_yards_transformed, y_td, yards_shift


def train_model(model, train_loader, val_loader, device, args):
    """Train the multi-output BNN"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # KL annealing schedule
    def kl_weight(epoch):
        return min(1.0, epoch / 20)  # Linear annealing over 20 epochs
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logger.info(f"Training on {device} for {args.epochs} epochs")
    logger.info(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_yards_loss = 0
        train_td_loss = 0
        train_kl = 0
        
        for batch_x, batch_y_yards, batch_y_td in train_loader:
            batch_x = batch_x.to(device)
            batch_y_yards = batch_y_yards.to(device)
            batch_y_td = batch_y_td.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with sampling
            yards_pred, td_logits, noise_std = model(batch_x, sample=True)
            
            # Negative log likelihood for yards (Gaussian)
            yards_nll = 0.5 * torch.mean(
                torch.log(2 * np.pi * noise_std**2) + 
                (batch_y_yards - yards_pred)**2 / noise_std**2
            )
            
            # Binary cross entropy for TDs
            td_loss = F.binary_cross_entropy_with_logits(td_logits, batch_y_td)
            
            # KL divergence
            kl_div = model.kl_divergence() / len(train_loader.dataset)
            
            # Total loss with KL annealing
            loss = yards_nll + td_loss + kl_weight(epoch) * args.kl_scale * kl_div
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_yards_loss += yards_nll.item()
            train_td_loss += td_loss.item()
            train_kl += kl_div.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_yards_mae = 0
        val_td_acc = 0
        
        with torch.no_grad():
            for batch_x, batch_y_yards, batch_y_td in val_loader:
                batch_x = batch_x.to(device)
                batch_y_yards = batch_y_yards.to(device)
                batch_y_td = batch_y_td.to(device)
                
                # Multiple forward passes for uncertainty
                n_samples = 10
                yards_samples = []
                td_probs = []
                
                for _ in range(n_samples):
                    yards_pred, td_logits, _ = model(batch_x, sample=True)
                    yards_samples.append(yards_pred)
                    td_probs.append(torch.sigmoid(td_logits))
                
                # Average predictions
                yards_mean = torch.stack(yards_samples).mean(dim=0)
                td_prob_mean = torch.stack(td_probs).mean(dim=0)
                
                # Validation metrics
                val_yards_mae += torch.mean(torch.abs(yards_mean - batch_y_yards)).item()
                val_td_acc += ((td_prob_mean > 0.5).float() == batch_y_td).float().mean().item()
                
                # Validation loss (simplified)
                val_loss += F.mse_loss(yards_mean, batch_y_yards).item()
        
        # Average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_yards_mae = val_yards_mae / len(val_loader)
        avg_val_td_acc = val_td_acc / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Checkpoint if best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if args.checkpoint_dir:
                checkpoint_path = Path(args.checkpoint_dir) / f'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_yards_mae': avg_val_yards_mae,
                    'val_td_acc': avg_val_td_acc
                }, checkpoint_path)
        
        # Save periodic checkpoints
        if args.checkpoint_dir and epoch % 50 == 0 and epoch > 0:
            checkpoint_path = Path(args.checkpoint_dir) / f'epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
        
        # Logging
        elapsed = time.time() - start_time
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Yards MAE: {avg_val_yards_mae:.2f} | "
                f"Val TD Acc: {avg_val_td_acc:.2%} | "
                f"Time: {elapsed:.1f}s"
            )
            
            # GPU memory monitoring
            if device.type == 'cuda':
                logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, device, yards_shift):
    """Evaluate the trained model"""
    model.eval()
    
    all_yards_true = []
    all_yards_pred = []
    all_yards_std = []
    all_td_true = []
    all_td_pred = []
    
    with torch.no_grad():
        for batch_x, batch_y_yards, batch_y_td in test_loader:
            batch_x = batch_x.to(device)
            
            # Multiple forward passes for uncertainty
            n_samples = 100
            yards_samples = []
            td_samples = []
            
            for _ in range(n_samples):
                yards_pred, td_logits, _ = model(batch_x, sample=True)
                yards_samples.append(yards_pred.cpu())
                td_samples.append(torch.sigmoid(td_logits).cpu())
            
            yards_samples = torch.stack(yards_samples)  # (n_samples, batch_size)
            td_samples = torch.stack(td_samples)
            
            # Get statistics
            yards_mean = yards_samples.mean(dim=0)
            yards_std = yards_samples.std(dim=0)
            td_prob_mean = td_samples.mean(dim=0)
            
            all_yards_true.append(batch_y_yards.numpy())
            all_yards_pred.append(yards_mean.numpy())
            all_yards_std.append(yards_std.numpy())
            all_td_true.append(batch_y_td.numpy())
            all_td_pred.append(td_prob_mean.numpy())
    
    # Concatenate all batches
    yards_true = np.concatenate(all_yards_true)
    yards_pred = np.concatenate(all_yards_pred)
    yards_std = np.concatenate(all_yards_std)
    td_true = np.concatenate(all_td_true)
    td_pred = np.concatenate(all_td_pred)
    
    # Transform yards back to original scale
    yards_true_original = np.expm1(yards_true) - yards_shift
    yards_pred_original = np.expm1(yards_pred) - yards_shift
    
    # Calculate metrics
    yards_mae = np.mean(np.abs(yards_pred_original - yards_true_original))
    yards_rmse = np.sqrt(np.mean((yards_pred_original - yards_true_original)**2))
    
    # 90% prediction interval coverage
    yards_lower = np.expm1(yards_pred - 1.645 * yards_std) - yards_shift
    yards_upper = np.expm1(yards_pred + 1.645 * yards_std) - yards_shift
    coverage = np.mean((yards_true_original >= yards_lower) & (yards_true_original <= yards_upper))
    
    # TD metrics
    td_accuracy = np.mean((td_pred > 0.5) == td_true)
    td_brier = np.mean((td_pred - td_true)**2)
    
    results = {
        'yards_mae': float(yards_mae),
        'yards_rmse': float(yards_rmse),
        'yards_coverage_90': float(coverage),
        'yards_mean_std': float(yards_std.mean()),
        'td_accuracy': float(td_accuracy),
        'td_brier_score': float(td_brier),
        'td_mean_prob': float(td_pred.mean()),
        'td_actual_rate': float(td_true.mean())
    }
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Rushing Yards:")
    logger.info(f"  MAE: {yards_mae:.2f} yards")
    logger.info(f"  RMSE: {yards_rmse:.2f} yards")
    logger.info(f"  90% Coverage: {coverage:.1%}")
    logger.info(f"  Mean Uncertainty: {yards_std.mean():.2f}")
    logger.info(f"\nRushing TDs:")
    logger.info(f"  Accuracy: {td_accuracy:.1%}")
    logger.info(f"  Brier Score: {td_brier:.4f}")
    logger.info(f"  Mean Pred Prob: {td_pred.mean():.1%}")
    logger.info(f"  Actual TD Rate: {td_true.mean():.1%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-Output BNN for NFL Rushing')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--kl-scale', type=float, default=0.01)
    parser.add_argument('--n-experts', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='models/bayesian/bnn_multioutput_checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/training')
    parser.add_argument('--output-dir', type=str, default='models/bayesian')
    
    args = parser.parse_args()
    
    # Setup directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    df = load_data(start_season=2020, end_season=2024)
    
    # Prepare features
    X, y_yards, y_td, yards_shift = prepare_features(df)
    
    # Train/test split
    X_train, X_test, y_yards_train, y_yards_test, y_td_train, y_td_test = train_test_split(
        X, y_yards, y_td, test_size=0.2, random_state=42, stratify=y_td
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_yards_train = torch.FloatTensor(y_yards_train)
    y_td_train = torch.FloatTensor(y_td_train)
    
    X_test = torch.FloatTensor(X_test)
    y_yards_test = torch.FloatTensor(y_yards_test)
    y_td_test = torch.FloatTensor(y_td_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_yards_train, y_td_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_test, y_yards_test, y_td_test)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    model = MultiOutputMoEBNN(
        input_dim=X_train.shape[1],
        n_experts=args.n_experts,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Number of experts: {args.n_experts}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, args)
    
    # Load best model for evaluation
    if args.checkpoint_dir:
        best_checkpoint = Path(args.checkpoint_dir) / 'best_model.pt'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Evaluate
    results = evaluate_model(model, val_loader, device, yards_shift)
    
    # Save final model and results
    final_model_path = Path(args.output_dir) / 'bnn_multioutput_v1_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'yards_shift': yards_shift,
        'args': vars(args),
        'results': results,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, final_model_path)
    
    logger.info(f"\n✅ Model saved to {final_model_path}")
    
    # Save results JSON
    results_path = Path(args.output_dir) / 'bnn_multioutput_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # GPU memory final report
    if device.type == 'cuda':
        logger.info(f"\nFinal GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        logger.info(f"Peak GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == '__main__':
    main()