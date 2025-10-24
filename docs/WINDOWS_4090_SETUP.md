# Windows RTX 4090 Setup for GNN Training

## Overview
Transfer GNN training from M1 Mac to Windows desktop with RTX 4090 for 5-10x speedup.

## Prerequisites
- Windows desktop with RTX 4090
- CUDA 12.1+ drivers installed
- Git installed
- Python 3.11+

## Setup Steps

### 1. Clone Repository
```bash
# On Windows
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics
git checkout main  # Start from main branch
```

### 2. Install Dependencies

#### Install uv package manager
```bash
# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

#### Install PyTorch with CUDA support
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Install project dependencies
```bash
uv sync
```

### 3. Transfer Checkpoint Files

Copy these files from M1 Mac to Windows:

**From M1:**
```bash
# On M1 Mac - create transfer package
tar -czf gnn_checkpoint_transfer.tar.gz \
  models/gnn/checkpoints/ \
  py/models/train_hierarchical_gnn.py \
  py/models/hierarchical_gnn_v1.py \
  py/models/gnn_graph_builder.py
```

**To Windows:**
Transfer `gnn_checkpoint_transfer.tar.gz` via:
- USB drive
- Network share
- Cloud storage (Dropbox, Google Drive, etc.)

Extract on Windows:
```bash
tar -xzf gnn_checkpoint_transfer.tar.gz
```

### 4. Database Connection

The training script connects to PostgreSQL on M1 Mac at:
```
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

**Option A: SSH Tunnel (Recommended)**
```bash
# On Windows - create SSH tunnel to M1 Mac
ssh -L 5544:localhost:5544 your-user@your-m1-ip
```

**Option B: Direct Connection**
Update connection string in training script to point to M1 Mac's IP:
```python
db_url = "postgresql://dro:sicillionbillions@<M1-IP>:5544/devdb01"
```

### 5. Verify CUDA Setup
```bash
# On Windows
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

### 6. Resume Training

```bash
# Resume from checkpoint (epoch 4)
uv run python py/models/train_hierarchical_gnn.py \
  --seasons 2020 2021 2022 2023 2024 2025 \
  --epochs 100 \
  --embedding-dim 64 \
  --device cuda \
  --resume \
  --checkpoint-dir models/gnn/checkpoints
```

## Performance Expectations

| Metric | M1 GPU (MPS) | RTX 4090 (CUDA) | Speedup |
|--------|--------------|-----------------|---------|
| Sec/game | 2.5-2.7 | 0.25-0.5 | 5-10x |
| Epoch time | ~47 min | ~5-9 min | 5-10x |
| Total (100 epochs) | ~80 hours | ~8-16 hours | 5-10x |

## Monitoring

```bash
# Monitor training progress
tail -f logs/gnn/hierarchical_gnn_training_2020_2025.log

# Or use monitoring script
./scripts/monitor_gnn_training.sh
```

## Checkpoint Info

Current checkpoint (from M1):
- Epoch: 4/100
- Best validation Brier: 0.2311
- Model parameters: 74,369
- Graph: 526 nodes, 2,217 edges

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or embedding dimension:
```bash
uv run python py/models/train_hierarchical_gnn.py \
  --embedding-dim 32 \  # Reduce from 64
  ...
```

### Database Connection Failed
- Check SSH tunnel is running
- Verify M1 Mac is accessible on network
- Check PostgreSQL allows remote connections

### Slow Performance (Not Using GPU)
```bash
# Verify device is set to 'cuda'
# Check CUDA installation with nvidia-smi
nvidia-smi
```

## Alternative: Fresh Training on Windows

If checkpoint transfer has issues, you can start fresh training:

```bash
# Start new training run on 4090
uv run python py/models/train_hierarchical_gnn.py \
  --seasons 2020 2021 2022 2023 2024 2025 \
  --epochs 100 \
  --embedding-dim 64 \
  --device cuda \
  --checkpoint-dir models/gnn/checkpoints_4090
```

With 4090's speed, starting fresh only costs ~16 hours vs ~80 hours saved.

## Files Overview

**Required Python files:**
- `py/models/train_hierarchical_gnn.py` - Training script
- `py/models/hierarchical_gnn_v1.py` - Model architecture
- `py/models/gnn_graph_builder.py` - Graph construction

**Checkpoint files:**
- `models/gnn/checkpoints/checkpoint_latest.pt` - Resume point
- `models/gnn/checkpoints/checkpoint_epoch_4.pt` - Backup

**Database schema:**
- Uses `mart.player_hierarchy` materialized view
- Uses `mart.player_game_stats` materialized view
- Queries `games` table for matchups

## Next Steps After Training

1. Copy trained model back to M1: `models/gnn/hierarchical_gnn_best.pt`
2. Run evaluation against baselines
3. Integrate with Bayesian ensemble
4. Document results for dissertation
