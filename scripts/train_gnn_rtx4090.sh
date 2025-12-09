#!/bin/bash
# =============================================================================
# GNN Training Script for RTX 4090
# =============================================================================
#
# Optimized for NVIDIA RTX 4090 (24GB VRAM):
# - Large batch size (512) to maximize GPU utilization
# - Mixed precision (FP16) for 2-3x speedup
# - Per-season graphs to avoid temporal data leakage
# - 51-dimensional player features (NextGen + Bayesian ratings)
#
# Expected performance:
# - Time per epoch: ~12 seconds
# - Total training time (50 epochs): ~10 minutes
# - Target AUC: >0.57 (baseline: 0.48 random)
#
# Usage:
#   ./scripts/train_gnn_rtx4090.sh
#
# Requirements:
#   - NVIDIA RTX 4090 with CUDA drivers
#   - PyTorch with CUDA support
#   - PostgreSQL database running on localhost:5544
#
# =============================================================================

set -e  # Exit on error

cd "$(dirname "$0")/.."  # Navigate to project root

echo "=============================================="
echo "  GNN Training - RTX 4090 Optimized"
echo "=============================================="

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set environment variables
export DB_PASSWORD="${DB_PASSWORD:-sicillionbillions}"
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  Seasons: 2020-2024 (5 seasons)"
echo "  Batch size: 512 (optimized for 24GB VRAM)"
echo "  Epochs: 50"
echo "  Learning rate: 0.001"
echo "  Mixed precision: Enabled (FP16)"
echo "  Player features: 51 dimensions"
echo "  Per-season graphs: Enabled (no temporal leakage)"
echo ""

# Run training
python py/models/train_hierarchical_gnn.py \
    --seasons 2020 2021 2022 2023 2024 \
    --batch-size 512 \
    --epochs 50 \
    --lr 0.001 \
    --embedding-dim 64 \
    --hidden-dim 128 \
    --num-message-rounds 3 \
    --use-amp \
    --device cuda \
    --output-dir models/gnn \
    --checkpoint-dir models/gnn/checkpoints \
    --checkpoint-every 5

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
echo "Model saved to: models/gnn/hierarchical_gnn_best.pt"
echo ""
echo "Next steps:"
echo "  1. Check test AUC (target: >0.57)"
echo "  2. If AUC >= 0.57: Proceed to Phase 6b (GNN integration)"
echo "  3. If AUC < 0.57: Review training logs and adjust hyperparameters"
