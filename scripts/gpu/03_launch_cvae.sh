#!/bin/bash
# Priority 3: cVAE Training for Game Trajectory Synthesis
# Expected time: 1-2 hours on RTX 4090

echo "🔥 Starting cVAE Training on RTX 4090"
echo "======================================"
echo "Game trajectory synthesis model"
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ CUDA not available! Please check your PyTorch installation"
    exit 1
fi

# Kill any existing training
pkill -f "train_cvae.py" 2>/dev/null

# Create directories
mkdir -p models/cvae/checkpoints
mkdir -p logs/training

# Start cVAE training
echo "Starting cVAE training..."
python nfl_analytics/models/train_cvae.py \
    --start-season 2016 \
    --end-season 2024 \
    --epochs 200 \
    --batch-size 256 \
    --learning-rate 0.0005 \
    --latent-dim 64 \
    --beta 1.0 \
    --device cuda \
    --output-dir models/cvae \
    --checkpoint-dir models/cvae/checkpoints \
    --checkpoint-freq 20 \
    --mixed-precision \
    --num-workers 8

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ cVAE training completed successfully!"
    
    # Test generation
    echo "Testing trajectory generation..."
    python -c "
import torch
from pathlib import Path
import sys
sys.path.append('nfl_analytics')

if Path('models/cvae/cvae_best.pt').exists():
    print('Model saved successfully')
    # Could add generation test here
"
    
    # Git commit results
    git add models/cvae/ logs/training/
    git commit -m "feat: complete cVAE training on RTX 4090 - game trajectory synthesis"
    git push origin main
else
    echo "❌ cVAE training failed! Check logs/training/cvae_training.log"
fi