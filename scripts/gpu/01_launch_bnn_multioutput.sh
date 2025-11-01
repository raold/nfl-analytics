#!/bin/bash
# Priority 1: BNN Multi-Output Training
# Expected time: 2-3 hours on RTX 4090

echo "🔥 Starting BNN Multi-Output Training on RTX 4090"
echo "=============================================="
echo "This will kill any existing Mac training and restart on GPU"
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ CUDA not available! Please check your PyTorch installation"
    exit 1
fi

# Kill any existing processes (if running)
pkill -f "bnn_multioutput_v1.py" 2>/dev/null

# Create directories
mkdir -p models/bayesian/bnn_multioutput_checkpoints
mkdir -p logs/training

# Start training with GPU optimization
echo "Starting training..."
python py/models/bnn_multioutput_v1.py \
    --device cuda \
    --batch-size 128 \
    --epochs 200 \
    --learning-rate 0.001 \
    --checkpoint-dir models/bayesian/bnn_multioutput_checkpoints \
    --log-dir logs/training \
    --mixed-precision \
    --num-workers 8 \
    --pin-memory

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Git commit results
    git add models/bayesian/bnn_multioutput* logs/training/
    git commit -m "feat: complete BNN Multi-Output training on RTX 4090"
    git push origin main
else
    echo "❌ Training failed! Check logs/training/bnn_multioutput_v1.log"
fi