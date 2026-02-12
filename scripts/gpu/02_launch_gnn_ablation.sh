#!/bin/bash
# Priority 2: GNN Ablation Study
# Expected time: 3-4 hours on RTX 4090

echo "🔥 Starting GNN Ablation Study on RTX 4090"
echo "==========================================="
echo "9 configurations × 50 epochs each"
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "❌ CUDA not available! Please check your PyTorch installation"
    exit 1
fi

# Kill any existing CPU training
pkill -f "gnn_ablation_study.py" 2>/dev/null

# Create output directories
mkdir -p models/gnn/ablation
mkdir -p logs/training

# Start ablation study
echo "Starting ablation study..."
python py/models/gnn_ablation_study.py \
    --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 \
    --epochs 50 \
    --batch-size 256 \
    --device cuda \
    --output-dir models/gnn/ablation \
    --checkpoint-freq 10 \
    --mixed-precision \
    --num-workers 8

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Ablation study completed successfully!"
    
    # Generate summary report
    python -c "
import json
from pathlib import Path

results = []
for config_dir in Path('models/gnn/ablation').glob('config_*/'):
    if (config_dir / 'metrics.json').exists():
        with open(config_dir / 'metrics.json') as f:
            metrics = json.load(f)
            results.append({
                'config': config_dir.name,
                'best_val_acc': metrics.get('best_val_accuracy', 0),
                'best_val_auc': metrics.get('best_val_auc', 0)
            })

# Sort by validation accuracy
results.sort(key=lambda x: x['best_val_acc'], reverse=True)

print('\n📊 Ablation Study Results:')
print('=' * 50)
for r in results[:5]:  # Top 5
    print(f\"{r['config']}: Acc={r['best_val_acc']:.3f}, AUC={r['best_val_auc']:.3f}\")
"
    
    # Git commit results
    git add models/gnn/ablation/ logs/training/
    git commit -m "feat: complete GNN ablation study on RTX 4090 - 9 configs"
    git push origin main
else
    echo "❌ Ablation study failed! Check logs/training/gnn_ablation.log"
fi