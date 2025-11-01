#!/bin/bash
# Master script to run all GPU training jobs sequentially
# Total expected time: ~23-32 hours

echo "🚀 RTX 4090 GPU Training Pipeline"
echo "=================================="
echo "This will run all training jobs sequentially"
echo "Estimated total time: 23-32 hours"
echo ""
echo "Phases:"
echo "  1. Critical Models (5-7 hours)"
echo "  2. Experiment-RL Integration (9-14 hours)"
echo "  3. Hyperparameter Optimization (8-10 hours)"
echo "  4. Evaluation & Backtesting (1-1.5 hours)"
echo ""

# Function to display GPU stats
show_gpu_stats() {
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
}

# Function to run a training phase
run_phase() {
    local phase=$1
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Starting Phase $phase"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python scripts/gpu_training_orchestrator.py --phase $phase
    
    # Cool down between phases
    if [ $phase -lt 4 ]; then
        echo "Cooling down GPU for 60 seconds before next phase..."
        sleep 60
    fi
}

# Check if running on Windows with WSL
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows environment"
fi

# Initial GPU check
echo "Current GPU Status:"
show_gpu_stats
echo ""

# Ask for confirmation
read -p "Start training pipeline? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 1
fi

# Start time
start_time=$(date +%s)

# Run all phases
for phase in 1 2 3 4; do
    run_phase $phase
    
    # Check if user wants to continue
    if [ $phase -lt 4 ]; then
        echo ""
        read -p "Continue to Phase $((phase + 1))? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Pipeline paused at Phase $phase"
            break
        fi
    fi
done

# End time
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ GPU Training Pipeline Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total time: ${hours}h ${minutes}m"
echo ""
echo "Final GPU Status:"
show_gpu_stats

# Generate summary report
python scripts/aggregate_results.py --output reports/gpu_training_summary.json