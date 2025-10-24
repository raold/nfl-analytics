#!/bin/bash

# Monitor Hierarchical GNN Training Progress
# Usage: ./scripts/monitor_gnn_training.sh

LOG_FILE="logs/gnn/hierarchical_gnn_training_2020_2025.log"

echo "=================================================="
echo "Hierarchical GNN Training Monitor"
echo "=================================================="
echo ""

# Check if training is running
if pgrep -f "train_hierarchical_gnn.py" > /dev/null; then
    echo "✓ Training is RUNNING"
    echo ""
else
    echo "✗ Training is NOT running"
    echo ""
    exit 1
fi

# Show latest epoch results
echo "Latest Epoch Results:"
echo "--------------------"
grep "Epoch" "$LOG_FILE" | tail -5
echo ""

# Show best validation Brier
echo "Best Validation Brier:"
echo "---------------------"
grep "New best" "$LOG_FILE" | tail -1
echo ""

# Show test results if available
if grep -q "Test Set Results" "$LOG_FILE"; then
    echo "Test Set Results:"
    echo "-----------------"
    grep -A 10 "Test Set Results" "$LOG_FILE" | tail -11
    echo ""
fi

# Estimate completion time
CURRENT_EPOCH=$(grep "Epoch" "$LOG_FILE" | tail -1 | awk '{print $2}' | cut -d'/' -f1)
TOTAL_EPOCHS=$(grep "Epochs:" "$LOG_FILE" | head -1 | awk '{print $2}')

if [ ! -z "$CURRENT_EPOCH" ] && [ ! -z "$TOTAL_EPOCHS" ]; then
    REMAINING=$((TOTAL_EPOCHS - CURRENT_EPOCH))
    echo "Progress: Epoch $CURRENT_EPOCH/$TOTAL_EPOCHS ($REMAINING remaining)"
    echo ""
fi

echo "Monitor live output:"
echo "  tail -f $LOG_FILE"
echo ""
echo "=================================================="
