#!/bin/bash
#
# Run Batched Multi-Season Backtest: Ensemble v7.0
# ================================================
#
# Runs 10 batches of ~160 games each to avoid R session crashes.
# Total: 1,632 games across 2022-2024 seasons.
#
# Usage:
#   bash scripts/run_batched_backtest_v7.sh
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 17, 2025
# Version: 1.0

set -e  # Exit on error

# Configuration
TOTAL_BATCHES=10
SEASONS="2022 2023 2024"
OUTPUT_DIR="results"
LOG_DIR="results/logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "==============================================="
echo "Batched Multi-Season Backtest: Ensemble v7.0"
echo "==============================================="
echo "Batches: $TOTAL_BATCHES"
echo "Seasons: $SEASONS"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Run each batch
for batch in $(seq 1 $TOTAL_BATCHES); do
    echo ""
    echo "=============================================="
    echo "Running Batch $batch/$TOTAL_BATCHES"
    echo "=============================================="
    echo "Start time: $(date)"

    # Run batch
    log_file="$LOG_DIR/batch_${batch}_of_${TOTAL_BATCHES}.log"

    if python scripts/backtest_ensemble_v7_multi_season.py \
        --seasons $SEASONS \
        --batch $batch \
        --total-batches $TOTAL_BATCHES \
        > "$log_file" 2>&1; then

        echo "✅ Batch $batch completed successfully"
        echo "   Log: $log_file"

        # Check if output file was created
        output_file="$OUTPUT_DIR/backtest_v7_multi_season_results_batch${batch}of${TOTAL_BATCHES}.csv"
        if [ -f "$output_file" ]; then
            num_predictions=$(tail -n +2 "$output_file" | wc -l | tr -d ' ')
            echo "   Predictions: $num_predictions"
        else
            echo "   ⚠️  Output file not found: $output_file"
        fi
    else
        echo "❌ Batch $batch FAILED"
        echo "   See log for details: $log_file"
        echo ""
        echo "=== Last 50 lines of log ==="
        tail -50 "$log_file"
        echo "============================"
        echo ""
        echo "Stopping batch processing due to failure"
        exit 1
    fi

    echo "End time: $(date)"

    # Brief pause between batches
    if [ $batch -lt $TOTAL_BATCHES ]; then
        echo "Pausing 5 seconds before next batch..."
        sleep 5
    fi
done

echo ""
echo "=============================================="
echo "All Batches Complete!"
echo "=============================================="
echo "Total batches: $TOTAL_BATCHES"
echo ""

# Aggregate results
echo "Aggregating batch results..."
if python scripts/aggregate_batch_results_v7.py \
    --batches $TOTAL_BATCHES \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/aggregation.log" 2>&1; then

    echo "✅ Aggregation completed successfully"
    echo "   Log: $LOG_DIR/aggregation.log"
    echo ""
    echo "=== Final Results ==="
    cat "$LOG_DIR/aggregation.log" | grep -A 50 "MULTI-SEASON BACKTEST RESULTS"
    echo "===================="
else
    echo "❌ Aggregation FAILED"
    echo "   See log for details: $LOG_DIR/aggregation.log"
    exit 1
fi

echo ""
echo "✅ Batched backtest complete!"
echo ""
echo "Final results:"
echo "  - Combined CSV: $OUTPUT_DIR/backtest_v7_multi_season_results_FINAL.csv"
echo "  - Analysis JSON: $OUTPUT_DIR/backtest_v7_multi_season_analysis_FINAL.json"
echo "  - Logs directory: $LOG_DIR"
echo ""
echo "Completion time: $(date)"
