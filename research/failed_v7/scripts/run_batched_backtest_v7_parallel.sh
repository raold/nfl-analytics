#!/bin/bash
#
# Run Batched Multi-Season Backtest: Ensemble v7.0 (PARALLEL)
# ===========================================================
#
# Runs 40 batches of ~40 games each with 5 parallel processes.
# Total: 1,632 games across 2022-2024 seasons.
#
# Usage:
#   bash scripts/run_batched_backtest_v7_parallel.sh
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 17, 2025
# Version: 1.1 (Parallel)

set -e  # Exit on error

# Configuration
TOTAL_BATCHES=40
PARALLEL_JOBS=1  # Run sequentially to avoid R session conflicts
SEASONS="2022 2023 2024"
OUTPUT_DIR="results"
LOG_DIR="results/logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "Batched Multi-Season Backtest: Ensemble v7.0 (PARALLEL)"
echo "=========================================================="
echo "Total batches: $TOTAL_BATCHES"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Games per batch: ~40"
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

# Function to run a single batch
run_batch() {
    local batch=$1
    local log_file="$LOG_DIR/batch_${batch}_of_${TOTAL_BATCHES}.log"

    echo "[$(date '+%H:%M:%S')] Starting batch $batch/$TOTAL_BATCHES..."

    if python scripts/backtest_ensemble_v7_multi_season.py \
        --seasons $SEASONS \
        --batch $batch \
        --total-batches $TOTAL_BATCHES \
        > "$log_file" 2>&1; then

        echo "[$(date '+%H:%M:%S')] ✅ Batch $batch completed"
        return 0
    else
        echo "[$(date '+%H:%M:%S')] ❌ Batch $batch FAILED"
        return 1
    fi
}

export -f run_batch
export SEASONS OUTPUT_DIR LOG_DIR TOTAL_BATCHES

# Run batches in parallel using GNU parallel or xargs
echo ""
echo "=========================================================="
echo "Running $TOTAL_BATCHES batches with $PARALLEL_JOBS parallel jobs"
echo "=========================================================="
echo ""

start_time=$(date +%s)

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution..."
    seq 1 $TOTAL_BATCHES | parallel -j $PARALLEL_JOBS --bar run_batch {}
    parallel_exit_code=$?
else
    echo "Using xargs for parallel execution..."
    # Use xargs as fallback
    seq 1 $TOTAL_BATCHES | xargs -n 1 -P $PARALLEL_JOBS -I {} bash -c "run_batch {}"
    parallel_exit_code=$?
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))
elapsed_sec=$((elapsed % 60))

echo ""
echo "=========================================================="
echo "All Batches Complete!"
echo "=========================================================="
echo "Total time: ${elapsed_min}m ${elapsed_sec}s"
echo ""

# Check if any batches failed
failed_batches=0
for batch in $(seq 1 $TOTAL_BATCHES); do
    output_file="$OUTPUT_DIR/backtest_v7_multi_season_results_batch${batch}of${TOTAL_BATCHES}.csv"
    if [ ! -f "$output_file" ]; then
        echo "⚠️  Missing batch $batch: $output_file"
        ((failed_batches++))
    fi
done

if [ $failed_batches -gt 0 ]; then
    echo ""
    echo "❌ $failed_batches batches failed or incomplete"
    echo "Cannot proceed with aggregation"
    exit 1
fi

echo "✅ All $TOTAL_BATCHES batches completed successfully"
echo ""

# Aggregate results
echo "=========================================================="
echo "Aggregating Results"
echo "=========================================================="
echo ""

if python scripts/aggregate_batch_results_v7.py \
    --batches $TOTAL_BATCHES \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/aggregation.log" 2>&1; then

    echo "✅ Aggregation completed successfully"
    echo ""
    echo "=== Final Results ==="
    tail -100 "$LOG_DIR/aggregation.log" | grep -A 50 "MULTI-SEASON BACKTEST RESULTS"
    echo "===================="
else
    echo "❌ Aggregation FAILED"
    echo "   See log for details: $LOG_DIR/aggregation.log"
    exit 1
fi

echo ""
echo "=========================================================="
echo "✅ Batched Backtest Complete!"
echo "=========================================================="
echo ""
echo "Final results:"
echo "  - Combined CSV: $OUTPUT_DIR/backtest_v7_multi_season_results_FINAL.csv"
echo "  - Analysis JSON: $OUTPUT_DIR/backtest_v7_multi_season_analysis_FINAL.json"
echo "  - Logs directory: $LOG_DIR"
echo ""
echo "Total execution time: ${elapsed_min}m ${elapsed_sec}s"
echo "Completion time: $(date)"
