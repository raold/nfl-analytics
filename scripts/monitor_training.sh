#!/bin/bash
# Monitor CQL training progress
# Usage: ./monitor_training.sh

echo "=== CQL Training Monitor ==="
echo ""

while true; do
    clear
    echo "=== CQL Training Monitor ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""

    # Queue status
    echo "📊 Queue Status:"
    PENDING=$(redis-cli zcard training_queue 2>/dev/null)
    echo "  Pending tasks: ${PENDING}"
    echo ""

    # Completed runs
    COMPLETED=$(ls -1 models/cql 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ Completed runs: ${COMPLETED}/4"
    echo ""

    # Latest training progress
    echo "🔄 Current Progress:"
    tail -5 worker_m4.log 2>/dev/null | grep "Epoch" | tail -1
    echo ""

    # Completed run summaries
    if [ -d "models/cql" ]; then
        echo "📈 Completed Runs:"
        for run_dir in models/cql/*/; do
            if [ -f "${run_dir}metadata.json" ]; then
                RUN_ID=$(basename "$run_dir")
                ALPHA=$(jq -r '.config.alpha' "${run_dir}metadata.json" 2>/dev/null)
                LR=$(jq -r '.config.lr' "${run_dir}metadata.json" 2>/dev/null)
                LOSS=$(jq -r '.latest_metrics.loss' "${run_dir}metadata.json" 2>/dev/null)
                CQL_LOSS=$(jq -r '.latest_metrics.cql_loss' "${run_dir}metadata.json" 2>/dev/null)
                Q_MEAN=$(jq -r '.latest_metrics.q_mean' "${run_dir}metadata.json" 2>/dev/null)

                printf "  %s: alpha=%.1f lr=%.0e | Loss=%.3f CQL=%.3f Q=%.3f\n" \
                    "$RUN_ID" "$ALPHA" "$LR" "$LOSS" "$CQL_LOSS" "$Q_MEAN" 2>/dev/null
            fi
        done
    fi
    echo ""
    echo "Press Ctrl+C to exit"

    sleep 5
done
