#!/bin/bash
#
# Watch for compute tasks to finish their round and restart with high intensity
#

echo "🔍 Watching compute tasks for round completion"
echo "=============================================="

# Task 1: PID 64774 (low -> high)
PID1=64774
CMD1="python run_compute.py --intensity high --workers 1"

# Task 2: PID 52241 (medium -> high)
PID2=52241
CMD2="python run_compute.py --intensity high"

# Check if processes are still running
check_pid() {
    kill -0 $1 2>/dev/null
    return $?
}

# Initial status
echo ""
if check_pid $PID1; then
    echo "✅ PID $PID1 is running (low intensity)"
else
    echo "❌ PID $PID1 has already finished"
fi

if check_pid $PID2; then
    echo "✅ PID $PID2 is running (medium intensity)"
else
    echo "❌ PID $PID2 has already finished"
fi

echo ""
echo "📊 Monitoring for completion..."
echo "(Checking every 10 seconds)"
echo ""

# Monitor loop
UPGRADED1=false
UPGRADED2=false

while true; do
    # Check task 1
    if [ "$UPGRADED1" = false ]; then
        if ! check_pid $PID1; then
            echo "🏁 PID $PID1 completed! Starting with high intensity..."
            cd $(cd "$(dirname "")/.." \&\& pwd)
            nohup $CMD1 > compute_high_1.log 2>&1 &
            NEW_PID1=$!
            echo "✅ Started new process PID $NEW_PID1 with HIGH intensity"
            UPGRADED1=true
        fi
    fi

    # Check task 2
    if [ "$UPGRADED2" = false ]; then
        if ! check_pid $PID2; then
            echo "🏁 PID $PID2 completed! Starting with high intensity..."
            cd $(cd "$(dirname "")/.." \&\& pwd)
            nohup $CMD2 > compute_high_2.log 2>&1 &
            NEW_PID2=$!
            echo "✅ Started new process PID $NEW_PID2 with HIGH intensity"
            UPGRADED2=true
        fi
    fi

    # Check if both upgraded
    if [ "$UPGRADED1" = true ] && [ "$UPGRADED2" = true ]; then
        echo ""
        echo "🎉 Both tasks upgraded to HIGH intensity!"
        echo "New PIDs: $NEW_PID1 and $NEW_PID2"
        echo "Logs: compute_high_1.log and compute_high_2.log"
        exit 0
    fi

    # Status indicator
    if [ "$UPGRADED1" = false ] || [ "$UPGRADED2" = false ]; then
        printf "⏳ Waiting... (PID1: %s, PID2: %s)\r" \
               "$([ "$UPGRADED1" = true ] && echo "✅" || echo "⏳")" \
               "$([ "$UPGRADED2" = true ] && echo "✅" || echo "⏳")"
    fi

    sleep 10
done