#!/bin/bash
#
# BFS Experiment Runner - Execute All 6 Remaining Experiments in Parallel
#
# Experiment 1: Interactions (FAILED - see postmortem)
# Experiments 2-7: Remaining approaches to fix opponent feature regression
#
# Target: Find ANY approach with MAE < 40.20 (v2 baseline)
#
# Author: NFL Analytics Research Team
# Date: November 14, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "")/.." \&\& pwd)"
cd "$PROJECT_ROOT"

# Log directory
LOG_DIR="/tmp/bfs_experiments"
mkdir -p "$LOG_DIR"

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo ""
echo "========================================================================"
echo " BFS EXPERIMENTS 2-7: PARALLEL EXECUTION"
echo "========================================================================"
echo " Target: MAE < 40.20 (beat v2 baseline)"
echo " Baseline: v3.1 with opponent features = 57.37 MAE (-42.71% regression)"
echo " Timestamp: $TIMESTAMP"
echo " Log directory: $LOG_DIR"
echo "========================================================================"
echo ""

# Experiment metadata
declare -A EXPERIMENTS
EXPERIMENTS[2]="Defensive Clustering (20 min)"
EXPERIMENTS[3]="Temporal Opponent Features (25 min)"
EXPERIMENTS[4]="Sample Weighting (15 min)"
EXPERIMENTS[5]="L1 Regularization (25 min)"
EXPERIMENTS[6]="Two-Stage Modeling (20 min)"
EXPERIMENTS[7]="Ridge Residuals (30 min - MOST PROMISING)"

# Start experiments in background
echo "${BLUE}Launching 6 experiments in parallel...${NC}"
echo ""

for exp_num in 2 3 4 5 6 7; do
    script_map=(
        [2]="xgb_v3_2_def_clustering.py"
        [3]="xgb_v3_2_temporal_opp.py"
        [4]="xgb_v3_2_sample_weights.py"
        [5]="xgb_v3_2_l1_regularization.py"
        [6]="xgb_v3_2_two_stage.py"
        [7]="xgb_v3_2_ridge_residuals.py"
    )

    script="${script_map[$exp_num]}"
    log_file="${LOG_DIR}/exp${exp_num}_${TIMESTAMP}.log"

    echo "  [Exp $exp_num] ${EXPERIMENTS[$exp_num]}"
    echo "             Script: scripts/$script"
    echo "             Log: $log_file"

    # Launch in background
    python "scripts/$script" > "$log_file" 2>&1 &
    pid=$!
    echo "             PID: $pid"
    echo ""
done

echo "${BLUE}All experiments launched!${NC}"
echo ""
echo "========================================================================"
echo " MONITORING INSTRUCTIONS"
echo "========================================================================"
echo ""
echo "Track progress:"
echo "  tail -f $LOG_DIR/exp*_${TIMESTAMP}.log"
echo ""
echo "Check individual experiment:"
echo "  tail -f $LOG_DIR/exp2_${TIMESTAMP}.log  # Defensive Clustering"
echo "  tail -f $LOG_DIR/exp3_${TIMESTAMP}.log  # Temporal Opponent"
echo "  tail -f $LOG_DIR/exp4_${TIMESTAMP}.log  # Sample Weighting"
echo "  tail -f $LOG_DIR/exp5_${TIMESTAMP}.log  # L1 Regularization"
echo "  tail -f $LOG_DIR/exp6_${TIMESTAMP}.log  # Two-Stage"
echo "  tail -f $LOG_DIR/exp7_${TIMESTAMP}.log  # Ridge Residuals (most promising)"
echo ""
echo "Check which are still running:"
echo "  ps aux | grep 'xgb_v3_2_'"
echo ""
echo "Kill all experiments (if needed):"
echo "  pkill -f 'xgb_v3_2_'"
echo ""
echo "========================================================================"
echo " EXPECTED COMPLETION TIME"
echo "========================================================================"
echo ""
echo "  Fastest (Exp 4):  ~15 minutes"
echo "  Average:          ~22 minutes"
echo "  Slowest (Exp 7):  ~30 minutes"
echo ""
echo "  All complete by:  $(date -v+35M '+%H:%M:%S')"
echo ""
echo "========================================================================"
echo ""

# Wait for all background jobs to complete
wait

echo ""
echo "========================================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""

# Parse results
echo "${BLUE}Parsing results...${NC}"
echo ""

declare -A RESULTS

for exp_num in 2 3 4 5 6 7; do
    log_file="${LOG_DIR}/exp${exp_num}_${TIMESTAMP}.log"

    if [ -f "$log_file" ]; then
        # Extract MAE from log (pattern: "MAE: XX.XX yards")
        mae=$(grep -oP "(?<=^MAE: )[0-9]+\.[0-9]+" "$log_file" | tail -1 || echo "ERROR")

        RESULTS[$exp_num]="$mae"
    else
        RESULTS[$exp_num]="MISSING"
    fi
done

# Print results table
echo "========================================================================"
echo " RESULTS SUMMARY"
echo "========================================================================"
echo ""
printf "%-5s %-40s %-12s %-10s\n" "Exp" "Approach" "MAE (yards)" "Status"
echo "------------------------------------------------------------------------"

v2_baseline=40.20

for exp_num in 2 3 4 5 6 7; do
    mae="${RESULTS[$exp_num]}"
    approach="${EXPERIMENTS[$exp_num]}"

    if [ "$mae" = "ERROR" ] || [ "$mae" = "MISSING" ]; then
        status="${RED}FAILED${NC}"
    elif [ "$(echo "$mae < $v2_baseline" | bc -l)" -eq 1 ]; then
        status="${GREEN}SUCCESS${NC}"
    else
        status="${YELLOW}NO IMPROVEMENT${NC}"
    fi

    printf "%-5s %-40s %-12s " "$exp_num" "$approach" "$mae"
    echo -e "$status"
done

echo "------------------------------------------------------------------------"
echo "Baseline: v2 = 40.20 MAE (target to beat)"
echo "========================================================================"
echo ""

# Check for success
success_count=0
best_mae=100.0
best_exp=0

for exp_num in 2 3 4 5 6 7; do
    mae="${RESULTS[$exp_num]}"

    if [ "$mae" != "ERROR" ] && [ "$mae" != "MISSING" ]; then
        if [ "$(echo "$mae < $v2_baseline" | bc -l)" -eq 1 ]; then
            ((success_count++))

            if [ "$(echo "$mae < $best_mae" | bc -l)" -eq 1 ]; then
                best_mae="$mae"
                best_exp="$exp_num"
            fi
        fi
    fi
done

echo "========================================================================"
echo " FINAL DECISION"
echo "========================================================================"
echo ""

if [ $success_count -gt 0 ]; then
    echo -e "${GREEN}✓ SUCCESS: $success_count experiment(s) beat v2 baseline${NC}"
    echo ""
    echo "Best result:"
    echo "  Experiment $best_exp: ${EXPERIMENTS[$best_exp]}"
    echo "  MAE: $best_mae yards"
    echo "  Improvement: $(echo "scale=2; ($v2_baseline - $best_mae) / $v2_baseline * 100" | bc -l)%"
    echo ""
    echo "RECOMMENDATION: Deploy Experiment $best_exp to production ensemble"
    echo ""
else
    echo -e "${RED}✗ ALL EXPERIMENTS FAILED${NC}"
    echo ""
    echo "No approach beat v2 baseline (40.20 MAE)"
    echo ""
    echo "RECOMMENDATION: Abandon opponent features"
    echo "  - Use v2 baseline (no opponent features) in ensemble"
    echo "  - Focus on alternative approaches (temporal, causal, etc.)"
    echo ""
fi

echo "========================================================================"
echo ""

# Create final report
REPORT_FILE="docs/BFS_EXPERIMENT_2_7_REPORT_${TIMESTAMP}.md"

cat > "$REPORT_FILE" << EOF
# BFS Experiments 2-7: Final Report

**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Status**: Complete
**Goal**: Fix XGBoost v3.1 opponent feature regression (57.37 MAE → target < 40.20 MAE)

## Executive Summary

- **Success Count**: $success_count / 6 experiments
- **Best Result**: Experiment $best_exp (${EXPERIMENTS[$best_exp]}) - $best_mae MAE

## Results

| Exp | Approach | MAE (yards) | vs v2 Baseline | Status |
|-----|----------|-------------|----------------|--------|
EOF

for exp_num in 2 3 4 5 6 7; do
    mae="${RESULTS[$exp_num]}"
    approach="${EXPERIMENTS[$exp_num]}"

    if [ "$mae" = "ERROR" ] || [ "$mae" = "MISSING" ]; then
        improvement="N/A"
        status="FAILED"
    else
        improvement=$(echo "scale=2; ($v2_baseline - $mae) / $v2_baseline * 100" | bc -l)

        if [ "$(echo "$mae < $v2_baseline" | bc -l)" -eq 1 ]; then
            status="SUCCESS"
        else
            status="NO IMPROVEMENT"
        fi
    fi

    echo "| $exp_num | $approach | $mae | ${improvement}% | $status |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

## Baseline Comparison

- v2 Baseline (no opponent features): 40.20 MAE
- v3.1 Regression (with opponent features): 57.37 MAE (-42.71%)

## Recommendation

EOF

if [ $success_count -gt 0 ]; then
    cat >> "$REPORT_FILE" << EOF
**DEPLOY**: Experiment $best_exp (${EXPERIMENTS[$best_exp]}) achieved $best_mae MAE

### Next Steps
1. Integrate into ensemble v6.2 (20% weight)
2. Run full backtest (2022-2024)
3. A/B test on 2025 season

### Model Files
- Model: models/xgboost/xgb_v3_2_*.json
- Importance: models/xgboost/xgb_v3_2_*_importance.csv
EOF
else
    cat >> "$REPORT_FILE" << EOF
**ABANDON**: No approach fixed the opponent feature regression

### Next Steps
1. Use v2 baseline (no opponent features) in ensemble
2. Focus on alternative improvements:
   - Temporal features (recent form)
   - Causal inference (matchup adjustments)
   - Advanced interactions (QB-WR chemistry)
   - Hierarchical models (position-specific)
EOF
fi

cat >> "$REPORT_FILE" << EOF

## Experiment Details

EOF

for exp_num in 2 3 4 5 6 7; do
    log_file="${LOG_DIR}/exp${exp_num}_${TIMESTAMP}.log"

    cat >> "$REPORT_FILE" << EOF
### Experiment $exp_num: ${EXPERIMENTS[$exp_num]}

**MAE**: ${RESULTS[$exp_num]} yards

**Log**: $log_file

EOF

    if [ -f "$log_file" ]; then
        echo '```' >> "$REPORT_FILE"
        tail -50 "$log_file" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

---

**Generated**: $(date '+%Y-%m-%d %H:%M:%S')
**Total Runtime**: ~30 minutes (parallel execution)
EOF

echo "Report saved: $REPORT_FILE"
echo ""
