#!/bin/bash
# Quick Phase 2 Status Checker
# Usage: ./check_phase2_status.sh

echo "========================================="
echo "PHASE 2 TRAINING STATUS"
echo "========================================="
echo ""

# Check if training processes are running
echo "Active Processes:"
ps aux | grep -E "(bnn_simpler|bnn_mixture|autonomous_phase2)" | grep -v grep || echo "  No training processes found"
echo ""

# Check for completed models
echo "Completed Models:"
[ -f "models/bayesian/bnn_simpler_v2.pkl" ] && echo "  ✓ Phase 2.1: Simpler BNN" || echo "  ⏳ Phase 2.1: In progress..."
[ -f "models/bayesian/bnn_mixture_experts_v2.pkl" ] && echo "  ✓ Phase 2.2: Mixture-of-Experts" || echo "  ⏳ Phase 2.2: Pending..."
echo ""

# Check results
echo "Results:"
if [ -f "experiments/calibration/simpler_bnn_v2_results.json" ]; then
    echo "  Phase 2.1 Results:"
    python3 -c "import json; r=json.load(open('experiments/calibration/simpler_bnn_v2_results.json')); print(f'    90% Coverage: {r[\"coverage_90\"]:.1f}% (target: 90%)'); print(f'    MAE: {r[\"mae\"]:.1f} yards')"
else
    echo "  ⏳ Phase 2.1: No results yet"
fi
echo ""

if [ -f "experiments/calibration/mixture_experts_v2_results.json" ]; then
    echo "  Phase 2.2 Results:"
    python3 -c "import json; r=json.load(open('experiments/calibration/mixture_experts_v2_results.json')); print(f'    90% Coverage: {r[\"coverage_90\"]:.1f}% (target: 90%)'); print(f'    MAE: {r[\"mae\"]:.1f} yards')"
else
    echo "  ⏳ Phase 2.2: No results yet"
fi
echo ""

# Check logs
echo "Recent Log Activity:"
echo "  Phase 2.1:"
[ -f "logs/bnn_simpler_v2_"* ] && tail -3 logs/bnn_simpler_v2_* 2>/dev/null | head -3 || echo "    No log found"
echo ""
echo "  Autonomous Runner:"
[ -f "logs/autonomous_phase2_"* ] && tail -3 logs/autonomous_phase2_* 2>/dev/null | head -3 || echo "    No log found"
echo ""

echo "========================================="
echo "To monitor live:"
echo "  tail -f logs/bnn_simpler_v2_*.log"
echo "  tail -f logs/autonomous_phase2_*.log"
echo "========================================="
