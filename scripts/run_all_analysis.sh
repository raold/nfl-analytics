#!/bin/bash
set -e

source .venv/bin/activate
export POSTGRES_PASSWORD=sicillionbillions

echo "=== Regenerating All Tables and Figures ==="
echo ""

# Analysis scripts
echo "[1/16] Generating results tables..."
python -m nfl_analytics.analysis.generate_results_tables || echo "  [SKIP] generate_results_tables failed"

echo "[2/16] Generating OOS table..."
python -m nfl_analytics.analysis.generate_oos_table || echo "  [SKIP] generate_oos_table failed"

echo "[3/16] Generating betting metrics..."
python -m nfl_analytics.analysis.generate_betting_metrics || echo "  [SKIP] generate_betting_metrics failed"

echo "[4/16] Generating weather tables..."
python -m nfl_analytics.analysis.generate_weather_tables || echo "  [SKIP] generate_weather_tables failed"

echo "[5/16] Generating zero bet weeks..."
python -m nfl_analytics.analysis.generate_zero_bet_weeks || echo "  [SKIP] generate_zero_bet_weeks failed"

echo "[6/16] Generating slippage model..."
python -m nfl_analytics.analysis.generate_slippage_model || echo "  [SKIP] generate_slippage_model failed"

echo "[7/16] Benchmark comparison..."
python -m nfl_analytics.analysis.benchmark_comparison || echo "  [SKIP] benchmark_comparison failed"

echo "[8/16] Statistical significance..."
python -m nfl_analytics.analysis.statistical_significance || echo "  [SKIP] statistical_significance failed"

echo "[9/16] RL agent comparison..."
python -m nfl_analytics.analysis.rl_agent_comparison || echo "  [SKIP] rl_agent_comparison failed"

echo "[10/16] RL vs baseline comparison..."
python -m nfl_analytics.analysis.rl_vs_baseline_comparison || echo "  [SKIP] rl_vs_baseline_comparison failed"

echo "[11/16] Key mass calibration..."
python -m nfl_analytics.analysis.keymass_calibration || echo "  [SKIP] keymass_calibration failed"

echo "[12/16] Stadium weather clustering..."
python -m nfl_analytics.analysis.stadium_weather_clustering || echo "  [SKIP] stadium_weather_clustering failed"

# Visualization scripts
echo "[13/16] Reliability diagram..."
python -m nfl_analytics.viz.reliability_diagram || echo "  [SKIP] reliability_diagram failed"

echo "[14/16] Bankroll trajectories..."
python -m nfl_analytics.viz.bankroll_trajectories || echo "  [SKIP] bankroll_trajectories failed"

echo "[15/16] Bankroll histogram..."
python -m nfl_analytics.viz.bankroll_histogram || echo "  [SKIP] bankroll_histogram failed"

echo "[16/16] Copula joint exceedance..."
python -m nfl_analytics.viz.copula_joint_exceedance || echo "  [SKIP] copula_joint_exceedance failed"

echo ""
echo "=== Analysis Complete ==="
