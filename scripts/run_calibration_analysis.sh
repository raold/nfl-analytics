#!/bin/bash
#
# Run calibration diagnostic analysis
#

cd $(cd "$(dirname "")/.." \&\& pwd)

echo "Running calibration diagnostic analysis..."
echo

python3 scripts/analyze_calibration.py

echo
echo "Analysis complete. Results saved to:"
echo "  - docs/diagnostics/calibration_diagnostic_nov7.md"
echo "  - docs/diagnostics/calibration_diagnostic_nov7.json"
