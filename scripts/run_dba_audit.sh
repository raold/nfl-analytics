#!/bin/bash
# Run DBA audit agent
# Schedule via cron: 0 2 * * * (daily at 2 AM)

set -e  # Exit on error

# Change to project directory
cd "$(dirname "$0")/.."

echo "========================================="
echo "Running DBA Audit Agent - $(date)"
echo "========================================="
echo ""

# Run the R script
Rscript R/dba_audit_agent.R

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ DBA Audit completed successfully (passed or warnings)"
else
    echo "❌ DBA Audit FAILED - critical issues detected!"
    echo "Check logs/dba_audits/ for details"
fi
echo "========================================="

exit $EXIT_CODE
