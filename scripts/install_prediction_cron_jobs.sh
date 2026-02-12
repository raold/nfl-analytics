#!/bin/bash
#
# Install NFL Prediction Generation Cron Jobs
#
# This script installs two cron jobs:
# 1. Tuesday 3:00 AM - R model regeneration (State-Space, Hierarchical, Informative Priors)
# 2. Tuesday 12:00 PM - Ensemble prediction generation (v6.0)
#
# Created: November 22, 2025
# Author: Claude Code (Sonnet 4.5)

set -e

PROJECT_DIR="$(cd "$(dirname "")/.." \&\& pwd)"

echo "=============================================================================="
echo "NFL Prediction Generation Cron Job Installation"
echo "=============================================================================="

# Check if scripts exist
if [ ! -f "$PROJECT_DIR/scripts/regenerate_r_predictions_weekly.sh" ]; then
    echo "ERROR: R regeneration script not found"
    echo "Expected: $PROJECT_DIR/scripts/regenerate_r_predictions_weekly.sh"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/generate_weekly_predictions_automated.sh" ]; then
    echo "ERROR: Weekly prediction script not found"
    echo "Expected: $PROJECT_DIR/scripts/generate_weekly_predictions_automated.sh"
    exit 1
fi

# Make scripts executable
chmod +x "$PROJECT_DIR/scripts/regenerate_r_predictions_weekly.sh"
chmod +x "$PROJECT_DIR/scripts/generate_weekly_predictions_automated.sh"

echo "✅ Scripts found and made executable"

# Create log directory
mkdir -p "$PROJECT_DIR/logs/predictions"
mkdir -p "$PROJECT_DIR/logs/r_models"

echo "✅ Log directories created"

# Show current crontab
echo ""
echo "Current crontab entries:"
echo "---"
crontab -l 2>/dev/null || echo "(No existing crontab)"
echo "---"
echo ""

# Create new cron entries
CRON_TEMP=$(mktemp)

# Preserve existing crontab (if any)
crontab -l 2>/dev/null > "$CRON_TEMP" || true

# Check if cron jobs already exist
if grep -q "regenerate_r_predictions_weekly.sh" "$CRON_TEMP" 2>/dev/null; then
    echo "⚠️  R regeneration cron job already exists - skipping"
else
    echo "# NFL Analytics: R Model Regeneration (Tuesday 3:00 AM)" >> "$CRON_TEMP"
    echo "0 3 * * 2 cd $PROJECT_DIR && ./scripts/regenerate_r_predictions_weekly.sh >> $PROJECT_DIR/logs/r_models/weekly_regeneration.log 2>&1" >> "$CRON_TEMP"
    echo "✅ Added R regeneration cron job"
fi

if grep -q "generate_weekly_predictions_automated.sh" "$CRON_TEMP" 2>/dev/null; then
    echo "⚠️  Ensemble prediction cron job already exists - skipping"
else
    echo "# NFL Analytics: Ensemble Prediction Generation (Tuesday 12:00 PM)" >> "$CRON_TEMP"
    echo "0 12 * * 2 cd $PROJECT_DIR && ./scripts/generate_weekly_predictions_automated.sh >> $PROJECT_DIR/logs/predictions/weekly_automated.log 2>&1" >> "$CRON_TEMP"
    echo "✅ Added ensemble prediction cron job"
fi

# Install new crontab
crontab "$CRON_TEMP"
rm "$CRON_TEMP"

echo ""
echo "✅ Cron jobs installed successfully!"
echo ""
echo "New crontab:"
echo "---"
crontab -l
echo "---"
echo ""

echo "=============================================================================="
echo "Schedule Summary"
echo "=============================================================================="
echo "Tuesday 3:00 AM  - R model regeneration (State-Space, Hierarchical, Informative Priors)"
echo "Tuesday 12:00 PM - Ensemble prediction generation (v6.0)"
echo ""
echo "Logs:"
echo "  R models:   $PROJECT_DIR/logs/r_models/weekly_regeneration.log"
echo "  Predictions: $PROJECT_DIR/logs/predictions/weekly_automated.log"
echo ""

echo "=============================================================================="
echo "Next Steps"
echo "=============================================================================="
echo "1. Verify cron jobs: crontab -l"
echo "2. Test R regeneration: ./scripts/regenerate_r_predictions_weekly.sh"
echo "3. Test prediction generation: ./scripts/generate_weekly_predictions_automated.sh"
echo "4. Monitor logs: tail -f logs/predictions/weekly_automated.log"
echo ""

echo "✅ Installation complete!"
