#!/bin/bash
# Example Usage Script for Model Registry CLI
# Demonstrates common workflows and use cases

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="uv run python ${SCRIPT_DIR}/query_model_registry.py"

echo "=================================================="
echo "Model Registry CLI - Example Usage"
echo "=================================================="
echo

# 1. List all production models
echo "1. List all production models:"
echo "   Command: list --production-only"
echo
$CLI list --production-only
echo
read -p "Press Enter to continue..."
echo

# 2. Search for BNN models
echo "2. Search for BNN models:"
echo "   Command: search --model-type bnn"
echo
$CLI search --model-type bnn
echo
read -p "Press Enter to continue..."
echo

# 3. Get detailed info on a specific model
echo "3. Get detailed info on bnn_multioutput_v1:"
echo "   Command: info bnn_multioutput_v1"
echo
$CLI info bnn_multioutput_v1
echo
read -p "Press Enter to continue..."
echo

# 4. Compare two models
echo "4. Compare two BNN versions:"
echo "   Command: compare bnn_multioutput_v1 bnn_calibrated_v2"
echo
$CLI compare bnn_multioutput_v1 bnn_calibrated_v2
echo
read -p "Press Enter to continue..."
echo

# 5. Check if model is safe to delete
echo "5. Check if bnn_mixture_experts_v2 is safe to delete:"
echo "   Command: check-safe-delete bnn_mixture_experts_v2"
echo
$CLI check-safe-delete bnn_mixture_experts_v2
echo
read -p "Press Enter to continue..."
echo

# 6. Export to different formats
echo "6. Export models to various formats:"
echo

echo "   a) LaTeX format:"
echo "   Command: export --format latex --status completed"
echo
$CLI export --format latex --status completed | head -n 15
echo "   [... truncated ...]"
echo
read -p "Press Enter to continue..."
echo

echo "   b) CSV format:"
echo "   Command: export --format csv --status completed"
echo
$CLI export --format csv --status completed | head -n 5
echo "   [... truncated ...]"
echo
read -p "Press Enter to continue..."
echo

echo "   c) Markdown format:"
echo "   Command: export --format markdown"
echo
$CLI export --format markdown | head -n 10
echo "   [... truncated ...]"
echo

echo
echo "=================================================="
echo "Example Usage Complete!"
echo "=================================================="
echo
echo "For full documentation, see:"
echo "  - User Guide: ${SCRIPT_DIR}/README_MODEL_REGISTRY_CLI.md"
echo "  - Quick Ref:  ${SCRIPT_DIR}/MODEL_REGISTRY_CLI_QUICKREF.md"
echo
