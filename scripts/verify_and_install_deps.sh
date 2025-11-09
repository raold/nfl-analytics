#!/usr/bin/env bash
# Quick verification and installation of dissertation dependencies

set -e

cd "$(dirname "$0")/.."

echo "Current directory: $(pwd)"
echo ""

# Check if Python virtual environment is active, if not try to activate
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [ -d ".venv" ]; then
        echo "Activating Python virtual environment..."
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        echo "Activating Python virtual environment..."
        source venv/bin/activate
    else
        echo "ERROR: No virtual environment found"
        echo "Create with: python3 -m venv .venv"
        echo "Then activate: source .venv/bin/activate"
        exit 1
    fi
fi

echo "Using Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# Run the Python installation script
echo "Installing dissertation dependencies..."
echo ""

python3 /Users/dro/rice/nfl-analytics/scripts/install_dissertation_deps.py
