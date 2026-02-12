#!/usr/bin/env bash
# Setup testing environment and run pre-commit hooks

set -e

echo "🔧 Setting up testing environment..."

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run 'python -m venv .venv' first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements-dev.txt

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files (first time)
echo "🔍 Running pre-commit checks on all files..."
pre-commit run --all-files || echo "⚠️  Some pre-commit hooks failed. This is normal on first run."

# Initialize secrets baseline for detect-secrets
if [ ! -f ".secrets.baseline" ]; then
    echo "🔐 Initializing secrets detection baseline..."
    detect-secrets scan > .secrets.baseline
fi

# Create pytest cache directory
mkdir -p .pytest_cache

echo ""
echo "✅ Testing environment setup complete!"
echo ""
echo "Available commands:"
echo "  pytest tests/unit              # Run unit tests"
echo "  pytest tests/integration       # Run integration tests (requires DB)"
echo "  pytest -m unit                 # Run only unit tests"
echo "  pytest --cov=py                # Run with coverage report"
echo "  pre-commit run --all-files     # Run all pre-commit hooks"
echo "  black py/                      # Format Python code"
echo "  ruff check py/                 # Lint Python code"
echo ""
