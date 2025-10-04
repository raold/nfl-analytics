#!/usr/bin/env bash
set -euo pipefail

# Migrate from pip to uv for faster dependency management
# This preserves your existing setup while creating a new uv-managed environment

echo "=== Migrating to uv-managed Python environment ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install with: brew install uv"
    exit 1
fi

# Backup existing requirements
if [ -f requirements.txt ]; then
    cp requirements.txt requirements.txt.backup
    echo "✅ Backed up requirements.txt"
fi

# Create new uv-managed venv
echo "Creating new uv-managed virtual environment..."
uv venv .venv-uv --python 3.11

# Install dependencies using uv
echo "Installing dependencies with uv (this will be FAST)..."
uv pip install -e ".[dev,torch,research]"

# Create activation script
cat > scripts/dev/activate_uv.sh << 'EOF'
#!/usr/bin/env bash
# Quick activation script for uv-managed environment
source .venv-uv/bin/activate
echo "✅ Activated uv environment (Python $(python --version))"
echo "📦 Packages: $(pip list | wc -l) installed"
EOF
chmod +x scripts/dev/activate_uv.sh

# Generate lock file for reproducibility
echo "Generating uv.lock for exact reproducibility..."
uv pip compile pyproject.toml -o requirements.lock

echo ""
echo "=== Migration Complete ==="
echo ""
echo "To use the new environment:"
echo "  source scripts/dev/activate_uv.sh"
echo ""
echo "Your old environment is preserved at .venv"
echo "New uv environment is at .venv-uv"
echo ""
echo "Benefits:"
echo "  ✅ 10x faster installs"
echo "  ✅ Better caching"
echo "  ✅ Reproducible with requirements.lock"
echo "  ✅ Modern pyproject.toml instead of requirements.txt"