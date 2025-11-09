#!/usr/bin/env bash
# Install dissertation enhancement plan dependencies
# Phase 1: MAPIE (conformal prediction)
# Phase 2: linearmodels (Difference-in-Differences)
# Phase 4: river (online learning & drift detection)
# Phase 5: torchuncertainty (evidential deep learning)

set -e

echo "========================================================================"
echo "DISSERTATION ENHANCEMENT PLAN - DEPENDENCY INSTALLATION"
echo "========================================================================"
echo ""

# Check for Python and uv
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo "uv version: $(uv --version)"
echo ""

# Ensure virtual environment exists and is active
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    uv venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

echo "Virtual environment activated: $(which python)"
echo ""

# Function to check if package is installed
check_package() {
    local package_name=$1
    if uv pip show "$package_name" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get package version
get_version() {
    local package_name=$1
    uv pip show "$package_name" 2>/dev/null | grep "^Version:" | awk '{print $2}'
}

# Array to track installation results
declare -A install_results
declare -A versions

echo "========================================================================"
echo "PHASE 1: MAPIE (Conformal Prediction)"
echo "========================================================================"
echo "Purpose: Hierarchical conformal wrappers for uncertainty quantification"
echo ""

if check_package "mapie"; then
    version=$(get_version "mapie")
    versions["mapie"]=$version
    install_results["mapie"]="ALREADY_INSTALLED"
    echo "✓ MAPIE already installed (v$version)"
else
    echo "Installing MAPIE..."
    if uv pip install mapie -q; then
        version=$(get_version "mapie")
        versions["mapie"]=$version
        install_results["mapie"]="NEWLY_INSTALLED"
        echo "✓ MAPIE installed successfully (v$version)"
    else
        install_results["mapie"]="FAILED"
        echo "✗ MAPIE installation failed"
    fi
fi
echo ""

echo "========================================================================"
echo "PHASE 2: linearmodels (Difference-in-Differences)"
echo "========================================================================"
echo "Purpose: Panel regression analysis of injury shocks (DiD methodology)"
echo ""

if check_package "linearmodels"; then
    version=$(get_version "linearmodels")
    versions["linearmodels"]=$version
    install_results["linearmodels"]="ALREADY_INSTALLED"
    echo "✓ linearmodels already installed (v$version)"
else
    echo "Installing linearmodels..."
    if uv pip install linearmodels -q; then
        version=$(get_version "linearmodels")
        versions["linearmodels"]=$version
        install_results["linearmodels"]="NEWLY_INSTALLED"
        echo "✓ linearmodels installed successfully (v$version)"
    else
        install_results["linearmodels"]="FAILED"
        echo "✗ linearmodels installation failed"
    fi
fi
echo ""

echo "========================================================================"
echo "PHASE 4: river (Online Learning & Drift Detection)"
echo "========================================================================"
echo "Purpose: Online learning and concept drift detection in production models"
echo ""

if check_package "river"; then
    version=$(get_version "river")
    versions["river"]=$version
    install_results["river"]="ALREADY_INSTALLED"
    echo "✓ river already installed (v$version)"
else
    echo "Installing river..."
    if uv pip install river -q; then
        version=$(get_version "river")
        versions["river"]=$version
        install_results["river"]="NEWLY_INSTALLED"
        echo "✓ river installed successfully (v$version)"
    else
        install_results["river"]="FAILED"
        echo "✗ river installation failed"
    fi
fi
echo ""

echo "========================================================================"
echo "PHASE 5: torchuncertainty (Evidential Deep Learning)"
echo "========================================================================"
echo "Purpose: Evidential deep learning with uncertainty decomposition"
echo ""

if check_package "torchuncertainty"; then
    version=$(get_version "torchuncertainty")
    versions["torchuncertainty"]=$version
    install_results["torchuncertainty"]="ALREADY_INSTALLED"
    echo "✓ torchuncertainty already installed (v$version)"
else
    echo "Installing torchuncertainty..."
    if uv pip install torchuncertainty -q; then
        version=$(get_version "torchuncertainty")
        versions["torchuncertainty"]=$version
        install_results["torchuncertainty"]="NEWLY_INSTALLED"
        echo "✓ torchuncertainty installed successfully (v$version)"
    else
        install_results["torchuncertainty"]="FAILED"
        echo "✗ torchuncertainty installation failed"
    fi
fi
echo ""

echo "========================================================================"
echo "DEPENDENCY CONFLICT CHECK"
echo "========================================================================"
echo ""

# Run pip check to detect conflicts
if uv pip check > /dev/null 2>&1; then
    echo "✓ No dependency conflicts detected"
else
    echo "⚠ WARNING: Potential dependency conflicts detected:"
    uv pip check || true
fi
echo ""

echo "========================================================================"
echo "INSTALLATION SUMMARY"
echo "========================================================================"
echo ""

newly_installed=0
already_installed=0
failed=0

for package in "mapie" "linearmodels" "river" "torchuncertainty"; do
    status=${install_results[$package]}
    version=${versions[$package]}

    case $status in
        NEWLY_INSTALLED)
            echo "NEW:     $package (v$version)"
            ((newly_installed++))
            ;;
        ALREADY_INSTALLED)
            echo "EXISTING: $package (v$version)"
            ((already_installed++))
            ;;
        FAILED)
            echo "FAILED:  $package"
            ((failed++))
            ;;
    esac
done

echo ""
echo "Summary:"
echo "  Newly Installed:  $newly_installed"
echo "  Already Installed: $already_installed"
echo "  Failed:           $failed"
echo ""

if [ $failed -gt 0 ]; then
    echo "ERROR: $failed package(s) failed to install"
    exit 1
fi

echo "========================================================================"
echo "VERIFICATION"
echo "========================================================================"
echo ""

# Test imports
echo "Testing Python imports..."
python3 << 'EOF'
import sys

packages = ['mapie', 'linearmodels', 'river', 'torchuncertainty']
failed_imports = []

for package in packages:
    try:
        __import__(package)
        print(f"✓ {package}: Import successful")
    except ImportError as e:
        print(f"✗ {package}: Import failed - {e}")
        failed_imports.append(package)

if failed_imports:
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All packages imported successfully"
else
    echo ""
    echo "✗ Some packages failed to import"
    exit 1
fi

echo ""
echo "========================================================================"
echo "INSTALLATION COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Update requirements.txt with new dependencies"
echo "2. Review Phase 1-5 dissertation enhancement documentation"
echo "3. Begin implementing MAPIE conformal prediction (Phase 1)"
echo ""
echo "Virtual environment: .venv/bin/activate"
echo "Activate with: source .venv/bin/activate"
echo ""
