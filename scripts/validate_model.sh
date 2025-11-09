#!/bin/bash
#
# Statistical Model Validator - Validate Bayesian models before production deployment
#
# Usage:
#   ./scripts/validate_model.sh MODEL_METADATA.json    # Validate specific model
#   ./scripts/validate_model.sh --all                  # Validate all trained models
#   ./scripts/validate_model.sh --quick MODEL.pkl      # Quick validation (basic checks)
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ "$1" == "--all" ]; then
    echo "=== Validating All Trained Models ==="
    echo ""

    python3 -c "
from py.utils.statistical_validator import StatisticalValidator
from pathlib import Path
import json

validator = StatisticalValidator()

# Find all model metadata files
metadata_files = list(Path('models/').rglob('*_metadata.json'))

if not metadata_files:
    print('No model metadata files found')
    exit(0)

print(f'Found {len(metadata_files)} models to validate\\n')

all_valid = True

for metadata_file in metadata_files:
    print(f'Validating: {metadata_file.name}')

    try:
        model_data = json.load(metadata_file.open())

        # Add model name
        model_data['model_name'] = metadata_file.stem.replace('_metadata', '')

        # Validate
        report = validator.validate_model(model_data)

        if report.has_critical_issues():
            print(f'  ⛔ FAILED - {len(report.critical_issues)} critical issues')
            all_valid = False

            for issue in report.critical_issues:
                print(f'     - {issue.description}')
        else:
            print(f'  ✅ PASSED - {len(report.issues)} warnings')

    except Exception as e:
        print(f'  ❌ ERROR: {e}')
        all_valid = False

    print()

if all_valid:
    print('✅ All models passed validation')
    exit(0)
else:
    print('⛔ Some models failed validation - DO NOT DEPLOY')
    exit(1)
"

elif [ "$1" == "--quick" ]; then
    if [ -z "$2" ]; then
        echo "Error: Please provide model file path"
        echo "Usage: ./scripts/validate_model.sh --quick MODEL.pkl"
        exit 1
    fi

    MODEL_FILE="$2"

    if [ ! -f "$MODEL_FILE" ]; then
        echo "Error: Model file not found: $MODEL_FILE"
        exit 1
    fi

    echo "=== Quick Model Validation: $MODEL_FILE ==="
    echo ""

    python3 -c "
from py.utils.statistical_validator import StatisticalValidator
from pathlib import Path
import pickle

validator = StatisticalValidator()

model_file = Path('$MODEL_FILE')

# Load model
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    print(f'✅ Model loaded successfully')
    print(f'   Type: {type(model).__name__}')

    # Quick checks
    print('\\nQuick validation checks:')
    print('  - Model file exists: ✅')
    print('  - Model loads: ✅')

    # Try to get predictions if possible
    if hasattr(model, 'predict'):
        print('  - Has predict method: ✅')
    else:
        print('  - Has predict method: ⚠️  Not found')

    print('\\n✅ Quick validation passed')
    print('\\nFor full validation:')
    print(f'  ./scripts/validate_model.sh {model_file.stem}_metadata.json')

except Exception as e:
    print(f'❌ ERROR: {e}')
    exit(1)
"

elif [ -n "$1" ]; then
    # Validate specific model from metadata file
    METADATA_FILE="$1"

    if [ ! -f "$METADATA_FILE" ]; then
        echo "Error: Metadata file not found: $METADATA_FILE"
        exit 1
    fi

    echo "=== Statistical Model Validation ==="
    echo "Metadata: $METADATA_FILE"
    echo ""

    python3 -c "
import sys
import json
from py.utils.statistical_validator import StatisticalValidator

validator = StatisticalValidator()

# Load model metadata
try:
    with open('$METADATA_FILE', 'r') as f:
        model_data = json.load(f)
except Exception as e:
    print(f'❌ ERROR loading metadata: {e}')
    sys.exit(1)

# Validate model
report = validator.validate_model(model_data)

# Print report
print(validator.generate_report(report))
print()

# Check for critical issues
if report.has_critical_issues():
    print('⛔ MODEL VALIDATION FAILED')
    print(f'Critical issues: {len(report.critical_issues)}')
    print()
    print('DO NOT DEPLOY THIS MODEL TO PRODUCTION')
    print()
    print('Critical issues:')
    for issue in report.critical_issues:
        print(f'  - {issue.category.value}: {issue.description}')
    sys.exit(1)
else:
    print('✅ Model validation passed')

    if report.issues:
        print(f'\\nWarnings: {len(report.issues)}')
        for issue in report.issues:
            print(f'  - {issue.category.value}: {issue.description}')

    print()
    print('Model approved for production deployment')
    sys.exit(0)
"

else
    echo "=== Statistical Model Validator ==="
    echo ""
    echo "Usage:"
    echo "  ./scripts/validate_model.sh METADATA.json     # Validate specific model"
    echo "  ./scripts/validate_model.sh --all             # Validate all models"
    echo "  ./scripts/validate_model.sh --quick MODEL.pkl # Quick validation"
    echo ""
    echo "Examples:"
    echo "  ./scripts/validate_model.sh models/bayesian/bnn_passing_v1_metadata.json"
    echo "  ./scripts/validate_model.sh --all"
    echo "  ./scripts/validate_model.sh --quick models/bayesian/bnn_passing_v1.pkl"
    echo ""
    echo "Validation checks:"
    echo "  - MCMC diagnostics (R-hat <1.01, ESS >400, divergences <10)"
    echo "  - Model calibration (coverage >85%, target: 90%)"
    echo "  - Prior/posterior comparison"
    echo "  - Cross-validation metrics"
    echo ""
    echo "⛔ Models with critical issues will be blocked from deployment"
    exit 1
fi
