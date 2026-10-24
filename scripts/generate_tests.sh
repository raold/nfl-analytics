#!/bin/bash
#
# Test Suite Generator - Generate comprehensive tests targeting 80%+ coverage
#
# Usage:
#   ./scripts/generate_tests.sh nfl_analytics/models/train_bnn_passing.py    # Generate tests for specific file
#   ./scripts/generate_tests.sh --full                             # Generate full test suite
#   ./scripts/generate_tests.sh --coverage                         # Check current coverage
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ "$1" == "--full" ]; then
    echo "=== Generating Full Test Suite ==="
    echo "Target: 80%+ code coverage"
    echo ""

    python3 -c "
from nfl_analytics.utils.test_generator import TestGenerator

generator = TestGenerator()

# Generate tests for all Python files in nfl_analytics/
generator.generate_full_suite(target_coverage=0.80, source_dirs=['nfl_analytics/'])

print('\\n✅ Full test suite generated')
print('\\nNext steps:')
print('  1. Review generated tests in tests/unit/')
print('  2. Run: pytest tests/unit')
print('  3. Check coverage: pytest --cov=nfl_analytics --cov-report=html')
print('  4. Open: htmlcov/index.html')
"

elif [ "$1" == "--coverage" ]; then
    echo "=== Current Test Coverage ==="
    echo ""

    if command -v pytest &> /dev/null; then
        pytest --cov=nfl_analytics --cov-report=term-missing --cov-report=html -q || true
        echo ""
        echo "Detailed coverage report: htmlcov/index.html"
    else
        echo "Error: pytest not installed"
        echo "Install: pip install pytest pytest-cov"
        exit 1
    fi

elif [ -n "$1" ]; then
    # Generate tests for specific file
    FILE_PATH="$1"

    if [ ! -f "$FILE_PATH" ]; then
        echo "Error: File not found: $FILE_PATH"
        exit 1
    fi

    echo "=== Generating Tests for $FILE_PATH ==="
    echo ""

    python3 -c "
from nfl_analytics.utils.test_generator import TestGenerator
from pathlib import Path

generator = TestGenerator()

# Generate tests for specific file
suite = generator.generate_tests_for_file('$FILE_PATH')

if suite.test_cases:
    # Determine output path
    file_path = Path('$FILE_PATH')
    test_filename = f'test_{file_path.name}'

    # Place in tests/unit/
    output_path = Path('tests/unit') / test_filename

    # Write tests
    generator.write_tests(str(output_path), suite)

    print(f'Generated {len(suite.test_cases)} tests → {output_path}')
    print('\\nTest types:')

    from collections import Counter
    test_types = Counter(test.test_type.value for test in suite.test_cases)
    for test_type, count in test_types.items():
        print(f'  - {test_type}: {count}')

    print('\\nNext steps:')
    print(f'  1. Review: {output_path}')
    print(f'  2. Run: pytest {output_path}')
    print(f'  3. Customize tests as needed')
else:
    print('No public functions found in file')
"

else
    echo "=== Test Suite Generator ==="
    echo ""
    echo "Usage:"
    echo "  ./scripts/generate_tests.sh FILE              # Generate tests for specific file"
    echo "  ./scripts/generate_tests.sh --full            # Generate full test suite (80%+ coverage)"
    echo "  ./scripts/generate_tests.sh --coverage        # Check current coverage"
    echo ""
    echo "Examples:"
    echo "  ./scripts/generate_tests.sh nfl_analytics/models/train_bnn_passing.py"
    echo "  ./scripts/generate_tests.sh nfl_analytics/ensemble/enhanced_ensemble_v5_canonical.py"
    echo "  ./scripts/generate_tests.sh --full"
    echo ""
    echo "Test types generated:"
    echo "  - Unit tests (basic functionality, edge cases, error handling)"
    echo "  - Property tests (Hypothesis library, idempotence, invariants)"
    echo "  - Temporal tests (sports betting specific, no future data leakage)"
    echo ""
    echo "Target: 80%+ code coverage"
    exit 1
fi

echo ""
echo "✅ Test generation complete"
