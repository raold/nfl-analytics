#!/usr/bin/env bash
#
# Code Review Script for NFL Analytics
#
# Reviews code changes for:
# - Data leakage (temporal, target, player)
# - Security vulnerabilities (credentials, SQL injection)
# - Bayesian modeling issues (priors, MCMC config)
# - Code quality (type hints, docstrings)
#
# Usage:
#   ./scripts/review_code.sh                     # Review all git changes
#   ./scripts/review_code.sh py/models/train.py   # Review specific file
#   ./scripts/review_code.sh --full              # Review entire codebase

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=================================="
echo "NFL Analytics Code Reviewer"
echo "=================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Run code reviewer
python3 -c "
import sys
sys.path.insert(0, '.')

from py.utils.code_reviewer import CodeReviewer

reviewer = CodeReviewer()

# Check command line arguments
import sys
if len(sys.argv) > 1:
    # Review specific file
    file_path = sys.argv[1]
    print(f'Reviewing: {file_path}')
    issues = reviewer.review_file(file_path)

    for issue in issues:
        print(f'{issue.file_path}:{issue.line_number}')
        print(f'  [{issue.severity.value}] {issue.title}')
        print(f'  {issue.description}')
        if issue.suggestion:
            print(f'  💡 {issue.suggestion}')
        print()
else:
    # Review all git changes
    print('Reviewing git changes...')
    report = reviewer.review_changes(git_diff=True)
    print(reviewer.generate_report())

    # Exit with error if critical issues found
    if report.has_blocking_issues():
        sys.exit(1)
" "$@"
