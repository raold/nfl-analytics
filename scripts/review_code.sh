#!/usr/bin/env bash
#
# Code Review Script for NFL Analytics
#
# Reviews code changes for:
# - Data leakage (temporal, target, player)
# - Security vulnerabilities (credentials, SQL injection)
# - Bayesian modeling issues (priors, MCMC config)
# - SQL schema errors (mart.player_game_stats.game_id bug)
# - Code quality (type hints, docstrings)
#
# Usage:
#   ./scripts/review_code.sh                     # Review all git changes
#   ./scripts/review_code.sh nfl_analytics/models/train.py   # Review specific file
#   ./scripts/review_code.sh --check-sql         # Check SQL files only
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
import glob
sys.path.insert(0, '.')

from nfl_analytics.utils.code_reviewer import CodeReviewer

reviewer = CodeReviewer()

# Check command line arguments
if len(sys.argv) > 1 and sys.argv[1] == '--check-sql':
    # Review all SQL files (Python, R, SQL)
    print('Checking SQL schema errors in all files...')
    print()

    sql_files = (
        glob.glob('**/*.sql', recursive=True) +
        glob.glob('**/*.py', recursive=True) +
        glob.glob('**/*.R', recursive=True)
    )

    all_issues = []
    for file_path in sql_files:
        if 'venv' in file_path or '.venv' in file_path or 'node_modules' in file_path:
            continue

        issues = reviewer.review_file(file_path)
        # Filter to only SQL schema issues
        sql_issues = [i for i in issues if 'player_game_stats' in i.title or 'schema' in i.title.lower()]

        if sql_issues:
            all_issues.extend(sql_issues)
            for issue in sql_issues:
                print(f'{issue.file_path}:{issue.line_number}')
                print(f'  [{issue.severity.value}] {issue.title}')
                print(f'  {issue.description}')
                if issue.suggestion:
                    print(f'  💡 {issue.suggestion}')
                if issue.code_snippet:
                    print(f'  Code: {issue.code_snippet}')
                print()

    print(f'Total SQL schema issues found: {len(all_issues)}')
    if any(i.severity.value == 'CRITICAL' for i in all_issues):
        sys.exit(1)

elif len(sys.argv) > 1:
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
