#!/usr/bin/env python3
"""
Pre-commit hook to validate brms posterior_predict calls.

Ensures all posterior_predict() calls have both:
- allow_new_levels = TRUE
- sample_new_levels = "gaussian"
"""

import re
import sys
from pathlib import Path


def check_file(filepath: str) -> int:
    """Check R file for incomplete brms predictions."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 1

    # Find posterior_predict calls
    calls = re.finditer(r'posterior_predict\([^)]{0,500}\)', content, re.DOTALL)

    errors = []
    for match in calls:
        call_text = match.group(0)
        line_num = content[:match.start()].count('\n') + 1

        has_allow_new = 'allow_new_levels' in call_text
        has_sample_new = 'sample_new_levels' in call_text

        if has_allow_new and not has_sample_new:
            errors.append(f"Line {line_num}: Missing sample_new_levels parameter")

    if errors:
        print(f"❌ {filepath}:")
        for error in errors:
            print(f"  {error}")
        print("\n💡 Fix: Add sample_new_levels = \"gaussian\" to posterior_predict()")
        return 1

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: check_brms_predictions.py <file.R>")
        sys.exit(1)

    sys.exit(check_file(sys.argv[1]))
