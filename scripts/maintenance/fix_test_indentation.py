#!/usr/bin/env python3
"""Fix indentation issues in generated test files."""
import re
import subprocess
from pathlib import Path


def fix_indentation(content: str) -> str:
    """Fix incorrect indentation after comments in test files."""
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Fix duplicate self parameters
        if "(self, self)" in line:
            line = line.replace("(self, self)", "(val)")

        # Check previous line for context
        prev_line = lines[i - 1] if i > 0 else ""
        prev_stripped = prev_line.strip()

        # If previous line is a comment or "# Act" and current line starts with 8 spaces
        # and is not already 12+ spaces (nested block), reduce to 4 spaces
        if line.startswith("        ") and not line.startswith("            "):
            # Check if this should be a 4-space indent line
            if (
                prev_stripped.startswith("#")
                or prev_stripped == ""
                or ("# Assert" in prev_stripped)
                or ("# Act" in prev_stripped)
                or ("# Arrange" in prev_stripped)
            ):
                # Remove 4 spaces of indentation
                fixed_lines.append(line[4:])
                continue

        # Fix with block indentation: if prev line is "with pytest.raises"
        # and current line doesn't start with enough spaces, add proper indentation
        if "with pytest.raises" in prev_line and stripped:
            if not line.startswith("        "):
                fixed_lines.append("        " + stripped)
                continue

        # Fix self -> val in function calls
        if "(self)" in line and ("result = " in line or "assert " in line):
            line = line.replace("(self)", "(val)")

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def main():
    # Get all test files that black says it cannot parse
    result = subprocess.run(
        ["bash", "-c", 'source .venv/bin/activate && black --check . 2>&1 | grep "Cannot parse"'],
        capture_output=True,
        text=True,
        cwd="/Users/dro/rice/nfl-analytics",
    )

    files_to_fix = []
    for line in result.stdout.split("\n"):
        if "cannot format" in line:
            # Extract file path
            match = re.search(r"cannot format (.+?):", line)
            if match:
                files_to_fix.append(match.group(1))

    print(f"Found {len(files_to_fix)} files to fix")

    fixed_count = 0
    for filepath in files_to_fix:
        try:
            path = Path(filepath)
            if not path.exists():
                print(f"Skipping {filepath} - not found")
                continue

            content = path.read_text()
            fixed_content = fix_indentation(content)

            if content != fixed_content:
                path.write_text(fixed_content)
                fixed_count += 1

        except Exception as e:
            print(f"Error fixing {filepath}: {e}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
