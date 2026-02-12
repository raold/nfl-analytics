#!/usr/bin/env python3
"""Fix bare except clauses by adding Exception."""
import json
import subprocess
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())



def main():
    # Get files with bare except
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source .venv/bin/activate && ruff check . --select E722 --output-format=json",
        ],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )

    errors = json.loads(result.stdout)

    files_to_fix = {}
    for error in errors:
        filepath = error["filename"]
        line_num = error["location"]["row"]
        if filepath not in files_to_fix:
            files_to_fix[filepath] = []
        files_to_fix[filepath].append(line_num)

    print(f"Found {len(files_to_fix)} files with bare except")

    fixed_count = 0
    for filepath, line_nums in files_to_fix.items():
        try:
            path = Path(filepath)
            lines = path.read_text().split("\n")

            # Sort in reverse so line numbers don't shift
            for line_num in sorted(set(line_nums), reverse=True):
                idx = line_num - 1  # Convert to 0-indexed
                if idx < len(lines) and "except:" in lines[idx]:
                    # Replace bare except with except Exception
                    lines[idx] = lines[idx].replace("except:", "except Exception:")
                    print(f"Fixed {filepath}:{line_num}")

            path.write_text("\n".join(lines))
            fixed_count += 1
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
