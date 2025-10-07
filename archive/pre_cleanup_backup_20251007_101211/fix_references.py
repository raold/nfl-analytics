#!/usr/bin/env python3
"""
Convert manual Figure~\ref and Table~\ref to \Cref{}.
"""

import re
from pathlib import Path

def fix_references(tex_dir):
    """Convert manual references to cleveref commands."""

    tex_files = list(Path(tex_dir).rglob("chapter_*.tex"))

    modifications = 0

    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Convert Figure~\ref{...} to \Cref{...}
            content = re.sub(r'Figure~\\ref\{([^}]+)\}', r'\\Cref{\1}', content)

            # Convert Table~\ref{...} to \Cref{...}
            content = re.sub(r'Table~\\ref\{([^}]+)\}', r'\\Cref{\1}', content)

            # Convert Section~\ref{...} to \Cref{...}
            content = re.sub(r'Section~\\ref\{([^}]+)\}', r'\\Cref{\1}', content)

            # Convert Chapter~\ref{...} to \Cref{...}
            content = re.sub(r'Chapter~\\ref\{([^}]+)\}', r'\\Cref{\1}', content)

            if content != original_content:
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                changes = len(re.findall(r'\\Cref\{', content)) - len(re.findall(r'\\Cref\{', original_content))
                print(f"✓ {tex_file.name}: {changes} references converted")
                modifications += changes

        except Exception as e:
            print(f"  Error processing {tex_file}: {e}")

    return modifications

def main():
    tex_dir = Path("analysis/dissertation")

    if not tex_dir.exists():
        print(f"Error: {tex_dir} not found")
        return

    print("=" * 70)
    print("CONVERTING MANUAL REFERENCES TO \\Cref{}")
    print("=" * 70)
    print()

    mods = fix_references(tex_dir)

    print()
    print("=" * 70)
    print(f"✓ Converted {mods} manual references to \\Cref{{}}")
    print("=" * 70)

if __name__ == "__main__":
    main()
