#!/usr/bin/env python3
"""
Intelligent LaTeX underscore fixer.
Escapes underscores ONLY in plain text, not in:
- Math mode (between $ or \[ \])
- \texttt{} commands
- lstlisting environments
- verbatim environments
- Comments (after %)
"""

import re
import sys

def fix_underscores_in_line(line, in_lstlisting, in_verbatim, in_algorithm):
    """Fix underscores in a single line, respecting context."""

    # Don't touch lines inside lstlisting, verbatim, or algorithm blocks
    if in_lstlisting or in_verbatim or in_algorithm:
        return line

    # Don't touch comment lines
    if line.strip().startswith('%'):
        return line

    # Split on % to handle inline comments
    if '%' in line:
        code_part, comment_part = line.split('%', 1)
        return fix_underscores_in_text(code_part) + '%' + comment_part

    return fix_underscores_in_text(line)

def fix_underscores_in_text(text):
    """Fix underscores in text, preserving math mode and \texttt{}."""

    # Pattern to match math mode: $ ... $, \[ ... \], \( ... \)
    # Pattern to match \texttt{...}
    # We'll process text in chunks, skipping protected regions

    result = []
    i = 0
    while i < len(text):
        # Check for start of math mode
        if text[i] == '$':
            # Find matching closing $
            j = i + 1
            while j < len(text) and text[j] != '$':
                if text[j] == '\\':
                    j += 2  # Skip escaped character
                else:
                    j += 1
            if j < len(text):
                j += 1  # Include closing $
            result.append(text[i:j])
            i = j
            continue

        # Check for \texttt{
        if i < len(text) - 7 and text[i:i+7] == '\\texttt{':
            # Find matching closing }
            j = i + 7
            brace_count = 1
            while j < len(text) and brace_count > 0:
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                elif text[j] == '\\':
                    j += 1  # Skip next char
                j += 1
            result.append(text[i:j])
            i = j
            continue

        # Check for \[ or \(
        if i < len(text) - 1 and text[i:i+2] in ['\\[', '\\(']:
            end_marker = '\\]' if text[i:i+2] == '\\[' else '\\)'
            j = text.find(end_marker, i + 2)
            if j != -1:
                j += 2
                result.append(text[i:j])
                i = j
                continue

        # Check for already escaped underscore
        if i < len(text) - 1 and text[i:i+2] == '\\_':
            result.append('\\_')
            i += 2
            continue

        # Check for unescaped underscore in plain text
        if text[i] == '_':
            result.append('\\_')
            i += 1
            continue

        # Regular character
        result.append(text[i])
        i += 1

    return ''.join(result)

def process_file(filepath):
    """Process a LaTeX file to fix underscores."""

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_lines = []
    in_lstlisting = False
    in_verbatim = False
    in_algorithm = False
    changes_made = 0

    for line_num, line in enumerate(lines, 1):
        original_line = line

        # Track environment state
        if '\\begin{lstlisting}' in line:
            in_lstlisting = True
        elif '\\end{lstlisting}' in line:
            in_lstlisting = False
        elif '\\begin{verbatim}' in line:
            in_verbatim = True
        elif '\\end{verbatim}' in line:
            in_verbatim = False
        elif '\\begin{algorithm}' in line or '\\begin{algorithmic}' in line:
            in_algorithm = True
        elif '\\end{algorithm}' in line or '\\end{algorithmic}' in line:
            in_algorithm = False

        # Fix underscores in this line
        fixed_line = fix_underscores_in_line(line, in_lstlisting, in_verbatim, in_algorithm)

        if fixed_line != original_line:
            changes_made += 1
            print(f"Line {line_num}: Fixed underscore(s)")

        output_lines.append(fixed_line)

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    return changes_made

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_underscores.py <file1.tex> [file2.tex ...]")
        sys.exit(1)

    total_changes = 0
    for filepath in sys.argv[1:]:
        print(f"\nProcessing {filepath}...")
        changes = process_file(filepath)
        print(f"  Made {changes} changes")
        total_changes += changes

    print(f"\nTotal changes across all files: {total_changes}")
