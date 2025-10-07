#!/usr/bin/env python3
"""
Check notation and formatting consistency across dissertation chapters.
"""

import re
from pathlib import Path
from collections import defaultdict

def check_notation_consistency(tex_dir):
    """Check for consistent mathematical notation across chapters."""

    tex_files = list(Path(tex_dir).rglob("*.tex"))

    issues = []
    notation_usage = defaultdict(set)

    # Check for common notation patterns
    patterns = {
        'Expected value': [r'\\E\[', r'\\mathbb\{E\}\[', r'E\[', r'\\text\{E\}\['],
        'Probability': [r'\\Prob', r'\\mathbb\{P\}', r'P\(', r'\\text\{Pr\}'],
        'Variance': [r'\\Var', r'\\text\{Var\}', r'Var\('],
        'CVaR': [r'CVaR', r'\\text\{CVaR\}', r'C-VaR'],
        'CLV': [r'CLV', r'\\text\{CLV\}'],
        'EPA': [r'EPA', r'\\text\{EPA\}'],
        'NFL': [r'NFL', r'\\text\{NFL\}'],
    }

    print("=" * 70)
    print("NOTATION CONSISTENCY CHECK")
    print("=" * 70)

    for pattern_name, variations in patterns.items():
        print(f"\n{pattern_name}:")
        found_variations = set()

        for tex_file in tex_files:
            if 'main.tex' in str(tex_file) or 'figures' in str(tex_file):
                continue

            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for variant in variations:
                    if re.search(variant, content):
                        found_variations.add(variant)
                        notation_usage[pattern_name].add((variant, tex_file.name))
            except:
                pass

        if len(found_variations) > 1:
            print(f"  ⚠️  Multiple notations found:")
            for usage in notation_usage[pattern_name]:
                print(f"     - {usage[0]} in {usage[1]}")
            issues.append(f"{pattern_name}: {len(found_variations)} variations")
        elif len(found_variations) == 1:
            print(f"  ✓ Consistent: {list(found_variations)[0]}")
        else:
            print(f"  - Not found")

    return issues

def check_table_formatting(tex_dir):
    """Check for consistent table formatting."""

    print("\n" + "=" * 70)
    print("TABLE FORMATTING CHECK")
    print("=" * 70)

    tex_files = list(Path(tex_dir).rglob("*table*.tex"))

    issues = []

    # Check for threeparttable usage
    tables_with_threeparttable = 0
    tables_without_threeparttable = 0

    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if r'\begin{table}' in content:
                if r'\begin{threeparttable}' in content:
                    tables_with_threeparttable += 1
                else:
                    tables_without_threeparttable += 1
                    issues.append(f"{tex_file.name}: Missing threeparttable")
        except:
            pass

    print(f"\nTables with threeparttable: {tables_with_threeparttable}")
    print(f"Tables without threeparttable: {tables_without_threeparttable}")

    if tables_without_threeparttable > 0:
        print(f"\n⚠️  {tables_without_threeparttable} tables missing threeparttable")

    return issues

def check_citation_style(tex_dir):
    """Check for consistent citation style."""

    print("\n" + "=" * 70)
    print("CITATION STYLE CHECK")
    print("=" * 70)

    tex_files = list(Path(tex_dir).rglob("chapter_*.tex"))

    cite_commands = defaultdict(int)

    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Count different citation commands
            cite_commands['\\cite{'] += len(re.findall(r'\\cite\{', content))
            cite_commands['\\citep{'] += len(re.findall(r'\\citep\{', content))
            cite_commands['\\citet{'] += len(re.findall(r'\\citet\{', content))
            cite_commands['\\citeauthor{'] += len(re.findall(r'\\citeauthor\{', content))
        except:
            pass

    print("\nCitation command usage:")
    for cmd, count in sorted(cite_commands.items(), key=lambda x: -x[1]):
        print(f"  {cmd}: {count} occurrences")

    if cite_commands['\\cite{'] > 0:
        print("\n⚠️  Using \\cite{} - consider \\citep{} for parenthetical or \\citet{} for textual")

    return []

def check_figure_references(tex_dir):
    """Check for consistent figure/table references."""

    print("\n" + "=" * 70)
    print("CROSS-REFERENCE STYLE CHECK")
    print("=" * 70)

    tex_files = list(Path(tex_dir).rglob("chapter_*.tex"))

    ref_styles = defaultdict(int)

    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Count different reference styles
            ref_styles['\\ref{'] += len(re.findall(r'\\ref\{', content))
            ref_styles['\\Cref{'] += len(re.findall(r'\\Cref\{', content))
            ref_styles['\\cref{'] += len(re.findall(r'\\cref\{', content))
            ref_styles['Figure~\\ref'] += len(re.findall(r'Figure~\\ref', content))
            ref_styles['Table~\\ref'] += len(re.findall(r'Table~\\ref', content))
        except:
            pass

    print("\nReference command usage:")
    for cmd, count in sorted(ref_styles.items(), key=lambda x: -x[1]):
        print(f"  {cmd}: {count} occurrences")

    if ref_styles['Figure~\\ref'] > 0 or ref_styles['Table~\\ref'] > 0:
        print("\n⚠️  Using manual 'Figure~\\ref' - consider \\Cref{} for consistency")

    return []

def main():
    """Run all consistency checks."""

    tex_dir = Path("analysis/dissertation")

    if not tex_dir.exists():
        print(f"Error: Directory {tex_dir} not found")
        return

    all_issues = []

    all_issues.extend(check_notation_consistency(tex_dir))
    all_issues.extend(check_table_formatting(tex_dir))
    all_issues.extend(check_citation_style(tex_dir))
    all_issues.extend(check_figure_references(tex_dir))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_issues:
        print(f"\n⚠️  Found {len(all_issues)} formatting/notation issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No major consistency issues found!")

    print("\nRecommendations:")
    print("  1. Use \\E for expected value (already defined)")
    print("  2. Use \\Prob for probability (already defined)")
    print("  3. Use \\Var for variance (already defined)")
    print("  4. Use \\Cref{} or \\cref{} for all cross-references")
    print("  5. Use \\citep{} for parenthetical citations, \\citet{} for textual")
    print("  6. Ensure all tables use threeparttable environment")

if __name__ == "__main__":
    main()
