#!/usr/bin/env python3
"""
Move existing chapter summaries to the end and add missing summaries.
"""

import re
from pathlib import Path

DISSERTATION_DIR = Path('/Users/dro/rice/nfl-analytics/analysis/dissertation')

# Chapters that need summaries added (identified from audit)
CHAPTERS_NEEDING_SUMMARIES = {
    'chapter_1_intro.tex': {
        'summary': 'This chapter established the motivation, scope, and core contributions of the dissertation. We argued that profitable NFL prediction requires a hybrid stack combining calibrated statistical models with explicit uncertainty quantification and disciplined risk management.',
        'next': 'We now survey the foundational literature—canonical models (Harville, Stern, Glickman), score distributions (Poisson, Skellam, bivariate), evaluation frameworks (proper scoring, calibration), and market microstructure—in \\Cref{chap:litreview}.'
    },
    'chapter_8_results_discussion.tex': {
        'summary': 'This chapter presented comprehensive empirical results validating the hybrid prediction and betting framework. We demonstrated that the ensemble achieves sustained profitability with positive CLV, strong calibration, and robust risk management.',
        'next': 'Having validated the system offline, \\Cref{chap:production} details the production deployment architecture, monitoring infrastructure, and operational protocols that enable reliable real-time execution.'
    },
    'chapter_10_conclusion.tex': {
        'summary': 'This dissertation demonstrated that profitable NFL betting is achievable through a disciplined hybrid framework combining classical statistical models, machine learning, and reinforcement learning under explicit risk constraints. The system achieved positive CLV and ROI while maintaining calibrated uncertainty and operational robustness.',
        'next': 'Future work should focus on extending the framework to player props, live betting, and portfolio optimization across multiple sports, while maintaining the core principles of uncertainty quantification and governance that underpin sustainable profitability.'
    }
}

def find_chaptersummary_position(content):
    """Find the position of \\chaptersummary in content."""
    match = re.search(r'\\chaptersummary\{', content)
    if match:
        return match.start()
    return None

def extract_chaptersummary(content):
    """Extract the full \\chaptersummary{...}{...} command."""
    pos = find_chaptersummary_position(content)
    if pos is None:
        return None, None

    # Find matching braces for the two arguments
    # This is a simplified brace matcher - good enough for our use case
    start = pos
    depth = 0
    in_first_arg = False
    in_second_arg = False
    first_arg_start = None
    first_arg_end = None
    second_arg_start = None
    second_arg_end = None

    i = pos + len('\\chaptersummary')
    while i < len(content):
        if content[i] == '{':
            if depth == 0 and not in_first_arg:
                in_first_arg = True
                first_arg_start = i + 1
            elif depth == 0 and in_first_arg and first_arg_end is not None and not in_second_arg:
                in_second_arg = True
                second_arg_start = i + 1
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0 and in_first_arg and first_arg_end is None:
                first_arg_end = i
            elif depth == 0 and in_second_arg and second_arg_end is None:
                second_arg_end = i
                # We've found both arguments
                return content[start:i+1], (start, i+1)
        i += 1

    return None, None

def remove_chaptersummary(content):
    """Remove \\chaptersummary from content and return cleaned content."""
    summary_text, summary_span = extract_chaptersummary(content)
    if summary_text is None:
        return content, None

    # Remove the summary and any surrounding whitespace
    before = content[:summary_span[0]].rstrip()
    after = content[summary_span[1]:].lstrip()

    return before + '\n\n' + after, summary_text

def add_chaptersummary_to_end(content, summary_text):
    """Add \\chaptersummary to the end of the content."""
    # Find the last non-whitespace content
    content = content.rstrip()

    # Add the summary
    return content + '\n\n' + summary_text + '\n'

def create_chaptersummary_command(summary, next_text):
    """Create a \\chaptersummary command."""
    return f'\\chaptersummary{{\n{summary}\n}}{{\n{next_text}\n}}'

def fix_chapter(chapter_path):
    """Fix a single chapter file."""
    print(f"\nProcessing {chapter_path.name}...")

    content = chapter_path.read_text()

    # Check if this chapter needs a summary added
    if chapter_path.name in CHAPTERS_NEEDING_SUMMARIES:
        info = CHAPTERS_NEEDING_SUMMARIES[chapter_path.name]
        summary_cmd = create_chaptersummary_command(info['summary'], info['next'])
        new_content = add_chaptersummary_to_end(content, summary_cmd)
        chapter_path.write_text(new_content)
        print(f"  ✓ Added summary to {chapter_path.name}")
        return True

    # Check if chapter has summary that needs to be moved
    summary_text, summary_span = extract_chaptersummary(content)
    if summary_text is None:
        print(f"  - No summary found in {chapter_path.name}")
        return False

    # Check if summary is already at the end (within last 200 chars)
    content_stripped = content.rstrip()
    if summary_span[1] >= len(content_stripped) - 200:
        print(f"  ✓ Summary already at end in {chapter_path.name}")
        return False

    # Move summary to end
    content_without_summary, extracted = remove_chaptersummary(content)
    new_content = add_chaptersummary_to_end(content_without_summary, extracted)
    chapter_path.write_text(new_content)
    print(f"  ✓ Moved summary to end in {chapter_path.name}")
    return True

def main():
    print("=" * 80)
    print("FIXING CHAPTER SUMMARIES")
    print("=" * 80)

    # Find all chapter files (exclude wrappers)
    chapter_files = sorted([
        f for f in DISSERTATION_DIR.glob('chapter_*/*.tex')
        if not f.name.endswith('_wrapper.tex')
    ])

    modified = []
    for chapter_file in chapter_files:
        if fix_chapter(chapter_file):
            modified.append(chapter_file.name)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total chapters processed: {len(chapter_files)}")
    print(f"Chapters modified: {len(modified)}")
    if modified:
        print("\nModified chapters:")
        for name in modified:
            print(f"  - {name}")

if __name__ == '__main__':
    main()
