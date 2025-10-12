#!/usr/bin/env python3
"""Check all chapters for summaries and verify they're at the end."""

import re
from pathlib import Path

def check_chapter(chapter_file):
    """Check if chapter has summary and if it's at the end."""
    content = chapter_file.read_text()

    # Look for \chaptersummary
    has_summary = r'\chaptersummary' in content

    # Check if summary is near the end (within last 500 chars of substantial content)
    # Ignore trailing whitespace and \end{document}
    content_stripped = content.rstrip()
    last_500 = content_stripped[-500:]

    summary_at_end = r'\chaptersummary' in last_500

    # Find position of summary
    match = re.search(r'\\chaptersummary', content)
    summary_pos = match.start() if match else None

    # Find last \section or \subsection before end
    sections = list(re.finditer(r'\\(sub)?section\*?{', content))
    last_section_pos = sections[-1].start() if sections else None

    return {
        'file': chapter_file.name,
        'has_summary': has_summary,
        'summary_at_end': summary_at_end,
        'summary_position': summary_pos,
        'last_section_position': last_section_pos,
        'summary_after_last_section': (summary_pos and last_section_pos and summary_pos > last_section_pos) if (summary_pos and last_section_pos) else None
    }

def main():
    dissertation_dir = Path('/Users/dro/rice/nfl-analytics/analysis/dissertation')

    # Find all chapter files
    chapter_files = sorted(dissertation_dir.glob('chapter_*/*.tex'))

    print("=" * 80)
    print("CHAPTER SUMMARY AUDIT")
    print("=" * 80)

    results = []
    for chapter_file in chapter_files:
        result = check_chapter(chapter_file)
        results.append(result)

        status = "✓" if result['has_summary'] and result['summary_at_end'] else "✗"
        print(f"\n{status} {result['file']}")
        print(f"   Has summary: {result['has_summary']}")
        print(f"   Summary at end: {result['summary_at_end']}")
        if result['summary_after_last_section'] is not None:
            print(f"   Summary after last section: {result['summary_after_last_section']}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(results)
    with_summary = sum(1 for r in results if r['has_summary'])
    summary_at_end = sum(1 for r in results if r['summary_at_end'])

    print(f"Total chapters: {total}")
    print(f"With summary: {with_summary} ({with_summary/total*100:.1f}%)")
    print(f"Summary at end: {summary_at_end} ({summary_at_end/total*100:.1f}%)")
    print(f"Missing summaries: {total - with_summary}")
    print(f"Summaries not at end: {with_summary - summary_at_end}")

if __name__ == '__main__':
    main()
