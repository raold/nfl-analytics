#!/usr/bin/env python3
"""
Audit bibliography for completeness, duplicates, and missing information.
"""

import re
from pathlib import Path
from collections import defaultdict

def parse_bib_file(bib_path):
    """Parse BibTeX file and extract entries."""

    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract all entries
    entries = re.findall(r'@(\w+)\{([^,]+),([^}]+)\}', content, re.DOTALL)

    parsed_entries = []
    for entry_type, cite_key, fields in entries:
        entry = {
            'type': entry_type,
            'key': cite_key,
            'fields': {}
        }

        # Parse fields
        field_pattern = r'(\w+)\s*=\s*\{([^}]+)\}|(\w+)\s*=\s*"([^"]+)"'
        for match in re.finditer(field_pattern, fields):
            if match.group(1):
                field_name = match.group(1)
                field_value = match.group(2)
            else:
                field_name = match.group(3)
                field_value = match.group(4)

            entry['fields'][field_name.lower()] = field_value

        parsed_entries.append(entry)

    return parsed_entries

def check_citations_used(tex_dir, bib_entries):
    """Check which citations are actually used in the dissertation."""

    tex_files = list(Path(tex_dir).rglob("*.tex"))

    cited_keys = set()

    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all citations
            citations = re.findall(r'\\cite[pt]?\{([^}]+)\}', content)
            for citation_list in citations:
                for key in citation_list.split(','):
                    cited_keys.add(key.strip())
        except:
            pass

    bib_keys = {entry['key'] for entry in bib_entries}

    unused = bib_keys - cited_keys
    missing = cited_keys - bib_keys

    return cited_keys, unused, missing

def check_entry_completeness(entries):
    """Check if entries have all required fields."""

    required_fields = {
        'article': ['author', 'title', 'journal', 'year'],
        'book': ['author', 'title', 'publisher', 'year'],
        'inproceedings': ['author', 'title', 'booktitle', 'year'],
        'incollection': ['author', 'title', 'booktitle', 'publisher', 'year'],
    }

    recommended_fields = {
        'article': ['volume', 'number', 'pages', 'doi'],
        'book': ['isbn'],
        'inproceedings': ['pages', 'doi'],
    }

    issues = []

    for entry in entries:
        entry_type = entry['type'].lower()
        fields = entry['fields']

        if entry_type in required_fields:
            for req_field in required_fields[entry_type]:
                if req_field not in fields:
                    issues.append(f"{entry['key']}: Missing required field '{req_field}'")

        if entry_type in recommended_fields:
            missing_rec = []
            for rec_field in recommended_fields[entry_type]:
                if rec_field not in fields:
                    missing_rec.append(rec_field)

            if missing_rec:
                issues.append(f"{entry['key']}: Missing recommended fields: {', '.join(missing_rec)}")

    return issues

def find_duplicate_entries(entries):
    """Find potential duplicate entries."""

    title_map = defaultdict(list)

    for entry in entries:
        title = entry['fields'].get('title', '').lower()
        if title:
            # Normalize title
            title = re.sub(r'[^a-z0-9\s]', '', title)
            title = ' '.join(title.split())
            title_map[title].append(entry['key'])

    duplicates = {title: keys for title, keys in title_map.items() if len(keys) > 1}

    return duplicates

def main():
    """Run bibliography audit."""

    bib_path = Path("analysis/dissertation/references.bib")
    tex_dir = Path("analysis/dissertation")

    if not bib_path.exists():
        print(f"Error: Bibliography file not found at {bib_path}")
        return

    print("=" * 70)
    print("BIBLIOGRAPHY AUDIT")
    print("=" * 70)

    # Parse bibliography
    entries = parse_bib_file(bib_path)
    print(f"\n✓ Found {len(entries)} entries in bibliography")

    # Check citation usage
    cited_keys, unused, missing = check_citations_used(tex_dir, entries)

    print(f"\n{len(cited_keys)} citations used in text")

    if unused:
        print(f"\n⚠️  {len(unused)} unused entries in bibliography:")
        for key in sorted(unused)[:10]:  # Show first 10
            print(f"     - {key}")
        if len(unused) > 10:
            print(f"     ... and {len(unused) - 10} more")

    if missing:
        print(f"\n❌ {len(missing)} missing entries (cited but not in bibliography):")
        for key in sorted(missing):
            print(f"     - {key}")

    # Check completeness
    completeness_issues = check_entry_completeness(entries)

    if completeness_issues:
        print(f"\n⚠️  {len(completeness_issues)} completeness issues:")
        for issue in completeness_issues[:15]:  # Show first 15
            print(f"     - {issue}")
        if len(completeness_issues) > 15:
            print(f"     ... and {len(completeness_issues) - 15} more")

    # Check for duplicates
    duplicates = find_duplicate_entries(entries)

    if duplicates:
        print(f"\n⚠️  {len(duplicates)} potential duplicate entries:")
        for title, keys in list(duplicates.items())[:5]:
            print(f"     - {keys}")
    else:
        print("\n✓ No duplicate entries found")

    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    entry_types = defaultdict(int)
    for entry in entries:
        entry_types[entry['type']] += 1

    print("\nEntry types:")
    for entry_type, count in sorted(entry_types.items(), key=lambda x: -x[1]):
        print(f"  {entry_type}: {count}")

    # Check for DOIs
    with_doi = sum(1 for e in entries if 'doi' in e['fields'])
    print(f"\nEntries with DOI: {with_doi}/{len(entries)} ({100*with_doi/len(entries):.1f}%)")

    # Check for URLs
    with_url = sum(1 for e in entries if 'url' in e['fields'])
    print(f"Entries with URL: {with_url}/{len(entries)} ({100*with_url/len(entries):.1f}%)")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. Add DOIs where missing (improves discoverability)")
    print("2. Remove unused entries to keep bibliography clean")
    print("3. Ensure all cited works are in references.bib")
    print("4. Check for duplicate entries with similar titles")
    print("5. Add volume/number/pages for journal articles")

if __name__ == "__main__":
    main()
