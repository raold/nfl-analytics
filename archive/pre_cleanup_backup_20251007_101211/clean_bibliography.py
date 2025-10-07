#!/usr/bin/env python3
"""
Clean bibliography by removing unused entries and identifying those needing DOIs.
"""

import re
from pathlib import Path

def get_cited_keys(tex_dir):
    """Extract all citation keys used in the dissertation."""
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

    return cited_keys

def parse_bib_entries(bib_path):
    """Parse BibTeX file into individual entries."""
    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into entries
    entries = []
    current_entry = []
    in_entry = False
    brace_count = 0

    for line in content.split('\n'):
        if line.strip().startswith('@'):
            if current_entry:
                entries.append('\n'.join(current_entry))
            current_entry = [line]
            in_entry = True
            brace_count = line.count('{') - line.count('}')
        elif in_entry:
            current_entry.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and line.strip().endswith('}'):
                entries.append('\n'.join(current_entry))
                current_entry = []
                in_entry = False

    if current_entry:
        entries.append('\n'.join(current_entry))

    return entries

def extract_entry_key(entry):
    """Extract the citation key from a BibTeX entry."""
    match = re.search(r'@\w+\{([^,]+),', entry)
    return match.group(1) if match else None

def has_doi(entry):
    """Check if entry has DOI field."""
    return bool(re.search(r'^\s*doi\s*=', entry, re.MULTILINE | re.IGNORECASE))

def main():
    bib_path = Path("analysis/dissertation/references.bib")
    tex_dir = Path("analysis/dissertation")

    print("=" * 70)
    print("CLEANING BIBLIOGRAPHY")
    print("=" * 70)

    # Get cited keys
    cited_keys = get_cited_keys(tex_dir)
    print(f"\n✓ Found {len(cited_keys)} unique citations in dissertation")

    # Parse bibliography
    entries = parse_bib_entries(bib_path)
    print(f"✓ Found {len(entries)} entries in bibliography")

    # Classify entries
    used_entries = []
    unused_entries = []
    entries_without_doi = []

    for entry in entries:
        key = extract_entry_key(entry)
        if not key:
            continue

        if key in cited_keys:
            used_entries.append(entry)
            if not has_doi(entry):
                entries_without_doi.append(key)
        else:
            unused_entries.append((key, entry))

    print(f"\n✓ {len(used_entries)} entries are cited (will keep)")
    print(f"⚠️  {len(unused_entries)} entries are unused (will remove)")
    print(f"⚠️  {len(entries_without_doi)} cited entries lack DOIs")

    # Show unused entries
    if unused_entries:
        print("\nUnused entries to be removed:")
        for key, _ in unused_entries:
            print(f"  - {key}")

    # Create cleaned bibliography
    cleaned_bib = '\n\n'.join(used_entries)

    # Backup original
    backup_path = bib_path.with_suffix('.bib.backup')
    with open(bib_path, 'r') as f:
        with open(backup_path, 'w') as f_backup:
            f_backup.write(f.read())
    print(f"\n✓ Backed up original to {backup_path}")

    # Write cleaned bibliography
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_bib)

    print(f"✓ Wrote cleaned bibliography ({len(used_entries)} entries)")

    # Report entries needing DOIs
    if entries_without_doi:
        print(f"\n📋 Entries needing DOIs ({len(entries_without_doi)}):")
        for key in sorted(entries_without_doi)[:10]:
            print(f"  - {key}")
        if len(entries_without_doi) > 10:
            print(f"  ... and {len(entries_without_doi) - 10} more")

    return entries_without_doi

if __name__ == "__main__":
    main()
