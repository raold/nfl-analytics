#!/usr/bin/env python3
"""
Add DOIs to bibliography entries.
"""

import re
from pathlib import Path

# DOI mappings for common academic papers
DOIS = {
    'dixon1997': '10.2307/2986290',
    'baio2010': '10.1080/02664760802684177',
    'stern1991': '10.1080/00031305.1991.10475812',
    'koopman2015': '10.1111/rssa.12042',
    'karlis2003': '10.1111/1467-9884.00366',
    'harville1980': '10.1080/01621459.1980.10477504',
    'lock2014': '10.1515/jqas-2013-0100',
    'sauer1998': 'https://www.jstor.org/stable/2565043',
    'levitt2004': '10.1111/j.0013-0133.2004.00207.x',
    'skellam1946': '10.2307/2981372',
    'maher1982': '10.2307/2347625',
    'bradleyterry1952': '10.1093/biomet/39.3-4.324',
    'szalkowski2012': '10.48550/arXiv.1201.0309',
    'daniels1954': '10.1214/aoms/1177728652',
    'dudik2014': '10.1214/14-STS500',
    'vanhasselt2016': '10.1609/aaai.v30i1.10295',
    'achiam2017cpo': '10.48550/arXiv.1705.10528',
    'levine2020': '10.48550/arXiv.2005.01643',
    'schulman2017ppo': '10.48550/arXiv.1707.06347',
    'wang2016dueling': '10.48550/arXiv.1511.06581',
    'schaul2016per': '10.48550/arXiv.1511.05952',
    'schulman2016gae': '10.48550/arXiv.1506.02438',
    'schulman2015trpo': '10.48550/arXiv.1502.05477',
    'wu2019brac': '10.48550/arXiv.1910.01708',
    'kostrikov2021iql': '10.48550/arXiv.2110.06169',
    'nair2020awac': '10.48550/arXiv.2006.09359',
    'fujimoto2021td3bc': 'https://proceedings.neurips.cc/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html',
}

def add_dois_to_bib(bib_path):
    """Add DOIs to bibliography entries."""

    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()

    modifications = 0

    for key, doi in DOIS.items():
        # Find the entry
        pattern = r'(@\w+\{' + key + r',.*?)\n(\})'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            entry_content = match.group(1)
            closing_brace = match.group(2)

            # Check if DOI already exists
            if 'doi' not in entry_content.lower():
                # Add DOI before closing brace
                if doi.startswith('http'):
                    new_entry = f"{entry_content},\n  url={{{doi}}}\n{closing_brace}"
                else:
                    new_entry = f"{entry_content},\n  doi={{{doi}}}\n{closing_brace}"

                content = content.replace(match.group(0), new_entry)
                modifications += 1
                print(f"✓ Added DOI to {key}")
            else:
                print(f"  - {key} already has DOI")

    # Write updated content
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return modifications

def main():
    bib_path = Path("analysis/dissertation/references.bib")

    if not bib_path.exists():
        print(f"Error: {bib_path} not found")
        return

    print("=" * 70)
    print("ADDING DOIs TO BIBLIOGRAPHY")
    print("=" * 70)
    print()

    mods = add_dois_to_bib(bib_path)

    print()
    print("=" * 70)
    print(f"✓ Added {mods} DOIs to bibliography")
    print("=" * 70)

if __name__ == "__main__":
    main()
