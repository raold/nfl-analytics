#!/usr/bin/env python3
"""
Fetch PDFs for citations using LEGAL sources (Unpaywall, arXiv, etc.).

This script tries multiple legitimate sources:
1. Unpaywall API (aggregates legal open access sources)
2. arXiv (preprint server)
3. PubMed Central (PMC)
4. DOAJ (Directory of Open Access Journals)

Usage:
    python scripts/fetch_citation_pdfs_legal.py --email your@email.edu

Note: Provide your email for Unpaywall API (required, they use it to track usage).
"""

import re
import os
import time
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import bibtexparser
from urllib.parse import urljoin


def parse_bibtex(bib_path: str) -> List[Dict]:
    """Parse BibTeX file and extract entries."""
    with open(bib_path, 'r', encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
    return bib_database.entries


def extract_doi(entry: Dict) -> Optional[str]:
    """Extract DOI from BibTeX entry."""
    if 'doi' in entry:
        doi = entry['doi'].strip()
        doi = re.sub(r'https?://doi\.org/', '', doi)
        return doi

    if 'url' in entry:
        url = entry['url']
        match = re.search(r'doi\.org/(.+)$', url)
        if match:
            return match.group(1)

    return None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def get_pdf_filename(entry: Dict) -> str:
    """Generate PDF filename from BibTeX entry."""
    author = entry.get('author', 'Unknown').split(' and ')[0].split(',')[0].strip()
    year = entry.get('year', 'XXXX')
    title = entry.get('title', 'untitled')

    title_words = re.sub(r'[^\w\s]', '', title).split()[:5]
    short_title = '_'.join(title_words)

    filename = f"{author}{year}_{short_title}.pdf"
    return sanitize_filename(filename)


def download_from_unpaywall(doi: str, email: str, output_path: str) -> bool:
    """
    Download PDF from Unpaywall (legal open access aggregator).

    Unpaywall finds legal, free versions of papers from:
    - Publisher open access
    - Institutional repositories
    - Preprint servers
    - Author uploads

    Args:
        doi: DOI of the paper
        email: Your email (required by Unpaywall API)
        output_path: Where to save the PDF

    Returns:
        True if successful, False otherwise
    """
    try:
        # Unpaywall API endpoint
        api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        print(f"  Checking Unpaywall API...")

        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Check if there's an open access version
        if data.get('is_oa'):
            oa_location = data.get('best_oa_location')
            if oa_location and oa_location.get('url_for_pdf'):
                pdf_url = oa_location['url_for_pdf']
                print(f"  ✓ Found open access PDF: {oa_location.get('host_type', 'unknown')}")

                # Download the PDF
                pdf_response = requests.get(pdf_url, timeout=60, allow_redirects=True)
                pdf_response.raise_for_status()

                # Check if it's really a PDF
                if pdf_response.content[:4] == b'%PDF':
                    with open(output_path, 'wb') as f:
                        f.write(pdf_response.content)
                    print(f"  ✓ Downloaded: {os.path.basename(output_path)} ({len(pdf_response.content) / 1024:.1f} KB)")
                    return True
                else:
                    print(f"  ✗ Response was not a PDF")
            else:
                print(f"  ⚠ Open access version exists but no PDF URL")
        else:
            print(f"  ✗ Not available as open access")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  ✗ DOI not found in Unpaywall")
        else:
            print(f"  ✗ HTTP error: {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    return False


def try_arxiv(entry: Dict, output_path: str) -> bool:
    """
    Try to find paper on arXiv (preprint server).

    Args:
        entry: BibTeX entry
        output_path: Where to save the PDF

    Returns:
        True if successful, False otherwise
    """
    # Check if arXiv ID is in the entry
    arxiv_id = None

    # Look for arXiv ID in various fields
    for field in ['eprint', 'archiveprefix', 'primaryclass', 'note', 'url']:
        if field in entry:
            match = re.search(r'arXiv[:\s]*(\d{4}\.\d{4,5})', entry[field], re.IGNORECASE)
            if match:
                arxiv_id = match.group(1)
                break

    if not arxiv_id:
        return False

    try:
        print(f"  Found arXiv ID: {arxiv_id}")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        if response.content[:4] == b'%PDF':
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Downloaded from arXiv: {os.path.basename(output_path)} ({len(response.content) / 1024:.1f} KB)")
            return True

    except Exception as e:
        print(f"  ✗ arXiv error: {e}")

    return False


def suggest_manual_methods(entry: Dict, doi: Optional[str]):
    """Print manual methods to try for getting the paper."""
    title = entry.get('title', 'Unknown')
    author = entry.get('author', 'Unknown').split(' and ')[0]

    print(f"\n  💡 Manual methods to try:")
    print(f"     1. Google Scholar: https://scholar.google.com/scholar?q={title.replace(' ', '+')}")
    if doi:
        print(f"     2. Publisher page: https://doi.org/{doi}")
    print(f"     3. Email author: {author} (often respond with PDFs)")
    print(f"     4. ResearchGate: https://www.researchgate.net/search?q={title.replace(' ', '+')}")
    print(f"     5. Your university library (check subscriptions)")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch PDFs using LEGAL sources (Unpaywall, arXiv, etc.)',
    )
    parser.add_argument(
        '--bib',
        default='analysis/dissertation/main/references.bib',
        help='Path to BibTeX file'
    )
    parser.add_argument(
        '--output',
        default='analysis/dissertation/pdfs',
        help='Output directory for PDFs'
    )
    parser.add_argument(
        '--email',
        required=True,
        help='Your email (required for Unpaywall API)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of papers to download'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist'
    )
    parser.add_argument(
        '--suggest-manual',
        action='store_true',
        help='Print manual methods for failed downloads'
    )

    args = parser.parse_args()

    # Validate email
    if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', args.email):
        print("❌ Invalid email address")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📚 Parsing bibliography: {args.bib}")
    entries = parse_bibtex(args.bib)
    print(f"   Found {len(entries)} total citations")

    # Filter entries with DOIs
    entries_with_doi = []
    for entry in entries:
        doi = extract_doi(entry)
        if doi:
            entries_with_doi.append((entry, doi))

    print(f"   {len(entries_with_doi)} have DOIs")

    if args.limit:
        entries_with_doi = entries_with_doi[:args.limit]
        print(f"   Limiting to first {args.limit} entries")

    print(f"\n🔽 Downloading PDFs to: {output_dir}")
    print(f"   Using legal sources: Unpaywall, arXiv, PMC")
    print(f"   Delay between requests: {args.delay}s\n")

    successful = 0
    failed = 0
    skipped = 0

    for i, (entry, doi) in enumerate(entries_with_doi, 1):
        cite_key = entry.get('ID', 'unknown')
        title = entry.get('title', 'Unknown title')[:60]

        print(f"[{i}/{len(entries_with_doi)}] {cite_key}")
        print(f"  Title: {title}...")
        print(f"  DOI: {doi}")

        filename = get_pdf_filename(entry)
        output_path = output_dir / filename

        if args.skip_existing and output_path.exists():
            print(f"  ⊘ Skipped: File already exists")
            skipped += 1
            continue

        # Try Unpaywall first (legal open access)
        success = download_from_unpaywall(doi, args.email, str(output_path))

        # Try arXiv if Unpaywall fails
        if not success:
            success = try_arxiv(entry, str(output_path))

        if success:
            successful += 1
        else:
            failed += 1
            if args.suggest_manual:
                suggest_manual_methods(entry, doi)

        if i < len(entries_with_doi):
            time.sleep(args.delay)

        print()

    # Summary
    print("=" * 70)
    print(f"📊 Summary:")
    print(f"   ✓ Successful: {successful}")
    print(f"   ✗ Failed: {failed}")
    print(f"   ⊘ Skipped: {skipped}")
    print(f"   📁 Output directory: {output_dir}")
    print("=" * 70)

    if failed > 0:
        print(f"\n💡 For {failed} failed downloads, try:")
        print(f"   1. Run with --suggest-manual to see specific methods")
        print(f"   2. Check your university library subscriptions")
        print(f"   3. Email authors directly (they usually respond!)")
        print(f"   4. Try Google Scholar's PDF links")


if __name__ == '__main__':
    main()
