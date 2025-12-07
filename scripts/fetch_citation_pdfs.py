#!/usr/bin/env python3
"""
Fetch PDFs for citations from bibliography using Sci-Hub.

Usage:
    python scripts/fetch_citation_pdfs.py [--bib PATH] [--output DIR] [--limit N]

Note: Try legitimate sources first (university library, author websites).
Sci-Hub should be a last resort for papers you cannot access otherwise.
"""

import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import bibtexparser
import requests

# Sci-Hub mirrors (rotate if one is down)
# Updated list as of Nov 2024 - these domains change frequently
SCIHUB_MIRRORS = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.ren",
    "https://sci-hub.wf",
    "https://sci-hub.mksa.top",
]

# LibGen mirrors as fallback
LIBGEN_MIRRORS = [
    "https://libgen.is",
    "https://libgen.rs",
    "https://libgen.li",
]


def parse_bibtex(bib_path: str) -> list[dict]:
    """Parse BibTeX file and extract entries."""
    with open(bib_path, encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
    return bib_database.entries


def extract_doi(entry: dict) -> str | None:
    """Extract DOI from BibTeX entry."""
    # Check DOI field
    if 'doi' in entry:
        doi = entry['doi'].strip()
        # Clean up DOI (remove URL prefix if present)
        doi = re.sub(r'https?://doi\.org/', '', doi)
        return doi

    # Check URL field for DOI
    if 'url' in entry:
        url = entry['url']
        match = re.search(r'doi\.org/(.+)$', url)
        if match:
            return match.group(1)

    return None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    # Replace invalid characters with underscore
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def get_pdf_filename(entry: dict) -> str:
    """Generate PDF filename from BibTeX entry."""
    # Try to use: FirstAuthorYear_ShortTitle.pdf
    author = entry.get('author', 'Unknown').split(' and ')[0].split(',')[0].strip()
    year = entry.get('year', 'XXXX')
    title = entry.get('title', 'untitled')

    # Clean title (take first few words)
    title_words = re.sub(r'[^\w\s]', '', title).split()[:5]
    short_title = '_'.join(title_words)

    filename = f"{author}{year}_{short_title}.pdf"
    return sanitize_filename(filename)


def download_from_scihub(doi: str, output_path: str, mirrors: list[str]) -> bool:
    """
    Download PDF from Sci-Hub using DOI.

    Args:
        doi: DOI of the paper
        output_path: Where to save the PDF
        mirrors: List of Sci-Hub mirrors to try

    Returns:
        True if successful, False otherwise
    """
    # More complete headers to avoid detection
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
    }

    # Create a session to maintain cookies
    session = requests.Session()
    session.headers.update(headers)

    for mirror in mirrors:
        try:
            # Request the DOI page
            url = f"{mirror}/{doi}"
            print(f"  Trying {mirror}...")

            response = session.get(url, timeout=30, allow_redirects=True)

            # Don't raise for 403, just skip to next mirror
            if response.status_code == 403:
                print("  ✗ 403 Forbidden (may be blocked or Cloudflare protection)")
                continue

            response.raise_for_status()

            # Look for PDF iframe or direct link
            # Sci-Hub typically embeds PDF in iframe with src="//..."
            pdf_match = re.search(r'<iframe[^>]+src="([^"]+\.pdf[^"]*)"', response.text, re.IGNORECASE)

            if not pdf_match:
                # Try alternative pattern
                pdf_match = re.search(r'location\.href\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\']', response.text)

            if pdf_match:
                pdf_url = pdf_match.group(1)

                # Handle protocol-relative URLs
                if pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url
                elif not pdf_url.startswith('http'):
                    pdf_url = urljoin(mirror, pdf_url)

                print(f"  Found PDF URL: {pdf_url[:80]}...")

                # Download the PDF
                pdf_response = requests.get(pdf_url, headers=headers, timeout=60)
                pdf_response.raise_for_status()

                # Check if we got a PDF (check magic bytes)
                if pdf_response.content[:4] == b'%PDF':
                    with open(output_path, 'wb') as f:
                        f.write(pdf_response.content)
                    print(f"  ✓ Downloaded: {os.path.basename(output_path)} ({len(pdf_response.content) / 1024:.1f} KB)")
                    return True
                else:
                    print("  ✗ Response was not a PDF")
                    continue
            else:
                print("  ✗ Could not find PDF link on page")
                continue

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error with {mirror}: {e}")
            continue
        except Exception as e:
            print(f"  ✗ Unexpected error with {mirror}: {e}")
            continue

    return False


def download_from_libgen(doi: str, output_path: str, mirrors: list[str]) -> bool:
    """
    Download PDF from LibGen using DOI.

    Args:
        doi: DOI of the paper
        output_path: Where to save the PDF
        mirrors: List of LibGen mirrors to try

    Returns:
        True if successful, False otherwise
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    for mirror in mirrors:
        try:
            # LibGen search by DOI
            search_url = f"{mirror}/scimag/?q={doi}"
            print(f"  Trying LibGen {mirror}...")

            response = requests.get(search_url, headers=headers, timeout=30)

            if response.status_code == 403:
                print("  ✗ 403 Forbidden")
                continue

            response.raise_for_status()

            # Look for download links in the results
            # LibGen has links like: /scimag/...md5...
            link_match = re.search(r'href="(/scimag/[^"]+)"', response.text)

            if link_match:
                download_path = link_match.group(1)
                pdf_url = urljoin(mirror, download_path)
                print("  Found LibGen link...")

                # Follow the link to get actual PDF
                pdf_response = requests.get(pdf_url, headers=headers, timeout=60, allow_redirects=True)
                pdf_response.raise_for_status()

                # Check if we got a PDF
                if pdf_response.content[:4] == b'%PDF':
                    with open(output_path, 'wb') as f:
                        f.write(pdf_response.content)
                    print(f"  ✓ Downloaded from LibGen: {os.path.basename(output_path)} ({len(pdf_response.content) / 1024:.1f} KB)")
                    return True

        except Exception as e:
            print(f"  ✗ Error with {mirror}: {e}")
            continue

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Fetch PDFs for bibliography citations from Sci-Hub',
        epilog='Note: Try legitimate sources first (university library, author websites).'
    )
    parser.add_argument(
        '--bib',
        default='analysis/dissertation/main/references.bib',
        help='Path to BibTeX file (default: analysis/dissertation/main/references.bib)'
    )
    parser.add_argument(
        '--output',
        default='analysis/dissertation/pdfs',
        help='Output directory for PDFs (default: analysis/dissertation/pdfs)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of papers to download (for testing)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between requests in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist'
    )

    args = parser.parse_args()

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

    # Download PDFs
    print(f"\n🔽 Downloading PDFs to: {output_dir}")
    print(f"   Using mirrors: {', '.join(SCIHUB_MIRRORS)}")
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

        # Generate filename
        filename = get_pdf_filename(entry)
        output_path = output_dir / filename

        # Skip if exists
        if args.skip_existing and output_path.exists():
            print("  ⊘ Skipped: File already exists")
            skipped += 1
            continue

        # Try Sci-Hub first
        success = download_from_scihub(doi, str(output_path), SCIHUB_MIRRORS)

        # If Sci-Hub fails, try LibGen
        if not success:
            print("  → Trying LibGen as fallback...")
            success = download_from_libgen(doi, str(output_path), LIBGEN_MIRRORS)

        if success:
            successful += 1
        else:
            failed += 1
            print("  ✗ Failed to download from all sources")

        # Rate limiting
        if i < len(entries_with_doi):
            time.sleep(args.delay)

        print()

    # Summary
    print("=" * 70)
    print("📊 Summary:")
    print(f"   ✓ Successful: {successful}")
    print(f"   ✗ Failed: {failed}")
    print(f"   ⊘ Skipped: {skipped}")
    print(f"   📁 Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
