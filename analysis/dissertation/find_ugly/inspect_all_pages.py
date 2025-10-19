#!/usr/bin/env python3
"""
Comprehensive Page-by-Page Dissertation Inspector
Converts all PDF pages to PNG, then inspects each systematically.
"""

import subprocess
import sys
from pathlib import Path

PDF_PATH = "/Users/dro/rice/nfl-analytics/analysis/dissertation/main/main.pdf"
OUTPUT_DIR = Path(__file__).parent / "pages"
DPI = 150

def get_page_count():
    """Get total pages from log file."""
    log_path = Path(PDF_PATH).parent / "main.log"
    with open(log_path, 'r') as f:
        for line in f:
            if "Output written on main.pdf" in line:
                parts = line.split('(')[1].split()[0]
                return int(parts)
    return None

def convert_all_pages(total_pages):
    """Convert ALL pages to PNG sequentially."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Converting all {total_pages} pages to PNG...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Convert all at once with Ghostscript
    cmd = [
        "gs",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        "-sDEVICE=png16m",
        f"-r{DPI}",
        f"-dFirstPage=1",
        f"-dLastPage={total_pages}",
        f"-sOutputFile={OUTPUT_DIR}/page_%03d.png",
        PDF_PATH
    ]

    print("Starting conversion (this will take a few minutes)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    # Verify all pages were created
    created = list(OUTPUT_DIR.glob("page_*.png"))
    print(f"\n✓ Successfully converted {len(created)} pages")

    return True

if __name__ == "__main__":
    page_count = get_page_count()
    if not page_count:
        print("Error: Could not determine page count")
        sys.exit(1)

    print(f"PDF has {page_count} pages\n")

    if "--convert" in sys.argv or not list(OUTPUT_DIR.glob("page_*.png")):
        success = convert_all_pages(page_count)
        if not success:
            sys.exit(1)
    else:
        existing = len(list(OUTPUT_DIR.glob("page_*.png")))
        print(f"Found {existing} existing PNGs (use --convert to reconvert)")

    print("\n" + "="*80)
    print("ALL PAGES CONVERTED - Ready for systematic inspection")
    print("="*80)
    print(f"\nImages location: {OUTPUT_DIR}/")
    print(f"Total pages: {page_count}")
    print("\nNext: Use Claude to inspect each page systematically")
