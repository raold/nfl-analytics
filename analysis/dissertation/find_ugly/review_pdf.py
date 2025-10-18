#!/usr/bin/env python3
"""
PDF Visual Review Tool
Converts PDF pages to PNG images for visual inspection, then auto-deletes them.
"""

import subprocess
import os
import sys
from pathlib import Path

# Configuration
PDF_PATH = "/Users/dro/rice/nfl-analytics/analysis/dissertation/main/main.pdf"
OUTPUT_DIR = Path(__file__).parent / "pages"
BATCH_SIZE = 12
DPI = 150  # Balance between quality and file size

def get_page_count():
    """Get total number of pages from LaTeX log."""
    log_path = Path(PDF_PATH).parent / "main.log"
    with open(log_path, 'r') as f:
        for line in f:
            if "Output written on main.pdf" in line:
                # Extract page count from: "Output written on main.pdf (362 pages, 5356034 bytes)."
                parts = line.split('(')[1].split()[0]
                return int(parts)
    return None

def convert_page_range(start_page, end_page):
    """Convert a range of PDF pages to PNG images using Ghostscript."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ghostscript command for batch conversion
    cmd = [
        "gs",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        "-sDEVICE=png16m",
        f"-r{DPI}",
        f"-dFirstPage={start_page}",
        f"-dLastPage={end_page}",
        f"-sOutputFile={OUTPUT_DIR}/page_%03d.png",
        PDF_PATH
    ]

    print(f"Converting pages {start_page}-{end_page}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error converting pages: {result.stderr}")
        return False

    print(f"✓ Converted pages {start_page}-{end_page}")
    return True

def delete_images():
    """Delete all PNG images in the output directory."""
    if OUTPUT_DIR.exists():
        png_files = list(OUTPUT_DIR.glob("*.png"))
        for f in png_files:
            f.unlink()
        print(f"✓ Deleted {len(png_files)} images")

def cleanup_directory():
    """Remove the pages directory entirely."""
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print("✓ Cleaned up pages directory")

if __name__ == "__main__":
    page_count = get_page_count()
    if not page_count:
        print("Error: Could not determine page count from main.log")
        sys.exit(1)

    print(f"PDF has {page_count} pages")
    print(f"Will process in batches of {BATCH_SIZE}")
    print(f"Images will be saved temporarily to: {OUTPUT_DIR}")
    print()

    # Ask which mode to run in
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            # Batch mode: specify which batch to convert
            batch_num = int(sys.argv[2])
            start_page = (batch_num - 1) * BATCH_SIZE + 1
            end_page = min(batch_num * BATCH_SIZE, page_count)
            convert_page_range(start_page, end_page)
        elif sys.argv[1] == "cleanup":
            # Cleanup mode: delete images
            delete_images()
        elif sys.argv[1] == "cleanup-all":
            # Full cleanup: remove directory
            cleanup_directory()
        elif sys.argv[1] == "pages":
            # Single page or range: e.g., "pages 50" or "pages 50-55"
            page_spec = sys.argv[2]
            if '-' in page_spec:
                start, end = page_spec.split('-')
                convert_page_range(int(start), int(end))
            else:
                page = int(page_spec)
                convert_page_range(page, page)
    else:
        print("Usage:")
        print("  python review_pdf.py batch <N>        # Convert batch N")
        print("  python review_pdf.py pages <N>        # Convert single page N")
        print("  python review_pdf.py pages <N>-<M>    # Convert pages N to M")
        print("  python review_pdf.py cleanup          # Delete all images")
        print("  python review_pdf.py cleanup-all      # Delete images and directory")
        print()
        total_batches = (page_count + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total batches: {total_batches}")
        print(f"Batch size: {BATCH_SIZE} pages")
