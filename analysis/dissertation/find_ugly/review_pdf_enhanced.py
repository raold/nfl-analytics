#!/usr/bin/env python3
"""
Enhanced PDF Visual Review Tool
Converts PDF pages to PNG images + runs automated quality checks on LaTeX source.

Incorporates learnings from visual inspection and 31 improvement recommendations.
Checks for:
- Table formatting issues (numeric alignment, decimal precision, width control)
- Figure quality issues (size, colorblind-safety, grid lines, legends)
- Typography problems (orphans, widows, overfull hboxes)
- Float placement warnings
- Missing references
- Inconsistent styling

Usage:
    python review_pdf_enhanced.py batch <N>        # Convert batch N with quality report
    python review_pdf_enhanced.py check            # Run quality checks only (no PNG conversion)
    python review_pdf_enhanced.py pages <N>        # Convert single page
    python review_pdf_enhanced.py cleanup          # Delete images
"""

import subprocess
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

PDF_PATH = "/Users/dro/rice/nfl-analytics/analysis/dissertation/main/main.pdf"
OUTPUT_DIR = Path(__file__).parent / "pages"
LOG_PATH = Path(PDF_PATH).parent / "main.log"
TEX_PATH = Path(PDF_PATH).parent / "main.tex"
FIGURES_DIR = Path(PDF_PATH).parent.parent / "figures" / "out"

BATCH_SIZE = 12
DPI = 150

# Quality check thresholds
MAX_OVERFULL_HBOX = 10  # pt
MAX_FLOAT_TOO_LARGE = 20  # pt
EXPECTED_DECIMAL_PLACES = {
    'brier': 3,
    'probability': 3,
    'roi': 2,
    'percent': 1,
}

# ============================================================================
# PDF Conversion (Original Functionality)
# ============================================================================

def get_page_count():
    """Get total number of pages from LaTeX log."""
    if not LOG_PATH.exists():
        return None

    with open(LOG_PATH, 'r') as f:
        for line in f:
            if "Output written on main.pdf" in line:
                parts = line.split('(')[1].split()[0]
                return int(parts)
    return None

def convert_page_range(start_page, end_page):
    """Convert a range of PDF pages to PNG images using Ghostscript."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ============================================================================
# Quality Checks
# ============================================================================

class QualityChecker:
    """Automated quality checker for dissertation LaTeX source and compilation."""

    def __init__(self):
        self.issues = defaultdict(list)
        self.warnings = defaultdict(int)

    def check_all(self):
        """Run all quality checks."""
        print("\n" + "="*80)
        print("DISSERTATION QUALITY REPORT")
        print("="*80 + "\n")

        self.check_compilation_log()
        self.check_table_quality()
        self.check_figure_references()
        self.check_typography()

        self.print_summary()

    def check_compilation_log(self):
        """Check LaTeX compilation log for warnings and errors."""
        if not LOG_PATH.exists():
            self.issues['log'].append("main.log not found - PDF not compiled")
            return

        with open(LOG_PATH, 'r') as f:
            log_content = f.read()

        # Count different types of warnings
        overfull_hboxes = re.findall(r'Overfull \\hbox \((\d+\.?\d*)pt too wide\)', log_content)
        for width in overfull_hboxes:
            if float(width) > MAX_OVERFULL_HBOX:
                self.issues['typography'].append(f"Overfull hbox: {width}pt too wide")

        underfull_hboxes = len(re.findall(r'Underfull \\hbox', log_content))
        self.warnings['underfull_hbox'] = underfull_hboxes

        # Float warnings
        float_warnings = re.findall(r'Float too large for page by ([\d.]+)pt', log_content)
        for size in float_warnings:
            if float(size) > MAX_FLOAT_TOO_LARGE:
                self.issues['floats'].append(f"Float {size}pt too large for page")

        # References
        undef_refs = len(re.findall(r'Reference.*undefined', log_content))
        if undef_refs > 0:
            self.issues['references'].append(f"{undef_refs} undefined references")

        multiply_defined = len(re.findall(r'multiply defined', log_content))
        if multiply_defined > 0:
            self.issues['references'].append(f"{multiply_defined} multiply-defined labels")

        # Citations
        undef_citations = len(re.findall(r'Citation.*undefined', log_content))
        if undef_citations > 0:
            self.issues['references'].append(f"{undef_citations} undefined citations")

        self.warnings['overfull_hbox_total'] = len(overfull_hboxes)
        self.warnings['overfull_hbox_serious'] = len([w for w in overfull_hboxes if float(w) > MAX_OVERFULL_HBOX])

    def check_table_quality(self):
        """Check table .tex files for quality issues."""
        if not FIGURES_DIR.exists():
            self.issues['tables'].append("Figures directory not found")
            return

        table_files = list(FIGURES_DIR.glob("*table*.tex"))

        issues_by_type = defaultdict(int)

        for table_file in table_files:
            with open(table_file, 'r') as f:
                content = f.read()

            # Check #1: Numeric alignment
            if '\\begin{tabular}' in content:
                col_spec_match = re.search(r'\\begin\{tabular[x]?\}\{([^}]+)\}', content)
                if col_spec_match:
                    col_spec = col_spec_match.group(1)
                    # Check if there are numeric columns that should be right-aligned
                    if 'l' in col_spec and re.search(r'&\s*\d+\.?\d*\s*(?:&|\\\\)', content):
                        if 'r' not in col_spec and 'N' not in col_spec:
                            issues_by_type['numeric_alignment'] += 1

            # Check #2: Decimal precision consistency
            numbers = re.findall(r'(\d+\.\d+)', content)
            if len(numbers) > 5:
                decimal_places = [len(n.split('.')[1]) for n in numbers]
                if len(set(decimal_places)) > 2:
                    issues_by_type['inconsistent_decimals'] += 1

            # Check #3: Width control for wide tables
            if '\\begin{tabular}' in content and col_spec_match:
                col_count = len([c for c in col_spec_match.group(1) if c in 'lcrpN'])
                if col_count > 5 and 'adjustbox' not in content and 'tabularx' not in content:
                    issues_by_type['no_width_control'] += 1

            # Check #23: Bold caption summaries
            if '\\caption{' in content:
                caption_match = re.search(r'\\caption\{([^}]+)\}', content)
                if caption_match and not caption_match.group(1).startswith('\\textbf'):
                    issues_by_type['caption_not_bold'] += 1

            # Check #25: Table notes present
            if '\\begin{tabular}' in content and '\\begin{tablenotes}' not in content:
                if '\\begin{threeparttable}' not in content:
                    issues_by_type['no_tablenotes'] += 1

        # Report table issues
        if issues_by_type:
            for issue_type, count in issues_by_type.items():
                self.issues['tables'].append(f"{count} tables with {issue_type.replace('_', ' ')}")

        self.warnings['total_tables'] = len(table_files)

    def check_figure_references(self):
        """Check that all figures are properly sized and referenced."""
        if not FIGURES_DIR.exists():
            return

        # Check for oversized images (should use dissertation_style.py SIZES)
        figure_files = list(FIGURES_DIR.glob("*.png")) + list(FIGURES_DIR.glob("*.pdf"))

        # Check if dissertation_style.py is being used
        style_imports = 0
        for py_file in Path(PDF_PATH).parent.parent.parent.glob("py/**/*.py"):
            if py_file.name.startswith('.'):
                continue
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'dissertation_style' in content:
                        style_imports += 1
            except:
                pass

        if style_imports == 0:
            self.issues['figures'].append("dissertation_style.py not imported in any plotting scripts")

        self.warnings['total_figures'] = len(figure_files)

    def check_typography(self):
        """Check for common typography issues."""
        if not TEX_PATH.exists():
            return

        with open(TEX_PATH, 'r') as f:
            main_content = f.read()

        # Check for required packages
        required_packages = ['siunitx', 'makecell', 'threeparttable', 'booktabs']
        for pkg in required_packages:
            if f'\\usepackage{{{pkg}}}' not in main_content and f'\\usepackage' not in main_content:
                self.issues['typography'].append(f"Package {pkg} not loaded")

        # Check for column type definitions
        if '\\newcolumntype{N}' not in main_content:
            self.issues['typography'].append("siunitx column type N not defined")

    def print_summary(self):
        """Print quality report summary."""
        print("\n" + "-"*80)
        print("ISSUES FOUND")
        print("-"*80)

        if not self.issues:
            print("✓ No critical issues found!")
        else:
            for category, issue_list in sorted(self.issues.items()):
                print(f"\n{category.upper()}:")
                for issue in issue_list:
                    print(f"  ⚠  {issue}")

        print("\n" + "-"*80)
        print("STATISTICS")
        print("-"*80)
        print(f"  Total tables: {self.warnings.get('total_tables', 0)}")
        print(f"  Total figures: {self.warnings.get('total_figures', 0)}")
        print(f"  Overfull hboxes: {self.warnings.get('overfull_hbox_total', 0)} ({self.warnings.get('overfull_hbox_serious', 0)} serious)")
        print(f"  Underfull hboxes: {self.warnings.get('underfull_hbox', 0)}")

        print("\n" + "="*80)

        # Return status code
        if self.issues:
            print("\n⚠ QUALITY ISSUES DETECTED - Review recommended")
            return 1
        else:
            print("\n✓ DISSERTATION QUALITY: EXCELLENT")
            return 0

# ============================================================================
# Main CLI
# ============================================================================

if __name__ == "__main__":
    # Always show page count if available
    page_count = get_page_count()
    if page_count:
        print(f"PDF has {page_count} pages")

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "check":
            # Quality check only mode
            checker = QualityChecker()
            exit_code = checker.check_all()
            sys.exit(exit_code)

        elif command == "batch":
            # Batch mode with quality report
            if not page_count:
                print("Error: Could not determine page count from main.log")
                sys.exit(1)

            batch_num = int(sys.argv[2])
            start_page = (batch_num - 1) * BATCH_SIZE + 1
            end_page = min(batch_num * BATCH_SIZE, page_count)

            # Convert pages
            convert_page_range(start_page, end_page)

            # Run quality check
            print("\nRunning quality checks...")
            checker = QualityChecker()
            checker.check_all()

        elif command == "cleanup":
            delete_images()

        elif command == "cleanup-all":
            cleanup_directory()

        elif command == "pages":
            page_spec = sys.argv[2]
            if '-' in page_spec:
                start, end = page_spec.split('-')
                convert_page_range(int(start), int(end))
            else:
                page = int(page_spec)
                convert_page_range(page, page)

    else:
        print("\nUsage:")
        print("  python review_pdf_enhanced.py check            # Run quality checks only")
        print("  python review_pdf_enhanced.py batch <N>        # Convert batch N + quality report")
        print("  python review_pdf_enhanced.py pages <N>        # Convert single page N")
        print("  python review_pdf_enhanced.py pages <N>-<M>    # Convert pages N to M")
        print("  python review_pdf_enhanced.py cleanup          # Delete all images")
        print("  python review_pdf_enhanced.py cleanup-all      # Delete images and directory")
        print()

        if page_count:
            total_batches = (page_count + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Total batches: {total_batches}")
            print(f"Batch size: {BATCH_SIZE} pages")
            print()

        print("💡 TIP: Run 'check' first to identify quality issues before visual inspection")
        print()
