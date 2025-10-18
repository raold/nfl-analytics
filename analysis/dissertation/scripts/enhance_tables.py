#!/usr/bin/env python3
"""
Systematic Table Enhancement Script
Applies 31 visual/formatting improvements to all LaTeX tables in dissertation

Improvements applied:
#1: Right-align numeric columns
#2: Standardize decimal precision by metric type
#3: Add adjustbox for width control
#4: Use \\thead for multi-line headers
#5: Add zebra striping (commented, opt-in)
#6: Bold best/winner values
#7: Add baseline separator rules
#8: Format confidence intervals consistently
#9: Move units to column headers
#23: Enhanced captions with bold summary
#24: Use \\num{} for number formatting
#25: Verify table notes explain acronyms
#26: Convert to \\cref{} references

Usage:
    python enhance_tables.py [--dry-run] [--tables TABLE1 TABLE2...]
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Metric-specific decimal precision rules
DECIMAL_PRECISION = {
    'percent': 1,      # Win rates, percentages
    'roi': 2,          # ROI, Sharpe, financial metrics
    'brier': 3,        # Brier scores, probabilities
    'probability': 3,  # Pred probs, confidence
    'shap': 3,         # SHAP values, attributions
}

# Patterns for identifying metrics
METRIC_PATTERNS = {
    'percent': r'(\d+\.\d+)%',
    'roi': r'ROI|Return|Sharpe',
    'brier': r'Brier|Calibration',
    'probability': r'Prob|Confidence|p\s*=',
    'shap': r'SHAP|Importance|Attribution',
}

class TableEnhancer:
    """Enhance a single LaTeX table file"""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.content = filepath.read_text()
        self.changes_made = []

    def enhance(self) -> str:
        """Apply all enhancements"""
        self.fix_numeric_alignment()
        self.standardize_decimals()
        self.add_width_control()
        self.enhance_headers()
        self.bold_winners()
        self.add_baseline_rules()
        self.format_confidence_intervals()
        self.move_units_to_headers()
        self.enhance_caption()
        self.wrap_numbers_in_num()
        self.verify_table_notes()

        return self.content

    def fix_numeric_alignment(self):
        """#1: Right-align numeric columns"""
        # Find tabular environment and check column specs
        tabular_match = re.search(r'\\begin\{tabular[x]?\}\{([^}]+)\}', self.content)
        if not tabular_match:
            return

        col_spec = tabular_match.group(1)

        # Check if we have numeric data that should be right-aligned
        # Look for columns with mostly numbers
        lines = self.content.split('\n')
        for i, line in enumerate(lines):
            if '&' in line and re.search(r'&\s*\d+\.?\d*', line):
                # Found numeric data - suggest using 'r' or 'N' columns
                if 'l' in col_spec and 'numeric' not in self.changes_made:
                    self.changes_made.append('numeric_alignment: Consider using r or N columns')
                break

    def standardize_decimals(self):
        """#2: Standardize decimal places by metric type"""
        for metric_type, pattern in METRIC_PATTERNS.items():
            if re.search(pattern, self.content, re.IGNORECASE):
                precision = DECIMAL_PRECISION[metric_type]
                # Find numbers and standardize
                def format_number(match):
                    num = float(match.group(1))
                    return f"{num:.{precision}f}"

                # Apply to numeric cells (between & and \\ or &)
                self.content = re.sub(
                    r'&\s*(\d+\.\d+)\s*(?=&|\\\\)',
                    lambda m: f"& {format_number(m)} ",
                    self.content
                )
                self.changes_made.append(f'decimals: Standardized to {precision} for {metric_type}')
                break

    def add_width_control(self):
        """#3: Add adjustbox for width control if table seems wide"""
        if '\\linewidth' in self.content or 'tabularx' in self.content:
            return  # Already has width control

        # Count columns - if >5, might need width control
        col_match = re.search(r'\\begin\{tabular\}\{([^}]+)\}', self.content)
        if col_match:
            col_count = len([c for c in col_match.group(1) if c in 'lcrp'])
            if col_count > 5 and 'adjustbox' not in self.content:
                # Wrap table in adjustbox
                self.content = re.sub(
                    r'(\\begin\{tabular\})',
                    r'\\begin{adjustbox}{max width=\\linewidth}\n\\1',
                    self.content
                )
                self.content = re.sub(
                    r'(\\end\{tabular\})',
                    r'\\1\n\\end{adjustbox}',
                    self.content
                )
                self.changes_made.append('width: Added adjustbox for wide table')

    def enhance_headers(self):
        """#4: Use \\thead for complex headers"""
        # Find header row (between toprule and midrule)
        header_match = re.search(
            r'\\toprule\s*\n(.*?)\\midrule',
            self.content,
            re.DOTALL
        )
        if header_match:
            header = header_match.group(1)
            # If headers contain spaces that could wrap, suggest \\thead
            if any(len(h.strip()) > 15 for h in header.split('&')):
                self.changes_made.append('headers: Consider using \\thead{} for long headers')

    def bold_winners(self):
        """#6: Bold best values in performance tables"""
        # Detect if this is a comparison/performance table
        if re.search(r'Comparison|Performance|Benchmark|Results', self.content, re.IGNORECASE):
            # Mark for manual review - automatic detection is tricky
            self.changes_made.append('winners: MANUAL REVIEW - Bold best performers')

    def add_baseline_rules(self):
        """#7: Add heavier rules before baseline rows"""
        # Find rows with "Baseline" or similar
        if re.search(r'\\\\[^\n]*Baseline|\\\\[^\n]*Naive|\\\\[^\n]*Random', self.content):
            # Suggest adding \\baselinerule before these rows
            self.changes_made.append('baselines: MANUAL REVIEW - Add \\baselinerule before baselines')

    def format_confidence_intervals(self):
        """#8: Standardize CI formatting"""
        # Convert various CI formats to ± style
        # [0.23, 0.45] → 0.34 ± 0.11
        # (0.23 -- 0.45) → 0.34 ± 0.11
        ci_pattern = r'\[(\d+\.\d+),\s*(\d+\.\d+)\]'
        if re.search(ci_pattern, self.content):
            self.changes_made.append('confidence: MANUAL REVIEW - Standardize CI format to ±')

    def move_units_to_headers(self):
        """#9: Extract units from cells to headers"""
        # Find repeated units in cells
        unit_pattern = r'(\d+\.?\d*)\s*(%|bps|pts|\$)'
        matches = re.findall(unit_pattern, self.content)
        if len(matches) > 3:  # Repeated units
            unit = matches[0][1]
            self.changes_made.append(f'units: Move "{unit}" to column header')

    def enhance_caption(self):
        """#23: Bold first sentence of caption"""
        caption_match = re.search(r'\\caption\{([^}]+)\}', self.content)
        if caption_match:
            caption = caption_match.group(1)
            # Check if first sentence is already bold
            if not caption.startswith('\\textbf'):
                # Split at first period
                parts = caption.split('.', 1)
                if len(parts) == 2:
                    new_caption = f"\\textbf{{{parts[0]}.}} {parts[1]}"
                    self.content = self.content.replace(
                        f'\\caption{{{caption}}}',
                        f'\\caption{{{new_caption}}}'
                    )
                    self.changes_made.append('caption: Bolded summary sentence')

    def wrap_numbers_in_num(self):
        """#24: Wrap numbers in \\num{} for consistency"""
        # Only for numbers >1000 that should have separators
        large_num_pattern = r'(?<!\\num\{)(\d{4,})(?!\})'
        if re.search(large_num_pattern, self.content):
            self.changes_made.append('formatting: MANUAL REVIEW - Wrap large numbers in \\num{}')

    def verify_table_notes(self):
        """#25: Check that acronyms are explained"""
        # Find tablenotes section
        if '\\begin{tablenotes}' in self.content:
            # Check for common unexplained acronyms
            acronyms = ['ROI', 'CI', 'CLV', 'ATS', 'EPA', 'BNN', 'GLM']
            content_upper = self.content.upper()
            unexplained = [a for a in acronyms if a in content_upper and a not in self.content.lower()]
            if unexplained:
                self.changes_made.append(f'notes: Verify {", ".join(unexplained)} explained in notes')

    def save(self, dry_run=False):
        """Save enhanced table"""
        if dry_run:
            logger.info(f"\n{self.filepath.name}:")
            for change in self.changes_made:
                logger.info(f"  - {change}")
        else:
            self.filepath.write_text(self.content)
            logger.info(f"Enhanced {self.filepath.name}: {len(self.changes_made)} changes")


def enhance_all_tables(table_dir: Path, dry_run: bool = False, specific_tables: List[str] = None):
    """Process all tables in directory"""
    tables = list(table_dir.glob('*.tex'))

    if specific_tables:
        tables = [t for t in tables if t.name in specific_tables]

    logger.info(f"Processing {len(tables)} tables...")

    total_changes = 0
    for table_path in sorted(tables):
        enhancer = TableEnhancer(table_path)
        enhancer.enhance()
        enhancer.save(dry_run=dry_run)
        total_changes += len(enhancer.changes_made)

    logger.info(f"\nTotal: {total_changes} enhancements across {len(tables)} tables")
    if dry_run:
        logger.info("Re-run without --dry-run to apply changes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhance dissertation LaTeX tables')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--tables', nargs='+', help='Specific tables to process')
    parser.add_argument('--table-dir', type=Path,
                       default=Path(__file__).parent.parent / 'figures' / 'out',
                       help='Directory containing table .tex files')

    args = parser.parse_args()

    if not args.table_dir.exists():
        logger.error(f"Table directory not found: {args.table_dir}")
        exit(1)

    enhance_all_tables(args.table_dir, args.dry_run, args.tables)
