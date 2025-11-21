# Bibliography Helper Usage Guide

**Created:** November 16, 2025
**Status:** Production-ready
**Location:** `py/utils/bibliography_helper.py`

## Overview

The Bibliography Helper provides automated bibliography management for the dissertation, including:
- 33 pre-configured common citations (NFL analytics, Bayesian stats, ML, causal inference)
- BibTeX validation and duplicate detection
- Missing citation detection from .tex files
- Automated bibliography reports

## Quick Start

```python
from py.utils.bibliography_helper import BibliographyHelper

# Initialize with bibliography file
helper = BibliographyHelper('analysis/dissertation/main/references.bib')

# Suggest a common citation
entry = helper.suggest_entry('yurko2020going')
print(entry)  # Formatted BibTeX entry

# Check for missing citations in dissertation
tex_files = Path('analysis/dissertation/main').glob('*.tex')
missing = helper.check_missing_citations(list(tex_files))

# Generate full report
report = helper.generate_bibliography_report(list(tex_files))
print(report)
```

## Common Citations Library (33 entries)

### NFL Analytics (4)
- `yurko2020going` - nflfastR expected points model
- `burke2009advanced` - Advanced NFL Stats blog
- `lopez2020bigger` - NFL tracking data introduction
- `schatz2003football` - Football Outsiders DVOA

### Bayesian Statistics (6)
- `gelman2013` - Bayesian Data Analysis (3rd ed)
- `burkner2017` - brms package reference
- `carpenter2017stan` - Stan language reference
- `vehtari2017practical` - LOO-CV and WAIC
- `hoffman2014no` - NUTS sampler
- `betancourt2017conceptual` - HMC diagnostics

### Machine Learning (4)
- `chen2016xgboost` - XGBoost algorithm
- `friedman2001greedy` - Gradient boosting
- `breiman2001random` - Random forests
- `hastie2009elements` - Elements of Statistical Learning

### Calibration & Uncertainty (3)
- `gneiting2007strictly` - Proper scoring rules
- `guo2017calibration` - Neural network calibration
- `kuleshov2018accurate` - Calibrated regression

### Causal Inference (4)
- `pearl2009causality` - Causality textbook
- `imbens2015causal` - Causal inference textbook
- `angrist2009mostly` - Mostly Harmless Econometrics
- `athey2015machine` - ML for causal effects

### Reinforcement Learning (3)
- `sutton2018reinforcement` - RL textbook
- `kumar2020conservative` - Conservative Q-Learning (CQL)
- `levine2020offline` - Offline RL tutorial

### Time Series & State-Space (3)
- `durbin2012time` - State space methods
- `petris2009dynamic` - Dynamic linear models in R
- `harvey1990forecasting` - Kalman filter

### Sports Betting (3)
- `kelly1956new` - Kelly criterion (original paper)
- `thorp2006kelly` - Kelly in sports betting
- `haghighat2021review` - Player rating systems

### Statistical Testing (3)
- `diebold1995comparing` - Diebold-Mariano test
- `bergmeir2018note` - Time series cross-validation
- `tashman2000out` - Out-of-sample testing

## Core Features

### 1. Suggest BibTeX Entry

```python
# Get formatted BibTeX entry for common citation
entry = helper.suggest_entry('gelman2013')
```

**Output:**
```bibtex
@book{gelman2013,
  author = {Gelman, Andrew and Carlin, John B and Stern, Hal S and Dunson, David B and Vehtari, Aki and Rubin, Donald B},
  title = {Bayesian Data Analysis},
  edition = {Third},
  year = {2013},
  publisher = {CRC Press},
  isbn = {978-1439840955},
  note = {Standard Bayesian reference}
}
```

### 2. Validate Bibliography Entry

```python
# Check if entry has required fields
issues = helper.validate_entry('my_citation')

if issues:
    print(f"Missing fields: {', '.join(issues)}")
```

**Validation Rules:**
- **All entries:** author, title, year
- **Articles:** journal, volume
- **Books:** publisher
- **Inproceedings:** booktitle

### 3. Find Duplicate Entries

```python
# Detect duplicates (same title or author+year)
duplicates = helper.find_duplicates()

for key1, key2, reason in duplicates:
    print(f"{key1} ↔ {key2}: {reason}")
```

### 4. Check Missing Citations

```python
# Scan .tex files for \cite{} commands
tex_files = Path('analysis/dissertation/main').glob('*.tex')
missing = helper.check_missing_citations(list(tex_files))

if missing:
    print(f"Missing {len(missing)} citations:")
    for key in missing:
        print(f"  - {key}")

        # Check if it's a common citation
        suggestion = helper.suggest_entry(key)
        if suggestion:
            print(f"    SUGGESTION: Add to references.bib:\n{suggestion}")
```

### 5. Generate Bibliography Report

```python
# Comprehensive report with all checks
report = helper.generate_bibliography_report(tex_files)
```

**Sample Output:**
```
================================================================================
BIBLIOGRAPHY REPORT
================================================================================

STATISTICS:
  total_entries: 87
  articles: 52
  books: 18
  inproceedings: 14
  other: 3

MISSING CITATIONS (3):
  - yurko2020going
  - kumar2020conservative
  - gelman2013

DUPLICATE ENTRIES (1):
  - gelman2013a ↔ gelman2013b (Same title)

VALIDATION ISSUES (2):
  - smith2020analysis: Missing required fields: journal, volume
  - jones2019study: Entry not found in references.bib

SUGGESTED COMMON CITATIONS (5):
  - yurko2020going
  - kumar2020conservative
  - gelman2013
  - chen2016xgboost
  - pearl2009causality

================================================================================
```

## LaTeX Templates

### Safe Table Template

**Location:** `analysis/dissertation/templates/safe_table_template.tex`

**Features:**
- Auto-constrains to page width (no overfull hbox)
- Custom column types (MAE, ROI, Brier)
- Table notes in minipage (full width)

**Usage:**
```latex
\begin{table}[htbp]
\centering
\caption{Model Performance Comparison}
\label{tab:model-comparison}

\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{l MAE ROI Brier}
\toprule
\textbf{Model} & \textbf{MAE} & \textbf{ROI (\%)} & \textbf{Brier} \\
\midrule
XGBoost v4      & 42.53 & 6.2 & 0.185 \\
Ensemble v5     & 40.12 & 7.1 & 0.172 \\
\bottomrule
\end{tabular}
\end{adjustbox}

\vspace{2mm}
\begin{minipage}{\linewidth}
\small
\textbf{Notes:} Metrics computed on 2024 hold-out set.
\end{minipage}
\end{table}
```

### Safe TikZ Template

**Location:** `analysis/dissertation/templates/safe_tikz_template.tex`

**Features:**
- Generous node spacing (prevents overlap)
- Safe node styles (safenode, compactnode, widenode)
- Safe arrow style (thick blue, rounded corners)

**Usage:**
```latex
\begin{figure}[htbp]
\centering

\begin{tikzpicture}[
  node distance=1.8cm and 2.5cm
]

\node[safenode] (data) {Training\\Data};
\node[safenode, below=of data] (model) {Model\\Training};
\node[compactnode, right=of model] (pred) {Predict};

\draw[safearrow] (data) -- (model);
\draw[safearrow] (model) -- (pred);

\end{tikzpicture}

\caption{Prediction Pipeline}
\label{fig:pipeline}
\end{figure}
```

## Integration with Dissertation Workflow

### Step 1: Check Missing Citations

```bash
# From project root
python3 -c "
from pathlib import Path
from py.utils.bibliography_helper import BibliographyHelper

helper = BibliographyHelper('analysis/dissertation/main/references.bib')
tex_files = list(Path('analysis/dissertation/main').glob('*.tex'))
missing = helper.check_missing_citations(tex_files)

if missing:
    print(f'Missing {len(missing)} citations:')
    for key in missing:
        print(f'  {key}')
        suggestion = helper.suggest_entry(key)
        if suggestion:
            print(f'    Add to references.bib:\n{suggestion}\n')
"
```

### Step 2: Generate Full Report

```bash
python3 -c "
from pathlib import Path
from py.utils.bibliography_helper import BibliographyHelper

helper = BibliographyHelper('analysis/dissertation/main/references.bib')
tex_files = list(Path('analysis/dissertation/main').glob('*.tex'))
report = helper.generate_bibliography_report(tex_files)

print(report)

# Save to file
with open('bibliography_report.txt', 'w') as f:
    f.write(report)
"
```

### Step 3: Add Common Citations

```bash
# Add all suggested common citations to references.bib
python3 -c "
from py.utils.bibliography_helper import BibliographyHelper

helper = BibliographyHelper('analysis/dissertation/main/references.bib')

# Add specific citations
for cite_key in ['yurko2020going', 'gelman2013', 'chen2016xgboost']:
    entry = helper.suggest_entry(cite_key)
    if entry:
        print(entry)
        print()
" >> analysis/dissertation/main/references.bib
```

## Common Patterns

### Pattern 1: Pre-Compilation Check

```python
# Check before LaTeX compilation
from pathlib import Path
from py.utils.bibliography_helper import BibliographyHelper

helper = BibliographyHelper('analysis/dissertation/main/references.bib')
tex_files = list(Path('analysis/dissertation/main').glob('*.tex'))

missing = helper.check_missing_citations(tex_files)
if missing:
    print("❌ Missing citations - compilation will fail:")
    for key in missing:
        print(f"  - {key}")
    exit(1)

duplicates = helper.find_duplicates()
if duplicates:
    print("⚠️ Duplicate entries found:")
    for key1, key2, reason in duplicates:
        print(f"  - {key1} ↔ {key2}: {reason}")

print("✅ Bibliography validation passed")
```

### Pattern 2: Auto-Suggest Workflow

```python
# Suggest citations based on document content
helper = BibliographyHelper('analysis/dissertation/main/references.bib')
tex_files = list(Path('analysis/dissertation/main').glob('*.tex'))

suggestions = helper.suggest_missing_common_citations(tex_files)
if suggestions:
    print("💡 Suggested citations based on content:")
    for key in suggestions:
        print(f"\n{key}:")
        print(helper.suggest_entry(key))
```

### Pattern 3: Validation Before Commit

```bash
# Add to .git/hooks/pre-commit
python3 << 'EOF'
from pathlib import Path
from py.utils.bibliography_helper import BibliographyHelper

helper = BibliographyHelper('analysis/dissertation/main/references.bib')
tex_files = list(Path('analysis/dissertation/main').glob('*.tex'))

# Block commit if missing citations
missing = helper.check_missing_citations(tex_files)
if missing:
    print("❌ Commit blocked - missing citations:")
    for key in missing:
        print(f"  - {key}")
    exit(1)

print("✅ Bibliography check passed")
EOF
```

## Statistics

**Total Common Citations:** 33
**Categories:** 9
**Coverage:**
- NFL analytics fundamentals (4 citations)
- Bayesian methodology (6 citations)
- Machine learning algorithms (4 citations)
- Model calibration (3 citations)
- Causal inference theory (4 citations)
- Reinforcement learning (3 citations)
- Time series methods (3 citations)
- Sports betting theory (3 citations)
- Statistical validation (3 citations)

## Files Created

1. **`py/utils/bibliography_helper.py`** (850 lines)
   - Core bibliography management
   - 33 pre-configured common citations
   - Validation and reporting

2. **`analysis/dissertation/templates/safe_table_template.tex`**
   - Copy-paste table template
   - Auto-width constraint
   - Table notes pattern

3. **`analysis/dissertation/templates/safe_tikz_template.tex`**
   - Copy-paste TikZ template
   - Safe node styles
   - 10+ common patterns

4. **`docs/guides/BIBLIOGRAPHY_HELPER_USAGE.md`** (this file)
   - Comprehensive usage guide
   - Integration examples
   - Common patterns

## Next Steps

1. **Add missing citations** - Run check and add common citations to references.bib
2. **Validate entries** - Fix any validation issues found
3. **Remove duplicates** - Consolidate duplicate entries
4. **Pre-commit hook** - Add bibliography check to git workflow

## See Also

- [FILE_ORGANIZATION.md](/Users/dro/rice/nfl-analytics/FILE_ORGANIZATION.md) - File structure rules
- [docs/DATABASE_QUICK_REFERENCE.md](/Users/dro/rice/nfl-analytics/docs/database/DATABASE_QUICK_REFERENCE.md) - Database schema
- [CLAUDE.md](/Users/dro/rice/nfl-analytics/CLAUDE.md) - Project overview
