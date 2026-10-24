# Common Tasks Reference Guide

**Purpose**: Quick reference for frequently performed development tasks in the NFL Analytics project.

**Last Updated**: November 8, 2025

---

## Table of Contents

1. [Training a New Bayesian Model](#training-a-new-bayesian-model)
2. [Running a Backtest](#running-a-backtest)
3. [Updating Dissertation](#updating-dissertation)
4. [Creating a New Figure](#creating-a-new-figure)
5. [Adding a New Test](#adding-a-new-test)

---

## Training a New Bayesian Model

### Overview

Training a new Bayesian model involves three main steps: preparing data, training with R, and verifying outputs.

### Prerequisites

- PostgreSQL database running (`localhost:5544`)
- R environment with renv packages installed
- Training data already in `mart.player_game_stats` table

### Step-by-Step Instructions

**1. Verify data exists in database**

```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT COUNT(*) FROM mart.player_game_stats;"
```

This should return a positive count. If the table is empty, run data ingestion scripts first:

```bash
Rscript --vanilla R/ingestion/ingest_schedules.R
Rscript --vanilla R/ingestion/ingest_pbp.R
```

**2. Train the model**

Run the R training script:

```bash
Rscript R/train_and_save_passing_model.R
```

The script will:
- Load data from database
- Fit Bayesian hierarchical model using brms/Stan
- Save model to `models/bayesian/`
- Print diagnostic summaries

**3. Verify output**

Check that the model was saved successfully:

```bash
ls -lh models/bayesian/
```

Look for the newly created `.rds` file. Expected file sizes:
- Simple models: 5-50 MB
- Complex hierarchical models: 50-200 MB
- QB-WR chemistry models: 200+ MB

**4. (Optional) Load and inspect the model**

```r
library(brms)

# Load model
model <- readRDS("models/bayesian/your_model_v1.rds")

# View summary
summary(model)

# Extract posterior samples for manual inspection
posterior_samples <- as_draws_df(model)
head(posterior_samples)

# Generate predictions on new data
new_data <- data.frame(
  player_id = c("123", "456"),
  position = c("QB", "WR"),
  team = c("KC", "KC")
)
predictions <- posterior_predict(model, new_data)
```

### Common Model Training Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `R/train_and_save_passing_model.R` | Passing yards model | `passing_yards_hierarchical_v*.rds` |
| `R/bayesian_receiving_with_qb_chemistry.R` | Receiving yards + QB-WR dyadic | `receiving_qb_chemistry_v*.rds` |
| `R/state_space_player_skills.R` | Dynamic player skill trajectories | `player_skill_trajectories_v*.csv` |
| `R/advanced_priors_elicitation.R` | Informative priors model | `passing_informative_priors_v*.rds` |
| `R/bayesian_team_ratings_brms.R` | Team hierarchical ratings | `team_ratings_v*.rds` |

### Troubleshooting

**Problem**: `Error: connection to database failed`
- **Solution**: Check PostgreSQL is running, verify connection string

**Problem**: `Error: Table mart.player_game_stats does not exist`
- **Solution**: Run data ingestion scripts (`data/ingest_*.R`)

**Problem**: `Warning: Some Pareto k diagnostic values are too high`
- **Solution**: Model may benefit from different priors or more data; document and monitor

**Problem**: MCMC chains not converging (Rhat > 1.01)
- **Solution**: Increase iterations, adjust priors, check data for outliers

---

## Running a Backtest

### Overview

A complete backtest involves two main steps: building features and running the backtest engine. Features must be computed as-of dates to avoid look-ahead bias.

### Prerequisites

- Database with historical data (1999-2024)
- All trained models in `models/` directory
- PostgreSQL running
- Python environment with scikit-learn, XGBoost, etc.

### Step-by-Step Instructions

**1. Build features**

Generate as-of features for your backtest period:

```bash
python py/features/asof_features.py \
  --output analysis/features/asof_team_features.csv \
  --season-start 1999 \
  --season-end 2024 \
  --validate
```

**Flags**:
- `--output`: Where to save feature CSV
- `--season-start`: First season (inclusive)
- `--season-end`: Last season (inclusive)
- `--validate`: Run data quality checks (recommended)

**Output**: CSV file with as-of features for all games in period, with strict temporal boundaries to prevent leakage.

**2. Run backtest**

Run the backtest harness:

```bash
python py/backtest/baseline_glm.py \
  --start-season 2003 \
  --end-season 2024 \
  --output-csv analysis/results/glm_baseline_metrics.csv \
  --tex analysis/dissertation/figures/out/glm_baseline_table.tex
```

**Flags**:
- `--start-season`: First season for backtest
- `--end-season`: Last season for backtest
- `--output-csv`: Path to save metrics CSV
- `--tex`: Path to save LaTeX table (optional)

**3. Analyze results**

Review the output CSV:

```bash
head -20 analysis/results/glm_baseline_metrics.csv
```

Expected columns: `week`, `game_date`, `prediction`, `actual`, `error`, `roi`, `win`

Generate summary statistics:

```python
import pandas as pd

results = pd.read_csv('analysis/results/glm_baseline_metrics.csv')

print(f"Accuracy: {results['win'].mean():.1%}")
print(f"ROI: {results['roi'].mean():.2%}")
print(f"Sharpe: {results['roi'].mean() / results['roi'].std():.2f}")
print(f"Max drawdown: {results['roi'].cumsum().min():.2%}")
```

### Available Backtest Engines

| Script | Purpose | Notes |
|--------|---------|-------|
| `py/backtest/baseline_glm.py` | Generalized Linear Model baseline | Fast, reference implementation |
| `py/backtest/bayesian_props_backtest.py` | Full Bayesian hierarchical | Produces calibration metrics |
| `py/backtest/enhanced_ensemble_v3.py` | 4-way ensemble (Bayesian + XGBoost + BNN + State-Space) | Production model |
| `py/backtest/harness.py` | Generic backtest framework | Extensible for custom models |

### Key Considerations

**Temporal Leakage**: Features are computed as-of each game date with strict boundaries:
- Use `.shift(1)` for features from previous weeks
- Never include post-game data (EPA, outcomes)
- Test with `tests/integration/test_temporal_leakage.py`

**Odds/Lines**: Use closing line (`spread_close`, `total_close`), not opening lines

**Sample Size**: Typical backtest periods:
- Full history: 1999-2024 (6,991 games)
- Recent seasons: 2018-2024 (1,232 games)
- Validation: 2022-2024 (462 games)

---

## Updating Dissertation

### Overview

The dissertation is compiled from LaTeX source files. Changes to content require editing LaTeX files and recompiling the PDF.

### File Structure

```
analysis/dissertation/
├── main/
│   ├── main.tex                      # Main document (includes all chapters)
│   └── main.pdf                      # Compiled output (do not edit)
├── chapter_1_introduction/
│   └── chapter_1_introduction.tex    # Chapter content
├── chapter_2_.../
│   └── chapter_*.tex                 # More chapters
├── appendix/
│   ├── appendix_*.tex                # Appendix chapters
│   └── consolidated_appendix.tex     # Index of appendices
├── style/
│   ├── dissertation_preamble.tex     # LaTeX preamble (packages, commands)
│   └── tikz_diagram_style.tex        # Standardized TikZ styling
├── figures/
│   ├── out/                          # Generated figures and tables
│   └── R/                            # R scripts to generate figures
└── references.bib                    # Bibliography database
```

### Prerequisites

- LaTeX distribution (TeX Live, MacTeX, or MiKTeX)
- `pdflatex` and `bibtex` command-line tools
- All chapter files saved

### Step-by-Step Instructions

**1. Edit chapter content**

Edit the relevant chapter file in `chapter_*/` directory:

```bash
# Example: Edit chapter on Bayesian models
nano analysis/dissertation/chapter_5_bayesian_models/chapter_5_bayesian_models.tex
```

**2. Navigate to main directory**

```bash
cd analysis/dissertation/main
```

**3. Compile the dissertation**

Run the complete compilation pipeline (must run pdflatex + bibtex + pdflatex twice):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**First compilation**: Generates auxiliary files and identifies references
**bibtex**: Processes bibliography database (`references.bib`)
**Second pdflatex**: Inserts bibliography into document
**Third pdflatex**: Updates cross-references and page numbers

**4. Check for errors and warnings**

Examine the log file:

```bash
grep -i "warning\|error" main.log
```

Common issues:
- `Undefined control sequence`: Typo in LaTeX command (check preamble)
- `Undefined reference`: Citation not in .bib file
- `Overfull hbox`: Line too long (consider rewording)

**5. View the compiled PDF**

```bash
# macOS
open main.pdf

# Linux
evince main.pdf

# or use any PDF viewer
```

### Common Editing Tasks

**Add a new paragraph**:
```latex
\section{My New Section}

This is my new content.
```

**Add a figure**:
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{path/to/figure.pdf}
  \caption{Figure caption here.}
  \label{fig:my_figure}
\end{figure}

As shown in Figure \ref{fig:my_figure}...
```

**Add a citation**:
```latex
\cite{author2020key} or \citep{author2020key}
```

**Add a table**:
```latex
\begin{table}[h]
  \centering
  \begin{tabular}{lcc}
    \hline
    Column 1 & Column 2 & Column 3 \\
    \hline
    Data 1 & Data 2 & Data 3 \\
    \hline
  \end{tabular}
  \caption{Table caption.}
  \label{tab:my_table}
\end{table}
```

**Add a TikZ diagram** (uses standardized style):
```latex
\input{../figures/out/my_diagram.tex}

% Or inline:
\begin{tikzpicture}[diagram style]
  \node[action box] (A) {Action};
  \node[process box, right=of A] (B) {Process};
  \draw[arrow] (A) -- (B);
\end{tikzpicture}
```

### Compilation Troubleshooting

**Problem**: `pdflatex: command not found`
- **Solution**: Install LaTeX distribution (TeX Live on Linux/Mac, MiKTeX on Windows)

**Problem**: `I can't find file 'chapter_*.tex'`
- **Solution**: Make sure you're in `main/` directory and chapter files exist

**Problem**: Bibliography doesn't appear
- **Solution**: Run full pipeline (pdflatex → bibtex → pdflatex → pdflatex)

**Problem**: `Undefined references` remain after compilation
- **Solution**: Add citations to `references.bib`, re-run bibtex, then pdflatex twice

---

## Creating a New Figure

### Overview

Figures are typically generated with R (using ggplot2) and saved as PDF files included in dissertation.

### File Structure

```
analysis/dissertation/figures/
├── R/
│   ├── figure_1_model_architecture.R    # R script to generate figure
│   ├── figure_2_results_comparison.R
│   └── ...
└── out/
    ├── figure_1_model_architecture.pdf  # Generated output
    ├── figure_2_results_comparison.pdf
    └── my_table.tex                     # LaTeX tables
```

### Step-by-Step Instructions

**1. Create R script**

Create a new R script in `analysis/dissertation/figures/R/`:

```r
# analysis/dissertation/figures/R/my_figure.R
# Purpose: Visualize model performance comparison
# Author: Your Name
# Date: 2025-11-08

library(ggplot2)
library(dplyr)
library(RPostgres)

# Connect to database
con <- DBI::dbConnect(RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

# Query data
data <- DBI::dbGetQuery(con, "
  SELECT
    model_type,
    season,
    mae,
    rmse,
    calibration
  FROM ml.model_metrics
  WHERE model_type IN ('Bayesian', 'XGBoost', 'BNN')
  ORDER BY season, model_type
")

# Close connection
DBI::dbDisconnect(con)

# Create visualization
p <- ggplot(data, aes(x = season, y = mae, color = model_type, linetype = model_type)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Bayesian" = "#1f77b4", "XGBoost" = "#ff7f0e", "BNN" = "#2ca02c")) +
  scale_linetype_manual(values = c("Bayesian" = "solid", "XGBoost" = "dashed", "BNN" = "dotted")) +
  labs(
    title = "Model Performance Comparison (2018-2024)",
    x = "Season",
    y = "Mean Absolute Error (yards)",
    color = "Model Type",
    linetype = "Model Type"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Save as PDF
ggsave("../out/my_figure.pdf", p, width = 6, height = 4, dpi = 300)

cat("Figure saved to: analysis/dissertation/figures/out/my_figure.pdf\n")
```

**2. Run the script**

```bash
Rscript analysis/dissertation/figures/R/my_figure.R
```

**3. Verify output**

Check that the PDF was created:

```bash
ls -lh analysis/dissertation/figures/out/my_figure.pdf
```

**4. Include in dissertation**

Add to relevant chapter LaTeX file:

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{../figures/out/my_figure.pdf}
  \caption{Description of figure content and key insights.}
  \label{fig:my_figure}
\end{figure}

Figure~\ref{fig:my_figure} shows...
```

**5. Recompile dissertation**

```bash
cd analysis/dissertation/main
pdflatex main.tex
pdflatex main.tex
```

### Best Practices for Dissertation Figures

**Size and Spacing**:
- Typical width: `0.8\textwidth` (leaves margins)
- Wide figures: `1.0\textwidth`
- Multi-panel: use `subcaption` package

**Resolution**:
- Use `dpi = 300` in `ggsave()` for print quality
- Vector formats (PDF) preferred over raster (PNG)

**Styling**:
- Use `theme_minimal()` for clean appearance
- Consistent color palette across all figures
- Font sizes: readable at figure width (check with PDF)

**Captions**:
- Descriptive: Should be understandable without text
- Start with main finding: "Figure X shows that..."
- Reference in text: "As shown in Figure~\ref{fig:my_figure}"

---

## Adding a New Test

### Overview

The project uses pytest for unit and integration testing. Tests are organized by type in the `tests/` directory.

### File Structure

```
tests/
├── unit/                              # Fast, no dependencies
│   ├── test_features.py
│   ├── test_bayesian_models.py
│   └── ...
├── integration/                       # Database, fixtures required
│   ├── test_ensemble.py
│   ├── test_temporal_leakage.py
│   └── ...
├── e2e/                               # Full pipeline tests
│   └── test_backtest_complete.py
├── conftest.py                        # Shared fixtures
├── pytest.ini                         # Configuration
└── README.md                          # Testing guide
```

### Prerequisites

- Python environment with pytest installed
- For integration tests: PostgreSQL running

### Step-by-Step Instructions

**1. Create test file**

Create a new test file in appropriate directory (`tests/unit/`, `tests/integration/`, or `tests/e2e/`):

```python
# tests/unit/test_my_module.py
"""
Test module: Test suite for my_module.

Covers:
- Basic functionality
- Edge cases
- Error handling
"""

import pytest
from py.my_module import my_function, MyClass


class TestMyFunction:
    """Tests for my_function."""

    def test_my_function_basic(self):
        """Test basic functionality of my_function."""
        # Arrange
        input_data = {"key": "value"}
        expected_output = 42

        # Act
        result = my_function(input_data)

        # Assert
        assert result == expected_output

    def test_my_function_edge_case_empty_input(self):
        """Test edge case: empty input."""
        result = my_function({})
        assert result == 0  # or expected default

    def test_my_function_invalid_input(self):
        """Test error handling with invalid input."""
        with pytest.raises(ValueError):
            my_function("invalid_string")

    def test_my_function_type_validation(self):
        """Test that function validates input types."""
        with pytest.raises(TypeError):
            my_function(None)


class TestMyClass:
    """Tests for MyClass."""

    @pytest.fixture
    def instance(self):
        """Create a test instance."""
        return MyClass(param1="test", param2=42)

    def test_initialization(self, instance):
        """Test class initialization."""
        assert instance.param1 == "test"
        assert instance.param2 == 42

    def test_method_basic(self, instance):
        """Test basic method functionality."""
        result = instance.my_method()
        assert result is not None

    def test_method_with_parameters(self, instance):
        """Test method with parameters."""
        result = instance.my_method(arg1=10, arg2=20)
        assert result == 30  # or expected output

    @pytest.mark.parametrize("input_val,expected", [
        (0, "zero"),
        (1, "one"),
        (-1, "negative"),
    ])
    def test_parameterized(self, instance, input_val, expected):
        """Test with multiple parameter combinations."""
        result = instance.categorize(input_val)
        assert result == expected
```

**2. Write test cases**

Follow the Arrange-Act-Assert (AAA) pattern:
- **Arrange**: Set up test data and fixtures
- **Act**: Call the function/method
- **Assert**: Verify the results

**3. Run tests**

Run a single test file:

```bash
pytest tests/unit/test_my_module.py -v
```

Run all unit tests:

```bash
pytest tests/unit -v
```

Run with coverage:

```bash
pytest tests/unit --cov=py --cov-report=html
open htmlcov/index.html
```

**4. Verify test passes**

Output should show:
```
tests/unit/test_my_module.py::TestMyFunction::test_my_function_basic PASSED
tests/unit/test_my_module.py::TestMyFunction::test_my_function_edge_case_empty_input PASSED
...
```

### Common Test Patterns

**Testing with fixtures** (shared setup):

```python
@pytest.fixture
def sample_data():
    """Fixture: Create sample DataFrame."""
    return pd.DataFrame({
        'player_id': [1, 2, 3],
        'yards': [100, 150, 200]
    })

def test_feature_with_fixture(sample_data):
    """Test function using fixture."""
    result = compute_feature(sample_data)
    assert len(result) == 3
```

**Mocking external dependencies**:

```python
from unittest.mock import patch, MagicMock

@patch('py.module.external_function')
def test_with_mock(mock_external):
    """Test function that calls external code."""
    mock_external.return_value = 42
    result = my_function()
    assert result == 42
    mock_external.assert_called_once()
```

**Testing exceptions**:

```python
def test_raises_error():
    """Test that function raises appropriate error."""
    with pytest.raises(ValueError, match="Invalid input"):
        my_function("bad_input")
```

**Parametrized tests** (multiple inputs):

```python
@pytest.mark.parametrize("input_val,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiple_cases(input_val, expected):
    """Test function with multiple input combinations."""
    assert multiply_by_two(input_val) == expected
```

### Integration Tests (with Database)

```python
# tests/integration/test_with_database.py
"""Integration tests requiring database connection."""

import pytest
from sqlalchemy import create_engine

@pytest.fixture(scope="session")
def db_engine():
    """Create database connection for tests."""
    engine = create_engine("postgresql://dro:sicillionbillions@localhost:5544/devdb01")
    yield engine
    engine.dispose()

def test_feature_engineering_with_db(db_engine):
    """Test feature engineering against real database."""
    with db_engine.connect() as conn:
        result = conn.execute("SELECT COUNT(*) FROM mart.player_game_stats")
        count = result.scalar()
        assert count > 0
```

### Running All Tests

```bash
# Unit tests (fast)
pytest tests/unit

# Integration tests (requires database)
pytest tests/integration

# All tests with coverage
pytest tests/ --cov=py --cov-report=html

# Tests with specific marker
pytest tests/ -m unit
pytest tests/ -m integration
```

### Test Configuration

Configuration in `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    unit: Unit tests (no external dependencies)
    integration: Integration tests (require database)
    slow: Tests that take more than 1 second
```

### Continuous Integration

Tests run automatically via GitHub Actions on:
- Every commit to feature branches
- Pull requests
- Merges to main branch

Check results in `.github/workflows/test.yml`

---

## Additional Resources

- **CLAUDE.md**: Complete project documentation
- **PROJECT_STATUS.md**: Current status and recent work
- **tests/README.md**: Detailed testing guide
- **docs/database/DATABASE_SCHEMA_AUDIT.md**: Database schema reference
- **PHASE_PLAN.md**: Development roadmap

---

**Version**: 1.0
**Last Updated**: November 8, 2025
**Maintained by**: Claude Code
