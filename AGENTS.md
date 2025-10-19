# AGENTS.md - Repository Guidelines for AI Assistants

**Last Updated:** October 19, 2025
**Purpose:** Guidelines, patterns, and best practices for AI agents working on this codebase
**Related:** See CLAUDE.md for comprehensive project documentation

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Code Style & Conventions](#code-style--conventions)
3. [Common Patterns](#common-patterns)
4. [Anti-Patterns (Avoid)](#anti-patterns-avoid)
5. [Git & Version Control](#git--version-control)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation Standards](#documentation-standards)
8. [LaTeX & Dissertation](#latex--dissertation)
9. [Database Operations](#database-operations)
10. [Model Training](#model-training)

---

## Core Principles

### 1. Reproducibility First

Every analysis, model, and experiment must be **fully reproducible** from code.

**Good:**
```python
# py/analysis/experiment_123.py
import numpy as np
np.random.seed(42)  # Fixed seed for reproducibility

# Document all parameters
CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'seed': 42
}

# Save config with model
joblib.dump({'model': model, 'config': CONFIG}, 'models/experiment_123.pkl')
```

**Bad:**
```python
# No seed, magic numbers, config not saved
model.fit(X, y)  # What were the hyperparameters?
joblib.dump(model, 'model.pkl')
```

### 2. Fail Fast, Fail Loudly

Validate assumptions explicitly. Don't let errors propagate silently.

**Good:**
```python
def train_model(data, labels):
    """Train model with explicit validation."""
    assert len(data) == len(labels), f"Data/label mismatch: {len(data)} vs {len(labels)}"
    assert len(data) > 100, f"Insufficient data: {len(data)} samples (need >100)"

    if data.isnull().any().any():
        raise ValueError(f"Found {data.isnull().sum().sum()} null values in training data")

    # Training code...
```

**Bad:**
```python
def train_model(data, labels):
    """Train model (assumptions implicit)."""
    # What if data/labels are mismatched? What if nulls exist?
    model.fit(data, labels)
```

### 3. Document Why, Not What

Code should be self-documenting for the "what". Comments explain "why".

**Good:**
```python
# Use L2 regularization to prevent overfitting on small sample (n=147 shock events)
# Alpha=0.1 chosen via 5-fold CV (see experiments/regularization_sweep.json)
model = Ridge(alpha=0.1)
```

**Bad:**
```python
# Create ridge regression model with alpha 0.1
model = Ridge(alpha=0.1)
```

### 4. Prefer Explicit Over Clever

Clarity beats brevity. Future readers (including you) will thank you.

**Good:**
```python
def calculate_kelly_fraction(win_prob, odds, max_fraction=0.25):
    """
    Calculate Kelly criterion bet fraction.

    Args:
        win_prob: Probability of winning (0-1)
        odds: Decimal odds (e.g., 1.91 for -110)
        max_fraction: Maximum bet size as fraction of bankroll

    Returns:
        Kelly fraction, capped at max_fraction
    """
    b = odds - 1  # Net profit per unit bet
    q = 1 - win_prob
    kelly = (win_prob * b - q) / b

    # Cap at max_fraction to limit risk
    return min(kelly, max_fraction)
```

**Bad:**
```python
def kelly(p, o, m=0.25):
    return min((p*(o-1)-(1-p))/(o-1), m)  # What does this do?
```

---

## Code Style & Conventions

### Python

**Formatting:**
- **Black** formatter (enforced by pre-commit hooks)
- Maximum line length: **100 characters**
- 4 spaces for indentation (no tabs)

**Type Hints:**
```python
from typing import List, Dict, Optional, Tuple
import pandas as pd

def process_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    normalize: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Process feature columns with optional normalization.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        normalize: Whether to normalize features (default True)

    Returns:
        Tuple of (processed DataFrame, normalization stats dict)
    """
    # Implementation...
    return processed_df, stats
```

**Imports:**
```python
# Standard library first
import os
import sys
from pathlib import Path

# Third-party packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from py.features.asof_features import build_asof_features
from py.models.bayesian_neural_network import BayesianNeuralNetwork
```

**Naming:**
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

```python
# Good naming
MAX_ITERATIONS = 1000
player_stats_df = load_player_stats()

class BayesianEnsemble:
    def __init__(self):
        self._models = []  # Private attribute

    def predict(self, X):
        """Public prediction method."""
        return self._aggregate_predictions(X)

    def _aggregate_predictions(self, X):
        """Private helper method."""
        # Implementation...
```

### R

**Style:**
- Follow [tidyverse style guide](https://style.tidyverse.org/)
- Use `<-` for assignment (not `=`)
- Pipe operator: `%>%` (magrittr) or `|>` (base R 4.1+)

**Formatting:**
```r
# Good R style
player_stats <- dbGetQuery(con, "
  SELECT
    player_id,
    season,
    AVG(stat_yards) AS avg_yards,
    STDDEV(stat_yards) AS sd_yards
  FROM mart.player_game_stats
  WHERE position = 'RB'
  GROUP BY player_id, season
") %>%
  filter(avg_yards > 0) %>%
  mutate(cv = sd_yards / avg_yards) %>%
  arrange(desc(cv))

# Load libraries explicitly at top
library(dplyr)
library(ggplot2)
library(brms)
library(DBI)
```

**Avoid:**
```r
# Bad: Use of = for assignment
x = 5

# Bad: Implicit library loading
dplyr::filter(...)  # Load library explicitly instead

# Bad: Unclear variable names
x <- data %>% filter(y > 5) %>% select(z)
```

### SQL

**Style:**
- UPPERCASE for keywords
- lowercase for table/column names
- Indent subqueries

```sql
-- Good SQL style
SELECT
    g.game_id,
    g.home_team,
    g.away_team,
    gs.home_score,
    gs.away_score,
    w.temperature,
    w.wind_speed
FROM public.games g
INNER JOIN mart.game_summary gs
    ON g.game_id = gs.game_id
LEFT JOIN public.weather w
    ON g.game_id = w.game_id
WHERE g.season >= 2020
    AND g.season <= 2024
ORDER BY g.game_date DESC;
```

---

## Common Patterns

### 1. Database Connections

**Always use context managers** for database connections:

```python
import psycopg2
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """Get database connection with automatic cleanup."""
    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        dbname="devdb01",
        user="dro",
        password="sicillionbillions"  # Use env vars in production
    )
    try:
        yield conn
    finally:
        conn.close()

# Usage
with get_db_connection() as conn:
    df = pd.read_sql("SELECT * FROM mart.game_summary", conn)
# Connection automatically closed
```

**R Pattern:**
```r
library(DBI)
library(RPostgres)

# Open connection
con <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

# Always disconnect when done
tryCatch({
  data <- dbGetQuery(con, "SELECT ...")
  # Process data...
}, finally = {
  dbDisconnect(con)
})
```

### 2. Model Persistence

**Save models with metadata:**

```python
import joblib
import json
from datetime import datetime

def save_model(model, filepath, metadata=None):
    """Save model with training metadata."""
    if metadata is None:
        metadata = {}

    # Add timestamp
    metadata['saved_at'] = datetime.now().isoformat()

    # Save model and metadata
    artifact = {
        'model': model,
        'metadata': metadata
    }
    joblib.dump(artifact, filepath)

    # Also save metadata as JSON for easy inspection
    json_path = filepath.replace('.pkl', '_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

# Usage
save_model(
    bnn_model,
    'models/bayesian/bnn_passing_v1.pkl',
    metadata={
        'training_samples': len(train_data),
        'test_mae': 58.70,
        'test_rmse': 73.45,
        'features': feature_cols,
        'hyperparameters': {'learning_rate': 0.001, 'epochs': 2000}
    }
)
```

### 3. Feature Engineering

**Use consistent naming and validation:**

```python
def build_player_features(player_game_df, lookback_weeks=4):
    """
    Build player features with rolling averages.

    Args:
        player_game_df: DataFrame with player-game records
        lookback_weeks: Number of weeks for rolling average

    Returns:
        DataFrame with engineered features
    """
    # Validate inputs
    required_cols = ['player_id', 'week', 'stat_yards', 'stat_carries']
    missing = set(required_cols) - set(player_game_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by player and week (critical for rolling calculations)
    df = player_game_df.sort_values(['player_id', 'week'])

    # Rolling features (group by player to avoid leakage across players)
    df['yards_per_carry'] = df['stat_yards'] / df['stat_carries'].clip(lower=1)
    df['rolling_avg_yards'] = (
        df.groupby('player_id')['stat_yards']
        .rolling(lookback_weeks, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )

    # Document feature names
    feature_cols = ['yards_per_carry', 'rolling_avg_yards']
    df.attrs['feature_cols'] = feature_cols

    return df
```

### 4. Backtesting

**Walk-forward validation to prevent look-ahead bias:**

```python
def walk_forward_backtest(
    data,
    train_func,
    predict_func,
    train_years,
    test_year
):
    """
    Walk-forward backtest with expanding training window.

    Args:
        data: Full dataset
        train_func: Function to train model (takes train_data, returns model)
        predict_func: Function to predict (takes model, test_data, returns predictions)
        train_years: List of years to use for initial training
        test_year: Year to test on

    Returns:
        Dict with predictions and metrics
    """
    # Split data
    train_data = data[data['season'].isin(train_years)]
    test_data = data[data['season'] == test_year]

    # Validate no leakage
    assert train_data['season'].max() < test_year, "Training data leaks into test period!"

    # Train and predict
    model = train_func(train_data)
    predictions = predict_func(model, test_data)

    # Calculate metrics
    mae = np.mean(np.abs(predictions - test_data['actual']))

    return {
        'predictions': predictions,
        'actual': test_data['actual'].values,
        'mae': mae,
        'train_years': train_years,
        'test_year': test_year
    }
```

---

## Anti-Patterns (Avoid)

### ❌ Hardcoded Paths

**Bad:**
```python
# Breaks on different machines
df = pd.read_csv('/Users/dro/Desktop/data.csv')
```

**Good:**
```python
from pathlib import Path

# Relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
data_path = PROJECT_ROOT / 'data' / 'processed' / 'data.csv'
df = pd.read_csv(data_path)
```

### ❌ Ignoring Null Values

**Bad:**
```python
# Silently propagates nulls
df['ratio'] = df['numerator'] / df['denominator']
model.fit(df[features], df['target'])  # NaNs will cause issues!
```

**Good:**
```python
# Explicit null handling
null_count = df[features].isnull().sum().sum()
if null_count > 0:
    raise ValueError(f"Found {null_count} null values. Handle before training.")

df['ratio'] = df['numerator'] / df['denominator'].replace(0, np.nan)
df = df.dropna()  # Explicit drop
```

### ❌ Modifying DataFrames In-Place Without Copy

**Bad:**
```python
def normalize(df):
    """Modifies original DataFrame - dangerous!"""
    df['feature'] = (df['feature'] - df['feature'].mean()) / df['feature'].std()
    return df

# Original df is modified!
normalized = normalize(df)
```

**Good:**
```python
def normalize(df):
    """Returns new DataFrame, original unchanged."""
    df_copy = df.copy()
    df_copy['feature'] = (df_copy['feature'] - df_copy['feature'].mean()) / df_copy['feature'].std()
    return df_copy

# Original df unchanged
normalized = normalize(df)
```

### ❌ Overly Broad Exception Handling

**Bad:**
```python
try:
    result = risky_operation()
except:
    result = None  # What went wrong? Silent failure!
```

**Good:**
```python
try:
    result = risky_operation()
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    result = default_value
```

---

## Git & Version Control

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature/bug change)
- `test`: Adding or updating tests
- `chore`: Maintenance (dependencies, config, etc.)

**Examples:**

```
feat(causal): add synthetic control method for player injuries

Implements Abadie et al. (2010) synthetic control for estimating
counterfactual player performance after injuries. Uses constrained
optimization to create synthetic player from control pool.

Closes #45
```

```
fix(bnn): correct prior specification for hierarchical effects

Prior sigma was too wide (10.0) causing under-dispersion. Reduced to
2.0 based on empirical Bayes analysis (see experiments/prior_sensitivity.json).

Fixes #67
```

```
docs(dissertation): standardize TikZ figure styling

Created analysis/dissertation/style/tikz_diagram_style.tex with
consistent color scheme (accent blue RGB 31,119,180) and rounded
corners for all flowchart diagrams.
```

### Branching Strategy

**main branch:** `main` (always deployable)

**Feature branches:** `feature/<descriptive-name>`
```bash
git checkout -b feature/add-conformal-prediction
# Make changes, commit
git push origin feature/add-conformal-prediction
gh pr create
```

**Hot fixes:** `hotfix/<issue>`
```bash
git checkout -b hotfix/fix-database-connection
# Fix, commit
git push origin hotfix/fix-database-connection
gh pr create
```

### Pre-commit Hooks

Always run pre-commit hooks before committing:

```bash
# Install hooks (one-time)
pre-commit install

# Hooks run automatically on git commit
git commit -m "feat: add new feature"

# Or run manually
pre-commit run --all-files
```

**Hooks enforce:**
- Black formatting (Python)
- Flake8 linting (Python)
- mypy type checking (Python)
- Trailing whitespace removal
- YAML validation
- Large file prevention

---

## Testing Guidelines

### Test Structure

Follow pytest conventions:

```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/           # Tests with database/external services
│   ├── test_database.py
│   └── test_pipeline.py
└── e2e/                   # End-to-end workflow tests
    └── test_backtest_pipeline.py
```

### Test Naming

```python
# tests/unit/test_features.py
import pytest
from py.features.asof_features import build_asof_features

def test_asof_features_basic():
    """Test basic as-of feature construction."""
    # Arrange
    sample_data = create_sample_data()

    # Act
    result = build_asof_features(sample_data, lookback=4)

    # Assert
    assert len(result) == len(sample_data)
    assert 'rolling_avg_yards' in result.columns

def test_asof_features_handles_null_input():
    """Test that null inputs raise ValueError."""
    with pytest.raises(ValueError, match="Missing required columns"):
        build_asof_features(pd.DataFrame())

def test_asof_features_no_leakage():
    """Test that features don't leak future information."""
    data = create_temporal_data()
    result = build_asof_features(data, lookback=4)

    # Rolling average at week 5 should only use weeks 1-4
    week5_rolling = result[result['week'] == 5]['rolling_avg_yards'].iloc[0]
    expected = data[data['week'] < 5]['stat_yards'].mean()

    assert abs(week5_rolling - expected) < 0.01
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_features.py

# Run specific test
pytest tests/unit/test_features.py::test_asof_features_basic

# Run with coverage
pytest --cov=py --cov-report=html
open htmlcov/index.html

# Run only fast unit tests
pytest -m unit

# Run integration tests (requires database)
pytest -m integration
```

---

## Documentation Standards

### Docstrings

Use **Google-style** docstrings:

```python
def estimate_treatment_effect(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounders: List[str],
    method: str = 'did'
) -> Dict[str, float]:
    """
    Estimate causal treatment effect with confounder adjustment.

    Uses difference-in-differences (DiD) or synthetic control to estimate
    the causal effect of a treatment on an outcome, adjusting for confounders.

    Args:
        df: Panel data with treatment, outcome, and confounders
        treatment_col: Column name for binary treatment indicator
        outcome_col: Column name for outcome variable
        confounders: List of confounder column names
        method: Estimation method ('did' or 'synthetic_control')

    Returns:
        Dictionary with:
            - 'estimate': Point estimate of treatment effect
            - 'ci_lower': Lower 95% confidence interval
            - 'ci_upper': Upper 95% confidence interval
            - 'p_value': Statistical significance p-value

    Raises:
        ValueError: If method not in ['did', 'synthetic_control']
        ValueError: If required columns missing from df

    Example:
        >>> effect = estimate_treatment_effect(
        ...     panel_df,
        ...     treatment_col='injury',
        ...     outcome_col='yards',
        ...     confounders=['ability', 'team_quality'],
        ...     method='did'
        ... )
        >>> print(f"Effect: {effect['estimate']:.2f} yards")
        Effect: -15.32 yards

    See Also:
        - py.causal.diff_in_diff.DifferenceInDifferences
        - py.causal.synthetic_control.SyntheticControl
    """
    # Implementation...
```

### README Files

Every major directory should have a README:

```markdown
# py/causal/

Causal inference framework for NFL analytics.

## Purpose

Move beyond correlation to understand true causal effects of shock events
(injuries, coaching changes, trades) on player/team performance.

## Modules

- `panel_constructor.py`: Build longitudinal panel datasets
- `treatment_definitions.py`: Define treatment events
- `diff_in_diff.py`: Difference-in-differences estimators
- `synthetic_control.py`: Synthetic control methods

## Quick Start

```python
from causal.diff_in_diff import DifferenceInDifferences

did = DifferenceInDifferences()
did.fit(panel_df, outcome_col='yards', treatment_col='injury', ...)
print(did.treatment_effect_)
```

## Documentation

See `docs/CAUSAL_INFERENCE_FRAMEWORK.md` for detailed usage.
```

---

## LaTeX & Dissertation

### File Organization

```
analysis/dissertation/
├── main/
│   ├── main.tex              # Main document (includes only)
│   └── references.bib        # Bibliography
├── chapter_*/                # Individual chapters
│   └── chapter_*.tex
├── appendix/                 # Appendices
│   └── appendix_consolidated.tex
├── figures/                  # Figure generation
│   ├── R/                    # R figure scripts
│   └── out/                  # Generated figures (.tex, .pdf)
└── style/                    # LaTeX styles
    ├── tikz_diagram_style.tex  # Standardized TikZ style
    └── dissertation_preamble.tex
```

### TikZ Figures

**Always use standardized style:**

```latex
% In your figure file
\input{../style/tikz_diagram_style.tex}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=1.5cm
]
  % Define accent color
  \definecolor{accent}{RGB}{31,119,180}

  % Use standardized styles
  \node[flowbox] (node1) {First Step};
  \node[flowbox, below=of node1] (node2) {Second Step};

  \draw[arrow] (node1) -- (node2);
\end{tikzpicture}
\caption{My standardized figure.}
\label{fig:my-figure}
\end{figure}
```

### Compilation

```bash
cd analysis/dissertation/main

# Full compilation cycle
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Check for errors/warnings
grep -i "warning\|error" main.log | head -20
```

### Cross-References

**Use `\label` and `\ref` (or `\Cref` with cleveref):**

```latex
% Define label
\section{Introduction}
\label{sec:introduction}

% Reference later
As discussed in Section~\ref{sec:introduction}...

% Or with cleveref (automatic "Section" prefix)
As discussed in \Cref{sec:introduction}...
```

**Label conventions:**
- Sections: `\label{sec:name}`
- Figures: `\label{fig:name}`
- Tables: `\label{tab:name}`
- Equations: `\label{eq:name}`

---

## Database Operations

### Queries

**Always use parameterized queries to prevent SQL injection:**

```python
# Good: Parameterized
player_id = 'some_id'
query = "SELECT * FROM mart.player_game_stats WHERE player_id = %s"
df = pd.read_sql(query, conn, params=(player_id,))

# Bad: String interpolation (SQL injection risk!)
query = f"SELECT * FROM mart.player_game_stats WHERE player_id = '{player_id}'"
df = pd.read_sql(query, conn)
```

### Materialized View Refresh

```bash
# Refresh specific view
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "REFRESH MATERIALIZED VIEW mart.game_summary;"

# Refresh all enhanced features
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT mart.refresh_game_features();"
```

### Migrations

**Always create migration scripts for schema changes:**

```sql
-- db/migrations/024_add_player_prop_predictions.sql

-- Add new table for prop predictions
CREATE TABLE IF NOT EXISTS mart.player_prop_predictions (
    prediction_id SERIAL PRIMARY KEY,
    player_id VARCHAR(255) NOT NULL,
    game_id INTEGER NOT NULL REFERENCES public.games(game_id),
    stat_type VARCHAR(50) NOT NULL,  -- 'passing_yards', 'rushing_yards', etc.
    predicted_value FLOAT NOT NULL,
    confidence_lower FLOAT NOT NULL,
    confidence_upper FLOAT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(player_id, game_id, stat_type, model_version)
);

CREATE INDEX idx_prop_predictions_player ON mart.player_prop_predictions(player_id);
CREATE INDEX idx_prop_predictions_game ON mart.player_prop_predictions(game_id);
```

---

## Model Training

### Bayesian Models (R/brms)

**Template:**

```r
# R/train_my_bayesian_model.R
library(brms)
library(cmdstanr)
library(DBI)
library(RPostgres)

# 1. Load data
con <- dbConnect(Postgres(), host="localhost", port=5544,
                 dbname="devdb01", user="dro", password="sicillionbillions")

data <- dbGetQuery(con, "
  SELECT ...
  FROM mart.player_game_stats
  WHERE season >= 2020
")

dbDisconnect(con)

# 2. Define model
model_formula <- bf(
  stat_yards ~ (1 | player_id) + carries + opponent_defense_rank,
  sigma ~ (1 | player_id)  # Model variance hierarchically too
)

# 3. Set priors (document reasoning!)
priors <- c(
  prior(normal(0, 5), class = "b"),  # Regression coefficients
  prior(student_t(3, 0, 10), class = "sd"),  # Random effect SDs
  prior(student_t(3, 0, 10), class = "sigma")  # Residual SD
)

# 4. Fit model
fit <- brm(
  formula = model_formula,
  data = data,
  prior = priors,
  family = gaussian(),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  backend = "cmdstanr",
  cores = 4,
  seed = 42,
  control = list(adapt_delta = 0.95)  # Increase if divergences occur
)

# 5. Check convergence
summary(fit)
plot(fit)  # Trace plots

# 6. Save model
saveRDS(fit, "models/bayesian/my_model_v1.rds")

# 7. Save metadata
metadata <- list(
  training_samples = nrow(data),
  chains = 4,
  iterations = 2000,
  divergences = sum(nuts_params(fit)$divergent),
  rhat_max = max(rhat(fit)),
  trained_at = Sys.time()
)

jsonlite::write_json(metadata, "models/bayesian/my_model_v1_metadata.json",
                     auto_unbox = TRUE, pretty = TRUE)
```

### Neural Networks (Python/PyTorch/PyMC)

**Template:**

```python
# py/models/train_my_nn.py
import torch
import pymc as pm
import arviz as az
import pandas as pd
import joblib
from datetime import datetime

def train_bnn(X_train, y_train, X_test, y_test):
    """Train Bayesian neural network with PyMC."""

    # Standardize features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train_std = (X_train - X_mean) / X_std
    X_test_std = (X_test - X_mean) / X_std

    # Build model
    with pm.Model() as model:
        # Input layer
        w1 = pm.Normal('w1', mu=0, sigma=1, shape=(X_train.shape[1], 64))
        b1 = pm.Normal('b1', mu=0, sigma=1, shape=(64,))

        # Hidden layer
        w2 = pm.Normal('w2', mu=0, sigma=1, shape=(64, 32))
        b2 = pm.Normal('b2', mu=0, sigma=1, shape=(32,))

        # Output layer
        w3 = pm.Normal('w3', mu=0, sigma=1, shape=(32, 1))
        b3 = pm.Normal('b3', mu=0, sigma=1, shape=(1,))

        # Forward pass
        h1 = pm.math.tanh(pm.math.dot(X_train_std, w1) + b1)
        h2 = pm.math.tanh(pm.math.dot(h1, w2) + b2)
        y_pred = pm.math.dot(h2, w3) + b3

        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=10)
        likelihood = pm.Normal('y', mu=y_pred.flatten(), sigma=sigma, observed=y_train)

        # Sample
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, random_seed=42)

    # Evaluate
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=['y'])

    y_pred_mean = ppc.posterior_predictive['y'].mean(dim=['chain', 'draw']).values
    mae = np.mean(np.abs(y_pred_mean - y_test))

    # Save
    artifact = {
        'trace': trace,
        'model': model,
        'X_mean': X_mean,
        'X_std': X_std,
        'metadata': {
            'train_samples': len(X_train),
            'test_mae': float(mae),
            'trained_at': datetime.now().isoformat()
        }
    }

    joblib.dump(artifact, 'models/bayesian/my_bnn_v1.pkl')

    return artifact

if __name__ == '__main__':
    # Load data, train, save
    ...
```

---

## Summary

**Key Takeaways:**

1. **Reproducibility**: Fixed seeds, saved configs, documented hyperparameters
2. **Validation**: Explicit null checks, input validation, fail fast
3. **Documentation**: Why not what, clear docstrings, comprehensive READMEs
4. **Testing**: Unit tests for logic, integration tests for database, e2e for workflows
5. **Version Control**: Conventional commits, feature branches, pre-commit hooks
6. **Code Style**: Black (Python), tidyverse (R), consistent naming
7. **Database**: Parameterized queries, migrations for schema changes
8. **LaTeX**: Standardized TikZ styles, cross-references, full compilation cycle

**When in Doubt:**
- Check CLAUDE.md for project context
- Read existing code for patterns
- Ask before breaking conventions

---

**Version:** 1.0
**Last Updated:** October 19, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
