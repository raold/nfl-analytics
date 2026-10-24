# NFL Analytics - Developer Workflow Guide

**Purpose**: Complete lifecycle for developing, testing, and deploying code changes

**Status**: Production-ready (November 15, 2025)
**Version**: 1.0
**Target Audience**: Contributing developers

---

## Table of Contents

1. [Development Environment](#development-environment)
2. [Git Workflow](#git-workflow)
3. [Pre-Commit Hooks and Enforcement](#pre-commit-hooks-and-enforcement)
4. [Testing Strategy](#testing-strategy)
5. [Code Review Checklist](#code-review-checklist)
6. [Model Training Workflow](#model-training-workflow)
7. [Backtest Validation Process](#backtest-validation-process)
8. [Documentation Standards](#documentation-standards)
9. [Common Development Tasks](#common-development-tasks)
10. [Debugging Techniques](#debugging-techniques)

---

## Development Environment

### Initial Setup

```bash
# 1. Fork and clone repository
git clone https://github.com/YOUR_USERNAME/nfl-analytics.git
cd nfl-analytics

# 2. Add upstream remote
git remote add upstream https://github.com/raold/nfl-analytics.git

# 3. Install Git LFS
git lfs install
git lfs pull

# 4. Create virtual environment
uv venv .venv
source .venv/bin/activate  # macOS/Linux

# 5. Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# 6. Install R packages
Rscript -e 'renv::restore()'

# 7. Setup database
bash scripts/dev/init_dev.sh

# 8. Install pre-commit hooks
pre-commit install
```

### Daily Development Workflow

```bash
# Morning: Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, commit frequently
git add .
git commit -m "feat: add new feature"  # Pre-commit hooks run automatically

# Push to your fork
git push origin feature/your-feature-name

# Create pull request
gh pr create --title "Add new feature" --body "Description..."
```

### Environment Configuration

**Development `.env`:**

```bash
# Use weak passwords for local development
DB_PASSWORD=sicillionbillions  # pragma: allowlist secret
REDIS_PASSWORD=changeme_redis_password

# Enable debug mode
DEBUG=true
LOG_LEVEL=DEBUG

# Disable rate limiting
API_RELOAD=true  # Hot reload for development
```

**Testing `.env.test`:**

```bash
# Use separate test database
DATABASE_URL=postgresql://dro:sicillionbillions@localhost:5544/testdb  # pragma: allowlist secret
TEST_DATABASE_URL=postgresql://dro:sicillionbillions@localhost:5544/testdb  # pragma: allowlist secret

# Fast iteration
MCMC_CHAINS=2  # Fewer chains for faster testing
MCMC_ITERATIONS=500
```

---

## Git Workflow

### Branch Strategy

**Main Branch**: `main` (protected, requires PR + review)

**Branch Naming Convention:**

```
feature/    - New features
fix/        - Bug fixes
refactor/   - Code refactoring
docs/       - Documentation updates
test/       - Test additions/improvements
chore/      - Maintenance tasks
```

**Examples:**

```bash
git checkout -b feature/ensemble-v6
git checkout -b fix/sql-schema-bug
git checkout -b refactor/r-model-interface
git checkout -b docs/update-quick-start
```

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
- `refactor`: Code refactoring
- `test`: Test additions/improvements
- `docs`: Documentation updates
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**

```bash
# Good commits
git commit -m "feat(ensemble): add temperature scaling for calibration"
git commit -m "fix(sql): correct player_game_stats join pattern"
git commit -m "test(temporal): add 25+ leakage detection tests"
git commit -m "docs(guides): create quick start guide"

# Bad commits (avoid these)
git commit -m "updates"  # Too vague
git commit -m "fixed bug"  # No context
git commit -m "WIP"  # Work in progress (use --amend instead)
```

**Multi-line commits:**

```bash
git commit -m "feat(ensemble): add hybrid XGBoost v1

- 70% static + 30% incremental blend
- +4.2% improvement over pure static
- Expected ROI: +5.5-7.5%

Closes #123"
```

### Pull Request Process

**1. Create PR:**

```bash
# Via GitHub CLI (recommended)
gh pr create \
  --title "feat(ensemble): add hybrid XGBoost v1" \
  --body "## Summary
- Implements 70/30 static/incremental blend
- Validated on 2022-2024 seasons

## Testing
- Unit tests: 15 new tests
- Integration tests: 3-season backtest
- Performance: 38.03 MAE (4.2% improvement)

## Checklist
- [x] Tests pass
- [x] Documentation updated
- [x] No security issues
- [x] Code reviewed by self"

# Via GitHub web interface
# (Push branch first, then create PR manually)
```

**2. PR Template:**

```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Testing
How was this tested?

## Performance Impact
Any performance changes?

## Security Impact
Any security considerations?

## Checklist
- [ ] Tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Documentation updated
- [ ] No security issues introduced
- [ ] Backward compatible (or migration plan documented)
```

**3. Review Process:**

- **Self-review**: Review your own PR diff first
- **Automated checks**: CI/CD pipeline runs tests
- **Peer review**: At least 1 approving review required
- **Merge**: Squash and merge (keeps history clean)

---

## Pre-Commit Hooks and Enforcement

### Unified Enforcement Framework

The project uses an automated enforcement framework to catch errors at commit time.

**Framework Components:**

1. **Git Pre-Commit Hook** (`.git/hooks/pre-commit`)
2. **Enforcement Router** (`nfl_analytics/utils/enforcement_router.py`)
3. **Specialized Validators** (11 agents)

**What Gets Validated:**

| Rule | Trigger | Level | Prevents |
|------|---------|-------|----------|
| SQL Validation | `.sql`, SQL strings | BLOCKING | Schema errors |
| Backtest Health | `*_validation.py` | BLOCKING | Expensive failures |
| Model Training | `train_*.py`, `train_*.R` | BLOCKING | Data leakage |
| Security Audit | `.env`, credentials | BLOCKING | Hardcoded secrets |
| Ensemble Guard | `enhanced_ensemble_*.py` | WARNING | Degradation |
| Schema Change | `CREATE VIEW`, `ALTER TABLE` | WARNING | Cascading failures |
| Docs Sync | `docs/milestones/*.md` | INFO | Stale docs |

### Pre-Commit Hook Usage

**Automatic (runs on every commit):**

```bash
# Make changes
vim nfl_analytics/features/build_features.py

# Commit (hook runs automatically)
git add nfl_analytics/features/build_features.py
git commit -m "feat(features): add new rolling features"

# If validation fails, commit is blocked:
# ❌ BLOCKING: model_training: Missing data leakage validation
# 💡 Suggestion: Add DataLeakageDetector.validate_split() before model.fit()
```

**Manual validation (before commit):**

```bash
# Validate specific file
python nfl_analytics/utils/enforcement_router.py scripts/train_new_model.py

# Validate all staged files
pre-commit run --all-files
```

**Bypass (EMERGENCY ONLY):**

```bash
# Only use if you're certain validation is incorrect
git commit --no-verify

# Document why in commit message:
git commit --no-verify -m "fix(urgent): hotfix for production issue

Bypassing pre-commit validation because:
- Production API is down (P0 incident)
- Fix is trivial (typo in config)
- Full validation will run in CI/CD

Issue: #456"
```

### Installing Additional Hooks

```bash
# Install all pre-commit hooks
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg

# Update hooks to latest version
pre-commit autoupdate
```

---

## Testing Strategy

### Test Pyramid

```
      ┌─────────────┐
      │  End-to-End │  (5% of tests)
      │  Integration│  (25% of tests)
      │    Unit     │  (70% of tests)
      └─────────────┘
```

### Unit Tests (70%)

**Location**: `tests/unit/`
**Run Time**: <30 seconds
**Coverage Target**: 80%+

**Example:**

```python
# tests/unit/test_feature_engineering.py

import pytest
from nfl_analytics.features.rolling_stats import compute_rolling_average

def test_rolling_average_basic():
    """Test rolling average with simple data."""
    values = [1, 2, 3, 4, 5]
    window = 3

    result = compute_rolling_average(values, window)

    expected = [None, None, 2.0, 3.0, 4.0]  # First 2 are None
    assert result == expected

def test_rolling_average_window_larger_than_data():
    """Test rolling average when window > data length."""
    values = [1, 2]
    window = 5

    result = compute_rolling_average(values, window)

    assert all(v is None for v in result)

def test_rolling_average_invalid_input():
    """Test rolling average raises on invalid input."""
    with pytest.raises(ValueError):
        compute_rolling_average([], window=3)
```

**Run unit tests:**

```bash
# Run all unit tests
pytest tests/unit -v

# Run specific test file
pytest tests/unit/test_feature_engineering.py -v

# Run specific test function
pytest tests/unit/test_feature_engineering.py::test_rolling_average_basic -v

# Run with coverage
pytest tests/unit --cov=py --cov-report=html
open htmlcov/index.html
```

### Integration Tests (25%)

**Location**: `tests/integration/`
**Run Time**: 2-5 minutes
**Requires**: Docker (database, Redis)

**Example:**

```python
# tests/integration/test_ensemble_predictions.py

import pytest
from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble

@pytest.fixture
def ensemble(db_connection):
    """Initialize ensemble with database."""
    ens = EnhancedEnsemble(db_connection=db_connection)
    ens.load_models()
    return ens

def test_ensemble_prediction_complete(ensemble):
    """Test full ensemble prediction pipeline."""
    result = ensemble.predict(
        player_id="HurtJa00",
        season=2024,
        week=10
    )

    # Validate prediction structure
    assert "mean" in result
    assert "std" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "components" in result

    # Validate prediction values
    assert 0 < result["mean"] < 600  # Reasonable passing yards
    assert 0 < result["std"] < 100   # Reasonable uncertainty
    assert len(result["components"]) == 4  # 4-way ensemble

def test_ensemble_handles_missing_player(ensemble):
    """Test ensemble gracefully handles unknown player."""
    result = ensemble.predict(
        player_id="INVALID",
        season=2024,
        week=10
    )

    assert result is None or "error" in result
```

**Run integration tests:**

```bash
# Start dependencies first
docker compose up -d pg-1 redis

# Run integration tests
pytest tests/integration -v -m integration

# Run specific integration test
pytest tests/integration/test_ensemble_predictions.py -v
```

### End-to-End Tests (5%)

**Location**: `tests/e2e/`
**Run Time**: 10-30 minutes
**Purpose**: Validate complete workflows

**Example:**

```bash
# tests/e2e/test_model_training_to_deployment.sh

#!/bin/bash
set -euo pipefail

echo "=== E2E Test: Model Training to Deployment ==="

# 1. Train new model
echo "[1/5] Training model..."
Rscript R/train_state_space_v2.R

# 2. Validate model
echo "[2/5] Validating model..."
python nfl_analytics/utils/statistical_model_validator.py \
  models/bayesian/state_space_v2_metadata.json

# 3. Register model
echo "[3/5] Registering model..."
python scripts/model_registry.py register \
  --name "state_space_v2_e2e_test" \
  --path "models/bayesian/state_space_v2_canonical.rds"

# 4. Run backtest
echo "[4/5] Running backtest..."
python nfl_analytics/backtest/historical_validation_4way.py \
  --season 2024 \
  --weeks 1-5

# 5. Verify predictions
echo "[5/5] Verifying predictions..."
python scripts/quick_api_test.py

echo "✅ E2E test passed!"
```

**Run E2E tests:**

```bash
# Run all E2E tests
bash tests/e2e/run_all_e2e_tests.sh

# Run specific E2E test
bash tests/e2e/test_model_training_to_deployment.sh
```

### Test Markers

```python
# Mark tests by category
@pytest.mark.unit
def test_simple_function():
    ...

@pytest.mark.integration
def test_database_query():
    ...

@pytest.mark.slow
def test_expensive_operation():
    ...

@pytest.mark.gpu
def test_cuda_acceleration():
    ...
```

**Run tests by marker:**

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run GPU tests only if CUDA available
pytest -m gpu --gpu
```

---

## Code Review Checklist

### Self-Review (Before PR)

**Code Quality:**
- [ ] No hardcoded credentials (use environment variables)
- [ ] No commented-out code (delete or document why)
- [ ] No debug print statements (use logging instead)
- [ ] Type hints added for public functions
- [ ] Docstrings added (Google style)
- [ ] No overly long functions (>50 lines)
- [ ] No deeply nested logic (>3 levels)

**Testing:**
- [ ] Unit tests added for new functions
- [ ] Integration tests added for new workflows
- [ ] All tests pass locally
- [ ] Code coverage >80% for new code
- [ ] Edge cases tested (empty inputs, None, errors)

**Performance:**
- [ ] No N+1 database queries
- [ ] No unnecessary loops (use vectorization)
- [ ] Large data structures not held in memory
- [ ] SQL queries use indexes
- [ ] Caching used where appropriate

**Security:**
- [ ] No SQL injection vulnerabilities (use parameterized queries)
- [ ] User input validated (Pydantic models)
- [ ] Secrets not logged
- [ ] File uploads validated (if applicable)
- [ ] Authentication/authorization checked

**Documentation:**
- [ ] README updated (if user-facing change)
- [ ] CLAUDE.md updated (if architecture change)
- [ ] Docstrings added for new functions
- [ ] Comments added for complex logic
- [ ] Migration guide (if breaking change)

### Peer Review (For Reviewers)

**Functional Correctness:**
- [ ] Code does what PR description says
- [ ] Logic is correct (no off-by-one errors, etc.)
- [ ] Edge cases handled
- [ ] Error handling appropriate

**Code Style:**
- [ ] Follows project conventions (naming, structure)
- [ ] Black formatter applied (Python)
- [ ] Tidyverse style (R)
- [ ] SQL keywords uppercase

**Testing:**
- [ ] Tests are comprehensive
- [ ] Tests are not flaky (deterministic)
- [ ] Tests are fast (<30 sec for unit tests)
- [ ] Test names are descriptive

**Impact Assessment:**
- [ ] Backward compatible (or migration documented)
- [ ] No performance regression
- [ ] No security vulnerabilities introduced
- [ ] Database migrations safe

---

## Model Training Workflow

### Standard Training Process

```bash
# 1. Validate data exists
psql $DATABASE_URL -c "SELECT COUNT(*) FROM mart.player_game_stats;"

# 2. Pre-training validation
python nfl_analytics/utils/data_leakage_detector.py \
  --validate-split \
  --train-start 2018 \
  --train-end 2023 \
  --test-start 2024

# 3. Train model
Rscript R/train_state_space_v3.R

# 4. Post-training validation
python nfl_analytics/utils/statistical_model_validator.py \
  models/bayesian/state_space_v3_metadata.json

# 5. Register to database
python scripts/model_registry.py register \
  --name "state_space_v3_canonical" \
  --path "models/bayesian/state_space_v3_canonical.rds" \
  --version "3.0.0"

# 6. Run backtest
python nfl_analytics/backtest/historical_validation_4way.py \
  --season 2022-2024

# 7. Compare to baseline
python scripts/compare_models.py \
  --baseline state_space_v2 \
  --candidate state_space_v3 \
  --seasons 2022-2024
```

### Model Training Checklist

**Pre-Training:**
- [ ] Database connection verified
- [ ] Training window validated (2018-2024 canonical)
- [ ] Data leakage checks passed
- [ ] Temporal splits validated (no player overlap)
- [ ] Feature engineering reviewed (no future data)

**During Training:**
- [ ] MCMC diagnostics monitored (R-hat, ESS)
- [ ] Training logs reviewed (no warnings/errors)
- [ ] Convergence achieved (R-hat < 1.01)
- [ ] Model file saved successfully

**Post-Training:**
- [ ] Model metadata created (.json file)
- [ ] Convergence validated (R-hat, ESS, divergences)
- [ ] Calibration checked (90% coverage target)
- [ ] Predictions realistic (200-350 yards for QBs)
- [ ] Backtest results documented
- [ ] Model registered to database

---

## Backtest Validation Process

### Standard Backtest

```bash
# Run 3-season backtest (canonical validation)
python nfl_analytics/backtest/historical_validation_4way.py \
  --seasons 2022 2023 2024 \
  --weeks 1-18 \
  --output results/backtest_$(date +%Y%m%d).pkl

# Generate report
python scripts/generate_backtest_report.py \
  --input results/backtest_20251115.pkl \
  --output docs/milestones/BACKTEST_REPORT_20251115.md
```

### Backtest Validation Gates

**Performance Gates:**
- [ ] MAE < 45 yards (target: <42 yards)
- [ ] Calibration 85-95% (target: 90%)
- [ ] Win rate > 59% (target: 62%+)
- [ ] ROI > 5% (target: 7%+)
- [ ] Sharpe ratio > 2.2 (target: 2.5+)

**Statistical Significance:**
- [ ] Improvement vs baseline statistically significant (p < 0.05)
- [ ] Diebold-Mariano test passed
- [ ] Consistency across seasons (std < 2 yards MAE)

**Risk Gates:**
- [ ] Max drawdown < -8%
- [ ] No catastrophic losses (>-20% in single day)
- [ ] Kelly criterion sizing validated
- [ ] Confidence filtering effective (>80% when betting)

---

## Documentation Standards

### Code Documentation

**Python Docstrings (Google Style):**

```python
def predict_passing_yards(
    player_id: str,
    season: int,
    week: int,
    opponent: str,
    weather: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Predict passing yards for a player in a specific game.

    Uses 4-way ensemble (state-space, hierarchical, XGBoost, informative priors)
    with temperature scaling for calibration.

    Args:
        player_id: NFL player ID (e.g., "HurtJa00")
        season: Season year (1999-2030)
        week: Week number (1-22)
        opponent: Opponent team abbreviation (e.g., "DAL")
        weather: Optional weather conditions (temp, wind, precipitation)

    Returns:
        Dictionary with keys:
            - mean: Expected passing yards
            - std: Standard deviation (uncertainty)
            - ci_lower: 90% confidence interval lower bound
            - ci_upper: 90% confidence interval upper bound
            - components: Predictions from each model

    Raises:
        ValueError: If player_id is invalid or not found in database
        ValueError: If season/week out of valid range

    Example:
        >>> result = predict_passing_yards("HurtJa00", 2024, 10, "DAL")
        >>> print(f"Prediction: {result['mean']:.1f} ± {result['std']:.1f}")
        Prediction: 268.5 ± 32.4
    """
    ...
```

**R Documentation (roxygen2 style):**

```r
#' Predict Passing Yards Using State-Space Model
#'
#' @description
#' Uses Kalman filtering to predict passing yards with time-varying
#' effects for player skill, opponent strength, and game context.
#'
#' @param player_id Character. NFL player ID (e.g., "HurtJa00")
#' @param season Integer. Season year (1999-2030)
#' @param week Integer. Week number (1-22)
#' @param model brmsfit object. Trained state-space model
#'
#' @return List with:
#'   \item{mean}{Expected passing yards}
#'   \item{sd}{Standard deviation}
#'   \item{ci_lower}{90% CI lower bound}
#'   \item{ci_upper}{90% CI upper bound}
#'
#' @examples
#' \dontrun{
#' model <- readRDS("models/bayesian/state_space_v2_canonical.rds")
#' pred <- predict_passing_yards("HurtJa00", 2024, 10, model)
#' cat(sprintf("Prediction: %.1f ± %.1f\n", pred$mean, pred$sd))
#' }
#'
#' @export
predict_passing_yards <- function(player_id, season, week, model) {
  ...
}
```

### File Headers

**Python:**

```python
"""Feature engineering for NFL player statistics.

This module provides functions for building temporal features while
preventing data leakage. All features are computed using as-of joins
to ensure only historical data is used.

Author: NFL Analytics Team
Created: 2025-11-15
Last Modified: 2025-11-15
"""

import pandas as pd
from typing import List, Dict, Optional
```

**R:**

```r
# Train State-Space Model v2 (Canonical Window)
#
# Purpose: Train state-space passing yards model using canonical
#          training window (2018-2024 + H=4 decay on 2015-2017)
#
# Author: NFL Analytics Team
# Created: 2025-11-08
# Last Modified: 2025-11-15
#
# Dependencies:
#   - brms (>=2.18.0)
#   - cmdstanr (>=0.6.0)
#   - tidyverse
#   - RPostgres

library(brms)
library(cmdstanr)
library(tidyverse)
```

---

## Common Development Tasks

### Adding a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/new-rolling-stats

# 2. Implement feature
# Edit nfl_analytics/features/rolling_stats.py

# 3. Add unit tests
# Edit tests/unit/test_rolling_stats.py

# 4. Add integration test
# Edit tests/integration/test_feature_pipeline.py

# 5. Update documentation
# Edit docs/guides/COMMON_TASKS_REFERENCE.md

# 6. Run tests
pytest tests/unit/test_rolling_stats.py -v
pytest tests/integration/test_feature_pipeline.py -v

# 7. Commit and push
git add .
git commit -m "feat(features): add exponential weighted rolling stats"
git push origin feature/new-rolling-stats

# 8. Create PR
gh pr create
```

### Fixing a Bug

```bash
# 1. Create fix branch
git checkout -b fix/sql-join-error

# 2. Write failing test (demonstrates bug)
# Edit tests/unit/test_sql_queries.py

# 3. Run test (should fail)
pytest tests/unit/test_sql_queries.py::test_player_game_join -v

# 4. Fix the bug
# Edit nfl_analytics/data/sql_queries.py

# 5. Run test again (should pass)
pytest tests/unit/test_sql_queries.py::test_player_game_join -v

# 6. Commit with descriptive message
git commit -m "fix(sql): correct player_game_stats join pattern

- Use (season, week, team) composite key instead of game_id
- Fixes issue where game_id doesn't exist in player_game_stats
- Adds test to prevent regression

Fixes #456"

# 7. Create PR
gh pr create
```

### Refactoring Code

```bash
# 1. Create refactor branch
git checkout -b refactor/simplify-ensemble

# 2. Add tests first (ensure current behavior captured)
pytest tests/unit/test_ensemble.py -v --cov=nfl_analytics/ensemble

# 3. Refactor code
# Edit nfl_analytics/ensemble/enhanced_ensemble_v5.py

# 4. Verify tests still pass
pytest tests/unit/test_ensemble.py -v

# 5. Commit
git commit -m "refactor(ensemble): extract component loading logic

- Create _load_component() method to reduce duplication
- No functional changes (all tests pass)
- Improves readability and maintainability"
```

---

## Debugging Techniques

### Python Debugging

**Using pdb:**

```python
import pdb

def problematic_function(data):
    # ... some code ...
    pdb.set_trace()  # Debugger will stop here
    result = complex_calculation(data)
    return result
```

**Using logging:**

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def feature_engineering(df):
    logger.debug(f"Input shape: {df.shape}")
    logger.debug(f"Columns: {df.columns.tolist()}")

    result = df.groupby("player_id").agg({"yards": "mean"})

    logger.debug(f"Output shape: {result.shape}")
    return result
```

**Using pytest debugger:**

```bash
# Drop into debugger on test failure
pytest tests/unit/test_features.py --pdb

# Drop into debugger on specific test
pytest tests/unit/test_features.py::test_rolling_average --pdb
```

### R Debugging

**Using browser():**

```r
predict_yards <- function(player_id, model) {
  # ... some code ...
  browser()  # Debugger will stop here
  predictions <- posterior_predict(model, newdata)
  return(predictions)
}
```

**Using print debugging:**

```r
train_model <- function(data) {
  cat("Input rows:", nrow(data), "\n")
  cat("Input cols:", ncol(data), "\n")

  model <- brm(formula, data = data)

  cat("Model summary:\n")
  print(summary(model))

  return(model)
}
```

### SQL Debugging

**Using EXPLAIN:**

```sql
-- Check query plan
EXPLAIN ANALYZE
SELECT *
FROM mart.player_game_stats pgs
JOIN public.games g
  ON pgs.season = g.season
  AND pgs.week = g.week
WHERE pgs.player_id = 'HurtJa00';
```

**Using timing:**

```sql
-- Enable timing
\timing

-- Run query
SELECT COUNT(*) FROM mart.player_game_stats;

-- Output: Time: 45.123 ms
```

---

**Status**: Developer workflow guide complete.

**Total Lines**: 588 (target: 500-600)

**Next**: Create Troubleshooting Master Guide.
