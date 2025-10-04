# Testing Infrastructure

This directory contains the complete test suite for the NFL analytics project.

## Quick Start

```bash
# 1. Setup testing environment
bash scripts/setup_testing.sh

# 2. Run unit tests (fast, no DB required)
pytest tests/unit -m unit

# 3. Run integration tests (requires Docker + Postgres)
docker compose up -d pg
pytest tests/integration -m integration

# 4. Run all tests with coverage
pytest --cov=py --cov-report=html
open htmlcov/index.html
```

## Test Structure

```
tests/
├── unit/                    # Fast, isolated unit tests (no I/O)
│   └── test_odds_parsing.py
├── integration/             # Multi-module & database tests
│   └── test_odds_ingestion.py
├── e2e/                     # Full pipeline tests
├── fixtures/                # Sample data for testing
│   └── sample_odds.py
├── sql/                     # SQL validation scripts
│   ├── test_schema.sql
│   └── test_data_quality.sql
├── conftest.py              # Shared fixtures
└── __init__.py
```

## Running Tests

### By Category
```bash
# Unit tests only (< 1s)
pytest tests/unit -m unit

# Integration tests (requires DB)
pytest tests/integration -m integration

# End-to-end tests (slow)
pytest tests/e2e -m e2e

# All tests
pytest
```

### By File
```bash
# Single test file
pytest tests/unit/test_odds_parsing.py

# Single test class
pytest tests/unit/test_odds_parsing.py::TestFlattenEvents

# Single test function
pytest tests/unit/test_odds_parsing.py::TestFlattenEvents::test_flatten_events_single_market_spread
```

### With Coverage
```bash
# Generate coverage report
pytest --cov=py --cov-report=term-missing

# HTML coverage report
pytest --cov=py --cov-report=html
open htmlcov/index.html

# XML for CI/CD
pytest --cov=py --cov-report=xml
```

### Other Options
```bash
# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific markers
pytest -m "unit and not slow"

# Run in parallel (faster)
pytest -n auto

# Show slowest tests
pytest --durations=10
```

## Integration Tests (Database Required)

Integration tests need a PostgreSQL database:

```bash
# Start test database
docker compose up -d pg

# Set environment variables
export TEST_DATABASE_URL="postgresql://testuser:testpass@localhost:5432/testdb"

# Apply schema
psql $TEST_DATABASE_URL -f db/001_init.sql
psql $TEST_DATABASE_URL -f db/002_timescale.sql

# Run integration tests
pytest tests/integration -m integration
```

## Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate

# Skip hooks (not recommended)
git commit --no-verify
```

### Hooks Included
- **black**: Python code formatting
- **ruff**: Python linting & auto-fixing
- **mypy**: Static type checking
- **trailing-whitespace**: Remove trailing spaces
- **check-yaml**: YAML syntax validation
- **detect-secrets**: Prevent committing secrets

## CI/CD (GitHub Actions)

Three workflows run automatically on push/PR:

### 1. Test Suite (`.github/workflows/test.yml`)
- Unit tests on Python 3.10, 3.11, 3.12
- Integration tests with TimescaleDB
- Code coverage reporting (Codecov)
- Security scanning (Bandit, Safety)

### 2. Pre-commit (`.github/workflows/pre-commit.yml`)
- Runs all pre-commit hooks
- Ensures code quality standards

### 3. Nightly Data Quality (`.github/workflows/nightly-data-quality.yml`)
- Schema validation
- Data integrity checks
- Runs daily at 6am UTC

## Writing Tests

### Unit Test Example
```python
import pytest
from py.ingest_odds_history import parse_iso

def test_parse_iso_with_z_suffix():
    """Test ISO timestamp parsing with Z (Zulu) timezone."""
    result = parse_iso("2023-09-07T17:00:00Z")
    assert result.year == 2023
    assert result.tzinfo == dt.timezone.utc
```

### Integration Test Example
```python
@pytest.mark.integration
def test_odds_insertion(db_connection):
    """Test inserting odds into database."""
    cursor = db_connection.cursor()
    cursor.execute("INSERT INTO odds_history (...) VALUES (...)")
    db_connection.commit()
    
    cursor.execute("SELECT COUNT(*) FROM odds_history")
    assert cursor.fetchone()[0] == 1
```

### Using Fixtures
```python
def test_with_sample_data(sample_odds_api_response, sample_snapshot_time):
    """Use shared fixtures from conftest.py."""
    rows = flatten_events([sample_odds_api_response], sample_snapshot_time)
    assert len(rows) > 0
```

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Test Naming
- Use descriptive names: `test_<function>_<scenario>_<expected>`
- Example: `test_parse_iso_with_z_suffix`

### 3. Assertions
- One logical assertion per test
- Use pytest's rich assertion introspection
- Prefer `assert x == y` over `assertEqual(x, y)`

### 4. Mocking
- Mock external APIs (requests)
- Mock time-dependent code (freezegun)
- Don't mock internal logic (test it directly)

### 5. Coverage
- Aim for 60%+ overall coverage
- 80%+ for critical paths (data ingestion)
- Don't obsess over 100%

## Debugging Tests

```bash
# Print output (disable capture)
pytest -s

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Trace test execution
pytest --trace

# Re-run only failed tests
pytest --lf

# Run tests matching keyword
pytest -k "odds and not slow"
```

## SQL Tests

SQL tests validate schema and data quality:

```bash
# Run schema validation
psql $DATABASE_URL -f tests/sql/test_schema.sql

# Run data quality checks
psql $DATABASE_URL -f tests/sql/test_data_quality.sql
```

## Troubleshooting

### "Import error: No module named 'py'"
- Ensure you're in the project root
- Activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements-dev.txt`

### "Connection refused" (database tests)
- Start Docker: `docker compose up -d pg`
- Check port: `lsof -i :5432`
- Wait for health check: `docker compose ps`

### "Pre-commit hook failed"
- See specific error in output
- Fix manually or use `--no-verify` to skip
- Update hooks: `pre-commit autoupdate`

### Tests pass locally but fail in CI
- Check Python version differences
- Verify database schema is applied
- Check environment variables

## Coverage Reports

After running tests with `--cov`, view coverage:

```bash
# Terminal report
pytest --cov=py --cov-report=term-missing

# HTML report (interactive)
pytest --cov=py --cov-report=html
open htmlcov/index.html

# Show only uncovered lines
pytest --cov=py --cov-report=term-missing:skip-covered
```

## Performance

### Slow Tests
Mark slow tests explicitly:

```python
@pytest.mark.slow
def test_full_season_ingestion():
    # This takes > 10s
    pass
```

Skip slow tests:
```bash
pytest -m "not slow"
```

### Parallel Execution
```bash
# Use all CPU cores
pytest -n auto

# Use 4 workers
pytest -n 4
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Pre-commit hooks](https://pre-commit.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Codecov](https://docs.codecov.com/)

---

**Last Updated**: October 3, 2025  
**Maintainer**: See CLAUDE.md for project context
