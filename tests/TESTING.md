# Testing Strategy & CI/CD Plan

**Last Updated**: October 3, 2025  
**Status**: Planning → Implementation

---

## 🎯 Testing Philosophy

### Goals
1. **Prevent regressions** in data pipelines (ingestion, transformations)
2. **Validate data quality** (schema, constraints, null checks)
3. **Ensure reproducibility** (same inputs → same outputs)
4. **Fast feedback** (local tests < 30s, CI tests < 5min)
5. **Isolated tests** (no dependency on live APIs or production DB)

### Non-Goals
- Testing third-party libraries (nflfastR, The Odds API)
- Performance benchmarking (separate from functional tests)
- Integration with production systems (this is research code)

---

## 🧪 Test Pyramid

```
         /\
        /  \  E2E Integration Tests (10%)
       /____\  
      /      \  
     / Module \ Integration Tests (30%)
    /  Tests  \
   /___________\
  /             \
 /  Unit Tests   \ (60%)
/_________________\
```

### Unit Tests (60% coverage target)
- **Scope**: Individual functions, pure logic
- **Speed**: < 1s per module
- **Mocking**: External APIs, database, file I/O
- **Examples**:
  - `py/ingest_odds_history.py::flatten_events()` - JSON parsing
  - `py/risk/generate_scenarios.py::simulate_returns()` - Monte Carlo logic
  - `py/rl/ope_gate.py` - DR estimator calculations

### Integration Tests (30%)
- **Scope**: Multi-module workflows, database interactions
- **Speed**: < 30s per test suite
- **Setup**: Dockerized test DB (separate from dev DB)
- **Examples**:
  - Ingest 1 day of odds → verify row count & schema
  - Load 1 season of plays → check EPA aggregation
  - End-to-end: CSV → scenarios → CVaR → JSON

### E2E Tests (10%)
- **Scope**: Full pipeline acceptance tests
- **Speed**: < 5min per test
- **Setup**: Full Docker compose stack
- **Examples**:
  - `scripts/run_reports.sh` produces valid TeX outputs
  - Quarto notebooks render without errors
  - LaTeX builds successfully

---

## 📁 Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures (DB, mock API)
│
├── unit/                       # Fast, isolated unit tests
│   ├── __init__.py
│   ├── test_odds_parsing.py   # flatten_events, parse_iso
│   ├── test_risk_scenarios.py # Monte Carlo simulation
│   ├── test_ope_estimators.py # DR, IPS calculations
│   └── test_pricing.py         # Teaser EV, middle breakeven
│
├── integration/                # DB + multi-module tests
│   ├── __init__.py
│   ├── test_odds_ingestion.py # Mock API → DB roundtrip
│   ├── test_pbp_pipeline.py   # Mock nflfastR → plays table
│   └── test_feature_eng.py    # plays → mart.team_epa
│
├── e2e/                        # Full pipeline tests
│   ├── __init__.py
│   ├── test_report_pipeline.py # run_reports.sh validation
│   └── test_notebook_render.py # Quarto rendering
│
├── fixtures/                   # Test data
│   ├── sample_odds_response.json
│   ├── sample_pbp_1game.csv
│   └── sample_games_10rows.csv
│
└── sql/                        # SQL test scripts
    ├── test_schema.sql         # Schema validation queries
    └── test_data_quality.sql   # Constraint checks
```

---

## 🐍 Python Testing Setup

### Dependencies (`requirements-dev.txt`)
```txt
# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.1
pytest-timeout>=2.1.0
pytest-xdist>=3.3.1  # Parallel execution

# Mocking
responses>=0.23.0     # HTTP mocking
freezegun>=1.2.2      # Time mocking

# Code quality
black>=23.7.0
ruff>=0.0.286
mypy>=1.5.0
pre-commit>=3.3.3

# Database testing
testing.postgresql>=1.3.0  # Disposable Postgres instances
```

### Pytest Configuration (`pytest.ini`)
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --strict-markers
    --cov=py
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=60
    --timeout=300
markers =
    unit: Fast unit tests (no I/O)
    integration: Integration tests (DB required)
    e2e: End-to-end tests (full stack)
    slow: Tests that take > 10s
```

### Example Unit Test
```python
# tests/unit/test_odds_parsing.py
import datetime as dt
from py.ingest_odds_history import flatten_events, parse_iso

def test_parse_iso_with_timezone():
    """Test ISO timestamp parsing with Z suffix."""
    result = parse_iso("2023-09-07T12:00:00Z")
    assert result.tzinfo == dt.timezone.utc
    assert result.year == 2023
    assert result.month == 9

def test_parse_iso_none():
    """Test parse_iso handles None gracefully."""
    assert parse_iso(None) is None

def test_flatten_events_empty():
    """Test flatten_events with empty list."""
    result = flatten_events([], dt.datetime.now(dt.timezone.utc))
    assert result == []

def test_flatten_events_single_market():
    """Test flatten_events with single event, single bookmaker."""
    snapshot = dt.datetime(2023, 9, 1, tzinfo=dt.timezone.utc)
    events = [{
        "id": "abc123",
        "sport_key": "americanfootball_nfl",
        "commence_time": "2023-09-07T17:00:00Z",
        "home_team": "Buffalo Bills",
        "away_team": "Arizona Cardinals",
        "bookmakers": [{
            "key": "fanduel",
            "title": "FanDuel",
            "last_update": "2023-09-01T12:00:00Z",
            "markets": [{
                "key": "spreads",
                "last_update": "2023-09-01T12:00:00Z",
                "outcomes": [
                    {"name": "Buffalo Bills", "price": 1.91, "point": -6.5},
                    {"name": "Arizona Cardinals", "price": 1.91, "point": 6.5}
                ]
            }]
        }]
    }]
    
    result = flatten_events(events, snapshot)
    
    assert len(result) == 2
    assert result[0]["event_id"] == "abc123"
    assert result[0]["bookmaker_key"] == "fanduel"
    assert result[0]["market_key"] == "spreads"
    assert result[0]["outcome_point"] == -6.5
    assert result[1]["outcome_point"] == 6.5
```

### Example Integration Test
```python
# tests/integration/test_odds_ingestion.py
import pytest
import responses
from datetime import datetime, timezone
from py.ingest_odds_history import build_request, flatten_events, upsert_rows

@pytest.fixture
def test_db(postgresql):
    """Create a temporary test database with schema."""
    conn = postgresql.cursor()
    # Apply 001_init.sql schema
    with open("db/001_init.sql") as f:
        conn.execute(f.read())
    yield postgresql
    postgresql.close()

@responses.activate
def test_odds_ingestion_roundtrip(test_db):
    """Test full ingestion: API mock → DB → verify."""
    # Mock The Odds API response
    responses.add(
        responses.GET,
        "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds",
        json={"data": [/* sample event */]},
        status=200,
        headers={"x-requests-remaining": "19999"}
    )
    
    # Run ingestion
    snapshot_at = datetime(2023, 9, 1, tzinfo=timezone.utc)
    # ... (actual ingestion logic)
    
    # Verify DB state
    cursor = test_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM odds_history WHERE snapshot_at = %s", (snapshot_at,))
    count = cursor.fetchone()[0]
    
    assert count == 6  # 3 outcomes × 2 bookmakers (example)
```

---

## 🔧 Pre-Commit Hooks

### `.pre-commit-config.yaml`
```yaml
repos:
  # Python code formatting
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.10
        files: ^py/

  # Python linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.0.286
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
        files: ^py/

  # Python type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
        files: ^py/

  # R code formatting
  - repo: https://github.com/lorenzwalthert/precommit
    rev: v0.3.2
    hooks:
      - id: style-files
        args: [--scope=line_breaks, --scope=spacing, --scope=tokens]
        files: \\.(R|r)$

  # SQL formatting (optional)
  - repo: https://github.com/sqlfluff/sqlfluff
    rev: 2.3.2
    hooks:
      - id: sqlfluff-lint
        files: ^db/.*\\.sql$
      - id: sqlfluff-fix
        files: ^db/.*\\.sql$

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=5000]  # Block files > 5MB
      - id: check-merge-conflict
      - id: detect-private-key

  # Security scanning
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: [--baseline, .secrets.baseline]
```

### Installation
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Test on existing code
```

---

## 🔄 GitHub Actions CI/CD

### Workflow: Test Suite (`.github/workflows/test.yml`)
```yaml
name: Test Suite

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    name: Unit Tests (Python)
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run unit tests
        run: pytest tests/unit -m unit --cov=py --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unit
          name: unit-tests-${{ matrix.python-version }}

  integration-tests:
    name: Integration Tests (Python + DB)
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: timescale/timescaledb:latest-pg16
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Apply database schema
        env:
          POSTGRES_HOST: localhost
          POSTGRES_PORT: 5432
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        run: |
          psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB" \
            -f db/001_init.sql
          psql "postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB" \
            -f db/002_timescale.sql
      
      - name: Run integration tests
        env:
          POSTGRES_HOST: localhost
          POSTGRES_PORT: 5432
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        run: pytest tests/integration -m integration --cov=py --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: integration

  r-tests:
    name: R Tests
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: r-lib/actions/setup-r@v2
        with:
          r-version: '4.3.0'
      
      - name: Cache R packages
        uses: actions/cache@v3
        with:
          path: ${{ env.R_LIBS_USER }}
          key: ${{ runner.os }}-r-${{ hashFiles('renv.lock') }}
      
      - name: Install R dependencies
        run: |
          Rscript -e 'install.packages("renv")'
          Rscript -e 'renv::restore()'
      
      - name: Run R tests
        run: Rscript -e 'testthat::test_dir("R/tests")'

  lint:
    name: Code Quality (Lint)
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install linters
        run: pip install black ruff mypy
      
      - name: Check formatting (black)
        run: black --check py/
      
      - name: Lint (ruff)
        run: ruff check py/
      
      - name: Type check (mypy)
        run: mypy py/ --ignore-missing-imports

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### Workflow: Nightly Data Quality (`.github/workflows/nightly-data-quality.yml`)
```yaml
name: Nightly Data Quality Checks

on:
  schedule:
    - cron: '0 6 * * *'  # 6am UTC daily
  workflow_dispatch:  # Manual trigger

jobs:
  data-quality:
    name: Check Data Integrity
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: timescale/timescaledb:latest-pg16
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Apply schema
        run: |
          psql "postgresql://testuser:testpass@localhost:5432/testdb" -f db/001_init.sql
          psql "postgresql://testuser:testpass@localhost:5432/testdb" -f db/002_timescale.sql
      
      - name: Run data quality SQL tests
        run: |
          psql "postgresql://testuser:testpass@localhost:5432/testdb" -f tests/sql/test_schema.sql
          psql "postgresql://testuser:testpass@localhost:5432/testdb" -f tests/sql/test_data_quality.sql
      
      - name: Report to Slack (on failure)
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "❌ Nightly data quality check failed!",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Data Quality Check Failed*\nRepo: ${{ github.repository }}\nRun: ${{ github.run_id }}"
                  }
                }
              ]
            }
```

### Workflow: Documentation (`.github/workflows/docs.yml`)
```yaml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - '**.md'
      - 'notebooks/**.qmd'

jobs:
  render-notebooks:
    name: Render Quarto Notebooks
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: quarto-dev/quarto-actions/setup@v2
      
      - name: Render notebooks
        run: |
          quarto render notebooks/04_score_validation.qmd
          quarto render notebooks/12_risk_sizing.qmd
      
      - name: Upload rendered HTML
        uses: actions/upload-artifact@v3
        with:
          name: rendered-notebooks
          path: notebooks/*.html
```

---

## 📊 Coverage Targets

### Phase 1 (MVP - Week 1)
- **Unit tests**: 40% coverage on core modules
  - `py/ingest_odds_history.py`: parsing functions
  - `py/risk/generate_scenarios.py`: simulation logic
  - `py/pricing/teaser.py`: EV calculations
- **Integration tests**: 2-3 critical paths
  - Odds ingestion smoke test
  - Feature engineering pipeline
- **CI**: Basic GitHub Actions (unit + lint)

### Phase 2 (Production - Month 1)
- **Unit tests**: 60% coverage
- **Integration tests**: All data pipelines
- **E2E tests**: `run_reports.sh` validation
- **Pre-commit hooks**: Enforced on all commits

### Phase 3 (Maintenance - Ongoing)
- **Maintain**: 60%+ coverage
- **Add**: Property-based testing (Hypothesis)
- **Monitor**: Coverage trends, flaky tests

---

## 🚨 Critical Test Cases

### Must-Have Tests
1. **Odds ingestion idempotency**: Running twice doesn't duplicate rows
2. **Play-by-play schema validation**: All expected columns exist
3. **EPA calculation correctness**: Known game → expected EPA
4. **CVaR optimizer feasibility**: Always returns valid portfolio
5. **OPE estimator bounds**: DR estimate within [min_return, max_return]
6. **TeX output validity**: Generated tables compile in LaTeX

### Edge Cases
- Empty API responses
- Missing/null values in database
- Zero-variance scenarios (simulation)
- Negative odds (invalid data)
- Division by zero (importance sampling)

---

## 🛠️ Implementation Checklist

### Week 1: Foundation
- [ ] Create `tests/` directory structure
- [ ] Add `requirements-dev.txt`
- [ ] Configure `pytest.ini`
- [ ] Write 3-5 unit tests for `ingest_odds_history.py`
- [ ] Set up GitHub Actions basic workflow
- [ ] Add pre-commit config (black + ruff only)

### Week 2: Integration
- [ ] Add `conftest.py` with DB fixtures
- [ ] Write integration test for odds ingestion
- [ ] Write integration test for play-by-play loading
- [ ] Add SQL test scripts (`test_schema.sql`)
- [ ] Enable coverage reporting (Codecov)

### Week 3: CI/CD
- [ ] Add R testing workflow
- [ ] Add security scanning (Trivy)
- [ ] Set up nightly data quality checks
- [ ] Document test running in CLAUDE.md
- [ ] Enforce pre-commit hooks in CI

### Week 4: Polish
- [ ] Add E2E test for `run_reports.sh`
- [ ] Add property-based tests (Hypothesis)
- [ ] Set up test coverage badges
- [ ] Write testing guidelines for contributors
- [ ] Review and refactor flaky tests

---

## 📚 Resources

### Python Testing
- pytest: https://docs.pytest.org/
- pytest-cov: https://pytest-cov.readthedocs.io/
- responses (HTTP mocking): https://github.com/getsentry/responses
- testing.postgresql: https://github.com/tk0miya/testing.postgresql

### R Testing
- testthat: https://testthat.r-lib.org/
- mockery: https://github.com/r-lib/mockery

### CI/CD
- GitHub Actions: https://docs.github.com/en/actions
- Codecov: https://docs.codecov.com/
- Pre-commit: https://pre-commit.com/

---

**Next Steps**: Start with Week 1 checklist, prioritize unit tests for data parsing and transformation logic.
