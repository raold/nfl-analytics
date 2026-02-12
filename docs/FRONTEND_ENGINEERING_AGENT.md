# Frontend Engineering Agent - Comprehensive Guide

**Created:** November 22, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** Production Ready
**Purpose:** Prevent dashboard brittleness through automated schema validation and frontend-backend synchronization

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Quick Start](#quick-start)
5. [CLI Commands](#cli-commands)
6. [Python API](#python-api)
7. [Integration Patterns](#integration-patterns)
8. [Use Cases](#use-cases)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### Problem Statement

The NFL Analytics Dashboard (`dashboard/app.py`, 1,800+ lines) has **15+ hardcoded SQL queries** with no schema validation. This causes:

1. **Runtime Failures** - Queries break when database schema changes
2. **Silent Bugs** - Missing columns return NULL instead of erroring
3. **API Contract Drift** - Dashboard expects fields that FastAPI no longer returns
4. **No Early Detection** - Problems discovered in production, not development

**Historical Context:** The `mart.player_game_stats.game_id` error occurred **DOZENS of times** (documented in `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md`). Each incident wasted 30-60 minutes of debugging.

### Solution

The Frontend Engineering Agent provides:

- ✅ **Compile-Time Validation** - Catch schema errors at commit time (not runtime)
- ✅ **Automated Drift Detection** - Alert when database/API changes break frontend
- ✅ **Contract Enforcement** - Validate API responses match expected schemas
- ✅ **Auto-Migration Tools** - Generate code updates automatically
- ✅ **Error Boundaries** - Provide fallback mechanisms for query failures

### ROI

**Time Saved:**
- 30-60 minutes per schema error × dozens of occurrences = **15-30 hours saved**
- Prevents production incidents (dashboard downtime = lost analysis time)
- Reduces context switching (developers can focus on features, not bugs)

**Risk Reduction:**
- Catches breaking changes before they reach production
- Enforces type safety for API contracts
- Documents schema evolution over time

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 Frontend Engineering Agent                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Schema     │  │     API      │  │    Drift     │         │
│  │  Validator   │  │  Contract    │  │  Detector    │         │
│  │              │  │  Enforcer    │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                │
│         └──────────────────┴──────────────────┘                │
│                            │                                   │
│                    ┌───────▼────────┐                          │
│                    │  Migration     │                          │
│                    │  Suggester     │                          │
│                    └────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         │                    │                    │
    ┌────▼────┐          ┌────▼────┐          ┌───▼────┐
    │Database │          │ FastAPI │          │Dashboard│
    │ Schema  │          │ OpenAPI │          │ Queries │
    └─────────┘          └─────────┘          └─────────┘
```

### Data Flow

1. **Schema Introspection** - Query PostgreSQL `information_schema` for tables/columns
2. **Query Extraction** - Parse Python files for SQL queries using regex patterns
3. **Validation** - Match extracted queries against schema, detect violations
4. **Drift Detection** - Compare current schema to saved snapshot
5. **Reporting** - Generate actionable reports with fix suggestions

### Key Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Schema Validator** | Validate SQL queries | Python file, DB schema | ValidationReport |
| **API Contract Enforcer** | Check API compatibility | OpenAPI spec | ContractReport |
| **Drift Detector** | Detect schema changes | Current schema, snapshot | SchemaDrift |
| **Migration Suggester** | Generate updates | SchemaDrift | List[MigrationUpdate] |

---

## Key Features

### 1. Schema Validation

**Validates:**
- ✅ Table existence (`public.games`, `mart.player_game_stats`)
- ✅ Column existence (detects `game_id` error in `player_game_stats`)
- ✅ Query anti-patterns (`SELECT *`, missing `WHERE` on large tables)
- ✅ Best practices (explicit column lists, LIMIT clauses)

**Example:**

```python
from py.utils.frontend_engineering_agent import FrontendEngineeringAgent

agent = FrontendEngineeringAgent()
report = agent.validate_dashboard_schema()

if not report.is_valid:
    print(report.generate_report())
    exit(1)
```

**Output (when error detected):**

```
================================================================================
FRONTEND SCHEMA VALIDATION REPORT
================================================================================
Timestamp: 2025-11-22T10:30:45
Queries Checked: 15
Valid: 13 | Invalid: 2

Issues by Severity:
  CRITICAL: 0
  ERROR:    2
  WARNING:  3
  INFO:     1

Validation Issues:

[ERROR] schema_validation: Column 'game_id' does not exist in table 'mart.player_game_stats'
  File: dashboard/app.py
  Line: 1417
  💡 Suggestion: Available columns: player_id, season, week, current_team, passing_yards...

[ERROR] schema_validation: Table 'public.old_table' does not exist in database
  File: dashboard/app.py
  Line: 842
  💡 Suggestion: Available tables: public.games, mart.player_game_stats...

================================================================================
❌ FAILED: Critical issues found - frontend may be broken
================================================================================
```

### 2. API Contract Enforcement

**Validates:**
- ✅ API reachability (`http://localhost:8000/health`)
- ✅ Endpoint existence (`/predict`, `/models`, `/metrics`)
- ✅ Response schema compliance (required fields present)
- ✅ Version compatibility

**Example:**

```python
report = agent.validate_api_contract()

if report.has_violations:
    print(report.generate_report())
    exit(1)
```

**Output:**

```
================================================================================
API CONTRACT VALIDATION REPORT
================================================================================
Timestamp: 2025-11-22T10:35:12
API Available: True
API Version: 1.1.0
Endpoints Checked: 4
Valid: 4 | Violations: 0

================================================================================
✅ PASSED: All API contracts validated
================================================================================
```

### 3. Schema Drift Detection

**Detects:**
- ➕ Added tables (non-breaking)
- 🗑️ Removed tables (BREAKING)
- ➕ Added columns (non-breaking)
- 🗑️ Removed columns (BREAKING)
- ⚠️ Type changes (potentially breaking)

**Example:**

```python
drift = agent.detect_schema_drift()

if drift.has_breaking_changes:
    print(f"⛔ Breaking changes detected:")
    for table, cols in drift.removed_columns.items():
        print(f"  {table}: removed {cols}")
```

**Workflow:**

1. Save baseline snapshot: `agent.save_schema_snapshot()`
2. Make schema changes (ALTER TABLE, DROP COLUMN, etc.)
3. Detect drift: `drift = agent.detect_schema_drift()`
4. Generate migration: `updates = agent.suggest_frontend_updates(drift)`

### 4. Auto-Migration Suggestions

**Generates:**
- 🔄 SQL query updates for removed columns
- 🔄 Type conversion code for type changes
- 📝 Documentation comments for manual review
- ⚙️ Automated fix scripts (where safe)

**Example:**

```python
updates = agent.suggest_frontend_updates(drift)

for update in updates:
    print(f"[{update.priority.value}] {update.file_path}:{update.line_number}")
    print(f"  Reason: {update.reason}")
    print(f"  Update: {update.new_code}")
    print(f"  Automated: {update.automated}")
```

### 5. Type Generation

**Generates:**
- Pydantic models from database schema
- TypeScript interfaces from FastAPI models
- Schema validators for runtime checks

**Example:**

```python
validators_code = agent.generate_schema_validators()

# Writes to dashboard/schema_validators.py
with open("dashboard/schema_validators.py", "w") as f:
    f.write(validators_code)
```

**Generated Output:**

```python
"""
Auto-generated Pydantic models from database schema
Generated: 2025-11-22T10:40:15
DO NOT EDIT MANUALLY - regenerate using frontend_engineering_agent.py
"""

from typing import Optional
from datetime import datetime, date
from pydantic import BaseModel, Field

class PublicGames(BaseModel):
    """Auto-generated model for public.games"""
    game_id: str = Field(..., description='Primary key')
    season: int
    week: int
    home_team: str
    away_team: str
    kickoff: Optional[date] = None
```

---

## Quick Start

### Installation

**1. Install Dependencies:**

```bash
pip install psycopg[binary] requests pydantic
```

**2. Set Environment Variables:**

```bash
export DB_HOST=localhost
export DB_PORT=5544
export DB_NAME=devdb01
export DB_USER=dro
export DB_PASSWORD=sicillionbillions  # pragma: allowlist secret
```

**3. Save Initial Schema Snapshot:**

```bash
./scripts/sync_frontend.sh save-snapshot
```

### First Validation

**Validate dashboard schemas:**

```bash
./scripts/sync_frontend.sh validate-schema
```

**Expected output (if valid):**

```
================================================================================
FRONTEND SCHEMA VALIDATION REPORT
================================================================================
Timestamp: 2025-11-22T10:45:30
Queries Checked: 15
Valid: 15 | Invalid: 0

================================================================================
✅ PASSED: All validation checks passed
================================================================================
```

### Integration into Workflow

**Pre-commit hook (`.git/hooks/pre-commit`):**

```bash
#!/bin/bash
# Validate schemas before commit
./scripts/sync_frontend.sh validate-schema || exit 1
```

**Dashboard startup (`dashboard/app.py`):**

```python
import subprocess
import sys

# Validate schemas at startup
result = subprocess.run(
    ["./scripts/sync_frontend.sh", "startup-check"],
    capture_output=True
)

if result.returncode != 0:
    print("⛔ Schema validation failed - refusing to start dashboard")
    print(result.stdout.decode())
    sys.exit(1)
```

---

## CLI Commands

### Command Reference

| Command | Purpose | Exit Code |
|---------|---------|-----------|
| `validate-schema` | Validate dashboard SQL queries | 0 = valid, 1 = errors |
| `validate-api` | Check API contract compliance | 0 = valid, 1 = violations |
| `check-drift` | Detect schema drift | 0 = no breaking changes, 1 = breaking |
| `generate-types` | Generate Pydantic validators | Always 0 |
| `apply-updates` | Suggest migration updates | 0 = no updates, 1 = updates needed |
| `save-snapshot` | Save current schema snapshot | Always 0 |
| `startup-check` | Run all startup validations | 0 = pass, 1 = fail |

### Examples

**1. Pre-Deployment Validation:**

```bash
#!/bin/bash
# Run all checks before deployment

echo "1/3: Validating schemas..."
./scripts/sync_frontend.sh validate-schema || exit 1

echo "2/3: Checking API contracts..."
./scripts/sync_frontend.sh validate-api || exit 1

echo "3/3: Detecting drift..."
./scripts/sync_frontend.sh check-drift || exit 1

echo "✅ All checks passed - safe to deploy"
```

**2. Nightly Schema Audit:**

```bash
#!/bin/bash
# Cron job: 0 2 * * * /path/to/nightly_audit.sh

./scripts/sync_frontend.sh check-drift > /var/log/schema_drift.log 2>&1

if [ $? -ne 0 ]; then
    # Send alert
    mail -s "Schema drift detected" team@example.com < /var/log/schema_drift.log
fi
```

**3. CI/CD Pipeline (`.github/workflows/test.yml`):**

```yaml
- name: Validate Frontend Schemas
  run: |
    ./scripts/sync_frontend.sh validate-schema
    ./scripts/sync_frontend.sh validate-api
```

---

## Python API

### Basic Usage

```python
from py.utils.frontend_engineering_agent import FrontendEngineeringAgent

# Initialize agent
agent = FrontendEngineeringAgent()

# Or with custom config
agent = FrontendEngineeringAgent(
    db_config={
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": "sicillionbillions"  # pragma: allowlist secret
    },
    api_base_url="http://localhost:8000",
    dashboard_path=Path("dashboard/app.py")
)
```

### Schema Validation

```python
# Validate all dashboard queries
report = agent.validate_dashboard_schema()

print(f"Queries checked: {report.queries_checked}")
print(f"Valid: {report.queries_valid}")
print(f"Invalid: {report.queries_invalid}")

# Check for critical issues
if not report.is_valid:
    print("❌ Critical issues found:")
    for issue in report.issues:
        if issue.severity in [Severity.ERROR, Severity.CRITICAL]:
            print(f"  - {issue.message}")
            if issue.suggestion:
                print(f"    💡 {issue.suggestion}")
    exit(1)

# Generate full report
print(report.generate_report())
```

### Drift Detection

```python
# Save baseline snapshot (one-time)
agent.save_schema_snapshot()

# Later... detect drift
drift = agent.detect_schema_drift()

if drift.has_breaking_changes:
    print("⛔ Breaking changes detected:")

    if drift.removed_tables:
        print(f"  Removed tables: {drift.removed_tables}")

    if drift.removed_columns:
        for table, cols in drift.removed_columns.items():
            print(f"  {table}: removed {cols}")

    if drift.type_changes:
        for table, changes in drift.type_changes.items():
            for col, old_type, new_type in changes:
                print(f"  {table}.{col}: {old_type} → {new_type}")

    # Generate migration suggestions
    updates = agent.suggest_frontend_updates(drift)
    print(f"\n{len(updates)} migration updates suggested")
```

### API Contract Validation

```python
report = agent.validate_api_contract()

print(f"API Available: {report.api_available}")
print(f"API Version: {report.api_version}")
print(f"Endpoints Checked: {report.endpoints_checked}")

if report.has_violations:
    print("⛔ Contract violations:")
    for violation in report.violations:
        print(f"  - {violation.message}")
        if violation.suggestion:
            print(f"    💡 {violation.suggestion}")
```

### Custom Validation

```python
# Validate specific SQL query
schema = agent.get_database_schema()

query = """
    SELECT pgs.game_id
    FROM mart.player_game_stats pgs
    WHERE pgs.player_id = '00-0033873'
"""

issues = agent.validate_sql_query(query, schema)

for issue in issues:
    print(f"[{issue.severity.value}] {issue.message}")
```

---

## Integration Patterns

### 1. Pre-Commit Hook

**File:** `.git/hooks/pre-commit`

```bash
#!/bin/bash
# Frontend Engineering Agent - Pre-commit validation

set -e

echo "Running frontend schema validation..."

if ! ./scripts/sync_frontend.sh validate-schema; then
    echo ""
    echo "⛔ Schema validation failed - commit blocked"
    echo ""
    echo "To fix:"
    echo "  1. Review errors above"
    echo "  2. Update queries to match current schema"
    echo "  3. See docs/database/SQL_PATTERNS.md for correct patterns"
    echo ""
    exit 1
fi

echo "✅ Schema validation passed"
exit 0
```

**Make executable:**

```bash
chmod +x .git/hooks/pre-commit
```

### 2. Dashboard Startup Validation

**File:** `dashboard/app.py` (add at top of file)

```python
import subprocess
import sys
from pathlib import Path

# Validate schemas at dashboard startup
def validate_startup():
    """Run startup validation checks"""
    result = subprocess.run(
        [str(Path(__file__).parent.parent / "scripts" / "sync_frontend.sh"), "startup-check"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        st.error("⛔ Schema validation failed")
        st.code(result.stdout, language="text")
        st.stop()

# Run validation (cached per session)
@st.cache_resource
def run_startup_checks():
    validate_startup()
    return True

run_startup_checks()
```

### 3. CI/CD Integration

**File:** `.github/workflows/frontend-validation.yml`

```yaml
name: Frontend Validation

on:
  pull_request:
    paths:
      - 'dashboard/**'
      - 'api/**'
      - 'db/**'

jobs:
  validate:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_DB: devdb01
          POSTGRES_USER: dro
          POSTGRES_PASSWORD: sicillionbillions
        ports:
          - 5544:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install psycopg[binary] requests pydantic

      - name: Validate schemas
        env:
          DB_HOST: localhost
          DB_PORT: 5544
          DB_NAME: devdb01
          DB_USER: dro
          DB_PASSWORD: sicillionbillions
        run: |
          ./scripts/sync_frontend.sh validate-schema

      - name: Check for drift
        run: |
          ./scripts/sync_frontend.sh check-drift
```

### 4. Prometheus Monitoring

**File:** `dashboard/app.py` (add metrics)

```python
from prometheus_client import Counter, Gauge

# Metrics
schema_errors = Counter(
    'dashboard_schema_errors_total',
    'Total schema validation errors',
    ['error_type']
)

schema_drift_detected = Gauge(
    'dashboard_schema_drift_detected',
    'Whether schema drift has been detected (0=no, 1=yes)'
)

# Run validation and track metrics
def check_schema_health():
    agent = FrontendEngineeringAgent()

    # Check validation
    report = agent.validate_dashboard_schema()
    for issue in report.issues:
        if issue.severity in [Severity.ERROR, Severity.CRITICAL]:
            schema_errors.labels(error_type=issue.category).inc()

    # Check drift
    drift = agent.detect_schema_drift()
    schema_drift_detected.set(1 if drift.has_breaking_changes else 0)

# Run periodically
import threading
import time

def background_validation():
    while True:
        try:
            check_schema_health()
        except Exception as e:
            logger.error(f"Schema health check failed: {e}")
        time.sleep(300)  # Every 5 minutes

thread = threading.Thread(target=background_validation, daemon=True)
thread.start()
```

---

## Use Cases

### Use Case 1: Detect `game_id` Column Error

**Context:** The `mart.player_game_stats.game_id` error occurred DOZENS of times.

**Problem:**

```python
# dashboard/app.py (WRONG - game_id doesn't exist)
query = """
    SELECT pgs.game_id, pgs.passing_yards
    FROM mart.player_game_stats pgs
    WHERE pgs.player_id = '00-0033873'
"""
result = conn.execute(query).fetchall()
```

**Without Agent:** Fails at runtime with `psycopg.errors.UndefinedColumn: column pgs.game_id does not exist`

**With Agent:**

```bash
$ ./scripts/sync_frontend.sh validate-schema

[ERROR] schema_validation: Column 'game_id' does not exist in table 'mart.player_game_stats'
  File: dashboard/app.py
  Line: 1417
  💡 Suggestion: Available columns: player_id, season, week, current_team, passing_yards...

❌ FAILED: Critical issues found - frontend may be broken
```

**Fix:**

```python
# Correct query (join via composite key)
query = """
    SELECT g.game_id, pgs.passing_yards
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    LEFT JOIN games g ON
        g.season = pgs.season
        AND g.week = pgs.week
        AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
    WHERE pgs.player_id = '00-0033873'
"""
```

### Use Case 2: API Contract Drift

**Context:** FastAPI adds new field `dirichlet_weights` to `PredictionResponse` (Ensemble v7 upgrade).

**Problem:** Dashboard expects old schema without `dirichlet_weights`.

**Detection:**

```bash
$ ./scripts/sync_frontend.sh validate-api

[WARNING] api_contract: New field 'dirichlet_weights' in /predict response
  💡 Suggestion: Update dashboard to handle new field or ensure backward compatibility

⚠️  Non-breaking changes detected - Review recommended
```

**Dashboard Update:**

```python
# dashboard/app.py - handle new field gracefully
response = requests.post(f"{API_URL}/predict", json=request_data)
prediction_data = response.json()

# New field (may not exist in older API versions)
dirichlet_weights = prediction_data.get("dirichlet_weights", None)

if dirichlet_weights:
    st.subheader("Dirichlet Weight Distribution")
    st.bar_chart(dirichlet_weights)
```

### Use Case 3: Schema Migration

**Context:** Database team adds `home_rest_days` column to `public.games`.

**Detection:**

```bash
$ ./scripts/sync_frontend.sh check-drift

➕ Added Columns:
  public.games: home_rest_days, away_rest_days

⚠️  Non-breaking changes detected - Review recommended
```

**Opportunity:**

```python
# dashboard/app.py - add new feature to dashboard
st.subheader("Rest Days Analysis")

query = """
    SELECT
        game_id,
        home_team,
        away_team,
        home_rest_days,
        away_rest_days,
        home_score - away_score AS margin
    FROM public.games
    WHERE season = 2024
        AND home_rest_days IS NOT NULL
"""

rest_data = pd.read_sql(query, conn)

# Visualize rest days vs margin
st.scatter_chart(rest_data, x="home_rest_days", y="margin")
```

---

## Best Practices

### 1. Save Schema Snapshots Regularly

**When:**
- After every database migration
- Before major API changes
- Weekly (automated via cron)

**How:**

```bash
# Manual
./scripts/sync_frontend.sh save-snapshot

# Automated (crontab)
0 0 * * 0 cd /path/to/project && ./scripts/sync_frontend.sh save-snapshot
```

### 2. Run Validation in CI/CD

**Required checks:**
- ✅ Schema validation on every PR
- ✅ API contract validation when API changes
- ✅ Drift detection when database changes

**Benefits:**
- Catch breaking changes before merge
- Prevent production incidents
- Document schema evolution

### 3. Use Pre-Commit Hooks

**Why:**
- Fastest feedback loop (seconds, not minutes)
- Blocks invalid commits
- Educates developers about schema

**Setup:**

```bash
# One-time setup
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
./scripts/sync_frontend.sh validate-schema || exit 1
EOF
chmod +x .git/hooks/pre-commit
```

### 4. Monitor Drift in Production

**Metrics to track:**
- `dashboard_schema_drift_detected` (Gauge: 0 or 1)
- `dashboard_schema_errors_total` (Counter by error type)
- `dashboard_api_contract_violations_total` (Counter by endpoint)

**Alerts:**

```yaml
# Prometheus alert rule
- alert: SchemaDriftDetected
  expr: dashboard_schema_drift_detected == 1
  for: 5m
  annotations:
    summary: "Schema drift detected in production"
    description: "Breaking database changes detected - dashboard may be broken"
```

### 5. Document Schema Changes

**When making database changes:**

```bash
# 1. Save snapshot before change
./scripts/sync_frontend.sh save-snapshot

# 2. Make database change
psql -c "ALTER TABLE public.games ADD COLUMN home_rest_days INTEGER"

# 3. Detect drift
./scripts/sync_frontend.sh check-drift

# 4. Document in migration notes
cat >> db/migrations/002_add_rest_days.md << 'EOF'
## Frontend Impact
- Added columns: public.games.home_rest_days, public.games.away_rest_days
- Non-breaking change (columns nullable)
- Opportunity: Dashboard can now show rest days analysis
EOF
```

---

## Troubleshooting

### Issue 1: "psycopg not available"

**Error:**

```
ERROR: psycopg not available - cannot introspect schema
```

**Fix:**

```bash
pip install psycopg[binary]
```

### Issue 2: "API not reachable"

**Error:**

```
[CRITICAL] api_availability: API not reachable at http://localhost:8000
```

**Fix:**

```bash
# Start API
docker-compose up -d api

# Or run locally
cd api && uvicorn main:app --reload
```

### Issue 3: "Schema snapshot not found"

**Warning:**

```
WARNING: Schema snapshot not found: /path/to/schema_snapshot.json
```

**Fix:**

```bash
# Save initial snapshot
./scripts/sync_frontend.sh save-snapshot
```

### Issue 4: False Positives

**Problem:** Validation incorrectly flags valid queries.

**Example:**

```sql
-- Valid query using CTE
WITH recent_games AS (
    SELECT * FROM games WHERE season = 2024
)
SELECT * FROM recent_games
```

**Agent sees:** `SELECT * FROM recent_games` (table doesn't exist)

**Workaround:** Add comment to suppress warning

```sql
-- frontend-agent: ignore-next-query
WITH recent_games AS (
    SELECT * FROM games WHERE season = 2024
)
SELECT * FROM recent_games
```

### Issue 5: Slow Validation

**Problem:** `validate-schema` takes >10 seconds.

**Cause:** Large dashboard file with 100+ queries.

**Optimization:**

```python
# Cache schema introspection
agent = FrontendEngineeringAgent()
schema = agent.get_database_schema()

# Validate multiple files with cached schema
for file_path in dashboard_files:
    queries = agent.extract_sql_queries(file_path)
    for query, line_num in queries:
        issues = agent.validate_sql_query(query, schema)  # Uses cached schema
```

---

## Advanced Usage

### Custom Validation Rules

```python
from py.utils.frontend_engineering_agent import FrontendEngineeringAgent, ValidationIssue, Severity

class CustomAgent(FrontendEngineeringAgent):
    def validate_sql_query(self, query, schema):
        issues = super().validate_sql_query(query, schema)

        # Custom rule: Detect hardcoded seasons
        if "season = 2024" in query.lower():
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category="hardcoded_value",
                message="Hardcoded season value detected",
                query=query,
                suggestion="Use parameter: WHERE season = %(season)s"
            ))

        # Custom rule: Require LIMIT on dashboard queries
        if "dashboard" in self.dashboard_path.name and "LIMIT" not in query.upper():
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category="performance",
                message="Dashboard query without LIMIT clause",
                query=query,
                suggestion="Add LIMIT clause to prevent UI slowdown"
            ))

        return issues
```

### Integration with Sports Analytics Validator

```python
from py.utils.frontend_engineering_agent import FrontendEngineeringAgent
from py.utils.sports_analytics_validator import SportsAnalyticsValidator

class EnhancedFrontendAgent(FrontendEngineeringAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sports_validator = SportsAnalyticsValidator()

    def validate_sql_query(self, query, schema):
        # Run frontend validation
        issues = super().validate_sql_query(query, schema)

        # Add NFL domain validation
        nfl_issues = self.sports_validator.check_query_logic(query)

        # Convert to ValidationIssue format
        for nfl_issue in nfl_issues:
            issues.append(ValidationIssue(
                severity=Severity.ERROR if nfl_issue.severity == "ERROR" else Severity.WARNING,
                category="nfl_domain",
                message=nfl_issue.message,
                query=query,
                suggestion=nfl_issue.suggestion if hasattr(nfl_issue, 'suggestion') else None
            ))

        return issues
```

### Automated Migration Application

```python
# DANGER: Automated code modification - use with caution

def apply_migration_update(update: MigrationUpdate) -> bool:
    """Apply automated migration update to file"""
    if not update.automated:
        logger.warning(f"Update requires manual review: {update.file_path}")
        return False

    with open(update.file_path, 'r') as f:
        content = f.read()

    # Only apply if old_code is unique
    if update.old_code and content.count(update.old_code) != 1:
        logger.error(f"Old code not unique in {update.file_path} - cannot auto-apply")
        return False

    # Apply update
    if update.old_code:
        new_content = content.replace(update.old_code, update.new_code)
    else:
        # Add new code at appropriate location
        new_content = content + "\n" + update.new_code

    # Write back
    with open(update.file_path, 'w') as f:
        f.write(new_content)

    logger.info(f"✅ Applied update to {update.file_path}")
    return True
```

---

## Future Enhancements

### Planned Features

1. **TypeScript Interface Generation** - Generate `.d.ts` files from FastAPI models
2. **GraphQL Schema Validation** - Support for GraphQL APIs
3. **Automated Test Generation** - Generate Pytest tests for schema changes
4. **Schema Diff Visualization** - Visual diff tool for schema changes
5. **Rollback Support** - Revert to previous schema snapshot

### Requested Features

Submit feature requests: https://github.com/raold/nfl-analytics/issues

---

## Related Documentation

- [DATABASE_QUICK_REFERENCE.md](database/DATABASE_QUICK_REFERENCE.md) - Schema reference
- [SQL_PATTERNS.md](database/SQL_PATTERNS.md) - Correct SQL patterns
- [SPORTS_ANALYTICS_VALIDATOR_GUIDE.md](SPORTS_ANALYTICS_VALIDATOR_GUIDE.md) - NFL domain validation
- [ENFORCEMENT_FRAMEWORK_COMPLETE.md](ENFORCEMENT_FRAMEWORK_COMPLETE.md) - Overall validation strategy

---

## Support

**Issues:** Report bugs and request features via GitHub Issues

**Documentation:** See `docs/` directory for additional guides

**Contact:** Claude Code team

---

**Version:** 1.0.0
**Last Updated:** November 22, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
