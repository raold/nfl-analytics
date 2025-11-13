# Data Layer Analyzer - Column Lineage and Impact Analysis

**Purpose:** Track column usage across codebase to prevent breaking changes, detect dead code, and identify feature opportunities.

**Status:** ✅ Complete (November 13, 2025)

**Author:** Claude Code (Sonnet 4.5)

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Matters](#why-this-matters)
3. [Quick Start](#quick-start)
4. [Core Capabilities](#core-capabilities)
5. [CLI Reference](#cli-reference)
6. [Python API](#python-api)
7. [Database Schema](#database-schema)
8. [Integration with Agent Coordinator](#integration-with-agent-coordinator)
9. [Real-World Examples](#real-world-examples)
10. [Best Practices](#best-practices)

---

## Overview

The Data Layer Analyzer is a static analysis tool that tracks which database columns are used where in your codebase. It enables:

- **Impact Analysis:** Know what breaks before you change a schema
- **Dead Column Detection:** Find unused columns wasting storage
- **Feature Opportunities:** Discover which models could use new data
- **Dependency Graphs:** Visualize column → consumer relationships
- **Migration Validation:** Block breaking changes automatically

**Key Stats:**
- Scans Python, R, and SQL files
- Tracks 56 tables across 5 schemas
- Handles ~351 files with SQL queries
- Identifies 830+ column references across 161 Python files

---

## Why This Matters

### The Problem

In a complex codebase like NFL Analytics:
- **56 tables** across 5 schemas (public, mart, ml, staging, metadata)
- **Multi-language** (Python 68%, R 7%, SQL 3%, LaTeX 19%)
- **Real money at stake** - Silent failures = Lost bets

**Without lineage tracking:**
- Add `weather.humidity` → No idea which models could benefit
- Remove `injuries.report_status` → Silently break 3 models
- Refactor feature engineering → Miss downstream consumers in R scripts

**Lesson learned (from CLAUDE.md):**
> "Dead code can break models - remove unused variables from queries and feature engineering"

### The Solution

Data Layer Analyzer provides:
1. **Forward impact analysis:** "I'm adding column X - who should use it?"
2. **Backward impact analysis:** "I'm removing column Y - what breaks?"
3. **Continuous validation:** Pre-commit hooks block breaking migrations

---

## Quick Start

### Installation

No installation needed - already in your project:

```bash
# Located at:
py/utils/data_layer_analyzer.py     # Core analyzer (770 lines)
scripts/analyze_data_layer.sh       # CLI wrapper
db/migrations/043_column_lineage_tracking.sql  # Database schema
```

### First Scan

```bash
# Scan entire codebase (takes ~30 seconds)
./scripts/analyze_data_layer.sh scan

# Output:
# 🔍 Scanning codebase...
# INFO:DataLayerAnalyzer:Scanning 351 Python files...
# INFO:DataLayerAnalyzer:Scanning 85 R files...
# INFO:DataLayerAnalyzer:Scanning 47 SQL files...
# ✅ Scan complete. Found 267 unique columns.
# 📁 Lineage cached to: .cache/column_lineage.json
```

Cache lasts 24 hours. Re-scan anytime with `./scripts/analyze_data_layer.sh scan`.

### Basic Usage

```bash
# Trace where a column is used
./scripts/analyze_data_layer.sh trace plays.passing_yards

# Analyze impact of removing a column
./scripts/analyze_data_layer.sh impact injuries.report_status remove

# Find opportunities for a new column
./scripts/analyze_data_layer.sh opportunities weather.humidity

# Generate dependency graph
./scripts/analyze_data_layer.sh graph graphviz > deps.dot
dot -Tpng deps.dot -o deps.png  # Visualize with Graphviz
```

---

## Core Capabilities

### 1. Column Lineage Tracking

**What:** Build reverse index of column → consumers

**How it works:**
- Parses Python files for SQL queries (`read_sql`, `execute`, triple-quoted strings)
- Parses R files for database calls (`dbGetQuery`, `dbReadTable`)
- Parses raw SQL files (migrations, queries, views)
- Extracts `table.column` references using regex
- Filters to known tables (56 tables tracked)

**Output:**
```json
{
  "plays.passing_yards": [
    {
      "table": "plays",
      "column": "passing_yards",
      "file": "py/features/player_features.py",
      "line": 45,
      "usage_type": "select",
      "consumer_type": "feature",
      "context": "SELECT passing_yards FROM plays WHERE..."
    },
    ...
  ]
}
```

### 2. Impact Analysis

**What:** Predict consequences of schema changes before deployment

**Supported change types:**
- `add` - Adding new column (identifies opportunities)
- `modify` - Changing column type/constraints (flags potential breakage)
- `remove` - Dropping column (blocks if breaking changes)

**Example:**
```bash
./scripts/analyze_data_layer.sh impact injuries.report_status remove

# Output:
# 📊 Impact Analysis: injuries.report_status (remove)
# Risk Level: CRITICAL
# Breaking Changes: 1
# Affected Models: 1
# Affected Features: 1
# Estimated Integration Effort: 0.5 hours
#
# ⛔ BREAKING CHANGES:
#   - R/features/injury_load_features.R:36 (feature)
#
# 🤖 AFFECTED MODELS (1):
#   - R/train_hierarchical_v2_production.R
#
# ⚠️  WARNING: This change has breaking changes!
# ⛔ Migration blocked due to breaking changes
```

**Risk Levels:**
- `low` - No usages, safe to change
- `medium` - 1-5 usages, manageable
- `high` - 6-20 usages, significant effort
- `critical` - 20+ usages or production impact

### 3. Dead Column Detection

**What:** Find columns that exist in schema but are never used

**Use case:**
```bash
./scripts/analyze_data_layer.sh unused

# Output:
# Total unique columns tracked: 267
#
# Top 10 most-used columns:
#   plays.passing_yards: 47 usages
#   games.spread_close: 38 usages
#   injuries.report_status: 12 usages
#   weather.temperature: 9 usages
#   ...
```

**Database view:**
```sql
-- View unused columns with storage cost
SELECT * FROM metadata.unused_columns;

-- View dead columns sorted by size
SELECT * FROM metadata.dead_columns_by_size LIMIT 20;
```

### 4. Feature Opportunity Detection

**What:** Suggest which code could benefit from a new column

**Heuristic:** Files that already use the same table + semantic keyword matching

**Example:**
```bash
./scripts/analyze_data_layer.sh opportunities weather.humidity

# Output:
# 💡 Finding opportunities for: weather.humidity
#
# Found 3 potential consumers:
#   - py/models/weather_totals_model.py
#   - py/features/environmental_contextual_features.py
#   - R/features/nflseedR_features.R
```

### 5. Dependency Graph Generation

**What:** Visualize column → consumer relationships

**Formats:**
- `graphviz` - DOT format for Graphviz rendering
- `mermaid` - Mermaid.js format for GitHub/documentation
- `json` - Structured data for custom visualization

**Example:**
```bash
# Generate Graphviz graph
./scripts/analyze_data_layer.sh graph graphviz > column_deps.dot

# Render to PNG
dot -Tpng column_deps.dot -o column_deps.png
open column_deps.png

# Generate Mermaid graph for GitHub
./scripts/analyze_data_layer.sh graph mermaid > column_deps.mmd
```

### 6. Migration Validation

**What:** Validate migration files before applying to database

**Example:**
```bash
./scripts/analyze_data_layer.sh validate db/migrations/042_add_weather_humidity.sql

# Output:
# 🔍 Validating migration: 042_add_weather_humidity.sql
#
# 📋 Analyzing ADD COLUMN statements...
# Checking: weather.humidity
# 📊 Impact Analysis: weather.humidity (add)
# Risk Level: LOW
# Opportunities: 3
#
# ✅ Migration validation passed
```

**Pre-commit hook integration:**
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: validate-schema-changes
      name: Data Layer Impact Analysis
      entry: ./scripts/analyze_data_layer.sh validate
      language: system
      files: ^db/migrations/.*\.sql$
```

---

## CLI Reference

### `scan`
```bash
./scripts/analyze_data_layer.sh scan
```
Scan entire codebase and build column lineage. Cache valid for 24 hours.

### `trace <column>`
```bash
./scripts/analyze_data_layer.sh trace plays.passing_yards
```
Show all usages of a specific column. Groups by consumer type (feature, model, backtest, etc.).

### `impact <column> <change_type>`
```bash
./scripts/analyze_data_layer.sh impact weather.humidity add
./scripts/analyze_data_layer.sh impact injuries.report_status remove
```
Analyze impact of schema change. Change types: `add`, `modify`, `remove`.

**Exit codes:**
- `0` - No breaking changes
- `1` - Breaking changes detected (blocks migration)

### `unused`
```bash
./scripts/analyze_data_layer.sh unused
```
Show column usage summary. Requires database schema for accurate dead column detection.

### `opportunities <column>`
```bash
./scripts/analyze_data_layer.sh opportunities weather.humidity
```
Find code that could benefit from a new column.

### `graph [format]`
```bash
./scripts/analyze_data_layer.sh graph graphviz > deps.dot
./scripts/analyze_data_layer.sh graph mermaid > deps.mmd
./scripts/analyze_data_layer.sh graph json > deps.json
```
Generate dependency graph. Formats: `graphviz`, `mermaid`, `json`.

### `validate <migration_file>`
```bash
./scripts/analyze_data_layer.sh validate db/migrations/042_add_weather_humidity.sql
```
Validate migration file for breaking changes. Blocks deployment if breaking.

### `clear-cache`
```bash
./scripts/analyze_data_layer.sh clear-cache
```
Clear cached lineage data. Forces re-scan on next command.

---

## Python API

### Basic Usage

```python
from py.utils.data_layer_analyzer import DataLayerAnalyzer

# Initialize analyzer
analyzer = DataLayerAnalyzer(project_root="/Users/dro/rice/nfl-analytics")

# Scan codebase
analyzer.scan_codebase()

# Save lineage to cache
analyzer.save_lineage('.cache/column_lineage.json')
```

### Trace Column Usage

```python
# Trace a column
usages = analyzer.trace_column('plays.passing_yards')

print(f"Found {len(usages)} usages")

for usage in usages:
    print(f"{usage.file_path}:{usage.line_number}")
    print(f"  Type: {usage.consumer_type}")
    print(f"  Usage: {usage.usage_type}")
    print(f"  Context: {usage.context}")
```

### Impact Analysis

```python
# Analyze impact of removing a column
report = analyzer.analyze_impact('injuries.report_status', 'remove')

print(report.summary())

if report.has_breaking_changes():
    print("⛔ BLOCKING MIGRATION")
    for change in report.breaking_changes:
        print(f"  - {change.file_path}:{change.line_number}")
    exit(1)
else:
    print("✅ Safe to proceed")
```

### Feature Opportunities

```python
# Find opportunities for new column
report = analyzer.analyze_impact('weather.humidity', 'add')

print(f"Found {len(report.opportunities)} opportunities:")
for opp in report.opportunities:
    print(f"  - {opp}")
```

### Dependency Graph

```python
# Generate dependency graph
graph_dot = analyzer.generate_dependency_graph('graphviz')
graph_mermaid = analyzer.generate_dependency_graph('mermaid')
graph_json = analyzer.generate_dependency_graph('json')

# Save to file
with open('column_deps.dot', 'w') as f:
    f.write(graph_dot)
```

### Load from Cache

```python
# Load lineage from cache (faster than re-scanning)
analyzer.load_lineage('.cache/column_lineage.json')

# Now you can run queries without re-scanning
usages = analyzer.trace_column('plays.passing_yards')
```

---

## Database Schema

### Migration: `043_column_lineage_tracking.sql`

**Tables:**

#### `metadata.column_lineage`
Stores column usage tracking data.

```sql
CREATE TABLE metadata.column_lineage (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    consumer_file TEXT NOT NULL,
    consumer_line INTEGER NOT NULL,
    consumer_type TEXT NOT NULL,  -- feature, model, backtest, validation, production
    usage_type TEXT NOT NULL,     -- select, insert, update, filter, groupby, orderby
    context TEXT,
    last_validated TIMESTAMP DEFAULT NOW(),
    UNIQUE (table_name, column_name, consumer_file, consumer_line)
);
```

#### `metadata.column_change_history`
Audit log of schema changes and their impact.

```sql
CREATE TABLE metadata.column_change_history (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    change_type TEXT NOT NULL,  -- add, modify, remove, rename
    migration_file TEXT,
    impact_analysis JSONB,      -- Store ImpactReport as JSON
    executed_at TIMESTAMP DEFAULT NOW()
);
```

**Views:**

- `metadata.unused_columns` - Columns never referenced in code
- `metadata.column_usage_summary` - Usage counts by column
- `metadata.high_impact_columns` - Columns used in many places (high risk)
- `metadata.dead_columns_by_size` - Unused columns sorted by storage cost

**Functions:**

- `metadata.refresh_column_lineage()` - Update validation timestamps

**Example Queries:**

```sql
-- Find high-impact columns (>10 usages)
SELECT * FROM metadata.high_impact_columns;

-- Find unused columns wasting storage
SELECT * FROM metadata.dead_columns_by_size LIMIT 20;

-- Check usage of specific column
SELECT * FROM metadata.column_usage_summary
WHERE table_name = 'plays' AND column_name = 'passing_yards';

-- Audit recent schema changes
SELECT * FROM metadata.column_change_history
WHERE executed_at > NOW() - INTERVAL '30 days'
ORDER BY executed_at DESC;
```

---

## Integration with Agent Coordinator

The Data Layer Analyzer is integrated into the **Data Layer Agent** in the Agent Coordinator.

**Updated capabilities:**

```python
AgentType.DATA_LAYER: AgentCapability(
    responsibilities=[
        # ... existing responsibilities ...
        "Track column usage across codebase",
        "Build column lineage (column → consumers)",
        "Analyze impact of schema changes (add/modify/remove)",
        "Detect unused/dead columns",
        "Identify feature opportunities for new columns",
        "Validate migrations before deployment",
        "Generate dependency graphs (column dependencies)",
        "Prevent breaking changes in production",
    ],
    keywords={
        # ... existing keywords ...
        "column lineage", "impact analysis", "schema change", "breaking changes",
        "unused columns", "dead columns", "column usage", "column tracking",
        "dependency graph", "column dependencies", "feature opportunities",
        "add column", "remove column", "drop column", "alter column",
        "migration validation", "trace column", "column consumers",
    }
)
```

**Usage with coordinator:**

```python
from py.utils.agent_coordinator import get_coordinator

coordinator = get_coordinator()

# Coordinator will route to Data Layer Agent
routing = coordinator.analyze_task(
    "Analyze impact of removing injuries.report_status column",
    files=["db/migrations/042_drop_injury_columns.sql"]
)

print(f"Primary agent: {routing.primary_agent.value}")
# Output: data_layer
```

---

## Real-World Examples

### Example 1: Adding Weather Humidity

**Scenario:** You want to add a new `weather.humidity` column to improve totals predictions.

**Workflow:**

```bash
# 1. Create migration
cat > db/migrations/042_add_weather_humidity.sql << 'EOF'
ALTER TABLE weather ADD COLUMN humidity NUMERIC;
EOF

# 2. Validate migration (no breaking changes expected)
./scripts/analyze_data_layer.sh validate db/migrations/042_add_weather_humidity.sql
# ✅ Migration validation passed

# 3. Identify feature opportunities
./scripts/analyze_data_layer.sh opportunities weather.humidity
# 💡 Found 3 potential consumers:
#   - py/models/weather_totals_model.py
#   - py/features/environmental_contextual_features.py
#   - R/features/nflseedR_features.R

# 4. Apply migration
psql -f db/migrations/042_add_weather_humidity.sql

# 5. Update feature engineering (guided by opportunities)
# Edit py/features/environmental_contextual_features.py
# Add humidity features

# 6. Re-scan to update lineage
./scripts/analyze_data_layer.sh scan
```

### Example 2: Removing Legacy Column

**Scenario:** You think `injuries.report_primary` is unused and want to remove it.

**Workflow:**

```bash
# 1. Trace column usage
./scripts/analyze_data_layer.sh trace injuries.report_primary
# Found 1 usage:
#   - R/features/injury_load_features.R:36 (feature)

# 2. Analyze impact of removal
./scripts/analyze_data_layer.sh impact injuries.report_primary remove
# ⛔ BREAKING CHANGES:
#   - R/features/injury_load_features.R:36
# 🤖 AFFECTED MODELS:
#   - R/train_hierarchical_v2_production.R (25% ensemble weight)
#
# ⚠️  WARNING: This change has breaking changes!
# ⛔ Migration blocked due to breaking changes

# 3. Decision: Refactor code first
# Edit R/features/injury_load_features.R
# Remove dependency on injuries.report_primary

# 4. Re-scan to verify
./scripts/analyze_data_layer.sh scan
./scripts/analyze_data_layer.sh trace injuries.report_primary
# No usages found

# 5. Now safe to remove
./scripts/analyze_data_layer.sh impact injuries.report_primary remove
# ✅ No breaking changes detected

# 6. Create and apply migration
cat > db/migrations/043_drop_legacy_columns.sql << 'EOF'
ALTER TABLE injuries DROP COLUMN report_primary;
EOF

psql -f db/migrations/043_drop_legacy_columns.sql
```

### Example 3: Pre-Commit Hook

**Scenario:** Prevent developers from creating breaking migrations.

**Setup:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-schema-changes
        name: Validate Schema Changes
        entry: ./scripts/analyze_data_layer.sh validate
        language: system
        files: ^db/migrations/.*\.sql$
        pass_filenames: true
```

**Install hook:**

```bash
pre-commit install
```

**Behavior:**

```bash
# Developer creates breaking migration
cat > db/migrations/044_drop_spread_close.sql << 'EOF'
ALTER TABLE games DROP COLUMN spread_close;
EOF

git add db/migrations/044_drop_spread_close.sql
git commit -m "Remove spread_close column"

# Pre-commit hook runs:
# 🔍 Validating migration: 044_drop_spread_close.sql
# ⛔ BREAKING CHANGES:
#   - py/features/vegas_line_bias_features.py:16 (feature)
#   - py/ensemble/enhanced_ensemble_v5.py:234 (model)
#   ... 38 more usages
#
# ⛔ Migration validation failed - breaking changes detected
#
# [ERROR] Commit aborted
```

---

## Best Practices

### 1. Run Daily Scans

```bash
# Add to cron or daily workflow
0 2 * * * cd /Users/dro/rice/nfl-analytics && ./scripts/analyze_data_layer.sh scan
```

Cache stays fresh, lineage always up-to-date.

### 2. Validate Before Every Migration

```bash
# ALWAYS validate migrations before applying
./scripts/analyze_data_layer.sh validate db/migrations/XXX_your_migration.sql

if [[ $? -eq 0 ]]; then
    psql -f db/migrations/XXX_your_migration.sql
else
    echo "⛔ Migration blocked - fix breaking changes first"
    exit 1
fi
```

### 3. Check Opportunities for New Columns

```bash
# Before adding a column, check where it should be used
./scripts/analyze_data_layer.sh opportunities weather.humidity

# Create GitHub issues for each opportunity
for file in $(./scripts/analyze_data_layer.sh opportunities weather.humidity | grep "py/\|R/"); do
    echo "TODO: Integrate weather.humidity into $file"
done
```

### 4. Audit Dead Columns Quarterly

```bash
# Generate report of unused columns
./scripts/analyze_data_layer.sh unused > unused_columns_$(date +%Y%m%d).txt

# Review with team - remove if truly unused
# Saves storage and improves query performance
```

### 5. Use Impact Analysis in Code Reviews

When reviewing a PR that touches the database:

```bash
# For each changed column, run impact analysis
git diff main --name-only | grep "db/migrations" | while read file; do
    ./scripts/analyze_data_layer.sh validate "$file"
done
```

### 6. Document High-Impact Columns

```sql
-- Query high-impact columns
SELECT * FROM metadata.high_impact_columns;

-- Add comments to protect critical columns
COMMENT ON COLUMN games.spread_close IS
    'CRITICAL: Used in 38 files across ensemble. Coordinate changes with team.';
```

### 7. Integrate with CI/CD

```yaml
# .github/workflows/validate-schema.yml
name: Validate Schema Changes

on:
  pull_request:
    paths:
      - 'db/migrations/*.sql'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Scan codebase
        run: ./scripts/analyze_data_layer.sh scan
      - name: Validate migrations
        run: |
          for file in db/migrations/*.sql; do
            ./scripts/analyze_data_layer.sh validate "$file" || exit 1
          done
```

---

## Limitations and Future Work

### Current Limitations

1. **SQL Parsing:** Regex-based, may miss complex dynamic SQL
2. **Alias Detection:** Limited ability to resolve table aliases
3. **Cross-Language Tracking:** R/Python interop not fully tracked
4. **Schema Inference:** Requires manual table list (56 known tables)

### Future Enhancements

1. **Full SQL Parser:** Use `sqlparse` or `sqlglot` for robust parsing
2. **Database Schema Integration:** Auto-discover tables from PostgreSQL
3. **Cross-File Analysis:** Track data flow across file boundaries
4. **ML-Based Suggestions:** Use embeddings to suggest better opportunities
5. **Real-Time Monitoring:** Watch file changes and update lineage automatically
6. **Visual Dashboard:** Interactive dependency graph visualization

---

## Related Documentation

- [Database Schema Audit](DATABASE_SCHEMA_AUDIT.md) - Complete schema reference
- [Data Leakage Detector](TEMPORAL_LEAKAGE_PROTECTION.md) - Temporal validation
- [Agent Coordinator](AGENT_COORDINATOR_GUIDE.md) - Multi-agent coordination
- [Phase 2+3 Integration](PHASE_2_3_INTEGRATION_COMPLETE.md) - Agent enhancements

---

## Troubleshooting

### Cache Not Found

```bash
# If you see "Cache missing or stale"
./scripts/analyze_data_layer.sh scan  # Re-scan codebase
```

### False Positives

If the analyzer reports usages that don't exist:

```bash
# Inspect the context
./scripts/analyze_data_layer.sh trace your.column | grep "context"

# If false positive is in a comment or string literal, consider:
# 1. Improving regex patterns in data_layer_analyzer.py
# 2. Excluding specific file patterns
```

### Performance Issues

Large codebases may take time to scan:

```bash
# Check scan progress
./scripts/analyze_data_layer.sh scan 2>&1 | tee scan.log

# Reduce scope by excluding directories
# Edit py/utils/data_layer_analyzer.py:
self.python_dirs = ['py/', 'scripts/']  # Remove tests/ if slow
```

### Missing Table

If a table isn't tracked:

```python
# Edit py/utils/data_layer_analyzer.py
self.known_tables = {
    # ... existing tables ...
    'your_new_table',  # Add here
}
```

---

## Support

For issues or questions:
- GitHub Issues: [nfl-analytics/issues](https://github.com/raold/nfl-analytics/issues)
- Documentation: `docs/`
- Code: `py/utils/data_layer_analyzer.py`

**Version:** 1.0
**Last Updated:** November 13, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
