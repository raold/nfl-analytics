# Sports Analytics Validator Guide

**Last Updated:** November 14, 2025
**Status:** Production Ready
**Priority:** P1 HIGH (prevents recurring schema bugs)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Validation Categories](#validation-categories)
4. [Usage Examples](#usage-examples)
5. [Integration Patterns](#integration-patterns)
6. [NFL Domain Rules](#nfl-domain-rules)
7. [API Reference](#api-reference)
8. [CLI Reference](#cli-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

The Sports Analytics Validator enforces NFL domain-specific rules to prevent logic violations that cause production bugs. Created to address recurring SQL schema errors documented in `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md`.

### Problem Solved

**Before Validator:**
- `game_id` column errors occurred "DOZENS of times"
- Ambiguous "LA" team abbreviations
- Invalid week numbers (Week 25 in queries)
- QB with 300 rushing yards (impossible)
- Team plays 20 games in season (should be 17)

**After Validator:**
- Catch errors at development time
- Block invalid SQL queries before execution
- Validate feature DataFrames before model training
- Prevent impossible stat combinations

### Key Features

1. **Position Eligibility** - QB/RB/WR/TE stat validation
2. **Stat Range Validation** - NFL physical limits (passing: 0-600 yards)
3. **Schedule Constraints** - Week 1-18, 17 games per team
4. **Team Abbreviations** - LAR/LAC enforcement, relocation tracking
5. **Game Result Consistency** - Score validation, spread checks
6. **SQL Query Logic** - Detects schema errors before execution

---

## Quick Start

### Installation

The validator is already included in the project. No installation needed.

**Files:**
- `py/utils/sports_analytics_validator.py` (891 lines)
- `scripts/validate_sports_analytics.sh` (CLI wrapper)
- `tests/unit/test_sports_analytics_validator.py` (comprehensive tests)

### Basic Usage

**Python API:**
```python
from py.utils.sports_analytics_validator import SportsAnalyticsValidator
import pandas as pd

# Create validator
validator = SportsAnalyticsValidator()

# Validate DataFrame
df = pd.read_csv('features.csv')
report = validator.validate_dataframe(df)

# Check for errors
if report.has_errors():
    print(report.generate_report())
    exit(1)  # Block production

# Check SQL query
issues = validator.check_query_logic("SELECT * FROM games WHERE week = 25")
for issue in issues:
    print(f"[{issue.severity.value}] {issue.message}")
```

**CLI:**
```bash
# Validate DataFrame
./scripts/validate_sports_analytics.sh --validate-df data/features.csv

# Check SQL query
./scripts/validate_sports_analytics.sh --check-query query.sql

# Generate report
./scripts/validate_sports_analytics.sh --report
```

---

## Validation Categories

### 1. Position Eligibility

**Rules:**
- QB: Can have `passing` and `rushing` stats (scrambles allowed)
- RB: Can have `rushing` and `receiving` stats
- WR: Can have `receiving` and `rushing` stats (jet sweeps)
- TE: Can have `receiving` stats only
- K/P: No yardage stats

**Special Cases:**
- QB rushing yards >200 is ERROR (historical max ~200, Michael Vick)
- RB with passing stats is ERROR (RBs don't throw)
- WR with <50 rushing yards is OK (reverses, jet sweeps)

**Example Violations:**
```python
# ERROR: RB with passing stats
df = pd.DataFrame([{
    'player_position': 'RB',
    'stat_category': 'passing',
    'stat_yards': 100
}])
# Result: ERROR - RB cannot have passing stats

# ERROR: QB with excessive rushing
df = pd.DataFrame([{
    'player_position': 'QB',
    'stat_category': 'rushing',
    'stat_yards': 250
}])
# Result: ERROR - QB with 250 rushing yards is impossible
```

### 2. Stat Range Validation

**Physical Limits (based on historical maximums + 10% buffer):**

| Stat Category | Yards | Attempts | Touchdowns |
|--------------|-------|----------|------------|
| Passing | 0-600 | 0-80 | 0-8 |
| Rushing | 0-300 | 0-50 | 0-6 |
| Receiving | 0-350 | 0-30 targets | 0-5 |

**Historical Context:**
- **Passing yards record**: 554 yards (Norm Van Brocklin, 1951)
- **Rushing yards record**: 296 yards (Adrian Peterson, 2007)
- **Receiving yards record**: 336 yards (Willie Anderson, 1989)

**Example Violations:**
```python
# ERROR: Passing yards over 600
df = pd.DataFrame([{
    'stat_category': 'passing',
    'stat_yards': 650
}])
# Result: ERROR - Passing yards out of range: 650 (max: 600)

# WARNING: High but plausible yards
df = pd.DataFrame([{
    'stat_category': 'passing',
    'stat_yards': 550
}])
# Result: WARNING - Unusually high but historically possible
```

### 3. Schedule Constraints

**Rules:**
- Regular season weeks: 1-18
- Playoff weeks: 19-22 (Wild Card through Super Bowl)
- Games per team: 17 (2021+), 16 (pre-2021)
- Bye weeks: 1 per team per season
- No team plays twice in same week

**Example Violations:**
```python
# ERROR: Invalid week
df = pd.DataFrame([{'week': 25, 'season': 2024}])
# Result: ERROR - Invalid week number: 25 (must be 1-22)

# ERROR: Too many games
# Team plays 25 games in 2024 season
df = pd.DataFrame([
    {'season': 2024, 'week': i % 18 + 1, 'home_team': 'KC'}
    for i in range(25)
])
# Result: ERROR - Team KC has 25 games in 2024 (expected: 17 regular + ~4 playoffs)
```

### 4. Team Abbreviation Validation

**32 Current Teams:**
```
ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN,
DET, GB, HOU, IND, JAX, KC, LAC, LAR, LV, MIA,
MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB,
TEN, WAS
```

**Historical Relocations:**
- **SD → LAC** (2017): San Diego Chargers relocated to Los Angeles
- **STL → LAR** (2016): St. Louis Rams relocated to Los Angeles
- **OAK → LV** (2020): Oakland Raiders relocated to Las Vegas

**Special Case: "LA" is AMBIGUOUS**
- ❌ WRONG: `home_team = 'LA'`
- ✅ CORRECT: `home_team = 'LAR'` (LA Rams) or `home_team = 'LAC'` (LA Chargers)

**Example Violations:**
```python
# ERROR: Ambiguous LA
df = pd.DataFrame([{'home_team': 'LA', 'season': 2024}])
# Result: ERROR - Ambiguous team abbreviation: 'LA' (use LAR or LAC)

# ERROR: Outdated SD in 2024
df = pd.DataFrame([{'home_team': 'SD', 'season': 2024}])
# Result: ERROR - Outdated team abbreviation: 'SD' in 2024 (relocated to LAC in 2017)

# OK: SD in 2016 (before relocation)
df = pd.DataFrame([{'home_team': 'SD', 'season': 2016}])
# Result: PASS - SD valid in 2016
```

### 5. Game Result Consistency

**Rules:**
- Scores: 0-70 (historical max: 73 in 1940)
- Total points: <150 (conservative upper bound)
- Negative scores: ERROR
- Spread consistency: INFO if >30 point discrepancy

**Example Violations:**
```python
# ERROR: Negative score
df = pd.DataFrame([{'home_score': -7, 'away_score': 14}])
# Result: ERROR - Negative home score: -7

# WARNING: Unusually high score
df = pd.DataFrame([{'home_score': 75, 'away_score': 14}])
# Result: WARNING - Unusually high home score: 75 (historical max: 73)

# ERROR: Excessive total points
df = pd.DataFrame([{'home_score': 80, 'away_score': 75}])
# Result: ERROR - Total points exceeds maximum: 155
```

---

## Usage Examples

### Pre-Training Validation

**ALWAYS validate data before model training:**

```python
from py.utils.sports_analytics_validator import SportsAnalyticsValidator
import pandas as pd

# Load feature data
train_df = pd.read_csv('data/features/train.csv')
test_df = pd.read_csv('data/features/test.csv')

# Create validator
validator = SportsAnalyticsValidator()

# Validate training data
report = validator.validate_dataframe(train_df, dataset_name='train.csv')

if report.has_errors():
    print("=" * 80)
    print("TRAINING DATA VALIDATION FAILED")
    print("=" * 80)
    print(report.generate_report())
    exit(1)  # BLOCK TRAINING

# Validate test data
report = validator.validate_dataframe(test_df, dataset_name='test.csv')

if report.has_errors():
    print("TEST DATA VALIDATION FAILED")
    print(report.generate_report())
    exit(1)

# Safe to proceed with training
print("✅ Data validation passed - safe to train")
model.fit(X_train, y_train)
```

### SQL Query Validation

**Check queries before execution:**

```python
from py.utils.sports_analytics_validator import SportsAnalyticsValidator

# Load SQL query
with open('queries/player_stats.sql', 'r') as f:
    query = f.read()

# Validate query logic
validator = SportsAnalyticsValidator()
issues = validator.check_query_logic(query)

# Check for errors
errors = [i for i in issues if i.severity.value == 'ERROR']

if errors:
    print("⛔ QUERY VALIDATION FAILED")
    for error in errors:
        print(f"  - {error.message}")
        print(f"    {error.details}")
    exit(1)

# Safe to execute
import psycopg2
conn = psycopg2.connect("postgresql://...")
df = pd.read_sql(query, conn)
```

### Feature Engineering Pipeline

**Validate features after generation:**

```python
from py.utils.sports_analytics_validator import SportsAnalyticsValidator
from py.data.feature_engineering import generate_player_features

# Generate features
features_df = generate_player_features(
    player_id='00-0019596',
    season=2024,
    week=10
)

# Validate features
validator = SportsAnalyticsValidator()
report = validator.validate_dataframe(features_df, dataset_name='player_features')

if report.has_errors():
    logger.error("Feature generation produced invalid data")
    logger.error(report.generate_report())
    raise ValueError("Invalid features - see log for details")

# Features are valid
return features_df
```

### CI/CD Integration

**Block commits with invalid data:**

```yaml
# .github/workflows/validate-data.yml
name: Validate Data

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2

      - name: Install dependencies
        run: pip install pandas numpy

      - name: Validate feature files
        run: |
          for file in data/features/*.csv; do
            python py/utils/sports_analytics_validator.py --validate-df "$file"
            if [ $? -ne 0 ]; then
              echo "Validation failed for $file"
              exit 1
            fi
          done

      - name: Check SQL queries
        run: |
          for file in queries/*.sql; do
            python py/utils/sports_analytics_validator.py --check-query "$file"
          done
```

---

## Integration Patterns

### Pattern 1: Validation Decorator

**Automatically validate function outputs:**

```python
from functools import wraps
from py.utils.sports_analytics_validator import SportsAnalyticsValidator

def validate_output(validator=None):
    """Decorator to validate DataFrame outputs"""
    if validator is None:
        validator = SportsAnalyticsValidator()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call function
            result = func(*args, **kwargs)

            # Validate if DataFrame
            if isinstance(result, pd.DataFrame):
                report = validator.validate_dataframe(result, dataset_name=func.__name__)

                if report.has_errors():
                    raise ValueError(f"Function {func.__name__} produced invalid data:\n{report.generate_report()}")

            return result
        return wrapper
    return decorator

# Usage
@validate_output()
def get_player_stats(player_id: str, season: int) -> pd.DataFrame:
    """Fetch player stats - automatically validated"""
    query = f"SELECT * FROM mart.player_game_stats WHERE player_id = '{player_id}'"
    return pd.read_sql(query, conn)

# If data is invalid, raises ValueError before returning
stats = get_player_stats('00-0019596', 2024)
```

### Pattern 2: Pre-Commit Hook

**Validate SQL files before commit:**

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Validate all staged SQL files
for file in $(git diff --cached --name-only --diff-filter=ACM | grep '.sql$'); do
    echo "Validating SQL file: $file"
    python py/utils/sports_analytics_validator.py --check-query "$file"

    if [ $? -ne 0 ]; then
        echo "⛔ SQL validation failed for $file"
        echo "Fix errors or use 'git commit --no-verify' to skip (NOT RECOMMENDED)"
        exit 1
    fi
done

echo "✅ SQL validation passed"
```

### Pattern 3: Data Pipeline Checkpoint

**Validate at each pipeline stage:**

```python
from py.utils.sports_analytics_validator import SportsAnalyticsValidator

class DataPipeline:
    def __init__(self):
        self.validator = SportsAnalyticsValidator()

    def run(self):
        # Stage 1: Extract
        raw_data = self.extract_data()
        self._validate(raw_data, 'raw_data')

        # Stage 2: Transform
        features = self.transform_features(raw_data)
        self._validate(features, 'features')

        # Stage 3: Load
        self.load_to_db(features)

    def _validate(self, df: pd.DataFrame, stage: str):
        """Validate data at pipeline stage"""
        report = self.validator.validate_dataframe(df, dataset_name=stage)

        if report.has_errors():
            logger.error(f"Validation failed at stage: {stage}")
            logger.error(report.generate_report())
            raise PipelineError(f"Data validation failed at {stage}")

        if report.has_warnings():
            logger.warning(f"Validation warnings at stage: {stage}")
            for warning in report.get_issues_by_severity(Severity.WARNING):
                logger.warning(f"  - {warning.message}")
```

---

## NFL Domain Rules

### Team Abbreviations Reference

**Current Teams (2020+):**
```python
NFL_TEAMS = {
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
    'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA',
    'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB',
    'TEN', 'WAS'
}
```

**Relocation History:**
```python
TEAM_RELOCATIONS = {
    'SD': ('LAC', 2017),   # San Diego -> LA Chargers
    'STL': ('LAR', 2016),  # St. Louis -> LA Rams
    'OAK': ('LV', 2020),   # Oakland -> Las Vegas
    'LA': None,            # Ambiguous - ERROR
}
```

### Stat Limits Reference

**Passing Stats:**
- Yards: 0-600 (record: 554 yards, Norm Van Brocklin, 1951)
- Attempts: 0-80 (record: 70, Drew Bledsoe, 1994)
- Completions: 0-55 (record: 48, Matt Ryan, 2016)
- Touchdowns: 0-8 (record: 7, multiple QBs)

**Rushing Stats:**
- Yards: 0-300 (record: 296 yards, Adrian Peterson, 2007)
- Attempts: 0-50 (record: 43, multiple RBs)
- Touchdowns: 0-6 (record: 5, multiple RBs)

**Receiving Stats:**
- Yards: 0-350 (record: 336 yards, Willie Anderson, 1989)
- Receptions: 0-25 (record: 20, Terrance Williams, 2013)
- Targets: 0-30 (record: 25+, various WRs)
- Touchdowns: 0-5 (record: 5, Jerry Rice, 1990)

### Schedule Reference

**Regular Season:**
- Weeks: 1-18
- Games per team: 17 (2021+), 16 (pre-2021)
- Bye weeks: 1 per team (typically weeks 6-14)

**Playoffs:**
- Week 19: Wild Card Round (6 games)
- Week 20: Divisional Round (4 games)
- Week 21: Conference Championships (2 games)
- Week 22: Super Bowl (1 game)

**Historical Changes:**
- **2021**: 17-game regular season (was 16)
- **2020**: 14-team playoffs (was 12)
- **1990**: Bye weeks introduced

---

## API Reference

### SportsAnalyticsValidator

**Class:** `py.utils.sports_analytics_validator.SportsAnalyticsValidator`

#### Methods

**`validate_dataframe(df, dataset_name=None)`**

Validate DataFrame against NFL domain rules.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `dataset_name` (str, optional): Name for reporting

**Returns:**
- `ValidationReport`: Report with all issues found

**Example:**
```python
validator = SportsAnalyticsValidator()
report = validator.validate_dataframe(df, dataset_name='features.csv')

if report.has_errors():
    print(report.generate_report())
    exit(1)
```

**`check_query_logic(query)`**

Check SQL query for NFL domain logic violations.

**Parameters:**
- `query` (str): SQL query string

**Returns:**
- `List[ValidationIssue]`: List of issues found

**Example:**
```python
issues = validator.check_query_logic("SELECT * FROM games WHERE week = 25")
for issue in issues:
    print(f"[{issue.severity.value}] {issue.message}")
```

**`validate_query_file(query_path)`**

Validate SQL query from file.

**Parameters:**
- `query_path` (Path): Path to SQL file

**Returns:**
- `List[ValidationIssue]`: List of issues found

**Example:**
```python
from pathlib import Path
issues = validator.validate_query_file(Path('query.sql'))
```

### ValidationReport

**Class:** `py.utils.sports_analytics_validator.ValidationReport`

#### Attributes

- `issues` (List[ValidationIssue]): All validation issues
- `rows_checked` (int): Number of rows validated
- `dataset_name` (str): Name of dataset

#### Methods

**`has_errors()`** → bool
- Returns True if report contains ERROR-level issues

**`has_warnings()`** → bool
- Returns True if report contains WARNING-level issues

**`get_issues_by_severity(severity)`** → List[ValidationIssue]
- Filter issues by severity (ERROR, WARNING, INFO)

**`generate_report()`** → str
- Generate human-readable validation report

### ValidationIssue

**Class:** `py.utils.sports_analytics_validator.ValidationIssue`

#### Attributes

- `severity` (Severity): ERROR, WARNING, or INFO
- `category` (str): Issue category (e.g., 'position_eligibility')
- `message` (str): Short description
- `details` (str, optional): Detailed explanation
- `row_index` (int, optional): DataFrame row index
- `column` (str, optional): Column name
- `value` (Any, optional): Violating value

---

## CLI Reference

### validate_sports_analytics.sh

**Location:** `scripts/validate_sports_analytics.sh`

**Usage:**
```bash
./scripts/validate_sports_analytics.sh <command> [arguments]
```

**Commands:**

**`--validate-df <file.csv>`**
- Validate CSV file against NFL domain rules
- Exit code: 0 (pass), 1 (fail)

**`--check-query <query.sql>`**
- Check SQL query for logic violations
- Exit code: 0 (pass), 1 (fail with errors)

**`--report`**
- Generate validation report for last validation
- Exit code: 0

**`--help`**
- Show help message

**Examples:**
```bash
# Validate feature DataFrame
./scripts/validate_sports_analytics.sh --validate-df data/features.csv

# Check SQL query
./scripts/validate_sports_analytics.sh --check-query queries/player_stats.sql

# Save report to file
./scripts/validate_sports_analytics.sh --validate-df data.csv > report.txt
```

---

## Troubleshooting

### Common Issues

**Issue 1: "Column 'game_id' does not exist in 'mart.player_game_stats'"**

**Cause:** Query references `game_id` in `player_game_stats` table (column doesn't exist)

**Solution:** Use `(season, week)` composite key join
```sql
-- WRONG
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id  -- ❌ Column doesn't exist

-- CORRECT
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = pgs.current_team OR g.away_team = pgs.current_team)
```

**Issue 2: "Ambiguous team abbreviation: 'LA'"**

**Cause:** Using "LA" instead of LAR/LAC

**Solution:** Always specify LAR (Rams) or LAC (Chargers)
```python
# WRONG
df[df['home_team'] == 'LA']

# CORRECT
df[df['home_team'].isin(['LAR', 'LAC'])]
```

**Issue 3: "Passing yards out of range: 650"**

**Cause:** Data error or data leakage (impossible value)

**Solution:** Investigate data source
```python
# Check for data errors
df[df['stat_yards'] > 600]

# Check for leakage (future data, target contamination)
from py.utils.data_leakage_detector import DataLeakageDetector
detector = DataLeakageDetector()
detector.validate_split(train_df, test_df, date_col='kickoff')
```

**Issue 4: "Invalid week number: 25"**

**Cause:** Query references non-existent week

**Solution:** Use valid week ranges
```sql
-- WRONG
SELECT * FROM games WHERE week = 25

-- CORRECT (regular season)
SELECT * FROM games WHERE week BETWEEN 1 AND 18

-- CORRECT (include playoffs)
SELECT * FROM games WHERE week BETWEEN 1 AND 22
```

### Debugging Tips

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from py.utils.sports_analytics_validator import SportsAnalyticsValidator
validator = SportsAnalyticsValidator()
```

**Inspect specific issues:**
```python
report = validator.validate_dataframe(df)

# Get all errors
errors = report.get_issues_by_severity(Severity.ERROR)

# Inspect first error
error = errors[0]
print(f"Category: {error.category}")
print(f"Message: {error.message}")
print(f"Details: {error.details}")
print(f"Row: {error.row_index}")
print(f"Column: {error.column}")
print(f"Value: {error.value}")

# Get problematic rows
if error.row_index is not None:
    print(df.iloc[error.row_index])
```

**Check validation report:**
```python
report = validator.validate_dataframe(df)
print(report.generate_report())

# Save to file
with open('validation_report.txt', 'w') as f:
    f.write(report.generate_report())
```

---

## Related Documentation

- **Schema Reference:** `docs/database/DATABASE_QUICK_REFERENCE.md`
- **SQL Patterns:** `docs/database/SQL_PATTERNS.md`
- **Schema Audit:** `docs/fixes/SQL_SCHEMA_AUDIT_NOV14.md`
- **Data Leakage Detector:** `docs/PHASE_1_AGENT_ENHANCEMENTS.md`

---

**Status:** ✅ Production Ready
**Priority:** P1 HIGH
**Last Updated:** November 14, 2025
