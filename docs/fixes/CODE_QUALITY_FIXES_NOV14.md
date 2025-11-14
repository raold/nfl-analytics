# Code Quality Fixes - November 14, 2025

**Status:** ✅ COMPLETE
**Files Fixed:** 1 (sports_analytics_validator.py)
**Files Already Fixed:** 2 (model_registry_auditor.py, ensemble_health_monitor.py)
**Total Lines Changed:** ~50 lines (type annotations and casts)

---

## Summary

Completed all remaining code quality fixes for 3 new agent files to ensure they pass pre-commit hooks (black, mypy, detect-secrets, ruff).

---

## Files Fixed

### 1. py/utils/sports_analytics_validator.py ✅ FIXED

**Issues Addressed:**

1. **Line 365: Type narrowing for `stat_category not in eligible_stats`**
   - **Problem:** pandas object dtype caused mypy error
   - **Fix:** Added explicit type conversion:
     ```python
     stat_category_str = str(stat_category) if not isinstance(stat_category, str) else stat_category
     if stat_category_str not in eligible_stats:
     ```

2. **Line 475: DataFrame indexing type issue**
   - **Problem:** Boolean indexing with pandas Series caused type mismatch
   - **Fix:** Extracted Series to variable for type inference:
     ```python
     week_series = df["week"]
     min_week = SCHEDULE_CONSTRAINTS["weeks_regular_season"][0]
     max_week = SCHEDULE_CONSTRAINTS["max_playoff_week"]
     invalid_weeks = df[(week_series < min_week) | (week_series > max_week)]
     ```

3. **Line 513: Comparison operator type mismatch**
   - **Problem:** Season column could be float/object type, not int
   - **Fix:** Added explicit int cast:
     ```python
     expected_games = (
         SCHEDULE_CONSTRAINTS["games_per_team_2021_plus"]
         if int(season) >= 2021
         else SCHEDULE_CONSTRAINTS["games_per_team_pre_2021"]
     )
     ```

4. **Row index type safety (multiple locations)**
   - **Problem:** pandas index could be int, np.integer, or other types
   - **Fix:** Added type-safe conversion throughout:
     ```python
     row_index=int(idx) if isinstance(idx, (int, np.integer)) else None
     ```

**Changes:**
- Added type narrowing with `isinstance()` checks
- Added explicit type conversions (`str()`, `int()`)
- Improved pandas Series extraction for type inference
- Made row index handling type-safe

---

### 2. py/utils/model_registry_auditor.py ✅ ALREADY FIXED

**Status:** No issues found

**Previous Fixes Applied:**
- Credentials already marked with `# pragma: allowlist secret`
- Type annotations already complete
- All mypy errors already resolved

---

### 3. py/utils/ensemble_health_monitor.py ✅ ALREADY FIXED

**Status:** No issues found

**Previous Fixes Applied:**
- Type annotations already complete
- All mypy errors already resolved
- No credentials or security issues

---

## Verification Commands

```bash
# Black formatting check
python -m black --check py/utils/sports_analytics_validator.py
python -m black --check py/utils/model_registry_auditor.py
python -m black --check py/utils/ensemble_health_monitor.py

# Mypy type checking
python -m mypy py/utils/sports_analytics_validator.py --ignore-missing-imports
python -m mypy py/utils/model_registry_auditor.py --ignore-missing-imports
python -m mypy py/utils/ensemble_health_monitor.py --ignore-missing-imports

# Ruff linting
python -m ruff check py/utils/sports_analytics_validator.py
python -m ruff check py/utils/model_registry_auditor.py
python -m ruff check py/utils/ensemble_health_monitor.py

# Detect secrets
detect-secrets scan py/utils/sports_analytics_validator.py
detect-secrets scan py/utils/model_registry_auditor.py
detect-secrets scan py/utils/ensemble_health_monitor.py

# Pre-commit on all files
pre-commit run --files py/utils/sports_analytics_validator.py \
                       py/utils/model_registry_auditor.py \
                       py/utils/ensemble_health_monitor.py
```

---

## Key Patterns Applied

### 1. Type Narrowing for pandas Object Dtypes

**Problem:** pandas columns with object dtype fail mypy checks

**Solution:**
```python
# Bad (fails mypy)
if stat_category not in eligible_stats:

# Good (passes mypy)
stat_category_str = str(stat_category) if not isinstance(stat_category, str) else stat_category
if stat_category_str not in eligible_stats:
```

### 2. Series Extraction for Type Inference

**Problem:** Inline boolean indexing obscures types

**Solution:**
```python
# Bad (type obscured)
invalid_weeks = df[(df["week"] < min_week) | (df["week"] > max_week)]

# Good (type clear)
week_series = df["week"]
invalid_weeks = df[(week_series < min_week) | (week_series > max_week)]
```

### 3. Explicit Type Casts for Comparisons

**Problem:** pandas columns can have unexpected types (float, object)

**Solution:**
```python
# Bad (fails if season is float)
if season >= 2021:

# Good (always works)
if int(season) >= 2021:
```

### 4. Type-Safe Index Handling

**Problem:** pandas index can be int, np.integer, or other types

**Solution:**
```python
# Bad (fails mypy)
row_index=idx

# Good (type-safe)
row_index=int(idx) if isinstance(idx, (int, np.integer)) else None
```

---

## Pre-Commit Hook Compliance

All 3 files now pass:

✅ **black** - Code formatting
✅ **ruff** - Linting and auto-fixes
✅ **mypy** - Type checking (with `--ignore-missing-imports`)
✅ **detect-secrets** - Credential scanning
✅ **bandit** - Security linting
✅ **trailing-whitespace** - Whitespace check
✅ **end-of-file-fixer** - EOF newline
✅ **mixed-line-ending** - LF line endings

---

## Impact

**Development Velocity:**
- 3 files ready for production integration
- Zero mypy errors blocking commits
- All pre-commit hooks passing

**Code Quality:**
- Type safety improved with explicit casts
- pandas type issues resolved systematically
- Consistent patterns across all 3 agent files

**Next Steps:**
1. ✅ Commit all 3 files with passing pre-commit hooks
2. ✅ Integrate into CI/CD pipeline
3. ✅ Add to agent coordinator architecture

---

## Related Documentation

- **Pre-commit config:** `.pre-commit-config.yaml`
- **Agent architecture:** `docs/AGENT_COORDINATOR_GUIDE.md`
- **Type checking:** Python 3.11+ with mypy 1.5.0
- **Pandas typing:** Using pandas-stubs for better type inference

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 14, 2025
**Duration:** 15 minutes
**Files Modified:** 1 (sports_analytics_validator.py)
**Lines Changed:** ~50 lines
