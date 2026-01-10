# NFL Analytics - Code Quality Assessment
**Date:** January 9, 2026
**Assessed by:** Claude Code (Code Review Specialist Agent)
**Scope:** 573 Python files (247K LOC), 74 R files (21K LOC)

---

## Executive Summary

| Language | Grade | Key Issues | Status |
|----------|-------|------------|--------|
| **Python** | **C+** | 1,326 type errors, 14 undefined names, 67 unformatted files | Needs attention |
| **R** | **B+** | 1,135 linting warnings (mostly false positives), 25 edge case bugs | Good shape |

**Overall Assessment:** Solid foundation with good documentation, but accumulated technical debt in type safety and code quality standards.

---

## Critical Issues (Fix Immediately)

### 1. F821 - Undefined Names (14 occurrences) - RUNTIME CRASHES
**Files affected:**
- `py/backtest/pinnacle_historical_simulation.py:78`
- `py/causal/did_injury_analysis.py:137`
- `py/rl/experiment_feature_extractor.py:166`

**Problem:**
```python
"password": os.getenv("DB_PASSWORD", "sicillionbillions")
# ERROR: 'os' is not imported - this WILL crash at runtime
```

**Fix:**
```bash
# Add missing imports
ruff check py/ --select F821 --fix
```

### 2. R seq_len Edge Case Bugs (25 occurrences) - DATA BUGS
**Files affected:**
- `R/analysis/line_movement_analysis.R:652`
- `R/backfill_advanced_metrics.R:417,440,462`

**Problem:**
```r
for (i in 1:nrow(data)) { ... }
# When nrow(data) == 0, this becomes 1:0 = c(1, 0)
# Loop runs TWICE with invalid indices
```

**Fix:**
```r
for (i in seq_len(nrow(data))) { ... }
# seq_len(0) = integer(0), loop doesn't run
```

---

## Python Detailed Findings

### Black Formatting: 88% Compliant (67 files need formatting)

**One-line fix:**
```bash
black py/
```

**Files needing formatting:**
- `py/features/game_feature_extractor.py` (core feature extraction)
- `py/semantic/features/semantic_features.py` (LLM features)
- `py/feedback/semantic_attribution.py` (attribution tracking)
- 64 more files...

### Ruff Linting: ~1,000 Issues

**Issue Breakdown:**

| Code | Count | Description | Severity | Auto-fix? |
|------|-------|-------------|----------|-----------|
| N806 | 466 | Variable name should be lowercase | LOW | No |
| UP045 | 137 | Deprecated typing import | MEDIUM | Yes |
| E402 | 135 | Import not at top of file | MEDIUM | Yes |
| N803 | 95 | Argument name should be lowercase | LOW | No |
| F401 | 77 | Unused imports | MEDIUM | Yes |
| **F821** | **14** | **Undefined name (CRASHES)** | **CRITICAL** | **Yes** |

**Auto-fix many issues:**
```bash
ruff check py/ --fix
```

**Top 5 files with most issues:**
1. `py/api/routes/game_outcome_predictions.py` - 48 issues
2. `py/models/improve_bnn_calibration.py` - 22 issues
3. `py/models/game_outcome_model.py` - 19 issues
4. `py/features/asof_features_unified.py` - 15 issues
5. `py/models/train_bnn_rushing_simple.py` - 14 issues

### MyPy Type Checking: 1,326 Errors (319/573 files affected)

**Top 5 error types:**

| Count | Error Type | Impact |
|-------|------------|--------|
| 73 | Returning Any from function declared to return float | Type safety violations |
| 27 | "None" has no attribute "predict" | Potential crashes |
| 26 | Returning Any from function declared to return int | Type safety violations |
| 25 | "None" has no attribute "r" | Potential crashes |
| 22 | Returning Any from function declared to return dict | Type safety violations |

**Top 10 files with most type errors:**

| File | Errors | Module |
|------|--------|--------|
| `py/features/game_feature_extractor.py` | 47 | Core features |
| `py/compute/performance_tracker.py` | 29 | Monitoring |
| `py/ensemble/r_model_interface_batched.py` | 26 | R interface |
| `py/compute/adaptive_scheduler.py` | 25 | Scheduling |
| `py/models/game_outcome_model_fixed.py` | 22 | Predictions |
| `py/training/train_meta_classifier.py` | 20 | Meta-learning |
| `py/features/line_movement_tracker.py` | 19 | Odds tracking |
| `py/models/game_outcome_model.py` | 17 | Predictions |
| `py/models/evidential_regression.py` | 17 | EDL |
| `py/models/bayesian_neural_network.py` | 16 | BNN |

**Sample critical errors:**
```python
# py/features/game_feature_extractor.py (47 errors)
def extract_feature() -> float:
    return dict_result["value"]  # Returns Any, not float

# py/ensemble/r_model_interface_batched.py (26 errors)
self.r.predict(X)  # self.r could be None
```

---

## R Detailed Findings

### styler: 100% Compliant (All 74 files pass)

All R files follow tidyverse style guide. No action needed.

### lintr: 1,135 Warnings

**Issue Breakdown:**

| Linter | Count | Description | Real Issue? |
|--------|-------|-------------|-------------|
| object_usage | 1,013 | Unused variables | Mostly false positives (brms NSE) |
| line_length | 51 | Lines >80 chars | Yes (readability) |
| seq_linter | 25 | Use seq_len() not 1:n | **YES - EDGE CASE BUGS** |
| commented_code | 23 | Remove commented code | Yes (cleanup) |
| indentation | 19 | Indentation issues | Yes (style) |

**Top 5 files with most warnings:**

| File | Warnings | Primary Issue |
|------|----------|---------------|
| `R/bayesian_player_props.R` | 178 | Unused vars (false positives) |
| `R/dissertation_figures.R` | 171 | Unused ggplot aesthetics |
| `R/team_strength_ratings.R` | 141 | Unused parameters |
| `R/rolling_features_slider.R` | 124 | Unused intermediate vars |
| `R/bayesian_team_ratings_brms.R` | 99 | Unused model components |

**Critical: 25 seq_len bugs to fix manually**

---

## Quick Win Action Plan

### Week 1 (3 hours)

1. **Fix F821 undefined names** (15 min)
   ```bash
   ruff check py/ --select F821 --fix
   ```

2. **Run Black formatter** (5 min)
   ```bash
   black py/
   ```

3. **Auto-fix ruff issues** (10 min)
   ```bash
   ruff check py/ --select F401,I001,UP045 --fix
   ```

4. **Fix R seq_len bugs** (2 hours)
   ```bash
   # Find all occurrences
   grep -rn "1:nrow\|1:ncol\|1:length" R/ --include="*.R"

   # Replace with seq_len() manually (25 occurrences)
   ```

**Impact:** Eliminates all critical runtime bugs + formatting issues

### Month 1 (40 hours)

5. **Add None checks for top 5 type error files** (20 hours)
   - `game_feature_extractor.py` (47 errors)
   - `performance_tracker.py` (29 errors)
   - `r_model_interface_batched.py` (26 errors)
   - `adaptive_scheduler.py` (25 errors)
   - `game_outcome_model_fixed.py` (22 errors)

6. **Fix variable naming (N806)** (10 hours)
   - Focus on public APIs first
   - `homeTeam` → `home_team`

7. **Add type stubs** (2 hours)
   ```bash
   pip install types-requests types-PyYAML types-redis
   ```

8. **Configure R lintr to reduce false positives** (1 hour)
   ```r
   # .lintr
   linters: linters_with_defaults(
     object_usage_linter = NULL,
     line_length_linter(120)
   )
   ```

**Impact:** Reduces mypy errors from 1,326 → ~800

### Quarter 1 (80 hours)

9. **Setup pre-commit hooks** (4 hours)
10. **CI/CD quality gates** (8 hours)
11. **Incremental mypy strictness** (40 hours)
12. **Documentation standards enforcement** (8 hours)
13. **Refactor remaining type errors** (20 hours)

**Impact:** Grade improves from C+ to B+ (Python), B+ to A- (R)

---

## Recommended Quality Standards Going Forward

### Python

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "UP", "B", "A"]
ignore = ["N806"]  # Allow some naming flexibility

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

### R

```r
# .lintr
linters: linters_with_defaults(
  object_usage_linter = NULL,        # Too many brms false positives
  line_length_linter(120),           # 120 instead of 80
  commented_code_linter = NULL       # Allow some commented alternatives
)
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix, --select, "F821,F401,I001,UP045"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        files: ^py/
```

---

## Progress Tracking

### Target Metrics

| Metric | Current | 3 Months | 6 Months |
|--------|---------|----------|----------|
| **Python Grade** | C+ | B | A- |
| **R Grade** | B+ | A- | A |
| Black compliance | 88% | 100% | 100% |
| Ruff issues | ~1,000 | <200 | <50 |
| mypy errors | 1,326 | <500 | <100 |
| Files with type errors | 56% | <30% | <10% |
| R lintr warnings | 1,135 | <300 | <100 |
| **Critical bugs (F821)** | **14** | **0** | **0** |

---

## Tool Commands Reference

```bash
# Python Quality Checks
black --check py/                    # Check formatting
black py/                            # Apply formatting
ruff check py/                       # Lint code
ruff check py/ --fix                 # Auto-fix linting
mypy py/ --ignore-missing-imports    # Type check

# R Quality Checks
Rscript -e "styler::style_dir('R/', dry='on')"     # Check style
Rscript -e "styler::style_dir('R/')"               # Apply style
Rscript -e "library(lintr); lint_dir('R/')"        # Lint code

# Pre-commit
pre-commit install                   # Install hooks
pre-commit run --all-files           # Run all hooks

# Stats
find py/ -name "*.py" | wc -l       # Count Python files
find R/ -name "*.R" | wc -l         # Count R files
```

---

## Conclusion

The NFL Analytics project has **good bones** but needs **systematic cleanup** to reach production-grade quality.

**Key Points:**

1. **14 critical runtime bugs (F821)** - fix immediately
2. **25 R edge case bugs (seq_len)** - fix this week
3. **88% of Python already formatted** - finish the job
4. **1,326 type errors** - tackle incrementally
5. **R code is already high quality** - just cleanup false positives

**Effort Estimate:**
- Week 1: 3 hours (critical fixes)
- Month 1: 40 hours (systematic cleanup)
- Quarter 1: 80 hours (quality standards)

**Result:** Grade A codebase in 6 months with focused effort.

---

**Full detailed report:** `/tmp/code_quality_assessment_report.md` (19,000+ words)

**Next Steps:**
1. Review this assessment
2. Execute Week 1 plan (fix F821 + format)
3. Schedule Month 1 cleanup work
4. Setup pre-commit hooks for ongoing enforcement
