# Rule 20: Data Sufficiency Gate - Quick Reference

**Purpose:** Prevent underpowered experiments by enforcing minimum sample sizes

**When It Triggers:** Code containing `meta_learning`, `adaptive_weight`, `online_learning`, `incremental`, `lightgbm`, `xgboost`, or `sklearn.fit`

**Level:** BLOCKING (will prevent commit if validation missing)

---

## Quick Fixes

### Meta-Learning

**Before (FAILS):**
```python
def train_meta_model(context_features, optimal_weights):
    meta_model = lgb.LGBMRegressor()
    meta_model.fit(context_features, optimal_weights)  # ❌ No validation
    return meta_model
```

**After (PASSES):**
```python
def train_meta_model(context_features, optimal_weights):
    # ✅ Validate data sufficiency
    assert len(context_features) >= 5000, "Insufficient data for meta-learning"

    meta_model = lgb.LGBMRegressor()
    meta_model.fit(context_features, optimal_weights)
    return meta_model
```

### Online Learning

**Before (FAILS):**
```python
def train_online_model(stream_data):
    model = SGDRegressor()
    for batch in stream_data:
        model.partial_fit(batch['X'], batch['y'])  # ❌ No init check
    return model
```

**After (PASSES):**
```python
def train_online_model(stream_data):
    model = SGDRegressor()

    # ✅ Validate initialization data
    initial_batch_size = 1200
    assert len(stream_data[0]['X']) >= 1000, "Need ≥1,000 samples for warm start"

    # Warmstart
    model.partial_fit(stream_data[0]['X'][:initial_batch_size],
                     stream_data[0]['y'][:initial_batch_size])

    for batch in stream_data[1:]:
        model.partial_fit(batch['X'], batch['y'])

    return model
```

### LightGBM/XGBoost

**Before (FAILS):**
```python
def train_lgb(X, y):
    n_features = 15
    model = lgb.train({'objective': 'regression'}, lgb.Dataset(X, y))  # ❌ No validation
    return model
```

**After (PASSES):**
```python
def train_lgb(X, y):
    n_features = X.shape[1]

    # ✅ Validate samples per feature
    assert len(X) >= n_features * 100, f"Need {n_features * 100}+ samples for LightGBM"

    model = lgb.train({'objective': 'regression'}, lgb.Dataset(X, y))
    return model
```

### Ensemble Reweighting

**Before (FAILS):**
```python
def optimize_weights(component_predictions, actuals):
    weights = cvx.Variable(len(component_predictions))
    problem = cvx.Problem(objective)
    problem.solve()  # ❌ No component validation
    return weights.value
```

**After (PASSES):**
```python
def optimize_weights(component_predictions, actuals):
    # ✅ Validate per-component data
    games_per_component = [len(pred) for pred in component_predictions]
    assert min(games_per_component) >= 100, "Need ≥100 games per component"

    weights = cvx.Variable(len(component_predictions))
    problem = cvx.Problem(objective)
    problem.solve()
    return weights.value
```

### Statistical Tests

**Before (FAILS):**
```python
def compare_models(errors_a, errors_b):
    t_stat, p_val = ttest_ind(errors_a, errors_b)  # ❌ No CLT check
    return t_stat, p_val
```

**After (PASSES):**
```python
def compare_models(errors_a, errors_b):
    # ✅ Validate Central Limit Theorem requirements
    n = min(len(errors_a), len(errors_b))
    assert n >= 30, "Need ≥30 samples per group for CLT"

    t_stat, p_val = ttest_ind(errors_a, errors_b)
    return t_stat, p_val
```

---

## Minimum Sample Thresholds

| Use Case | Minimum Samples | Reasoning |
|----------|-----------------|-----------|
| **Meta-learning** | 5,000 | Need diverse contexts to learn weight adjustments |
| **Online learning (init)** | 1,000 | Cold start requires sufficient bootstrap |
| **LightGBM/XGBoost** | 100 × n_features | Prevent overfitting (rule of thumb) |
| **Ensemble reweighting** | 100 per component | Stable weight estimation |
| **Statistical tests (t-test)** | 30 per group | Central Limit Theorem (CLT) |

---

## Error Messages

### Example 1: Meta-Learning

```bash
❌ BLOCKING: data_sufficiency_gate: Data sufficiency validation missing
   Pattern: Meta-learning without data validation
   Severity: CRITICAL
   Required: ≥5,000 samples

   Suggestion: Add: assert len(training_data) >= 5000, "Insufficient data for meta-learning"

   Reference: v7.0 failure (1,701 games insufficient)
   Postmortem: docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md
```

**Fix:** Add `assert len(training_data) >= 5000` before `.fit()` call

### Example 2: LightGBM

```bash
❌ BLOCKING: data_sufficiency_gate: Data sufficiency validation missing
   Pattern: Complex tree model without sample validation
   Severity: HIGH
   Required: 1,500 samples (15 features × 100)
   Features detected: 15

   Suggestion: Add: assert len(X) >= n_features * 100, "Need 1500+ samples for LightGBM"
```

**Fix:** Add `assert len(X) >= n_features * 100` before `.train()` call

---

## Runtime Decorator (Optional)

```python
from py.utils.enforcement_router import enforce_data_sufficiency

@enforce_data_sufficiency
def train_meta_model(X, y):
    """This function won't run if data validation missing."""
    # Rule 20 checks source file for data validation
    # Blocks execution if validation absent
    model.fit(X, y)
    return model
```

**When to use:** Long-running training functions that should fail fast

---

## Manual Validation (Pre-Flight Check)

```bash
# Check if file will pass Rule 20 before committing
python py/utils/enforcement_router.py scripts/train_meta_model.py

# Output if PASSED:
# ✅ PASSED: data_sufficiency_gate: Data sufficiency validation present

# Output if FAILED:
# ❌ BLOCKING: data_sufficiency_gate: Meta-learning without data validation
```

---

## FAQ

**Q: Why 5,000 samples for meta-learning?**
A: v7.0 had 1,701 games for 15 features = 113 samples/feature. LightGBM needs 50-100 samples/feature for complex patterns → 750-1,500 minimum. Meta-learning adds context complexity → 5,000+ recommended.

**Q: Can I bypass Rule 20 in emergencies?**
A: Yes, use `git commit --no-verify`, but document why and add validation ASAP.

**Q: What if my test files trigger Rule 20?**
A: Test files (`test_*.py` or `*_test.py`) are automatically skipped.

**Q: What about linear models (not tree models)?**
A: Linear models need 10-20 samples/feature. Rule 20 currently focuses on complex models (tree-based, meta-learning) where underpowering is most costly.

**Q: How do I know if my data is sufficient?**
A: Check against thresholds in table above. If unsure, use conservative estimate (100 samples/feature for complex models).

---

## See Also

- **Implementation:** `py/utils/enforcement_router.py` (Rule 20, lines 1252-1441)
- **Tests:** `tests/unit/test_rule_20_data_sufficiency.py` (16 comprehensive tests)
- **Full Documentation:** `docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md` (Rule 20 section)
- **Motivation:** `docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md` (v7.0 failure)

---

**Last Updated:** November 19, 2025
**Rule Status:** ✅ Active (BLOCKING level)
