# Rule 19: Example Files (Would Be Caught)

This document shows real code patterns that Rule 19 detects and blocks.

---

## Example 1: Hardcoded Mean in predict() - BLOCKED

**File:** `py/ensemble/bad_ensemble_v1.py`

```python
#!/usr/bin/env python3
"""
Bad ensemble - uses placeholder predictions.
RULE 19 BLOCKS THIS FILE.
"""

class BadEnsemble:
    def __init__(self):
        self.models = []

    def predict(self, player_id, season, week):
        """
        Predict passing yards - USES PLACEHOLDER.
        Rule 19 detects: Hardcoded mean = 250.0 in predict() method
        Severity: CRITICAL
        """
        # TODO: integrate with real models
        mean = 250.0  # BLOCKED by Rule 19 line 156
        std = 30.0    # BLOCKED by Rule 19 line 157
        confidence = 0.95

        return {
            'player_id': player_id,
            'season': season,
            'week': week,
            'mean': mean,          # CRITICAL: Direct hardcoded value
            'std': std,            # CRITICAL: Direct hardcoded value
            'confidence': confidence
        }
```

**Rule 19 Detection:**

```
BLOCKING: placeholder_detection: Placeholder predictions detected: 2 CRITICAL, 0 HIGH

Details:
  1. Line 25: match="mean = 250.0", description="Direct assignment: mean = 250"
     severity=CRITICAL, location="predict() method"

  2. Line 26: match="std = 30.0", description="Direct std assignment: 20-49"
     severity=CRITICAL, location="predict() method"

Recommendation: Remove all placeholder predictions before running validation/backtest
Blocked for: 18+ hours (Nov 16-18)
```

---

## Example 2: Dict with Hardcoded Values - BLOCKED

**File:** `py/ensemble/broken_v2.py`

```python
#!/usr/bin/env python3
"""
Broken ensemble v2 - returns hardcoded dict.
RULE 19 BLOCKS THIS FILE.
"""

class BrokenEnsemble:
    def predict(self, player_id, season, week):
        """
        Return hardcoded prediction dict.
        Rule 19 detects: Hardcoded mean: 250.0 in dict
        """
        # Placeholder implementation - to be replaced
        return {
            'mean': 250.0,  # BLOCKED: Hardcoded mean
            'std': 35.0,    # BLOCKED: Hardcoded std without np.std()
            'n_samples': 1000
        }
```

**Rule 19 Detection:**

```
BLOCKING: placeholder_detection: Placeholder predictions detected: 1 CRITICAL, 1 HIGH

Details:
  1. Line 20: match='"mean": 250.0', description="Hardcoded mean: 250.0"
     severity=CRITICAL, location="predict() method"

  2. Line 21: match='"std": 35.0', description="Hardcoded std without computation"
     severity=CRITICAL, location="predict() method"

Recommendation: Compute mean/std from actual model predictions
```

---

## Example 3: Placeholder Keywords - BLOCKED

**File:** `py/ensemble/incomplete_ensemble.py`

```python
#!/usr/bin/env python3
"""
Incomplete ensemble with placeholder indicators.
RULE 19 BLOCKS THIS FILE.
"""

class IncompleteEnsemble:
    def predict(self, player_id, season, week):
        """Not yet integrated with production models."""
        # placeholder: using dummy values for now
        return {
            'mean': 250.0,
            'std': 30.0
        }

    def get_confidence(self):
        # TODO: integrate with uncertainty quantification
        return {'ci_lower': 220.0, 'ci_upper': 280.0}

    def predict_with_uncertainty(self):
        # temporary return for testing
        return self.dummy_prediction()

    def dummy_prediction(self):
        return {'mean': 250}
```

**Rule 19 Detection:**

```
BLOCKING: placeholder_detection: Placeholder predictions detected: 3 CRITICAL, 0 HIGH

Details:
  1. Line 14: match="placeholder", description="Placeholder indicator"
     severity=CRITICAL, location="predict/return path"

  2. Line 23: match="TODO.*integrate", description="TODO: integrate indicator"
     severity=CRITICAL, location="predict/return path"

  3. Line 27: match="temporary.*return", description="Temporary return indicator"
     severity=CRITICAL, location="predict/return path"

Recommendation: Remove placeholder indicators and implement real logic
```

---

## Example 4: Mixed Range Placeholders - BLOCKED

**File:** `py/ensemble/lazy_ensemble.py`

```python
#!/usr/bin/env python3
"""
Lazy ensemble with multiple placeholder patterns.
RULE 19 BLOCKS THIS FILE.
"""

class LazyEnsemble:
    def predict(self, player_id, season, week):
        """Uses multiple placeholder patterns."""

        # All of these will be BLOCKED
        if week == 1:
            mean = 245.0  # BLOCKED: In range 245-249
        elif week == 2:
            mean = 248.0  # BLOCKED: In range 245-249
        elif week == 3:
            mean = 250.0  # BLOCKED: Exact 250
        elif week == 4:
            mean = 252.0  # BLOCKED: In range 250-255
        else:
            mean = 255.0  # BLOCKED: In range 250-255

        # Standard deviation placeholder
        std = 30.0  # BLOCKED: Hardcoded, no computation

        return {
            'mean': mean,    # CRITICAL
            'std': std,      # CRITICAL
            'week': week
        }
```

**Rule 19 Detection:**

```
BLOCKING: placeholder_detection: Placeholder predictions detected: 6 CRITICAL, 0 HIGH

Details (first 10):
  1. Line 15: match="mean = 245.0", severity=CRITICAL, location="predict() method"
  2. Line 17: match="mean = 248.0", severity=CRITICAL, location="predict() method"
  3. Line 19: match="mean = 250.0", severity=CRITICAL, location="predict() method"
  4. Line 21: match="mean = 252.0", severity=CRITICAL, location="predict() method"
  5. Line 23: match="mean = 255.0", severity=CRITICAL, location="predict() method"
  6. Line 26: match="std = 30.0", severity=CRITICAL, location="predict() method"

Recommendation: Implement real prediction logic instead of hardcoded values
```

---

## Example 5: Production File - PASSES

**File:** `py/ensemble/enhanced_ensemble_v6_production.py` (Real Production Code)

```python
#!/usr/bin/env python3
"""
Production ensemble with real predictions.
RULE 19 PASSES THIS FILE.
"""

import numpy as np

class EnhancedEnsemble5Way:
    def __init__(self):
        self.models = [...]

    def predict(self, player_id, season, week):
        """
        Production prediction - properly computed.
        Rule 19: PASSES
        """

        # Collect predictions from real components
        predictions = []
        for component in self.components:
            try:
                pred = component.predict(player_id, season, week)
                if pred is not None and not np.isnan(pred['mean']):
                    predictions.append(pred['mean'])
            except Exception as e:
                logger.warning(f"Component failed: {e}")
                continue

        # Validate minimum components
        if len(predictions) < 3:
            raise RuntimeError("Ensemble unhealthy: < 3 components loaded")

        # COMPUTED - not hardcoded
        ensemble_mean = np.mean(predictions)      # PASSES: Computed from data
        ensemble_std = np.std(predictions)        # PASSES: Computed from data
        ensemble_confidence = self.compute_calibration()  # PASSES: Method call

        return {
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'ensemble_confidence': ensemble_confidence,
            'n_components': len(predictions)
        }

    def compute_calibration(self):
        """Compute confidence from historical accuracy."""
        # Actual computation (not placeholder)
        return self.calibration_tracker.get_calibration_score()
```

**Rule 19 Detection:**

```
✅ PASSED: placeholder_detection - No placeholder predictions detected
```

---

## Example 6: Test File (Placeholders Allowed) - SKIPPED

**File:** `tests/test_ensemble_placeholders.py`

```python
#!/usr/bin/env python3
"""
Test file with placeholder predictions.
RULE 19 SKIPS THIS FILE (test files are allowed to use placeholders).
"""

import pytest
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way


def test_ensemble_with_placeholder():
    """Test ensemble behavior with placeholder values."""

    # Placeholders are OK in test files
    placeholder_mean = 250.0  # SKIPPED: In test file
    placeholder_std = 30.0    # SKIPPED: In test file

    result = {
        'mean': placeholder_mean,
        'std': placeholder_std
    }

    assert result['mean'] == 250.0
    assert result['std'] == 30.0


def test_predict_returns_dict():
    """Test that predict returns proper dictionary."""
    ensemble = EnhancedEnsemble5Way()

    # Placeholder for mock data
    mock_prediction = {
        'mean': 250.0,  # SKIPPED: In test file
        'std': 25.0     # SKIPPED: In test file
    }

    assert 'mean' in mock_prediction
    assert 'std' in mock_prediction
```

**Rule 19 Detection:**

```
SKIPPED: placeholder_detection - Test file - placeholders allowed for testing
```

---

## Example 7: Code with Computation Context - PASSES

**File:** `py/ensemble/smart_ensemble.py`

```python
#!/usr/bin/env python3
"""
Smart ensemble - properly structured predictions.
RULE 19 PASSES THIS FILE.
"""

import numpy as np
import pandas as pd


class SmartEnsemble:
    def predict(self, player_id, season, week):
        """
        Make predictions with proper computation.
        Rule 19: PASSES
        """

        # Collect ensemble predictions
        component_predictions = []
        for component in self.components:
            pred = component.predict(player_id, season, week)
            if pred is not None:
                component_predictions.append(pred['mean'])

        # Compute from data (PASSES: has computation context)
        ensemble_mean = np.mean(component_predictions)  # np.mean() context
        ensemble_std = np.std(component_predictions)    # np.std() context

        # Uncertainty from percentiles
        q25 = np.percentile(component_predictions, 25)
        q75 = np.percentile(component_predictions, 75)
        uncertainty = (q75 - q25) / 1.35  # Proper computation

        return {
            'mean': ensemble_mean,      # PASSES: Computed
            'std': ensemble_std,        # PASSES: Computed
            'uncertainty': uncertainty, # PASSES: Computed from percentiles
            'n_models': len(component_predictions)
        }


class BayesianEnsemble:
    def predict(self, player_id, season, week):
        """
        Bayesian ensemble with uncertainty quantification.
        Rule 19: PASSES
        """

        # Get posterior samples from Bayesian model
        posterior_samples = self.bayesian_model.sample_posterior(
            player_id=player_id,
            season=season,
            week=week,
            n_samples=4000
        )

        # Compute credible interval from samples (PASSES: proper computation)
        mean_estimate = np.mean(posterior_samples)      # Computed
        std_estimate = np.std(posterior_samples)        # Computed
        ci_lower = np.percentile(posterior_samples, 2.5)  # Percentile
        ci_upper = np.percentile(posterior_samples, 97.5)  # Percentile

        return {
            'mean': mean_estimate,
            'std': std_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_posterior_samples': len(posterior_samples)
        }
```

**Rule 19 Detection:**

```
✅ PASSED: placeholder_detection - No placeholder predictions detected
```

---

## Example 8: Misleading Comment (No Return) - PASSES

**File:** `py/ensemble/config_ensemble.py`

```python
#!/usr/bin/env python3
"""
Ensemble with placeholder comments (but no nearby return).
RULE 19 PASSES THIS FILE.
"""

class ConfigEnsemble:
    # Configuration constants (PASSES: not in predict or near return)
    DEFAULT_MEAN = 250.0  # placeholder value for docs (no return nearby)
    DEFAULT_STD = 30.0    # standard deviation placeholder

    def predict(self, player_id, season, week):
        """
        Real prediction method.
        Rule 19: PASSES
        """

        # Load config (constants above are OK - not in predict return)
        mean = self._compute_mean_from_config()
        std = self._compute_std_from_config()

        return {
            'mean': mean,  # PASSES: Loaded from method, not hardcoded
            'std': std     # PASSES: Loaded from method, not hardcoded
        }

    def _compute_mean_from_config(self):
        """Compute mean from configuration."""
        # Proper computation even if using defaults
        return self.config['ensemble_mean']

    def _compute_std_from_config(self):
        """Compute std from configuration."""
        # Proper computation
        return self.config['ensemble_std']
```

**Rule 19 Detection:**

```
✅ PASSED: placeholder_detection - No placeholder predictions detected
```

Note: Placeholder comments in configuration constants are OK if not directly in predict() method and not near return statements.

---

## Summary Table

| File | Pattern | Severity | Status | Reason |
|------|---------|----------|--------|--------|
| bad_ensemble_v1.py | Hardcoded mean=250.0 | CRITICAL | BLOCKED | In predict() method |
| broken_v2.py | Dict with mean: 250.0 | CRITICAL | BLOCKED | In predict() method |
| incomplete_ensemble.py | Placeholder keyword | CRITICAL | BLOCKED | Near return statement |
| lazy_ensemble.py | Multiple hardcoded values | CRITICAL | BLOCKED | All in predict() |
| enhanced_ensemble_v6_production.py | np.mean() computation | - | PASSED | Proper computation |
| smart_ensemble.py | np.std() computation | - | PASSED | Proper computation |
| test_ensemble_placeholders.py | Any placeholder | - | SKIPPED | Test file |
| config_ensemble.py | Config constants | - | PASSED | Not in predict/return |

---

## Testing These Examples

```bash
# Test each file against Rule 19
python py/utils/enforcement_router.py \
  docs/enforcement/bad_ensemble_v1.py \
  docs/enforcement/broken_v2.py \
  docs/enforcement/incomplete_ensemble.py \
  docs/enforcement/lazy_ensemble.py

# Expected: All BLOCKED

python py/utils/enforcement_router.py \
  py/ensemble/enhanced_ensemble_v6_production.py

# Expected: PASSED
```

---

**Related Documentation:**
- [Rule 19 Full Documentation](/Users/dro/rice/nfl-analytics/docs/enforcement/RULE_19_PLACEHOLDER_DETECTION.md)
- [Enforcement Framework](/Users/dro/rice/nfl-analytics/docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md)
- [Implementation](/Users/dro/rice/nfl-analytics/py/utils/enforcement_router.py)
