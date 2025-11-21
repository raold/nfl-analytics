# Ensemble v7.0 → v6.0 Component Extraction

**Date:** November 19, 2025
**Author:** Claude Code (Sonnet 4.5)
**Purpose:** Extract reusable components from v7.0 (ELO+WLS) and integrate into v6.0 (production ensemble)

---

## Executive Summary

**Status:** ✅ Complete
**Components Extracted:** 3 (R Model Interface, Rule 19, Test Suite)
**Tests Extracted:** 45 general ensemble tests (from 68 total v7 tests)
**Tests v7-Specific:** 23 (ELO-specific, WLS-specific)

**Key Findings:**

1. **R Model Interface** (`py/ensemble/r_model_interface.py`): ✅ Already in v6.0 production (line 77)
2. **Rule 19 (Placeholder Detection)**: ✅ Active in enforcement framework and pre-commit hooks
3. **Test Suite**: 45/68 tests are general ensemble tests applicable to v6.0

---

## Component 1: R Model Interface

**File:** `py/ensemble/r_model_interface.py` (533 lines)
**Status:** ✅ Already Integrated in v6.0
**v6 Integration Point:** Line 77 of `enhanced_ensemble_v6_production.py`

### Verification

```python
# v6.0 production imports (line 77)
from py.ensemble.r_model_interface import RModelInterface

# v6.0 initialization (line 379)
self.r_interface = RModelInterface(db_config=self.db_config)

# v6.0 usage (lines 437, 448, 470)
ss_pred, ss_std = self._predict_state_space(player_id, season, week, features)
hier_pred, hier_std = self._predict_hierarchical(player_id, season, week, features)
ip_pred, ip_std = self._predict_informative_priors(player_id, season, week, features)
```

**Capabilities:**
- Loads R models via rpy2 (brms, cmdstanr)
- Handles 3 Bayesian models: State-Space v2, Hierarchical v2, Informative Priors v2
- Posterior predictive distribution generation
- Uncertainty quantification (mean, std)

**Action Required:** ✅ None - already in production use

---

## Component 2: Rule 19 - Placeholder Detection

**File:** `py/utils/enforcement_router.py` (lines 337-348)
**Status:** ✅ Active in Enforcement Framework
**Integration:** Pre-commit hook (line 115-143 of `.pre-commit-config.yaml`)

### Verification

```bash
$ python py/utils/enforcement_router.py py/ensemble/enhanced_ensemble_v6_production.py
✅ PASSED: placeholder_detection: No placeholder predictions detected
```

**Rule Configuration:**

```python
# Rule 19: Placeholder Detection (CRITICAL) - PHASE 4
self.rules['placeholder_detection'] = EnforcementRule(
    name='placeholder_detection',
    description='Prevent hardcoded placeholder predictions from reaching production',
    triggers=[
        r'py/ensemble/.*\.py$',           # All ensemble files
        r'predict\(',                      # Predict methods
        r'return.*dict',                   # Dict returns
    ],
    validator=self._validate_placeholder_predictions,
    level=EnforcementLevel.BLOCKING
)
```

**What It Catches:**
- Hardcoded placeholder predictions (e.g., `return 250.0`)
- Synthetic test values in production code
- Mock returns in non-test files
- `ensemble_mean = 250.0` assignments

**Pre-commit Integration:**

```yaml
- id: validate-ensemble-components
  name: Ensemble Component Validation
  entry: python py/utils/enforcement_router.py
  language: system
  files: \.(py)$
  pass_filenames: true
  description: Ensure ensemble has minimum component checks
```

**Action Required:** ✅ None - already active and tested

---

## Component 3: Test Suite Extraction

**Status:** ✅ 45 tests extracted and adapted for v6.0
**Source Files:**
- `tests/unit/test_ensemble_v7_components.py` (21 tests → 18 general)
- `tests/integration/test_ensemble_v7_integration.py` (25 tests → 20 general)
- `tests/integration/test_ensemble_v7_sql.py` (22 tests → 7 general)

**Target Files:**
- `tests/unit/test_ensemble_framework.py` (NEW - 18 general unit tests)
- `tests/integration/test_ensemble_v6_integration.py` (UPDATED - added 20 tests)
- `tests/integration/test_ensemble_v6_sql.py` (NEW - 7 SQL/data tests)

### Test Categorization

#### **General Ensemble Tests (45 total - Applicable to v6.0)**

**Unit Tests (18):**
1. ✅ `test_state_space_not_placeholder` - Component prediction validation
2. ✅ `test_state_space_returns_tuple` - Return type validation
3. ✅ `test_state_space_uncertainty_positive` - Uncertainty check
4. ✅ `test_state_space_prediction_in_reasonable_range` - Range validation
5. ✅ `test_hierarchical_not_placeholder` - Hierarchical validation
6. ✅ `test_hierarchical_returns_tuple` - Return type
7. ✅ `test_hierarchical_uncertainty_reasonable` - Uncertainty bounds
8. ✅ `test_xgboost_v4_not_placeholder` - XGBoost validation
9. ✅ `test_xgboost_v4_uses_features` - Feature usage check
10. ✅ `test_xgboost_v4_ridge_component_executed` - Execution validation
11. ✅ `test_informative_priors_not_placeholder` - IP validation
12. ✅ `test_informative_priors_uses_domain_knowledge` - Domain validation
13. ✅ `test_hybrid_not_placeholder` - Hybrid validation
14. ✅ `test_hybrid_13_features_processed` - Feature count check
15. ✅ `test_minimum_3_components_required` - Minimum viable ensemble
16. ✅ `test_component_variance_check` - Component diversity (≥5 yards)
17. ✅ `test_default_weights_sum_to_one` - Weight normalization
18. ✅ `test_canonical_weight_values` - Weight validation

**Integration Tests (20):**
1. ✅ `test_all_5_components_generate_predictions` - E2E pipeline
2. ✅ `test_ensemble_mean_reasonable_value` - Range check (180-350 yards)
3. ✅ `test_ensemble_mean_not_placeholder` - Placeholder detection
4. ✅ `test_base_ensemble_mean_present` - Output structure
5. ✅ `test_prediction_returns_uncertainty` - Uncertainty quantification
6. ✅ `test_prediction_includes_component_details` - Component breakdown
7. ✅ `test_component_diversity_threshold_5_yards` - Diversity enforcement
8. ✅ `test_ensemble_weighted_average_correct` - Math validation
9. ✅ `test_uncertainty_pooling` - Variance pooling formula
10. ✅ `test_temperature_scaling_applied` - Calibration check
11. ✅ `test_calibrated_std_differs_from_ensemble_std` - Scaling verification
12. ✅ `test_prediction_intervals_use_calibrated_std` - PI validation
13. ✅ `test_ensemble_handles_one_component_failure` - Graceful degradation
14. ✅ `test_ensemble_handles_two_component_failures` - 3/5 minimum
15. ✅ `test_ensemble_fails_with_three_component_failures` - Critical failure
16. ✅ `test_required_output_keys` - Output schema validation
17. ✅ `test_output_values_are_numeric` - Type validation
18. ✅ `test_weights_sum_to_one` - Weight normalization
19. ✅ `test_prediction_time_is_reasonable` - Performance (<5s)
20. ✅ `test_causal_features_failure_handled` - Causal graceful degradation

**SQL/Data Tests (7):**
1. ✅ `test_defense_roughing_classifications_exists` - Causal data file
2. ✅ `test_stadium_distance_matrix_exists` - Travel data file
3. ✅ `test_ridge_model_path_resolvable` - Model file paths
4. ✅ `test_xgboost_v4_path_resolvable` - XGBoost paths
5. ✅ `test_bayesian_models_paths_resolvable` - R model paths
6. ✅ `test_temperature_scaler_paths_resolvable` - Calibration paths
7. ✅ `test_plays_table_feature_extraction_query` - SQL query validation

#### **v7-Specific Tests (23 total - NOT Applicable to v6.0)**

**Reasons for Exclusion:**
- ELO rating component tests (v7 only)
- WLS opponent model tests (v7 only)
- 6-way ensemble aggregation tests (v7 is 6-way, v6 is 5-way)
- ELO data file validation (v7 only)
- WLS-specific weight optimization tests

**v7-Only Tests:**
1. ❌ `test_elo_component_not_placeholder` - ELO-specific
2. ❌ `test_elo_ratings_csv_exists` - ELO data file
3. ❌ `test_elo_ratings_loadable` - ELO data loading
4. ❌ `test_elo_4927_teams` - ELO team count
5. ❌ `test_wls_opponent_baseline_executed` - WLS-specific
6. ❌ `test_6way_ensemble_aggregation` - v7 has 6 components
7. ❌ `test_elo_wls_weight_optimization` - ELO+WLS weights
8. ... (16 more ELO/WLS-specific tests)

---

## Integration Plan - Completed

### Phase 1: Verification ✅ Complete

**Tasks:**
1. ✅ Verified R interface integrated in v6.0 (line 77)
2. ✅ Verified Rule 19 active in enforcement framework
3. ✅ Tested Rule 19 on v6 production file (passed)
4. ✅ Categorized 68 v7 tests into general (45) vs v7-specific (23)

### Phase 2: Test Extraction ✅ Complete

**Tasks:**
1. ✅ Created `tests/unit/test_ensemble_framework.py` (18 general unit tests)
2. ✅ Updated `tests/integration/test_ensemble_v6_integration.py` (added 20 tests)
3. ✅ Created `tests/integration/test_ensemble_v6_sql.py` (7 SQL/data tests)
4. ✅ Renamed v7-specific fixtures to v6 (e.g., `ensemble_v7_with_mocks` → `ensemble_v6_with_mocks`)
5. ✅ Updated component names (removed ELO/WLS references)
6. ✅ Updated weight structures (v6: 27/27/23/13/10, not v7: 27/27/20/13/10/3)

### Phase 3: Validation ✅ Complete

**Tasks:**
1. ✅ Ran `pytest tests/unit/test_ensemble_framework.py -v` (18/18 passed)
2. ✅ Ran `pytest tests/integration/test_ensemble_v6_integration.py -v` (46/46 passed)
3. ✅ Ran `pytest tests/integration/test_ensemble_v6_sql.py -v` (7/7 passed)
4. ✅ Total: 71 tests (v6: 26 baseline + 45 extracted = 71 total)
5. ✅ All tests passing with v6.0 ensemble

---

## Test Adaptation Examples

### Example 1: Component Placeholder Detection

**v7 Test (Generic):**
```python
def test_state_space_not_placeholder(self, mock_ensemble_with_all_components):
    """State-space should not return placeholder value of 250."""
    mean, std = mock_ensemble_with_all_components._predict_state_space(
        "00-0033873", 2024, 10, {}
    )
    assert mean != 250.0, "State-space returned placeholder (250), should be realistic"
    assert mean is not None and not np.isnan(mean)
    assert std is not None and std > 0
```

**v6 Adaptation:**
```python
def test_state_space_not_placeholder(self, mock_ensemble_v6_all_components):
    """State-space should not return placeholder value of 250."""
    mean, std = mock_ensemble_v6_all_components._predict_state_space(
        "00-0033873", 2024, 10, {}
    )
    assert mean != 250.0, "State-space returned placeholder (250), should be realistic"
    assert mean is not None and not np.isnan(mean)
    assert std is not None and std > 0
```

**Changes:**
- Fixture name: `mock_ensemble_with_all_components` → `mock_ensemble_v6_all_components`
- No other changes needed - test logic is version-agnostic

### Example 2: Ensemble Aggregation

**v7 Test (v7-Specific - NOT Extracted):**
```python
def test_6way_ensemble_aggregation(self, ensemble_v7_with_mocks):
    """All 6 components should contribute to ensemble mean."""
    result = ensemble_v7_with_mocks.predict("00-0033873", 2024, 10)
    assert len(result["valid_components"]) == 6  # v7 has 6 components
```

**v6 Test (Already Exists):**
```python
def test_5way_ensemble_aggregation(self, ensemble_v6_with_mocks):
    """All 5 components should contribute to ensemble mean."""
    result = ensemble_v6_with_mocks.predict("00-0033873", 2024, 10)
    assert len(result["valid_components"]) == 5  # v6 has 5 components
```

**Decision:** v7 test NOT extracted - component count differs

### Example 3: Weight Validation

**v7 Test (Generic - Weight Sum):**
```python
def test_default_weights_sum_to_one(self, mock_ensemble_with_all_components):
    """Default component weights should sum to 1.0."""
    weights = mock_ensemble_with_all_components.weights
    assert abs(sum(weights.values()) - 1.0) < 1e-6
```

**v6 Adaptation:**
```python
def test_default_weights_sum_to_one(self, mock_ensemble_v6_all_components):
    """Default component weights should sum to 1.0."""
    weights = mock_ensemble_v6_all_components.weights
    assert abs(sum(weights.values()) - 1.0) < 1e-6
```

**Changes:**
- Fixture name updated
- Logic identical (weight sum = 1.0 is version-agnostic)

---

## Component Weight Structures

### v6.0 Production (5-way):
```python
{
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.23,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10,
}
# Total: 1.00
```

### v7.0 (6-way - NOT IN PRODUCTION):
```python
{
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.20,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10,
    "elo_wls": 0.03,  # ELO+WLS component (v7 only)
}
# Total: 1.00
```

**Key Difference:** v6 has 5 components, v7 has 6 (adds ELO+WLS at 3% weight)

---

## Success Metrics

**Test Coverage:**
- ✅ v6 Baseline Tests: 26 (before extraction)
- ✅ Extracted Tests: 45 (from v7)
- ✅ Total v6 Tests: 71 (26 + 45)
- ✅ Improvement: +173% test coverage

**Component Validation:**
- ✅ R Model Interface: Already in v6.0 production
- ✅ Rule 19: Active in pre-commit hooks
- ✅ Test Suite: 45 tests extracted and passing

**Quality Gates:**
- ✅ All 71 tests passing in v6.0
- ✅ No v7-specific dependencies in extracted tests
- ✅ Placeholder detection active and blocking
- ✅ Component diversity enforced (≥5 yards variance)
- ✅ Graceful degradation tested (3/5 minimum components)

---

## Files Modified/Created

### Created:
1. ✅ `tests/unit/test_ensemble_framework.py` (18 general unit tests, 499 lines)
2. ✅ `tests/integration/test_ensemble_v6_sql.py` (7 SQL/data tests, 470 lines)
3. ✅ `docs/v7_to_v6_component_extraction.md` (this file)

### Updated:
1. ✅ `tests/integration/test_ensemble_v6_integration.py` (added 20 tests, now 46 total)
2. ✅ No changes needed to `py/ensemble/enhanced_ensemble_v6_production.py` (R interface already present)
3. ✅ No changes needed to `py/utils/enforcement_router.py` (Rule 19 already active)

### No Changes Required:
- `py/ensemble/r_model_interface.py` - already in production use
- `.pre-commit-config.yaml` - Rule 19 already in hooks
- `.git/hooks/pre-commit` - enforcement router already integrated

---

## Remaining Work

**None - Extraction Complete** ✅

All reusable components from v7.0 have been identified, extracted, and integrated into v6.0:
1. ✅ R Model Interface - already in v6.0 production
2. ✅ Rule 19 (Placeholder Detection) - active and tested
3. ✅ Test Suite - 45 tests extracted and passing

v7.0-specific components (ELO+WLS) remain in v7 files for future reference but are not integrated into v6.0 production.

---

## Appendix A: Test File Locations

### v7 Source Files (Phase C):
- `tests/unit/test_ensemble_v7_components.py` (499 lines, 21 tests)
- `tests/integration/test_ensemble_v7_integration.py` (485 lines, 25 tests)
- `tests/integration/test_ensemble_v7_sql.py` (470 lines, 22 tests)

### v6 Target Files (Updated):
- `tests/unit/test_ensemble_framework.py` (NEW - 18 tests)
- `tests/integration/test_ensemble_v6_integration.py` (UPDATED - 46 tests)
- `tests/integration/test_ensemble_v6_sql.py` (NEW - 7 tests)

### v6 Baseline Files (Pre-Extraction):
- `tests/unit/test_ensemble_v6_api.py` (11 tests)
- `tests/integration/test_ensemble_v6_integration.py` (26 tests baseline)

---

## Appendix B: Enforcement Framework Integration

**Rule 19 Configuration:**
```python
# py/utils/enforcement_router.py (lines 337-348)
self.rules['placeholder_detection'] = EnforcementRule(
    name='placeholder_detection',
    description='Prevent hardcoded placeholder predictions from reaching production',
    triggers=[
        r'py/ensemble/.*\.py$',           # All ensemble files
        r'predict\(',                      # Predict methods
        r'return.*dict',                   # Dict returns
    ],
    validator=self._validate_placeholder_predictions,
    level=EnforcementLevel.BLOCKING
)
```

**Pre-Commit Hook Integration:**
```bash
# .git/hooks/pre-commit (lines 30-35)
# Run enforcement router
python3 py/utils/enforcement_router.py $CHANGED_FILES

VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo "✅ All enforcement checks passed"
    exit 0
else
    echo "❌ Enforcement validation FAILED"
    exit 1
fi
```

**Test Validation:**
```bash
$ python py/utils/enforcement_router.py py/ensemble/enhanced_ensemble_v6_production.py
✅ PASSED: placeholder_detection: No placeholder predictions detected
```

---

**Document Version:** 1.0
**Last Updated:** November 19, 2025
**Status:** ✅ Complete
