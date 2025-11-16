# Test Quick Start Guide - NFL Analytics

**Date:** November 15, 2025
**Purpose:** Get started writing tests immediately
**Level:** Beginner to Intermediate

---

## The 5-Minute Setup

### 1. Install Testing Framework

```bash
# Already in requirements.txt, but just in case
pip install pytest pytest-cov pytest-mock hypothesis

# Verify installation
pytest --version
```

### 2. Understand Test Structure

```
tests/
├── conftest.py              # Shared fixtures & setup
├── unit/                    # Unit tests (fast, no DB)
├── integration/             # Integration tests (with DB)
├── e2e/                     # End-to-end tests
├── performance/             # Performance benchmarks
└── production/              # Production smoke tests
```

### 3. Run Existing Tests

```bash
# Run all tests
pytest

# Run fast tests only (skip slow)
pytest -m "not slow"

# Run P0 critical tests
pytest -m "critical"

# Run with output
pytest -v

# Run specific file
pytest tests/unit/test_ensemble_v6_api.py
```

---

## Writing Your First Test

### Template: Simple Unit Test

```python
# tests/unit/test_my_component.py
"""
Test suite for MyComponent

Author: Your Name
Date: November 15, 2025
"""

import pytest
from my_module import MyComponent


class TestMyComponent:
    """Test suite for MyComponent class."""

    def test_component_import(self):
        """Test component can be imported."""
        assert MyComponent is not None

    def test_component_init(self):
        """Test component initialization."""
        component = MyComponent(param1="value1")
        assert component is not None

    def test_happy_path(self):
        """Test basic functionality."""
        component = MyComponent(param1="value1")
        result = component.do_something()

        assert result is not None
        assert isinstance(result, dict)
        assert "key" in result

    def test_error_handling(self):
        """Test error handling."""
        component = MyComponent()

        with pytest.raises(ValueError):
            component.do_something(invalid_input=None)
```

### Template: Test with Mocking

```python
# tests/unit/test_with_mocks.py
"""Test suite with mocked dependencies."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestWithMocks:
    """Tests using mocks for external dependencies."""

    @patch('my_module.external_service')
    def test_with_mock(self, mock_service):
        """Test component with mocked service."""
        # Configure mock
        mock_service.return_value = {"status": "success"}

        # Use component
        from my_module import MyComponent
        component = MyComponent()
        result = component.call_service()

        # Assert
        assert result["status"] == "success"
        mock_service.assert_called_once()

    def test_with_mock_setup(self):
        """Test with explicit mock setup."""
        # Create mock
        mock_db = Mock()
        mock_db.query.return_value = [{"id": 1, "name": "test"}]

        # Use in component
        from my_module import MyComponent
        component = MyComponent(db=mock_db)
        result = component.get_data()

        # Verify mock was called
        assert len(result) == 1
        mock_db.query.assert_called()
```

### Template: Test with Fixtures

```python
# tests/unit/test_with_fixtures.py
"""Test suite using fixtures for reusable test data."""

import pytest
import pandas as pd
import numpy as np


class TestWithFixtures:
    """Tests using fixtures for common setup."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            'player_id': ['00-0019596'] * 10,
            'week': list(range(1, 11)),
            'yards': np.random.uniform(200, 300, 10),
        })

    @pytest.fixture
    def mock_ensemble(self):
        """Create mock ensemble."""
        from unittest.mock import Mock
        return Mock(predict=Mock(return_value={"mean": 250, "std": 30}))

    def test_with_data(self, sample_data):
        """Test with sample data fixture."""
        assert len(sample_data) == 10
        assert 'yards' in sample_data.columns

    def test_with_ensemble(self, mock_ensemble):
        """Test with mock ensemble fixture."""
        result = mock_ensemble.predict()
        assert result["mean"] == 250

    def test_with_multiple_fixtures(self, sample_data, mock_ensemble):
        """Test combining multiple fixtures."""
        # Use both fixtures
        assert len(sample_data) > 0
        result = mock_ensemble.predict()
        assert result is not None
```

---

## Running Tests

### Common Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_ensemble_v6_api.py

# Run specific test function
pytest tests/unit/test_ensemble_v6_api.py::TestEnsembleV6API::test_default_weights_structure

# Run tests matching pattern
pytest -k "ensemble"

# Run with coverage
pytest --cov=py

# Run and show coverage HTML
pytest --cov=py --cov-report=html
open htmlcov/index.html

# Run fast tests only (skip slow)
pytest -m "not slow"

# Run critical tests (P0)
pytest -m "critical"

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Show print statements
pytest -s
```

---

## Test Markers

Mark tests with decorators:

```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    """This test is slow, skip with: pytest -m 'not slow'"""
    pass

@pytest.mark.critical
def test_critical_path():
    """This test is critical, run with: pytest -m 'critical'"""
    pass

@pytest.mark.database
def test_with_database():
    """This test requires database"""
    pass

@pytest.mark.r_only
def test_requires_r():
    """This test requires R/rpy2"""
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """Skip this test"""
    pass

@pytest.mark.xfail(reason="Known issue")
def test_known_failure():
    """Expected to fail"""
    pass
```

---

## Assertions Cheatsheet

```python
# Basic assertions
assert x == 5
assert x != 5
assert x > 5
assert x < 5
assert x in [1, 2, 3]
assert x is not None
assert isinstance(x, dict)

# String assertions
assert "text" in "some text"
assert s.startswith("prefix")
assert s.endswith("suffix")

# List/dict assertions
assert len(my_list) == 5
assert "key" in my_dict
assert my_dict["key"] == "value"

# Float assertions (with tolerance)
assert abs(x - 5.0) < 0.01  # Within 0.01

# Exception assertions
with pytest.raises(ValueError):
    function_that_raises()

with pytest.raises(ValueError, match="error message"):
    function_that_raises()

# Boolean assertions
assert my_function() is True
assert my_function() is False

# Type assertions
assert isinstance(x, (list, tuple))
assert callable(my_function)

# Pandas assertions
pd.testing.assert_frame_equal(df1, df2)
pd.testing.assert_series_equal(s1, s2)

# NumPy assertions
np.testing.assert_array_equal(arr1, arr2)
np.testing.assert_almost_equal(arr1, arr2, decimal=3)
```

---

## Parametrized Tests

Run same test with different inputs:

```python
import pytest

class TestParametrized:
    """Tests using parametrization."""

    @pytest.mark.parametrize("player_id,expected_position", [
        ("00-0019596", "QB"),
        ("00-0036355", "QB"),
        ("00-0033816", "WR"),
    ])
    def test_player_position(self, player_id, expected_position):
        """Test player position (runs 3 times with different inputs)."""
        from my_module import get_player_position
        assert get_player_position(player_id) == expected_position

    @pytest.mark.parametrize("week,expected_games", [
        (1, 16),
        (10, 16),
        (9, 16),  # Bye weeks still have 16 games (just no bye team)
    ])
    def test_games_per_week(self, week, expected_games):
        """Test number of games per week."""
        assert expected_games == 16
```

---

## Testing Ensemble Component

### Step 1: Test Model Loads

```python
def test_ensemble_models_load():
    """Test all models load successfully."""
    from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way
    from unittest.mock import patch
    from pathlib import Path

    # Mock model files exist
    with patch.object(Path, 'exists', return_value=True):
        with patch('pickle.load', return_value=MagicMock()):
            ensemble = EnhancedEnsemble4Way(
                state_space_path="models/bayesian/simplified_random_walk_v2.rds",
                hierarchical_path="models/bayesian/passing_yards_hierarchical_v2.rds",
                informative_path="models/bayesian/passing_informative_priors_v2.rds",
                xgboost_path="models/xgboost/passing_xgboost_v2_canonical.pkl",
            )

    assert ensemble is not None
```

### Step 2: Test Weights

```python
def test_ensemble_weights():
    """Test ensemble weights configuration."""
    from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

    weights = EnhancedEnsemble4Way.DEFAULT_WEIGHTS

    # Should sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.01

    # Should have 4 models
    assert len(weights) == 4

    # All positive
    assert all(w > 0 for w in weights.values())
```

### Step 3: Test Prediction

```python
def test_ensemble_prediction_structure():
    """Test ensemble returns correct prediction structure."""
    from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way
    from unittest.mock import Mock

    ensemble = Mock()
    ensemble.predict.return_value = {
        "player_id": "00-0019596",
        "ensemble_mean": 245.5,
        "ensemble_std": 32.1,
        "component_predictions": {
            "state_space_v2": 243.0,
            "hierarchical_v2": 248.0,
            "xgboost_v2": 244.0,
            "informative_priors_v2": 246.0,
        }
    }

    result = ensemble.predict("00-0019596", 2024, 9)

    assert "player_id" in result
    assert "ensemble_mean" in result
    assert "ensemble_std" in result
    assert "component_predictions" in result
```

---

## Testing R Model Integration

### Step 1: Test Interface Imports

```python
def test_r_interface_imports():
    """Test R interface can be imported."""
    from py.ensemble.r_model_interface import RModelInterface
    assert RModelInterface is not None
```

### Step 2: Test R Session Initialization

```python
@patch('py.ensemble.r_model_interface.rpy2')
def test_r_session_init(mock_rpy2):
    """Test R session initialization."""
    from py.ensemble.r_model_interface import RModelInterface

    interface = RModelInterface({"host": "localhost"})
    assert not interface.rpy2_initialized  # Not initialized yet
```

### Step 3: Test Error Handling

```python
def test_missing_model_file():
    """Test handling of missing RDS file."""
    from py.ensemble.r_model_interface import RModelInterface

    interface = RModelInterface({"host": "localhost"})

    with pytest.raises(FileNotFoundError):
        interface._load_model("nonexistent.rds")
```

---

## Testing Data Leakage

### Test Future Data Prevention

```python
def test_future_data_prevention():
    """Test no future data leaks."""
    features_for_week_10 = {
        'week': 10,
        'games_through': 9,
        'rolling_weeks': [6, 7, 8, 9],
    }

    # No week 10+ data
    assert features_for_week_10['games_through'] < 10
    assert all(w < 10 for w in features_for_week_10['rolling_weeks'])
```

### Test Player Contamination Prevention

```python
def test_no_player_contamination():
    """Test same player not in train and test."""
    train_players = {'00-0019596', '00-0036355'}
    test_players = {'00-0033816', '00-0026209'}

    overlap = train_players & test_players

    # Should have no overlap
    assert len(overlap) == 0
```

---

## Common Issues & Solutions

### Issue 1: Import Errors

```python
# Problem: ModuleNotFoundError when running tests

# Solution: conftest.py already handles this
# The conftest.py in tests/ adds project root to path
# Just run: pytest from project root
```

### Issue 2: Database Connection Errors

```python
# Problem: Tests fail because database not running

# Solution 1: Skip database tests
pytest -m "not database"

# Solution 2: Start database
docker-compose up -d pg

# Solution 3: Use mocks instead
@patch('my_module.database')
def test_with_mock_db(mock_db):
    pass
```

### Issue 3: R/rpy2 Not Available

```python
# Problem: R tests fail because R not installed

# Solution 1: Skip R tests
pytest -m "not r_only"

# Solution 2: Install R
brew install r  # macOS

# Solution 3: Use mocks
@patch('py.ensemble.r_model_interface.rpy2')
def test_with_mock_r(mock_rpy2):
    pass
```

### Issue 4: Tests Run Slowly

```python
# Problem: Some tests are slow

# Solution: Skip slow tests
pytest -m "not slow"

# Solution: Run in parallel (need pytest-xdist)
pytest -n auto

# Solution: Focus on fast tests
pytest tests/unit
```

---

## Best Practices

### ✅ DO

```python
# DO: Clear, descriptive test names
def test_ensemble_prediction_within_bounds():
    pass

# DO: One assertion per test (usually)
def test_weight_sums_to_one():
    weights = {"a": 0.3, "b": 0.7}
    assert sum(weights.values()) == 1.0

# DO: Arrange-Act-Assert structure
def test_prediction():
    # Arrange
    ensemble = create_ensemble()
    player_id = "00-0019596"

    # Act
    result = ensemble.predict(player_id, 2024, 9)

    # Assert
    assert result["mean"] > 0

# DO: Use fixtures for reusable data
@pytest.fixture
def sample_data():
    return {"player_id": "00-0019596"}

# DO: Mock external dependencies
@patch('external_service')
def test_with_service(mock_service):
    pass
```

### ❌ DON'T

```python
# DON'T: Vague test names
def test_stuff():
    pass

# DON'T: Multiple assertions per test
def test_ensemble():
    assert weights == 1.0
    assert predictions > 0
    assert std > 0

# DON'T: Complex setup in test
def test_with_complex_setup():
    # 50 lines of setup
    pass

# DON'T: Test implementation details
def test_internal_variable():
    ensemble._internal_var = 5

# DON'T: Use real external services
def test_with_real_database():
    db = psycopg2.connect("postgresql://...")
    pass
```

---

## Next Steps

1. **Read Full Docs:** Review `TEST_COVERAGE_AUDIT_NOV15_2025.md` (comprehensive)
2. **Study Examples:** Check `TEST_IMPLEMENTATION_PLAN_NOV15_2025.md` (detailed examples)
3. **Start Writing:** Pick a component from P0 list and write 5-10 tests
4. **Run Tests:** Execute with `pytest -v`
5. **Check Coverage:** Run `pytest --cov=py` to see coverage gaps
6. **Iterate:** Refine tests based on feedback

---

## Quick Reference Files

| Document | Purpose |
|----------|---------|
| [TEST_COVERAGE_AUDIT_NOV15_2025.md](TEST_COVERAGE_AUDIT_NOV15_2025.md) | Comprehensive coverage analysis & 257 test cases |
| [TEST_IMPLEMENTATION_PLAN_NOV15_2025.md](TEST_IMPLEMENTATION_PLAN_NOV15_2025.md) | Detailed examples & implementation strategy |
| [TEST_QUICK_START_GUIDE.md](TEST_QUICK_START_GUIDE.md) | This file - quick start for developers |

---

## Questions?

- **How do I test component X?** → See TEST_COVERAGE_AUDIT.md for specific recommendations
- **What's the test for this?** → Search TEST_IMPLEMENTATION_PLAN.md for examples
- **How do I run just P0 tests?** → `pytest -m "critical"`
- **How do I mock R?** → See R Model Interface section above

---

**Generated:** November 15, 2025
**Level:** Beginner-Friendly
**Time to Productivity:** 30 minutes
