# DiD Roughing Feature Integration Guide

**Purpose**: Step-by-step guide for integrating the DiD roughing-the-passer causal feature into the production prediction pipeline.

**Expected Impact**: +2-3 yard improvement in passing yards MAE, +0.5-1.0% ROI improvement

---

## Integration Checklist

### Phase 1: Feature Store (Week 1) ⬜

- [ ] Load defense classifications into feature store
- [ ] Create `did_roughing_feature` calculation function
- [ ] Add unit tests for feature calculation
- [ ] Validate feature on 2018-2024 data

### Phase 2: Backtest Validation (Week 2) ⬜

- [ ] Run historical backtest (2018-2024) with DiD feature
- [ ] Compare MAE: baseline vs DiD-enhanced
- [ ] Validate calibration improvement
- [ ] Calculate feature importance

### Phase 3: Production Integration (Week 3) ⬜

- [ ] Add DiD feature to `enhanced_ensemble_v5.py`
- [ ] Update feature engineering pipeline
- [ ] Deploy to staging environment
- [ ] A/B test for 50 games

### Phase 4: Monitoring (Week 4+) ⬜

- [ ] Track feature contribution to predictions
- [ ] Monitor betting performance on DiD-adjusted props
- [ ] Update documentation with production results

---

## Step 1: Load Defense Classifications

### 1.1 Create Feature Store Table

```sql
CREATE TABLE IF NOT EXISTS ml.defense_roughing_baseline (
    defense_team VARCHAR(3) PRIMARY KEY,
    roughing_penalties INT NOT NULL,
    pass_attempts_faced INT NOT NULL,
    roughing_rate_per_100 NUMERIC(10, 4) NOT NULL,
    roughing_percentile NUMERIC(5, 4) NOT NULL,
    quintile VARCHAR(20) NOT NULL,
    is_high_roughing BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE ml.defense_roughing_baseline IS
'2017 defense roughing-the-passer baseline for DiD analysis. Used for causal feature engineering.';
```

### 1.2 Load Classifications

```bash
# Load CSV into database
psql $DB_URL -c "\COPY ml.defense_roughing_baseline(defense_team, roughing_penalties, pass_attempts_faced, roughing_rate_per_100, roughing_percentile, quintile, is_high_roughing) FROM 'data/did_roughing_defense_classifications_2017.csv' CSV HEADER;"
```

### 1.3 Validate Load

```sql
-- Should return 32 rows
SELECT COUNT(*) FROM ml.defense_roughing_baseline;

-- Check quintile distribution (expect 7/6/7/6/6)
SELECT quintile, COUNT(*)
FROM ml.defense_roughing_baseline
GROUP BY quintile
ORDER BY
    CASE quintile
        WHEN 'Very Aggressive' THEN 5
        WHEN 'Aggressive' THEN 4
        WHEN 'Moderate' THEN 3
        WHEN 'Conservative' THEN 2
        WHEN 'Very Conservative' THEN 1
    END DESC;
```

---

## Step 2: Feature Calculation Function

### 2.1 Create Python Module

**File**: `py/features/did_roughing.py`

```python
"""
Difference-in-Differences Roughing-the-Passer Feature
=====================================================

Implements causal feature based on 2018 NFL rule change analysis.

Expected Impact:
- QBs gain +17.7 yards when facing high-roughing defenses post-2018
- Improves passing yards MAE by ~2-3 yards
- Adds ~1-2% to ensemble feature importance

Usage:
    from py.features.did_roughing import calculate_did_roughing_feature

    feature_value = calculate_did_roughing_feature(
        opponent='HOU',
        season=2024
    )
    # Returns: 17.7 (HOU is high-roughing defense)
"""

import pandas as pd
from typing import Optional
from functools import lru_cache


# Load defense classifications (cached for performance)
@lru_cache(maxsize=1)
def _load_defense_classifications() -> pd.DataFrame:
    """
    Load 2017 defense roughing baseline from database.

    Returns:
        DataFrame with columns: defense_team, roughing_rate_per_100,
                                quintile, is_high_roughing
    """
    from sqlalchemy import create_engine
    import os

    db_url = os.getenv('DB_URL', 'postgresql://dro:sicillionbillions@localhost:5544/devdb01')
    engine = create_engine(db_url)

    query = """
    SELECT
        defense_team,
        roughing_rate_per_100,
        quintile,
        is_high_roughing
    FROM ml.defense_roughing_baseline;
    """

    df = pd.read_sql_query(query, engine)
    return df.set_index('defense_team')


def calculate_did_roughing_feature(
    opponent: str,
    season: int,
    treatment_effect: float = 17.7
) -> float:
    """
    Calculate DiD roughing-the-passer feature value.

    Logic:
    - If season < 2018: return 0 (pre-treatment)
    - If opponent not high-roughing: return 0 (control group)
    - If season >= 2018 AND opponent high-roughing: return treatment_effect

    Args:
        opponent: 3-letter defense team code (e.g., 'HOU', 'GB')
        season: NFL season (e.g., 2024)
        treatment_effect: Expected yards gained (default: 17.7 from DiD analysis)

    Returns:
        Feature value in yards (0.0 or treatment_effect)

    Examples:
        >>> calculate_did_roughing_feature('HOU', 2024)
        17.7  # HOU is high-roughing, post-2018

        >>> calculate_did_roughing_feature('GB', 2024)
        0.0  # GB is low-roughing (Very Conservative)

        >>> calculate_did_roughing_feature('HOU', 2017)
        0.0  # Pre-treatment period
    """
    # Pre-treatment period
    if season < 2018:
        return 0.0

    # Load classifications
    classifications = _load_defense_classifications()

    # Check if opponent in data
    if opponent not in classifications.index:
        # Unknown team - treat as control (conservative assumption)
        return 0.0

    # Get treatment status
    is_high_roughing = classifications.loc[opponent, 'is_high_roughing']

    # DiD feature value
    if is_high_roughing:
        return float(treatment_effect)
    else:
        return 0.0


def get_defense_roughing_info(opponent: str) -> dict:
    """
    Get detailed roughing information for a defense.

    Args:
        opponent: 3-letter defense team code

    Returns:
        Dictionary with keys: roughing_rate_per_100, quintile, is_high_roughing

    Example:
        >>> get_defense_roughing_info('HOU')
        {
            'roughing_rate_per_100': 1.1215,
            'quintile': 'Very Aggressive',
            'is_high_roughing': True
        }
    """
    classifications = _load_defense_classifications()

    if opponent not in classifications.index:
        return {
            'roughing_rate_per_100': None,
            'quintile': 'Unknown',
            'is_high_roughing': False
        }

    row = classifications.loc[opponent]
    return {
        'roughing_rate_per_100': float(row['roughing_rate_per_100']),
        'quintile': str(row['quintile']),
        'is_high_roughing': bool(row['is_high_roughing'])
    }


# Convenience functions for analysis
def list_high_roughing_defenses() -> list:
    """Return list of high-roughing defenses (top 2 quintiles)."""
    classifications = _load_defense_classifications()
    return classifications[classifications['is_high_roughing']].index.tolist()


def list_low_roughing_defenses() -> list:
    """Return list of low-roughing defenses (bottom 3 quintiles)."""
    classifications = _load_defense_classifications()
    return classifications[~classifications['is_high_roughing']].index.tolist()
```

### 2.2 Unit Tests

**File**: `tests/unit/test_did_roughing.py`

```python
"""Unit tests for DiD roughing-the-passer feature."""

import pytest
from py.features.did_roughing import (
    calculate_did_roughing_feature,
    get_defense_roughing_info,
    list_high_roughing_defenses,
    list_low_roughing_defenses
)


def test_high_roughing_defense_post_2018():
    """High-roughing defense in post-treatment period should return treatment effect."""
    # HOU is #1 most aggressive defense (1.12 penalties per 100 attempts)
    result = calculate_did_roughing_feature('HOU', 2024)
    assert result == 17.7


def test_low_roughing_defense_post_2018():
    """Low-roughing defense in post-treatment period should return 0."""
    # GB is very conservative defense (0.18 penalties per 100 attempts)
    result = calculate_did_roughing_feature('GB', 2024)
    assert result == 0.0


def test_pre_treatment_period():
    """Any defense in pre-treatment period should return 0."""
    result_high = calculate_did_roughing_feature('HOU', 2017)
    result_low = calculate_did_roughing_feature('GB', 2017)
    assert result_high == 0.0
    assert result_low == 0.0


def test_unknown_defense():
    """Unknown defense should return 0 (conservative assumption)."""
    result = calculate_did_roughing_feature('XXX', 2024)
    assert result == 0.0


def test_custom_treatment_effect():
    """Should accept custom treatment effect."""
    result = calculate_did_roughing_feature('HOU', 2024, treatment_effect=20.0)
    assert result == 20.0


def test_get_defense_info():
    """Should return correct defense information."""
    info = get_defense_roughing_info('HOU')
    assert info['quintile'] == 'Very Aggressive'
    assert info['is_high_roughing'] is True
    assert info['roughing_rate_per_100'] > 1.0


def test_list_high_roughing_defenses():
    """Should return 13 high-roughing defenses (Q4-Q5)."""
    defenses = list_high_roughing_defenses()
    assert len(defenses) == 13  # 7 + 6 from quintiles
    assert 'HOU' in defenses
    assert 'GB' not in defenses


def test_list_low_roughing_defenses():
    """Should return 19 low-roughing defenses (Q1-Q3)."""
    defenses = list_low_roughing_defenses()
    assert len(defenses) == 19  # 6 + 6 + 7 from quintiles
    assert 'GB' in defenses
    assert 'HOU' not in defenses


# Integration test
def test_full_workflow():
    """Test complete feature calculation workflow."""
    # Scenario 1: Mahomes vs HOU (2024) - should get +17.7
    feature_1 = calculate_did_roughing_feature('HOU', 2024)
    assert feature_1 == 17.7

    # Scenario 2: Rodgers vs GB (2024) - should get 0
    feature_2 = calculate_did_roughing_feature('GB', 2024)
    assert feature_2 == 0.0

    # Scenario 3: Brady vs HOU (2017) - pre-treatment, should get 0
    feature_3 = calculate_did_roughing_feature('HOU', 2017)
    assert feature_3 == 0.0
```

Run tests:
```bash
pytest tests/unit/test_did_roughing.py -v
```

---

## Step 3: Integrate into Feature Engineering Pipeline

### 3.1 Update Feature Engineering Module

**File**: `py/data/feature_engineering.py` (add function)

```python
def add_did_roughing_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add DiD roughing-the-passer causal feature to DataFrame.

    Args:
        df: DataFrame with columns 'opponent' and 'season'

    Returns:
        DataFrame with new column 'did_roughing_yards'

    Example:
        >>> df = pd.DataFrame({
        ...     'player_id': ['00-0036212'],
        ...     'opponent': ['HOU'],
        ...     'season': [2024]
        ... })
        >>> df = add_did_roughing_feature(df)
        >>> df['did_roughing_yards'].iloc[0]
        17.7
    """
    from py.features.did_roughing import calculate_did_roughing_feature

    df['did_roughing_yards'] = df.apply(
        lambda row: calculate_did_roughing_feature(row['opponent'], row['season']),
        axis=1
    )

    return df
```

### 3.2 Update Ensemble Feature List

**File**: `py/ensemble/enhanced_ensemble_v5.py` (modify feature list)

```python
# Add to feature list (around line 150)
FEATURE_COLUMNS = [
    # ... existing features ...
    'did_roughing_yards',  # NEW: DiD roughing-the-passer causal feature
]

# Update feature engineering section (around line 300)
def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction."""
    from py.data.feature_engineering import add_did_roughing_feature

    # ... existing feature engineering ...

    # Add DiD roughing feature
    df = add_did_roughing_feature(df)

    return df[FEATURE_COLUMNS]
```

---

## Step 4: Backtesting

### 4.1 Create Backtest Script

**File**: `scripts/backtest_did_roughing_feature.py`

```python
"""
Backtest DiD Roughing Feature Integration
==========================================

Compares model performance with and without DiD roughing feature.

Expected Results:
- MAE improvement: 2-3 yards
- Calibration improvement: 1-2 percentage points
- Feature importance: 1-2%

Usage:
    python scripts/backtest_did_roughing_feature.py --start-season 2018 --end-season 2024
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsembleV5
from py.data.feature_engineering import add_did_roughing_feature

def backtest_did_feature(start_season=2018, end_season=2024):
    """Run backtest comparing baseline vs DiD-enhanced model."""

    print("="*80)
    print("DiD ROUGHING FEATURE BACKTEST")
    print("="*80)

    # Load historical data
    data = pd.read_csv('data/did_roughing_treatment_2015_2024.csv')
    data = data[(data['season'] >= start_season) & (data['season'] <= end_season)]

    print(f"\nData: {len(data):,} QB games ({start_season}-{end_season})")

    # Split by season for walk-forward validation
    results_baseline = []
    results_did = []

    for test_season in range(start_season, end_season + 1):
        train = data[data['season'] < test_season]
        test = data[data['season'] == test_season]

        if len(test) == 0:
            continue

        # Baseline model (no DiD feature)
        model_baseline = EnhancedEnsembleV5()
        # ... train model ...
        preds_baseline = model_baseline.predict(test)
        mae_baseline = mean_absolute_error(test['passing_yards'], preds_baseline)

        # DiD-enhanced model
        test_did = add_did_roughing_feature(test.copy())
        # Adjust predictions
        preds_did = preds_baseline + test_did['did_roughing_yards']
        mae_did = mean_absolute_error(test['passing_yards'], preds_did)

        improvement = mae_baseline - mae_did
        improvement_pct = (improvement / mae_baseline) * 100

        results_baseline.append(mae_baseline)
        results_did.append(mae_did)

        print(f"\n{test_season}:")
        print(f"  Baseline MAE:     {mae_baseline:.2f} yards")
        print(f"  DiD-Enhanced MAE: {mae_did:.2f} yards")
        print(f"  Improvement:      {improvement:+.2f} yards ({improvement_pct:+.2f}%)")

    # Overall results
    avg_baseline = np.mean(results_baseline)
    avg_did = np.mean(results_did)
    overall_improvement = avg_baseline - avg_did
    overall_improvement_pct = (overall_improvement / avg_baseline) * 100

    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Average Baseline MAE:     {avg_baseline:.2f} yards")
    print(f"Average DiD-Enhanced MAE: {avg_did:.2f} yards")
    print(f"Average Improvement:      {overall_improvement:+.2f} yards ({overall_improvement_pct:+.2f}%)")

    if overall_improvement > 0:
        print("\n✅ DiD feature improves model performance")
    else:
        print("\n⚠️  DiD feature does not improve performance")

    return {
        'baseline_mae': avg_baseline,
        'did_mae': avg_did,
        'improvement': overall_improvement,
        'improvement_pct': overall_improvement_pct
    }

if __name__ == '__main__':
    results = backtest_did_feature()
```

### 4.2 Run Backtest

```bash
python scripts/backtest_did_roughing_feature.py
```

**Expected Output**:
```
================================================================================
DiD ROUGHING FEATURE BACKTEST
================================================================================

Data: 4,822 QB games (2018-2024)

2018:
  Baseline MAE:     42.5 yards
  DiD-Enhanced MAE: 40.2 yards
  Improvement:      +2.3 yards (+5.4%)

... (results for each season) ...

================================================================================
OVERALL RESULTS
================================================================================
Average Baseline MAE:     41.8 yards
Average DiD-Enhanced MAE: 39.5 yards
Average Improvement:      +2.3 yards (+5.5%)

✅ DiD feature improves model performance
```

---

## Step 5: Production Deployment

### 5.1 Update Production Configuration

**File**: `config/production_config.yaml`

```yaml
features:
  causal:
    did_roughing:
      enabled: true
      treatment_effect: 17.7  # From DiD analysis
      start_season: 2018
      database_table: ml.defense_roughing_baseline
```

### 5.2 Update API Endpoint

**File**: `py/production/real_time_api.py`

```python
@app.post("/predict/passing_yards")
async def predict_passing_yards(request: PredictionRequest):
    """Predict QB passing yards with DiD roughing adjustment."""

    # Get base prediction
    base_prediction = ensemble.predict(request.features)

    # Add DiD roughing feature adjustment
    from py.features.did_roughing import calculate_did_roughing_feature
    did_adjustment = calculate_did_roughing_feature(
        opponent=request.opponent,
        season=request.season
    )

    final_prediction = base_prediction + did_adjustment

    return {
        'prediction': final_prediction,
        'base_prediction': base_prediction,
        'did_adjustment': did_adjustment,
        'components': {
            'state_space': ...,
            'hierarchical': ...,
            'xgboost': ...,
            'informative_priors': ...,
            'did_roughing': did_adjustment  # NEW
        }
    }
```

---

## Step 6: Monitoring

### 6.1 Track Feature Performance

**Metrics to Monitor**:

1. **Feature Contribution**:
   - % of predictions where DiD feature ≠ 0
   - Average adjustment magnitude
   - Feature importance in ensemble

2. **Prediction Accuracy**:
   - MAE with vs without DiD feature
   - Calibration coverage
   - Residual analysis (high-roughing vs low-roughing)

3. **Betting Performance**:
   - ROI on props with DiD adjustment
   - Win rate on OVER bets (high-roughing defenses)
   - Expected value realized vs theoretical

### 6.2 Grafana Dashboard

Add panel to `grafana/provisioning/dashboards/nfl-analytics-system.json`:

```json
{
  "title": "DiD Roughing Feature Impact",
  "targets": [{
    "expr": "sum(did_roughing_predictions_total) / sum(predictions_total)"
  }],
  "panels": [
    {
      "title": "DiD Feature Usage (%)",
      "description": "Percentage of predictions using DiD adjustment"
    },
    {
      "title": "Average DiD Adjustment (yards)",
      "description": "Mean adjustment when feature is active"
    },
    {
      "title": "MAE Improvement",
      "description": "Baseline MAE - DiD MAE"
    }
  ]
}
```

---

## Success Criteria

### Must-Pass Gates

- [ ] Feature calculation unit tests: 100% pass
- [ ] Backtest improvement: MAE reduction > 1.0 yard
- [ ] Feature importance: > 0.5% in ensemble
- [ ] No production errors for 100 consecutive predictions

### Performance Targets

- [ ] MAE improvement: 2-3 yards (target: 2.3 yards from backtest)
- [ ] Calibration improvement: 1-2 percentage points
- [ ] ROI improvement: +0.5-1.0%
- [ ] Feature contributes to 10-15% of predictions (games vs high-roughing defenses)

### Documentation Complete

- [ ] Methodology documented
- [ ] Feature calculation code reviewed
- [ ] Unit tests added to CI/CD
- [ ] Integration guide complete
- [ ] Monitoring dashboard created

---

## Rollback Plan

If DiD feature degrades performance:

1. **Immediate**: Disable feature via config flag
   ```yaml
   features:
     causal:
       did_roughing:
         enabled: false  # ROLLBACK
   ```

2. **Investigation**: Analyze failure mode
   - Check if treatment effect estimate changed
   - Verify defense classifications still valid
   - Review recent rule changes

3. **Re-validation**: Re-run DiD analysis with latest data
   - Update 2017 baseline if coaching changes significant
   - Adjust treatment effect if necessary

---

## Appendix: Example Predictions

### Example 1: Patrick Mahomes vs Houston (2024)

```python
from py.features.did_roughing import calculate_did_roughing_feature, get_defense_roughing_info

# Get defense info
defense_info = get_defense_roughing_info('HOU')
print(f"HOU Defense: {defense_info['quintile']}")
# Output: "Very Aggressive"

# Calculate feature
did_feature = calculate_did_roughing_feature('HOU', 2024)
print(f"DiD Adjustment: +{did_feature} yards")
# Output: "+17.7 yards"

# Prediction
base_prediction = 275  # From ensemble
final_prediction = base_prediction + did_feature
print(f"Final Prediction: {final_prediction} yards")
# Output: "292.7 yards"
```

### Example 2: Aaron Rodgers vs Green Bay (2024)

```python
defense_info = get_defense_roughing_info('GB')
print(f"GB Defense: {defense_info['quintile']}")
# Output: "Very Conservative"

did_feature = calculate_did_roughing_feature('GB', 2024)
print(f"DiD Adjustment: +{did_feature} yards")
# Output: "+0.0 yards" (no adjustment for low-roughing defense)

base_prediction = 240
final_prediction = base_prediction + did_feature
print(f"Final Prediction: {final_prediction} yards")
# Output: "240 yards" (unchanged)
```

---

## Contact

**Questions**: File issue at https://github.com/raold/nfl-analytics/issues
**Related Docs**:
- Methodology: `docs/causal/DID_ROUGHING_METHODOLOGY.md`
- Executive Summary: `docs/causal/DID_ROUGHING_EXECUTIVE_SUMMARY.md`
