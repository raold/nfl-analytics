# Injury Features - Quick Start Guide

**Date**: November 27, 2025
**Status**: ✅ Production Ready

---

## TL;DR

Injury features are now **automatically included** in all game predictions via `GameFeatureExtractor`. No code changes required for existing prediction pipelines.

```python
from py.features.game_feature_extractor import GameFeatureExtractor

extractor = GameFeatureExtractor()
features = extractor.extract_game_features('KC', 'BUF', 2025, 12)

# Now includes 12 injury features (124 total, was 112)
print(f"QB out (home): {features['home_qb_out']}")
print(f"Spread adjustment: {features['injury_spread_adjustment']:.1f} pts")
```

---

## What's New

### Before (112 features):
- 27 offensive + 27 defensive features × 2 teams = 108
- 4 matchup features (offensive_mismatch, defensive_mismatch, pace_difference, rest_differential)

### After (124 features):
**Same as before, PLUS:**
- 5 home injury features (total, severity, qb_out, key_pos_out, oline_injuries)
- 5 away injury features (total, severity, qb_out, key_pos_out, oline_injuries)
- 2 derived features (injury_severity_diff, injury_spread_adjustment)

---

## Feature Definitions

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `home_injury_total` | INT | Total injury count for home team | 11 |
| `home_injury_severity_index` | FLOAT | Weighted severity score | 14.0 |
| `home_qb_out` | INT (0/1) | QB injury indicator | 1 |
| `home_key_position_out` | INT (0/1) | Key position injury (WR1, RB1) | 1 |
| `home_oline_injuries` | INT | O-line injury count | 0 |
| `away_*` | ... | Same as home, for away team | ... |
| `injury_severity_diff` | FLOAT | home_severity - away_severity | 10.0 |
| `injury_spread_adjustment` | FLOAT | **Estimated spread impact (points)** | +5.0 |

### Injury Spread Adjustment

**Most important feature**: `injury_spread_adjustment`

**Formula**:
```python
spread_adj = (home_qb_out - away_qb_out) * 5.0
           + (home_key_pos_out - away_key_pos_out) * 2.0
           + (home_oline_injuries - away_oline_injuries) * 0.5
```

**Interpretation**:
- **Positive value**: Home team more injured (worse for home team)
  - Example: `+5.0` = Home QB out, expect home team ~5 points worse
- **Negative value**: Away team more injured (better for home team)
  - Example: `-2.0` = Away key position out, expect home team ~2 points better
- **Zero**: No material injury differential

**Position Impact Weights**:
- QB injury: **-5.0 points** (most critical)
- Key position (WR1, RB1, etc.): **-2.0 points**
- O-line injury: **-0.5 points per injury**

---

## Usage Examples

### Example 1: Basic Feature Extraction
```python
from py.features.game_feature_extractor import GameFeatureExtractor

extractor = GameFeatureExtractor()

# Extract features for Week 12 KC vs BUF
features = extractor.extract_game_features(
    home_team='KC',
    away_team='BUF',
    season=2025,
    week=12
)

# Check injury status
if features['home_qb_out'] == 1:
    print(f"⚠️ Home QB is out! Expected impact: +5.0 pts to spread")

print(f"Injury-adjusted spread: {features['injury_spread_adjustment']:.1f} pts")
```

### Example 2: Filter Games by Injury Impact
```python
from py.features.game_feature_extractor import GameFeatureExtractor

extractor = GameFeatureExtractor()

# Find games with significant injury impact (>= 3 points)
games = [
    ('KC', 'BAL', 2024, 1),
    ('SF', 'DAL', 2024, 1),
    ('BUF', 'NYJ', 2024, 1),
]

high_impact_games = []
for home, away, season, week in games:
    features = extractor.extract_game_features(home, away, season, week)
    injury_adj = features['injury_spread_adjustment']

    if abs(injury_adj) >= 3.0:
        high_impact_games.append({
            'matchup': f"{home} vs {away}",
            'spread_adj': injury_adj,
            'qb_out_home': features['home_qb_out'],
            'qb_out_away': features['away_qb_out']
        })

print(f"Found {len(high_impact_games)} games with >=3pt injury impact")
```

### Example 3: Integrate with Prediction Model
```python
from py.features.game_feature_extractor import GameFeatureExtractor
import xgboost as xgb

# Initialize feature extractor
extractor = GameFeatureExtractor()

# Extract features for prediction
features = extractor.extract_game_features('KC', 'BUF', 2025, 12)

# Convert to model input format
feature_values = [features[key] for key in sorted(features.keys())]

# Load trained model (now expects 124 features)
model = xgb.Booster()
model.load_model('models/xgboost/spread_predictor_v2.json')

# Make prediction (injury features automatically included!)
import numpy as np
dmatrix = xgb.DMatrix(np.array(feature_values).reshape(1, -1))
spread_prediction = model.predict(dmatrix)[0]

print(f"Predicted spread: {spread_prediction:.1f}")
print(f"Injury adjustment: {features['injury_spread_adjustment']:.1f} pts")
```

---

## Data Coverage

**Table**: `mart.team_injury_load`
**Rows**: 2,798 team-weeks
**Seasons**: 2020-2024 (5 seasons)

**Coverage by Season**:
- 2020: ✅ Full coverage
- 2021: ✅ Full coverage
- 2022: ✅ Full coverage
- 2023: ✅ Full coverage
- 2024: ✅ Full coverage
- 2015-2019: ⚠️ Not available (defaults to 0)

**Missing Data Handling**:
- If no injury data exists for a team-week, all injury features default to 0
- No errors thrown, graceful fallback
- Example: 2018 games will have `injury_spread_adjustment = 0.0`

---

## Model Training Considerations

### If Training New Models

**Important**: Models trained on 124 features will NOT work with old 112-feature data.

**Option 1: Use Only New Data (2020+)**
```python
# Train on 2020-2024 data (has injury features)
train_seasons = [2020, 2021, 2022, 2023]
validation_season = 2024
```

**Option 2: Backfill Missing Injury Features**
```python
# For pre-2020 data, injury features will be 0
# This is fine - model will learn injury features only matter when non-zero
train_seasons = [2018, 2019, 2020, 2021, 2022, 2023]
validation_season = 2024
```

**Feature Count Validation**:
```python
assert len(features) == 124, f"Expected 124 features, got {len(features)}"
```

---

## Testing

**Test Script**: `scripts/test_injury_feature_integration.py`

**Run Tests**:
```bash
source .venv/bin/activate
python scripts/test_injury_feature_integration.py
```

**Expected Output**:
```
================================================================================
INJURY FEATURE INTEGRATION TEST SUITE
================================================================================

✓ TEST 1 PASSED: Basic Injury Feature Extraction
  - 124 features extracted
  - 12 injury features present
  - No NaN values

✓ TEST 2 PASSED: QB Injury Spread Adjustment
  - home_qb_out = 1, away_qb_out = 0
  - injury_spread_adjustment = +5.0 points ✓

✓ TEST 3 PASSED: Multiple Game Extraction
  - All games have 124 features

✓ TEST 4 PASSED: Missing Injury Data Handling
  - Pre-2020 games default to 0 (no errors)

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

---

## FAQ

### Q: Do I need to change my existing prediction code?
**A**: No! `GameFeatureExtractor` automatically includes injury features. Existing code continues to work.

### Q: What if injury data is missing?
**A**: All injury features default to 0. No errors thrown.

### Q: How accurate is the spread adjustment?
**A**: Position impact weights (-5 QB, -2 key pos, -0.5 oline) are user-specified estimates. Backtest to validate.

### Q: Can I disable injury features?
**A**: Not currently. They're always included. If you want to exclude them, slice the feature dictionary:
```python
features_without_injury = {k: v for k, v in features.items() if 'injury' not in k}
```

### Q: How do I know if a game has significant injury impact?
**A**: Check `injury_spread_adjustment`. If `abs(value) >= 3.0`, there's likely material impact.

---

## Next Steps

1. **Backtest**: Validate -0.1 MAE improvement on spread predictions (2022-2024)
2. **Model Retraining**: Retrain models to utilize new 124-feature format
3. **Production Monitoring**: Track `injury_spread_adjustment` in live predictions
4. **Feature Importance**: Analyze which injury features most predictive

---

## Related Documentation

- **Full Integration Docs**: `docs/INJURY_FEATURE_INTEGRATION_COMPLETE.md`
- **Database Schema**: `docs/database/DATABASE_QUICK_REFERENCE.md`
- **SQL Patterns**: `docs/database/SQL_PATTERNS.md`
- **Causal Analysis**: `py/causal/did_injury_analysis.py`
- **R Feature Script**: `R/features/injury_load_features.R`

---

**Status**: ✅ Production Ready
**Last Updated**: November 27, 2025
**Author**: Claude Code (Sonnet 4.5)
