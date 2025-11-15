# Temperature Scaler v6.0 Usage Guide

**Quick Reference for Production Deployment**

---

## Quick Start

### 1. Load Ensemble with Temperature Scaler

```python
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

# Initialize ensemble (automatically loads temperature_scaler_v6.pkl)
ensemble = EnhancedEnsemble5Way()

# Verify temperature loaded correctly
print(f"Temperature: {ensemble.temperature:.4f}")  # Should be 0.7083
```

### 2. Generate Calibrated Prediction

```python
# Get prediction for a player-game
result = ensemble.predict(
    player_id="00-0019596",  # Patrick Mahomes
    season=2024,
    week=10,
    return_components=False,
    return_uncertainty=True
)

print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['calibrated_std']:.1f} yards")
print(f"90% PI: [{result['lower_90']:.1f}, {result['upper_90']:.1f}]")
print(f"Temperature: {result['temperature']:.4f}")
```

**Expected Output:**
```
Prediction: 285.3 ± 42.1 yards
90% PI: [216.1, 354.5]
Temperature: 0.7083
```

---

## Temperature Scaler Files

### Location

```
models/calibration/
├── temperature_scaler_v6.pkl           # Scaler object (84 bytes)
└── temperature_scaler_v6_metadata.json  # Training metadata (816 bytes)
```

### Loading Manually

```python
import pickle

# Load temperature scaler
with open('models/calibration/temperature_scaler_v6.pkl', 'rb') as f:
    scaler_data = pickle.load(f)

print(f"Temperature: {scaler_data['temperature']:.4f}")
print(f"Target Coverage: {scaler_data['target_coverage']:.1%}")
print(f"Method: {scaler_data['method']}")
```

---

## Calibration Formula

Temperature scaling adjusts prediction variance:

```python
# Raw ensemble prediction
sigma_raw = ensemble_std

# Temperature scaling
T = 0.7083  # Optimal temperature
sigma_calibrated = sigma_raw / T

# 90% prediction interval
z_90 = 1.645  # 90% quantile of standard normal
lower_90 = mean - z_90 * sigma_calibrated
upper_90 = mean + z_90 * sigma_calibrated
```

**Note:** Division by T (not multiplication by √T) in ensemble code due to implementation choice.

---

## Verification

### Check Temperature Scaler Integration

```bash
python scripts/verify_temperature_scaler_v6.py
```

**Expected Output:**
```
✅ PASS Temperature Scaler
✅ PASS Ensemble Integration
✅ All checks passed - ensemble v6.0 ready for deployment
```

---

## Retraining

### Fast Training (2-3 minutes)

Uses existing ensemble predictions as proxy:

```bash
python scripts/train_temperature_scaler_v6_fast.py
```

**Use when:**
- Quick deployment needed
- Ensemble weights changed slightly
- Coverage target adjustment

### Full Training (30-45 minutes)

Generates v6 ensemble predictions from scratch:

```bash
python scripts/train_temperature_scaler_v6.py
```

**Use when:**
- Final production validation
- Ensemble architecture changed significantly
- After major model updates

---

## Troubleshooting

### Issue: Ensemble using fallback temperature (0.5534)

**Symptom:**
```python
ensemble = EnhancedEnsemble5Way()
print(ensemble.temperature)  # Shows 0.5534 instead of 0.7083
```

**Solution:**
```python
# Explicitly specify v6 scaler path
ensemble = EnhancedEnsemble5Way(
    temperature_scaler_path="models/calibration/temperature_scaler_v6.pkl"
)
```

### Issue: Temperature scaler not found

**Symptom:**
```
WARNING - Temperature scaler not found: models/calibration/temperature_scaler_v6.pkl
WARNING - Will attempt to load from v4 temperature scaler as fallback
```

**Solution:**
```bash
# Check file exists
ls -lh models/calibration/temperature_scaler_v6.pkl

# If missing, retrain
python scripts/train_temperature_scaler_v6_fast.py
```

### Issue: Coverage outside target range

**Symptom:**
Coverage < 88% or > 92% on new predictions

**Solution:**
```bash
# Retrain temperature scaler on latest data
python scripts/train_temperature_scaler_v6.py --start-season 2023 --end-season 2024
```

---

## API Integration

### FastAPI Endpoint

```python
from fastapi import FastAPI
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

app = FastAPI()
ensemble = EnhancedEnsemble5Way()  # Load on startup

@app.get("/predict")
def predict(player_id: str, season: int, week: int):
    result = ensemble.predict(
        player_id=player_id,
        season=season,
        week=week,
        return_uncertainty=True
    )

    return {
        "mean": result['ensemble_mean'],
        "std": result['calibrated_std'],
        "lower_90": result['lower_90'],
        "upper_90": result['upper_90'],
        "temperature": result['temperature']
    }
```

### Example Request

```bash
curl "http://localhost:8000/predict?player_id=00-0019596&season=2024&week=10"
```

**Response:**
```json
{
  "mean": 285.3,
  "std": 42.1,
  "lower_90": 216.1,
  "upper_90": 354.5,
  "temperature": 0.7083
}
```

---

## Calibration Monitoring

### Validate Coverage in Production

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load production predictions
df = pd.read_csv('production_predictions.csv')

# Calculate z-scores
z_scores = (df['actual'] - df['pred_mean']) / df['calibrated_std']

# Calculate coverage
alpha = 0.1  # 90% CI
lower_q = stats.norm.ppf(alpha / 2)
upper_q = stats.norm.ppf(1 - alpha / 2)
within_ci = ((z_scores >= lower_q) & (z_scores <= upper_q)).mean()

print(f"Coverage: {within_ci:.1%}")  # Should be 88-92%

if within_ci < 0.88:
    print("⚠️  Undercalibrated - intervals too narrow")
elif within_ci > 0.92:
    print("⚠️  Overcalibrated - intervals too wide")
else:
    print("✅ Calibration on target")
```

---

## Configuration

### Custom Temperature

For experimentation:

```python
# Override temperature manually
ensemble = EnhancedEnsemble5Way()
ensemble.temperature = 0.65  # Custom value

result = ensemble.predict(player_id="00-0019596", season=2024, week=10)
print(f"Custom temperature: {result['temperature']:.4f}")
```

**Warning:** Only use for testing. Production should use trained temperature scaler.

---

## Performance

### Inference Speed

Temperature scaling adds **<1ms** overhead:

```python
import time

ensemble = EnhancedEnsemble5Way()

start = time.time()
result = ensemble.predict(player_id="00-0019596", season=2024, week=10)
elapsed_ms = (time.time() - start) * 1000

print(f"Total inference: {elapsed_ms:.1f}ms")
print(f"Temperature overhead: <1ms")
```

---

## References

- **Training Script:** `scripts/train_temperature_scaler_v6_fast.py`
- **Verification:** `scripts/verify_temperature_scaler_v6.py`
- **Implementation:** `py/calibration/temperature_scaling.py`
- **Ensemble:** `py/ensemble/enhanced_ensemble_v6_production.py`
- **Milestone:** `docs/milestones/TEMPERATURE_SCALER_V6_TRAINING_COMPLETE.md`

---

**Last Updated:** November 15, 2025
