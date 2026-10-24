# R Model Interface Guide

**Quick Reference for rpy2 Integration**
**Version:** 1.0
**Date:** November 15, 2025

---

## Quick Start

### Basic Usage

```python
from nfl_analytics.ensemble.r_model_interface import RModelInterface

# Initialize interface
db_config = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": os.getenv("DB_PASSWORD")
}

interface = RModelInterface(db_config)

# Predict with Hierarchical v2
mean, std = interface.predict_hierarchical(
    player_id="00-0033873",  # Patrick Mahomes
    season=2024,
    week=9,
    model_path=Path("models/bayesian/passing_yards_hierarchical_v2.rds"),
    ndraws=1000
)

print(f"Prediction: {mean:.1f} ± {std:.1f} yards")
# Output: Prediction: 308.7 ± 60.6 yards
```

---

## Available Methods

### 1. `predict_hierarchical()`

Predict using Hierarchical Bayesian v2 model.

**Arguments:**
- `player_id` (str): NFLverse player ID (e.g., "00-0033873")
- `season` (int): Season year (e.g., 2024)
- `week` (int): Week number (1-18)
- `model_path` (Path): Path to hierarchical_v2.rds file
- `ndraws` (int, optional): Number of posterior draws (default: 1000)

**Returns:**
- `(mean, std)` tuple or `None` if prediction fails

**Example:**
```python
result = interface.predict_hierarchical(
    player_id="00-0033873",
    season=2024,
    week=9,
    model_path=Path("models/bayesian/passing_yards_hierarchical_v2.rds"),
    ndraws=500  # Faster inference
)

if result:
    mean, std = result
    print(f"Hierarchical: {mean:.1f} ± {std:.1f} yards")
else:
    print("Prediction failed")
```

---

### 2. `predict_informative_priors()`

Predict using Informative Priors v2 model.

**Arguments:** Same as `predict_hierarchical()`

**Returns:** `(mean, std)` tuple or `None`

**Example:**
```python
result = interface.predict_informative_priors(
    player_id="00-0033873",
    season=2024,
    week=9,
    model_path=Path("models/bayesian/passing_informative_priors_v2.rds")
)
```

---

### 3. `predict_state_space()`

Predict using State-Space Random Walk v2 (vectorized skills method).

**Arguments:**
- `player_id` (str): Player ID
- `season` (int): Season year
- `week` (int): Week number
- `skills_df` (DataFrame): Player skills with columns [player_id, skill_mean, skill_std]

**Returns:** `(mean, std)` tuple or `None`

**Example:**
```python
import pandas as pd

skills_df = pd.read_csv("models/bayesian/simplified_random_walk_skills_v2.csv")
result = interface.predict_state_space(
    player_id="00-0033873",
    season=2024,
    week=9,
    skills_df=skills_df
)
```

**Note:** This method does NOT call R (uses vectorized calculation), so it's much faster (~10ms vs ~1500ms).

---

## Integration with Ensemble v6.0

The ensemble automatically uses R interface for predictions:

```python
from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

ensemble = EnhancedEnsemble5Way()

result = ensemble.predict(
    player_id="00-0033873",
    season=2024,
    week=9,
    return_components=True
)

# Access individual R model predictions
print(f"State-Space: {result['components']['state_space_v2']['mean']:.1f} yards")
print(f"Hierarchical: {result['components']['hierarchical_v2']['mean']:.1f} yards")
print(f"Inf Priors: {result['components']['informative_priors_v2']['mean']:.1f} yards")
```

---

## Performance Considerations

### Inference Time Comparison

| Method | Time | Use Case |
|--------|------|----------|
| `predict_state_space()` (skills) | ~10 ms | Production, real-time |
| `predict_hierarchical()` | ~1,500 ms | Batch backtest |
| `predict_informative_priors()` | ~1,500 ms | Batch backtest |

**Optimization Tips:**

1. **Reduce ndraws for faster inference:**
   ```python
   # Fast (500 draws): ~750ms, slight quality loss
   result = interface.predict_hierarchical(..., ndraws=500)

   # Standard (1000 draws): ~1500ms, production quality
   result = interface.predict_hierarchical(..., ndraws=1000)

   # High quality (2000 draws): ~3000ms, research only
   result = interface.predict_hierarchical(..., ndraws=2000)
   ```

2. **Use vectorized state-space when possible:**
   ```python
   # FAST: Vectorized skills (10ms)
   if skills_df is not None:
       result = interface.predict_state_space(player_id, season, week, skills_df)

   # SLOW: R model fallback (1500ms)
   else:
       result = interface.predict_brms_model(...)
   ```

3. **Cache R model objects (advanced):**
   ```python
   # Load R model once, reuse for multiple predictions
   interface._init_rpy2()
   ro = interface._r_session
   ro.r(f'model <- readRDS("{model_path}")')

   # Now predict multiple times without reloading
   for player_id in player_ids:
       result = interface.predict_hierarchical(player_id, season, week, model_path)
   ```

---

## Error Handling

### Common Errors

#### 1. rpy2 ImportError
```python
ImportError: No module named 'rpy2'
```
**Fix:** `pip install rpy2`

#### 2. R Library Missing
```python
RRuntimeError: Error in library(brms) : there is no package called 'brms'
```
**Fix:**
```r
# In R console
install.packages("brms")
```

#### 3. Database Connection Error
```python
psycopg2.OperationalError: could not connect to server
```
**Fix:** Ensure PostgreSQL is running and DB_PASSWORD env var is set

#### 4. No Data for Player
```python
WARNING: No features found for player_id=..., season=..., week=...
```
**Fix:** Check if player played in that game (may be bye week or injury)

### Graceful Fallback Pattern

```python
# Best practice: 3-tier fallback
result = None

# Try R model
if interface.r_interface:
    result = interface.predict_hierarchical(player_id, season, week, model_path)

# Fallback to skills (state-space only)
if result is None and skills_df is not None:
    result = interface.predict_state_space(player_id, season, week, skills_df)

# Ultimate fallback: league average
if result is None:
    result = (225.0, 50.0)  # league average QB passing yards

mean, std = result
```

---

## Testing

### Run Integration Tests

```bash
# Full test suite (Test 1: R interface, Test 2: Ensemble v6.0)
python scripts/test_r_integration.py

# Expected output:
# Test 1 (R Interface): ✅ PASSED
# Test 2 (Ensemble v6.0): ✅ PASSED
# ✅ ALL TESTS PASSED - R integration ready for production
```

### Manual Testing

```python
# Test individual R model
from nfl_analytics.ensemble.r_model_interface import RModelInterface
from pathlib import Path

interface = RModelInterface(db_config)

result = interface.predict_hierarchical(
    player_id="00-0033873",
    season=2024,
    week=9,
    model_path=Path("models/bayesian/passing_yards_hierarchical_v2.rds"),
    ndraws=100  # Faster for testing
)

assert result is not None, "Prediction failed"
mean, std = result
assert 150 <= mean <= 400, f"Unreasonable prediction: {mean}"
assert 20 <= std <= 100, f"Unreasonable uncertainty: {std}"
print(f"✅ Test passed: {mean:.1f} ± {std:.1f} yards")
```

---

## Troubleshooting

### Issue: Slow Predictions

**Symptoms:** Predictions take >5 seconds

**Diagnosis:**
```python
import time

start = time.time()
result = interface.predict_hierarchical(...)
elapsed = time.time() - start
print(f"Inference time: {elapsed:.2f}s")
```

**Solutions:**
1. Reduce ndraws: 1000 → 500 (saves ~750ms)
2. Use vectorized state-space (10ms vs 1500ms)
3. Implement async parallel prediction (see Advanced Usage)

### Issue: Unreasonable Predictions

**Symptoms:** Predictions <100 or >500 yards

**Diagnosis:**
```python
result = interface.predict_hierarchical(...)
if result:
    mean, std = result
    if mean < 100 or mean > 500:
        print(f"⚠️ Unreasonable prediction: {mean:.1f} yards")
        print(f"   Player: {player_id}")
        print(f"   Season: {season}, Week: {week}")
```

**Solutions:**
1. Check if player actually played that game
2. Verify model file is not corrupted: `readRDS("model.rds")` in R
3. Check feature values (attempts, is_home, opponent, etc.)
4. Regenerate skills CSV if state-space prediction is wrong

### Issue: R Cleanup Error on Exit

**Symptoms:** Error message on script exit:
```
Error: unable to find an inherited method for function 'dbListConnections' for signature 'drv = "PqDriver"'
```

**Fix:** This is a harmless R cleanup error. Suppress with:
```python
import warnings
warnings.filterwarnings("ignore", category=rpy2.rinterface_lib.callbacks.RRuntimeWarning)
```

---

## Advanced Usage

### Batch Prediction (Future)

```python
# TODO: Implement batch prediction for backtest efficiency
def predict_batch(interface, games):
    """
    Predict multiple games with single R call.

    Args:
        games: List of dicts with keys [player_id, season, week]

    Returns:
        List of (mean, std) tuples
    """
    # Fetch all features at once
    features_df = fetch_features_batch(games)

    # Call R once for all games
    ro.r.assign("new_data", features_df)
    ro.r("posteriors <- posterior_predict(model, newdata = new_data, ndraws = 1000)")

    # Extract predictions for each game
    results = []
    for i in range(len(games)):
        ro.r(f"pred_mean <- mean(exp(posteriors[,{i+1}]))")
        ro.r(f"pred_sd <- sd(exp(posteriors[,{i+1}]))")
        mean = float(ro.r("pred_mean")[0])
        std = float(ro.r("pred_sd")[0])
        results.append((mean, std))

    return results
```

### Async Parallel Prediction (Future)

```python
import asyncio
import concurrent.futures

async def predict_all_models_async(player_id, season, week):
    """
    Run all 3 R models in parallel.

    Reduces inference time from 3s → 1.5s (3 models × 1.5s each → max(1.5s, 1.5s, 1.5s))
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        loop = asyncio.get_event_loop()

        # Submit all 3 R predictions
        futures = [
            loop.run_in_executor(executor, interface.predict_hierarchical, player_id, season, week, hier_path),
            loop.run_in_executor(executor, interface.predict_informative_priors, player_id, season, week, ip_path),
            loop.run_in_executor(executor, interface.predict_state_space, player_id, season, week, skills_df)
        ]

        # Wait for all to complete
        results = await asyncio.gather(*futures)

    return results
```

---

## Best Practices

### 1. Always Use Environment Variables for Passwords

```python
# ❌ BAD: Hardcoded password
db_config = {"password": "sicillionbillions"}

# ✅ GOOD: Environment variable
import os
from dotenv import load_dotenv
load_dotenv()

db_config = {"password": os.getenv("DB_PASSWORD")}
```

### 2. Validate Predictions

```python
def validate_prediction(mean, std, player_id):
    """Validate prediction is reasonable for NFL QB."""
    if mean < 100 or mean > 500:
        logger.warning(f"Unreasonable prediction for {player_id}: {mean:.1f} yards")
        return False

    if std < 10 or std > 150:
        logger.warning(f"Unreasonable uncertainty for {player_id}: {std:.1f} yards")
        return False

    return True
```

### 3. Log R Model Calls

```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Calling Hierarchical v2 for {player_id} S{season} W{week}")
result = interface.predict_hierarchical(...)
if result:
    logger.info(f"  Result: {result[0]:.1f} ± {result[1]:.1f} yards")
else:
    logger.warning(f"  Prediction failed, using fallback")
```

### 4. Handle Errors Gracefully

```python
try:
    result = interface.predict_hierarchical(...)
    if result is None:
        # Fallback strategy
        result = (225.0, 50.0)
except Exception as e:
    logger.error(f"R prediction failed: {e}")
    result = (225.0, 50.0)  # League average fallback
```

---

## FAQ

**Q: How do I reduce inference time?**
A: (1) Reduce ndraws to 500, (2) Use vectorized state-space, (3) Implement async parallel prediction

**Q: What if R model prediction fails?**
A: Ensemble uses 3-tier fallback: R model → skills → league average

**Q: Can I use this for rushing yards?**
A: Yes, but you'll need to train separate R models for rushing (different features)

**Q: How do I know if rpy2 is working?**
A: Run `python scripts/test_r_integration.py` - should see "✅ ALL TESTS PASSED"

**Q: What R libraries are required?**
A: brms, cmdstanr, DBI, RPostgres (see R/setup_packages.R)

---

## Related Documentation

- **Implementation:** `nfl_analytics/ensemble/r_model_interface.py`
- **Tests:** `scripts/test_r_integration.py`
- **Milestone:** `docs/milestones/R_MODEL_INTEGRATION_COMPLETE.md`
- **R Training:** `R/train_hierarchical_v2_production.R`, `R/train_informative_priors_v2_production.R`
- **Database Schema:** `docs/database/DATABASE_QUICK_REFERENCE.md`

---

**Last Updated:** November 15, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
