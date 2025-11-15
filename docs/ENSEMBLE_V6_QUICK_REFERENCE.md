# Ensemble v6 Quick Reference Card

**Last Updated:** November 15, 2025

---

## Production Version

### Import
```python
from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way
```

### Initialization
```python
ensemble = EnhancedEnsemble4Way(
    # Optional: Custom paths (defaults provided)
    ridge_path="models/ridge_residuals_v7.pkl",
    xgboost_path="models/xgb_ridge_residuals_v7.json",

    # Optional: Custom weights (if None, uses DEFAULT_WEIGHTS)
    weights={
        'state_space_v2': 0.30,
        'hierarchical_v2': 0.25,
        'xgboost_v4_ridge': 0.20,
        'informative_priors_v2': 0.25
    },

    # Optional: Temperature scaling (if None, no calibration)
    temperature_scaler_path="models/calibration/temperature_scaler_4way.pkl",

    # Optional: Database config (reads from .env if not provided)
    db_password=None  # Reads from DB_PASSWORD env var
)
```

### Prediction
```python
result = ensemble.predict(
    player_id="00-0019596",  # Patrick Mahomes
    season=2024,
    week=9
)

print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")
print(f"Components: {result['component_predictions']}")
```

### Architecture
- **4-way ensemble** (XGBoost v4 Ridge Residuals)
- State-Space v2: 30%
- Hierarchical v2: 25%
- XGBoost v4 Ridge: 20%
- Informative Priors v2: 25%

---

## Research Version (ELO + Adaptive WLS)

### Import
```python
from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way
```

### Initialization
```python
ensemble = EnhancedEnsemble6Way(
    # ELO ratings (optional)
    elo_ratings_path="models/elo/team_ratings.csv",

    # Adaptive WLS weights (optional)
    adaptive_weights_path="models/ensemble/adaptive_wls_weights.json",

    # Standard params same as production version
    db_password=None
)
```

### Architecture
- **5-way ensemble** (ELO + Adaptive WLS)
- State-Space v2: 27%
- Hierarchical v2: 27%
- XGBoost v2: 23%
- Informative Priors v2: 13%
- ELO Win/Loss: 10%

**Status:** Experimental (pending multi-season validation)

---

## Common Errors

### ❌ TypeError: use_adaptive_weights
```python
# WRONG - This parameter doesn't exist
ensemble = EnhancedEnsemble6Way(use_adaptive_weights=True)
```

**Solution:** Use `adaptive_weights_path` instead
```python
# CORRECT
ensemble = EnhancedEnsemble6Way(
    adaptive_weights_path="models/ensemble/adaptive_wls_weights.json"
)
```

---

### ❌ ImportError: EnhancedEnsemble6Way
```python
# WRONG - Production version exports EnhancedEnsemble4Way
from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way
```

**Solution:** Use correct class name
```python
# CORRECT
from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way
```

---

### ❌ ValueError: DB_PASSWORD not set
```python
# WRONG - Missing database password
ensemble = EnhancedEnsemble4Way()
```

**Solution:** Set DB_PASSWORD environment variable
```bash
# In .env file
DB_PASSWORD=your_password_here  # pragma: allowlist secret
```

---

## File Locations

| File | Purpose |
|------|---------|
| `py/ensemble/enhanced_ensemble_v6.py` | Production (canonical) |
| `py/ensemble/enhanced_ensemble_v6_elo.py` | Research (ELO + WLS) |
| `archive/ensemble/enhanced_ensemble_v6_DEPRECATED_BAK.py` | Archived (DO NOT USE) |
| `archive/ensemble/enhanced_ensemble_v6_NEW.py` | Archived (DO NOT USE) |

---

## See Also

- `docs/API_CONSOLIDATION_COMPLETE.md` - Full consolidation summary
- `docs/API_CONSOLIDATION_ANALYSIS.md` - Technical analysis
- `tests/unit/test_ensemble_v6_api.py` - API contract tests
- `archive/ensemble/README.md` - Migration guide
