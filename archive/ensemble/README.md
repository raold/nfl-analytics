# Archived Ensemble Versions

**Date Archived:** November 15, 2025

These files were archived during API consolidation cleanup to resolve `EnhancedEnsemble6Way` import conflicts.

---

## Files

### 1. enhanced_ensemble_v6_DEPRECATED_BAK.py
- **Original Date:** November 14, 2025 14:06:56
- **Reason:** Deprecated architecture (BNN + Hybrid XGBoost)
- **Architecture:** 4-way ensemble (State-Space 30%, BNN 25%, Hierarchical 25%, Hybrid XGB 20%)
- **Status:** DEPRECATED - DO NOT USE

### 2. enhanced_ensemble_v6_NEW.py
- **Original Date:** November 14, 2025 13:59:23
- **Reason:** Experimental version, superseded by canonical
- **Architecture:** 5-way ensemble (State-Space 30%, Hierarchical 30%, XGBoost 25%, Informative 15%)
- **Status:** EXPERIMENTAL - Reference only

---

## Canonical Version

**File:** `nfl_analytics/ensemble/enhanced_ensemble_v6.py`
**Class:** `EnhancedEnsemble4Way` (production name)
**Architecture:** 4-way ensemble with XGBoost v4 Ridge Residuals
- State-Space v2: 30%
- Hierarchical v2: 25%
- XGBoost v4 Ridge: 20%
- Informative Priors v2: 25%

**Production Import:**
```python
from nfl_analytics.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way
```

---

## Research Code

**File:** `nfl_analytics/ensemble/enhanced_ensemble_v6_elo.py` (NOT archived)
**Class:** `EnhancedEnsemble6Way`
**Architecture:** 5-way ensemble with ELO + Adaptive WLS
**Status:** Active research, kept separate from production

---

## Migration Guide

If you have code importing deprecated versions:

### Before:
```python
from nfl_analytics.ensemble.enhanced_ensemble_v6_DEPRECATED_BAK import EnhancedEnsemble6Way
ensemble = EnhancedEnsemble6Way()
```

### After:
```python
from nfl_analytics.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way
ensemble = EnhancedEnsemble4Way()
```

---

## See Also

- `docs/API_CONSOLIDATION_ANALYSIS.md` - Full analysis of consolidation
- `nfl_analytics/ensemble/enhanced_ensemble_v6.py` - Canonical version
- `nfl_analytics/ensemble/enhanced_ensemble_v6_elo.py` - ELO research version
