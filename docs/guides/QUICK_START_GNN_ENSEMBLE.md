# GNN Ensemble Quick Start Guide

**Version:** 6.2 | **Date:** December 7, 2025

## Installation

No additional dependencies needed - GNN uses existing PyTorch setup.

## Quick Start

### 1. Basic Game Prediction

```python
from py.ensemble.enhanced_ensemble_v6_2_gnn import EnhancedEnsemble6WayGNN
import os

# Initialize
ensemble = EnhancedEnsemble6WayGNN(
    db_password=os.getenv("DB_PASSWORD"),
    enable_gnn=True
)

# Predict
result = ensemble.predict_game_outcome(
    home_team="KC",
    away_team="BUF",
    season=2024,
    week=1
)

print(f"KC Win Probability: {result['ensemble_mean']:.1%}")
print(f"90% CI: [{result['lower_90']:.1%}, {result['upper_90']:.1%}]")
```

### 2. Check GNN Availability

```python
if ensemble.gnn_available:
    print("✓ GNN ready")
else:
    print("✗ GNN unavailable (ensemble uses 5-way fallback)")
```

### 3. Player-Level Predictions (Unchanged)

```python
# Still uses 5-way ensemble (no GNN)
result = ensemble.predict(
    player_id="00-0019596",  # Patrick Mahomes
    season=2024,
    week=1
)
print(f"Passing Yards: {result['ensemble_mean']:.1f}")
```

## Testing

### Run Integration Tests
```bash
python scripts/test_gnn_ensemble_integration.py
```

### Run Unit Tests
```bash
pytest tests/unit/test_gnn_component.py -v
```

## Architecture

```
EnhancedEnsemble6WayGNN (v6.2)
├── GNNComponent (10% weight)           # NEW
├── StateSpace v2 (24% weight)          # Scaled down from 27%
├── Hierarchical v2 (24% weight)        # Scaled down from 27%
├── XGBoost v4 (20% weight)             # Scaled down from 23%
├── InformativePriors v2 (12% weight)   # Scaled down from 13%
└── Hybrid XGB (10% weight)             # Unchanged
```

## Weights

### v6.0 (5-way, no GNN)
```python
{
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.23,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10,
}
```

### v6.2 (6-way, with GNN)
```python
{
    "state_space_v2": 0.24,
    "hierarchical_v2": 0.24,
    "xgboost_v4_ridge": 0.20,
    "informative_priors_v2": 0.12,
    "hybrid_static_incremental": 0.10,
    "gnn_hierarchical_v1": 0.10,  # NEW
}
```

## Model Files

- **GNN Model:** `models/gnn/hierarchical_gnn_best.pt`
- **Metadata:** `models/gnn/hierarchical_gnn_metadata.json`
- **Performance:** AUC 0.61, Accuracy 55.9%, Brier 0.255

## Graceful Degradation

If GNN model unavailable:
1. `GNNComponent.is_available = False`
2. `ensemble.gnn_available = False`
3. `predict_game_outcome()` returns error message
4. Ensemble reverts to 5-way for player predictions
5. No crashes or exceptions

## Configuration Options

```python
ensemble = EnhancedEnsemble6WayGNN(
    enable_gnn=True,              # Enable/disable GNN
    gnn_model_path=None,          # Custom model path (or None for default)
    gnn_device="cpu",             # Device: "cpu" or "cuda"
    db_password="...",            # Database password
    # ... other v6.1 options
)
```

## Key Methods

### Game Outcome Prediction
```python
result = ensemble.predict_game_outcome(
    home_team: str,           # e.g., "KC", "SF"
    away_team: str,           # e.g., "BUF", "PHI"
    season: int,              # e.g., 2024
    week: int,                # 1-18
    return_components: bool   # Default: False
)
```

**Returns:**
```python
{
    "ensemble_mean": float,      # Win probability (0-1)
    "ensemble_std": float,       # Uncertainty
    "lower_90": float,           # 90% CI lower bound
    "upper_90": float,           # 90% CI upper bound
    "gnn_used": bool,            # Was GNN used?
    "valid_components": list,    # Which components contributed
}
```

### Player Prediction
```python
result = ensemble.predict(
    player_id: str,           # e.g., "00-0019596"
    season: int,              # e.g., 2024
    week: int,                # 1-18
    return_components: bool,  # Default: False
    return_uncertainty: bool  # Default: True
)
```

## Common Issues

### Issue: GNN model not found
**Solution:** Check that `models/gnn/hierarchical_gnn_best.pt` exists. If not, train GNN first:
```bash
python py/models/train_hierarchical_gnn.py --device cuda --batch-size 512
```

### Issue: Database connection failed
**Solution:** Check `DB_PASSWORD` environment variable is set:
```bash
export DB_PASSWORD="your_password"
```

### Issue: CUDA out of memory
**Solution:** Use CPU device:
```python
ensemble = EnhancedEnsemble6WayGNN(gnn_device="cpu", ...)
```

### Issue: Prediction returns None
**Solution:** Check `ensemble.gnn_available`. If False, GNN model failed to load.

## Performance Expectations

| Metric | GNN Standalone | 6-Way Ensemble (Expected) |
|--------|---------------|---------------------------|
| AUC | 0.61 | 0.62-0.65 |
| Accuracy | 55.9% | 56-58% |
| Brier Score | 0.255 | 0.24-0.25 |

## Next Steps

1. **Validate:** Run integration tests
2. **Baseline:** Generate predictions on 2024 test set
3. **Optimize:** Run Dirichlet weight optimization
4. **Deploy:** Monitor live performance in 2025

## Documentation

- **Full Guide:** `docs/GNN_ENSEMBLE_INTEGRATION.md`
- **Summary:** `GNN_ENSEMBLE_INTEGRATION_SUMMARY.md`
- **Code:** `py/ensemble/enhanced_ensemble_v6_2_gnn.py`
- **Tests:** `tests/unit/test_gnn_component.py`

## Support

See `CLAUDE.md` for project guidelines and conventions.

---

**Ready to use!** The GNN is integrated and ready for validation testing.
