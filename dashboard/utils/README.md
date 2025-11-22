# Dashboard Utilities

This directory contains shared utilities for the NFL Analytics dashboard.

## Modules

### api_client.py

Type-safe, validated API client for the FastAPI prediction service.

**Key Features:**
- Automatic response validation using TypedDict definitions
- Graceful error handling (returns None on failure)
- Optional Redis caching for predictions
- Detailed error logging

**Quick Start:**
```python
from dashboard.utils.api_client import APIClient

api = APIClient()
prediction = api.get_game_prediction("2024_12_KC_LV", 2024, 12)

if prediction:
    home_yards = prediction['home_qb_yards']
    away_yards = prediction['away_qb_yards']
```

**Methods:**
- `get_health()` - Check API health status
- `get_models()` - List loaded models
- `get_player_prediction()` - Get single player prediction
- `get_game_prediction()` - Get game-level QB passing yards predictions

**Documentation:**
- See `docs/guides/API_CLIENT_INTEGRATION_GUIDE.md` for comprehensive guide
- See `docs/API_CLIENT_IMPLEMENTATION_SUMMARY.md` for implementation details

## Future Modules

- `streamlit_utils.py` - Streamlit-specific utilities (caching, session state, etc.)
- `metrics.py` - Dashboard metrics and KPI calculations
- `visualization.py` - Plotting utilities for common dashboard charts
