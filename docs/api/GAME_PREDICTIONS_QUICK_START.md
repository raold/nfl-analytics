# Game Predictions API - Quick Start Guide

**Last Updated:** November 21, 2025

## TL;DR

```bash
# Start API
uvicorn py.api.main:app --host 0.0.0.0 --port 8000

# Predict single game
curl -X POST http://localhost:8000/api/v1/predict/game/full \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_09_CLE_PIT", "season": 2024, "week": 9}'

# Predict entire week
curl -X POST http://localhost:8000/api/v1/predict/week \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "week": 9}'

# Retrieve stored prediction
curl http://localhost:8000/api/v1/predictions/game/2024_09_CLE_PIT
```

---

## Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/predict/game/full` | Generate full game prediction |
| POST | `/api/v1/predict/week` | Batch predict all games in week |
| GET | `/api/v1/predictions/game/{game_id}` | Retrieve stored prediction |

---

## Example Response

```json
{
    "game_id": "2024_09_CLE_PIT",
    "home_team": "PIT",
    "away_team": "CLE",
    "predicted_home_score": 24.3,
    "predicted_away_score": 20.1,
    "predicted_total": 44.4,
    "predicted_spread": 4.2,
    "predicted_home_win_prob": 0.635,
    "spread_edge": 0.7,
    "total_edge": 1.9,
    "ml_edge": 0.022
}
```

---

## Python Usage

```python
import requests

# Single game
response = requests.post(
    "http://localhost:8000/api/v1/predict/game/full",
    json={"game_id": "2024_09_CLE_PIT", "season": 2024, "week": 9}
)
pred = response.json()

print(f"Total: {pred['predicted_total']:.1f}")
print(f"Spread: {pred['predicted_spread']:+.1f}")
print(f"Home Win %: {pred['predicted_home_win_prob']:.1%}")
```

---

## Edge Interpretation

| Edge Type | Positive Value | Negative Value |
|-----------|----------------|----------------|
| `spread_edge` | Value on **home** team | Value on **away** team |
| `total_edge` | Value on **OVER** | Value on **UNDER** |
| `ml_edge` | Value on **home** ML | Value on **away** ML |

---

## Database

**Table:** `ml.game_predictions`

```sql
-- Query recent predictions
SELECT game_id, predicted_total, predicted_spread, spread_edge
FROM ml.game_predictions
WHERE season = 2024 AND week = 9
ORDER BY predicted_at DESC;

-- Check storage
SELECT COUNT(*) FROM ml.game_predictions;
```

---

## Status

- **Phase 1:** ✅ Mock predictions (production ready)
- **Phase 2:** 🚧 Real ensemble integration (planned Dec 2025)

---

**Full Docs:** `docs/api/GAME_PREDICTIONS_API.md`
