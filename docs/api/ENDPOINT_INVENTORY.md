# NFL Analytics API Endpoint Inventory

**Last Updated:** November 28, 2025
**Total Endpoints:** 19
**OpenAPI Coverage:** 6/19 (32%) - **14 endpoints need documentation**
**Status:** Operational (v6.1 Ensemble)

---

## Summary

| Category | Documented | Undocumented | Total |
|----------|------------|--------------|-------|
| Root Level | 5 | 2 | 7 |
| Game Predictions | 0 | 7 | 7 |
| Props Predictions | 0 | 5 | 5 |
| **Total** | **5** | **14** | **19** |

---

## 1. Root Level Endpoints (main.py)

### Documented in OpenAPI

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/` | GET | API root with links | No |
| `/health` | GET | Service health status | No |
| `/models` | GET | List loaded models | No |
| `/predict` | POST | Player prop prediction | No |
| `/metrics` | GET | Prometheus metrics | No |

### Undocumented (NEED ADDITION)

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/status` | GET | Status and metrics summary | No |
| `/predict/game` | POST | Game-level QB predictions | No |

---

## 2. Game Predictions (/api/v1)

**Router:** `py/api/routes/game_predictions.py`
**Status:** All 7 endpoints undocumented

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/api/v1/predict/game/full` | POST | Full game prediction (scores, spreads, totals, win probs) | No |
| `/api/v1/predict/week` | POST | Batch predict all games in a week | No |
| `/api/v1/predict/games/batch` | POST | Batch predict 1-50 games | No |
| `/api/v1/predictions/game/{game_id}` | GET | Retrieve stored prediction | No |
| `/api/v1/predict/game` | POST | QB passing yards for both teams | No |
| `/api/v1/predict/cache/invalidate` | POST | Invalidate prediction cache | No |
| `/api/v1/predict/cache/stats` | GET | Cache statistics | No |

---

## 3. Props Predictions (/api/v1)

**Router:** `py/api/routes/props_predictions.py`
**Status:** All 5 endpoints undocumented

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/api/v1/predict/passing_yards/{player_id}` | POST | Passing yards prediction (Ensemble v6.1 + A/B test) | No |
| `/api/v1/predict/rushing_yards/{player_id}` | POST | Rushing yards prediction (Ensemble v6.1) | No |
| `/api/v1/predict/receiving_yards/{player_id}` | POST | Receiving yards prediction (Ensemble v6.1) | No |
| `/api/v1/predict/touchdown_prob/{player_id}` | POST | TD probability (XGBoost v1) | No |
| `/api/v1/props/health` | GET | Props model health check | No |

---

## Request/Response Schemas

### GamePredictionRequest

```json
{
  "game_id": "2024_12_BUF_KC",
  "season": 2024,
  "week": 12
}
```

### GamePredictionResponse

```json
{
  "game_id": "2024_12_BUF_KC",
  "season": 2024,
  "week": 12,
  "home_team": "KC",
  "away_team": "BUF",
  "predicted_home_score": 27.5,
  "predicted_away_score": 24.2,
  "predicted_total": 51.7,
  "predicted_spread": 3.3,
  "predicted_home_win_prob": 0.612,
  "spread_std": 7.1,
  "total_std": 9.5,
  "sportsbook_spread": -3.5,
  "sportsbook_total": 52.5,
  "sportsbook_home_ml": -180,
  "sportsbook_away_ml": 155,
  "spread_edge": 0.2,
  "total_edge": -0.8,
  "ml_edge": 0.022,
  "predicted_at": "2024-11-28T12:00:00Z",
  "ensemble_version": "v6.1",
  "processing_ms": 105.3
}
```

### PropsPredictionRequest

```json
{
  "player_id": "00-0034844",
  "stat_type": "rushing_yards",
  "week": 12,
  "season": 2025,
  "confidence_level": 0.90
}
```

### PropsPredictionResponse

```json
{
  "player_id": "00-0034844",
  "player_name": "Derrick Henry",
  "stat_type": "rushing_yards",
  "week": 12,
  "season": 2025,
  "prediction": 95.3,
  "prediction_unit": "yards",
  "uncertainty": 18.5,
  "confidence_level": 0.90,
  "confidence_interval_lower": 62.8,
  "confidence_interval_upper": 127.8,
  "model_used": "ensemble_v6.1",
  "ab_group": "treatment",
  "calibration_score": 0.895,
  "timestamp": "2025-11-28T12:00:00Z",
  "processing_ms": 145.3
}
```

### BatchGamePredictionRequest

```json
{
  "games": [
    {"game_id": "2024_12_BUF_KC", "season": 2024, "week": 12},
    {"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12},
    {"game_id": "2024_12_PHI_LAR", "season": 2024, "week": 12}
  ]
}
```

### BatchGamePredictionResponse

```json
{
  "total_requested": 3,
  "total_successful": 3,
  "total_failed": 0,
  "predictions": [...],
  "errors": [],
  "processing_ms": 320.5,
  "cache_hit_rate": 0.67
}
```

---

## Example API Calls

### Full Game Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/game/full \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_12_BUF_KC",
    "season": 2024,
    "week": 12
  }'
```

### Week Predictions

```bash
curl -X POST http://localhost:8000/api/v1/predict/week \
  -H "Content-Type: application/json" \
  -d '{
    "season": 2024,
    "week": 12
  }'
```

### Batch Game Predictions

```bash
curl -X POST http://localhost:8000/api/v1/predict/games/batch \
  -H "Content-Type: application/json" \
  -d '{
    "games": [
      {"game_id": "2024_12_BUF_KC", "season": 2024, "week": 12},
      {"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12}
    ]
  }'
```

### Player Props - Passing Yards

```bash
curl -X POST http://localhost:8000/api/v1/predict/passing_yards/00-0036355 \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0036355",
    "stat_type": "passing_yards",
    "week": 12,
    "season": 2025,
    "confidence_level": 0.90
  }'
```

### Player Props - Rushing Yards

```bash
curl -X POST http://localhost:8000/api/v1/predict/rushing_yards/00-0034844 \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0034844",
    "stat_type": "rushing_yards",
    "week": 12,
    "season": 2025,
    "confidence_level": 0.90
  }'
```

### Get Stored Prediction

```bash
curl http://localhost:8000/api/v1/predictions/game/2024_12_BUF_KC
```

### Invalidate Cache

```bash
curl -X POST http://localhost:8000/api/v1/predict/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_12_BUF_KC",
    "season": 2024,
    "week": 12
  }'
```

### Cache Stats

```bash
curl http://localhost:8000/api/v1/predict/cache/stats
```

### Props Health Check

```bash
curl http://localhost:8000/api/v1/props/health
```

---

## Models Used

### Ensemble v6.1 (5-Way)

| Component | Weight | Description |
|-----------|--------|-------------|
| State-Space v2 | 27% | Bayesian state-space (Stan/cmdstanr) |
| Hierarchical v2 | 27% | Bayesian hierarchical (brms) |
| XGBoost v4 Ridge | 23% | Ridge-boosted XGBoost (4.70 MAE) |
| Informative Priors v2 | 13% | Bayesian informative priors (brms) |
| Hybrid Incremental | 10% | 70% static + 30% incremental blend |

### Temperature Scaling

- **T = 0.7083** (v6.1, corrected Nov 23)
- **Calibration:** 90% coverage on holdout

### A/B Testing (Props)

- **Control (50%):** Default v6.0 weights
- **Treatment (50%):** Dirichlet v8.0 weights (31.5% MAE improvement)
- **Assignment:** MD5 hash of player_id + week + season

---

## Performance

| Metric | Value |
|--------|-------|
| Cold Start | ~4s (first prediction) |
| Warm Predictions | ~105ms |
| Cache Hit | <1ms |
| Batch (16 games) | <2s (with cache) |

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Validation error (invalid request) |
| 404 | Game/player not found |
| 503 | Models not available |
| 500 | Internal server error |

---

## Next Steps

1. **Update OpenAPI spec** with 14 missing endpoints (6-8 hours)
2. **Add authentication** to protected endpoints (optional)
3. **Add rate limiting** per documentation claims

---

**Author:** Claude Code (Opus 4.5)
**Version:** 1.0
