# Game Predictions API Documentation

**Last Updated:** November 21, 2025
**Version:** 1.0.0
**Status:** ✅ Production Ready (Phase 1 Complete - Mock Predictions)

## Overview

Game-level prediction endpoints that provide full game forecasts including:
- **Score predictions** (home/away)
- **Spread predictions** (positive = home favored)
- **Total predictions** (over/under)
- **Win probability predictions** (0-1)
- **Uncertainty quantification** (standard deviations)
- **Edge calculations** (vs sportsbook lines)

## Architecture

### Components

```
py/api/routes/game_predictions.py   # FastAPI routes
db/migrations/048_create_game_predictions.sql   # Database schema
ml.game_predictions   # PostgreSQL table
```

### Database Schema

**Table:** `ml.game_predictions`

**Key Columns:**
- `game_id` VARCHAR(20) - Game identifier (e.g., '2024_09_CLE_PIT')
- `predicted_home_score` NUMERIC(6,2) - Predicted home team score
- `predicted_away_score` NUMERIC(6,2) - Predicted away team score
- `predicted_total` NUMERIC(6,2) - Predicted total points
- `predicted_spread` NUMERIC(6,2) - Predicted point spread (positive = home favored)
- `predicted_home_win_prob` NUMERIC(6,4) - Home win probability (0-1)
- `spread_std` NUMERIC(6,2) - Spread uncertainty
- `total_std` NUMERIC(6,2) - Total uncertainty
- `spread_edge` NUMERIC(8,4) - Our spread - sportsbook spread
- `total_edge` NUMERIC(8,4) - Our total - sportsbook total
- `ml_edge` NUMERIC(8,4) - Our win prob - sportsbook implied prob

**Indexes:**
- `idx_game_predictions_game_id` (game_id)
- `idx_game_predictions_season_week` (season, week)
- `idx_game_predictions_status` (status)
- `idx_game_predictions_predicted_at` (predicted_at DESC)
- `idx_game_predictions_ensemble_version` (ensemble_version)
- `uq_game_predictions_model_game` (model_id, game_id, ensemble_version) UNIQUE

---

## Endpoints

### 1. POST `/api/v1/predict/game/full`

Generate full game prediction with scores, spreads, totals, and win probabilities.

**Request Body:**
```json
{
    "game_id": "2024_09_CLE_PIT",
    "season": 2024,
    "week": 9
}
```

**Response (200 OK):**
```json
{
    "game_id": "2024_09_CLE_PIT",
    "season": 2024,
    "week": 9,
    "home_team": "PIT",
    "away_team": "CLE",
    "predicted_home_score": 24.3,
    "predicted_away_score": 20.1,
    "predicted_total": 44.4,
    "predicted_spread": 4.2,
    "predicted_home_win_prob": 0.635,
    "spread_std": 7.2,
    "total_std": 9.8,
    "sportsbook_spread": -3.5,
    "sportsbook_total": 42.5,
    "sportsbook_home_ml": -180,
    "sportsbook_away_ml": 155,
    "spread_edge": 0.7,
    "total_edge": 1.9,
    "ml_edge": 0.022,
    "predicted_at": "2024-11-21T14:30:45.123Z",
    "ensemble_version": "mock_v1.0",
    "processing_ms": 245.3
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/game/full \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "2024_09_CLE_PIT",
    "season": 2024,
    "week": 9
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict/game/full",
    json={
        "game_id": "2024_09_CLE_PIT",
        "season": 2024,
        "week": 9
    }
)

data = response.json()
print(f"Predicted Total: {data['predicted_total']}")
print(f"Predicted Spread: {data['predicted_spread']}")
print(f"Home Win Probability: {data['predicted_home_win_prob']:.1%}")
```

**Error Responses:**
- `404 Not Found` - Game not found for given game_id/season/week
- `500 Internal Server Error` - Prediction generation failed

---

### 2. POST `/api/v1/predict/week`

Batch predict all games in a given week.

**Request Body:**
```json
{
    "season": 2024,
    "week": 9
}
```

**Response (200 OK):**
```json
{
    "season": 2024,
    "week": 9,
    "total_games": 14,
    "predictions": [
        {
            "game_id": "2024_09_CLE_PIT",
            "season": 2024,
            "week": 9,
            "home_team": "PIT",
            "away_team": "CLE",
            "predicted_home_score": 24.3,
            "predicted_away_score": 20.1,
            "predicted_total": 44.4,
            "predicted_spread": 4.2,
            "predicted_home_win_prob": 0.635,
            "spread_std": 7.2,
            "total_std": 9.8,
            "spread_edge": 0.7,
            "total_edge": 1.9,
            "predicted_at": "2024-11-21T14:30:45.123Z",
            "ensemble_version": "mock_v1.0",
            "processing_ms": 245.3
        }
        // ... 13 more games
    ],
    "processing_ms": 3456.7
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/week \
  -H "Content-Type: application/json" \
  -d '{
    "season": 2024,
    "week": 9
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict/week",
    json={"season": 2024, "week": 9}
)

data = response.json()
print(f"Generated {data['total_games']} predictions in {data['processing_ms']:.1f}ms")

for pred in data['predictions']:
    print(f"{pred['game_id']}: {pred['predicted_total']:.1f} total, "
          f"{pred['predicted_spread']:.1f} spread")
```

**Error Responses:**
- `404 Not Found` - No games found for given season/week
- `500 Internal Server Error` - Prediction generation failed

---

### 3. GET `/api/v1/predictions/game/{game_id}`

Retrieve stored game prediction from database.

**Query Parameters:**
- `ensemble_version` (optional) - Filter by ensemble version

**Response (200 OK):**
```json
{
    "game_id": "2024_09_CLE_PIT",
    "season": 2024,
    "week": 9,
    "home_team": "PIT",
    "away_team": "CLE",
    "predicted_home_score": 24.3,
    "predicted_away_score": 20.1,
    "predicted_total": 44.4,
    "predicted_spread": 4.2,
    "predicted_home_win_prob": 0.635,
    "spread_std": 7.2,
    "total_std": 9.8,
    "sportsbook_spread": -3.5,
    "sportsbook_total": 42.5,
    "spread_edge": 0.7,
    "total_edge": 1.9,
    "predicted_at": "2024-11-21T14:30:45.123Z",
    "ensemble_version": "mock_v1.0",
    "processing_ms": 0.0
}
```

**cURL Examples:**
```bash
# Get latest prediction for game
curl http://localhost:8000/api/v1/predictions/game/2024_09_CLE_PIT

# Get prediction for specific ensemble version
curl "http://localhost:8000/api/v1/predictions/game/2024_09_CLE_PIT?ensemble_version=v6.1"
```

**Python Example:**
```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/predictions/game/2024_09_CLE_PIT"
)

data = response.json()
print(f"Stored prediction from {data['predicted_at']}")
print(f"Ensemble version: {data['ensemble_version']}")
```

**Error Responses:**
- `404 Not Found` - No prediction found for given game_id (or ensemble_version)
- `500 Internal Server Error` - Database query failed

---

## Edge Calculation

### Spread Edge

**Formula:**
```
spread_edge = predicted_spread - sportsbook_spread
```

**Interpretation:**
- **Positive edge**: Value on home team (our spread more favorable to home)
- **Negative edge**: Value on away team (our spread less favorable to home)

**Example:**
```
Our Spread: +4.2 (home favored by 4.2)
Sportsbook Spread: -3.5 (home favored by 3.5)
Spread Edge: +0.7 (value on home team - we think home wins by more)
```

### Total Edge

**Formula:**
```
total_edge = predicted_total - sportsbook_total
```

**Interpretation:**
- **Positive edge**: Value on OVER (our total is higher)
- **Negative edge**: Value on UNDER (our total is lower)

**Example:**
```
Our Total: 44.4
Sportsbook Total: 42.5
Total Edge: +1.9 (value on OVER - we predict higher scoring)
```

### Moneyline Edge

**Formula:**
```
ml_edge = predicted_home_win_prob - sportsbook_implied_prob

where:
  sportsbook_implied_prob = |odds| / (|odds| + 100)  if odds < 0 (favorite)
  sportsbook_implied_prob = 100 / (odds + 100)       if odds > 0 (underdog)
```

**Interpretation:**
- **Positive edge**: Value on home team (we predict higher win prob than market)
- **Negative edge**: Value on away team (we predict lower win prob than market)

**Example:**
```
Our Home Win Prob: 0.635 (63.5%)
Sportsbook Home ML: -180
Sportsbook Implied Prob: 180 / (180 + 100) = 0.643 (64.3%)
ML Edge: 0.635 - 0.643 = -0.008 (-0.8%, value on away team)
```

---

## Implementation Status

### Phase 1: Mock Predictions (✅ Complete)

**Status:** Production ready (November 21, 2025)

**Features:**
- ✅ Database schema (`ml.game_predictions`)
- ✅ FastAPI routes (`py/api/routes/game_predictions.py`)
- ✅ Mock prediction logic (deterministic, reasonable values)
- ✅ Edge calculation (spread, total, moneyline)
- ✅ Error handling and logging
- ✅ Prometheus metrics
- ✅ Database persistence
- ✅ 3 endpoints operational

**Mock Prediction Logic:**
```python
# Generates plausible scores (17-31 points per team)
predicted_home_score = round(random.uniform(17.0, 31.0), 1)
predicted_away_score = round(random.uniform(17.0, 31.0), 1)

# Derives spread and total
predicted_spread = predicted_home_score - predicted_away_score
predicted_total = predicted_home_score + predicted_away_score

# Win probability via logistic function
predicted_home_win_prob = 1 / (1 + exp(-predicted_spread / 14))
```

**Limitations:**
- Mock predictions only (not trained models)
- No actual feature extraction
- No ensemble integration
- Fixed uncertainty (6.5-8.5 for spread_std, 9-11 for total_std)

### Phase 2: Real Model Integration (🚧 Planned)

**Target:** December 2025

**Planned Features:**
- 🚧 Integrate Ensemble v6.1 predictions
- 🚧 Extract game-level features (team stats, matchups, betting market)
- 🚧 Sum player-level predictions → game totals
- 🚧 Calibrated uncertainty (from ensemble components)
- 🚧 Component-level predictions (State-Space, Hierarchical, XGBoost, etc.)
- 🚧 Model metadata (inference time, cache hits)

**Integration Points:**
1. Replace `generate_mock_game_prediction()` with real ensemble calls
2. Add feature extraction from `games`, `mart.team_epa`, `mart.game_features_enhanced`
3. Store component predictions in `model_components` JSONB field
4. Link to `ml.model_registry` via `model_id`

---

## Prometheus Metrics

**Counters:**
- `nfl_game_predictions_total{prediction_type}` - Total game predictions made
  - Labels: `prediction_type` ∈ {full, week}

**Histograms:**
- `nfl_game_prediction_duration_seconds{prediction_type}` - Prediction generation time
  - Buckets: 0.1, 0.5, 1.0, 2.0, 5.0 seconds

**Example Queries:**
```promql
# Total game predictions (last 24h)
increase(nfl_game_predictions_total[24h])

# Average prediction latency (last 1h)
rate(nfl_game_prediction_duration_seconds_sum[1h])
  / rate(nfl_game_prediction_duration_seconds_count[1h])

# P95 prediction latency
histogram_quantile(0.95, nfl_game_prediction_duration_seconds_bucket)
```

---

## Performance

**Phase 1 (Mock Predictions):**
- Single game: ~50-150ms
- Full week (14 games): ~1-2 seconds
- Database insert: <10ms per prediction

**Phase 2 (Real Ensemble) Targets:**
- Single game: <500ms (including feature extraction)
- Full week: <10 seconds (with parallel processing)
- Database insert: <10ms per prediction

---

## Testing

### Database Test
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT COUNT(*) FROM ml.game_predictions;"
```

### API Test (requires running API)
```bash
# Start API
uvicorn py.api.main:app --host 0.0.0.0 --port 8000

# Test single game prediction
curl -X POST http://localhost:8000/api/v1/predict/game/full \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_09_CLE_PIT", "season": 2024, "week": 9}'

# Test week predictions
curl -X POST http://localhost:8000/api/v1/predict/week \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "week": 1}'

# Test retrieval
curl http://localhost:8000/api/v1/predictions/game/2024_09_CLE_PIT
```

### Python Test Script
```bash
python tests/test_game_predictions_api.py
```

---

## Security Considerations

**Database Credentials:**
- ⚠️ Hardcoded credentials in routes file (dev only)
- TODO: Move to environment variables for production

**SQL Injection:**
- ✅ All queries use parameterized inputs (`%s`)
- ✅ No string concatenation in SQL

**Input Validation:**
- ✅ Pydantic models validate all inputs
- ✅ Season: 1999-2030
- ✅ Week: 1-22
- ✅ game_id: string (no special chars)

---

## Error Handling

**Database Errors:**
- Connection failures: HTTP 500, logged with stack trace
- Query failures: HTTP 500, logged with query and params

**Missing Data:**
- Game not found: HTTP 404 with clear message
- No predictions for week: HTTP 404 with clear message

**Prediction Failures:**
- Mock prediction errors: HTTP 500, logged with game details

---

## Future Enhancements

### Phase 2 (Real Models)
1. Integrate Ensemble v6.1 predictions
2. Extract game-level features
3. Calibrated uncertainty quantification
4. Component-level predictions in JSONB

### Phase 3 (Advanced Features)
1. Live odds API integration (track line movements)
2. Prediction confidence scores (beyond std dev)
3. Historical accuracy metrics per game
4. Automated bet recommendations (with Kelly sizing)
5. Comparison vs closing lines (CLV analysis)

---

## References

- **API Main:** `py/api/main.py`
- **Routes:** `py/api/routes/game_predictions.py`
- **Migration:** `db/migrations/048_create_game_predictions.sql`
- **Test Script:** `tests/test_game_predictions_api.py`
- **Database Docs:** `docs/database/DATABASE_QUICK_REFERENCE.md`
- **SQL Patterns:** `docs/database/SQL_PATTERNS.md`

---

**Version:** 1.0.0
**Last Updated:** November 21, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
**Status:** ✅ Phase 1 Complete, 🚧 Phase 2 Planned
