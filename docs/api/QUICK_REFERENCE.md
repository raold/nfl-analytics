# NFL Analytics API - Quick Reference

**Base URL:** `http://localhost:8000`
**Interactive Docs:** `http://localhost:8000/docs`

---

## Most Common Endpoints

### Game Predictions

```bash
# Full game prediction (spread, total, scores, win prob)
curl -X POST http://localhost:8000/api/v1/predict/game/full \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_12_BUF_KC", "season": 2024, "week": 12}'

# Predict all games in a week
curl -X POST http://localhost:8000/api/v1/predict/week \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "week": 12}'

# Batch predict specific games (max 50)
curl -X POST http://localhost:8000/api/v1/predict/games/batch \
  -H "Content-Type: application/json" \
  -d '{"games": [
    {"game_id": "2024_12_BUF_KC", "season": 2024, "week": 12},
    {"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12}
  ]}'
```

### Player Props

```bash
# Passing yards (Patrick Mahomes)
curl -X POST http://localhost:8000/api/v1/predict/passing_yards/00-0036355 \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0036355",
    "stat_type": "passing_yards",
    "week": 12,
    "season": 2025,
    "confidence_level": 0.90
  }'

# Rushing yards (Derrick Henry)
curl -X POST http://localhost:8000/api/v1/predict/rushing_yards/00-0034844 \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0034844",
    "stat_type": "rushing_yards",
    "week": 12,
    "season": 2025,
    "confidence_level": 0.90
  }'

# Receiving yards
curl -X POST http://localhost:8000/api/v1/predict/receiving_yards/{player_id} \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0038120",
    "stat_type": "receiving_yards",
    "week": 12,
    "season": 2025,
    "confidence_level": 0.90
  }'

# Touchdown probability
curl -X POST http://localhost:8000/api/v1/predict/touchdown_prob/{player_id} \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "00-0036355",
    "stat_type": "touchdowns",
    "week": 12,
    "season": 2025,
    "confidence_level": 0.90
  }'
```

### Health & Status

```bash
# API health
curl http://localhost:8000/health

# Props models health
curl http://localhost:8000/api/v1/props/health

# Overall status
curl http://localhost:8000/status

# List models
curl http://localhost:8000/models

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Cache Management

```bash
# Invalidate specific game
curl -X POST http://localhost:8000/api/v1/predict/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"game_id": "2024_12_BAL_LAC", "season": 2024, "week": 12}'

# Invalidate entire week
curl -X POST http://localhost:8000/api/v1/predict/cache/invalidate \
  -H "Content-Type: application/json" \
  -d '{"season": 2024, "week": 12}'

# Cache statistics
curl http://localhost:8000/api/v1/predict/cache/stats
```

---

## Python Client

```python
import requests

class NFLClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def predict_game(self, game_id, season, week):
        """Get full game prediction."""
        response = requests.post(
            f"{self.base_url}/api/v1/predict/game/full",
            json={"game_id": game_id, "season": season, "week": week}
        )
        return response.json()

    def predict_passing_yards(self, player_id, week, season, confidence=0.90):
        """Get passing yards prediction."""
        response = requests.post(
            f"{self.base_url}/api/v1/predict/passing_yards/{player_id}",
            json={
                "player_id": player_id,
                "stat_type": "passing_yards",
                "week": week,
                "season": season,
                "confidence_level": confidence
            }
        )
        return response.json()

# Usage
client = NFLClient()

# Game prediction
game = client.predict_game("2024_12_BUF_KC", 2024, 12)
print(f"Spread: {game['predicted_spread']:.1f}")

# Player prediction
mahomes = client.predict_passing_yards("00-0036355", 12, 2024)
print(f"Mahomes: {mahomes['prediction']:.1f} ± {mahomes['uncertainty']:.1f} yards")
```

---

## JavaScript Client

```javascript
class NFLClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  async predictGame(gameId, season, week) {
    const response = await fetch(`${this.baseURL}/api/v1/predict/game/full`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId, season, week })
    });
    return await response.json();
  }

  async predictPassingYards(playerId, week, season, confidence = 0.90) {
    const response = await fetch(
      `${this.baseURL}/api/v1/predict/passing_yards/${playerId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          player_id: playerId,
          stat_type: 'passing_yards',
          week,
          season,
          confidence_level: confidence
        })
      }
    );
    return await response.json();
  }
}

// Usage
const client = new NFLClient();

const game = await client.predictGame('2024_12_BUF_KC', 2024, 12);
console.log(`Spread: ${game.predicted_spread}`);

const mahomes = await client.predictPassingYards('00-0036355', 12, 2024);
console.log(`Mahomes: ${mahomes.prediction} ± ${mahomes.uncertainty} yards`);
```

---

## Common Player IDs

| Player | Position | GSIS ID |
|--------|----------|---------|
| Patrick Mahomes | QB | 00-0036355 |
| Josh Allen | QB | 00-0034857 |
| Lamar Jackson | QB | 00-0035228 |
| Derrick Henry | RB | 00-0034844 |
| Christian McCaffrey | RB | 00-0033908 |
| Travis Kelce | TE | 00-0030506 |
| Tyreek Hill | WR | 00-0032764 |
| Davante Adams | WR | 00-0031503 |

---

## Response Fields

### Game Prediction
- `predicted_spread`: Point spread (positive = home favored)
- `predicted_total`: Total points
- `predicted_home_score`: Home team score
- `predicted_away_score`: Away team score
- `predicted_home_win_prob`: Win probability (0-1)
- `spread_std`: Spread uncertainty
- `total_std`: Total uncertainty
- `spread_edge`: Our spread - sportsbook spread
- `total_edge`: Our total - sportsbook total
- `ml_edge`: Our win prob - implied prob

### Player Props
- `prediction`: Point prediction (yards or probability)
- `uncertainty`: Standard deviation
- `confidence_interval_lower`: Lower bound
- `confidence_interval_upper`: Upper bound
- `model_used`: Model version
- `ab_group`: A/B test group (passing only)
- `calibration_score`: Calibration (if available)

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found (game/player doesn't exist) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (models not loaded) |

---

## Performance

| Scenario | Latency |
|----------|---------|
| Cache Hit | <1ms |
| Warm Prediction | ~105ms |
| Cold Start | ~4s |
| Batch 16 games (cached) | <2s |

---

## Full Documentation

- **Complete API Reference:** `/docs/api/API_REFERENCE.md`
- **Interactive Docs:** `http://localhost:8000/docs`
- **OpenAPI Spec:** `http://localhost:8000/openapi.json`

---

**Version:** 3.0.0
**Last Updated:** December 11, 2025
