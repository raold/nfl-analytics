# Data Pipeline Quick Start Guide

**Quick reference for common data pipeline operations**

## Quick Links

- **Full Documentation**: [docs/components/DATA_PIPELINE.md](../components/DATA_PIPELINE.md)
- **Database Reference**: [docs/database/DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md)
- **SQL Patterns**: [docs/database/SQL_PATTERNS.md](../database/SQL_PATTERNS.md)

---

## Common Tasks (Copy-Paste Ready)

### Task 1: Auto-Retrain Models for New Season

```bash
# Auto-detect latest completed season and retrain all models
python py/pipeline/auto_retrain.py --auto

# For specific season
python py/pipeline/auto_retrain.py --season 2024

# Dry run (see what would happen)
python py/pipeline/auto_retrain.py --auto --dry-run --verbose
```

**Expected output**:
```
[1/5] Detecting latest season... ✓ Season 2023 (100% complete)
[2/5] Regenerating features... ✓
[3/5] Retraining v3 model... ✓ Brier: 0.1985
[4/5] Retraining spread... ✓ MAE: 9.87
[5/5] Comparing models... ✓ Promoted to production
```

---

### Task 2: Refresh Predictions After Weekly Games

```bash
# Auto-detect completed week and refresh predictions
python py/pipeline/refresh_predictions.py --auto

# For specific week
python py/pipeline/refresh_predictions.py --season 2025 --week 5

# Predict 3 weeks ahead (default: 2)
python py/pipeline/refresh_predictions.py --auto --weeks-ahead 3

# Dry run
python py/pipeline/refresh_predictions.py --auto --dry-run
```

**Expected output**:
```
[1/5] Detecting completed week... ✓ Week 5 (14/14 games)
[2/5] Refreshing materialized views...
      ✓ mv_game_aggregates (2.1s)
      ✓ mv_team_rolling_stats (5.3s)
      ✓ mv_betting_features (12.4s)
[3/5] Generating win probability... ✓
[4/5] Generating spread predictions... ✓
[5/5] Logging refresh... ✓
```

---

### Task 3: Query Games Using Repository Pattern

```python
from py.data.repository.game_repository import GameRepository
from py.data.repository.base import DatabaseConfig

# Initialize
config = DatabaseConfig()  # Uses .env
repo = GameRepository(config)

# Get all games for season
result = repo.get_games_by_season(2024, cache_ttl=3600)
print(f"Found {result.row_count} games")

# Get specific week
result = repo.get_games_by_season(2025, week=5)
for game in result.rows:
    print(f"{game['home_team']} vs {game['away_team']}: "
          f"{game['home_score']} - {game['away_score']}")

# Get upcoming games
result = repo.get_upcoming_games(limit=10)
```

---

### Task 4: Fetch Latest Odds

```bash
# Fetch all prop odds for upcoming games
ODDS_API_KEY=your_key python py/data/fetch_prop_odds.py

# Fetch specific prop types
ODDS_API_KEY=your_key python py/data/fetch_prop_odds.py \
    --prop-types player_pass_yds player_rush_yds player_receptions

# Dry run (no database writes)
ODDS_API_KEY=your_key python py/data/fetch_prop_odds.py --dry-run

# Backfill historical data
ODDS_API_KEY=your_key python py/data/fetch_prop_odds.py --backfill
```

---

### Task 5: Start Continuous Odds Monitoring

```bash
# Start service (runs continuously during NFL season)
python py/data/odds_update_service.py

# Run once and exit (for testing)
python py/data/odds_update_service.py --once

# Custom poll interval (5 minutes)
python py/data/odds_update_service.py --poll-interval 300

# Set alert threshold for line moves
python py/data/odds_update_service.py --line-move-threshold 3.0

# Dry run
python py/data/odds_update_service.py --dry-run
```

---

### Task 6: Update Live Game Scores

```bash
# Update scores continuously (every 30 seconds)
python py/data/update_live_scores.py

# Update once and exit
python py/data/update_live_scores.py --once

# Custom update interval (60 seconds)
python py/data/update_live_scores.py --interval 60

# Dry run
python py/data/update_live_scores.py --dry-run
```

---

### Task 7: Execute Raw SQL Queries Safely

```python
from py.data.repository.base import BaseRepository

repo = BaseRepository()

# Simple query with caching
result = repo.query(
    "SELECT * FROM games WHERE season = %s AND week = %s",
    params=(2025, 5),
    cache_key="games_2025_week_5",
    cache_ttl=3600  # 1 hour
)

# Bulk update with cache invalidation
repo.execute(
    "UPDATE games SET home_score = %s WHERE game_id = %s",
    params=(31, 'game_123'),
    invalidate_cache=["game_123", "games_by_week"]
)

# Multiple operations in transaction
def update_game_and_stats(cursor):
    cursor.execute("UPDATE games SET home_score = %s WHERE game_id = %s",
                   (31, 'game_123'))
    cursor.execute("UPDATE team_stats SET wins = wins + 1 WHERE team = %s",
                   ('KC',))
    return cursor.rowcount

rows = repo.transaction(update_game_and_stats)
print(f"Updated {rows} rows")
```

---

### Task 8: Monitor Database Performance

```bash
# Check cache statistics
python -c "
from py.data.repository.base import BaseRepository
stats = BaseRepository.cache_stats()
print(f'Cache hit rate: {stats[\"hit_rate_pct\"]}%')
print(f'Backend: {stats[\"backend\"]}')
"

# Check connection pool status
python -c "
from py.data.repository.base import BaseRepository, DatabaseConfig
config = DatabaseConfig()
repo = BaseRepository(config)
print(f'Pool size: {repo._pool.getsize()}')
"

# Query slow queries (PostgreSQL)
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
SELECT query, calls, mean_exec_time FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;
EOF
```

---

### Task 9: Clear Cache When Data Changes

```python
from py.data.repository.base import BaseRepository

# Clear specific cache key
BaseRepository._cache.invalidate("games_2025_week_5")

# Clear all cache
BaseRepository.clear_cache()

# Check cache hit rate
stats = BaseRepository.cache_stats()
print(f"Hit rate: {stats['hit_rate_pct']}%")
```

---

### Task 10: Extract Game Trajectories for cVAE

```python
from py.data.cvae_data_pipeline import CVAEDataPipeline

pipeline = CVAEDataPipeline(db_config={
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions"
})

pipeline.connect()

# Extract trajectories
trajectories = pipeline.extract_game_trajectories(2024)
conditions = pipeline.extract_conditions(2024)

# Create dataloaders
train_loader, val_loader = pipeline.prepare_dataloaders(
    trajectories, conditions, batch_size=32
)

# Train model
for epoch in range(10):
    for trajectory, condition in train_loader:
        # Model training code here
        pass

pipeline.disconnect()
```

---

## Environment Setup

### Create .env file

```bash
cat > .env << 'EOF'
# Database
DB_HOST=localhost
DB_PORT=5544
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=sicillionbillions

# Connection pool
DB_MIN_CONNECTIONS=5
DB_MAX_CONNECTIONS=50
DB_TIMEOUT=30

# Odds service
ODDS_API_KEY=your_api_key_here
POLL_INTERVAL=900
LINE_MOVE_THRESHOLD=5.0

# Logging
LOG_LEVEL=INFO
EOF
```

### Verify connection

```bash
python -c "
from py.data.repository.base import BaseRepository, DatabaseConfig
config = DatabaseConfig()
repo = BaseRepository(config)
result = repo.query('SELECT 1 as test')
print('✓ Database connection OK')
"
```

---

## Common Errors & Solutions

### Error: "Use 'kickoff' column, not 'game_date'"

**Cause**: Query uses deprecated `game_date` column

**Fix**: Use `kickoff` instead
```python
# ❌ Wrong
"SELECT * FROM games WHERE game_date > %s"

# ✓ Correct
"SELECT * FROM games WHERE kickoff > %s"
```

---

### Error: "Connection pool exhausted"

**Cause**: Not returning connections to pool

**Fix**: Always use context manager
```python
# ✓ Correct: Connection returned automatically
with repo.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT ...")
```

---

### Error: "API quota exceeded"

**Cause**: Too many calls to The Odds API

**Fix**: Increase poll interval
```bash
# Poll every hour instead of 15 minutes
python py/data/odds_update_service.py --poll-interval 3600
```

---

### Error: "REFRESH MATERIALIZED VIEW timeout"

**Cause**: Missing unique index on view

**Fix**: Add index
```sql
CREATE UNIQUE INDEX mv_game_aggregates_idx ON mv_game_aggregates(game_id);
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_game_aggregates;
```

---

## Monitoring Checklist

- [ ] Database connection healthy
- [ ] Cache hit rate > 50%
- [ ] No long-running queries (> 5 min)
- [ ] Odds data updated (< 30 min old)
- [ ] Live scores current
- [ ] Materialized views recent
- [ ] Retraining logs clean
- [ ] API rate limit not exceeded

---

## Performance Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Cache hit rate | 50%+ | 65%+ |
| Query (cached) | <5ms | 2-3ms |
| Query (cold) | <50ms | 30-45ms |
| MV refresh (all) | <60s | 45-55s |
| Retrain cycle | <1h | 35-50m |

---

## Related Documentation

- [Database Schema Reference](../database/DATABASE_QUICK_REFERENCE.md)
- [SQL Patterns & Best Practices](../database/SQL_PATTERNS.md)
- [Component Architecture](../ARCHITECTURE_REFERENCE.md)
- [Ensemble Integration](./ENSEMBLE_INTEGRATION.md)
