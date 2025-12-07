# GNN Graph Builder (CUDA) - Handoff Document

**Date:** December 7, 2025
**Source Machine:** Windows 11 (Desktop)
**Target Machine:** Mac M4 (Laptop)
**Status:** Working

---

## Summary

Successfully configured and ran the NFL GNN Graph Builder for seasons 2020-2024. The graph builder creates heterogeneous graphs with player, position, and team nodes for use in hierarchical GNN training.

---

## Infrastructure Changes

### PostgreSQL Upgrade (PG16 → PG17)

**Reason:** Database dump was created with PostgreSQL 17 (format 1.16), incompatible with PG16.

**Change in `docker-compose.yml`:**
```yaml
# Before
image: timescale/timescaledb:latest-pg16

# After
image: timescale/timescaledb:latest-pg17
```

### Database Restore

**Dump File:** `backups/nfl_analytics_20251207_185613.dump` (84MB, Git LFS)

**Restore Command:**
```bash
docker cp backups/nfl_analytics_20251207_185613.dump nfl-analytics-pg-1:/tmp/restore.dump
docker exec nfl-analytics-pg-1 pg_restore -U dro -d devdb01 --no-owner --no-privileges --jobs=4 /tmp/restore.dump
```

---

## Code Changes

### 1. `py/models/gnn_graph_builder_cuda.py` - Multi-Season Support

**Lines 884-925:** Updated `main()` to build graphs for 2020-2024:

```python
def main():
    # Build multi-season graph for 2020-2024
    seasons = [2020, 2021, 2022, 2023, 2024]

    db = NFLDatabase()
    builder = NFLGraphBuilder(db)

    graph = builder.build_multi_season_graph(
        seasons=seasons,
        include_chemistry=True,
        include_matchups=False
    )

    # Load games for all seasons
    all_games = []
    for season in seasons:
        season_games = db.load_games(season=season)
        all_games.append(season_games)
    games = pd.concat(all_games, ignore_index=True)
```

### 2. `py/models/gnn_graph_builder_cuda.py` - Games Query Fix

**Lines 159-174:** Fixed `load_games()` query (game_type column was empty):

```python
# Before
WHERE season = {season}
  AND game_type = 'REG'

# After
WHERE season = {season}
  AND week <= 18
```

---

## Results

### Graph Structure

| Metric | Value |
|--------|-------|
| **Total Nodes** | 434 |
| Player Nodes | 392 |
| Position Nodes | 10 |
| Team Nodes | 32 |

### Edge Counts

| Edge Type | Count |
|-----------|-------|
| player_to_position | 392 |
| position_to_team | 112 |
| player_chemistry (QB-WR) | 712 |
| player_matchup | 0 (not implemented) |
| team_to_game | 0 (not implemented) |

### Data Volume

| Data Source | Records |
|-------------|---------|
| Player-game stats | 13,550 |
| NextGen passing | 3,026 |
| NextGen rushing | 3,055 |
| NextGen receiving | 7,469 |
| Player info | 24,341 |
| Games (2020-2024) | 1,349 |

### Player Features

- **Dimensions:** 51 features per player
- **Normalization:** Z-score normalized
- **Feature Categories:**
  - Basic stats (9): attempts, yards, TDs, completion%, etc.
  - NextGen Passing (14): passer_rating, air_yards, aggressiveness, etc.
  - NextGen Rushing (10): efficiency, time_to_los, RYOE, etc.
  - NextGen Receiving (13): separation, cushion, YAC, etc.
  - Player Info (4): weight, experience, draft_pick, age
  - Bayesian Ratings (1): rating_mean

---

## Database Schema

### Key Tables (Row Counts)

| Table | Rows |
|-------|------|
| games | 7,263 |
| plays | 1,264,341 |
| nextgen_passing | 5,696 |
| nextgen_rushing | 5,794 |
| nextgen_receiving | 14,187 |
| players | 24,341 |

### Materialized Views

| View | Purpose |
|------|---------|
| `mart.player_hierarchy` | 24,341 players with position groups |
| `mart.player_game_stats` | Aggregated player-game stats |
| `mart.game_summary` | Game-level aggregations |

---

## Hardware Performance (Windows Desktop)

| Metric | Value |
|--------|-------|
| **Platform** | Windows 11 |
| **Database** | PostgreSQL 17 (TimescaleDB) in Docker |
| **Container** | nfl-analytics-pg-1 |
| **Port** | 5544 |
| **Graph Build Time** | ~15 seconds |
| **Memory Usage** | ~500MB (Python process) |

---

## Reproduction Steps (Mac M4)

### 1. Pull Latest Code
```bash
cd ~/nfl-analytics
git pull origin main
git lfs pull  # Get the 84MB dump file
```

### 2. Start PostgreSQL 17
```bash
docker compose down pg-1
rm -rf pgdata  # Clean slate
docker compose up -d pg-1
sleep 20  # Wait for initialization
```

### 3. Restore Database
```bash
docker cp backups/nfl_analytics_20251207_185613.dump nfl-analytics-pg-1:/tmp/restore.dump
docker exec nfl-analytics-pg-1 pg_restore -U dro -d devdb01 --no-owner --no-privileges --jobs=4 /tmp/restore.dump
```

### 4. Run GNN Graph Builder
```bash
source .venv/bin/activate
python py/models/gnn_graph_builder_cuda.py
```

### Expected Output
```
Graph built: 434 nodes
  Players: 392
  Positions: 10
  Teams: 32
  Player→Position edges: 392
  Position→Team edges: 112
  Chemistry edges: 712

Games loaded: 1,349
  Seasons: 2020 - 2024
```

---

## Connection Details

```
Host: localhost
Port: 5544
Database: devdb01
User: dro
Password: sicillionbillions

Connection String:
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

---

## Files Modified

| File | Change |
|------|--------|
| `docker-compose.yml` | PG16 → PG17 |
| `py/models/gnn_graph_builder_cuda.py` | Multi-season support + games query fix (NVIDIA CUDA) |

---

## Platform-Specific Versions

| Platform | File | GPU Backend |
|----------|------|-------------|
| Windows (NVIDIA) | `gnn_graph_builder_cuda.py` | CUDA (RTX 4090) |
| Mac M4 | `gnn_graph_builder_mps.py` | Metal Performance Shaders |
| Linux (AMD) | `gnn_graph_builder_rocm.py` | ROCm |

## Next Steps

1. **Create M4 version:** Copy CUDA version, replace `torch.cuda` with `torch.mps`
2. **Create AMD/ROCm version:** Copy CUDA version, configure for ROCm
3. **Train Hierarchical GNN:** Use graph with `train_hierarchical_gnn.py`
4. **Add Matchup Edges:** Implement WR vs CB, OL vs DL matchups
5. **Temporal Graphs:** Build week-by-week graphs for temporal modeling

---

## Troubleshooting

### "relation mart.player_hierarchy does not exist"
- Run migration 023: `psql -f db/migrations/023_player_hierarchy_schema.sql`

### "unsupported version (1.16) in file header"
- Upgrade to PostgreSQL 17: Change `latest-pg16` → `latest-pg17` in docker-compose.yml

### 0 games loaded
- Check `game_type` column in games table
- Current fix: Use `week <= 18` instead of `game_type = 'REG'`

---

**Document Version:** 1.0
**Last Updated:** December 7, 2025
