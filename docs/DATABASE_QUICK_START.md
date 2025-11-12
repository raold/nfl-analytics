# Database Quick Start Guide
**Last Updated:** November 10, 2025

## Connection String

```bash
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

## Quick Access

```bash
# psql command line
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01

# R connection
library(DBI)
library(RPostgres)
con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)
```

## Database Statistics

| Metric | Value |
|--------|-------|
| Total Games | 7,263 (1999-2025) |
| Total Plays | 1,254,173 |
| Passing Plays | 515,996 |
| Rushing Plays | 387,402 |
| 2024 Games | 285 (including Super Bowl) |
| Unique Passers | 634 |

## Essential Tables

### games (7,263 rows)
```sql
SELECT * FROM games WHERE season = 2024 LIMIT 5;
```
Columns: `game_id`, `season`, `week`, `home_team`, `away_team`, `kickoff`, `spread_close`, `total_close`, `home_score`, `away_score`

### plays (1,254,173 rows)
```sql
SELECT * FROM plays 
WHERE passer_player_id IS NOT NULL 
LIMIT 5;
```
Key columns: `game_id`, `play_id`, `passer_player_id`, `passing_yards`, `yards_gained`, `epa`, `wp`, `down`, `ydstogo`

## Common Queries

### Training Data (2018-2024)
```sql
SELECT 
  p.passer_player_id,
  p.passer_player_name,
  p.passing_yards,
  p.yards_gained,
  p.epa,
  g.season,
  g.week,
  g.kickoff
FROM plays p
JOIN games g ON p.game_id = g.game_id
WHERE g.season BETWEEN 2018 AND 2024
  AND p.passer_player_id IS NOT NULL
  AND p.passing_yards IS NOT NULL;
```

### Player Season Stats
```sql
SELECT 
  passer_player_name,
  COUNT(*) as attempts,
  AVG(passing_yards) as avg_yards,
  MAX(passing_yards) as longest,
  SUM(CASE WHEN complete_pass = 1 THEN 1 ELSE 0 END) as completions
FROM plays
WHERE passer_player_id IS NOT NULL
  AND game_id IN (SELECT game_id FROM games WHERE season = 2024)
GROUP BY passer_player_name
ORDER BY attempts DESC;
```

### Recent Games
```sql
SELECT 
  season,
  week,
  kickoff::date,
  home_team,
  away_team,
  home_score,
  away_score
FROM games
WHERE season = 2024
ORDER BY kickoff DESC
LIMIT 10;
```

## Materialized Views

Refresh when needed:
```sql
REFRESH MATERIALIZED VIEW mart.game_summary;
REFRESH MATERIALIZED VIEW mart.player_game_stats;
REFRESH MATERIALIZED VIEW mart.game_features_enhanced;
```

List all:
```sql
SELECT schemaname, matviewname 
FROM pg_matviews 
ORDER BY schemaname, matviewname;
```

## Data Quality

- ✅ All 2024 games populated (including Super Bowl)
- ✅ Play-by-play complete through 2024 season
- ✅ All required columns present (`passer_player_id`, `passing_yards`)
- ✅ Materialized views refreshed (November 10, 2025)

## Next Steps

Ready for:
- Model training (2018-2024 canonical window)
- Feature engineering
- Backtesting validation
- Production predictions

See [DATABASE_INGESTION_VALIDATION.md](milestones/DATABASE_INGESTION_VALIDATION.md) for complete validation report.
