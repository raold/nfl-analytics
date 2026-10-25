# SQL Patterns for devdb01

**Updated:** 2026-10-25. Every query on this page except the two marked WRONG was run (EXPLAIN or
executed) in a read-only session against devdb01 that day. Table facts and counts:
[DATABASE_QUICK_REFERENCE.md](DATABASE_QUICK_REFERENCE.md).

## Common mistakes

### 1. `game_id` in `mart.player_game_stats`

The view has no `game_id`, and this error recurred dozens of times.

```sql
-- WRONG: ERROR: column pgs.game_id does not exist
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id;
```

Join on `(season, week)` and the player's team that week (pattern 1).

### 2. Wrong column names

```sql
-- WRONG: none of these columns exists
SELECT g.game_date, g.spread_line, g.total_line FROM games g;
```

Use `g.kickoff` (a `timestamptz`), `g.spread_close` and `g.total_close`. `plays` uses `quarter`,
not `qtr`.

### 3. Team codes

`games`, `plays` and the other nflverse tables use nflverse codes. The Rams are `LA` (and `STL`
before 2016) and never `LAR` there, so `WHERE home_team = 'LAR'` silently returns nothing. The
Chargers are `LAC` (`SD` before 2017), the Raiders `LV` (`OAK` before 2020).

Two tables use other code sets (checked 2026-10-25): `public.draft_picks.team` has
Pro-Football-Reference codes (`LAR` 89 rows, `LVR` 48, and `GNB`, `KAN`, `NOR`, `NWE`, `SFO`,
`TAM`, `SDG`, `RAI`, `RAM`, `PHO` for other franchises and eras), and `public.espn_qbr.team_abb`
has ESPN codes (`LAR` 186 rows, `WSH` 324). One `public.international_games` row
(`2019_08_CIN_LA`) has `home_team = 'LAR'`. Map these to nflverse codes before joining.
`nfl_analytics.utils.team_mapping.normalize_team(code, to_canonical=False)` turns `LAR` into `LA`,
but it knows only the Rams, Chargers and Raiders codes (`LA`, `LAR`, `STL`, `SD`, `OAK`) and
returns `LVR`, the other PFR codes and `WSH` unchanged.

### 4. Joining a player to games through `current_team`

`mart.player_hierarchy.current_team` (also copied into `mart.player_game_stats`) is the player's
latest team. Joining games on it attaches the wrong game, or none, to every week the player spent
elsewhere. For the 5,275 passing player-weeks of 2016-2025 that appear in `plays`, `current_team`
differs from the team the player actually passed for in 2,803 (53%). The older version of this
page, and the props models built from it, used that join. Take the team from `plays` instead.

### 5. Week 0

`mart.player_game_stats` (2,130 rows) and the `nextgen_*` tables hold season totals as week 0.
A filter such as `week < W` without `week >= 1` pulls in the whole season.

## Patterns

### Pattern 1: player stats with that week's game and line

```sql
WITH passer_games AS (
    SELECT DISTINCT p.game_id, p.passer_player_id AS player_id, p.posteam AS team
    FROM plays p
    WHERE p.passer_player_id IS NOT NULL
)
SELECT
    pgs.player_id, pgs.season, pgs.week,
    pgs.stat_yards, pgs.stat_attempts,
    pg.team,
    g.game_id,
    (pg.team = g.home_team)::int AS is_home,
    -- favored: home iff spread_close > 0, away iff spread_close < 0 (pick'em: neither)
    CASE WHEN pg.team = g.home_team THEN (g.spread_close > 0)::int
         ELSE (g.spread_close < 0)::int END AS is_favored,
    abs(g.spread_close) AS spread_abs,
    g.total_close,
    w.temp, w.wind,
    g.roof
FROM mart.player_game_stats pgs
JOIN games g ON g.season = pgs.season AND g.week = pgs.week
JOIN passer_games pg ON pg.game_id = g.game_id AND pg.player_id = pgs.player_id
LEFT JOIN weather w ON w.game_id = g.game_id
WHERE pgs.stat_category = 'passing'
  AND pgs.week >= 1
  AND pgs.season BETWEEN 2016 AND 2025;
```

For rushing or receiving stats use `rusher_player_id` or `receiver_player_id`. The older version of
this pattern computed `is_favored` with the signs reversed (it marked home underdogs as favored)
and took `temp` from `games`, which has no such column. `weather.wind` has a unit problem (ERRATA
E5); nflverse `wind` is the safer source.

### Pattern 2: one player-game, parameterized

```python
import pandas as pd
import psycopg2

from nfl_analytics.utils.db_config import get_db_config   # DB_* environment variables

QUERY = """
SELECT pgs.stat_yards, pgs.stat_attempts, g.game_id, g.spread_close, g.total_close
FROM mart.player_game_stats pgs
JOIN games g ON g.season = pgs.season AND g.week = pgs.week
JOIN (SELECT DISTINCT game_id, passer_player_id AS player_id FROM plays
      WHERE passer_player_id = %(player_id)s) pg
  ON pg.game_id = g.game_id AND pg.player_id = pgs.player_id
WHERE pgs.player_id = %(player_id)s AND pgs.season = %(season)s AND pgs.week = %(week)s
  AND pgs.stat_category = 'passing'
"""

with psycopg2.connect(**get_db_config()) as conn:
    df = pd.read_sql_query(QUERY, conn, params={"player_id": "00-0019596", "season": 2019, "week": 1})
```

(`00-0019596` is Tom Brady: `current_team` says TB, but in 2019 week 1 he played for NE.)

### Pattern 3: a player's last N games before a date

```sql
SELECT pgs.season, pgs.week, pgs.stat_yards, pgs.stat_attempts, g.game_id, g.kickoff
FROM mart.player_game_stats pgs
JOIN games g ON g.season = pgs.season AND g.week = pgs.week
JOIN (SELECT DISTINCT game_id, passer_player_id AS player_id FROM plays
      WHERE passer_player_id = '00-0019596') pg
  ON pg.game_id = g.game_id AND pg.player_id = pgs.player_id
WHERE pgs.player_id = '00-0019596'
  AND pgs.stat_category = 'passing'
  AND pgs.week >= 1
  AND g.kickoff < TIMESTAMPTZ '2020-10-01 00:00 America/New_York'   -- the prediction time
ORDER BY g.kickoff DESC
LIMIT 8;
```

### Pattern 4: cover, edge and line in the correct sign

```sql
SELECT
    g.game_id,
    g.home_score - g.away_score                             AS home_margin,
    g.spread_close,                                          -- expected home margin
    (g.home_score - g.away_score) - g.spread_close           AS home_cover_margin,
    CASE WHEN (g.home_score - g.away_score) > g.spread_close THEN 1
         WHEN (g.home_score - g.away_score) < g.spread_close THEN 0 END AS home_covered,  -- NULL = push
    -g.spread_close                                          AS book_home_line  -- sportsbook sign
FROM games g
WHERE g.season = 2018 AND g.game_id = '2018_01_BUF_BAL';
-- 47-3, spread_close 7.5: home_cover_margin 36.5, home_covered 1, book_home_line -7.5
```

A model's edge is `pred_home_margin - g.spread_close`. Python code uses
`nfl_analytics/utils/spread_convention.py` for the same rules.

### Pattern 5: local kickoff hour, weekday and date

```sql
SELECT g.game_id,
       g.kickoff                                                      AS kickoff_utc,
       (g.kickoff AT TIME ZONE 'America/New_York')::date              AS game_date_et,
       extract(hour FROM g.kickoff AT TIME ZONE 'America/New_York')   AS hour_et,
       extract(isodow FROM g.kickoff AT TIME ZONE 'America/New_York') AS isodow_et   -- 7 = Sunday
FROM games g
WHERE g.game_id = '2025_22_SEA_NE';
-- 2026-02-08 23:30 UTC; 2026-02-08, 18, 7
```

### Pattern 6: odds snapshots in nflverse sign

```sql
SELECT o.game_id, o.bookmaker, o.market_type, o.last_update,
       -o.home_spread AS home_line_nflverse_sign,
       g.spread_close
FROM odds_history o
JOIN games g ON g.game_id = o.game_id
WHERE o.game_id = '2023_01_CIN_CLE'
ORDER BY o.last_update;
```

`market_type` names the backfill request, not the opening or closing line; the close is
`games.spread_close`. The table has no Rams games and almost no night games (a Thursday opener
such as `2023_01_DET_KC` returns nothing). Keep row-level results private: the data is licensed.

### Pattern 7: a team's games in a season

```sql
SELECT g.game_id, g.week, g.kickoff, g.home_team, g.away_team, g.spread_close, g.total_close,
       g.home_score, g.away_score
FROM games g
WHERE 'KC' IN (g.home_team, g.away_team)
  AND g.season = 2024
ORDER BY g.kickoff;
```

`games` has no status column; an unplayed game has NULL scores (devdb01 currently has none).

## Indexes that exist

| Relation | Indexes |
|---|---|
| `games` | `games_pkey (game_id)`, `games_season_week_idx (season, week)`, `games_kickoff_idx`, `games_home_team_idx`, `games_away_team_idx` |
| `plays` | `plays_pkey (game_id, play_id)`, `plays_game_id_idx`, `plays_posteam_idx`, `plays_quarter_idx` |
| `mart.player_game_stats` | `idx_player_game_stats_unique (player_id, season, week, stat_category)`, `_player_id`, `_season_week`, `_category`, `_position_group` |
| `mart.player_hierarchy` | `idx_player_hierarchy_player_id` (unique), `_position_group`, `_team` |

The November 2025 optimization reports in `db/` describe further indexes
(`idx_player_game_stats_lookup`, `idx_games_season_week_kickoff`, `idx_games_betting_context`, ...)
that do not exist in devdb01. Pattern 1 over all passing rows takes well under a second.

## Checks

```sql
-- player_game_stats has no game_id (expect 0 rows)
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'mart' AND table_name = 'player_game_stats' AND column_name = 'game_id';

-- pooled home cover rate with the correct rule (expect about 0.49)
SELECT avg(((home_score - away_score) > spread_close)::int)
FROM games
WHERE spread_close IS NOT NULL AND (home_score - away_score) <> spread_close;
```

`uv run python -m nfl_analytics.utils.sql_validator "<query>"` catches mistakes 1 and 2 before a
query runs.
