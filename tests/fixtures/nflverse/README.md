# nflverse fixtures for the market benchmark tests

Real rows, unchanged values; only rows and columns are selected. Used by
`tests/unit/evaluation/test_market_benchmark_nflverse.py`.

## games_2025_2026w8.csv

- Source: <https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv>,
  downloaded 2026-10-24 16:54:34 EDT (20:54:34 UTC). The full file had 7,548 rows (1999-2026),
  sha256 `5d882d05c73ab11758cfe70e71e994e50b53e39b974943368dfee500f9a50693`.
- Rows (407): the first 1999 game (`1999_01_MIN_ATL`; nflverse has no `gametime` for 1999),
  every 2025 game (272 regular season + 13 playoff), and every 2026 game of weeks 1-8 (121).
  At download 94 of the 2026 games were final: weeks 1-6 and the week 7 Thursday game
  `2026_07_NE_CHI`. The other 27 have no score yet, which is what predict mode needs.
- Columns (18, nflverse order): the ones `market.read_nflverse_games` reads.
- sha256 `aadf74b3faf2a260f759a2016fa71d04c185cec10ff67ab76985c2517c7af97b`. Reproduce it
  byte for byte from that download with:

```python
import pandas as pd

g = pd.read_csv("games.csv", dtype=str, keep_default_na=False)
cols = ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team",
        "away_score", "home_team", "home_score", "away_moneyline", "home_moneyline",
        "spread_line", "away_spread_odds", "home_spread_odds", "total_line", "under_odds",
        "over_odds"]
rows = pd.concat([g[g.season == "1999"].head(1), g[g.season == "2025"],
                  g[(g.season == "2026") & (g.week.astype(int) <= 8)]])
rows[cols].to_csv("games_2025_2026w8.csv", index=False)
```

## lines_2025w15_20251207.csv

The nflverse `spread_line` of the 16 games of 2025 week 15 as it stood on 2025-12-07, before
any of them was played (they kicked off 2025-12-11 to 2025-12-15). Their closing lines are in
`games_2025_2026w8.csv`, so the two files together give real line moves for closing-line value.

- That day `nflreadr::load_schedules(1999:2025)` loaded nflverse schedules into devdb01
  `games`, copying `spread_line` into `spread_close` (`tmp_rovodev_ingest_schedules_direct.R`,
  commit 3c29d9c, 2025-12-07 16:56 UTC). The values here come from the backup taken
  afterwards, `backups/nfl_analytics_20251207_154232_pg17.dump` (local, not in git; file name
  time 2025-12-07 15:42:32 local), table `public.games`, rows with season 2025 and week 15.
  The 2025-12-07 20:19 and 2025-12-08 backups hold the same values.
- sha256 `21e6eaa8eee67240773e4eceeadef8ec0cd8812983d1ab98257e09577fb8521b`.
- Example: `2025_15_IND_SEA` was 3.5 (SEA favored by 3.5) on 2025-12-07 and closed 13.5;
  `2025_15_BUF_NE` moved from 1.5 (NE favored) to -2.5 (BUF favored).
