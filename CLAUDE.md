# CLAUDE.md: nfl-analytics

**Updated:** 2026-10-25 · **State:** wound-down research archive with an honest negative result.
No model in this project beats the NFL closing line. Trading work continues in the separate Sibyl
repository; nothing here is meant to be deployed.

## What the project is now

A PostgreSQL warehouse of nflverse data (devdb01), the models built on it in 2025-2026, the tools
that graded them, and the LaTeX source of a ~720-page dissertation. The October 2026 salvage fixed
the spread sign convention and the database, graded every model against the closing line,
retracted the edge claims, and removed everything that was dead, misleading or hazardous.

What is kept and why:

- `nfl_analytics/evaluation/market_benchmark/` with `nfl_analytics/utils/spread_convention.py`:
  the grading tool (ATS at real prices, MAE and log-loss against the market, information beyond
  the line, closing-line value, guardrails, prospective predict mode).
- `models/production/game_outcome_v2/`: the one game model with a stored out-of-sample season
  (2025), frozen as a reference (see its README), plus the code that loads and regrades it.
- `docs/investigations/2026-10-*`: the final evaluation, the retraction ledger, the 2004-2024
  walk-forward regrade, line movement from the licensed odds snapshots, market facts.
- `analysis/dissertation/` (unchanged text) with `analysis/dissertation/ERRATA.md`.

Results: [README.md](README.md#results) has the canonical table;
[docs/investigations/2026-10-final-evaluation/](docs/investigations/2026-10-final-evaluation/README.md)
has the details and the scripts. Open items and owner decisions: [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Critical rules

1. **Spread sign.** `games.spread_close` is nflverse `spread_line`: the expected HOME margin,
   positive when the home team is favored (2018_01_BUF_BAL = 7.5; BAL won 47-3 and covered). Home
   covers iff `(home_score - away_score) - spread_close > 0`; pushes are excluded. Model edge =
   `pred_home_margin - spread_close`. Never write `margin + spread`: that sportsbook rule produced
   the retracted "74.9% ATS". Use `nfl_analytics/utils/spread_convention.py` (`home_cover`,
   `model_edge`, `book_home_line`, `assert_nflverse_convention`). The pre-commit hook
   `spread-sign-convention` (also run in CI) rejects the known wrong forms; mark a deliberate
   reproduction of the old rule with `# sign-ok: <reason>`. Sanity check: pooled home cover ≈ 49%.
   `odds_history.home_spread` is the other way round (sportsbook sign, negative = home favored).
2. **Kickoff is a true UTC instant** (repaired 2026-10-24 for 2000-2025; 1999 is NULL). Derive a
   local hour, weekday or date with `kickoff AT TIME ZONE 'America/New_York'` in SQL or
   `nfl_analytics/features/kickoff_time.py` (`kickoff_wall_clock`) in pandas. Comparing instants
   (`kickoff < now()`) needs no conversion. Before the repair, 2000-2024 held Eastern wall-clock
   time labelled UTC and 2025 held midnight ET.
3. **Never `DELETE FROM games`.** `plays` references `games` and `participation` references `plays`,
   both `ON DELETE CASCADE`. Change games with `UPDATE` or `INSERT ... ON CONFLICT (game_id) DO
   UPDATE`. The DB is a dev copy; treat it as read-only unless the task is a DB change, and take a
   backup first.
4. **The odds data is licensed.** `public.odds_history` (The Odds API snapshots, 2020-2024) must not
   be redistributed as a data product. No raw or row-level odds in the repo (it is public); raw
   exports go to `backups/local/` (gitignored). Aggregates are fine.
5. **`models/production/game_outcome_v2` is DO NOT DEPLOY.** `load_models()` refuses it unless
   called with `allow_frozen_reference=True`. Use it only for evaluation and postmortems.
6. **No made-up numbers.** Every number in a doc, table or figure comes from a committed script or
   a query you ran; simulations are labelled as such. Grade any claim with `market_benchmark`; it
   refuses win rates above 60% or ROI above 15% on 100+ bets and broken market frames.
7. **No secrets in the repo.** Database settings come from `DB_*` environment variables
   (`.env.example`; dev defaults in `nfl_analytics/utils/db_config.py`); never write the password
   into a doc. Keep the database on 127.0.0.1.

## Commands

Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/); CI uses 3.11. The dependencies are locked in
`uv.lock`.

```bash
uv sync --extra dev                          # core + test and lint tools
uv run pytest -m "not integration"           # offline suite, what CI runs (227 passed, 1 skipped on 2026-10-25)
uv run pytest -m integration                 # 133 tests; need devdb01 and DB_PASSWORD (tests/integration: 66 pass, 28 fail or error)
uv run ruff check --select E9,F63,F7,F82 nfl_analytics tests scripts
uv run black --check nfl_analytics tests     # black 26.10.1, line length 100
uvx pre-commit@4.7.0 run --all-files         # local hooks, including spread-sign-convention

# Grade baselines against the closing line, no database needed
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline power_ratings \
    --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024
# The current season from nflverse; predictions for a future week; grading them later
T=$(mktemp -d)                               # keep downloads and outputs out of the repo
curl -sSfL -o $T/games.csv https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
uv run python -m nfl_analytics.evaluation.market_benchmark --baseline elo --nflverse-csv $T/games.csv --seasons 2026
uv run python -m nfl_analytics.evaluation.market_benchmark predict --baseline elo \
    --nflverse-csv $T/games.csv --season 2026 --week 9 --out $T/preds_2026_w09_elo.csv
uv run python -m nfl_analytics.evaluation.market_benchmark --predictions-csv $T/preds_2026_w09_elo.csv \
    --nflverse-csv $T/games.csv --seasons 2026
# Any model: a CSV with game_id and the expected home margin, against devdb01
uv run python -m nfl_analytics.evaluation.market_benchmark --predictions-csv preds.csv \
    --pred-col pred_home_margin --db --seasons 2025

# Canonical results (see docs/investigations/2026-10-final-evaluation/README.md#reproduce)
uv run python scripts/final_evaluation/grade_game_outcome_v2_2025.py --out-dir /tmp/v2   # no database
uv run python -m nfl_analytics.utils.sql_validator "SELECT ..."   # catches the usual schema traps
```

The `final_evaluation` scripts write into `docs/investigations/2026-10-final-evaluation/results/`
by default (most take `--out-dir`). Do not rerun predict mode into `prospective/`. `make` lists
the Makefile shortcuts: test, lint (the CI lint steps plus the sign guard and its self-test),
benchmark, db, backup, restore, refresh.

## Database (devdb01)

PostgreSQL 17.7 + TimescaleDB 2.24.0 in container `nfl-analytics-pg-1` (`make db`),
127.0.0.1:5544, database `devdb01`, user `dro`; password from `DB_PASSWORD` (export it; only two
scripts read `.env`). An empty database gets the schema snapshot from `bash scripts/dev/init_dev.sh`;
restores and the migration history: [db/README.md](db/README.md). Counts below were taken
read-only on 2026-10-25.

| Table | Rows | Notes |
|---|---|---|
| `games` | 7,276 | 1999-2025, all completed; `game_type`, scores, lines and kickoffs equal nflverse for every game |
| `plays` | 1,279,628 | 1999-2025; `quarter` is NULL for every 1999-2024 play (filled for 2025) |
| `participation` | 428,409 | one row per play, 2016-2024; key `(nflverse_game_id, play_id)` |
| `odds_history` | 117,266 | 'opening'/'closing' snapshots of 1,061 games of 2020-2024 from 25 books, plus 32 'game_lines' rows; one backfill (2025-12-07/08); licensed |
| `injuries` / `weather` | 84,682 / 2,061 | 2009-2024 / 2018-2025 |
| `nextgen_passing` / `_rushing` / `_receiving` | 5,696 / 5,794 / 14,187 | 2016-2025, include week-0 season totals |
| `mart.game_summary` (matview) | 7,276 | pooled home cover 0.4905 over 7,081 decided games (migration 104) |
| `mart.player_game_stats` (matview) | 25,677 | no `game_id`: join `games` on `(season, week)` plus the player's team that week, taken from `plays`, never `current_team` |
| `ml.game_outcome_predictions` | 285 | Game Outcome v2's 2025 run (also `data/game_outcome_v2_predictions_2025.csv`) |

Schemas: `public` 46 tables (13 `mlb_*`/`nhl_*` belong to Sibyl) and 1 view, `mart` 6 tables and 3
matviews, `ml` 12 tables and 8 views, `metadata` 1, `salvage` 1 (`nflverse_games`, the 2026-10-24
nflverse download used for the repair). `public.mv_betting_features` is the other matview.
Conventions: use `kickoff`, `spread_close`, `total_close`, `quarter`, `posteam`/`defteam` (not
`game_date`, `spread_line`, `total_line`, `qtr`). Team codes: `games`, `plays` and the other
nflverse tables use nflverse codes, so the Rams are `LA` (`STL` before 2016) and never `LAR`;
`OAK`→`LV` in 2020, `SD`→`LAC` in 2017. Two tables use other code sets: `public.draft_picks` has
PFR codes (`LAR`, `LVR`, `GNB`, `KAN`, `NWE`, `SFO`, ...) and `public.espn_qbr.team_abb` ESPN codes
(`LAR`, `WSH`); one `public.international_games` row has `LAR`. Map codes to the nflverse form
before joining. `nfl_analytics.utils.team_mapping.normalize_team()` maps only `LA`↔`LAR`, `STL`,
`SD` and `OAK`, and passes `LVR`, the other PFR codes and `WSH` through unchanged.
`odds_history.market_type` 'opening'/'closing' are mislabelled request times, not the open and
close ([line-movement write-up](docs/investigations/2026-10-line-movement/README.md)).
`ml.model_registry` still lists v2 (model_id 17) as `production`; ignore that. More:
[DATABASE_QUICK_REFERENCE.md](docs/database/DATABASE_QUICK_REFERENCE.md),
[SQL_PATTERNS.md](docs/database/SQL_PATTERNS.md).

There is no data ingestion path any more (the R ingest scripts were removed). New seasons come
from the nflverse release files, read directly with `--nflverse-csv`.

## What was removed and where it is

The October 2026 prune (committed 2026-10-25 00:05 EDT) took the tree from 3,562 to 771 tracked
files: the Kalshi trading stack, the API, dashboards, MCP server, ETL and monitoring, the semantic
(LLM) stack, the feedback loop, the props ensembles, the R pipeline, the RL code, notebooks,
one-off scripts, model artifacts and most docs. Everything is recoverable from local tags:

| Tag | What it holds |
|---|---|
| `archive/pre-prune-2026-10` | `c01e10b`, the tree just before the prune: every deleted file (the one commit in between, `6ffd370`, only added the errata audit) |
| `docs-archive-2026-10` | the ~850 docs files removed by the docs prune of 2026-10-24 (924 files in all, counting the dissertation, `archive/`, `research/` and other notes outside `docs/`; [HISTORY](docs/HISTORY.md)) |
| `archive/dissertation-visual-redesign-2026-02` | an orphaned dissertation styling branch (pre-retraction claims; do not merge) |
| `archive/bnn-2025-11` | where the BNN research stopped (ADR-004) |

Only `archive/pre-prune-2026-10` is pushed. The other three contain an Odds API key in five old docs
files; the key is dead (tested 2026-10-25), and pushing them is an open owner item (PROJECT_STATUS).

Paths cited by older docs, ADRs and the 2026-10 write-ups that no longer exist are almost always
in `archive/pre-prune-2026-10` (`git show archive/pre-prune-2026-10:<path>`) or, for session
logs, `docs-archive-2026-10`. Module map:
[docs/architecture/ARCHITECTURE_REFERENCE.md](docs/architecture/ARCHITECTURE_REFERENCE.md).

## Dissertation

Source `analysis/dissertation/main/main.tex`; build with
`cd analysis/dissertation/main && latexmk -pdf -interaction=nonstopmode main.tex` (a clean copy
built on 2026-10-25: exit 0, 721 pages; `main.pdf`, `main.toc`, `main.lof` and `main.lot` are build
outputs, untracked since 2026-10-25; the last committed PDF, from 2026-02-12, is in history). The
text was not corrected: [ERRATA.md](analysis/dissertation/ERRATA.md) records what is wrong,
including the full audit of every table and figure (176 of 290 table units (170 printed tables)
and 30 of 51 figure entries (28 figures) made up, 63 table units computed from real data with a
bug). Check a number there before citing it.

Helpers outside the repo, in `~/.local/bin`: `diss-watch`, `diss-stats`, `diss-diff`,
`diss-compress`, `diss-clean`, `diss-lint` (they use `DISS_DIR=.../analysis/dissertation/main`,
as exported by `~/.zshrc`), and `diss-chapter`, `diss-figures` (they want the dissertation root:
`DISS_DIR=$PWD/analysis/dissertation diss-chapter list`). Shell aliases: `dw`, `ds`, `dl`,
`dclean`, `dchap`. Tips: `main/references.bib` is the active bibliography; `texlogsieve main.log`
finds overfull boxes; `pdftoppm -png -r 150 -f 1 -l 10 main.pdf /tmp/p` renders pages for review.
Standards: [DISSERTATION_STANDARDS.md](docs/guides/DISSERTATION_STANDARDS.md).

## Conventions

- Python: black and the CI ruff subset must pass; type hints; Google-style docstrings; 100 columns.
  SQL: uppercase keywords. R (three dissertation figure scripts left): tidyverse style.
- Tests: `tests/{integration,infrastructure,e2e}` are auto-marked `integration`; pin conventions
  with real games, not synthetic examples.
- Git: main branch `main`; plain commit messages, no co-author or AI attribution; never
  `--no-verify`. Commit and push only when asked.
- Docs: write findings under `docs/investigations/<yyyy-mm>-<topic>/` with the script and its
  output; keep README, PROJECT_STATUS and this file in step.

## Lessons learned

1. **Pin conventions with real games.** The spread sign bug recurred three times (Nov 2025, the
   CQL rewards, Dec 2025) because each script re-derived the cover rule. One helper, tests on
   BUF@BAL 2018, a hook in CI.
2. **Too good to be true is a bug report.** 74% ATS steady across six seasons is arithmetic, not
   skill; so is picking 81.5% of winners when the closing favorite picks 65.5% (same-season data,
   ledger C19). Check the pooled base rate first.
3. **Measure against the market, not a strawman.** MAE and log-loss against the de-vigged market
   settle in one season what ATS ROI cannot: at 208 bets the ROI interval is about ±13 points.
4. **A retroactive run is not a holdout.** v2's 2025 predictions were made after the season; only
   predictions published before kickoff are prospective evidence.
5. **Early-stopping scores are not validation scores.** v2's "10.45 validation MAE" is the
   early-stopping score on 2024, the season it tuned on; its out-of-sample 2025 MAE was 11.05
   against 9.67 for the line (ledger C17, C04).
6. **Compare on the same games.** "v2.1 10.70 vs v2 10.45" set a test-tuned 2025 number against
   v2's 2024 score; on the same 162 games of 2025, stored v2 scored 11.09 and the line 10.00 (C16).
   Disabling Kalman/GP stands on engineering grounds only (the GP never trained).
7. **A/B tests need a real control and point-in-time inputs.** The semantic "A/B" compared a
   prediction with itself, the quick control was a constant +2.5, one committed result was
   simulated from the outcomes, and the semantic inputs were read as of extraction time (C23-C25).
   There is no valid test of the semantic features.
8. **Check what a model uses before blaming a data source.** v2's 2025 predictions were compressed
   (SD 2.30 against 6.18 for the line), but none of its trees splits on its 46 NextGen or 20 QBR
   inputs, so missing QBR/NextGen data cannot explain it. The line-movement inputs it does split on
   sat at their defaults for every 2025 game (`odds_history` has no 2025 snapshots).
9. **Small subsets mislead.** v2's 8.43 playoff MAE came with the line at 8.33 on the same 12 games;
   12-13 games say nothing about playoff dynamics (C29).
10. **Never type numbers into a table generator.** 176 of 290 dissertation table units have no
    source; every table needs a committed script that reads committed data.
11. **Store true instants.** Eastern wall-clock times labelled UTC (2000-2024) made the odds
    backfill's "closing" request 4.5-5.5 hours before the first kickoff, and date-only 2025
    kickoffs zeroed every 2025 primetime flag.
12. **Match keys explicitly.** The odds backfill mapped the Rams to `LAR` (`games` has `LA`) and
    matched UTC dates to Eastern ones, so it dropped every Rams game and nearly every night game.
13. **Season-based splits only**; percentage splits leak across train and test.
14. **Same-game inputs are leakage.** The props models used the game's own pass attempts; a
    no-skill attempts × league-YPA formula scores 45.28 against their claimed 44.25 (C09, C13).
15. **Validate baselines independently.** The "+21%" incremental XGBoost compared against a mean
    predictor (POSTMORTEMS §2).
16. **Audit feature coverage before training** (XGBoost v3: 35% of rows median-imputed).
17. **Schema first.** `mart.player_game_stats` has no `game_id` (this error recurred dozens of
    times), and `current_team` is a player's latest team: joining games on it picks the wrong team
    in 53% of passing player-weeks (SQL_PATTERNS, mistake 4). Run `sql_validator`, then check a
    few rows by hand.
18. **Prefer explicit casts** over custom aggregates: `AVG(col::int)` rather than the custom
    `avg(boolean)` this database carries.
19. **Know when to stop** (BNN: 84 hours, 6 approaches, ADR-004).
20. **Context managers for DB connections**; delete corrupt model files before retraining; check R
    factor levels before training.

---

**Version:** 10.0 | **Maintainer:** Claude Code | **Repository:** github.com/raold/nfl-analytics

**Project Rules:**
- Never take shortcuts, never use synthetic data
- Survey, read, audit, synthesize before writing code
- When doing git commit NEVER add co-authorship
