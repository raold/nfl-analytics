# Project History

A short timeline of the project, and where removed material lives.

## Archived material

Two clean-ups removed most of the tree in October 2026. Nothing was lost; it is in local git tags:

| Tag | Points at | Holds |
|---|---|---|
| `docs-archive-2026-10` | `e707826` | the ~850 docs files (session logs, milestone summaries, audits and fix reports) removed by the docs prune of 2026-10-24 (`0c5cbc9`): 845 under `docs/`, 924 in all with the 79 notes it removed elsewhere (31 dissertation, 19 `archive/`, 15 `research/`, 6 `reports/`, 8 others) |
| `archive/pre-prune-2026-10` | `c01e10b` | the tree just before the salvage prune (committed 2026-10-25 00:05 EDT; the one commit in between, `6ffd370`, only added the errata audit): every file the prune deleted (2,793 of 3,562 tracked files) |
| `archive/dissertation-visual-redesign-2026-02` | `57113a1` | an orphaned dissertation styling branch from 2026-02-13, never merged (it restates retracted claims) |
| `archive/bnn-2025-11` | `61e5cd2` | where the BNN research stopped (ADR-004) |

The tags have not been pushed (PROJECT_STATUS, owner decisions), so they exist only in the
owner's clone. All but `archive/pre-prune-2026-10` contain a live API key in five old docs files;
rotate the key before pushing them.

```bash
git show archive/pre-prune-2026-10:nfl_analytics/rl/dataset.py | less     # read one removed file
git checkout archive/pre-prune-2026-10 -- R/ingestion                      # restore a directory
git ls-tree -r --name-only docs-archive-2026-10 docs/ | less               # list the archived docs
```

Paths cited by comments, ADRs and older write-ups that no longer exist in the tree refer to
files in these tags: `docs/milestones/…`, `docs/fixes/…`, `docs/audits/…` and similar are in
`docs-archive-2026-10`; code, scripts, notebooks, models and the other docs are in
`archive/pre-prune-2026-10`. A few cited files were never committed at all.

What the docs prune removed (file counts): loose `docs/*.md` 169, milestones 128, archive 92,
audits 73, fixes 50, deployment 46, guides 32, performance 28, api 23, agent_context 23, reports
21, architecture 15, research 14, knowledge 14, optimization 13, and ~30 smaller directories.

## Timeline

| Date | Event |
|------|-------|
| 2025-10 | Database schema audit, feature engineering, TikZ standardization for the dissertation; the 2004-2024 GLM/XGB walk-forward harness (its predictions are still in `analysis/results/`) |
| 2025-11-02 | Windows RTX 4090 GPU training; BNN calibration work |
| 2025-11-08 | Canonical training window (ADR-007); state-space model approved. The window was chosen on the test weeks with same-game inputs (retraction ledger C13) |
| 2025-11-09 | BNN research abandoned after 84 hours / 6 approaches (ADR-004); convergence threshold relaxed to R-hat < 1.02 (ADR-020) |
| 2025-11-18 | Ensemble v7 meta-learning rejected; static weights kept (ADR-009) |
| 2025-11-21 | Placeholder lines written into `games.spread_close` for 94 games (2025 weeks 12-18); the script was deleted and the lines verified gone on 2026-10-24 (POSTMORTEMS §4) |
| 2025-11-27 | Phase 2 advanced stats; Dirichlet weight optimization for props (tuned on its own test season, C11) |
| 2025-12-07/08 | One-off backfill of The Odds API historical snapshots for 2020-2024 into `odds_history`, with mistimed requests and two matcher bugs (POSTMORTEMS §7) |
| 2025-12-09 | Synthetic features (ADR-010). "Power ratings 74.9% ATS / +43% ROI" claimed: **refuted**, a spread sign bug (POSTMORTEMS §4, C01) |
| 2025-12-14 | XGBoost feature optimization; baseline retained (ADR-011) |
| 2025-12-21 | Semantic (LLM) feature stack, read as of extraction time (look-ahead, POSTMORTEMS §9); a simulated A/B result committed the same day (§10) |
| 2025-12-22 | Weekly feedback loop (ADR-012); none of its tables exists in devdb01 |
| 2025-12-23 | Game Outcome v2 trained (fit on 2022-2023, early-stopped on 2024) and promoted on a look-ahead semantic A/B |
| 2025-12-28 | Rest/Kalman/GP features (ADR-013; the rest result was the sign bug, C15) |
| 2025-12-31 | Kalman/GP features disabled (ADR-015) |
| 2026-01-02 | CQL target leakage found and fixed (ADR-016; the ADR was committed on 2026-01-09) |
| 2026-02-12 | Full 2025 season reviewed (ADR-017, `investigations/2026-02-season-review/`). v2's 285 predictions were made in one run that day, after the season. Graded correctly: spread MAE 11.05 vs 9.67 for the line, ATS 139-145-1; the "-3.2% on 208 bets" quoted then holds only at \|edge\| > 2 |
| 2026-02-12 | Last dissertation build (`main/main.pdf`, 722 pages) |
| 2026-02-13 | Kalshi trading system (ADR-018), later superseded by the separate Sibyl repo; a dissertation styling branch left unmerged (tag `archive/dissertation-visual-redesign-2026-02`) |
| 2026-10-24 | Repair: spread sign convention fixed project-wide, docs pruned (tag `docs-archive-2026-10`), `py/` renamed to `nfl_analytics/`, dependencies locked, CI repaired, market benchmark added |
| 2026-10-24 | Salvage, part 1: migration 104 applied; `games` repaired from nflverse (true UTC kickoffs for 2000-2025, playoff `game_type` for 1999-2020, missing metadata); state-space sign fixed; code moved to Eastern wall-clock time; Game Outcome v2 frozen; odds snapshots re-timed into a private panel; market facts; dissertation errata |
| 2026-10-24 | Salvage, part 2: final evaluation and retraction ledger; 2004-2024 walk-forward regraded; prospective Elo and power-rating predictions for 2026 weeks 7-8; full audit of the dissertation's tables and figures |
| 2026-10-25 | Salvage, part 3: prune from 3,562 to 771 tracked files (committed 00:05 EDT; tag `archive/pre-prune-2026-10`); tooling and CI cut to what works (sign guard in CI, secret scan, compose reduced to the database, schema snapshot); documentation rewritten to match the tree and the database; the 2025-12-08 dump holding the licensed odds rows untracked and moved to `backups/local/` (it stays in history) |
