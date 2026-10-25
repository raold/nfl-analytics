# Project Status

**Updated:** 2026-10-25 · **State:** wound down. A research archive with an honest negative result:
no model in this project beats the NFL closing line ([README](README.md#results),
[final evaluation](docs/investigations/2026-10-final-evaluation/README.md)). The salvage work is
on branch `repair/2026-10`, which is not pushed; it fast-forwards `origin/main` (0 commits behind).

## Done in the October 2026 salvage

| Item | What was done | Evidence |
|---|---|---|
| CI | Offline suite green and pinned to the lock; lint subset and black; the temporal-safety job runs the leakage and convention tests plus the spread-sign hook; a gitleaks job; link checks | `uv run pytest -m "not integration"`: 218 passed, 1 skipped (torch not installed) on 2026-10-25; 1,511 passed before the prune removed the tests of deleted code. Temporal-safety tests: 72 passed. Workflows in `.github/workflows/` |
| Spread convention | One helper, `nfl_analytics/utils/spread_convention.py`, used by every label and grading site; tests pinned to real games; widened pre-commit guard with a `# sign-ok` escape; newly found sites fixed (state-space ATS probability, as-of features) | [POSTMORTEMS §4](docs/POSTMORTEMS.md#4-spread-sign-convention-found-october-24-2026) |
| Convention verified on the database | 97.2% of home moneyline favorites since 2015 have `spread_close` > 0 (the 54 exceptions are within 1.5 points); corr(spread_close, home moneyline) = -0.953; pooled 2020-2025 home cover 49.9%, against 55.7% under the old rule (n = 1,660) | POSTMORTEMS §4 queries, run read-only 2026-10-25 |
| Migration 104 applied | `mart.game_summary` and `mv_betting_features` grade covers correctly; recorded in `schema_migrations` (2026-10-24 22:41 UTC) | pooled home cover in `mart.game_summary`: 0.4905 over 7,081 decided games |
| 2025 placeholder lines | The EPA-derived placeholders written into 2025 weeks 12-18 on 2025-11-21 are gone | `spread_close` equals nflverse `spread_line` for all 285 games of 2025 and all 7,276 games of 1999-2025 |
| Database repair | `kickoff` is now the true instant for 2000-2025 (7,017 games; 1999 has no time upstream); `game_type` relabelled for 244 playoff games of 1999-2020; NULL metadata filled (roof, surface, rest, QBs, coaches, referee); scores and lines untouched | 0 mismatches with nflverse on `game_type`, scores, lines and kickoffs (read-only comparison with `salvage.nflverse_games`). Before-image and dumps in `backups/local/` |
| Game Outcome v2 graded and frozen | 285 predictions of 2025 exported to `data/game_outcome_v2_predictions_2025.csv` and graded without the database; the model is DO NOT DEPLOY and `load_models()` refuses it by default | [results/game_outcome_v2_2025.md](docs/investigations/2026-10-final-evaluation/results/game_outcome_v2_2025.md) (rerun byte-identical 2026-10-25); [model README](models/production/game_outcome_v2/README.md) |
| 2004-2024 walk-forward regraded | Labels verified; state-space sign fixed; the vig was charged twice. GLM: 51.0% of 3,826 bets, ROI -2.6% | [walkforward_2004_2024.md](docs/investigations/2026-10-final-evaluation/walkforward_2004_2024.md) |
| Final evaluation | v2 2025, Elo and power ratings (2019-2024, 2025, 2019-2025, 2026 holdout), the Kalman state-space walk-forward, prospective 2026 week 7 and 8 predictions | [final evaluation](docs/investigations/2026-10-final-evaluation/README.md); baselines rerun byte-identical 2026-10-25 |
| Retraction ledger | 44 headline numbers, where each still appears, what is wrong, corrected values | [retraction_ledger.md](docs/investigations/2026-10-final-evaluation/retraction_ledger.md) |
| Odds panel | `odds_history` snapshot times reconstructed and the rows cleaned into a private panel; aggregate line-movement and CLV results published | [line movement](docs/investigations/2026-10-line-movement/README.md); panel in `backups/local/odds_history_panel/` (licensed, not in git) |
| Market facts | How closing lines behaved in 1999-2025 from nflverse schedules, labelled as historical associations | [market facts](docs/investigations/2026-10-market-facts/README.md) |
| Dissertation errata | Every error found, with corrected values; a full audit of all 290 table units and 51 figures (176 tables and 30 figures made up, 11 more tables with made-up cells, 63 tables computed with a bug); chapters unchanged | [ERRATA.md](analysis/dissertation/ERRATA.md), `ERRATA_audit.csv` |
| Prune | 3,562 tracked files down to 771: trading stack, API, dashboards, ETL, semantic stack, feedback loop, R pipeline, notebooks, artifacts, most docs; hazards (simulated results, a DB-dump-to-public script, a Sci-Hub fetcher) removed | tag `archive/pre-prune-2026-10` holds everything removed ([HISTORY](docs/HISTORY.md)) |
| Tooling | Local pre-commit cut to hooks that pass; compose reduced to the database; Makefile reduced to working targets; schema snapshot as the DB bootstrap (`scripts/dev/init_dev.sh`); the infrastructure test that could stop the database removed, and the restore test limited to a test database | `.pre-commit-config.yaml`, `docker-compose.yml`, `Makefile`, [db/README.md](db/README.md), [tests/README.md](tests/README.md) |
| Docs | README, CLAUDE.md, this file, SETUP, the docs index, ADR statuses and banners, postmortems and database references rewritten to match the tree and the database | this commit |

## Open: decisions for the owner

1. **Rotate The Odds API key.** The live key is in 5 docs files at the tip of public `main` (removed
   on this branch) and in Sibyl's `.env`. Also confirm that the two keys committed in `.env` in
   October 2025 are revoked, and turn on GitHub secret scanning and push protection.
2. **Push `repair/2026-10` and open the PR.** It fast-forwards `main` and removes the live key from
   the public tip. Pushing also timestamps the prospective predictions in
   `docs/investigations/2026-10-final-evaluation/prospective/`: a game counts as prospective
   evidence only if the files are public before it kicks off (week 7 starts 2026-10-25 09:30 ET,
   week 8 on 2026-10-29 20:15 ET).
3. **Push the tags.** All four are local only. `archive/pre-prune-2026-10` is the only easy way
   back to the deleted files and holds no copy of the key. `docs-archive-2026-10`,
   `archive/dissertation-visual-redesign-2026-02` and `archive/bnn-2025-11` each contain the live
   key in 5 files (they predate its removal): rotate it before pushing them, or re-tag with those
   files redacted.
4. **Public LFS content.** The three dumps in `backups/` contain `odds_history` (The Odds API terms
   forbid redistributing the data as a standalone product) and predate the 2026-10-24 repair.
   `analysis/dissertation/main/main.pdf` carries the retracted and made-up claims. Remove them from
   the tip, replace the PDF with a corrected build, or keep them next to the errata. Removed files
   stay downloadable from LFS history unless purged.
5. **Branch cleanup.** Local: `backtest`, `experiments`, `hotfix`, `rovodev_arch`, the four
   `feature/*` branches, nine `tmp*` branches and the `refs/ai*` refs. Remote: `bnn-archive`,
   `rovodev_arch`, `feature/code-quality-remediation`. The salvage survey found all of them
   contained in `main` history or worthless.
6. **Disk.** The asd leftovers in `~/rice` (15 `.nfl-*-backup_asd-*` directories and tarballs,
   about 25 GB; the tarballs are owned by root). The `_asd-old` directories share files with the
   live repo through hardlinks, so they are not real backups; what only they held is the orphaned
   dissertation branch, now the tag `archive/dissertation-visual-redesign-2026-02`, and the old
   reflogs. The dangling Docker volume
   `3c6f2d424a18...` (created 2026-10-24 17:18 EDT) is a restored scratch copy of the database;
   `nfl-analytics_grafana_data`, `_prometheus_data` and `_redis_data` belong to the removed services.
7. **Sibyl handoff.** `backups/local/SIBYL_HANDOFF_2026-10-24.md`: Sibyl's NFL backtest still
   grades covers with the old rule, its news ingest stamps current articles onto past seasons, and
   it carries a byte-identical copy of Game Outcome v2.
8. **Backups.** `backups/local/` holds the only dumps of the repaired database and the before-image
   of the repair; check that the next borg home run picks it up.
9. **Dev database password.** It is public, and the role is a superuser. Keep the database on
   127.0.0.1 and never reuse the password; rotate it if the database is ever reachable off-host.

## Open: known technical issues (not planned)

- The dissertation is uncorrected; [ERRATA §6](analysis/dissertation/ERRATA.md#6-what-a-corrected-build-requires)
  lists what a corrected build needs.
- No data ingestion path remains (the R ingest scripts were removed); new seasons come from the
  nflverse release files. `scripts/pipeline_health_check.py` still prints the removed R commands as
  remediation, and it checks QBR and NextGen inputs that v2 never uses.
- `public.weather` wind speeds from the Meteostat ingest are in the wrong unit (ERRATA E5); use
  nflverse game-day wind. `plays.quarter` and the clock columns are NULL for every 1999-2024 play.
- `ml.model_registry` still lists v2 (model_id 17) as `production`.
- `nfl_analytics/backtest/harness_multimodel.py` fails at import (`DEFAULT_FEATURE_COLUMNS` no longer
  exists in `baseline_glm.py`), and its input feature file is not in git, so the 2004-2024 harness
  cannot be rerun; its published predictions are graded as stored.
- `scripts/final_evaluation/check_claims.py` reproduces only from a worktree of
  `archive/pre-prune-2026-10` (it scans files the prune removed).
- `scripts/generate_gnn_calibration_plot.py` draws its figure from random noise (ERRATA P46).
- Earlier open items closed without action: ADR re-tests and retrains (the project is wound down),
  the inverted feature encodings (the models that use them are frozen or removed), the
  meta-labeling retrain and the props Dirichlet weights (code removed in the prune).
