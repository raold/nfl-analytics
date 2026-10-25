# Project Status

**Updated:** 2026-10-25 · **State:** wound down. A research archive with an honest negative result:
no model in this project beats the NFL closing line ([README](README.md#results),
[final evaluation](docs/investigations/2026-10-final-evaluation/README.md)). The salvage was
merged into `main` on 2026-10-25 (pull request #24, a fast-forward) and CI is green there. What
is left for the owner is [at the end](#open-for-the-owner).

## Done in the October 2026 salvage

| Item | What was done | Evidence |
|---|---|---|
| CI | Offline suite green and pinned to the lock; lint subset and black; the temporal-safety job runs the leakage and convention tests plus the spread-sign hook; a gitleaks job; link checks | `uv run pytest -m "not integration"`: 227 passed, 1 skipped (torch not installed) at the branch tip on 2026-10-25; 1,571 passed, 11 skipped at `c01e10b`, just before the prune removed the tests of deleted code (1,511 at the start of the salvage). Temporal-safety tests: 72 passed. Workflows in `.github/workflows/` |
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
| Dissertation errata | Every error found, with corrected values; a full audit of all 290 table units and 51 figure entries (176 table units, 170 printed tables, and 30 figure entries, 28 figures, made up; 11 more tables with made-up cells; 63 table units computed with a bug); chapters unchanged | [ERRATA.md](analysis/dissertation/ERRATA.md), `ERRATA_audit.csv` |
| Prune | 3,562 tracked files down to 771 (committed 2026-10-25 00:05 EDT): trading stack, API, dashboards, ETL, semantic stack, feedback loop, R pipeline, notebooks, artifacts, most docs. Removed as hazards: fabricated and simulated result files and most of their generators, two scripts that wrote DB dumps into the repo, a Sci-Hub fetcher. Five figure generators that make their numbers up were kept as dissertation tooling, marked with a warning header (known issues below) | tag `archive/pre-prune-2026-10` holds everything removed ([HISTORY](docs/HISTORY.md)) |
| Tooling | Local pre-commit cut to hooks that pass; compose reduced to the database; Makefile reduced to working targets; schema snapshot as the DB bootstrap (`scripts/dev/init_dev.sh`); the infrastructure test that could stop the database removed, and the restore test limited to a test database | `.pre-commit-config.yaml`, `docker-compose.yml`, `Makefile`, [db/README.md](db/README.md), [tests/README.md](tests/README.md) |
| Docs | README, CLAUDE.md, this file, SETUP, the docs index, ADR statuses and banners, postmortems and database references rewritten to match the tree and the database | `289ea5d` and the docs commits after it |
| Publication | `repair/2026-10` pushed at 2026-10-25 07:37:53 UTC and fast-forwarded into `main` (PR #24, merged 07:47:16 UTC). The week 7-8 prospective files were public 5 h 52 min before the first predicted kickoff, so all 27 games count as prospective | [prospective README](docs/investigations/2026-10-final-evaluation/prospective/README.md#when-the-timestamp-becomes-verifiable); on `main`, lint-and-test, temporal safety, secret scan and local links pass |
| Credentials | All three keys for The Odds API found in history (the one in five old docs files, which Sibyl's `.env` also holds, and the two committed in `.env` in `f2b9c7b` and `b90e9d6`) were tested on 2026-10-25 against the API's quota-free `/v4/sports` endpoint. The docs-file key, the one in Sibyl's `.env`, answered "API key is not valid"; of the two `.env` keys, the one from `f2b9c7b` answered "not valid" and the one from `b90e9d6` "deactivated". GitHub secret scanning and push protection are on | tested without printing the keys; repository security settings |

## Decided on 2026-10-25

The ten owner decisions listed here before the push, and what came of them:

1. **The Odds API keys.** None of the keys in history works (above), so there is nothing left to
   rotate for this repository; the copy in Sibyl's `.env` is dead too. A new key, if Sibyl needs
   one, belongs only in Sibyl's environment.
2. **Push and pull request.** Done: PR #24, merged as a fast-forward, so every commit keeps its
   hash.
3. **Tags.** `archive/pre-prune-2026-10` is pushed. `docs-archive-2026-10`,
   `archive/dissertation-visual-redesign-2026-02` and `archive/bnn-2025-11` are not pushed yet;
   the key in their five old docs files is dead (open item 1).
4. **Public dumps and the PDF.** No history rewrite. On its own it would take nothing out of
   public reach: GitHub keeps every pull request's head ref (`refs/pull/1/head` to
   `refs/pull/24/head`), `refs/pull/24/head` alone reaches the whole history, and only GitHub
   Support can remove those refs and stored LFS objects. A rewrite would also change the hash of
   every commit after `46a9645` (2025-12-08), among them `c01e10b`, which the published
   prospective record cites. So the 2025-12-08 dump with the licensed `odds_history` rows, added
   in `46a9645`, stays reachable through history and GitHub's LFS storage (open item 3). The two
   2025-12-07 dumps stay tracked; their `odds_history` table is empty.
   `analysis/dissertation/main/main.pdf`, which carried the retracted and made-up claims, was
   untracked (`0bccf47`); `latexmk` rebuilds it, ERRATA.md records the errors, and its old
   versions stay in history.
5. **Owner housekeeping.** The off-machine backup of `backups/local/` was taken and verified, and
   the container volumes of the removed services were deleted. The old sync-daemon copies of the
   repository and its database (25 GiB) were audited: apart from 76 KB of unique files, now copied
   into `backups/local/`, everything in them is in the repository, its git objects or the live
   database. Deleting them is left to the owner (open item 7). Details are in the private note in
   `backups/local/`.
6. **Sibyl handoff.** Implemented in the private Sibyl repository as draft pull request #2718
   (branch `claude/nfl-sign-convention-2026-10-25`).
   - It grades spreads through a ported convention helper, guarded by a lint and by a runtime
     check.
   - It stops filing news under weeks it postdates.
   - It marks Sibyl's bundled v2 do-not-deploy.

   Each head was panel-reviewed before publication. Merging and deploying are the Sibyl
   operator's decisions. Sibyl's `[domains.nfl]` stays disabled and its copy of Game Outcome v2
   is not deployed.
7. **The database v2 was trained on.** Not checked: the machine it was most likely on is not
   reachable from this one (open item 5). devdb01 does not reproduce it: no `semantic` schema, no
   feedback-loop tables, no 2025 odds snapshots (one `game_lines` row) and no 2025 injuries.
8. **Missing odds: not re-fetched.** The 357 games of 2020-2024 without a game-day snapshot (93
   Rams games and 264 that start on the next UTC date, which the backfill's matcher dropped; 10 of
   the 264 kept early snapshots only) stay missing: there is no working key, the quota is paid,
   and nothing left in this project would use them (POSTMORTEMS §7,
   [line-movement coverage](docs/investigations/2026-10-line-movement/results.md)).
9. **Branch cleanup.** The local branches (`backtest`, `experiments`, `hotfix`, `rovodev_arch`,
   four `feature/*`, nine `tmp*`), the `refs/ai*` refs and two stashes are deleted; the salvage
   survey found all of them contained in `main` history or worthless. The remote branches
   `bnn-archive`, `rovodev_arch` and `feature/code-quality-remediation` remain (open item 2).
10. **Dev database password: unchanged.** It is public (`docker-compose.yml`), the database
    listens on 127.0.0.1 only, and it is never reused. Rotate it if the database is ever
    reachable off-host.

## Open: for the owner

1. Push the remaining tags:
   `git push origin refs/tags/docs-archive-2026-10 refs/tags/archive/dissertation-visual-redesign-2026-02 refs/tags/archive/bnn-2025-11`
2. Delete the old remote branches:
   `git push origin --delete bnn-archive rovodev_arch feature/code-quality-remediation`
3. Decide whether the licensed odds rows reachable through history matter enough to act: keep the
   repository public as it is, make it private, or rewrite history and ask GitHub Support to remove
   the pull-request refs and the LFS object.
4. GitGuardian's app check failed on PR #24 over four old incidents: the public
   development-database password and placeholder values, not live credentials. Resolve them as
   false positives in the GitGuardian dashboard, or remove the app; it is not a required check.
5. Export the database v2 was trained on, or record it as lost (private note, item 7).
6. Review Sibyl draft pull request #2718, starting with its "Production effect" section (the
   news timer and the weekly NFL train unit on gmk), and merge it or not. The NFL items it leaves
   open are in Sibyl's TODOS and in its Review section.
7. The machine-specific chores in the private note in `backups/local/`, chiefly deleting the
   audited sync-daemon copies (one `sudo rm` command there).
8. Optional, once week 8 is final (Monday 2026-11-02): grade the prospective week 7-8
   predictions with the recipe in
   [prospective/README.md](docs/investigations/2026-10-final-evaluation/prospective/README.md#grading-after-the-games).
   They are baselines with no claimed edge; 27 games settle nothing, but they are the project's
   only prospective record.

## Open: known technical issues (not planned)

- The dissertation is uncorrected; [ERRATA §6](analysis/dissertation/ERRATA.md#6-what-a-corrected-build-requires)
  lists what a corrected build needs.
- No data ingestion path remains (the R ingest scripts were removed); new seasons come from the
  nflverse release files. `scripts/pipeline_health_check.py` still prints the removed R commands as
  remediation, and it checks QBR and NextGen inputs that v2 never uses.
- `tests/integration` is red on stale expectations, not data problems: 66 passed, 20 failed,
  8 errors and 3 skipped against devdb01 on 2026-10-25 (read-only session;
  [tests/README.md](tests/README.md#integration-status)). `tests/infrastructure` was not run.
- Some scripts kept for provenance do not run as committed: `scripts/train_game_outcome_model.py`
  is a pre-v2 trainer (it did not produce v2), needs `psycopg_pool`, which `uv.lock` does not
  contain, and starts training when run; `scripts/validate_conformal_bnn.py` imports
  `nfl_analytics.models.bnn_conformal`, removed in the prune. `scripts/backfill_historical_odds.py`
  still has the matcher bugs of POSTMORTEMS §7 and spends paid API quota if run.
- `public.weather` wind speeds from the Meteostat ingest are in the wrong unit (ERRATA E5); use
  nflverse game-day wind. `plays.quarter` and the clock columns are NULL for every 1999-2024 play.
- `ml.model_registry` still lists v2 (model_id 17) as `production`.
- `nfl_analytics/backtest/harness_multimodel.py` fails at import (`DEFAULT_FEATURE_COLUMNS` no longer
  exists in `baseline_glm.py`), and its input feature file is not in git, so the 2004-2024 harness
  cannot be rerun; its published predictions are graded as stored.
- `scripts/final_evaluation/check_claims.py` reproduces only from a worktree of
  `archive/pre-prune-2026-10` (it scans files the prune removed).
- Five dissertation figure generators make their numbers up; each now carries a warning header.
  Do not regenerate figures from them: `scripts/generate_gnn_calibration_plot.py` (ERRATA P46),
  `scripts/generate_alpha_sensitivity_figure.py` (P01), `scripts/generate_bnn_improvement_figures.py`
  (P14, P15), `R/dissertation_figures.R` (P02, P07, P09, P28, P29/P51) and
  `analysis/dissertation/figures/R/gnn_optimization_figures.R` (P10-P12, T125, T259, X02).
- Earlier open items closed without action: ADR re-tests and retrains (the project is wound down),
  the inverted feature encodings (the models that use them are frozen or removed), the
  meta-labeling retrain and the props Dirichlet weights (code removed in the prune).
