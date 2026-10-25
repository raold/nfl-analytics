# Documentation

The project is a wound-down research archive; [../README.md](../README.md) has the results and
[../PROJECT_STATUS.md](../PROJECT_STATUS.md) what is left to decide. Docs, code and data removed
in October 2026 are in local git tags ([HISTORY.md](HISTORY.md)): `docs-archive-2026-10` for the
~850 docs files (session logs and reports) pruned on 2026-10-24, `archive/pre-prune-2026-10` for
everything the salvage prune removed. A path cited in a doc that no longer exists in the tree is
almost always in one of the two (`git show archive/pre-prune-2026-10:<path>`).

## Start here

- [../README.md](../README.md): what the project is, the canonical results, setup and commands
- [../CLAUDE.md](../CLAUDE.md): the critical rules (spread sign, kickoff time zone, database, odds licence)
- [POSTMORTEMS.md](POSTMORTEMS.md): what went wrong and why; read §4 (spread sign convention)
  before writing any betting math
- [adr/README.md](adr/README.md): design decisions, their status and how their evidence held up

## Evaluation and results (October 2026)

- [investigations/2026-10-final-evaluation/](investigations/2026-10-final-evaluation/README.md):
  the final grade against the closing line (Game Outcome v2 on 2025, walk-forward baselines, the
  Kalman state-space model, prospective 2026 predictions), with its
  [retraction ledger](investigations/2026-10-final-evaluation/retraction_ledger.md), the
  [2004-2024 walk-forward regrade](investigations/2026-10-final-evaluation/walkforward_2004_2024.md)
  and the [negative-result evidence](investigations/2026-10-final-evaluation/evidence/README.md)
  kept from November 2025 (a market blend that put all its weight on the line, XGBoost v6 losing
  to the line every season with the line as an input, GNN configurations all worse than a coin
  flip on Brier)
- [investigations/2026-10-market-benchmark.md](investigations/2026-10-market-benchmark.md): the
  grading tool, its options and guardrails
- [investigations/2026-10-line-movement/](investigations/2026-10-line-movement/README.md): line
  movement and closing-line value from the licensed 2020-2024 odds snapshots (aggregates only)
- [investigations/2026-10-market-facts/](investigations/2026-10-market-facts/README.md): how
  closing lines behaved in 1999-2025, from nflverse schedules
- [../analysis/dissertation/ERRATA.md](../analysis/dissertation/ERRATA.md): errors in the
  dissertation and the full audit of its tables and figures

## Earlier investigations

- [investigations/2026-02-season-review/](investigations/2026-02-season-review/README.md): the
  February 2026 review of the 2025 season (WS1-WS7), with notes on which findings hold
- [TRAINING_WINDOW_ABLATION_RESULTS.md](TRAINING_WINDOW_ABLATION_RESULTS.md): the November 2025
  training-window study behind ADR-007 (retracted; kept as the source cited by a dissertation figure)

## Data

- [database/DATABASE_QUICK_REFERENCE.md](database/DATABASE_QUICK_REFERENCE.md): tables, counts,
  semantics of `kickoff`, `spread_close` and `odds_history`, and objects created outside the migrations
- [database/SQL_PATTERNS.md](database/SQL_PATTERNS.md): joins that work and the schema traps
- [../db/README.md](../db/README.md): schema bootstrap, migrations history, restores

## Models, features and methods

- [../models/production/game_outcome_v2/README.md](../models/production/game_outcome_v2/README.md):
  the frozen Game Outcome v2 (how it was trained, its 2025 grade, its defects; DO NOT DEPLOY)
- [features/NEXTGEN_FEATURES_DOCUMENTATION.md](features/NEXTGEN_FEATURES_DOCUMENTATION.md): the
  Next Gen Stats feature module
- [TEMPORAL_SAFETY_TESTING_FRAMEWORK.md](TEMPORAL_SAFETY_TESTING_FRAMEWORK.md): what the leakage
  tests check, and what they missed
- [architecture/ARCHITECTURE_REFERENCE.md](architecture/ARCHITECTURE_REFERENCE.md): module and
  script map of the pruned tree

## Other

- [guides/DISSERTATION_STANDARDS.md](guides/DISSERTATION_STANDARDS.md): the standards the
  dissertation was written to (and, per the errata, missed)
- [HISTORY.md](HISTORY.md): timeline and how to restore archived docs
- [../tests/README.md](../tests/README.md), [../SETUP.md](../SETUP.md),
  [../CONTRIBUTING.md](../CONTRIBUTING.md)
