# Documentation

Current reference documentation. Dated session logs and reports were archived on 2026-10-24 —
see [HISTORY.md](HISTORY.md) for the timeline and how to restore them. Docs for the subsystems
removed in the 2026-10 prune (trading, API, operations, agents, causal designs, ensemble and R
model guides) are in tag `archive/pre-prune-2026-10`.

## Start here

- [../README.md](../README.md) — what the project is and its honest results
- [../SETUP.md](../SETUP.md) — environment setup (uv, R, database)
- [POSTMORTEMS.md](POSTMORTEMS.md) — failures and root causes (read §4, spread sign convention, before writing any betting math)
- [adr/README.md](adr/README.md) — architecture decision records, with evidence quality noted

## Evaluation and results

- [investigations/2026-10-final-evaluation/](investigations/2026-10-final-evaluation/) — final grade of the models against the closing line, the 2004–2024 walk-forward regrade and the retraction ledger
- [investigations/2026-10-market-benchmark.md](investigations/2026-10-market-benchmark.md) — baselines vs the closing line; reproduces the 74.9% retraction
- [investigations/2026-10-line-movement/](investigations/2026-10-line-movement/) — line movement and closing-line value from the 2020–2024 odds snapshots (aggregates only)
- [investigations/2026-10-market-facts/](investigations/2026-10-market-facts/) — how closing lines have behaved since 1999, from nflverse schedules
- [investigations/2026-02-season-review/](investigations/2026-02-season-review/) — full 2025 season review (WS1–WS7)
- [TRAINING_WINDOW_ABLATION_RESULTS.md](TRAINING_WINDOW_ABLATION_RESULTS.md) — data behind the dissertation's training-window figure (`R/dissertation_figures.R`)
- [TEMPORAL_LEAKAGE_PROTECTION.md](TEMPORAL_LEAKAGE_PROTECTION.md), [TEMPORAL_SAFETY_TESTING_FRAMEWORK.md](TEMPORAL_SAFETY_TESTING_FRAMEWORK.md)

## Data

- [database/DATABASE_QUICK_REFERENCE.md](database/DATABASE_QUICK_REFERENCE.md), [database/SQL_PATTERNS.md](database/SQL_PATTERNS.md)
- [CROSS_PLATFORM_SETUP.md](CROSS_PLATFORM_SETUP.md) — database objects created outside the migrations

## Models and methods

- [models/GAME_OUTCOME_MODEL_GUIDE.md](models/GAME_OUTCOME_MODEL_GUIDE.md)
- [features/NEXTGEN_FEATURES_DOCUMENTATION.md](features/NEXTGEN_FEATURES_DOCUMENTATION.md)

## Architecture

- [architecture/ARCHITECTURE_REFERENCE.md](architecture/ARCHITECTURE_REFERENCE.md)

## Guides

- [guides/DISSERTATION_STANDARDS.md](guides/DISSERTATION_STANDARDS.md)
