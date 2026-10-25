# Architecture Decision Records

Each ADR records a decision, its context, the alternatives and the consequences. Use
[template.md](template.md) for new ones and take the next free number.

Numbers 001–006 and 009–019 are original. 007, 008, 020 and 021 were moved here from the old
`docs/decisions/` directory on 2026-10-24 (each file says what it was renumbered from); existing
numbers were not changed because the dissertation cites them.

**State in October 2026.** The project is wound down. The ADRs are kept as the record of what was
decided and why; most of the code they describe was removed in the 2026-10 prune and is at tag
`archive/pre-prune-2026-10` ([../HISTORY.md](../HISTORY.md)), so paths in them often no longer
exist in the tree. Where a number in an ADR was later retracted, the ADR carries a banner citing
the row of the [retraction ledger](../investigations/2026-10-final-evaluation/retraction_ledger.md)
(C01-C44) that corrects it; the original text is left below the banner.

**Evidence column:** *point estimate* means a decision taken from a single-season comparison
with no uncertainty estimate; *invalid* means the evidence was retracted.

| ADR | Decision | Status | Evidence |
|-----|----------|--------|----------|
| [001](001-docker-compose-infrastructure.md) | Docker Compose for the service stack | Superseded: compose reduced to the database service (2026-10) | — |
| [002](002-git-lfs-migration.md) | Git LFS for model artifacts | Accepted; LFS now holds only the two 2025-12-07 database dumps in `backups/` (no odds rows) | — |
| [003](003-xgboost-v4-ridge-residuals.md) | XGBoost v4 ridge-residual features | **Superseded** (target leakage; honest MAE 59.14) | invalid (C44) |
| [004](004-bnn-research-abandonment.md) | Abandon BNN research | Accepted (tag `archive/bnn-2025-11`) | — |
| [005](005-temperature-scaling-resolution.md) | Temperature scaling resolution | Superseded: the v6 ensemble was removed (2026-10) | — |
| [006](006-postgresql-r-model-lookup.md) | PostgreSQL lookup for R models | Superseded: code removed (2026-10) | the batch script wrote placeholder draws |
| [007](007-canonical-training-window.md) | Canonical training window | Accepted; evidence retracted | invalid (C13: same-game inputs, chosen on the test weeks) |
| [008](008-hybrid-static-incremental.md) | Hybrid static/incremental XGBoost | Never deployed; code removed (2026-10) | point estimate, outcome-filtered sample, mix chosen on the test seasons |
| [009](009-abandon-ensemble-v7-meta-learning.md) | Abandon v7 meta-learning, keep static weights | Accepted | point estimate, in sample (C14) |
| [010](010-synthetic-features-dec9.md) | Add synthetic features | Accepted; its motivating numbers retracted | invalid (C12, C09) |
| [011](011-xgboost-feature-optimization.md) | Keep baseline XGBoost features | Accepted | point estimate (one 272-game season) |
| [012](012-weekly-feedback-loop.md) | Weekly feedback loop | Abandoned: its tables never existed in devdb01; code removed (2026-10) | no valid test (C25) |
| [013](013-dec28-2025-validation.md) | Dec 28 changes validation | Rest test **invalidated** (spread sign bug) | invalid (C15) |
| [014](014-xgboost-v2.1-ab-test-results.md) | Reject XGBoost v2.1 | Completed; comparison invalid | invalid (C16, C17: tuned on the test season, different seasons compared) |
| [015](015-kalman-gp-features-disabled.md) | Disable Kalman/GP features | Accepted on engineering grounds only | invalid comparison (C16) |
| [016](016-cql-target-leakage-fix.md) | CQL target leakage fix | Accepted; RL code removed (2026-10) | C39 |
| [017](017-2026-season-roadmap.md) | 2026 season roadmap | Not implemented (project wound down; §2.4 withdrawn) | C01-C04, C29, C30 |
| [018](018-kalshi-trading-system.md) | Kalshi trading system | Superseded by the Sibyl repo; code removed (2026-10) | C03 |
| [019](019-claude-api-trading-integration.md) | Claude API trade-signal validation | Superseded by the Sibyl repo; code removed (2026-10) | — |
| [020](020-relaxed-convergence-threshold.md) | Relax R-hat threshold to 1.02 | Accepted; its ROI figures are projections | projection (C09, C10) |
| [021](021-ensemble-calibration-production.md) | Ensemble calibration for production | Abandoned (projections only) | projection (C13, C36) |

Related: [../POSTMORTEMS.md](../POSTMORTEMS.md) records failures and their root causes.
