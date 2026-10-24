# Architecture Decision Records

Each ADR records a decision, its context, the alternatives and the consequences. Use
[template.md](template.md) for new ones and take the next free number.

Numbers 001–006 and 009–019 are original. 007, 008, 020 and 021 were moved here from the old
`docs/decisions/` directory on 2026-10-24 (each file says what it was renumbered from); existing
numbers were not changed because the dissertation cites them.

**Evidence column:** several decisions were made from a single-season point comparison with no
uncertainty estimate. Those are marked *point estimate* and are candidates for re-testing with
`nfl_analytics/evaluation/market_benchmark` (bootstrap intervals, Diebold–Mariano, market baseline).

| ADR | Decision | Status | Evidence |
|-----|----------|--------|----------|
| [001](001-docker-compose-infrastructure.md) | Docker Compose for the service stack | Accepted | — |
| [002](002-git-lfs-migration.md) | Git LFS for model artifacts | Accepted | — |
| [003](003-xgboost-v4-ridge-residuals.md) | XGBoost v4 ridge-residual features | **Superseded** (target leakage; honest MAE 59.14) | invalid |
| [004](004-bnn-research-abandonment.md) | Abandon BNN research | Accepted | — |
| [005](005-temperature-scaling-resolution.md) | Temperature scaling resolution | Accepted | — |
| [006](006-postgresql-r-model-lookup.md) | PostgreSQL lookup for R models | Accepted | — |
| [007](007-canonical-training-window.md) | Canonical training window | Accepted | point estimate |
| [008](008-hybrid-static-incremental.md) | Hybrid static/incremental XGBoost | Approved | point estimate (38.03 vs 39.66 MAE, no CI) |
| [009](009-abandon-ensemble-v7-meta-learning.md) | Abandon v7 meta-learning, keep static weights | Accepted | point estimate (DM p = NaN) |
| [010](010-synthetic-features-dec9.md) | Add synthetic features | Accepted | point estimate |
| [011](011-xgboost-feature-optimization.md) | Keep baseline XGBoost features | Accepted | point estimate (one 272-game season) |
| [012](012-weekly-feedback-loop.md) | Weekly feedback loop | Accepted | — |
| [013](013-dec28-2025-validation.md) | Dec 28 changes validation | Rest test **invalidated** (spread sign bug) | invalid |
| [014](014-xgboost-v2.1-ab-test-results.md) | Reject XGBoost v2.1 | Completed | confounded (different training windows) |
| [015](015-kalman-gp-features-disabled.md) | Disable Kalman/GP features | Accepted | point estimate (163 games) |
| [016](016-cql-target-leakage-fix.md) | CQL target leakage fix | Accepted | — |
| [017](017-2026-season-roadmap.md) | 2026 season roadmap | Proposed (§2.4 withdrawn) | — |
| [018](018-kalshi-trading-system.md) | Kalshi trading system (superseded by the Sibyl repo) | Accepted | — |
| [019](019-claude-api-trading-integration.md) | Claude API trade-signal validation | Proposed | — |
| [020](020-relaxed-convergence-threshold.md) | Relax R-hat threshold to 1.02 | Accepted | — |
| [021](021-ensemble-calibration-production.md) | Ensemble calibration for production | In progress | — |

Related: [../POSTMORTEMS.md](../POSTMORTEMS.md) records failures and their root causes.
