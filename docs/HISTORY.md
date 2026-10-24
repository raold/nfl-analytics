# Project History

A short timeline of the project. Detailed session logs, milestone summaries, audits and fix
reports (about 920 files) were removed from the working tree on 2026-10-24; they are preserved
in the git tag **`docs-archive-2026-10`**.

## Restoring archived docs

```bash
git fetch --tags                                    # the tag must exist on your clone
git show docs-archive-2026-10:docs/milestones/RECENT_WORK_HISTORY.md | less
git checkout docs-archive-2026-10 -- docs/milestones   # restore a whole directory
git ls-tree -r --name-only docs-archive-2026-10 docs/  # list everything that was there
```

Comments in code and older docs that cite `docs/milestones/…`, `docs/fixes/…`, `docs/audits/…`,
`docs/reports/…` and similar paths refer to files in that tag.

What was removed (file counts): loose `docs/*.md` 169, milestones 128, archive 92, audits 73,
fixes 50, deployment 46, guides 32, performance 28, api 23, agent_context 23, reports 21,
architecture 15, research 14, knowledge 14, optimization 13, and ~30 smaller directories.

## Timeline

| Date | Event |
|------|-------|
| 2025-10 | Database schema audit, feature engineering, TikZ standardization for the dissertation |
| 2025-11-02 | Windows RTX 4090 GPU training; BNN calibration work |
| 2025-11-08 | Canonical training window established (ADR-007); state-space model approved |
| 2025-11-09 | BNN research abandoned after 84 hours / 6 approaches (ADR-004) |
| 2025-11-09 | Convergence threshold relaxed to R-hat < 1.02 (ADR-020) |
| 2025-11-18 | Ensemble v7 meta-learning rejected; static weights kept (ADR-009) |
| 2025-11-21 | Placeholder lines written into `games.spread_close` for 94 games (2025 wk 12–18); script deleted 2026-10-24 (POSTMORTEMS §4) |
| 2025-11-27 | Phase 2 advanced stats; Dirichlet weight optimization for props |
| 2025-12-07 | GNN v2 integration and model retraining |
| 2025-12-09 | Synthetic features (ADR-010). "Power ratings 74.9% ATS / +43% ROI" claimed — **refuted**: spread sign bug (POSTMORTEMS §4) |
| 2025-12-14 | XGBoost feature optimization; baseline retained (ADR-011) |
| 2025-12-21 | Semantic (LLM) feature stack |
| 2025-12-22 | Weekly feedback loop (ADR-012) |
| 2025-12-23 | Game Outcome model v2 promoted; CQL target leakage found (ADR-016) |
| 2025-12-28 | Rest/Kalman/GP features (ADR-013 — rest result invalidated 2026-10-24) |
| 2025-12-31 | Kalman/GP features disabled (ADR-015) |
| 2026-02-12 | Full 2025 season evaluated: spread MAE 11.05 vs Vegas 9.67, −3.2% ATS ROI on 208 bets (ADR-017, `investigations/2026-02-season-review/`) |
| 2026-02-13 | Kalshi trading system (ADR-018); later superseded by the separate Sibyl repo |
| 2026-10-24 | Repair: spread sign convention fixed project-wide, docs pruned, `py/` renamed to `nfl_analytics/`, dependencies and CI repaired, market benchmark added |
