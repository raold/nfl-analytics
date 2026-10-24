# Project Status

**Updated:** 2026-10-24 · **State:** research platform, repaired; trading moved to the Sibyl repo

## Where things stand

- **Models do not beat the market.** Full 2025 season: spread MAE 11.05 vs the closing line's
  9.67; −3.2% ATS ROI on 208 bets (95% CI roughly −16% to +10%, so also not distinguishable
  from a small edge). Walk-forward power ratings and Elo are ~50% ATS on 2019–2024.
- **The "74.9% ATS / +43% ROI" power-ratings result (Dec 2025) is retracted.** It was a spread
  sign error. The same bug affected cover labels, backtests, two materialized views, the
  meta-labeling trade gate and ADR-013's rest result. Fixed project-wide on 2026-10-24
  ([POSTMORTEMS §4](docs/POSTMORTEMS.md)).
- **Dissertation:** 722 pages (`analysis/dissertation/main/main.pdf`, built 2026-02-12).
  Sections that rely on the affected labels need review (see open items).

## Repair of 2026-10-24 (branch `repair/2026-10`)

- Spread convention helper, pinning tests on real games, migration 104, pre-commit guard
- `py/` package renamed to `nfl_analytics/` (it shadowed pytest's `py` module)
- Dependencies declared in `pyproject.toml` (base + extras) with a committed `uv.lock`;
  CI, Dockerfiles and dev scripts install from the lock
- Offline test suite green: 1,511 passed (was 1,203 passed / 145 failed / 76 errors / 21
  collection errors); several real bugs fixed along the way (see git log)
- Docs pruned from ~1,035 to ~150 Markdown files; history in [docs/HISTORY.md](docs/HISTORY.md)
- New [`market_benchmark`](nfl_analytics/evaluation/market_benchmark/) with uncertainty
  intervals and plausibility guardrails

## Open items

Needs the database (`docker compose up -d pg-1`):

1. Apply `db/migrations/104_fix_spread_cover_sign.sql` and refresh the two views.
2. Confirm the convention and the retraction with the queries in POSTMORTEMS §4 (expected:
   `spread_close > 0` for ~all home-moneyline favorites; pooled home cover ≈ 49%).
3. Check that `games.spread_close` for 2025 weeks 12–18 holds real lines, not the EPA
   placeholders written on 2025-11-21 (the weekly ingest should have replaced them).
4. Benchmark Game Outcome v2 (`ml.game_outcome_predictions`) and the 2019–2025 ensemble CSV
   with `market_benchmark`; record results in `docs/investigations/`.
5. Re-test ADR-008/009/011/013/014/015 with intervals (they rest on single-season point
   comparisons; see the evidence column in [docs/adr/README.md](docs/adr/README.md)).

Needs a decision or a retrain:

6. Retrain the meta-labeling classifier (old models are now refused at load time).
7. Fix the inverted feature encodings at the next retrain (`home_is_favorite`, R props
   `is_favored`, swapped implied team totals); listed in POSTMORTEMS §4.
8. `EnhancedEnsemble5Way.DIRICHLET_WEIGHTS` (props) still gives the leaked XGBoost v4 model
   2.6%; equal weights already performed better on the 2025 holdout.
9. Review dissertation chapters that report ATS results built on the affected labels.
