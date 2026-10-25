# Temporal Safety: Tests and Their Limits

**Updated:** 2026-10-25. This replaces the November 2025 framework report, which declared "15/15
feature modules CLEAN" and the system "SAFE for production deployment". That verdict did not
hold: the leaks that actually inflated this project's results were found later, by other means.
The earlier text is at tag `archive/pre-prune-2026-10`.

## What leakage looked like here

| Kind | Example | Where it was found |
|---|---|---|
| Target leakage | XGBoost v4 residual features built from the target; CQL rewards from the outcome | ADR-003, ADR-016 |
| Same-game inputs | every props model fed the game's own pass attempts | retraction ledger C09, C13 |
| Same-season data | game-ensemble predictions built from full-season team stats of the season predicted (81.5% straight-up) | C19 |
| Look-ahead inputs | semantic features read as of extraction time for past games | POSTMORTEMS §9, C24 |
| Tuning on the evaluation set | early stopping on the validation or test season (v2's 10.45, v2.1's 10.70) | C16, C17 |
| Grading errors that look like skill | the spread sign bug | POSTMORTEMS §4 |

None of these was caught by the static checks below. Two kinds of check did catch them: comparing
a result with the market and with base rates (`market_benchmark` refuses win rates above 60% or
ROI above 15% on 100+ bets), and no-skill baselines built from the same inputs (the game's
attempts × league yards per attempt scores 45.28, against the 44.25 claimed for the props models).

## What the tests check

**In CI** (`.github/workflows/temporal-safety.yml`, no database):

- `tests/unit/test_spread_convention.py`: the cover rule on real games and the pooled base rate.
- `tests/unit/test_data_leakage_prevention.py` and `tests/unit/test_data_leakage_detector_comprehensive.py`:
  `nfl_analytics/utils/data_leakage_detector.py` on known-good and known-bad frames (temporal
  splits, rolling windows with and without `shift(1)`, negative shifts, the target among the
  features, code patterns).
- The `spread-sign-convention` pre-commit hook over the whole tree.

**With the database** (`uv run pytest -m integration tests/integration`, not in CI):

- `test_gfe_pinned_games.py`: the game feature extractor on three pinned games; it records every
  row the queries return and fails if a row with outcome columns comes from the target week or
  later.
- `test_temporal_safety_comprehensive.py`, `test_temporal_leakage.py`, `test_temporal_leakage_critical.py`:
  as-of joins, lag and rolling patterns, train/test ordering. Several of these fail on the
  current schema (stale expectations; [tests/README.md](../tests/README.md#integration-status)).

## Using the detector

```python
from nfl_analytics.utils.data_leakage_detector import DataLeakageDetector

detector = DataLeakageDetector()
detector.validate_split(train_df, test_df, date_col="kickoff")
if detector.report.has_critical_leakage():
    raise ValueError(detector.generate_report())
```

It catches mechanical mistakes (a test date before the last training date, an unshifted rolling
mean). It cannot tell whether an input was known before kickoff.

## Rules that would have prevented the real leaks

1. Every feature query takes the prediction time as a parameter; "now" is never a default.
2. Early stopping, model selection and thresholds use seasons that are not reported.
3. No input from the game being predicted, and no season aggregate of the season being predicted;
   `week >= 1` on tables that carry week-0 season totals.
4. A result is graded against the closing market with `market_benchmark` before it is reported,
   and compared with a no-skill baseline built from the same inputs.
5. A prediction counts as prospective only if it was published before kickoff.
