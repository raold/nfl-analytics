# ADR-016: CQL Target Leakage Fix

**Status:** Accepted
**Date:** 2026-01-02
**Deciders:** Claude Code, User
**Tags:** ml-model, data, rl

---

## Context

### Problem Statement

The CQL (Conservative Q-Learning) evaluation pipeline in `nfl_analytics/rl/` had critical target leakage that produced impossible metrics in the dissertation table `rl_vs_baseline_table.tex`:

- CQL (single): 93.1% win rate, +79.7% ROI
- CQL (ensemble): **100% win rate, +91.0% ROI, infinite Sharpe, 0% max DD**

These metrics are scientifically impossible in an efficient betting market where the breakeven win rate is 52.4% (at -110 odds).

### Background

The CQL pipeline was designed to:
1. Generate a logged dataset from historical games (`nfl_analytics/rl/dataset.py`)
2. Train CQL models on this dataset (models never actually existed)
3. Evaluate policies using `nfl_analytics/rl/evaluate_cql_betting.py`

However, the dataset generation included post-game outcomes (home_score, away_score) and pre-computed rewards, allowing models to "see the future."

### Decision Drivers

- **Scientific integrity:** Dissertation cannot contain fabricated metrics
- **Reproducibility:** Pipeline must produce realistic, verifiable results
- **Alignment with XGBoost fix:** Similar leakage was fixed in XGBoost v5.1 (Dec 10, 2025)

---

## Decision

Fix the CQL pipeline to prevent target leakage by:
1. Removing outcome columns from the dataset generation
2. Implementing proper temporal (season-based) train/test split
3. Loading outcomes separately at evaluation time
4. Fixing the spread cover formula sign error

### Details

**Files Modified:**

| File | Changes |
|------|---------|
| `nfl_analytics/rl/dataset.py` | Removed `home_score`, `away_score`, `r` from output columns |
| `nfl_analytics/rl/evaluate_cql_betting.py` | Added `load_game_outcomes()`, season-based split, fixed cover formula |
| `figures/out/rl_vs_baseline_table.tex` | Updated with honest baseline metrics |

**Critical Bug Found:**

The cover formula had an incorrect sign:
```sql
-- WRONG (gave 58.6% home cover rate)
CASE WHEN (home_score - away_score) > -spread_close THEN 1 ELSE 0 END

-- CORRECT (gives 49.3% home cover rate)
CASE WHEN (home_score - away_score) > spread_close THEN 1 ELSE 0 END
```

The database uses positive `spread_close` for home favorites (giving points), not the standard convention.

---

## Alternatives Considered

### Alternative 1: Document as Known Limitation

**Description:** Add footnote about data quality issues, keep fabricated metrics

**Pros:**
- No code changes required
- Faster to implement

**Cons:**
- Scientifically dishonest
- Could invalidate dissertation
- Perpetuates incorrect results in academic record

**Evaluation:** Rejected - academic integrity is non-negotiable

### Alternative 2: Remove Table Entirely

**Description:** Use `\IfFileExists` fallback to hide the table

**Pros:**
- Avoids showing any potentially incorrect data
- No risk of fabricated metrics

**Cons:**
- Loses legitimate baseline results
- Chapter 6 empirical claims would have no supporting data

**Evaluation:** Rejected - we can show honest baseline results

### Alternative 3: Full Fix + Retrain (Chosen)

**Description:** Fix leakage, regenerate dataset, show baselines, mark CQL as pending

**Pros:**
- Scientifically correct
- Shows honest market efficiency results
- Leaves path for future CQL training

**Cons:**
- CQL results unavailable until retraining
- Requires code changes

**Evaluation:** Selected - best balance of integrity and content preservation

---

## Consequences

### Positive

- **Scientific integrity restored:** Baseline metrics now show realistic ~50% win rates and negative ROI
- **Pipeline is correct:** Future CQL training will produce valid results
- **Audit trail:** This ADR documents the fix for reproducibility

### Negative

- **CQL results unavailable:** No trained CQL models exist; table shows "pending retraining"
- **Weaker empirical claims:** Chapter 6 cannot claim CQL outperforms baselines without valid models

### Neutral

- **Dataset structure changed:** Clean dataset has 14 columns (down from 16)
- **Evaluation requires database:** `load_game_outcomes()` fetches from PostgreSQL

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CQL never retrained | Medium | Medium | Document methodology value regardless of results |
| Future spread convention changes | Low | Low | Cover formula now has detailed docstring |

---

## Implementation

### Approach

1. **Fix dataset.py (lines 156-182):**
   - Removed reward computation
   - Removed outcome columns from output
   - Added comments explaining the fix

2. **Fix evaluate_cql_betting.py:**
   - Added `load_game_outcomes()` function
   - Changed `--test-split` to `--train-seasons`/`--test-seasons`
   - Updated `calculate_metrics()` to accept outcomes parameter
   - Fixed cover formula sign

3. **Regenerated clean dataset:**
   ```bash
   python nfl_analytics/rl/dataset.py --output data/rl/betting_features_clean.csv
   ```

4. **Updated dissertation table:**
   - Shows realistic baseline metrics
   - Notes CQL pending retraining

### Timeline

- **Discovered:** 2026-01-02
- **Fixed:** 2026-01-02
- **Validated:** 2026-01-02 (baseline metrics now realistic)

### Success Criteria

- [x] Dataset contains no outcome columns (home_score, away_score, r)
- [x] Temporal split uses season boundaries, not percentage
- [x] Baseline win rates are ~50% (market efficiency)
- [x] Baseline ROI is negative (due to vig)
- [x] Dissertation table updated with honest metrics

### Validation

```
BETTING PERFORMANCE EVALUATION (TEST SET)
Train: [2019, 2020, 2021, 2022] | Test: [2023, 2024]

RANDOM
  Win rate:     50.5%
  ROI:          -3.49%

MARKET
  Win rate:     50.9%
  ROI:          -2.78%
```

These results confirm:
- Market efficiency (~50% win rate)
- Vig impact (negative ROI)
- No leakage (would show >90% win rate if leaked)

---

## Related Decisions

- [ADR-015: Kalman/GP Features Disabled](./015-kalman-gp-features-disabled.md) - Similar pattern of disabling flawed components
- XGBoost v5.1 Leakage Fix (Dec 10, 2025) - Prior leakage fix in different pipeline

---

## References

### Documentation

- Plan file (no longer in the repo; see [HISTORY](../HISTORY.md))
- [CLAUDE.md](../../CLAUDE.md) - Project documentation

### Implementation

- `nfl_analytics/rl/dataset.py` (removed in the 2026-10 prune; tag `archive/pre-prune-2026-10`) - Dataset generation
- `nfl_analytics/rl/evaluate_cql_betting.py` (removed in the 2026-10 prune; tag `archive/pre-prune-2026-10`) - Evaluation script
- [figures/out/rl_vs_baseline_table.tex](../../analysis/dissertation/figures/out/rl_vs_baseline_table.tex) - Dissertation table

---

## Notes

The spread sign convention in this database **is** the nflverse convention (`spread_line` is copied
verbatim into `games.spread_close`); it is inverted only relative to sportsbook notation:
- Positive `spread_close` = home is **favorite** (giving points)
- Negative `spread_close` = home is **underdog** (receiving points)

*(Corrected 2026-10-24; this note previously said "inverted from standard nflverse". Use
`nfl_analytics/utils/spread_convention.py` for all cover/edge math.)*

This is documented in the `load_game_outcomes()` docstring for future reference.

---

**Version:** 1.0
**Last Updated:** 2026-01-02
**Author:** Claude Code
