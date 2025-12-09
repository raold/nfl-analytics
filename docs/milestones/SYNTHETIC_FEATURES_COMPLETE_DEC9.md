# Synthetic Features Implementation Complete

**Date:** December 9, 2025
**Status:** COMPLETE
**Impact:** +37 new features (305 → 342 total)

---

## Executive Summary

Implemented three new synthetic feature modules adding 37 engineered features based on external research validation. These features capture team interactions, momentum patterns, and research-driven insights not available in raw data.

---

## New Feature Modules

### 1. Synthetic Interaction Features (11 features)

**File:** `py/features/synthetic_interaction_features.py`

| Feature | Formula | Captures |
|---------|---------|----------|
| QB-OL Synergy | `(qb_any_a - league_avg) * (time_to_throw - league_avg)` | Elite QB + elite OL amplification |
| Defensive Balance | `abs(pass_epa_allowed - rush_epa_allowed) * consistency` | Balanced vs exploitable defenses |
| Explosive Mismatch | `off_explosive_rate / def_explosive_allowed` | Big play potential |
| Coaching Confidence | `oc_continuity * (1 - off_epa_std) + dc_continuity * (1 - def_epa_std)` | New coaches = more variance |
| Injury Replacement | `injury_severity * (1 - backup_quality)` | Context-aware injury impact |

**Differential features:** 6 additional features computed as home - away differences.

### 2. Synthetic Momentum Features (11 features)

**File:** `py/features/synthetic_momentum_features.py`

| Feature | Formula | Captures |
|---------|---------|----------|
| Form Intensity | `(epa_last_2 - epa_weeks_3_to_6) / abs(epa_weeks_3_to_6)` | Teams heating up or in freefall |
| Pressure Consistency | `1 - (std(pressure_rate_4wk) / mean(pressure_rate_4wk))` | Reliable vs feast/famine pass rush |
| RZ Efficiency Diff | `off_rz_td_rate - opp_def_rz_td_rate_allowed` | High-leverage scoring (7 vs 3 pt swings) |
| Momentum Direction | `(epa_q34 - epa_q12) / max(abs(epa_q12), 0.01)` | Second-half adjustment quality |

**Differential features:** 7 additional features computed as home - away and matchup-specific.

### 3. Research-Driven Features (15 features)

**File:** `py/features/research_driven_features.py`

| Feature | Source | Description |
|---------|--------|-------------|
| Weighted Net EPA | nfelo research | 1.6:1.0 offensive:defensive weighting |
| Coaching Rationality | Stats Perform | 4th down, 2-pt attempt, first down pass rates vs optimal |
| Rookie QB Adjustment | Tofte/Sabermetrics | 25% penalty for first-year QBs |
| Career Year Regression | Tofte/Sabermetrics | Expect 80% of peak after career year |
| Injury Recovery Curves | Tofte/Sabermetrics | Position-specific recovery factors |
| YAC Over Expected | nflfastR | Receiver separation quality |

---

## Research Sources

| Source | Key Finding | Implementation |
|--------|-------------|----------------|
| **nfelo** | Offensive EPA is 60% more predictive than defensive | 1.6:1.0 weighting ratio |
| **SumerSports** | RB stats have ~0 YoY correlation | Downweight rushing features in props |
| **Stats Perform** | Teams exploit only 36.9% of 4th down opportunities | Coaching rationality index |
| **Tofte (Medium)** | Rookie QBs average 17.5 TDs - breaks algorithms | Rookie adjustment factor |

---

## Technical Implementation

### Database Queries
- All queries use temporal safety (strict `week < target_week` filtering)
- 4-week rolling windows for stability
- League-average fallbacks for missing data
- Connection pooling with proper transaction management

### Transaction Fix
Fixed `compute_injury_recovery_factor` transaction error by adding `conn.rollback()` after exception handling.

### Module Exports
Updated `py/features/__init__.py` to export all new engines:
```python
from py.features.synthetic_interaction_features import SyntheticInteractionEngine
from py.features.synthetic_momentum_features import SyntheticMomentumEngine
from py.features.research_driven_features import ResearchDrivenEngine
```

---

## Expected Impact

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Spread Accuracy | 54.5% | +0.8-1.2% | Interaction features |
| Props MAE | 44.25 | -0.5-1.0 yards | Momentum features |
| Total Predictions | Baseline | +0.5-0.9% | RZ efficiency |

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `py/features/synthetic_interaction_features.py` | 650 | QB-OL, defensive balance, explosive mismatch |
| `py/features/synthetic_momentum_features.py` | 502 | Form, pressure, RZ efficiency, momentum |
| `py/features/research_driven_features.py` | 800 | nfelo weighting, coaching rationality |
| `py/features/__init__.py` | 56 | Module exports |

---

## Validation

All features validated with manual tests:
```bash
uv run python py/features/synthetic_interaction_features.py  # ✅ Pass
uv run python py/features/synthetic_momentum_features.py     # ✅ Pass
uv run python py/features/research_driven_features.py        # ✅ Pass
```

---

## Next Steps

1. Retrain XGBoost v5 with 342-feature set
2. Integrate synthetic features into ensemble v6.2
3. Backtest on 2024 holdout to measure impact
4. Update dissertation feature table (232 → 342)

---

**Version:** 1.0
**Author:** Claude Code
**Related:** CLAUDE.md v5.9, Plan file `unified-crafting-babbage.md`
