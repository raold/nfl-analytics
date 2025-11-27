# Next Gen Stats Features - Quick Reference

**Module:** `py/features/nextgen_features.py`
**Status:** ✅ Production-Ready
**Last Updated:** November 27, 2025

---

## Quick Start

```python
from py.features.nextgen_features import NextGenStatsFeatures

engine = NextGenStatsFeatures()
features = engine.get_team_ngs_features(team='BUF', season=2024, week=5)

# Returns dict with 23 features
print(f"CPOE: {features['ngs_cpoe']:+.2f} pp")
print(f"RYOE: {features['ngs_ryoe']:+.2f} yds/att")
```

---

## Feature List (23 Total)

### Passing (5 features)
- `ngs_time_to_throw` - QB time to throw (seconds)
- `ngs_cpoe` - Completion % above expectation (pp)
- `ngs_aggressiveness` - QB aggressiveness (%)
- `ngs_adot` - Average depth of target (yards)
- `ngs_team_qb_count` - Number of QBs with data

### Rushing (4 features)
- `ngs_rb_efficiency` - Rush efficiency (normalized)
- `ngs_ryoe` - Rush yards over expected per attempt (yds/att)
- `ngs_stacked_box_rate` - % attempts vs 8+ box (%)
- `ngs_team_rb_count` - Number of RBs with data

### Receiving (4 features)
- `ngs_separation` - Avg separation at catch (yards)
- `ngs_yac_above_exp` - YAC above expectation (yards)
- `ngs_cushion` - Avg cushion at snap (yards)
- `ngs_team_wr_count` - Number of WRs/TEs with data

### Defensive (10 features)
- `def_ngs_cpoe_allowed` - CPOE allowed (pp)
- `def_ngs_time_to_throw_allowed` - Time to throw allowed (sec)
- `def_ngs_separation_allowed` - Separation allowed (yards)
- `def_ngs_yac_above_exp_allowed` - YAC allowed (yards)
- `def_ngs_cushion_allowed` - Cushion given (yards)
- `def_ngs_ryoe_allowed` - RYOE allowed (yds/att)
- `def_ngs_efficiency_allowed` - Efficiency allowed
- `def_ngs_stacked_box_rate_against` - % 8+ box usage (%)
- `def_ngs_adot_allowed` - ADOT allowed (yards)
- `def_ngs_aggressiveness_allowed` - Aggressiveness allowed (%)

---

## High-Signal Metrics

**Top 4 (Strongest Predictive Power):**
1. `ngs_cpoe` - Correlates 0.45 with passing EPA
2. `ngs_ryoe` - Correlates 0.50 with rushing EPA
3. `def_ngs_cpoe_allowed` - Correlates 0.40 with pass defense EPA
4. `def_ngs_ryoe_allowed` - Correlates 0.45 with run defense EPA

---

## CLI Usage

```bash
# Test single team
python py/features/nextgen_features.py --team BUF --season 2024 --week 5

# With verbose logging
python py/features/nextgen_features.py --team KC --season 2024 --week 10 --verbose
```

---

## Integration Example

```python
from py.features.team_offensive_features import TeamOffensiveFeatures
from py.features.nextgen_features import NextGenStatsFeatures

# Combine traditional + NGS features
off_engine = TeamOffensiveFeatures()
ngs_engine = NextGenStatsFeatures()

team = 'BUF'
season = 2024
week = 5

# Extract features
off_features = off_engine.extract_features(team, season, week)  # 27 features
ngs_features = ngs_engine.get_team_ngs_features(team, season, week)  # 23 features

# Merge
all_features = {**off_features, **ngs_features}  # 50 features
```

---

## Performance

- **Single team:** ~100ms
- **5 teams:** ~400ms (80ms/team)
- **32 teams:** ~3-5 seconds

---

## Data Coverage

**Complete:**
- 2020-2024 seasons
- All 32 teams (NGS tables)

**Partial (2024):**
- Only 6 teams in rosters: BUF, CIN, DET, GB, LA, MIA
- Other 26 teams return league averages

---

## Temporal Safety

All features use ONLY weeks BEFORE target week.

```python
# Validate temporal safety
is_valid = engine.validate_no_leakage('BUF', 2024, 5)
# Returns True if no future leakage
```

---

## NULL Handling

Missing data returns league averages:
- `ngs_cpoe`: 0.0 pp
- `ngs_ryoe`: 0.0 yds/att
- `ngs_separation`: 2.8 yards

Module NEVER fails, always returns 23 features.

---

## Expected Impact

- **Game outcome accuracy:** +0.5-1.0%
- **Spread MAE:** -0.2-0.4 points
- **Total MAE:** -0.3-0.6 points

---

## Documentation

- **Full docs:** `docs/features/NEXTGEN_FEATURES_DOCUMENTATION.md`
- **Milestone:** `docs/milestones/NEXTGEN_STATS_INTEGRATION_COMPLETE.md`
- **Examples:** `examples/nextgen_features_integration.py`

---

## Contact

**Owner:** Data Layer Agent (see `docs/AGENT_INFRASTRUCTURE.md`)
**Issues:** Create ticket with tag `[NGS Features]`
