# ADR-010: Synthetic Features Addition (December 9, 2025)

**Status:** Accepted
**Date:** 2025-12-09
**Deciders:** Project Team
**Tags:** feature-engineering, research-driven, ml-model, ensemble-enhancement

---

## Context

### Problem Statement

After achieving 54.5% spread accuracy and 44.25 MAE on props with Ensemble v6.0, the feature set (305 features) had reached a maturity plateau. While existing features capture team performance metrics (EPA, QBR, NextGen stats), they miss critical second-order effects:

1. **Interaction effects:** QB + OL synergy is more than the sum of individual components
2. **Momentum patterns:** Recent form diverging from medium-term baseline signals trajectory shifts
3. **Research-validated insights:** Published analytics findings not yet incorporated (nfelo EPA weighting, rookie QB penalties)

**Core Challenge:** How to push beyond the 305-feature plateau without adding noise or overfitting?

**Business Impact:**
- Spread accuracy target: +0.8-1.2% improvement
- Props MAE target: -0.5-1.0 yards improvement
- Competitive advantage: Incorporate academic research before competitors

### Background

**Current State (December 8, 2025):**
- 305 total features across 10 categories
- Feature categories: TeamOffensive (42), TeamDefensive (38), NextGen (28), QBR (18), OpponentAdjusted (45), TurnoverLuck (12), Primetime (8), SoV (24), PossessionValue (16), Playcalling (22)
- Feature extraction time: <500ms via GameFeatureExtractor
- XGBoost v4 trained on full feature set with Ridge residuals

**Research Gap Analysis:**
- **nfelo research:** Offensive EPA is 60% more predictive than defensive EPA (not weighted in current features)
- **SumerSports:** RB stats have near-zero year-over-year correlation (current features treat all positions equally)
- **Stats Perform:** Teams exploit only 36.9% of optimal 4th-down opportunities (coaching rationality not captured)
- **Tofte (Medium):** Rookie QBs average 17.5 TDs and systematically break prediction algorithms (no rookie adjustment)

**Timing:** December 9, 2025 (post-2025 holdout validation, pre-XGBoost v5 retraining)

### Decision Drivers

- **Research Validation:** Multiple published sources confirm specific feature patterns
- **Feature Engineering Maturity:** 305 features represents comprehensive raw data coverage
- **Second-Order Effects:** Interaction and momentum features capture non-linear relationships
- **Competitive Advantage:** Academic research provides validated signal before market incorporation
- **Risk Management:** Synthetic features can be validated independently before integration

---

## Decision

**We will add 37 synthetic features across three new modules: Interaction Features (11), Momentum Features (11), and Research-Driven Features (15).**

### Details

**Feature Modules:**

1. **Synthetic Interaction Features** (`py/features/synthetic_interaction_features.py`)
   - QB-OL Synergy: `(qb_any_a - league_avg) * (time_to_throw - league_avg)`
   - Defensive Balance Index: `abs(pass_epa - rush_epa) * consistency_score`
   - Explosive Mismatch: `off_explosive_rate / def_explosive_allowed`
   - Coaching Confidence: `oc_continuity * (1 - off_epa_std) + dc_continuity * (1 - def_epa_std)`
   - Injury Replacement Value: `injury_severity * (1 - backup_quality)`
   - Plus 6 differential features (home - away)

2. **Synthetic Momentum Features** (`py/features/synthetic_momentum_features.py`)
   - Form Intensity: `(epa_last_2 - epa_weeks_3_to_6) / abs(epa_weeks_3_to_6)`
   - Pressure Consistency: `1 - (std(pressure_rate_4wk) / mean(pressure_rate_4wk))`
   - RZ Efficiency Differential: `off_rz_td_rate - opp_def_rz_td_rate_allowed`
   - Momentum Direction: `(epa_q34 - epa_q12) / max(abs(epa_q12), 0.01)`
   - Plus 7 differential/matchup features

3. **Research-Driven Features** (`py/features/research_driven_features.py`)
   - Weighted Net EPA: `(off_epa * 1.6 - def_epa * 1.0) / 2.6` (nfelo 1.6:1.0 weighting)
   - Coaching Rationality Index: Weighted blend of 4th-down go rate, 2-pt attempts, 1st-down pass rate vs optimal
   - Rookie QB Adjustment: 25% penalty for first-year QBs (Tofte research)
   - Career Year Regression: 80% expectation after career year >130% of average
   - Injury Recovery Curves: Position and injury-type specific factors (Achilles 80%, ACL 90%, etc.)
   - YAC Over Expected: `actual_yac - expected_yac` from nflfastR
   - Plus 9 team-specific and differential features

**Technical Implementation:**
- Database queries with strict temporal safety (`week < target_week`)
- 4-week rolling windows for stability
- League-average fallbacks for missing data
- Connection pooling with proper transaction management
- Transaction rollback handling in `compute_injury_recovery_factor`

**Total Feature Count:** 305 → 342 (+37)

---

## Alternatives Considered

### Alternative 1: Continue with 305 Features Only

**Description:**
Freeze feature engineering and focus solely on hyperparameter tuning and ensemble weighting optimization.

**Pros:**
- Zero overfitting risk from new features
- Simpler feature pipeline (no new modules)
- Lower maintenance burden
- Faster training time

**Cons:**
- Leaves validated research findings on the table
- Competitors may incorporate same research
- Limited upside potential (hyperparameter tuning yields ~1-2% gains max)
- Ignores known interaction effects (QB-OL synergy)

**Evaluation:**
Rejected. Research-validated features represent low-risk signal addition. The nfelo EPA weighting (1.6:1.0) alone is based on extensive empirical validation and unlikely to be noise.

---

### Alternative 2: Full Feature Explosion (100+ New Features)

**Description:**
Generate all possible interactions, polynomial terms, and rolling windows across all existing features.

**Pros:**
- Maximizes XGBoost's ability to find complex patterns
- Automated feature generation (less manual curation)
- Potential for discovering unexpected relationships

**Cons:**
- HIGH overfitting risk (100+ features on 2018-2024 data)
- Feature selection overhead (need to prune low-importance features)
- Computational explosion (training time, storage)
- Interpretability loss (harder to explain predictions)
- Many features will be correlated noise

**Evaluation:**
Rejected. Diminishing returns and high overfitting risk. The 37 synthetic features are research-validated and theoretically grounded, unlike automated interactions which are exploratory.

---

### Alternative 3: Research Features Only (15 Features)

**Description:**
Add only the research-driven features from published sources (nfelo, Stats Perform, Tofte), skip interaction and momentum features.

**Pros:**
- Highest confidence (peer-reviewed/published research)
- Lowest overfitting risk (15 features vs 37)
- Simplest implementation (single module)
- Direct literature citations for dissertation

**Cons:**
- Misses obvious interaction effects (QB-OL synergy well-established)
- Momentum features validated in sports analytics (form intensity standard)
- Incomplete solution (leaves validated signal on table)
- Limited upside (+0.3-0.5% vs +1.0-1.5% target)

**Evaluation:**
Rejected as too conservative. While research-driven features have highest validation, interaction and momentum features are standard in sports analytics and low-risk additions.

---

### Alternative 4: Staged Rollout (Research → Interaction → Momentum)

**Description:**
Add features in three separate phases over 3 months, validating each module before proceeding.

**Pros:**
- De-risks implementation (isolate impact of each module)
- Easier to attribute performance changes
- Can abort if early modules show no benefit
- More rigorous experimental design

**Cons:**
- Delays full benefit by 2-3 months (NFL season is short)
- Three separate XGBoost retraining cycles (computational cost)
- Features may have synergistic effects (testing individually misleading)
- Opportunity cost (slow rollout vs fast validation)

**Evaluation:**
Rejected. The 37 features are small enough to validate together. NFL season timing (December 9 with 4 weeks remaining) makes staged rollout impractical. Can still measure individual module importance post-hoc via SHAP values.

---

## Consequences

### Positive

- **Research-Validated Signal:** nfelo EPA weighting (1.6:1.0) based on extensive empirical validation
- **Interaction Effects Captured:** QB-OL synergy, defensive balance, explosive mismatch fill known gaps
- **Momentum Patterns:** Form intensity and RZ efficiency capture trajectory shifts
- **Competitive Advantage:** Incorporating academic research before market
- **Dissertation Contribution:** 37 research-driven features strengthen Chapter 5 (Feature Engineering)
- **Maintainability:** Three focused modules (interaction, momentum, research) with clear purpose
- **Temporal Safety:** All features use strict `week < target_week` filtering (no data leakage)

**Quantitative Targets:**
- Spread accuracy: +0.8-1.2% (54.5% → 55.3-55.7%)
- Props MAE: -0.5-1.0 yards (44.25 → 43.25-43.75)
- Total predictions: +0.5-0.9% via RZ efficiency features

### Negative

- **Overfitting Risk:** 37 new features on limited NFL data (2018-2024 = 6 seasons)
  - **Mitigation:** All features research-validated, use XGBoost's regularization (Ridge residuals), monitor 2025 holdout performance
- **Feature Drift:** Some features may lose predictive power over time (e.g., coaching rationality as league evolves)
  - **Mitigation:** Annual feature importance review, SHAP value monitoring
- **Computational Cost:** Feature extraction time may increase
  - **Mitigation:** Current <500ms budget has headroom, can optimize queries if needed
- **Maintenance Burden:** Three new modules to maintain
  - **Mitigation:** Comprehensive docstrings, unit tests, clear research citations
- **Transaction Complexity:** Database queries with rollback handling increase failure modes
  - **Mitigation:** Exception handling added, connection pooling tested

### Neutral

- **Feature Count:** 305 → 342 (+12% increase, within XGBoost capacity)
- **Training Time:** XGBoost v5 retraining required (estimated 2-4 hours on full dataset)
- **Documentation Update:** Dissertation Chapter 5 feature table needs update (232 → 342)
- **Module Exports:** Updated `py/features/__init__.py` with new engines

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Overfitting on 6 seasons** | Medium | Medium | Monitor 2025 holdout, use Ridge regularization, validate feature importance |
| **Research findings don't generalize** | Low | Medium | Multiple independent sources, empirically validated before adoption |
| **Database query performance** | Low | Low | 4-week rolling windows efficient, connection pooling tested |
| **Feature drift over time** | Medium | Low | Annual review, SHAP monitoring, can remove features if importance drops |
| **Transaction errors** | Low | Medium | Rollback handling added, comprehensive error logging |

---

## Implementation

### Approach

**Phase 1: Research & Design (4 hours)** - Complete
1. Survey external research (nfelo, Stats Perform, Tofte, SumerSports)
2. Identify feature gaps in current 305-feature set
3. Design interaction, momentum, and research modules
4. Document formulas and expected impact

**Phase 2: Implementation (6 hours)** - Complete
1. Implement `SyntheticInteractionEngine` (11 features)
2. Implement `SyntheticMomentumEngine` (11 features)
3. Implement `ResearchDrivenEngine` (15 features)
4. Add transaction error handling (rollback in injury recovery)
5. Update `py/features/__init__.py` module exports

**Phase 3: Validation (2 hours)** - Complete
1. Manual testing with 2024 game data
2. Verify temporal safety (no data leakage)
3. Confirm database query performance (<100ms per feature)
4. Validate league-average fallbacks

**Phase 4: Integration (Planned - 8 hours)**
1. Retrain XGBoost v5 with 342-feature set
2. Backtest on 2024 holdout to isolate synthetic feature impact
3. Integrate into Ensemble v6.2
4. Monitor 2025 Weeks 15-18 performance

### Timeline

- **Research Start:** December 8, 2025
- **Implementation Complete:** December 9, 2025
- **Validation Complete:** December 9, 2025
- **XGBoost v5 Retraining:** Planned Week of December 9
- **Ensemble v6.2 Integration:** Planned December 16
- **2025 Holdout Validation:** Weeks 15-18 (December 2025)

### Success Criteria

- [x] All 37 features implemented and tested
- [x] Temporal safety verified (no data leakage)
- [x] Database queries perform <100ms per feature
- [x] Module documentation complete with research citations
- [ ] XGBoost v5 training completes successfully
- [ ] 2024 backtest shows +0.5% improvement minimum
- [ ] 2025 Weeks 15-18 validation confirms no overfitting
- [ ] Feature importance analysis identifies top contributors
- [ ] Dissertation Chapter 5 updated with new features

### Validation

**How We'll Know This Decision Was Correct:**

1. **Immediate Validation (2024 Backtest):**
   - Ensemble MAE improves by ≥0.5 yards on 2024 holdout
   - SHAP values show synthetic features in top 50 importance
   - No calibration degradation (maintain 90% coverage)

2. **Production Validation (2025 Weeks 15-18):**
   - Spread accuracy maintains 54%+ on new data
   - Props MAE ≤44.5 yards (no regression from 44.25 baseline)
   - Feature importance stable (no wild swings indicating overfitting)

3. **Research Validation:**
   - nfelo weighted EPA shows higher SHAP importance than unweighted
   - Rookie QB adjustment correlates with prediction errors
   - RZ efficiency differential predicts total over/under

4. **Long-Term (2026 Season):**
   - Features maintain predictive power across season transition
   - No feature drift detected (importance stable ±20%)
   - Competitive advantage sustained (market hasn't adapted)

---

## Related Decisions

- [ADR-009: Abandon Ensemble v7 Meta-Learning](./009-abandon-ensemble-v7-meta-learning.md) - Simplicity preference guides feature design (focused modules vs feature explosion)
- [ADR-004: BNN Research Abandonment](./004-bnn-research-abandonment.md) - Resource efficiency lesson applied (research-validated features vs exploratory approaches)
- [ADR-003: XGBoost v4 Ridge Residuals](./003-xgboost-v4-ridge-residuals.md) - Regularization approach prevents overfitting with expanded feature set

---

## References

### Documentation

- Synthetic Features Implementation Complete (no longer in the repo; see [HISTORY](../HISTORY.md)) - Detailed implementation log
- 2025 Holdout Validation (no longer in the repo; see [HISTORY](../HISTORY.md)) - Baseline performance for comparison
- Feature Engineering Pattern Analysis (no longer in the repo; see [HISTORY](../HISTORY.md)) - Feature design patterns

### Code

- `py/features/synthetic_interaction_features.py` (650 lines) - QB-OL synergy, defensive balance, explosive mismatch, coaching confidence, injury replacement
- `py/features/synthetic_momentum_features.py` (502 lines) - Form intensity, pressure consistency, RZ efficiency, momentum direction
- `py/features/research_driven_features.py` (800 lines) - Weighted net EPA, coaching rationality, rookie adjustment, career year regression, injury recovery, YAC over expected
- `py/features/__init__.py` (56 lines) - Module exports

### External Research Sources

1. **nfelo** - Offensive EPA weighting research
   - Finding: Offensive EPA 60% more predictive than defensive EPA (1.6:1.0 optimal ratio)
   - Implementation: `compute_weighted_net_epa()` with OFFENSIVE_EPA_WEIGHT = 1.6, DEFENSIVE_EPA_WEIGHT = 1.0
   - Source: nfelo.com methodology papers

2. **Stats Perform** - 4th Down Analytics
   - Finding: 4th downs convert 63.3% of the time, but teams only attempt 36.9%
   - Implementation: `compute_coaching_rationality_index()` with OPTIMAL_4TH_DOWN_GO_RATE = 0.45
   - Insight: Rational coaches following analytics are more predictable

3. **Tofte (Medium/Sabermetrics)** - Rookie QB Performance
   - Finding: Rookie QBs average 17.5 TDs, systematically underperform projections by 25%
   - Implementation: `compute_rookie_qb_adjustment()` with 0.75 factor for rookies, graduated to 1.0 over 3 years
   - Application: Adjusts passing yards expectations for rookie QBs

4. **Tofte (Medium/Sabermetrics)** - Career Year Regression
   - Finding: Players exceeding 130% of career average regress to ~80% of peak
   - Implementation: `compute_career_year_regression()` with 0.80 factor for >130% seasons
   - Insight: Prevents overestimating players coming off career years

5. **SumerSports** - Running Back Year-Over-Year Correlation
   - Finding: RB stats have near-zero YoY correlation (most volatile position)
   - Implementation: Downweight rushing features in props models (not direct feature)
   - Application: Informs feature weighting strategy

6. **nflfastR** - Expected YAC Models
   - Finding: YAC over expected measures receiver quality beyond scheme
   - Implementation: `compute_yac_over_expected()` using xyac_mean_yardage when available
   - Fallback: League average comparison when expected YAC unavailable

---

## Notes

### Lessons Learned

1. **Research-Driven > Exploratory:** 37 validated features > 100 automated interactions
   - External research provides theoretical grounding
   - Reduces overfitting risk (not data mining on our dataset)
   - Easier to explain and defend in dissertation

2. **Modular Design Aids Maintenance:**
   - Three focused modules (interaction, momentum, research) easier to maintain than monolithic
   - Clear separation of concerns (each module has specific purpose)
   - Can disable individual modules if features underperform

3. **Transaction Handling Critical:**
   - Database queries with missing tables require rollback handling
   - `compute_injury_recovery_factor` initially broke on transaction errors
   - Fixed with `conn.rollback()` after exception in try/except blocks

4. **Temporal Safety Non-Negotiable:**
   - All features verified to use `week < target_week` filtering
   - League-average fallbacks prevent lookahead bias
   - 4-week rolling windows provide stability without data leakage

5. **Documentation During Development:**
   - Comprehensive docstrings with formulas, sources, and hypotheses
   - Research citations embedded in code comments
   - Makes dissertation writing easier (Chapter 5 feature table update)

### When to Revisit

**Triggers for Removing/Modifying Features:**
- SHAP importance <0.01 for any feature after 2025 season (prune low-value features)
- Feature drift detected (importance change >50% season-over-season)
- Overfitting signal (2025 performance significantly worse than 2024 backtest)
- Database query performance >200ms for any feature (optimize or remove)

**Triggers for Adding More Synthetic Features:**
- New research published validating additional feature types
- SHAP analysis identifies feature gaps (high residual error in specific contexts)
- Competitive analysis shows market incorporating features we lack

**Next Review:** March 2026 (post-2025 season completion, pre-2026 training)

### Feature Importance Hypotheses

**High Importance Expected:**
- Weighted Net EPA (nfelo 1.6:1.0) - direct research validation
- RZ Efficiency Differential - high-leverage situations (7 vs 3 points)
- QB-OL Synergy - well-established interaction effect

**Medium Importance Expected:**
- Form Intensity - momentum indicators useful but noisy
- Coaching Rationality - predictability signal but slow-changing
- Rookie QB Adjustment - affects subset of games only

**Low Importance Expected:**
- Injury Recovery Factor - sparse data (depends on injury reporting)
- Career Year Regression - affects small subset of players
- YAC Over Expected - may be captured by existing receiving metrics

**Validation:** Post-XGBoost v5 training, run SHAP analysis to confirm/reject hypotheses. Document discrepancies for future research.

---

**Version:** 1.0
**Last Updated:** 2025-12-09
**Author:** Claude Code (Sonnet 4.5)
