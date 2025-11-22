# Parallel Modeling Acceleration - 6 Concurrent Tracks

**Created**: November 22, 2025
**Strategy**: Maximize productivity via parallel subagent execution
**Timeline**: 4-8 weeks to complete all 6 opportunities
**Expected Combined Impact**: +8-12 yards MAE improvement, +8-12% ROI increase

---

## Executive Summary

**User Decisions**:
1. ✅ YES - Implement Player Projection Aggregation immediately
2. ❌ NO - Skip market microstructure (no $500/month API)
3. ℹ️ Include RL 4th down in dissertation (not main contribution)
4. 🚀 Attack ALL opportunities in parallel

**Parallelization Strategy**:
- 6 independent tracks (different codebase areas, no conflicts)
- Each track assigned to specialized subagent
- Weekly sync points to integrate completions
- Aggressive 4-8 week timeline (vs 6+ months sequential)

---

## Track Architecture

| Track | Opportunity | Agent Type | Codebase Area | Timeline | Dependencies |
|-------|-------------|------------|---------------|----------|--------------|
| **A** | Player Projection Aggregation | Feature Engineering | `py/features/` | 1-2 weeks | None |
| **B** | Rushing Props Expansion | API Development | `py/api/`, `py/ensemble/` | 2-3 weeks | None |
| **C** | Causal Inference Phase 1 | Statistical Analysis | `py/causal/`, `py/features/` | 4-6 weeks | None |
| **D** | RL 4th Down Decisions | Research | `py/rl/` | 4-8 weeks | None |
| **E** | Dirichlet Bayesian Stacking | Ensemble Optimization | `py/ensemble/` | 4 weeks | None |
| **F** | Drive-Level Prediction | Data Pipeline | `py/pipeline/`, `py/models/` | 3-4 weeks | None |

**Key Insight**: Zero dependency conflicts! All tracks can run fully parallel.

---

## TRACK A: Player Projection Aggregation (WEEKS 1-2)

### Agent Assignment
**Type**: Feature Engineering Specialist
**Autonomy**: HIGH (clear requirements, low risk)

### Scope
1. Aggregate BNN Rushing projections to team-level features
2. Engineer 10-15 team features:
   - Total offensive EPA (sum of all players)
   - Star power (Gini coefficient of projections)
   - Depth quality (backup player performance)
   - Projection uncertainty (std across players)
3. Train XGBoost on aggregated features
4. Backtest on 2024 → 2025

### Success Criteria
- +2-4 yards MAE improvement
- No data leakage (validate split)
- Feature importance analysis complete

### Deliverables
- `py/features/player_aggregation_features.py` (200-300 lines)
- `scripts/train_xgboost_player_agg_v5.py` (validation script)
- `docs/milestones/PLAYER_AGGREGATION_COMPLETE.md` (results)

### Risk: LOW
- Simple aggregation, proven approach
- BNN Rushing already 89.5% calibrated

---

## TRACK B: Rushing Props Expansion (WEEKS 1-3)

### Agent Assignment
**Type**: API Development + ML Integration
**Autonomy**: MEDIUM (API design decisions)

### Scope
1. Retrain BNN for receiving yards (WR props)
2. Adapt XGBoost for touchdown prediction (binary)
3. Build props API endpoints:
   - `/predict/rushing_yards/{player_id}`
   - `/predict/receiving_yards/{player_id}`
   - `/predict/touchdown_prob/{player_id}`
4. Deploy alongside existing passing yards props

### Success Criteria
- Receiving yards: 85%+ calibration (match BNN Rushing)
- TD prediction: 75%+ accuracy
- API latency: <500ms per prediction

### Deliverables
- `py/models/bnn_receiving_v1.py` (retrained BNN)
- `py/models/xgboost_touchdowns_v1.py` (binary classifier)
- `py/api/props_endpoints.py` (3 new endpoints)
- `docs/milestones/RUSHING_RECEIVING_PROPS_COMPLETE.md`

### Risk: LOW-MEDIUM
- Infrastructure exists, data ready
- BNN retraining may require hyperparameter tuning

---

## TRACK C: Causal Inference Phase 1 (WEEKS 1-6)

### Agent Assignment
**Type**: Statistical Analysis + Causal Inference
**Autonomy**: MEDIUM-HIGH (complex methods, need validation)

### Scope (from ADVANCED_FEATURE_ENGINEERING_ROADMAP.md)
1. **DiD Rule Changes**:
   - 2018 roughing the passer (+3-5 yards MAE)
   - 2011 kickoff rule
   - 2020 COVID protocols
2. **PSM Travel Fatigue**:
   - Distance matrix (already built ✅)
   - Timezone crossing effects
   - Rest days impact
3. **PSM Weather Familiarity**:
   - Cold-weather teams at home
   - Dome teams outdoors
   - Extreme weather games

### Success Criteria
- +1-2% ROI improvement
- Causal assumptions validated (parallel trends, common support)
- ATT/ATE estimates with confidence intervals

### Deliverables
- `py/causal/did_rule_changes.py` (300-400 lines)
- `py/causal/psm_travel_fatigue.py` (400-500 lines)
- `py/causal/psm_weather_familiarity.py` (300-400 lines)
- `py/features/causal_features_v1.py` (integration)
- `docs/milestones/CAUSAL_INFERENCE_PHASE1_COMPLETE.md`

### Risk: MEDIUM
- Causal assumptions may not hold
- Requires statistical validation (placebo tests, robustness)

---

## TRACK D: RL 4th Down Decisions (WEEKS 1-8)

### Agent Assignment
**Type**: Research + Deep RL
**Autonomy**: HIGH (research exploration)

### Scope
1. Extract 4th down decisions from play-by-play (nflverse)
2. Define MDP:
   - State: (down, distance, field position, score diff, time remaining)
   - Action: {go, punt, field goal}
   - Reward: Δ win probability + points bonus
3. Train CQL agent (Conservative Q-Learning, offline RL)
4. Validate against NYT 4th down bot
5. **Dissertation Integration**: Write Chapter 7 section

### Success Criteria
- Match/exceed NYT 4th down bot recommendations
- Publishable result (novel application)
- Dissertation chapter draft (15-20 pages)

### Deliverables
- `py/rl/fourth_down_agent.py` (500-700 lines)
- `py/rl/fourth_down_mdp.py` (state/action/reward definitions)
- `scripts/train_fourth_down_cql.py` (training script)
- `analysis/dissertation/chapter_7_reinforcement_learning/section_4th_down.tex` (dissertation)
- `docs/milestones/RL_4TH_DOWN_COMPLETE.md`

### Risk: MEDIUM-HIGH
- Offline RL may require large dataset (10K+ decisions)
- Reward function design critical

---

## TRACK E: Dirichlet Bayesian Stacking (WEEKS 1-4)

### Agent Assignment
**Type**: Ensemble Optimization + Bayesian Methods
**Autonomy**: MEDIUM (clear algorithm, needs tuning)

### Scope (from BAYESIAN_ONLINE_LEARNING_V8_PROPOSAL.md)
1. Implement DirichletBayesianStacking class (300 lines)
2. Online weight update: α_i^(new) = α_i^(old) + L(y | ŷ_i)
3. Hyperparameter tuning (α_0, σ) via grid search
4. Multi-season prequential backtest (2022-2024)
5. Compare vs static weights (v6.0 baseline)

### Success Criteria
- +1.5-2.5 yards MAE improvement (33 → 30.5 yards)
- Weights adapt smoothly (no oscillation)
- Calibration maintained (90%+)

### Deliverables
- `py/ensemble/dirichlet_bayesian_stacking.py` (300 lines)
- `scripts/train_dirichlet_stacking_v8.py` (tuning script)
- `scripts/backtest_ensemble_v8_prequential.py` (validation)
- `docs/milestones/DIRICHLET_STACKING_V8_COMPLETE.md`

### Risk: LOW
- Proven algorithm (CDC FluSight winner)
- 80 lines reference implementation available

---

## TRACK F: Drive-Level Prediction (WEEKS 1-4)

### Agent Assignment
**Type**: Data Pipeline + Sequence Modeling
**Autonomy**: MEDIUM (LSTM design decisions)

### Scope
1. Aggregate play-by-play to drive level:
   - Yards per drive, points per drive, TD probability
2. Engineer drive-level features:
   - Starting field position
   - Down/distance sequence
   - Drive duration, play count
3. Train LSTM over drive sequences
4. Predict next drive outcome
5. Aggregate drives → full-game predictions

### Success Criteria
- Drive-level MAE: <15 yards per drive
- Full-game aggregation matches/beats current MAE
- In-game betting opportunities identified

### Deliverables
- `py/pipeline/drive_aggregation.py` (400-500 lines)
- `py/models/lstm_drive_prediction.py` (LSTM implementation)
- `scripts/train_lstm_drives_v1.py` (training script)
- `docs/milestones/DRIVE_LEVEL_PREDICTION_COMPLETE.md`

### Risk: MEDIUM
- LSTM may require large dataset (50K+ drives)
- Sequence length tuning critical

---

## Parallel Execution Schedule

### Week 1 (Nov 22-29)
**Kickoff All 6 Tracks Simultaneously**

| Track | Agent Action | Deliverable ETA |
|-------|-------------|-----------------|
| A | Feature engineering complete | Week 2 (Dec 6) |
| B | BNN Receiving training started | Week 3 (Dec 13) |
| C | DiD rule changes extraction | Week 6 (Jan 3) |
| D | 4th down MDP defined, data extracted | Week 8 (Jan 17) |
| E | Dirichlet implementation complete | Week 4 (Dec 20) |
| F | Drive aggregation pipeline complete | Week 4 (Dec 20) |

### Week 2 (Nov 29 - Dec 6)
- **Track A COMPLETE** ✅ Player aggregation deployed
- Track B: API endpoints developed
- Track C: PSM travel fatigue analysis
- Track D: CQL agent training started
- Track E: Hyperparameter tuning
- Track F: LSTM architecture selected

### Week 3 (Dec 6-13)
- **Track B COMPLETE** ✅ Rushing/receiving props deployed
- Track C: PSM weather familiarity
- Track D: Validation vs NYT bot
- Track E: Prequential backtest running
- Track F: Drive-level model training

### Week 4 (Dec 13-20)
- **Track E COMPLETE** ✅ Dirichlet stacking deployed
- **Track F COMPLETE** ✅ Drive-level predictions deployed
- Track C: Feature integration + XGBoost retraining
- Track D: Dissertation writing

### Week 6 (Jan 3, 2026)
- **Track C COMPLETE** ✅ Causal inference Phase 1 deployed

### Week 8 (Jan 17, 2026)
- **Track D COMPLETE** ✅ RL 4th down deployed + dissertation section

---

## Integration Strategy

### Weekly Sync Points (Every Friday 5 PM)
1. **Status Review**: Each track reports progress, blockers
2. **Conflict Resolution**: Merge conflicts, API changes
3. **Integration Planning**: Which completions to merge next week
4. **Risk Assessment**: Track any emerging issues

### Integration Priority (REVISED - PARALLEL EXECUTION)

**✅ APPROVED**: Parallel Groups strategy (4 weeks vs 8 weeks sequential)

**Week 3**: Merge A+C features → Train XGBoost v5
- Combine player aggregation (10-15 features) + causal features (4 features)
- Single training run with 14-19 total features
- Prevents conflict (both tracks modify XGBoost)

**Week 4**: Deploy v6.1 + Start Track E
- v6.1 = v6.0 + XGBoost v5 (static weights re-optimized)
- Track B: API v2.0 deployed (props endpoints)
- Track F: Parallel validation deployed (drive-level LSTM)
- Track E: Dirichlet initialized with v6.1 weights

**Week 6**: Deploy v7.0 (FINAL ENSEMBLE)
- Track E complete: Dirichlet stacking
- Track C complete: Causal features integrated
- v7.0 = Dynamic weights + XGBoost v5 + causal features

**Week 8**: Track D dissertation complete

### Ensemble Version Roadmap (REVISED)
- **v6.0** (Nov 15): PRODUCTION (XGBoost v4, static weights)
- **v6.1** (Week 4, Dec 20): v6.0 + XGBoost v5 (player agg + causal, static weights)
- **v7.0** (Week 6, Jan 3): v6.1 + Dirichlet stacking (dynamic weights) **← FINAL**
- **v8.0** (Jan 3): Alias for v7.0 (all 6 tracks integrated)

**Expected v7.0/v8.0 Performance**:
- MAE: 24-28 yards (25-35% improvement over v6.0's 32-35 yards)
- ROI: +14-18% (2.3x improvement over v6.0's +6-9%)
- **Timeline**: Jan 3, 2026 (4 weeks faster than original Feb 1 target)

---

## Subagent Coordination

### Agent Assignments (6 Specialized Subagents)

**Subagent 1: Feature Engineering Lead** (Track A)
- **Primary**: Player projection aggregation
- **Secondary**: Assist Track C with causal feature integration
- **Skills**: Pandas, NumPy, sklearn, feature engineering

**Subagent 2: API Development Lead** (Track B)
- **Primary**: Rushing/receiving props expansion
- **Secondary**: API documentation + deployment
- **Skills**: FastAPI, async, endpoint design, BNN retraining

**Subagent 3: Causal Inference Lead** (Track C)
- **Primary**: DiD, PSM, causal feature engineering
- **Secondary**: Statistical validation (placebo tests)
- **Skills**: DoWhy, EconML, causal inference, statistical testing

**Subagent 4: Deep RL Research Lead** (Track D)
- **Primary**: 4th down RL agent training
- **Secondary**: Dissertation writing (Chapter 7)
- **Skills**: PyTorch, CQL, offline RL, academic writing

**Subagent 5: Ensemble Optimization Lead** (Track E)
- **Primary**: Dirichlet Bayesian stacking implementation
- **Secondary**: Ensemble v7.0-v8.0 integration
- **Skills**: Bayesian methods, ensemble learning, cvxpy

**Subagent 6: Sequence Modeling Lead** (Track F)
- **Primary**: Drive-level LSTM prediction
- **Secondary**: Data pipeline optimization
- **Skills**: PyTorch, LSTM/RNN, sequence modeling, data engineering

### Communication Protocol
- **Shared Repository**: All agents commit to feature branches (`track-a`, `track-b`, etc.)
- **Daily Standups**: Automated status posts (what done, what next, blockers)
- **Integration PRs**: Each track creates PR when ready to merge
- **Code Review**: Lead agent (you) reviews + approves integration

---

## Risk Mitigation

### Parallel Execution Risks
1. **Merge Conflicts**: Weekly sync prevents drift
2. **API Breaking Changes**: Versioned APIs (v1, v2)
3. **Resource Contention**: GPU allocation (Track D, F get priority)
4. **Integration Failures**: Staged rollout (v6.1 → v7.0 → v7.5 → v8.0)

### Technical Debt Management
- Each track includes documentation requirement
- No "quick hacks" - production-quality code only
- Test coverage: 80%+ per track
- Automated validation: pre-commit hooks enforced

---

## Success Metrics

### By Week 8 (Jan 17, 2026)
- ✅ 6/6 tracks complete
- ✅ Ensemble v8.0 deployed
- ✅ MAE: 24-28 yards (vs 32-35 baseline)
- ✅ ROI: +14-18% (vs +6-9% baseline)
- ✅ Props expanded: passing, rushing, receiving, TDs
- ✅ Dissertation chapter: RL 4th down section complete
- ✅ Technical debt: Zero (all code production-ready)

### ROI Breakdown (v6.0 → v8.0)
| Improvement | Source | Expected MAE Gain | Expected ROI Gain |
|-------------|--------|-------------------|-------------------|
| Player Aggregation | Track A | +2-4 yards | +1-2% |
| Rushing Props | Track B | N/A (new market) | +5-10% |
| Causal Features | Track C | +1-2 yards | +1-2% |
| RL 4th Down | Track D | N/A (research) | Dissertation value |
| Dirichlet Stacking | Track E | +1.5-2.5 yards | +1-2% |
| Drive-Level | Track F | +2-3 yards | +1-2% |
| **TOTAL** | **All Tracks** | **+8-12 yards** | **+9-18% ROI** |

---

## Next Steps (Immediate)

1. **NOW**: Spawn 6 subagents (one per track)
2. **Week 1**: All agents begin parallel development
3. **Friday Dec 6**: First integration (Track A complete)
4. **Friday Dec 13**: Second integration (Track B complete)
5. **Friday Dec 20**: Third integration (Tracks E, F complete)
6. **Friday Jan 3**: Fourth integration (Track C complete)
7. **Friday Jan 17**: Final integration (Track D complete, v8.0 deployed)

---

## Questions for User Approval

1. **Approve 6-track parallel execution?** (All tracks start Week 1)
2. **Approve integration schedule?** (Weekly syncs, staged rollout)
3. **Approve timeline?** (8 weeks to complete all 6)
4. **Any track priority adjustments?** (e.g., Track A must finish before Track E)

**Ready to spawn subagents and begin parallel execution?** 🚀
