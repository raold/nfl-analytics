# ADR-012: Weekly Feedback Loop System

**Date:** December 22, 2025
**Status:** Accepted
**Context:** Semantic Stack Phase 1 Complete, 2025 Holdout Testing Ongoing

## Decision

Implement a comprehensive weekly feedback loop system that enables bidirectional learning between prediction models and semantic features, with automatic feature selection based on performance.

## Context

With the semantic stack (Phase 1) complete and 26 LLM-derived features integrated into the GameFeatureExtractor, we need a systematic way to:

1. **Validate semantic feature contribution** - Do LLM features actually improve predictions?
2. **Close the learning loop** - Use prediction outcomes to improve LLM prompts and features
3. **Automate feature management** - Enable/disable features based on performance
4. **Generate training data** - Convert outcomes into annotations for LLM fine-tuning
5. **Detect model degradation** - Trigger retraining when accuracy drops

## Alternatives Considered

### Option A: Manual Weekly Review
- **Approach:** Human reviews prediction accuracy weekly, adjusts features manually
- **Pros:** High control, human judgment on edge cases
- **Cons:** Not scalable, prone to bias, delays, no systematic learning
- **Rejected:** Too slow and subjective for production system

### Option B: Continuous Online Learning
- **Approach:** Update weights after every game in real-time
- **Pros:** Immediate adaptation, always current
- **Cons:** Noisy updates, risk of overfitting to recent games, complex infrastructure
- **Rejected:** NFL has 16 games/week, not enough signal for real-time updates (see ADR-009)

### Option C: Weekly Batch Feedback Loop (Selected)
- **Approach:** Weekly batch processing after all games complete
- **Pros:** Sufficient data for stable updates, aligned with NFL schedule, simpler infrastructure
- **Cons:** 7-day delay in adaptation
- **Selected:** Best balance of learning signal and stability

## Implementation

### Architecture (3 Phases, 10 Modules)

```
Phase 6: Weekly Feedback Loop
├── 6.1 WeeklyOutcomeUpdater     - Populate actuals after games
├── 6.2 SemanticAttributionTracker - Track feature contribution
├── 6.3 EnsembleWeightAdjuster   - Adjust weights on performance
├── 6.4 RetrainTrigger           - Detect degradation
└── 6.5 AnnotationGenerator      - Generate LLM training data

Phase 7: Meta-Learning
├── 7.1 SemanticMetaLearner      - Bidirectional Model↔Semantic
├── 7.2 FeaturePerformanceMemory - Rolling contribution tracking
├── 7.3 AutoFeatureSelector      - Automatic enable/disable
└── 7.4 PromptEvolver            - Evolve LLM prompts

Phase 8: A/B Testing
└── 8.1 SemanticABTest           - McNemar test for significance
```

### Key Design Decisions

1. **Feature States:** Three-state lifecycle (probation → active → disabled)
   - New features start in probation (weight=0.5)
   - Promoted to active after 2+ weeks of positive contribution
   - Demoted to disabled after 4+ weeks of negative contribution

2. **Statistical Testing:** McNemar test for paired binary outcomes
   - More powerful than chi-square for matched pairs
   - Accounts for correlation in predictions on same games

3. **Weight Adjustment:** Inverse-MAE weighting with shrinkage
   - Regularization toward static weights prevents overreaction
   - Minimum 20 games required before weight changes

4. **Retraining Triggers:** Three thresholds
   - MAE degradation: +0.5 points from baseline
   - Accuracy drop: -5% from baseline
   - Calibration drift: +0.1 Brier score

5. **LLM Prompt Evolution:** A/B test prompts before deployment
   - Track games_tested, accuracy_rate, contribution_score
   - Require statistical significance before switching

### Database Schema (8 Tables)

```sql
-- Core tracking
ml.feedback_runs         -- Weekly orchestration runs
ml.semantic_attribution  -- Per-game feature attribution
ml.feature_performance   -- Rolling contribution tracking
ml.feature_selection_state -- Feature status (active/disabled/probation)

-- Prompt management
ml.prompt_versions       -- LLM prompt versioning
ml.prompt_ab_tests       -- Prompt A/B test configs

-- A/B testing
ml.ab_test_status        -- Test metadata and state
ml.ab_test_results       -- Per-game control vs treatment
```

### Usage

```bash
# Weekly workflow (Tuesday after games complete)
python scripts/run_weekly_feedback.py --season 2025 --week 16

# A/B test semantic features on 2025 holdout
python scripts/ab_test_semantic_2025.py --weeks 1-16 --dry-run
```

## Consequences

### Positive

- **Systematic validation:** Every semantic feature is measured weekly
- **Automatic pruning:** Poor features disabled without manual intervention
- **LLM improvement:** Outcomes converted to training data for fine-tuning
- **Early detection:** Retraining triggered before significant degradation
- **Transparent tracking:** All decisions logged in database

### Negative

- **7-day delay:** Adaptation lags by one week
- **Cold start:** New features need 2+ weeks to be promoted
- **Complexity:** 10 modules and 8 tables to maintain
- **Threshold tuning:** Trigger thresholds may need adjustment

### Neutral

- **Database size:** Attribution tracking adds ~500 rows/week
- **Runtime:** Weekly feedback loop takes 2-5 minutes

## Relation to ADR-009

ADR-009 rejected **dynamic meta-learning** that adjusted weights game-by-game in real-time. This ADR implements **weekly batch feedback** which is fundamentally different:

| Aspect | ADR-009 (Rejected) | ADR-012 (Accepted) |
|--------|-------------------|-------------------|
| Frequency | Per-game | Per-week |
| Data | 1 game | 16 games |
| Weights | Dynamic | Static (adjustable) |
| Validation | None | Statistical tests |
| Learning | Real-time | Batch |

The key insight from ADR-009 was that single games are too noisy. Weekly batches (16 games) provide sufficient signal.

## Validation

The system will be validated using:

1. **2025 Holdout A/B Test:** Compare semantic vs non-semantic predictions
2. **Feature Attribution Analysis:** Track which features improve accuracy
3. **Prompt Evolution A/B Tests:** Measure prompt improvement over time
4. **Retraining Trigger Accuracy:** Verify triggers fire when appropriate

## References

- ADR-009: Abandon Ensemble v7 Meta-Learning
- ADR-011: XGBoost Feature Optimization
- `nfl_analytics/feedback/__init__.py`: Module exports
- `scripts/run_weekly_feedback.py`: Weekly orchestrator
- `scripts/ab_test_semantic_2025.py`: 2025 holdout comparison
