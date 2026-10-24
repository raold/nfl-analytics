"""
Feedback Loop System for NFL Analytics.

This module implements a weekly feedback loop that:
1. Updates outcomes after games complete
2. Tracks semantic feature attribution
3. Adjusts ensemble weights based on performance
4. Triggers retraining when accuracy degrades
5. Generates annotations for LLM fine-tuning
6. Provides meta-learning for bidirectional improvement
7. WHY System for prediction explainability (Dec 28, 2025)

Components:
-----------
Phase 6: Weekly Feedback Loop
    - WeeklyOutcomeUpdater: Update predictions with actual outcomes
    - SemanticAttributionTracker: Track feature contribution (True SHAP)
    - EnsembleWeightAdjuster: Adjust weights based on performance
    - RetrainTrigger: Detect when retraining is needed
    - AnnotationGenerator: Generate LLM training data
    - WeeklyFeedbackOrchestrator: Orchestrate all components

Phase 7: Meta-Learning System
    - SemanticMetaLearner: Bidirectional learning (Model↔Semantic)
    - FeaturePerformanceMemory: Rolling feature performance tracking
    - AutoFeatureSelector: Automatic feature enable/disable
    - PromptEvolver: Evolve LLM prompts based on outcomes

Phase 8: A/B Testing
    - SemanticABTest: Compare semantic vs non-semantic predictions

Phase 9: WHY System (Dec 28, 2025)
    - ModelErrorDecomposer: Track which ensemble component caused errors
    - TemporalCausalityTracker: Trace News→Sentiment→Feature→Prediction→Outcome
    - FeatureInteractionAnalyzer: Detect semantic × base feature synergies

Usage:
------
Weekly workflow (Tuesday after games complete):

    from nfl_analytics.feedback import WeeklyFeedbackOrchestrator

    orchestrator = WeeklyFeedbackOrchestrator()
    result = orchestrator.run(season=2025, week=16)

    if result.triggers_fired:
        print(f"Retraining recommended: {result.triggers_fired}")

A/B Testing:

    from nfl_analytics.feedback import SemanticABTest

    test = SemanticABTest()
    test_id = test.setup_test(season=2025, start_week=1)
    results = test.analyze_results(test_id)
    test.print_results(results)

Meta-Learning:

    from nfl_analytics.feedback import SemanticMetaLearner

    learner = SemanticMetaLearner()
    result = learner.update_from_outcomes(season=2025, week=16)

    for insight in result.feature_insights:
        print(f"{insight.feature_name}: {insight.recommendation}")
"""

# Phase 6: Weekly Feedback Loop
from nfl_analytics.feedback.outcome_updater import WeeklyOutcomeUpdater
from nfl_analytics.feedback.semantic_attribution import SemanticAttributionTracker
from nfl_analytics.feedback.ensemble_weight_adjuster import EnsembleWeightAdjuster
from nfl_analytics.feedback.retrain_trigger import RetrainTrigger
from nfl_analytics.feedback.annotation_generator import AnnotationGenerator

# Phase 7: Meta-Learning
from nfl_analytics.feedback.meta_learner import SemanticMetaLearner
from nfl_analytics.feedback.feature_memory import FeaturePerformanceMemory
from nfl_analytics.feedback.auto_feature_selector import AutoFeatureSelector
from nfl_analytics.feedback.prompt_evolver import PromptEvolver

# Phase 8: A/B Testing
from nfl_analytics.feedback.semantic_ab_test import SemanticABTest

# Phase 9: WHY System (Dec 28, 2025)
from nfl_analytics.feedback.model_error_decomposition import ModelErrorDecomposer
from nfl_analytics.feedback.temporal_causality import TemporalCausalityTracker
from nfl_analytics.feedback.feature_interactions import FeatureInteractionAnalyzer

__all__ = [
    # Phase 6
    "WeeklyOutcomeUpdater",
    "SemanticAttributionTracker",
    "EnsembleWeightAdjuster",
    "RetrainTrigger",
    "AnnotationGenerator",
    # Phase 7
    "SemanticMetaLearner",
    "FeaturePerformanceMemory",
    "AutoFeatureSelector",
    "PromptEvolver",
    # Phase 8
    "SemanticABTest",
    # Phase 9: WHY System
    "ModelErrorDecomposer",
    "TemporalCausalityTracker",
    "FeatureInteractionAnalyzer",
]
