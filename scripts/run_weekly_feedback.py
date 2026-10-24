#!/usr/bin/env python3
"""
Weekly Feedback Loop Orchestrator.

Single CLI to run the complete weekly feedback loop after games complete.

Usage:
    python scripts/run_weekly_feedback.py --season 2025 --week 16
    python scripts/run_weekly_feedback.py --season 2025 --week 16 --dry-run
    python scripts/run_weekly_feedback.py --season 2025 --week 16 --auto-adjust

Workflow:
    1. Update outcomes (actual scores → ml.game_predictions)
    2. Compute semantic attribution (feature importance)
    3. Check retrain triggers (performance degradation)
    4. Optionally adjust weights (--auto-adjust)
    5. Generate annotations (LLM fine-tuning data)
    6. Log summary to ml.feedback_runs

Exit Codes:
    0: Success
    1: Partial failure (some steps failed)
    2: Critical failure
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2

from nfl_analytics.feedback.outcome_updater import WeeklyOutcomeUpdater
from nfl_analytics.feedback.semantic_attribution import SemanticAttributionTracker
from nfl_analytics.feedback.ensemble_weight_adjuster import EnsembleWeightAdjuster
from nfl_analytics.feedback.retrain_trigger import RetrainTrigger
from nfl_analytics.feedback.annotation_generator import AnnotationGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FeedbackLoopResult:
    """Complete result of feedback loop execution."""
    season: int
    week: int
    run_timestamp: datetime
    duration_seconds: float = 0.0
    dry_run: bool = False

    # Step results
    games_updated: int = 0
    attributions_computed: int = 0
    triggers_fired: list[str] = field(default_factory=list)
    weights_adjusted: bool = False
    annotations_generated: int = 0

    # Details
    outcome_result: Any = None
    attribution_result: Any = None
    trigger_result: Any = None
    weight_result: Any = None
    annotation_stats: Any = None

    # Errors
    errors: list[str] = field(default_factory=list)
    step_status: dict[str, str] = field(default_factory=dict)


class WeeklyFeedbackOrchestrator:
    """
    Orchestrate all weekly feedback components.

    Steps:
    1. Update outcomes (WeeklyOutcomeUpdater)
    2. Compute semantic attribution (SemanticAttributionTracker)
    3. Check retrain triggers (RetrainTrigger)
    4. Optionally adjust weights (EnsembleWeightAdjuster)
    5. Generate annotations (AnnotationGenerator)
    6. Log summary to ml.feedback_runs
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5544,
        database: str = "devdb01",
        user: str = "dro",
        password: str = "sicillionbillions",
    ):
        """Initialize with database connection parameters."""
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }

        # Initialize components
        self.outcome_updater = WeeklyOutcomeUpdater(**self.conn_params)
        self.attribution_tracker = SemanticAttributionTracker(**self.conn_params)
        self.weight_adjuster = EnsembleWeightAdjuster(**self.conn_params)
        self.retrain_trigger = RetrainTrigger(**self.conn_params)
        self.annotation_generator = AnnotationGenerator(**self.conn_params)

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.conn_params)

    def run(
        self,
        season: int,
        week: int,
        dry_run: bool = False,
        auto_adjust: bool = False,
        generate_annotations: bool = True,
        annotation_dir: Path | None = None
    ) -> FeedbackLoopResult:
        """
        Run the complete weekly feedback loop.

        Args:
            season: Season year
            week: Week number
            dry_run: If True, don't persist changes
            auto_adjust: If True, apply weight adjustments automatically
            generate_annotations: If True, generate LLM training data
            annotation_dir: Directory for annotations (default: data/semantic_training)

        Returns:
            FeedbackLoopResult with complete execution summary
        """
        start_time = time.time()
        result = FeedbackLoopResult(
            season=season,
            week=week,
            run_timestamp=datetime.now(),
            dry_run=dry_run
        )

        logger.info(f"Starting feedback loop for {season} Week {week}")
        if dry_run:
            logger.info("DRY RUN MODE - no changes will be persisted")

        # Step 1: Update outcomes
        logger.info("Step 1/5: Updating outcomes...")
        try:
            outcome_result = self.outcome_updater.update_week(
                season=season,
                week=week,
                dry_run=dry_run
            )
            result.outcome_result = outcome_result
            result.games_updated = outcome_result.predictions_updated
            result.step_status["outcomes"] = "success"
            logger.info(f"  Updated {outcome_result.predictions_updated} predictions")
        except Exception as e:
            logger.error(f"  Outcome update failed: {e}")
            result.errors.append(f"Outcomes: {e}")
            result.step_status["outcomes"] = "failed"

        # Step 2: Compute semantic attribution
        logger.info("Step 2/5: Computing semantic attribution...")
        try:
            attribution_result = self.attribution_tracker.compute_week_attribution(
                season=season,
                week=week,
                save_to_db=not dry_run
            )
            result.attribution_result = attribution_result
            result.attributions_computed = attribution_result.games_with_semantic
            result.step_status["attribution"] = "success"
            logger.info(f"  Computed attribution for {attribution_result.games_with_semantic} games")

            if attribution_result.top_features:
                top = list(attribution_result.top_features.items())[:3]
                for name, stats in top:
                    logger.info(f"    Top feature: {name} (mean={stats['mean']:.4f})")
        except Exception as e:
            logger.error(f"  Attribution failed: {e}")
            result.errors.append(f"Attribution: {e}")
            result.step_status["attribution"] = "failed"

        # Step 3: Check retrain triggers
        logger.info("Step 3/5: Checking retrain triggers...")
        try:
            trigger_result = self.retrain_trigger.check_triggers(
                model_id="ensemble",
                season=season,
                week=week
            )
            result.trigger_result = trigger_result
            result.triggers_fired = trigger_result.triggers_fired
            result.step_status["triggers"] = "success"

            if trigger_result.triggers_fired:
                logger.warning(f"  TRIGGERS FIRED: {trigger_result.triggers_fired}")
                if not dry_run:
                    self.retrain_trigger.log_trigger_event(trigger_result)
            else:
                logger.info("  No triggers fired - performance within bounds")
        except Exception as e:
            logger.error(f"  Trigger check failed: {e}")
            result.errors.append(f"Triggers: {e}")
            result.step_status["triggers"] = "failed"

        # Step 4: Weight adjustment (optional)
        logger.info("Step 4/5: Evaluating weight adjustments...")
        try:
            weight_result = self.weight_adjuster.compute_optimal_weights(
                season=season,
                end_week=week,
                lookback_weeks=4
            )
            result.weight_result = weight_result

            if weight_result.should_adjust:
                logger.info(f"  Weight adjustment recommended: {weight_result.reason}")
                if auto_adjust and not dry_run:
                    version = self.weight_adjuster.apply_weights(
                        weight_result.suggested_weights
                    )
                    result.weights_adjusted = True
                    logger.info(f"  Applied weights version {version}")
                else:
                    logger.info("  Adjustment not applied (use --auto-adjust)")
            else:
                logger.info(f"  No adjustment needed: {weight_result.reason}")

            result.step_status["weights"] = "success"
        except Exception as e:
            logger.error(f"  Weight evaluation failed: {e}")
            result.errors.append(f"Weights: {e}")
            result.step_status["weights"] = "failed"

        # Step 5: Generate annotations
        if generate_annotations:
            logger.info("Step 5/5: Generating annotations...")
            try:
                output_dir = annotation_dir or Path("data/semantic_training")
                annotation_stats = self.annotation_generator.export_training_batch(
                    output_dir=output_dir,
                    season=season,
                    weeks=[week]
                )
                result.annotation_stats = annotation_stats
                result.annotations_generated = (
                    annotation_stats.sentiment_annotations +
                    annotation_stats.ner_annotations
                )
                result.step_status["annotations"] = "success"
                logger.info(f"  Generated {result.annotations_generated} annotations")
            except Exception as e:
                logger.error(f"  Annotation generation failed: {e}")
                result.errors.append(f"Annotations: {e}")
                result.step_status["annotations"] = "failed"
        else:
            result.step_status["annotations"] = "skipped"

        # Record run
        result.duration_seconds = time.time() - start_time

        if not dry_run:
            self._log_feedback_run(result)

        return result

    def _log_feedback_run(self, result: FeedbackLoopResult) -> None:
        """Log feedback run to database."""
        query = """
        INSERT INTO ml.feedback_runs
            (season, week, run_timestamp, games_updated, games_with_outcomes,
             top_semantic_features, triggers_fired, weight_adjustments,
             weights_applied, sentiment_annotations, ner_annotations,
             dry_run, run_duration_seconds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Prepare JSON fields
        top_features = {}
        if result.attribution_result and result.attribution_result.top_features:
            top_features = {
                k: v["mean"] for k, v in result.attribution_result.top_features.items()
            }

        weight_adj = {}
        if result.weight_result:
            weight_adj = {
                "current": result.weight_result.current_weights,
                "suggested": result.weight_result.suggested_weights,
                "should_adjust": result.weight_result.should_adjust,
            }

        sent_annotations = 0
        ner_annotations = 0
        if result.annotation_stats:
            sent_annotations = result.annotation_stats.sentiment_annotations
            ner_annotations = result.annotation_stats.ner_annotations

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    result.season,
                    result.week,
                    result.run_timestamp,
                    result.games_updated,
                    result.attributions_computed,
                    json.dumps(top_features),
                    result.triggers_fired,
                    json.dumps(weight_adj),
                    result.weights_adjusted,
                    sent_annotations,
                    ner_annotations,
                    result.dry_run,
                    result.duration_seconds
                ))
            conn.commit()

        logger.info("Logged feedback run to ml.feedback_runs")


def print_summary(result: FeedbackLoopResult) -> None:
    """Print formatted summary of feedback loop execution."""
    print("\n" + "=" * 60)
    print("FEEDBACK LOOP SUMMARY")
    print("=" * 60)
    print(f"Season: {result.season}  Week: {result.week}")
    print(f"Timestamp: {result.run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    if result.dry_run:
        print("Mode: DRY RUN")

    print("\n" + "-" * 60)
    print("STEP RESULTS")
    print("-" * 60)
    for step, status in result.step_status.items():
        icon = "✓" if status == "success" else "✗" if status == "failed" else "-"
        print(f"  [{icon}] {step}: {status}")

    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    print(f"  Games updated:        {result.games_updated}")
    print(f"  Attributions:         {result.attributions_computed}")
    print(f"  Annotations:          {result.annotations_generated}")
    print(f"  Weights adjusted:     {result.weights_adjusted}")

    if result.triggers_fired:
        print(f"\n  ⚠️  TRIGGERS FIRED: {result.triggers_fired}")

    if result.errors:
        print("\n  ERRORS:")
        for err in result.errors:
            print(f"    - {err}")

    print("=" * 60 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run weekly feedback loop for NFL prediction models"
    )
    parser.add_argument(
        "--season", type=int, required=True,
        help="Season year (e.g., 2025)"
    )
    parser.add_argument(
        "--week", type=int, required=True,
        help="Week number (1-18)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without persisting changes"
    )
    parser.add_argument(
        "--auto-adjust", action="store_true",
        help="Automatically apply weight adjustments if recommended"
    )
    parser.add_argument(
        "--skip-annotations", action="store_true",
        help="Skip annotation generation"
    )
    parser.add_argument(
        "--annotation-dir", type=str, default=None,
        help="Directory for annotations (default: data/semantic_training)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("NFL WEEKLY FEEDBACK LOOP")
    print("=" * 60)
    print(f"Season: {args.season}  Week: {args.week}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Auto-adjust: {args.auto_adjust}")
    print("=" * 60 + "\n")

    try:
        orchestrator = WeeklyFeedbackOrchestrator()

        annotation_dir = Path(args.annotation_dir) if args.annotation_dir else None

        result = orchestrator.run(
            season=args.season,
            week=args.week,
            dry_run=args.dry_run,
            auto_adjust=args.auto_adjust,
            generate_annotations=not args.skip_annotations,
            annotation_dir=annotation_dir
        )

        print_summary(result)

        # Exit code based on results
        if result.errors:
            return 1
        return 0

    except Exception as e:
        logger.error(f"Critical failure: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
