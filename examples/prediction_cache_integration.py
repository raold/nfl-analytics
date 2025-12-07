"""
Prediction Cache Manager Integration Examples

This module demonstrates common integration patterns for the Prediction Cache Manager.

Examples:
1. Ensemble integration with cache-first lookup
2. Model training workflow with automatic caching
3. Batch prediction with bulk caching
4. Cache maintenance and monitoring
5. Migration from rpy2 to PostgreSQL

Author: Claude Code (Sonnet 4.5)
Created: November 22, 2025
"""

import logging

import numpy as np
import pandas as pd

from py.utils.prediction_cache_manager import PredictionCacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Ensemble Integration with Cache-First Lookup
# =============================================================================


class CachedEnsemble:
    """
    Ensemble predictor with PostgreSQL cache for R model predictions.

    Pattern: Try cache first (fast), fall back to rpy2 (slow), cache result.
    """

    def __init__(self):
        self.cache = PredictionCacheManager()
        # Initialize R interface (slow)
        # self.r_interface = RModelInterface()

    def predict_passing_yards(
        self, player_id: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Get ensemble prediction with cached R model components.

        Returns:
            Dict with model predictions and ensemble average
        """
        predictions = {}

        # Python models (always fast)
        predictions["xgboost"] = self._predict_xgboost(player_id, season, week)

        # R models (cache-first to avoid slow rpy2 calls)
        predictions["state_space"] = self._predict_state_space_cached(
            player_id, season, week
        )
        predictions["hierarchical"] = self._predict_hierarchical_cached(
            player_id, season, week
        )
        predictions["informative_priors"] = self._predict_informative_priors_cached(
            player_id, season, week
        )

        # Ensemble average
        predictions["ensemble"] = np.mean(list(predictions.values()))

        return predictions

    def _predict_state_space_cached(
        self, player_id: str, season: int, week: int
    ) -> float:
        """Get state-space prediction with cache lookup."""
        # Try cache first (<1ms)
        cached = self.cache.get_cached_prediction(
            player_id=player_id,
            stat_type="passing_yards",
            season=season,
            week=week,
            model_name="state_space_v2",
            max_age_hours=24,  # Only fresh predictions
        )

        if cached is not None:
            logger.debug(f"Cache HIT for {player_id} state_space_v2")
            return cached

        # Cache miss - compute via rpy2 (slow 50-200ms)
        logger.debug(f"Cache MISS for {player_id} state_space_v2 - computing")
        prediction = self._compute_state_space_prediction(player_id, season, week)

        # Cache for next time
        self._cache_single_prediction(
            player_id, season, week, prediction, "state_space_v2"
        )

        return prediction

    def _predict_hierarchical_cached(
        self, player_id: str, season: int, week: int
    ) -> float:
        """Get hierarchical prediction with cache lookup."""
        cached = self.cache.get_cached_prediction(
            player_id=player_id,
            stat_type="passing_yards",
            season=season,
            week=week,
            model_name="hierarchical_v2",
            max_age_hours=24,
        )

        if cached is not None:
            return cached

        prediction = self._compute_hierarchical_prediction(player_id, season, week)
        self._cache_single_prediction(
            player_id, season, week, prediction, "hierarchical_v2"
        )
        return prediction

    def _predict_informative_priors_cached(
        self, player_id: str, season: int, week: int
    ) -> float:
        """Get informative priors prediction with cache lookup."""
        cached = self.cache.get_cached_prediction(
            player_id=player_id,
            stat_type="passing_yards",
            season=season,
            week=week,
            model_name="informative_priors_v2",
            max_age_hours=24,
        )

        if cached is not None:
            return cached

        prediction = self._compute_informative_priors_prediction(
            player_id, season, week
        )
        self._cache_single_prediction(
            player_id, season, week, prediction, "informative_priors_v2"
        )
        return prediction

    def _cache_single_prediction(
        self, player_id: str, season: int, week: int, prediction: float, model_name: str
    ):
        """Cache a single prediction."""
        df = pd.DataFrame(
            [
                {
                    "player_id": player_id,
                    "season": season,
                    "week": week,
                    "prediction": prediction,
                }
            ]
        )

        self.cache.cache_predictions(
            df, model_name=model_name, stat_type="passing_yards", validate=True
        )

    def _predict_xgboost(self, player_id: str, season: int, week: int) -> float:
        """Placeholder for XGBoost prediction."""
        return 250.0  # Example

    def _compute_state_space_prediction(
        self, player_id: str, season: int, week: int
    ) -> float:
        """Placeholder for rpy2 state-space prediction."""
        # In production: self.r_interface.predict_state_space(...)
        return 245.0  # Example

    def _compute_hierarchical_prediction(
        self, player_id: str, season: int, week: int
    ) -> float:
        """Placeholder for rpy2 hierarchical prediction."""
        return 248.0  # Example

    def _compute_informative_priors_prediction(
        self, player_id: str, season: int, week: int
    ) -> float:
        """Placeholder for rpy2 informative priors prediction."""
        return 252.0  # Example


# =============================================================================
# Example 2: Model Training Workflow with Automatic Caching
# =============================================================================


def train_and_cache_model(
    model_name: str, stat_type: str, training_seasons: list[int]
) -> int:
    """
    Train model and automatically cache all predictions.

    Args:
        model_name: Name of model to train
        stat_type: Type of stat to predict
        training_seasons: Seasons to generate predictions for

    Returns:
        Number of predictions cached
    """
    logger.info(f"Training {model_name} for {stat_type}")

    # Initialize cache manager
    cache = PredictionCacheManager()

    # Step 1: Invalidate old predictions
    logger.info("Invalidating old predictions")
    cache.invalidate_cache(model_name=model_name, stat_type=stat_type)

    # Step 2: Train model (placeholder)
    logger.info("Training model")
    # model = train_r_model(model_name, stat_type, training_seasons)

    # Step 3: Generate predictions for all seasons/weeks
    all_predictions = []

    for season in training_seasons:
        logger.info(f"Generating predictions for season {season}")

        # Generate predictions for all weeks (placeholder)
        season_predictions = generate_season_predictions(season, stat_type)
        all_predictions.append(season_predictions)

    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Step 4: Cache predictions in bulk
    logger.info(f"Caching {len(predictions_df)} predictions")
    cached_count = cache.cache_predictions(
        predictions_df, model_name=model_name, stat_type=stat_type, validate=True
    )

    # Step 5: Validate cache
    logger.info("Validating cache integrity")
    report = cache.validate_cache_integrity(sample_size=1000)

    if not report.is_valid():
        logger.error(f"Cache validation failed:\n{report.summary()}")
        raise ValueError("Cache validation failed")

    logger.info(f"✅ Successfully trained and cached {cached_count} predictions")
    return cached_count


def generate_season_predictions(season: int, stat_type: str) -> pd.DataFrame:
    """Placeholder for generating season predictions."""
    # In production: call R model and return predictions
    num_predictions = 100  # Example
    return pd.DataFrame(
        {
            "player_id": [f"P{i:04d}" for i in range(num_predictions)],
            "season": [season] * num_predictions,
            "week": [1] * num_predictions,
            "prediction": np.random.uniform(100, 400, num_predictions),
        }
    )


# =============================================================================
# Example 3: Batch Prediction with Bulk Caching
# =============================================================================


def predict_week_batch(
    season: int, week: int, player_ids: list[str], model_name: str
) -> pd.DataFrame:
    """
    Generate predictions for all players in a week with bulk caching.

    Args:
        season: Season year
        week: Week number
        player_ids: List of player IDs to predict
        model_name: Name of model to use

    Returns:
        DataFrame with predictions
    """
    cache = PredictionCacheManager()

    # Check cache for existing predictions
    cached_predictions = []
    missing_player_ids = []

    for player_id in player_ids:
        cached = cache.get_cached_prediction(
            player_id=player_id,
            stat_type="passing_yards",
            season=season,
            week=week,
            model_name=model_name,
            max_age_hours=24,
        )

        if cached is not None:
            cached_predictions.append(
                {
                    "player_id": player_id,
                    "season": season,
                    "week": week,
                    "prediction": cached,
                    "source": "cache",
                }
            )
        else:
            missing_player_ids.append(player_id)

    logger.info(
        f"Cache hit rate: {len(cached_predictions)}/{len(player_ids)} "
        f"({len(cached_predictions)/len(player_ids)*100:.1f}%)"
    )

    # Compute missing predictions in batch
    if missing_player_ids:
        logger.info(f"Computing {len(missing_player_ids)} missing predictions")
        new_predictions = batch_compute_predictions(
            missing_player_ids, season, week, model_name
        )

        # Bulk cache new predictions
        cache.cache_predictions(
            new_predictions, model_name=model_name, stat_type="passing_yards"
        )

        # Add to results
        for _, row in new_predictions.iterrows():
            cached_predictions.append(
                {
                    "player_id": row["player_id"],
                    "season": row["season"],
                    "week": row["week"],
                    "prediction": row["prediction"],
                    "source": "computed",
                }
            )

    return pd.DataFrame(cached_predictions)


def batch_compute_predictions(
    player_ids: list[str], season: int, week: int, model_name: str
) -> pd.DataFrame:
    """Placeholder for batch prediction computation."""
    return pd.DataFrame(
        {
            "player_id": player_ids,
            "season": [season] * len(player_ids),
            "week": [week] * len(player_ids),
            "prediction": np.random.uniform(100, 400, len(player_ids)),
        }
    )


# =============================================================================
# Example 4: Cache Maintenance and Monitoring
# =============================================================================


def weekly_cache_maintenance() -> dict[str, any]:
    """
    Perform weekly cache maintenance and monitoring.

    Returns:
        Dict with maintenance results
    """
    cache = PredictionCacheManager()
    results = {}

    # 1. Get statistics before maintenance
    logger.info("Getting pre-maintenance statistics")
    stats_before = cache.get_cache_stats()
    results["stats_before"] = {
        "total_predictions": stats_before.total_predictions,
        "cache_size_mb": stats_before.cache_size_mb,
        "hit_rate": stats_before.hit_rate,
        "avg_lookup_time_ms": stats_before.avg_lookup_time_ms,
    }

    # 2. Invalidate stale predictions (older than 7 days)
    logger.info("Invalidating stale predictions (>7 days)")
    stale_count = cache.invalidate_cache(older_than_hours=168)
    results["stale_removed"] = stale_count

    # 3. Optimize indexes
    logger.info("Optimizing indexes")
    index_results = cache.optimize_indexes()
    results["indexes_optimized"] = sum(index_results.values())

    # 4. Run VACUUM ANALYZE
    logger.info("Running VACUUM ANALYZE")
    vacuum_success = cache.vacuum_analyze()
    results["vacuum_success"] = vacuum_success

    # 5. Validate cache integrity
    logger.info("Validating cache integrity (sample 10,000)")
    validation_report = cache.validate_cache_integrity(sample_size=10000)
    results["validation"] = {
        "is_valid": validation_report.is_valid(),
        "total_checked": validation_report.total_checked,
        "invalid_predictions": validation_report.invalid_predictions,
        "nan_predictions": validation_report.nan_predictions,
        "out_of_range_predictions": validation_report.out_of_range_predictions,
    }

    # 6. Get statistics after maintenance
    logger.info("Getting post-maintenance statistics")
    stats_after = cache.get_cache_stats()
    results["stats_after"] = {
        "total_predictions": stats_after.total_predictions,
        "cache_size_mb": stats_after.cache_size_mb,
        "hit_rate": stats_after.hit_rate,
        "avg_lookup_time_ms": stats_after.avg_lookup_time_ms,
    }

    # 7. Check for alerts
    alerts = []
    if stats_after.hit_rate < 90.0:
        alerts.append(f"⚠️ Cache hit rate below target: {stats_after.hit_rate:.1f}%")

    if stats_after.avg_lookup_time_ms > 2.0:
        alerts.append(
            f"⚠️ Lookup time above target: {stats_after.avg_lookup_time_ms:.2f}ms"
        )

    if not validation_report.is_valid():
        alerts.append("⚠️ Cache validation failed - manual review required")

    results["alerts"] = alerts

    # Log summary
    logger.info("=" * 60)
    logger.info("CACHE MAINTENANCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Stale predictions removed: {stale_count:,}")
    logger.info(f"Indexes optimized: {results['indexes_optimized']}/4")
    logger.info(f"VACUUM ANALYZE: {'✅' if vacuum_success else '❌'}")
    logger.info(
        f"Cache size: {stats_before.cache_size_mb:.1f}MB → "
        f"{stats_after.cache_size_mb:.1f}MB "
        f"({stats_after.cache_size_mb - stats_before.cache_size_mb:+.1f}MB)"
    )
    logger.info(f"Hit rate: {stats_after.hit_rate:.1f}%")
    logger.info(f"Avg lookup time: {stats_after.avg_lookup_time_ms:.2f}ms")

    if alerts:
        logger.warning("ALERTS:")
        for alert in alerts:
            logger.warning(f"  {alert}")
    else:
        logger.info("✅ No alerts - cache healthy")

    logger.info("=" * 60)

    return results


# =============================================================================
# Example 5: Migration from rpy2 to PostgreSQL
# =============================================================================


def migrate_rpy2_to_postgres(
    model_name: str, stat_type: str, seasons: list[int]
) -> dict[str, int]:
    """
    Migrate existing rpy2 model predictions to PostgreSQL cache.

    Args:
        model_name: Name of R model
        stat_type: Type of stat
        seasons: Seasons to migrate

    Returns:
        Dict with migration statistics
    """
    cache = PredictionCacheManager()

    # Define rpy2 prediction function
    def get_rpy2_predictions(season, stat_type, player_ids=None):
        """
        Call R model via rpy2 and return predictions.

        In production, this would use rpy2 to call R model.
        """
        logger.info(f"Fetching rpy2 predictions for season {season}")

        # Placeholder: In production, call R model
        # predictions = r_interface.predict_season(season, stat_type, player_ids)

        # Example predictions
        num_players = 100 if player_ids is None else len(player_ids)
        weeks = range(1, 18)  # 17 weeks

        predictions = []
        for week in weeks:
            for i in range(num_players):
                predictions.append(
                    {
                        "player_id": f"P{i:04d}",
                        "season": season,
                        "week": week,
                        "prediction": np.random.uniform(100, 400),
                    }
                )

        return pd.DataFrame(predictions)

    # Perform migration
    logger.info(f"Starting migration for {model_name}/{stat_type}")
    logger.info(f"Seasons: {seasons}")

    total, cached = cache.backfill_from_rpy2(
        model_name=model_name,
        stat_type=stat_type,
        seasons=seasons,
        rpy2_prediction_function=get_rpy2_predictions,
    )

    # Validate migration
    logger.info("Validating migration")
    report = cache.validate_cache_integrity(sample_size=10000)

    if not report.is_valid():
        logger.error(f"Migration validation failed:\n{report.summary()}")
        raise ValueError("Migration validation failed")

    # Get statistics
    stats = cache.get_cache_stats()

    results = {
        "total_predictions": total,
        "cached_predictions": cached,
        "cache_success_rate": (cached / total * 100) if total > 0 else 0,
        "cache_size_mb": stats.cache_size_mb,
        "validation_passed": report.is_valid(),
    }

    logger.info("=" * 60)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total predictions: {total:,}")
    logger.info(f"Successfully cached: {cached:,}")
    logger.info(f"Success rate: {results['cache_success_rate']:.1f}%")
    logger.info(f"Cache size: {stats.cache_size_mb:.2f} MB")
    logger.info(f"Validation: {'✅ PASSED' if report.is_valid() else '❌ FAILED'}")
    logger.info("=" * 60)

    return results


# =============================================================================
# Main: Run Examples
# =============================================================================


def main():
    """Run example integrations."""
    logger.info("=" * 60)
    logger.info("PREDICTION CACHE MANAGER - INTEGRATION EXAMPLES")
    logger.info("=" * 60)

    # Example 1: Ensemble integration
    logger.info("\n### Example 1: Ensemble Integration")
    ensemble = CachedEnsemble()
    predictions = ensemble.predict_passing_yards("P0001", 2024, 10)
    logger.info(f"Ensemble predictions: {predictions}")

    # Example 2: Model training workflow
    logger.info("\n### Example 2: Model Training Workflow")
    cached_count = train_and_cache_model(
        model_name="state_space_v2", stat_type="passing_yards", training_seasons=[2024]
    )
    logger.info(f"Cached {cached_count} predictions")

    # Example 3: Batch prediction
    logger.info("\n### Example 3: Batch Prediction")
    batch_predictions = predict_week_batch(
        season=2024,
        week=10,
        player_ids=[f"P{i:04d}" for i in range(50)],
        model_name="state_space_v2",
    )
    logger.info(f"Batch predictions: {len(batch_predictions)} players")

    # Example 4: Cache maintenance
    logger.info("\n### Example 4: Cache Maintenance")
    maintenance_results = weekly_cache_maintenance()
    logger.info(f"Maintenance complete: {maintenance_results}")

    # Example 5: Migration
    logger.info("\n### Example 5: Migration from rpy2")
    migration_results = migrate_rpy2_to_postgres(
        model_name="state_space_v2", stat_type="passing_yards", seasons=[2024]
    )
    logger.info(f"Migration complete: {migration_results}")

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXAMPLES COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
