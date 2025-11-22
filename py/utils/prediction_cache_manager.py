"""
Prediction Cache Manager Agent - PostgreSQL cache management for R model predictions.

This agent manages the ml.r_model_predictions table, providing:
- High-performance bulk inserts (1000+ predictions/batch)
- Fast lookups with composite indexing (<1ms target)
- Cache invalidation and freshness tracking
- Data integrity validation and audit trails
- Migration tools from rpy2 to PostgreSQL

Author: Claude Code (Sonnet 4.5)
Created: November 22, 2025
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics about prediction cache performance and state."""

    total_predictions: int
    unique_players: int
    unique_models: int
    cache_size_mb: float
    hit_rate: float
    avg_lookup_time_ms: float
    oldest_prediction: Optional[datetime]
    newest_prediction: Optional[datetime]
    predictions_by_model: Dict[str, int]
    predictions_by_stat_type: Dict[str, int]


@dataclass
class ValidationReport:
    """Results of cache integrity validation."""

    total_checked: int
    valid_predictions: int
    invalid_predictions: int
    duplicate_predictions: int
    nan_predictions: int
    out_of_range_predictions: int
    orphaned_predictions: int
    issues: List[str]

    def is_valid(self) -> bool:
        """Check if cache passed validation."""
        return (
            self.invalid_predictions == 0
            and self.nan_predictions == 0
            and self.out_of_range_predictions == 0
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Cache Validation Report",
            f"  Total Checked: {self.total_checked:,}",
            f"  Valid: {self.valid_predictions:,}",
            f"  Invalid: {self.invalid_predictions:,}",
            f"  Duplicates: {self.duplicate_predictions:,}",
            f"  NaN Values: {self.nan_predictions:,}",
            f"  Out of Range: {self.out_of_range_predictions:,}",
            f"  Orphaned: {self.orphaned_predictions:,}",
        ]
        if self.issues:
            lines.append(f"\nIssues Found:")
            for issue in self.issues[:10]:  # Show first 10 issues
                lines.append(f"  - {issue}")
            if len(self.issues) > 10:
                lines.append(f"  ... and {len(self.issues) - 10} more")
        return "\n".join(lines)


class PredictionCacheManager:
    """
    Manages PostgreSQL cache for R model predictions.

    Features:
    - Bulk insert optimization (batch 1000+ predictions)
    - Fast lookups with composite indexes (<1ms target)
    - Cache invalidation on model updates
    - Freshness tracking with last_updated timestamps
    - Data integrity validation (range checks, NaN detection)
    - Performance monitoring (hit rate, lookup time)
    - Migration tools from rpy2 to PostgreSQL
    """

    # Stat type ranges (min, max) for validation
    STAT_RANGES = {
        "passing_yards": (0, 600),
        "rushing_yards": (0, 300),
        "receiving_yards": (0, 350),
        "passing_attempts": (0, 70),
        "rushing_attempts": (0, 50),
        "receptions": (0, 20),
    }

    # Batch sizes for bulk operations
    BATCH_SIZE = 1000
    BACKFILL_BATCH_SIZE = 5000

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize cache manager.

        Args:
            db_url: Database connection URL (defaults to env var DB_URL)
        """
        self.db_url = db_url or os.getenv(
            "DB_URL", "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
        )
        self._conn = None
        self._hit_count = 0
        self._miss_count = 0
        self._lookup_times = []

        logger.info("PredictionCacheManager initialized")

    def _get_connection(self):
        """Get database connection (lazy initialization)."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
        return self._conn

    def _close_connection(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def cache_predictions(
        self,
        predictions: pd.DataFrame,
        model_name: str,
        stat_type: str,
        validate: bool = True,
    ) -> int:
        """
        Bulk insert predictions into cache with deduplication.

        Args:
            predictions: DataFrame with columns [player_id, season, week, prediction]
            model_name: Name of model generating predictions
            stat_type: Type of stat being predicted
            validate: Whether to validate predictions before caching

        Returns:
            Number of predictions successfully cached

        Raises:
            ValueError: If predictions DataFrame is invalid
            psycopg2.Error: If database operation fails
        """
        start_time = datetime.now()

        # Validate input DataFrame
        required_cols = {"player_id", "season", "week", "prediction"}
        if not required_cols.issubset(predictions.columns):
            raise ValueError(
                f"predictions must have columns {required_cols}, "
                f"got {predictions.columns.tolist()}"
            )

        # Validate predictions if requested
        if validate:
            predictions = self._validate_predictions(predictions, stat_type)
            if len(predictions) == 0:
                logger.warning("All predictions failed validation")
                return 0

        # Prepare data for insertion
        predictions["model_name"] = model_name
        predictions["stat_type"] = stat_type
        predictions["created_at"] = datetime.now()
        predictions["last_updated"] = datetime.now()

        # Remove duplicates (keep latest)
        predictions = predictions.sort_values("last_updated").drop_duplicates(
            subset=["player_id", "season", "week", "model_name", "stat_type"],
            keep="last",
        )

        # Batch insert
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            total_inserted = 0
            for i in range(0, len(predictions), self.BATCH_SIZE):
                batch = predictions.iloc[i : i + self.BATCH_SIZE]

                # Prepare values for insertion
                values = [
                    (
                        row["player_id"],
                        row["season"],
                        row["week"],
                        row["model_name"],
                        row["stat_type"],
                        float(row["prediction"]),
                        row["created_at"],
                        row["last_updated"],
                    )
                    for _, row in batch.iterrows()
                ]

                # Use INSERT ... ON CONFLICT to handle duplicates
                query = """
                    INSERT INTO ml.r_model_predictions
                    (player_id, season, week, model_name, stat_type,
                     prediction, created_at, last_updated)
                    VALUES %s
                    ON CONFLICT (player_id, season, week, model_name, stat_type)
                    DO UPDATE SET
                        prediction = EXCLUDED.prediction,
                        last_updated = EXCLUDED.last_updated
                """

                execute_values(cursor, query, values)
                total_inserted += len(batch)

                if (i + self.BATCH_SIZE) % (self.BATCH_SIZE * 10) == 0:
                    logger.info(f"Cached {total_inserted}/{len(predictions)} predictions")

            conn.commit()

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Successfully cached {total_inserted} predictions for "
                f"{model_name}/{stat_type} in {elapsed:.2f}s "
                f"({total_inserted/elapsed:.0f} predictions/sec)"
            )

            return total_inserted

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to cache predictions: {e}")
            raise
        finally:
            cursor.close()

    def get_cached_prediction(
        self,
        player_id: str,
        stat_type: str,
        season: int,
        week: int,
        model_name: Optional[str] = None,
        max_age_hours: Optional[int] = None,
    ) -> Optional[float]:
        """
        Fast lookup of cached prediction.

        Args:
            player_id: Player ID
            stat_type: Type of stat
            season: Season year
            week: Week number
            model_name: Specific model (None = any model)
            max_age_hours: Maximum age of prediction (None = any age)

        Returns:
            Cached prediction or None if not found
        """
        start_time = datetime.now()

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Build query with optional filters
            query = """
                SELECT prediction, last_updated
                FROM ml.r_model_predictions
                WHERE player_id = %s
                  AND stat_type = %s
                  AND season = %s
                  AND week = %s
            """
            params = [player_id, stat_type, season, week]

            if model_name:
                query += " AND model_name = %s"
                params.append(model_name)

            if max_age_hours:
                query += " AND last_updated >= %s"
                params.append(datetime.now() - timedelta(hours=max_age_hours))

            query += " ORDER BY last_updated DESC LIMIT 1"

            cursor.execute(query, params)
            result = cursor.fetchone()

            # Track lookup time
            lookup_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._lookup_times.append(lookup_time_ms)

            # Keep only last 1000 lookup times
            if len(self._lookup_times) > 1000:
                self._lookup_times = self._lookup_times[-1000:]

            if result:
                self._hit_count += 1
                logger.debug(
                    f"Cache HIT for {player_id}/{stat_type}/{season}/{week} "
                    f"({lookup_time_ms:.2f}ms)"
                )
                return float(result[0])
            else:
                self._miss_count += 1
                logger.debug(
                    f"Cache MISS for {player_id}/{stat_type}/{season}/{week} "
                    f"({lookup_time_ms:.2f}ms)"
                )
                return None

        finally:
            cursor.close()

    def invalidate_cache(
        self,
        model_name: Optional[str] = None,
        stat_type: Optional[str] = None,
        season: Optional[int] = None,
        older_than_hours: Optional[int] = None,
    ) -> int:
        """
        Invalidate (delete) cached predictions matching criteria.

        Args:
            model_name: Specific model (None = all models)
            stat_type: Specific stat type (None = all types)
            season: Specific season (None = all seasons)
            older_than_hours: Delete predictions older than N hours (None = all ages)

        Returns:
            Number of predictions deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Build DELETE query with filters
            query = "DELETE FROM ml.r_model_predictions WHERE 1=1"
            params = []

            if model_name:
                query += " AND model_name = %s"
                params.append(model_name)

            if stat_type:
                query += " AND stat_type = %s"
                params.append(stat_type)

            if season:
                query += " AND season = %s"
                params.append(season)

            if older_than_hours:
                query += " AND last_updated < %s"
                params.append(datetime.now() - timedelta(hours=older_than_hours))

            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            conn.commit()

            logger.info(
                f"Invalidated {deleted_count} cached predictions "
                f"(model={model_name}, stat_type={stat_type}, "
                f"season={season}, older_than={older_than_hours}h)"
            )

            return deleted_count

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to invalidate cache: {e}")
            raise
        finally:
            cursor.close()

    def get_cache_stats(self) -> CacheStats:
        """
        Get comprehensive cache statistics.

        Returns:
            CacheStats object with performance and state metrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM ml.r_model_predictions")
            total_predictions = cursor.fetchone()[0]

            # Unique players
            cursor.execute(
                "SELECT COUNT(DISTINCT player_id) FROM ml.r_model_predictions"
            )
            unique_players = cursor.fetchone()[0]

            # Unique models
            cursor.execute(
                "SELECT COUNT(DISTINCT model_name) FROM ml.r_model_predictions"
            )
            unique_models = cursor.fetchone()[0]

            # Cache size
            cursor.execute(
                """
                SELECT pg_total_relation_size('ml.r_model_predictions') / 1024.0 / 1024.0
            """
            )
            cache_size_mb = cursor.fetchone()[0] or 0.0

            # Oldest/newest predictions
            cursor.execute(
                """
                SELECT MIN(created_at), MAX(last_updated)
                FROM ml.r_model_predictions
            """
            )
            oldest, newest = cursor.fetchone()

            # Predictions by model
            cursor.execute(
                """
                SELECT model_name, COUNT(*)
                FROM ml.r_model_predictions
                GROUP BY model_name
                ORDER BY COUNT(*) DESC
            """
            )
            predictions_by_model = dict(cursor.fetchall())

            # Predictions by stat type
            cursor.execute(
                """
                SELECT stat_type, COUNT(*)
                FROM ml.r_model_predictions
                GROUP BY stat_type
                ORDER BY COUNT(*) DESC
            """
            )
            predictions_by_stat_type = dict(cursor.fetchall())

            # Calculate hit rate
            total_lookups = self._hit_count + self._miss_count
            hit_rate = (
                (self._hit_count / total_lookups * 100) if total_lookups > 0 else 0.0
            )

            # Average lookup time
            avg_lookup_time = (
                np.mean(self._lookup_times) if self._lookup_times else 0.0
            )

            return CacheStats(
                total_predictions=total_predictions,
                unique_players=unique_players,
                unique_models=unique_models,
                cache_size_mb=cache_size_mb,
                hit_rate=hit_rate,
                avg_lookup_time_ms=avg_lookup_time,
                oldest_prediction=oldest,
                newest_prediction=newest,
                predictions_by_model=predictions_by_model,
                predictions_by_stat_type=predictions_by_stat_type,
            )

        finally:
            cursor.close()

    def validate_cache_integrity(self, sample_size: Optional[int] = None) -> ValidationReport:
        """
        Validate cache data integrity.

        Args:
            sample_size: Number of predictions to check (None = all)

        Returns:
            ValidationReport with detailed findings
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        issues = []
        total_checked = 0
        valid_predictions = 0
        nan_predictions = 0
        out_of_range_predictions = 0
        duplicate_predictions = 0
        orphaned_predictions = 0

        try:
            # Get sample of predictions
            if sample_size:
                cursor.execute(
                    """
                    SELECT player_id, season, week, model_name, stat_type,
                           prediction, created_at, last_updated
                    FROM ml.r_model_predictions
                    ORDER BY RANDOM()
                    LIMIT %s
                """,
                    (sample_size,),
                )
            else:
                cursor.execute(
                    """
                    SELECT player_id, season, week, model_name, stat_type,
                           prediction, created_at, last_updated
                    FROM ml.r_model_predictions
                """
                )

            predictions = cursor.fetchall()
            total_checked = len(predictions)

            for row in predictions:
                (
                    player_id,
                    season,
                    week,
                    model_name,
                    stat_type,
                    prediction,
                    created_at,
                    last_updated,
                ) = row

                # Check for NaN
                if prediction is None or np.isnan(prediction):
                    nan_predictions += 1
                    issues.append(
                        f"NaN prediction for {player_id}/{stat_type}/{season}/{week}"
                    )
                    continue

                # Check range
                if stat_type in self.STAT_RANGES:
                    min_val, max_val = self.STAT_RANGES[stat_type]
                    if not (min_val <= prediction <= max_val):
                        out_of_range_predictions += 1
                        issues.append(
                            f"Out of range ({prediction:.1f}) for "
                            f"{player_id}/{stat_type}/{season}/{week} "
                            f"(expected {min_val}-{max_val})"
                        )
                        continue

                # Check timestamps
                if created_at > last_updated:
                    issues.append(
                        f"Invalid timestamps for {player_id}/{stat_type}/{season}/{week} "
                        f"(created_at > last_updated)"
                    )
                    continue

                valid_predictions += 1

            # Check for duplicates
            cursor.execute(
                """
                SELECT player_id, season, week, model_name, stat_type, COUNT(*)
                FROM ml.r_model_predictions
                GROUP BY player_id, season, week, model_name, stat_type
                HAVING COUNT(*) > 1
            """
            )
            duplicates = cursor.fetchall()
            duplicate_predictions = len(duplicates)
            for dup in duplicates[:10]:  # Show first 10
                issues.append(f"Duplicate prediction: {dup[:-1]} (count={dup[-1]})")

            # Check for orphaned predictions (player not in database)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM ml.r_model_predictions rp
                LEFT JOIN player_game_stats pgs
                  ON rp.player_id = pgs.player_id
                WHERE pgs.player_id IS NULL
            """
            )
            orphaned_predictions = cursor.fetchone()[0] or 0
            if orphaned_predictions > 0:
                issues.append(
                    f"{orphaned_predictions} predictions for players not in database"
                )

            invalid_predictions = (
                nan_predictions + out_of_range_predictions + duplicate_predictions
            )

            return ValidationReport(
                total_checked=total_checked,
                valid_predictions=valid_predictions,
                invalid_predictions=invalid_predictions,
                duplicate_predictions=duplicate_predictions,
                nan_predictions=nan_predictions,
                out_of_range_predictions=out_of_range_predictions,
                orphaned_predictions=orphaned_predictions,
                issues=issues,
            )

        finally:
            cursor.close()

    def optimize_indexes(self) -> Dict[str, bool]:
        """
        Optimize cache table indexes for performance.

        Returns:
            Dict of index names and creation status
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        indexes = {
            # Primary composite index for lookups
            "idx_r_predictions_lookup": """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_r_predictions_lookup
                ON ml.r_model_predictions (player_id, stat_type, season, week)
            """,
            # Model-specific queries
            "idx_r_predictions_model": """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_r_predictions_model
                ON ml.r_model_predictions (model_name, stat_type)
            """,
            # Freshness tracking
            "idx_r_predictions_updated": """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_r_predictions_updated
                ON ml.r_model_predictions (last_updated)
            """,
            # Season-based queries
            "idx_r_predictions_season": """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_r_predictions_season
                ON ml.r_model_predictions (season, week)
            """,
        }

        results = {}
        for name, query in indexes.items():
            try:
                cursor.execute(query)
                conn.commit()
                results[name] = True
                logger.info(f"Created/verified index: {name}")
            except Exception as e:
                conn.rollback()
                results[name] = False
                logger.error(f"Failed to create index {name}: {e}")

        cursor.close()
        return results

    def vacuum_analyze(self) -> bool:
        """
        Run VACUUM ANALYZE on cache table to optimize performance.

        Returns:
            True if successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Need to run VACUUM outside transaction
            old_isolation = conn.isolation_level
            conn.set_isolation_level(0)

            cursor.execute("VACUUM ANALYZE ml.r_model_predictions")

            conn.set_isolation_level(old_isolation)

            logger.info("Successfully ran VACUUM ANALYZE on cache table")
            return True

        except Exception as e:
            logger.error(f"Failed to VACUUM ANALYZE: {e}")
            return False
        finally:
            cursor.close()

    def backfill_from_rpy2(
        self,
        model_name: str,
        stat_type: str,
        seasons: List[int],
        rpy2_prediction_function,
        player_ids: Optional[List[str]] = None,
    ) -> Tuple[int, int]:
        """
        Backfill cache from existing rpy2 model predictions.

        Args:
            model_name: Name of R model
            stat_type: Type of stat being predicted
            seasons: Seasons to backfill
            rpy2_prediction_function: Function that returns predictions DataFrame
            player_ids: Specific players (None = all players)

        Returns:
            Tuple of (total_predictions, cached_predictions)
        """
        logger.info(
            f"Starting backfill for {model_name}/{stat_type} across {len(seasons)} seasons"
        )

        total_predictions = 0
        cached_predictions = 0

        for season in seasons:
            logger.info(f"Backfilling season {season}")

            try:
                # Get predictions from rpy2
                predictions_df = rpy2_prediction_function(
                    season=season, stat_type=stat_type, player_ids=player_ids
                )

                if predictions_df is None or len(predictions_df) == 0:
                    logger.warning(f"No predictions for season {season}")
                    continue

                total_predictions += len(predictions_df)

                # Cache in batches
                for i in range(0, len(predictions_df), self.BACKFILL_BATCH_SIZE):
                    batch = predictions_df.iloc[i : i + self.BACKFILL_BATCH_SIZE]
                    cached = self.cache_predictions(
                        batch, model_name, stat_type, validate=True
                    )
                    cached_predictions += cached

                    logger.info(
                        f"Season {season}: Cached {cached_predictions}/{total_predictions} "
                        f"predictions ({cached_predictions/total_predictions*100:.1f}%)"
                    )

            except Exception as e:
                logger.error(f"Failed to backfill season {season}: {e}")
                continue

        logger.info(
            f"Backfill complete: Cached {cached_predictions}/{total_predictions} "
            f"predictions ({cached_predictions/total_predictions*100:.1f}%)"
        )

        return total_predictions, cached_predictions

    def _validate_predictions(
        self, predictions: pd.DataFrame, stat_type: str
    ) -> pd.DataFrame:
        """
        Validate predictions before caching.

        Args:
            predictions: DataFrame with predictions
            stat_type: Type of stat

        Returns:
            Filtered DataFrame with valid predictions only
        """
        initial_count = len(predictions)

        # Remove NaN predictions
        predictions = predictions[predictions["prediction"].notna()]
        predictions = predictions[~predictions["prediction"].apply(np.isnan)]

        # Check range
        if stat_type in self.STAT_RANGES:
            min_val, max_val = self.STAT_RANGES[stat_type]
            predictions = predictions[
                (predictions["prediction"] >= min_val)
                & (predictions["prediction"] <= max_val)
            ]

        # Remove invalid player IDs
        predictions = predictions[predictions["player_id"].notna()]
        predictions = predictions[predictions["player_id"] != ""]

        # Remove invalid seasons/weeks
        predictions = predictions[
            (predictions["season"] >= 2015) & (predictions["season"] <= 2030)
        ]
        predictions = predictions[
            (predictions["week"] >= 1) & (predictions["week"] <= 22)
        ]

        removed = initial_count - len(predictions)
        if removed > 0:
            logger.warning(
                f"Removed {removed}/{initial_count} invalid predictions "
                f"({removed/initial_count*100:.1f}%)"
            )

        return predictions

    def __del__(self):
        """Cleanup on deletion."""
        self._close_connection()


# CLI helper functions
def main():
    """Command-line interface for cache management."""
    import argparse

    parser = argparse.ArgumentParser(description="Prediction Cache Manager")
    parser.add_argument(
        "command",
        choices=["backfill", "invalidate", "stats", "optimize", "validate"],
        help="Command to execute",
    )
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--stat-type", help="Stat type")
    parser.add_argument("--season", type=int, help="Season year")
    parser.add_argument("--older-than", type=int, help="Hours threshold")
    parser.add_argument("--sample-size", type=int, help="Validation sample size")

    args = parser.parse_args()

    manager = PredictionCacheManager()

    if args.command == "stats":
        stats = manager.get_cache_stats()
        print(f"\n{'='*60}")
        print("PREDICTION CACHE STATISTICS")
        print(f"{'='*60}")
        print(f"Total Predictions: {stats.total_predictions:,}")
        print(f"Unique Players: {stats.unique_players:,}")
        print(f"Unique Models: {stats.unique_models:,}")
        print(f"Cache Size: {stats.cache_size_mb:.2f} MB")
        print(f"Hit Rate: {stats.hit_rate:.1f}%")
        print(f"Avg Lookup Time: {stats.avg_lookup_time_ms:.2f} ms")
        print(f"Oldest Prediction: {stats.oldest_prediction}")
        print(f"Newest Prediction: {stats.newest_prediction}")
        print(f"\nPredictions by Model:")
        for model, count in stats.predictions_by_model.items():
            print(f"  {model}: {count:,}")
        print(f"\nPredictions by Stat Type:")
        for stat, count in stats.predictions_by_stat_type.items():
            print(f"  {stat}: {count:,}")
        print(f"{'='*60}\n")

    elif args.command == "invalidate":
        count = manager.invalidate_cache(
            model_name=args.model,
            stat_type=args.stat_type,
            season=args.season,
            older_than_hours=args.older_than,
        )
        print(f"Invalidated {count:,} cached predictions")

    elif args.command == "optimize":
        print("Optimizing indexes...")
        results = manager.optimize_indexes()
        for name, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {name}")

        print("\nRunning VACUUM ANALYZE...")
        success = manager.vacuum_analyze()
        status = "✅" if success else "❌"
        print(f"{status} VACUUM ANALYZE")

    elif args.command == "validate":
        print("Validating cache integrity...")
        report = manager.validate_cache_integrity(sample_size=args.sample_size)
        print(f"\n{report.summary()}\n")

        if report.is_valid():
            print("✅ Cache validation PASSED")
        else:
            print("❌ Cache validation FAILED")
            exit(1)

    else:
        print(f"Command '{args.command}' not yet implemented")
        exit(1)


if __name__ == "__main__":
    main()
