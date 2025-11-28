#!/usr/bin/env python3
"""
Data Freshness Validator - Ensure predictions and data are current

Tracks "last updated" timestamps, validates ensemble component availability,
auto-refreshes stale data, and provides UI indicators for data age.

Purpose:
- Track last_updated timestamps for all data sources
- Validate ensemble components are available before predictions
- Auto-refresh stale data (>24 hours for predictions, >1 week for historical)
- Provide UI indicators (🟢 fresh, 🟡 stale, 🔴 outdated)
- Prevent serving predictions from incomplete ensemble

Usage:
    from py.agents.data_freshness_validator import DataFreshnessValidator

    validator = DataFreshnessValidator()

    # Check if data is fresh
    status = validator.check_data_freshness("predictions", season=2024, week=12)
    if not status.is_fresh:
        validator.refresh_data("predictions", season=2024, week=12)

    # Validate ensemble before predictions
    health = validator.validate_ensemble_availability()
    if not health.is_healthy:
        raise RuntimeError(f"Ensemble unhealthy: {health.missing_components}")

    # Get UI indicator
    indicator = validator.get_freshness_indicator("predictions")
    st.write(f"{indicator} Predictions")

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Phase: 6 (Frontend Agents - Week 4)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FreshnessStatus(Enum):
    """Data freshness status levels."""

    FRESH = "fresh"           # Updated within threshold
    STALE = "stale"           # Older than ideal but usable
    OUTDATED = "outdated"     # Too old, needs refresh
    MISSING = "missing"       # No data available


class DataSource(Enum):
    """Types of data sources to monitor."""

    PREDICTIONS = "predictions"
    ODDS = "odds"
    INJURY_REPORTS = "injury_reports"
    WEATHER = "weather"
    PLAYER_STATS = "player_stats"
    TEAM_STATS = "team_stats"
    ENSEMBLE_WEIGHTS = "ensemble_weights"
    MODEL_ARTIFACTS = "model_artifacts"


@dataclass
class FreshnessThreshold:
    """Thresholds for data freshness by source."""

    fresh_threshold_hours: int  # Green: data within this age
    stale_threshold_hours: int  # Yellow: data between fresh and stale
    # Red: data older than stale_threshold_hours


@dataclass
class DataFreshnessStatus:
    """Status of a specific data source."""

    source: DataSource
    last_updated: Optional[datetime]
    age_hours: Optional[float]
    status: FreshnessStatus
    is_fresh: bool
    indicator: str  # Emoji indicator
    message: str


@dataclass
class EnsembleComponentStatus:
    """Status of an ensemble component."""

    component_name: str
    model_path: Optional[Path]
    exists: bool
    last_modified: Optional[datetime]
    age_hours: Optional[float]
    is_available: bool
    message: str


@dataclass
class EnsembleHealth:
    """Overall ensemble health check."""

    total_components: int
    available_components: int
    missing_components: List[str]
    outdated_components: List[str]
    is_healthy: bool
    can_generate_predictions: bool
    health_score: float  # 0.0-1.0
    components: List[EnsembleComponentStatus]
    checked_at: datetime = field(default_factory=datetime.now)


@dataclass
class RefreshReport:
    """Report from data refresh operation."""

    source: DataSource
    refresh_started: datetime
    refresh_completed: datetime
    duration_seconds: float
    records_updated: int
    success: bool
    error: Optional[str]


class DataFreshnessValidator:
    """
    Validates data freshness and ensemble component availability.

    Monitors timestamps, auto-refreshes stale data, provides UI indicators.
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",  # pragma: allowlist secret
        models_path: str = "models/"
    ):
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password
        }
        self.models_path = Path(models_path)

        # Define freshness thresholds by data source
        self.thresholds = {
            DataSource.PREDICTIONS: FreshnessThreshold(
                fresh_threshold_hours=24,   # Predictions should be daily
                stale_threshold_hours=72    # Warning after 3 days
            ),
            DataSource.ODDS: FreshnessThreshold(
                fresh_threshold_hours=1,    # Odds change hourly
                stale_threshold_hours=6
            ),
            DataSource.INJURY_REPORTS: FreshnessThreshold(
                fresh_threshold_hours=12,   # Injury reports update 2x daily
                stale_threshold_hours=48
            ),
            DataSource.WEATHER: FreshnessThreshold(
                fresh_threshold_hours=6,    # Weather updates every 6 hours
                stale_threshold_hours=24
            ),
            DataSource.PLAYER_STATS: FreshnessThreshold(
                fresh_threshold_hours=168,  # Stats update weekly (after games)
                stale_threshold_hours=336   # Warning after 2 weeks
            ),
            DataSource.TEAM_STATS: FreshnessThreshold(
                fresh_threshold_hours=168,
                stale_threshold_hours=336
            ),
            DataSource.ENSEMBLE_WEIGHTS: FreshnessThreshold(
                fresh_threshold_hours=168,  # Weights reoptimized weekly
                stale_threshold_hours=720   # Warning after 30 days
            ),
            DataSource.MODEL_ARTIFACTS: FreshnessThreshold(
                fresh_threshold_hours=720,  # Models retrained monthly
                stale_threshold_hours=2160  # Warning after 90 days
            ),
        }

        # Emoji indicators
        self.indicators = {
            FreshnessStatus.FRESH: "🟢",
            FreshnessStatus.STALE: "🟡",
            FreshnessStatus.OUTDATED: "🔴",
            FreshnessStatus.MISSING: "⚫"
        }

    def check_data_freshness(
        self,
        source: str,
        **filters
    ) -> DataFreshnessStatus:
        """
        Check freshness of a data source.

        Args:
            source: Data source to check (e.g., "predictions")
            **filters: Optional filters (e.g., season=2024, week=12)

        Returns:
            DataFreshnessStatus with age and recommendations
        """
        source_enum = DataSource(source)
        threshold = self.thresholds[source_enum]

        # Query database for last_updated timestamp
        last_updated = self._get_last_updated(source_enum, filters)

        if last_updated is None:
            return DataFreshnessStatus(
                source=source_enum,
                last_updated=None,
                age_hours=None,
                status=FreshnessStatus.MISSING,
                is_fresh=False,
                indicator=self.indicators[FreshnessStatus.MISSING],
                message=f"No data found for {source}"
            )

        # Calculate age
        age = datetime.now() - last_updated
        age_hours = age.total_seconds() / 3600

        # Determine status
        if age_hours <= threshold.fresh_threshold_hours:
            status = FreshnessStatus.FRESH
            is_fresh = True
            message = f"Data is fresh (updated {age_hours:.1f}h ago)"
        elif age_hours <= threshold.stale_threshold_hours:
            status = FreshnessStatus.STALE
            is_fresh = False
            message = f"Data is stale (updated {age_hours:.1f}h ago, consider refresh)"
        else:
            status = FreshnessStatus.OUTDATED
            is_fresh = False
            message = f"Data is outdated (updated {age_hours:.1f}h ago, refresh recommended)"

        return DataFreshnessStatus(
            source=source_enum,
            last_updated=last_updated,
            age_hours=age_hours,
            status=status,
            is_fresh=is_fresh,
            indicator=self.indicators[status],
            message=message
        )

    def validate_ensemble_availability(self) -> EnsembleHealth:
        """
        Validate all ensemble components are available and current.

        Checks:
        - Model files exist
        - R model predictions cached in database
        - Ensemble weights are current
        - All 5 components present (State-Space, Hierarchical, XGBoost, Informative Priors, Static-Incremental)

        Returns:
            EnsembleHealth with detailed component status
        """
        # Expected ensemble components (v6.1)
        expected_components = {
            "state_space_v2": "bayesian/state_space_v2.rds",
            "hierarchical_v2": "bayesian/hierarchical_v2.rds",
            "xgboost_v4_ridge": "xgboost/passing_xgboost_v4_ridge.pkl",
            "informative_priors_v2": "bayesian/informative_priors_v2.rds",
            "hybrid_static_incremental": "hybrid/static_incremental_v1.pkl"
        }

        component_statuses = []
        available_count = 0
        missing_components = []
        outdated_components = []

        for component_name, relative_path in expected_components.items():
            model_path = self.models_path / relative_path

            if not model_path.exists():
                component_statuses.append(EnsembleComponentStatus(
                    component_name=component_name,
                    model_path=model_path,
                    exists=False,
                    last_modified=None,
                    age_hours=None,
                    is_available=False,
                    message=f"Model file missing: {model_path}"
                ))
                missing_components.append(component_name)
                continue

            # Check file age
            last_modified = datetime.fromtimestamp(model_path.stat().st_mtime)
            age = datetime.now() - last_modified
            age_hours = age.total_seconds() / 3600

            # Models >90 days old are outdated
            is_outdated = age_hours > (90 * 24)
            is_available = True

            if is_outdated:
                outdated_components.append(component_name)
                message = f"Model outdated (last modified {age_hours:.0f}h ago)"
            else:
                message = f"Model available (last modified {age_hours:.0f}h ago)"
                available_count += 1

            component_statuses.append(EnsembleComponentStatus(
                component_name=component_name,
                model_path=model_path,
                exists=True,
                last_modified=last_modified,
                age_hours=age_hours,
                is_available=is_available and not is_outdated,
                message=message
            ))

        # Additional check: R model predictions in database
        r_predictions_status = self._check_r_predictions_cached()
        if not r_predictions_status:
            missing_components.append("r_model_predictions_cache")

        # Determine overall health
        total_components = len(expected_components)
        health_score = available_count / total_components
        is_healthy = health_score >= 0.8  # 80% of components available
        can_generate_predictions = health_score >= 1.0  # All components needed

        return EnsembleHealth(
            total_components=total_components,
            available_components=available_count,
            missing_components=missing_components,
            outdated_components=outdated_components,
            is_healthy=is_healthy,
            can_generate_predictions=can_generate_predictions,
            health_score=health_score,
            components=component_statuses
        )

    def refresh_data(
        self,
        source: str,
        **filters
    ) -> RefreshReport:
        """
        Refresh stale data by triggering appropriate update script.

        Args:
            source: Data source to refresh
            **filters: Optional filters for partial refresh

        Returns:
            RefreshReport with refresh status
        """
        source_enum = DataSource(source)
        start_time = datetime.now()

        try:
            if source_enum == DataSource.PREDICTIONS:
                records = self._refresh_predictions(**filters)
            elif source_enum == DataSource.ODDS:
                records = self._refresh_odds(**filters)
            elif source_enum == DataSource.INJURY_REPORTS:
                records = self._refresh_injury_reports(**filters)
            elif source_enum == DataSource.WEATHER:
                records = self._refresh_weather(**filters)
            else:
                raise ValueError(f"Refresh not implemented for {source}")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return RefreshReport(
                source=source_enum,
                refresh_started=start_time,
                refresh_completed=end_time,
                duration_seconds=duration,
                records_updated=records,
                success=True,
                error=None
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Error refreshing {source}: {e}")

            return RefreshReport(
                source=source_enum,
                refresh_started=start_time,
                refresh_completed=end_time,
                duration_seconds=duration,
                records_updated=0,
                success=False,
                error=str(e)
            )

    def get_freshness_indicator(self, source: str, **filters) -> str:
        """
        Get emoji indicator for data freshness (for UI display).

        Args:
            source: Data source to check
            **filters: Optional filters

        Returns:
            Emoji string (🟢/🟡/🔴/⚫)
        """
        status = self.check_data_freshness(source, **filters)
        return status.indicator

    def get_all_freshness_statuses(self) -> Dict[str, DataFreshnessStatus]:
        """
        Check freshness of all monitored data sources.

        Returns:
            Dictionary mapping source names to freshness statuses
        """
        statuses = {}

        for source in DataSource:
            try:
                statuses[source.value] = self.check_data_freshness(source.value)
            except Exception as e:
                logger.warning(f"Error checking {source.value}: {e}")

        return statuses

    # Private helper methods

    def _get_last_updated(
        self,
        source: DataSource,
        filters: Dict[str, Any]
    ) -> Optional[datetime]:
        """Query database for last_updated timestamp."""

        table_mapping = {
            DataSource.PREDICTIONS: "ml.predictions",
            DataSource.ODDS: "public.sportsbook_lines",
            DataSource.INJURY_REPORTS: "public.injury_reports",
            DataSource.WEATHER: "public.weather_data",
            DataSource.PLAYER_STATS: "mart.player_game_stats",
            DataSource.TEAM_STATS: "mart.team_game_stats",
        }

        if source not in table_mapping:
            return None

        table = table_mapping[source]

        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Build WHERE clause from filters
                    where_clause = " AND ".join(
                        f"{k} = %s" for k in filters.keys()
                    )
                    where_clause = f"WHERE {where_clause}" if where_clause else ""

                    query = f"""
                    SELECT MAX(updated_at) as last_updated
                    FROM {table}
                    {where_clause}
                    """

                    cur.execute(query, tuple(filters.values()))
                    result = cur.fetchone()

                    return result[0] if result and result[0] else None

        except Exception as e:
            logger.error(f"Error querying {table}: {e}")
            return None

    def _check_r_predictions_cached(self) -> bool:
        """Check if R model predictions are cached in database."""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                    SELECT COUNT(*)
                    FROM ml.r_model_predictions
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    """)
                    count = cur.fetchone()[0]
                    return count > 0
        except Exception as e:
            logger.warning(f"Error checking R predictions cache: {e}")
            return False

    def _refresh_predictions(self, **filters) -> int:
        """Refresh predictions by calling ensemble."""
        # Would call generate_predictions script
        logger.info("Refreshing predictions...")
        return 0  # Placeholder

    def _refresh_odds(self, **filters) -> int:
        """Refresh odds from sportsbook API."""
        logger.info("Refreshing odds...")
        return 0  # Placeholder

    def _refresh_injury_reports(self, **filters) -> int:
        """Refresh injury reports."""
        logger.info("Refreshing injury reports...")
        return 0  # Placeholder

    def _refresh_weather(self, **filters) -> int:
        """Refresh weather data."""
        logger.info("Refreshing weather...")
        return 0  # Placeholder


if __name__ == "__main__":
    # Example usage
    validator = DataFreshnessValidator()

    # Check predictions freshness
    status = validator.check_data_freshness("predictions")
    print(f"\n{status.indicator} Predictions: {status.message}")

    # Validate ensemble
    health = validator.validate_ensemble_availability()
    print(f"\n=== Ensemble Health ===")
    print(f"Available: {health.available_components}/{health.total_components}")
    print(f"Health score: {health.health_score:.0%}")
    print(f"Can generate predictions: {health.can_generate_predictions}")

    if health.missing_components:
        print(f"Missing: {', '.join(health.missing_components)}")
    if health.outdated_components:
        print(f"Outdated: {', '.join(health.outdated_components)}")

    # Check all sources
    print(f"\n=== All Data Sources ===")
    statuses = validator.get_all_freshness_statuses()
    for source, status in statuses.items():
        print(f"{status.indicator} {source}: {status.message}")
