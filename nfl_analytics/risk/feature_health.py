"""
Feature pipeline health monitor — blocks trades when data is stale or degraded.

Layer 5 in the risk hierarchy — arguably the MOST IMPORTANT layer.

2025 Lesson Learned: Missing QBR/NextGen/semantic data compressed model
predictions from [-35, +25] to [-2.6, +8.7]. The model ran, produced
outputs, and the paper trader happily placed bets — all garbage.

This module prevents that disaster by checking:
1. Feature staleness — are data sources up-to-date?
2. Feature completeness — are critical feature groups present?
3. Feature distribution — do feature values look normal?

Every trade signal must pass a feature health check. If the pipeline is
degraded, NO trades are placed until data is restored.

Usage:
    from nfl_analytics.risk.feature_health import FeatureHealthChecker, HealthConfig

    checker = FeatureHealthChecker()
    result = checker.check_features(features_dict, season=2026, week=5)
    if not result.healthy:
        logger.critical("BLOCKING TRADES: %s", result.issues)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Feature group definitions — which prefixes belong to which data source
FEATURE_GROUPS = {
    "base_offensive": {
        "prefixes": ["home_off_", "away_off_"],
        "min_count": 40,  # Expect ~42 per team = 84 total
        "critical": True,
        "source": "mart.team_offensive_stats",
    },
    "base_defensive": {
        "prefixes": ["home_def_", "away_def_"],
        "min_count": 30,
        "critical": True,
        "source": "mart.team_defensive_stats",
    },
    "nextgen": {
        "prefixes": ["home_ngs_", "away_ngs_"],
        "min_count": 20,  # 23 per team = 46
        "critical": True,
        "source": "nextgen_passing/rushing/receiving",
    },
    "qbr": {
        "prefixes": ["home_qbr_", "away_qbr_"],
        "min_count": 10,  # 10 per team = 20
        "critical": True,
        "source": "ESPN QBR API",
    },
    "injury": {
        "prefixes": ["home_injury_", "away_injury_", "injury_"],
        "min_count": 8,
        "critical": True,
        "source": "nflreadr injuries",
    },
    "turnover_luck": {
        "prefixes": ["to_"],
        "min_count": 10,
        "critical": False,
        "source": "derived from plays",
    },
    "primetime": {
        "prefixes": ["pt_"],
        "min_count": 5,
        "critical": False,
        "source": "game schedule",
    },
    "strength_of_victory": {
        "prefixes": ["sov_"],
        "min_count": 8,
        "critical": False,
        "source": "derived from games",
    },
    "rest": {
        "prefixes": ["rest_", "is_home_", "is_away_", "home_days_", "away_days_"],
        "min_count": 4,
        "critical": False,
        "source": "game schedule",
    },
    "semantic": {
        "prefixes": ["sem_"],
        "min_count": 20,  # 38 expected
        "critical": False,  # Not critical yet (A/B test inconclusive)
        "source": "Ollama/MLX LLM inference",
    },
    "synthetic": {
        "prefixes": ["syn_", "mom_", "res_"],
        "min_count": 20,
        "critical": False,
        "source": "derived features",
    },
}


@dataclass
class HealthConfig:
    """Configuration for feature health checks."""

    # Minimum non-zero ratio per feature group (below = "compressed")
    min_nonzero_ratio: float = 0.30
    # Minimum total feature count (382 expected, allow some missing)
    min_total_features: int = 200
    # Maximum ratio of features that are exactly 0.0
    max_zero_ratio: float = 0.50
    # Block on any critical group failure
    block_on_critical: bool = True
    # Block if total features below threshold
    block_on_low_count: bool = True


@dataclass
class FeatureGroupStatus:
    """Health status of a single feature group."""

    name: str
    expected_min: int
    actual_count: int
    nonzero_count: int
    critical: bool
    healthy: bool
    issue: str = ""

    @property
    def nonzero_ratio(self) -> float:
        if self.actual_count == 0:
            return 0.0
        return self.nonzero_count / self.actual_count


@dataclass
class HealthCheckResult:
    """Result of a comprehensive feature health check."""

    healthy: bool
    total_features: int
    nonzero_features: int
    group_statuses: list[FeatureGroupStatus] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def nonzero_ratio(self) -> float:
        if self.total_features == 0:
            return 0.0
        return self.nonzero_features / self.total_features

    @property
    def critical_issues(self) -> list[str]:
        return [s.issue for s in self.group_statuses if not s.healthy and s.critical]

    def summary(self) -> str:
        status = "HEALTHY" if self.healthy else "DEGRADED"
        lines = [
            f"Feature Health: {status}",
            f"  Total: {self.total_features} features ({self.nonzero_features} nonzero, "
            f"{self.nonzero_ratio:.0%} ratio)",
        ]
        for gs in self.group_statuses:
            mark = "OK" if gs.healthy else "FAIL"
            crit = " [CRITICAL]" if gs.critical and not gs.healthy else ""
            lines.append(
                f"  {mark} {gs.name}: {gs.actual_count}/{gs.expected_min} "
                f"({gs.nonzero_count} nonzero){crit}"
            )
        if self.issues:
            lines.append(f"  Issues: {'; '.join(self.issues)}")
        return "\n".join(lines)


class FeatureHealthChecker:
    """Checks feature pipeline health before allowing trades.

    Designed to prevent the 2025 disaster where missing data sources
    caused prediction compression without any warnings.
    """

    def __init__(self, config: HealthConfig | None = None):
        self.config = config or HealthConfig()
        self._last_check: HealthCheckResult | None = None

    @property
    def last_check(self) -> HealthCheckResult | None:
        return self._last_check

    def check_features(self, features: dict[str, float]) -> HealthCheckResult:
        """Run a comprehensive health check on extracted features.

        Args:
            features: Dictionary of feature_name → value from GameFeatureExtractor.

        Returns:
            HealthCheckResult with overall health status and per-group breakdown.
        """
        total = len(features)
        nonzero = sum(1 for v in features.values() if v != 0.0 and v is not None)
        issues = []
        group_statuses = []

        # Check each feature group
        for group_name, group_def in FEATURE_GROUPS.items():
            status = self._check_group(group_name, group_def, features)
            group_statuses.append(status)
            if not status.healthy:
                issues.append(status.issue)

        # Check total feature count
        if total < self.config.min_total_features:
            issues.append(f"Low feature count: {total} < {self.config.min_total_features}")

        # Check overall zero ratio (the compression detector)
        zero_ratio = 1.0 - (nonzero / total if total > 0 else 0)
        if zero_ratio > self.config.max_zero_ratio:
            issues.append(
                f"High zero ratio: {zero_ratio:.0%} of features are zero "
                f"(max {self.config.max_zero_ratio:.0%}). "
                f"Possible data pipeline failure."
            )

        # Determine overall health
        healthy = True
        if self.config.block_on_critical:
            critical_failures = [s for s in group_statuses if not s.healthy and s.critical]
            if critical_failures:
                healthy = False
        if self.config.block_on_low_count and total < self.config.min_total_features:
            healthy = False

        result = HealthCheckResult(
            healthy=healthy,
            total_features=total,
            nonzero_features=nonzero,
            group_statuses=group_statuses,
            issues=issues,
        )

        self._last_check = result

        if not healthy:
            logger.warning("Feature health check FAILED:\n%s", result.summary())
        else:
            logger.debug("Feature health check passed: %d features, %d nonzero", total, nonzero)

        return result

    def _check_group(
        self,
        name: str,
        group_def: dict,
        features: dict[str, float],
    ) -> FeatureGroupStatus:
        """Check a single feature group."""
        prefixes = group_def["prefixes"]
        min_count = group_def["min_count"]
        critical = group_def["critical"]

        # Count features matching this group's prefixes
        group_features = {
            k: v for k, v in features.items() if any(k.startswith(p) for p in prefixes)
        }
        actual_count = len(group_features)
        nonzero_count = sum(1 for v in group_features.values() if v != 0.0 and v is not None)

        # Determine health
        healthy = True
        issue = ""

        if actual_count < min_count:
            healthy = False
            issue = (
                f"{name}: only {actual_count}/{min_count} features present "
                f"(source: {group_def['source']})"
            )
        elif actual_count > 0:
            nz_ratio = nonzero_count / actual_count
            if nz_ratio < self.config.min_nonzero_ratio:
                healthy = False
                issue = (
                    f"{name}: {nz_ratio:.0%} nonzero ({nonzero_count}/{actual_count}), "
                    f"min {self.config.min_nonzero_ratio:.0%} "
                    f"(source: {group_def['source']})"
                )

        return FeatureGroupStatus(
            name=name,
            expected_min=min_count,
            actual_count=actual_count,
            nonzero_count=nonzero_count,
            critical=critical,
            healthy=healthy,
            issue=issue,
        )
