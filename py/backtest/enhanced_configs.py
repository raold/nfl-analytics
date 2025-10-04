"""Enhanced backtest harness configurations with new features.

This extends the default harness configs to include features from the backfilled data:
- Advanced play-level metrics (success rate, air yards, CPOE)
- Turnover and penalty rates
- Situational tendencies (shotgun rate, explosive plays)
"""

from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    features: Sequence[str]
    threshold: float


# Original baseline config (for comparison)
BASELINE_CONFIG = Config(
    name="baseline_core",
    features=[
        "epa_diff_prior",
        "prior_epa_mean_diff",
        "epa_pp_last3_diff",
        "prior_margin_avg_diff",
        "season_point_diff_avg_diff",
        "rest_diff",
    ],
    threshold=0.5,
)


# Enhanced config with new backfilled features
ENHANCED_CONFIG = Config(
    name="enhanced_full",
    features=[
        # Core EPA metrics (original)
        "epa_diff_prior",
        "prior_epa_mean_diff",
        "epa_pp_last3_diff",
        "prior_margin_avg_diff",
        "season_point_diff_avg_diff",
        
        # Rest and situational
        "rest_diff",
        
        # NEW: Success rate and efficiency
        "prior_success_rate_diff",
        "success_rate_last3_diff",
        
        # NEW: Passing efficiency
        "prior_air_yards_diff",
        "prior_cpoe_diff",
        "prior_completion_pct_diff",
        
        # NEW: Turnovers and penalties
        "prior_turnovers_avg_diff",
        "prior_penalties_avg_diff",
        
        # NEW: Play-calling tendencies
        "prior_shotgun_rate_diff",
        "prior_explosive_pass_rate_diff",
        
        # QB and coaching stability
        "qb_change_diff",
        "coach_change_diff",
        
        # Venue
        "surface_grass_diff",
        "roof_dome_diff",
    ],
    threshold=0.5,
)


# Selective config: Most impactful new features only
ENHANCED_SELECT_CONFIG = Config(
    name="enhanced_select",
    features=[
        # Core EPA (keep all)
        "epa_diff_prior",
        "prior_epa_mean_diff",
        "epa_pp_last3_diff",
        "prior_margin_avg_diff",
        "season_point_diff_avg_diff",
        "rest_diff",
        
        # NEW: Top 5 most impactful features (based on expected importance)
        "prior_success_rate_diff",
        "success_rate_last3_diff",
        "prior_turnovers_avg_diff",
        "prior_cpoe_diff",
        "prior_shotgun_rate_diff",
    ],
    threshold=0.5,
)


# All configs to sweep
ALL_CONFIGS = [
    BASELINE_CONFIG,
    ENHANCED_SELECT_CONFIG,
    ENHANCED_CONFIG,
]


def get_config_by_name(name: str) -> Config:
    """Get a config by name."""
    for config in ALL_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown config: {name}")


def list_config_names() -> list[str]:
    """List all available config names."""
    return [c.name for c in ALL_CONFIGS]
