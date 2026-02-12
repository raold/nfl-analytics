#!/usr/bin/env python3
"""
Register production Bayesian models to the model registry.

Registers the 3 core Bayesian models used in ensemble v6:
1. State-Space v2 (passing_yards_state_space_v2.rds)
2. Hierarchical v2 (passing_yards_hierarchical_v2.rds)
3. Informative Priors v2 (passing_informative_priors_v2.rds)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.registry.model_registry import ModelRegistry


def register_state_space_v2(registry: ModelRegistry):
    """Register State-Space v2 Bayesian model."""
    registry.register_model(
        name="bayesian_state_space_v2",
        version="v2.0.0",
        author="NFL Analytics Team",
        description=(
            "State-Space v2 Bayesian model for passing yards prediction. "
            "Dynamic player skill estimation with random walk priors. "
            "Uses cmdstanr with 10 MCMC chains (Mac M4 optimized). "
            "Training window: 2018-2024 with H=4 exponential decay. "
            "Ensemble weight: 27% (highest individual component)."
        ),
        metrics={
            "mae": 47.77,  # From CLAUDE.md
            "ensemble_weight": 0.27,
            "rhat_max": 1.01,  # Convergence diagnostic
            "ess_min": 400,  # Effective sample size
            "calibration": 0.90,
        },
        artifacts={
            "model": "models/bayesian/passing_yards_state_space_v2.rds",
            "training_script": "R/train_bayesian_state_space_v3.R",
        },
        tags=["production", "bayesian", "state-space", "cmdstan", "passing"],
    )
    print("✓ Registered State-Space v2 Bayesian model")


def register_hierarchical_v2(registry: ModelRegistry):
    """Register Hierarchical v2 Bayesian model."""
    registry.register_model(
        name="bayesian_hierarchical_v2",
        version="v2.0.0",
        author="NFL Analytics Team",
        description=(
            "Hierarchical v2 Bayesian model for passing yards prediction. "
            "Multi-level model with player, team, and position effects. "
            "Uses brms with 10 MCMC chains (Mac M4 optimized). "
            "Training window: 2018-2024 with H=4 exponential decay. "
            "Ensemble weight: 27% (tied for highest)."
        ),
        metrics={
            "ensemble_weight": 0.27,
            "rhat_max": 1.01,
            "ess_min": 400,
            "calibration": 0.90,
            "model_size_mb": 120,  # From file listing
        },
        artifacts={
            "model": "models/bayesian/passing_yards_hierarchical_v2.rds",
            "training_script": "R/train_and_save_passing_model.R",
        },
        tags=["production", "bayesian", "hierarchical", "brms", "passing"],
    )
    print("✓ Registered Hierarchical v2 Bayesian model")


def register_informative_priors_v2(registry: ModelRegistry):
    """Register Informative Priors v2 Bayesian model."""
    registry.register_model(
        name="bayesian_informative_priors_v2",
        version="v2.0.0",
        author="NFL Analytics Team",
        description=(
            "Informative Priors v2 Bayesian model for passing yards prediction. "
            "Bayesian regression with domain-informed priors on player/team effects. "
            "Uses brms with 10 MCMC chains (Mac M4 optimized). "
            "Training window: 2018-2024 with H=4 exponential decay. "
            "Ensemble weight: 13% (lowest but still valuable for diversity)."
        ),
        metrics={
            "ensemble_weight": 0.13,
            "rhat_max": 1.01,
            "ess_min": 400,
            "calibration": 0.90,
            "model_size_mb": 123,
        },
        artifacts={
            "model": "models/bayesian/passing_informative_priors_v2.rds",
            "training_script": "R/train_and_save_passing_model.R",
        },
        tags=["production", "bayesian", "informative-priors", "brms", "passing"],
    )
    print("✓ Registered Informative Priors v2 Bayesian model")


def register_bnn_rushing_v3(registry: ModelRegistry):
    """Register BNN Rushing v3.1 model (only successful BNN)."""
    registry.register_model(
        name="bnn_rushing_v3_1",
        version="v3.1.0",
        author="NFL Analytics Team",
        description=(
            "Bayesian Neural Network v3.1 for rushing yards prediction. "
            "Only production BNN model (BNN passing research abandoned after 84 hrs). "
            "Uses PyMC with variational inference. Calibrated to 89.5%. "
            "Note: Not used in ensemble v6 (focused on passing yards)."
        ),
        metrics={
            "calibration": 0.895,
            "mae": 38.0,  # Approximate
            "research_hours": 84,  # Total BNN investigation time
        },
        artifacts={
            "model": "models/bayesian/rushing_bnn_v3_1.pkl",
            "research_doc": "docs/milestones/BNN_INVESTIGATION_COMPLETE.md",
        },
        tags=["production", "bnn", "rushing", "pymc", "calibrated"],
    )
    print("✓ Registered BNN Rushing v3.1 model")


def main():
    """Main execution."""
    print("Initializing model registry...")
    registry = ModelRegistry(registry_path="models/registry.json")

    print("\nRegistering production Bayesian models...\n")

    # Register State-Space v2
    try:
        register_state_space_v2(registry)
    except ValueError as e:
        print(f"⚠️  State-Space v2 already registered: {e}")

    # Register Hierarchical v2
    try:
        register_hierarchical_v2(registry)
    except ValueError as e:
        print(f"⚠️  Hierarchical v2 already registered: {e}")

    # Register Informative Priors v2
    try:
        register_informative_priors_v2(registry)
    except ValueError as e:
        print(f"⚠️  Informative Priors v2 already registered: {e}")

    # Register BNN Rushing v3.1
    try:
        register_bnn_rushing_v3(registry)
    except ValueError as e:
        print(f"⚠️  BNN Rushing v3.1 already registered: {e}")

    print("\n✓ Bayesian model registration complete!")

    # Summary
    all_models = registry.list_models()
    bayesian_models = [m for m in all_models if "bayesian" in m.lower() or "bnn" in m.lower()]
    print(f"\nTotal Bayesian models registered: {len(bayesian_models)}")
    for model in sorted(bayesian_models):
        versions = registry.list_versions(model)
        if versions:
            latest = versions[0]
            print(f"  • {model:40} {latest.version}")


if __name__ == "__main__":
    main()
