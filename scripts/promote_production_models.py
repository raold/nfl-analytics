#!/usr/bin/env python3
"""
Promote production-ready models through staging to production.

This script promotes the core ensemble v6 production stack:
1. XGBoost v5 integrated → staging
2. Ensemble v6.1 Dirichlet weights → staging
3. Bayesian State-Space v2 → staging
4. Bayesian Hierarchical v2 → staging
5. Bayesian Informative Priors v2 → staging
6. BNN Rushing v3.1 → staging

Note: Models go to staging first for validation, then can be promoted
to production after successful backtesting/monitoring.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.registry.model_registry import ModelRegistry


def promote_to_staging(registry: ModelRegistry):
    """Promote production-ready models to staging."""
    models_to_promote = [
        ("xgboost_passing_v5_integrated", "v5.0.0"),
        ("ensemble_v6_dirichlet_weights", "v6.1.0"),
        ("bayesian_state_space_v2", "v2.0.0"),
        ("bayesian_hierarchical_v2", "v2.0.0"),
        ("bayesian_informative_priors_v2", "v2.0.0"),
        ("bnn_rushing_v3_1", "v3.1.0"),
    ]

    print("Promoting models to staging...\n")

    for name, version in models_to_promote:
        try:
            registry.promote_model(name, version, "staging")
        except ValueError as e:
            print(f"⚠️  Could not promote {name} {version}: {e}")

    print("\n✓ Staging promotion complete!")


def promote_to_production(registry: ModelRegistry):
    """Promote validated models from staging to production."""
    # Only promote core ensemble v6 components that are in active use
    production_models = [
        ("ensemble_v6_dirichlet_weights", "v6.1.0"),
        ("bayesian_state_space_v2", "v2.0.0"),
        ("bayesian_hierarchical_v2", "v2.0.0"),
        ("bayesian_informative_priors_v2", "v2.0.0"),
        ("xgboost_passing_v5_integrated", "v5.0.0"),
    ]

    print("\nPromoting validated models to production...\n")

    for name, version in production_models:
        try:
            registry.promote_model(name, version, "prod")
        except ValueError as e:
            print(f"⚠️  Could not promote {name} {version}: {e}")

    print("\n✓ Production promotion complete!")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Promote models to staging or production")
    parser.add_argument(
        "--to-production",
        action="store_true",
        help="Promote to production (default: staging only)",
    )

    args = parser.parse_args()

    print("Initializing model registry...")
    registry = ModelRegistry(registry_path="models/registry.json")

    if args.to_production:
        print("\n🚀 PRODUCTION PROMOTION MODE")
        print("=" * 80)
        print("This will promote models directly to production.")
        print("Ensure all models have been validated in staging first!")
        print("=" * 80)
        response = input("\nContinue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborting.")
            return
        promote_to_production(registry)
    else:
        print("\n🔄 STAGING PROMOTION MODE")
        print("=" * 80)
        promote_to_staging(registry)

    # Generate quick summary
    print("\n" + "=" * 80)
    print("UPDATED STATUS")
    print("=" * 80)

    all_models = registry.list_models()
    status_counts = {"prod": 0, "staging": 0, "dev": 0}

    for model_name in all_models:
        versions = registry.list_versions(model_name)
        for version in versions:
            status_counts[version.status] += 1

    print(f"\n  Production:  {status_counts['prod']:3d}")
    print(f"  Staging:     {status_counts['staging']:3d}")
    print(f"  Development: {status_counts['dev']:3d}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
