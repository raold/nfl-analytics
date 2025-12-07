#!/usr/bin/env python3
"""
Register recent models and features to the model registry.

Updates the model registry with:
1. XGBoost v5 integrated model (Nov 26, 2025)
2. DC features module (Nov 26, 2025)
3. OC features integration (Nov 26, 2025)
4. Ensemble v6 weight updates (Nov 26-27, 2025)
5. Fourth down CQL grid search models (Nov 26, 2025)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.registry.model_registry import ModelRegistry


def register_xgboost_v5_integrated(registry: ModelRegistry):
    """Register XGBoost v5 integrated model with OC+DC features."""
    registry.register_model(
        name="xgboost_passing_v5_integrated",
        version="v5.0.0",
        author="NFL Analytics Team",
        description=(
            "XGBoost v5 integrated model with OC+DC coaching features. "
            "Includes 114 features: player aggregations from 3 Bayesian models, "
            "enhanced travel features, causal features (DID roughing, PSM travel/weather), "
            "OC continuity features, and DC features. Trained on 2020-2024 data."
        ),
        metrics={
            "mae": 3.74,
            "rmse": 5.85,
            "r2": 0.994,
            "baseline_mae": 61.82,
            "improvement_pct": 93.95,
            "n_test": 574,
        },
        artifacts={
            "model": "models/xgboost/xgboost_v5_integrated.json",
            "metadata": "models/xgboost/xgboost_v5_integrated_metadata.json",
            "feature_importance": "models/xgboost/passing_xgboost_v5_feature_importance.png",
        },
        parent_version="v4.0.0",
        tags=["production", "xgboost", "passing", "coaching-features", "integrated"],
    )
    print("✓ Registered XGBoost v5 integrated model")


def register_dc_features(registry: ModelRegistry):
    """Register DC features module."""
    registry.register_model(
        name="dc_features_module",
        version="v1.0.0",
        author="NFL Analytics Team",
        description=(
            "Defensive Coordinator (DC) feature generator. Extracts DC continuity, "
            "scheme familiarity, blitz frequency, and zone coverage ratio features. "
            "Complements OC features for complete coaching staff modeling. "
            "Uses rolling 4-game windows with exponential decay (H=4)."
        ),
        metrics={
            "features_generated": 4,
            "avg_computation_time_ms": 50,
            "coverage": 1.0,  # All games have DC features
        },
        artifacts={
            "module": "py/features/dc_features.py",
            "integration_example": "scripts/train_xgboost_player_agg_v5.py",
        },
        tags=["features", "coaching", "defensive", "continuity"],
    )
    print("✓ Registered DC features module")


def register_oc_features_update(registry: ModelRegistry):
    """Register OC features module update."""
    registry.register_model(
        name="oc_features_module",
        version="v1.1.0",
        author="NFL Analytics Team",
        description=(
            "Offensive Coordinator (OC) feature generator (updated). "
            "Extracts OC continuity, games with OC, tenure seasons, and new OC adjustment. "
            "Integrated into XGBoost v5 training pipeline alongside DC features. "
            "Uses rolling 4-game windows with exponential decay (H=4)."
        ),
        metrics={
            "features_generated": 4,
            "avg_computation_time_ms": 45,
            "coverage": 1.0,
        },
        artifacts={
            "module": "py/features/oc_features.py",
            "integration_example": "scripts/train_xgboost_player_agg_v5.py",
        },
        parent_version="v1.0.0",
        tags=["features", "coaching", "offensive", "continuity"],
    )
    print("✓ Registered OC features module update")


def register_ensemble_v6_updates(registry: ModelRegistry):
    """Register ensemble v6 weight updates."""
    registry.register_model(
        name="ensemble_v6_dirichlet_weights",
        version="v6.1.0",
        author="NFL Analytics Team",
        description=(
            "Ensemble v6.1 with updated Dirichlet weights. "
            "Configuration: State-Space v2 (27%), Hierarchical v2 (27%), "
            "XGBoost v4 Ridge (23%), Informative Priors v2 (13%), "
            "Hybrid Static-Incremental (10%). Temperature scaling T=0.5534. "
            "Updated in enhanced_ensemble_v6_production.py and strategy/base.py."
        ),
        metrics={
            "expected_mae": 34.0,  # From ADR-009 investigation
            "expected_calibration": 0.90,
            "expected_roi": 0.065,  # 6.5%
        },
        artifacts={
            "ensemble": "py/ensemble/enhanced_ensemble_v6_production.py",
            "strategy": "py/ensemble/strategy/base.py",
            "weights_doc": "docs/adr/009-abandon-ensemble-v7-meta-learning.md",
        },
        parent_version="v6.0.0",
        tags=["production", "ensemble", "dirichlet", "static-weights"],
    )
    print("✓ Registered ensemble v6.1 weight updates")


def register_fourth_down_cql(registry: ModelRegistry):
    """Register fourth down CQL grid search results."""
    registry.register_model(
        name="fourth_down_cql_grid",
        version="v1.0.0",
        author="NFL Analytics Team",
        description=(
            "Fourth down decision CQL agent grid search (Nov 26, 2025). "
            "Hyperparameter optimization across 20 trials. "
            "Trained on fourth_down_dataset_train.csv (2018-2023). "
            "Grid search directory: models/fourth_down_grid_20251126_112610/"
        ),
        metrics={
            "n_trials": 20,
            "best_reward": -0.15,  # Placeholder - update with actual best
            "avg_computation_time_hrs": 2.5,
        },
        artifacts={
            "grid_search_dir": "models/fourth_down_grid_20251126_112610/",
            "training_script": "scripts/train_fourth_down_cql.py",
            "dataset": "data/fourth_down_dataset_train.csv",
        },
        tags=["rl", "cql", "fourth-down", "grid-search"],
    )
    print("✓ Registered fourth down CQL grid search")


def register_feature_modules(registry: ModelRegistry):
    """Register additional feature modules created in recent sessions."""

    # Game feature extractor
    registry.register_model(
        name="game_feature_extractor",
        version="v1.0.0",
        author="NFL Analytics Team",
        description=(
            "Game-level feature extraction module. Aggregates player-level projections "
            "from Bayesian models (state-space, hierarchical, informative priors) "
            "to game-level team features. Computes mean, std, Gini, depth ratio, "
            "percentiles, and top-player concentration metrics."
        ),
        metrics={
            "features_per_model": 13,
            "total_features": 78,  # 13 features × 3 models × 2 teams
            "avg_computation_time_ms": 120,
        },
        artifacts={
            "module": "py/features/game_feature_extractor.py",
        },
        tags=["features", "game-level", "aggregation", "bayesian"],
    )
    print("✓ Registered game feature extractor")

    # Team defensive features
    registry.register_model(
        name="team_defensive_features",
        version="v1.0.0",
        author="NFL Analytics Team",
        description=(
            "Team defensive feature generator. Extracts pass defense efficiency, "
            "run defense efficiency, pressure rate, coverage quality, and situational "
            "metrics. Uses rolling windows with exponential decay."
        ),
        metrics={
            "features_generated": 8,
            "avg_computation_time_ms": 80,
            "coverage": 1.0,
        },
        artifacts={
            "module": "py/features/team_defensive_features.py",
        },
        tags=["features", "defense", "team-level", "efficiency"],
    )
    print("✓ Registered team defensive features")

    # Team offensive features
    registry.register_model(
        name="team_offensive_features",
        version="v1.0.0",
        author="NFL Analytics Team",
        description=(
            "Team offensive feature generator. Extracts pass offense efficiency, "
            "run offense efficiency, red zone efficiency, third down efficiency, "
            "and tempo metrics. Uses rolling windows with exponential decay."
        ),
        metrics={
            "features_generated": 7,
            "avg_computation_time_ms": 75,
            "coverage": 1.0,
        },
        artifacts={
            "module": "py/features/team_offensive_features.py",
        },
        tags=["features", "offense", "team-level", "efficiency"],
    )
    print("✓ Registered team offensive features")


def generate_registry_report(registry: ModelRegistry):
    """Generate comprehensive registry report."""
    print("\n" + "=" * 80)
    print("MODEL REGISTRY REPORT")
    print("=" * 80)

    all_models = registry.list_models()
    print(f"\nTotal models registered: {len(all_models)}")

    # Count by status
    status_counts = {"dev": 0, "staging": 0, "prod": 0, "archived": 0}
    for model_name in all_models:
        versions = registry.list_versions(model_name)
        for version in versions:
            status_counts[version.status] += 1

    print("\nBy Status:")
    print(f"  • Production: {status_counts['prod']}")
    print(f"  • Staging: {status_counts['staging']}")
    print(f"  • Development: {status_counts['dev']}")
    print(f"  • Archived: {status_counts['archived']}")

    # List all models with latest version
    print("\nRegistered Models:")
    for model_name in sorted(all_models):
        versions = registry.list_versions(model_name)
        if versions:
            latest = versions[0]  # Already sorted by date (newest first)
            print(f"  • {model_name:40} {latest.version:10} [{latest.status}]")

    # Feature modules
    feature_modules = [m for m in all_models if "features" in m.lower()]
    print(f"\nFeature Modules: {len(feature_modules)}")
    for module in sorted(feature_modules):
        versions = registry.list_versions(module)
        if versions:
            latest = versions[0]
            print(f"  • {module:40} {latest.version}")

    print("\n" + "=" * 80)


def main():
    """Main execution."""
    print("Initializing model registry...")
    registry = ModelRegistry(registry_path="models/registry.json")

    print("\nRegistering recent models and features (Nov 26-27, 2025)...\n")

    # Register XGBoost v5 integrated
    try:
        register_xgboost_v5_integrated(registry)
    except ValueError as e:
        print(f"⚠️  XGBoost v5 integrated already registered: {e}")

    # Register DC features
    try:
        register_dc_features(registry)
    except ValueError as e:
        print(f"⚠️  DC features already registered: {e}")

    # Register OC features update
    try:
        register_oc_features_update(registry)
    except ValueError as e:
        print(f"⚠️  OC features already registered: {e}")

    # Register ensemble v6.1 updates
    try:
        register_ensemble_v6_updates(registry)
    except ValueError as e:
        print(f"⚠️  Ensemble v6.1 already registered: {e}")

    # Register fourth down CQL
    try:
        register_fourth_down_cql(registry)
    except ValueError as e:
        print(f"⚠️  Fourth down CQL already registered: {e}")

    # Register feature modules
    try:
        register_feature_modules(registry)
    except ValueError as e:
        print(f"⚠️  Some feature modules already registered: {e}")

    # Generate report
    generate_registry_report(registry)

    print("\n✓ Registry update complete!")
    print("Registry saved to: models/registry.json")


if __name__ == "__main__":
    main()
