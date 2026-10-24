#!/usr/bin/env python3
"""
Generate comprehensive model registry report.

Outputs:
1. Summary statistics (total models, by type, by status)
2. Model lineage (parent-child relationships)
3. Production-ready models
4. Pending actions (models to promote)
5. Registry health status
"""

import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nfl_analytics.registry.model_registry import ModelRegistry


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print subsection header."""
    print(f"\n{title}")
    print("-" * 80)


def generate_summary_stats(registry: ModelRegistry):
    """Generate summary statistics."""
    print_header("MODEL REGISTRY SUMMARY")

    all_models = registry.list_models()
    print(f"\n📊 Total Models Registered: {len(all_models)}")

    # Count by status
    status_counts = defaultdict(int)
    type_counts = defaultdict(int)

    for model_name in all_models:
        versions = registry.list_versions(model_name)
        for version in versions:
            status_counts[version.status] += 1

            # Categorize by type
            if "xgboost" in model_name.lower():
                type_counts["XGBoost"] += 1
            elif "bayesian" in model_name.lower() or "bnn" in model_name.lower():
                type_counts["Bayesian"] += 1
            elif "ensemble" in model_name.lower():
                type_counts["Ensemble"] += 1
            elif "features" in model_name.lower():
                type_counts["Features"] += 1
            elif "cql" in model_name.lower() or "rl" in version.tags:
                type_counts["Reinforcement Learning"] += 1
            else:
                type_counts["Other"] += 1

    print_subheader("By Status")
    print(f"  🚀 Production:   {status_counts['prod']:3d}")
    print(f"  🔄 Staging:      {status_counts['staging']:3d}")
    print(f"  🛠️  Development:  {status_counts['dev']:3d}")
    print(f"  📦 Archived:     {status_counts['archived']:3d}")

    print_subheader("By Type")
    for model_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  • {model_type:25} {count:3d}")

    return all_models


def list_all_models(registry: ModelRegistry, all_models: list[str]):
    """List all registered models with details."""
    print_header("REGISTERED MODELS")

    # Group by type
    bayesian = []
    xgboost = []
    ensemble = []
    features = []
    rl = []
    other = []

    for model_name in sorted(all_models):
        versions = registry.list_versions(model_name)
        if not versions:
            continue

        latest = versions[0]
        entry = (model_name, latest.version, latest.status, latest.created_at[:10])

        if "bayesian" in model_name.lower() or "bnn" in model_name.lower():
            bayesian.append(entry)
        elif "xgboost" in model_name.lower():
            xgboost.append(entry)
        elif "ensemble" in model_name.lower():
            ensemble.append(entry)
        elif "features" in model_name.lower():
            features.append(entry)
        elif "cql" in model_name.lower() or "rl" in latest.tags:
            rl.append(entry)
        else:
            other.append(entry)

    def print_model_group(title: str, models: list):
        if not models:
            return
        print_subheader(title)
        for name, version, status, date in models:
            status_icon = {"prod": "🟢", "staging": "🟡", "dev": "🔵", "archived": "⚫"}.get(
                status, "⚪"
            )
            print(f"  {status_icon} {name:45} {version:10} [{status:8}] {date}")

    print_model_group("Bayesian Models", bayesian)
    print_model_group("XGBoost Models", xgboost)
    print_model_group("Ensemble Models", ensemble)
    print_model_group("Reinforcement Learning", rl)
    print_model_group("Feature Modules", features)
    print_model_group("Other Models", other)


def analyze_lineage(registry: ModelRegistry, all_models: list[str]):
    """Analyze model lineage (parent-child relationships)."""
    print_header("MODEL LINEAGE")

    has_lineage = False
    for model_name in sorted(all_models):
        versions = registry.list_versions(model_name)
        if not versions:
            continue

        # Check if any version has a parent
        lineage_versions = [v for v in versions if v.parent_version]
        if lineage_versions:
            has_lineage = True
            print_subheader(f"{model_name}")
            for version in lineage_versions:
                print(f"  {version.version} ← {version.parent_version}")

    if not has_lineage:
        print("\n  No parent-child lineage relationships found.")
        print("  (All models are either v1.0.0 or independent versions)")


def identify_pending_actions(registry: ModelRegistry, all_models: list[str]):
    """Identify models that should be promoted to production."""
    print_header("PENDING ACTIONS")

    # Models tagged "production" but still in dev status
    candidates = []

    for model_name in all_models:
        versions = registry.list_versions(model_name, status="dev")
        for version in versions:
            if "production" in version.tags:
                candidates.append((model_name, version.version, version.description))

    if candidates:
        print_subheader("Models Ready for Production Promotion")
        print("\nThe following models are tagged 'production' but still in 'dev' status:")
        for name, version, desc in candidates:
            print(f"\n  📌 {name} {version}")
            print(f"     {desc[:100]}...")
            print(f"     Action: registry.promote_model('{name}', '{version}', 'staging')")
    else:
        print("\n✓ No pending promotion actions identified.")


def check_registry_health(registry: ModelRegistry, all_models: list[str]):
    """Check registry health and identify issues."""
    print_header("REGISTRY HEALTH CHECK")

    issues = []
    warnings = []

    # Check 1: At least one production model
    prod_models = []
    for model_name in all_models:
        prod_version = registry.get_production_model(model_name)
        if prod_version:
            prod_models.append(model_name)

    if not prod_models:
        issues.append("❌ No models in production status")
    else:
        print(f"✓ {len(prod_models)} model(s) in production")

    # Check 2: Essential components registered
    essential = [
        "xgboost",
        "bayesian_state_space",
        "bayesian_hierarchical",
        "ensemble",
    ]
    for component in essential:
        found = any(component in m.lower() for m in all_models)
        if found:
            print(f"✓ {component.replace('_', ' ').title()} model registered")
        else:
            warnings.append(f"⚠️  Missing {component.replace('_', ' ').title()} model")

    # Check 3: Metadata completeness
    incomplete_metadata = []
    for model_name in all_models:
        versions = registry.list_versions(model_name)
        for version in versions:
            if not version.metrics:
                incomplete_metadata.append(f"{model_name} {version.version}")

    if incomplete_metadata:
        warnings.append(f"⚠️  {len(incomplete_metadata)} version(s) missing metrics")

    # Print warnings and issues
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  {issue}")

    if not issues and not warnings:
        print("\n✅ Registry health: EXCELLENT")
        print("   All essential components registered with complete metadata.")
    elif issues:
        print("\n🔴 Registry health: NEEDS ATTENTION")
    else:
        print("\n🟡 Registry health: GOOD")


def main():
    """Main execution."""
    print("\nGenerating Model Registry Report...")
    print("Registry: models/registry.json")

    registry = ModelRegistry(registry_path="models/registry.json")

    # Generate sections
    all_models = generate_summary_stats(registry)
    list_all_models(registry, all_models)
    analyze_lineage(registry, all_models)
    identify_pending_actions(registry, all_models)
    check_registry_health(registry, all_models)

    print("\n" + "=" * 80)
    print("Report generation complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
