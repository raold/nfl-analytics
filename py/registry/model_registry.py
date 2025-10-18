"""
Model Registry for NFL Analytics.

Centralized model versioning, promotion, and comparison system.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelVersion:
    """Model version metadata with semantic versioning."""

    name: str  # Model name (e.g., "glm_baseline", "dqn_rl")
    version: str  # Semantic version (e.g., "v1.2.3")
    created_at: str  # ISO format datetime
    author: str  # Creator name
    description: str  # Model description
    metrics: dict[str, float]  # Performance metrics
    artifacts: dict[str, str]  # Artifact paths
    parent_version: str | None = None  # Parent version for lineage
    status: str = "dev"  # dev/staging/prod/archived
    tags: list[str] = None  # Optional tags

    def __post_init__(self):
        """Validate version after initialization."""
        if self.tags is None:
            self.tags = []

        # Validate status
        valid_statuses = ["dev", "staging", "prod", "archived"]
        if self.status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")

        # Validate version format
        if not self.version.startswith("v"):
            raise ValueError("Version must start with 'v' (e.g., v1.0.0)")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """Central model registry with version control."""

    def __init__(self, registry_path: str = "models/registry.json"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registry or create new
        self.models = self._load_registry()

    def _load_registry(self) -> dict[str, list[ModelVersion]]:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)

            # Convert dicts back to ModelVersion objects
            registry = {}
            for model_name, versions in data.items():
                registry[model_name] = [ModelVersion.from_dict(v) for v in versions]

            return registry
        else:
            return {}

    def _save_registry(self):
        """Save registry to disk."""
        # Convert ModelVersion objects to dicts
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = [v.to_dict() for v in versions]

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        name: str,
        version: str,
        author: str,
        description: str,
        metrics: dict[str, float],
        artifacts: dict[str, str],
        parent_version: str | None = None,
        tags: list[str] | None = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model name
            version: Semantic version
            author: Creator name
            description: Model description
            metrics: Performance metrics
            artifacts: Paths to model artifacts
            parent_version: Parent version for lineage
            tags: Optional tags

        Returns:
            Registered ModelVersion
        """
        # Create model version
        model = ModelVersion(
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            author=author,
            description=description,
            metrics=metrics,
            artifacts=artifacts,
            parent_version=parent_version,
            status="dev",
            tags=tags or [],
        )

        # Add to registry
        if name not in self.models:
            self.models[name] = []

        # Check for duplicate versions
        existing_versions = [v.version for v in self.models[name]]
        if version in existing_versions:
            raise ValueError(f"Version {version} already exists for {name}")

        self.models[name].append(model)
        self._save_registry()

        print(f"✓ Registered {name} {version}")
        return model

    def promote_model(self, name: str, version: str, new_status: str) -> ModelVersion:
        """
        Promote model to new status.

        Args:
            name: Model name
            version: Version to promote
            new_status: New status (staging/prod)

        Returns:
            Updated ModelVersion
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found")

        # Find version
        model = None
        for v in self.models[name]:
            if v.version == version:
                model = v
                break

        if model is None:
            raise ValueError(f"Version {version} not found for {name}")

        # Validate promotion path
        valid_transitions = {
            "dev": ["staging", "archived"],
            "staging": ["prod", "archived"],
            "prod": ["archived"],
        }

        if new_status not in valid_transitions.get(model.status, []):
            raise ValueError(f"Cannot promote from {model.status} to {new_status}")

        # If promoting to prod, demote current prod
        if new_status == "prod":
            for v in self.models[name]:
                if v.status == "prod":
                    v.status = "archived"
                    print(f"  Archived previous prod: {v.version}")

        model.status = new_status
        self._save_registry()

        print(f"✓ Promoted {name} {version} to {new_status}")
        return model

    def rollback_model(self, name: str, to_version: str) -> ModelVersion:
        """
        Rollback to a previous version.

        Args:
            name: Model name
            to_version: Version to rollback to

        Returns:
            Promoted ModelVersion
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found")

        # Find target version
        target = None
        for v in self.models[name]:
            if v.version == to_version:
                target = v
                break

        if target is None:
            raise ValueError(f"Version {to_version} not found")

        if target.status == "archived":
            # Restore from archive and promote to prod
            target.status = "staging"
            self._save_registry()
            return self.promote_model(name, to_version, "prod")
        elif target.status in ["staging", "dev"]:
            return self.promote_model(name, to_version, "prod")
        else:
            print(f"  {name} {to_version} already in production")
            return target

    def compare_versions(self, name: str, version1: str, version2: str) -> dict[str, Any]:
        """
        Compare two model versions.

        Args:
            name: Model name
            version1: First version
            version2: Second version

        Returns:
            Comparison dictionary
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found")

        # Find versions
        v1 = v2 = None
        for v in self.models[name]:
            if v.version == version1:
                v1 = v
            if v.version == version2:
                v2 = v

        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")

        # Compare metrics
        metric_comparison = {}
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())

        for metric in all_metrics:
            val1 = v1.metrics.get(metric, None)
            val2 = v2.metrics.get(metric, None)

            if val1 is not None and val2 is not None:
                delta = val2 - val1
                pct_change = (delta / val1 * 100) if val1 != 0 else 0
                metric_comparison[metric] = {
                    version1: val1,
                    version2: val2,
                    "delta": delta,
                    "pct_change": pct_change,
                }

        return {
            "model_name": name,
            "version1": version1,
            "version2": version2,
            "metric_comparison": metric_comparison,
            "v1_status": v1.status,
            "v2_status": v2.status,
            "v1_created": v1.created_at,
            "v2_created": v2.created_at,
        }

    def get_production_model(self, name: str) -> ModelVersion | None:
        """
        Get current production model.

        Args:
            name: Model name

        Returns:
            Production ModelVersion or None
        """
        if name not in self.models:
            return None

        for v in self.models[name]:
            if v.status == "prod":
                return v

        return None

    def get_version(self, name: str, version: str) -> ModelVersion | None:
        """Get specific model version."""
        if name not in self.models:
            return None

        for v in self.models[name]:
            if v.version == version:
                return v

        return None

    def list_models(self) -> list[str]:
        """List all model names."""
        return list(self.models.keys())

    def list_versions(self, name: str, status: str | None = None) -> list[ModelVersion]:
        """
        List versions for a model.

        Args:
            name: Model name
            status: Optional status filter

        Returns:
            List of ModelVersion objects
        """
        if name not in self.models:
            return []

        versions = self.models[name]

        if status:
            versions = [v for v in versions if v.status == status]

        # Sort by creation time (newest first)
        versions = sorted(versions, key=lambda v: v.created_at, reverse=True)

        return versions

    def get_lineage(self, name: str, version: str) -> list[ModelVersion]:
        """
        Get version lineage (ancestry chain).

        Args:
            name: Model name
            version: Starting version

        Returns:
            List of ModelVersion objects in lineage
        """
        if name not in self.models:
            return []

        lineage = []
        current_version = version

        while current_version:
            model = self.get_version(name, current_version)
            if model is None:
                break

            lineage.append(model)
            current_version = model.parent_version

        return lineage

    def export_model_card(self, name: str, version: str, output_path: Path):
        """
        Export model card (documentation).

        Args:
            name: Model name
            version: Version
            output_path: Output markdown path
        """
        model = self.get_version(name, version)
        if model is None:
            raise ValueError(f"Model {name} {version} not found")

        lines = [
            f"# Model Card: {name} {version}",
            "",
            f"**Status:** {model.status}",
            f"**Created:** {model.created_at}",
            f"**Author:** {model.author}",
            "",
            "## Description",
            model.description,
            "",
            "## Metrics",
            "",
        ]

        for metric, value in sorted(model.metrics.items()):
            lines.append(f"- **{metric}**: {value:.4f}")

        lines.extend(
            [
                "",
                "## Artifacts",
                "",
            ]
        )

        for artifact_type, path in sorted(model.artifacts.items()):
            lines.append(f"- **{artifact_type}**: `{path}`")

        if model.parent_version:
            lines.extend(
                [
                    "",
                    "## Lineage",
                    f"- Parent: {model.parent_version}",
                ]
            )

        if model.tags:
            lines.extend(
                [
                    "",
                    "## Tags",
                    "",
                ]
            )
            for tag in model.tags:
                lines.append(f"- {tag}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        print(f"✓ Exported model card: {output_path}")


def main():
    """Main execution for demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description="NFL Analytics Model Registry")
    parser.add_argument(
        "command",
        choices=["register", "promote", "rollback", "compare", "list", "export"],
        help="Registry command",
    )
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--version", type=str, help="Model version")
    parser.add_argument("--author", type=str, default="NFL Analytics Team")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--status", type=str, help="Target status")
    parser.add_argument("--version2", type=str, help="Second version for comparison")
    parser.add_argument("--output", type=Path, help="Output path")

    args = parser.parse_args()

    # Initialize registry
    registry = ModelRegistry()

    if args.command == "register":
        # Example registration
        registry.register_model(
            name=args.name,
            version=args.version,
            author=args.author,
            description=args.description,
            metrics={"accuracy": 0.65, "brier": 0.22, "roi": 0.03},
            artifacts={"model": f"models/{args.name}_{args.version}.pkl"},
            tags=["baseline"],
        )

    elif args.command == "promote":
        registry.promote_model(args.name, args.version, args.status)

    elif args.command == "rollback":
        registry.rollback_model(args.name, args.version)

    elif args.command == "compare":
        comparison = registry.compare_versions(args.name, args.version, args.version2)
        print(json.dumps(comparison, indent=2))

    elif args.command == "list":
        if args.name:
            versions = registry.list_versions(args.name, status=args.status)
            for v in versions:
                print(f"{v.version} ({v.status}) - {v.created_at}")
        else:
            models = registry.list_models()
            for m in models:
                print(m)

    elif args.command == "export":
        registry.export_model_card(args.name, args.version, args.output)


if __name__ == "__main__":
    main()
