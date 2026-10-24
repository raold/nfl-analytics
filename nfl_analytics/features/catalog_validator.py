"""
Feature Catalog Validator

Runtime validation that ensures only asof_safe features are used in
production prediction pipelines. Integrates with training scripts and
model deployment to enforce temporal safety guarantees.

Usage:
    from py.features.catalog_validator import validate_feature_list

    # In your training script:
    features_to_use = ['prior_epa_mean', 'rest_days', 'home_score']  # ❌ home_score!
    validate_feature_list(features_to_use)  # Raises error

    # OR as a decorator:
    @validate_training_features
    def train_model(X, y):
        ...

Author: Claude Code
Date: 2025-10-17
"""

from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path

import pandas as pd
import yaml


class FeatureCatalog:
    """Feature catalog loader and validator."""

    _instance = None
    _catalog = None

    def __new__(cls):
        """Singleton pattern to avoid reloading catalog."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Load catalog on first instantiation."""
        if self._catalog is None:
            self._load_catalog()

    def _load_catalog(self) -> None:
        """Load feature catalog YAML."""
        # Try multiple potential locations
        possible_paths = [
            Path(__file__).parent / "catalog.yaml",
            Path.cwd() / "py/features/catalog.yaml",
            Path(__file__).parent.parent.parent / "py/features/catalog.yaml",
        ]

        for catalog_path in possible_paths:
            if catalog_path.exists():
                with open(catalog_path) as f:
                    self._catalog = yaml.safe_load(f)
                return

        raise FileNotFoundError(
            "Feature catalog not found. Searched:\n" + "\n".join(str(p) for p in possible_paths)
        )

    def get_safe_features(self) -> set[str]:
        """Get all features marked as asof_safe:true."""
        safe_features = set()

        for group in self._catalog.get("feature_groups", []):
            if group.get("asof_safe", False):
                for feature in group.get("features", []):
                    safe_features.add(feature["name"])

        return safe_features

    def get_unsafe_features(self) -> set[str]:
        """Get all features marked as asof_safe:false."""
        unsafe_features = set()

        for group in self._catalog.get("feature_groups", []):
            if not group.get("asof_safe", True):
                for feature in group.get("features", []):
                    unsafe_features.add(feature["name"])

        return unsafe_features

    def get_all_documented_features(self) -> set[str]:
        """Get all features documented in catalog (safe or unsafe)."""
        all_features = set()

        for group in self._catalog.get("feature_groups", []):
            for feature in group.get("features", []):
                all_features.add(feature["name"])

        return all_features


class FeatureValidationError(ValueError):
    """Raised when unsafe features are detected in prediction pipeline."""

    pass


def validate_feature_list(
    features: Iterable[str],
    allow_undocumented: bool = False,
    strict: bool = True,
) -> None:
    """
    Validate that a feature list contains only asof_safe features.

    Args:
        features: List of feature names to validate
        allow_undocumented: If True, allow features not in catalog (with warning)
        strict: If True, raise error on unsafe features. If False, only warn.

    Raises:
        FeatureValidationError: If unsafe features detected and strict=True

    Example:
        >>> validate_feature_list(['prior_epa_mean', 'rest_days'])  # OK
        >>> validate_feature_list(['prior_epa_mean', 'home_score'])  # Error!
    """
    catalog = FeatureCatalog()
    safe_features = catalog.get_safe_features()
    unsafe_features = catalog.get_unsafe_features()
    documented_features = catalog.get_all_documented_features()

    features_set = set(features)
    unsafe_detected = features_set & unsafe_features
    undocumented = features_set - documented_features

    # Check for unsafe features
    if unsafe_detected:
        error_msg = (
            "❌ TEMPORAL LEAKAGE DETECTED!\n\n"
            "The following features are marked asof_safe:false and MUST NOT be "
            "used in prediction:\n"
        )
        for feat in sorted(unsafe_detected):
            error_msg += f"  - {feat}\n"

        error_msg += (
            "\nThese are post-game outcome variables that would cause leakage.\n"
            "Remove them from your feature list before training.\n"
        )

        if strict:
            raise FeatureValidationError(error_msg)
        else:
            print(f"⚠️  WARNING: {error_msg}")

    # Check for undocumented features
    if undocumented and not allow_undocumented:
        warning_msg = (
            "⚠️  UNDOCUMENTED FEATURES DETECTED:\n\n"
            "The following features are not in catalog.yaml:\n"
        )
        for feat in sorted(undocumented):
            warning_msg += f"  - {feat}\n"

        warning_msg += (
            "\nThese features have not been audited for temporal safety.\n"
            "Add them to py/features/catalog.yaml before using in production.\n"
        )

        print(warning_msg)
        if strict:
            raise FeatureValidationError(
                "Undocumented features detected. Set allow_undocumented=True to bypass."
            )

    # Success message
    n_safe = len(features_set & safe_features)
    if n_safe > 0:
        print(f"✅ Feature validation passed: {n_safe} safe features, 0 unsafe")


def validate_dataframe_columns(
    df: pd.DataFrame,
    allow_undocumented: bool = False,
    strict: bool = True,
) -> None:
    """
    Validate that a DataFrame contains only asof_safe feature columns.

    Args:
        df: DataFrame to validate
        allow_undocumented: Allow columns not in catalog
        strict: Raise error on unsafe features

    Example:
        >>> df = pd.read_csv('features.csv')
        >>> validate_dataframe_columns(df)
    """
    # Exclude common non-feature columns
    exclude_cols = {
        "game_id",
        "season",
        "week",
        "kickoff",
        "home_team",
        "away_team",
        "spread_close",
        "total_close",
    }

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    validate_feature_list(feature_cols, allow_undocumented=allow_undocumented, strict=strict)


def validate_training_features(func: Callable) -> Callable:
    """
    Decorator to validate features before training.

    Assumes function receives DataFrame as first argument (X).

    Example:
        @validate_training_features
        def train_model(X, y):
            model.fit(X, y)
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to extract DataFrame from args or kwargs
        X = None
        if args and isinstance(args[0], pd.DataFrame):
            X = args[0]
        elif "X" in kwargs and isinstance(kwargs["X"], pd.DataFrame):
            X = kwargs["X"]

        if X is not None:
            print("🔒 Validating features before training...")
            validate_dataframe_columns(X, allow_undocumented=True, strict=True)

        return func(*args, **kwargs)

    return wrapper


def get_safe_feature_subset(features: Iterable[str]) -> list[str]:
    """
    Filter a feature list to include only asof_safe features.

    Args:
        features: Input feature list

    Returns:
        Subset of features that are asof_safe:true

    Example:
        >>> all_features = ['prior_epa_mean', 'home_score', 'rest_days']
        >>> safe = get_safe_feature_subset(all_features)
        >>> print(safe)
        ['prior_epa_mean', 'rest_days']
    """
    catalog = FeatureCatalog()
    safe_features = catalog.get_safe_features()

    return [f for f in features if f in safe_features]


def print_feature_safety_report(features: Iterable[str]) -> None:
    """
    Print a detailed report on feature safety status.

    Args:
        features: Feature list to analyze
    """
    catalog = FeatureCatalog()
    safe_features = catalog.get_safe_features()
    unsafe_features = catalog.get_unsafe_features()
    documented_features = catalog.get_all_documented_features()

    features_set = set(features)

    safe_count = len(features_set & safe_features)
    unsafe_count = len(features_set & unsafe_features)
    undocumented_count = len(features_set - documented_features)

    print("\n" + "=" * 60)
    print("FEATURE SAFETY REPORT")
    print("=" * 60)
    print(f"Total features:        {len(features_set)}")
    print(f"✅ Safe (asof_safe):    {safe_count}")
    print(f"❌ Unsafe (leakage):    {unsafe_count}")
    print(f"⚠️  Undocumented:        {undocumented_count}")
    print("=" * 60)

    if unsafe_count > 0:
        print("\n❌ UNSAFE FEATURES (DO NOT USE IN PREDICTION):")
        for feat in sorted(features_set & unsafe_features):
            print(f"  - {feat}")

    if undocumented_count > 0:
        print("\n⚠️  UNDOCUMENTED FEATURES (NOT AUDITED):")
        for feat in sorted(features_set - documented_features):
            print(f"  - {feat}")

    print()


# CLI interface for ad-hoc validation
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate feature list against catalog")
    parser.add_argument("features", nargs="+", help="Feature names to validate")
    parser.add_argument(
        "--allow-undocumented",
        action="store_true",
        help="Allow features not in catalog",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Warn instead of error on unsafe features",
    )

    args = parser.parse_args()

    try:
        validate_feature_list(
            args.features,
            allow_undocumented=args.allow_undocumented,
            strict=not args.no_strict,
        )
        print("\n✅ All features are safe for prediction!")
    except FeatureValidationError as e:
        print(f"\n{e}")
        exit(1)
