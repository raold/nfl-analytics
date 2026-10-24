"""Feature weighting using Bayesian shrinkage and Dirichlet normalization.

This module provides intelligent feature weighting based on empirical importance
(SHAP values) rather than binary include/exclude decisions.

Two approaches:
1. Dirichlet Weighting: Normalize SHAP importance to sum to 1, treating features
   as components of a mixture model where high-signal features dominate.

2. Bayesian Shrinkage: Apply soft thresholding where low-importance features are
   shrunk toward zero but not eliminated, preserving weak signals.

Usage:
    from nfl_analytics.features.feature_weighting import FeatureWeighter

    weighter = FeatureWeighter.from_shap_csv("data/feature_importance_shap.csv")
    X_weighted = weighter.transform(X, method="bayesian_shrinkage")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from scipy.special import softmax


class FeatureWeighter:
    """Feature weighting based on SHAP importance.

    Provides methods to weight features proportionally to their importance,
    reducing noise from low-signal features without eliminating them entirely.
    """

    def __init__(self, feature_importance: pd.DataFrame, importance_col: str = "combined_shap"):
        """Initialize with feature importance data.

        Args:
            feature_importance: DataFrame with 'feature' and importance columns
            importance_col: Column name containing importance scores
        """
        self.importance_df = feature_importance.copy()
        self.importance_col = importance_col

        # Create feature -> importance mapping
        self.importance_map = dict(
            zip(self.importance_df["feature"], self.importance_df[importance_col])
        )

        # Precompute weights using different methods
        self._precompute_weights()

    @classmethod
    def from_shap_csv(cls, csv_path: str, importance_col: str = "combined_shap"):
        """Load feature importance from SHAP CSV file.

        Args:
            csv_path: Path to feature_importance_shap.csv
            importance_col: Column to use for importance

        Returns:
            FeatureWeighter instance
        """
        df = pd.read_csv(csv_path)
        return cls(df, importance_col)

    def _precompute_weights(self):
        """Precompute weights using different weighting methods."""
        importances = self.importance_df[self.importance_col].values
        features = self.importance_df["feature"].values

        # 1. Dirichlet-style normalization (sum to 1)
        # Add small epsilon to avoid division by zero
        dirichlet_weights = importances / (importances.sum() + 1e-10)

        # 2. Softmax normalization (smoother distribution)
        # Temperature controls sharpness: lower = sharper distinction
        temperature = 1.0
        softmax_weights = softmax(importances / temperature)

        # 3. Bayesian shrinkage (horseshoe-inspired)
        # Features below median get shrunk, above median get boosted
        median_imp = np.median(importances)
        std_imp = np.std(importances) + 1e-10

        # Shrinkage factor: sigmoid of z-score
        z_scores = (importances - median_imp) / std_imp
        shrinkage_weights = 1 / (1 + np.exp(-z_scores))  # Sigmoid

        # 4. Power-law weighting (amplify differences)
        # w_i = imp_i^alpha / sum(imp_j^alpha)
        alpha = 0.5  # Square root dampens extreme values
        power_weights = np.power(importances + 1e-10, alpha)
        power_weights = power_weights / power_weights.sum()

        # 5. Rank-based weighting (robust to outliers)
        ranks = self.importance_df[self.importance_col].rank(ascending=True)
        rank_weights = ranks.values / ranks.sum()

        # Store all weight types
        self.weights = {
            "dirichlet": dict(zip(features, dirichlet_weights)),
            "softmax": dict(zip(features, softmax_weights)),
            "bayesian_shrinkage": dict(zip(features, shrinkage_weights)),
            "power_law": dict(zip(features, power_weights)),
            "rank_based": dict(zip(features, rank_weights)),
        }

        # Store summary statistics
        self.n_features = len(features)
        self.median_importance = median_imp
        self.std_importance = std_imp

    def get_weights(
        self,
        feature_names: list[str],
        method: Literal[
            "dirichlet", "softmax", "bayesian_shrinkage", "power_law", "rank_based"
        ] = "bayesian_shrinkage",
    ) -> np.ndarray:
        """Get weights for specified features.

        Args:
            feature_names: List of feature names
            method: Weighting method to use

        Returns:
            Array of weights in same order as feature_names
        """
        weight_map = self.weights[method]

        weights = []
        for feat in feature_names:
            if feat in weight_map:
                weights.append(weight_map[feat])
            else:
                # Unknown feature gets median weight
                median_weight = np.median(list(weight_map.values()))
                weights.append(median_weight)

        return np.array(weights)

    def transform(
        self,
        X: pd.DataFrame,
        method: Literal[
            "dirichlet", "softmax", "bayesian_shrinkage", "power_law", "rank_based"
        ] = "bayesian_shrinkage",
        scale_factor: float = 1.0,
    ) -> pd.DataFrame:
        """Apply feature weighting to feature matrix.

        Args:
            X: Feature DataFrame
            method: Weighting method
            scale_factor: Multiplier for weights (higher = more aggressive weighting)

        Returns:
            Weighted feature DataFrame
        """
        weights = self.get_weights(X.columns.tolist(), method)

        # Scale weights: 1.0 = no change, >1 = amplify differences
        if scale_factor != 1.0:
            # Center weights around 1, then apply scale
            centered = weights - weights.mean()
            weights = 1.0 + (centered * scale_factor)
            # Clip to prevent negative weights
            weights = np.clip(weights, 0.01, None)

        # Apply weights as column multipliers
        X_weighted = X.copy()
        for i, col in enumerate(X.columns):
            X_weighted[col] = X[col] * weights[i]

        return X_weighted

    def get_feature_tiers(self, n_tiers: int = 4) -> dict[str, list[str]]:
        """Group features into importance tiers.

        Args:
            n_tiers: Number of tiers (default 4: elite, high, medium, low)

        Returns:
            Dict mapping tier name to list of features (elite = highest importance)
        """
        df = self.importance_df.copy()
        # Sort by importance descending so elite gets highest values
        df = df.sort_values(self.importance_col, ascending=False).reset_index(drop=True)

        tier_names = ["elite", "high", "medium", "low"][:n_tiers]
        features_per_tier = len(df) // n_tiers

        tiers = {}
        for i, tier_name in enumerate(tier_names):
            start_idx = i * features_per_tier
            end_idx = start_idx + features_per_tier if i < n_tiers - 1 else len(df)
            tier_features = df.iloc[start_idx:end_idx]["feature"].tolist()
            tiers[tier_name] = tier_features

        return tiers

    def get_summary(self) -> dict:
        """Get summary statistics about feature importance distribution.

        Returns:
            Dict with distribution statistics
        """
        importances = self.importance_df[self.importance_col].values

        return {
            "n_features": self.n_features,
            "median_importance": self.median_importance,
            "std_importance": self.std_importance,
            "max_importance": importances.max(),
            "min_importance": importances.min(),
            "n_zero_importance": (importances < 1e-6).sum(),
            "n_above_median": (importances > self.median_importance).sum(),
            "top_10_pct_share": np.sort(importances)[-int(self.n_features * 0.1) :].sum()
            / importances.sum(),
            "bottom_50_pct_share": np.sort(importances)[: int(self.n_features * 0.5)].sum()
            / importances.sum(),
        }

    def get_pruning_candidates(
        self, threshold: float = 0.001, min_keep: int = 100
    ) -> tuple[list[str], list[str]]:
        """Identify features that could be pruned vs kept.

        Args:
            threshold: Importance threshold below which features are candidates
            min_keep: Minimum number of features to always keep

        Returns:
            Tuple of (features_to_keep, pruning_candidates)
        """
        df = self.importance_df.sort_values(self.importance_col, ascending=False)

        keep = []
        prune = []

        for i, row in df.iterrows():
            if len(keep) < min_keep or row[self.importance_col] >= threshold:
                keep.append(row["feature"])
            else:
                prune.append(row["feature"])

        return keep, prune


class BayesianFeatureShrinker:
    """Bayesian shrinkage for features using horseshoe-like priors.

    Implements soft thresholding where:
    - High-importance features get weight ~1 (no shrinkage)
    - Low-importance features get weight ~0 (strong shrinkage)
    - Medium-importance features get proportional shrinkage

    This is inspired by the horseshoe prior which has:
    - Mass near zero (for noise features)
    - Heavy tails (for signal features)
    """

    def __init__(
        self,
        feature_importance: pd.DataFrame,
        importance_col: str = "combined_shap",
        global_shrinkage: float = 0.3,
        local_shrinkage: bool = True,
    ):
        """Initialize Bayesian shrinker.

        Args:
            feature_importance: DataFrame with importance scores
            importance_col: Column to use
            global_shrinkage: Global shrinkage intensity (0-1, higher = more shrinkage)
                             Default 0.3 provides moderate shrinkage while preserving signal
            local_shrinkage: Whether to use feature-specific shrinkage
        """
        self.importance_df = feature_importance.copy()
        self.importance_col = importance_col
        self.global_shrinkage = global_shrinkage
        self.local_shrinkage = local_shrinkage

        self._compute_shrinkage_factors()

    def _compute_shrinkage_factors(self):
        """Compute horseshoe-inspired shrinkage factors."""
        importances = self.importance_df[self.importance_col].values
        features = self.importance_df["feature"].values

        # Normalize importances to [0, 1] range
        imp_min = importances.min()
        imp_max = importances.max()
        imp_normalized = (importances - imp_min) / (imp_max - imp_min + 1e-10)

        if self.local_shrinkage:
            # Local shrinkage using percentile-based approach
            # Features in top percentiles get higher kappa (less shrinkage)
            # Features in bottom percentiles get lower kappa (more shrinkage)
            percentiles = np.argsort(np.argsort(importances)) / len(importances)

            # Sigmoid mapping: percentile -> shrinkage factor
            # Centered at 50th percentile, steepness controlled by temperature
            temperature = 0.15  # Lower = sharper cutoff
            kappa = 1 / (1 + np.exp(-(percentiles - 0.5) / temperature))

            # Apply global shrinkage as a floor adjustment
            kappa = kappa * (1 - self.global_shrinkage) + self.global_shrinkage * percentiles
        else:
            # Uniform shrinkage based on global parameter
            kappa = np.ones_like(importances) * (1 - self.global_shrinkage)

        # Ensure minimum shrinkage factor for stability
        kappa = np.clip(kappa, 0.05, 1.0)

        self.shrinkage_factors = dict(zip(features, kappa))
        self.feature_names = features

    def get_shrinkage_factors(self, feature_names: list[str]) -> np.ndarray:
        """Get shrinkage factors for specified features.

        Args:
            feature_names: List of feature names

        Returns:
            Array of shrinkage factors (0-1)
        """
        factors = []
        default_factor = np.median(list(self.shrinkage_factors.values()))

        for feat in feature_names:
            factors.append(self.shrinkage_factors.get(feat, default_factor))

        return np.array(factors)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply Bayesian shrinkage to features.

        Args:
            X: Feature DataFrame

        Returns:
            Shrunk feature DataFrame
        """
        factors = self.get_shrinkage_factors(X.columns.tolist())

        X_shrunk = X.copy()
        for i, col in enumerate(X.columns):
            # Shrinkage: multiply by factor, which shrinks toward zero
            X_shrunk[col] = X[col] * factors[i]

        return X_shrunk

    def get_effective_features(self, threshold: float = 0.1) -> list[str]:
        """Get features with shrinkage factor above threshold.

        These are features that retain meaningful signal after shrinkage.

        Args:
            threshold: Minimum shrinkage factor to be considered "effective"

        Returns:
            List of effective feature names
        """
        return [feat for feat, factor in self.shrinkage_factors.items() if factor >= threshold]


def create_weighted_training_data(
    X: pd.DataFrame,
    shap_csv_path: str = "data/feature_importance_shap.csv",
    method: str = "bayesian_shrinkage",
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """Convenience function to create weighted training data.

    Args:
        X: Raw feature DataFrame
        shap_csv_path: Path to SHAP importance CSV
        method: Weighting method
        scale_factor: Weight scaling factor

    Returns:
        Weighted feature DataFrame
    """
    weighter = FeatureWeighter.from_shap_csv(shap_csv_path)
    return weighter.transform(X, method=method, scale_factor=scale_factor)


if __name__ == "__main__":
    # Demo usage
    import sys

    csv_path = "data/feature_importance_shap.csv"
    if not Path(csv_path).exists():
        print(f"SHAP CSV not found at {csv_path}")
        sys.exit(1)

    weighter = FeatureWeighter.from_shap_csv(csv_path)

    print("=" * 60)
    print("FEATURE WEIGHTING ANALYSIS")
    print("=" * 60)

    print("\n=== Importance Distribution ===")
    summary = weighter.get_summary()
    print(f"  Total features:       {summary['n_features']}")
    print(
        f"  Zero importance:      {summary['n_zero_importance']} ({summary['n_zero_importance']/summary['n_features']*100:.1f}%)"
    )
    print(f"  Above median:         {summary['n_above_median']}")
    print(f"  Top 10% share:        {summary['top_10_pct_share']*100:.1f}% of total importance")
    print(f"  Bottom 50% share:     {summary['bottom_50_pct_share']*100:.1f}% of total importance")

    print("\n=== Feature Tiers (by importance) ===")
    tiers = weighter.get_feature_tiers()
    for tier, features in tiers.items():
        print(f"\n  {tier.upper()} ({len(features)} features):")
        for f in features[:5]:
            imp = weighter.importance_map.get(f, 0)
            print(f"    - {f}: {imp:.4f}")
        if len(features) > 5:
            print(f"    ... and {len(features) - 5} more")

    print("\n=== Bayesian Shrinkage Analysis ===")
    shrinker = BayesianFeatureShrinker(weighter.importance_df, global_shrinkage=0.3)

    for threshold in [0.9, 0.7, 0.5, 0.3]:
        effective = shrinker.get_effective_features(threshold=threshold)
        print(
            f"  Kappa >= {threshold}: {len(effective):3d} features retain {threshold*100:.0f}%+ signal"
        )

    print("\n=== Weighting Method Comparison ===")
    top_features = weighter.importance_df.nlargest(5, "combined_shap")["feature"].tolist()
    bottom_features = weighter.importance_df.nsmallest(5, "combined_shap")["feature"].tolist()

    print("\n  Top 5 features:")
    for method in ["dirichlet", "bayesian_shrinkage"]:
        weights = weighter.get_weights(top_features, method)
        print(f"    {method:20s}: {weights.round(4)}")

    print("\n  Bottom 5 features:")
    for method in ["dirichlet", "bayesian_shrinkage"]:
        weights = weighter.get_weights(bottom_features, method)
        print(f"    {method:20s}: {weights.round(4)}")

    print("\n=== Recommendations ===")
    keep, prune = weighter.get_pruning_candidates(threshold=0.001, min_keep=150)
    print(f"  Features to keep:     {len(keep)} (SHAP >= 0.001 or top 150)")
    print(f"  Pruning candidates:   {len(prune)} (low signal)")
    print(f"\n  Alternative to pruning: Use 'bayesian_shrinkage' method to")
    print(f"  soft-threshold these {len(prune)} features toward zero.")
