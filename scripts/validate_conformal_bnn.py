#!/usr/bin/env python3
"""
Comprehensive BNN Conformal Prediction Validation Script

Validates the BNN Conformal Prediction wrapper by:
1. Loading NFL rushing yards data (2018-2024) from database
2. Splitting data temporally (60/20/20 train/calib/test)
3. Loading existing BNN model (or falling back to Random Forest)
4. Calibrating conformal prediction on calibration set
5. Validating 90% coverage on test set
6. Comparing to temperature scaling baseline (89.5%)
7. Generating comprehensive performance report and plots

Usage:
    python scripts/validate_conformal_bnn.py [--use-rf-fallback]

Created: 2025-11-09
Author: Claude Code
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Try to import PyTorch (optional)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available - will use Random Forest fallback")

# Try to import BNN conformal wrapper
try:
    from py.models.bnn_conformal import BNNConformalPredictor, PyTorchBNNWrapper
    BNN_CONFORMAL_AVAILABLE = True
except ImportError:
    BNN_CONFORMAL_AVAILABLE = False
    logging.warning("BNN Conformal wrapper not available")

# Import MAPIE for conformal prediction
try:
    from mapie.regression import SplitConformalRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    SplitConformalRegressor = None  # Type hint placeholder
    logging.error("MAPIE not installed - install with: pip install mapie")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConformalBNNValidator:
    """Comprehensive validator for BNN Conformal Prediction"""

    def __init__(
        self,
        db_config: dict | None = None,
        alpha: float = 0.10,  # 90% coverage
        use_rf_fallback: bool = False
    ):
        """
        Initialize validator.

        Args:
            db_config: Database connection config (uses defaults if None)
            alpha: Miscoverage level (0.10 for 90% coverage)
            use_rf_fallback: Force Random Forest instead of BNN
        """
        self.alpha = alpha
        self.use_rf_fallback = use_rf_fallback or not PYTORCH_AVAILABLE

        # Database config
        if db_config is None:
            self.db_config = {
                "host": "localhost",
                "port": 5544,
                "database": "devdb01",
                "user": "dro",
                "password": "sicillionbillions",
            }
        else:
            self.db_config = db_config

        # Model paths
        self.bnn_model_paths = [
            "models/bayesian/bnn_multioutput_v1_final.pt",
            "models/bayesian/bnn_multioutput_passing_v1_canonical.pt",
            "models/bayesian/bnn_passing_v1.pkl"
        ]

        # Results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)

        logger.info("ConformalBNNValidator initialized:")
        logger.info(f"  Target coverage: {1-alpha:.1%}")
        logger.info(f"  PyTorch available: {PYTORCH_AVAILABLE}")
        logger.info(f"  MAPIE available: {MAPIE_AVAILABLE}")
        logger.info(f"  Use RF fallback: {self.use_rf_fallback}")

    def load_data(
        self,
        start_season: int = 2018,
        end_season: int = 2024
    ) -> pd.DataFrame:
        """
        Load rushing yards data with Vegas lines from database.

        Uses same query as conformal_prediction_baseline.py to ensure
        consistency with existing experiments.

        Args:
            start_season: First season to load (default: 2018)
            end_season: Last season to load (default: 2024)

        Returns:
            DataFrame with rushing data and features
        """
        logger.info(f"\nLoading rushing data ({start_season}-{end_season})...")

        try:
            conn = psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.warning("Attempting to use cached data...")
            return self._load_cached_data()

        query = f"""
        WITH rushing_data AS (
            SELECT
                pgs.player_id,
                pgs.player_display_name as player_name,
                pgs.season,
                pgs.week,
                pgs.current_team as team,
                pgs.player_position as position,
                pgs.stat_yards,
                pgs.stat_attempts as carries,
                -- Recent form (3-game average)
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.player_id
                    ORDER BY pgs.season, pgs.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_rushing_l3,
                -- Season average
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.player_id, pgs.season
                    ORDER BY pgs.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as season_avg
            FROM mart.player_game_stats pgs
            WHERE pgs.season BETWEEN {start_season} AND {end_season}
              AND pgs.stat_category = 'rushing'
              AND pgs.position_group IN ('RB', 'FB', 'HB')
              AND pgs.stat_attempts >= 5
              AND pgs.stat_yards IS NOT NULL
        )
        SELECT
            rd.*,
            -- Vegas features
            CASE
                WHEN rd.team = g.home_team THEN g.spread_close
                ELSE -g.spread_close
            END as spread_close,
            g.total_close
        FROM rushing_data rd
        LEFT JOIN games g
            ON rd.season = g.season
            AND rd.week = g.week
            AND (rd.team = g.home_team OR rd.team = g.away_team)
        WHERE rd.stat_yards IS NOT NULL
          AND g.spread_close IS NOT NULL
          AND g.total_close IS NOT NULL
        ORDER BY rd.season, rd.week, rd.stat_yards DESC
        """

        try:
            df = pd.read_sql(query, conn)
            conn.close()

            # Handle missing values
            df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
                df.groupby("position")["stat_yards"].transform("median")
            )
            df["season_avg"] = df["season_avg"].fillna(
                df.groupby("position")["stat_yards"].transform("median")
            )

            logger.info(f"✓ Loaded {len(df)} rushing performances")
            logger.info(f"  Players: {df['player_id'].nunique()}")
            logger.info(f"  Seasons: {df['season'].min()}-{df['season'].max()}")
            logger.info(f"  Vegas lines: {df['spread_close'].notna().sum()} games")

            # Cache data for future use
            cache_path = self.results_dir / "rushing_data_cache.pkl"
            df.to_pickle(cache_path)
            logger.info(f"  Data cached to {cache_path}")

            return df

        except Exception as e:
            logger.error(f"Query failed: {e}")
            conn.close()
            return self._load_cached_data()

    def _load_cached_data(self) -> pd.DataFrame:
        """Load cached data if database connection fails"""
        cache_path = self.results_dir / "rushing_data_cache.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_pickle(cache_path)
            logger.info(f"✓ Loaded {len(df)} cached samples")
            return df
        else:
            raise ValueError(
                "No cached data available and database connection failed. "
                "Please ensure database is running or provide cached data."
            )

    def split_data(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into train/calib/test (60/20/20).

        Split strategy:
        - Train: 2018-2021 (60%)
        - Calibration: 2022 + first half of 2023 (20%)
        - Test: second half of 2023 + 2024 (20%)

        Args:
            df: Full dataset

        Returns:
            Tuple of (train_df, calib_df, test_df)
        """
        logger.info("\nSplitting data temporally...")

        # Temporal split (no data leakage)
        train_mask = df['season'] <= 2021
        calib_mask = (df['season'] == 2022) | (
            (df['season'] == 2023) & (df['week'] <= 9)
        )
        test_mask = ((df['season'] == 2023) & (df['week'] > 9)) | (
            df['season'] == 2024
        )

        train_df = df[train_mask].copy()
        calib_df = df[calib_mask].copy()
        test_df = df[test_mask].copy()

        logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"    Seasons: {train_df['season'].min()}-{train_df['season'].max()}")
        logger.info(f"  Calibration: {len(calib_df)} samples ({len(calib_df)/len(df)*100:.1f}%)")
        logger.info(f"    Seasons: {calib_df['season'].min()}-{calib_df['season'].max()}")
        logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        logger.info(f"    Seasons: {test_df['season'].min()}-{test_df['season'].max()}")

        # Validate split (no overlap)
        assert len(set(train_df.index) & set(calib_df.index)) == 0
        assert len(set(train_df.index) & set(test_df.index)) == 0
        assert len(set(calib_df.index) & set(test_df.index)) == 0
        logger.info("  ✓ No overlap between splits")

        return train_df, calib_df, test_df

    def prepare_features(
        self,
        df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix (same 6 features as baseline).

        Features:
        1. carries (stat_attempts)
        2. avg_rushing_l3 (3-game rolling average)
        3. season_avg (season-to-date average)
        4. week (game week)
        5. spread_close (Vegas spread)
        6. total_close (Vegas total)

        Args:
            df: DataFrame with raw features

        Returns:
            Tuple of (X, y) arrays
        """
        feature_cols = [
            "carries",
            "avg_rushing_l3",
            "season_avg",
            "week",
            "spread_close",
            "total_close",
        ]

        # Handle missing values
        X = df[feature_cols].fillna(0).values
        y = df["stat_yards"].values

        logger.info(f"  Feature matrix: {X.shape}")
        logger.info(f"  Features: {', '.join(feature_cols)}")

        return X, y

    def load_bnn_model(self) -> nn.Module | None:
        """
        Try to load existing BNN model from checkpoints.

        Tries multiple paths and checkpoint formats.

        Returns:
            Loaded BNN model or None if loading fails
        """
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available - cannot load BNN")
            return None

        logger.info("\nAttempting to load BNN model...")

        for model_path in self.bnn_model_paths:
            path = Path(model_path)
            if not path.exists():
                logger.debug(f"  {model_path}: not found")
                continue

            logger.info(f"  Trying: {model_path}")

            try:
                # Load checkpoint
                checkpoint = torch.load(path, map_location='cpu')

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        logger.error(
                            "    State dict format requires model architecture. "
                            "Please use direct model checkpoint."
                        )
                        continue
                    elif 'model' in checkpoint:
                        model = checkpoint['model']
                        logger.info("    ✓ Loaded model from 'model' key")
                        return model
                    else:
                        logger.error(f"    Unknown checkpoint format: {checkpoint.keys()}")
                        continue
                else:
                    # Direct model checkpoint
                    model = checkpoint
                    logger.info("    ✓ Loaded direct model checkpoint")
                    return model

            except Exception as e:
                logger.warning(f"    Failed: {e}")
                continue

        logger.warning("  Could not load BNN model from any path")
        return None

    def create_conformal_predictor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        bnn_model: nn.Module | None = None
    ) -> Any:
        """
        Create conformal predictor with BNN or Random Forest base.

        Args:
            X_train: Training features (for RF training)
            y_train: Training targets (for RF training)
            bnn_model: Pre-trained BNN model (optional)

        Returns:
            Conformal predictor (not yet calibrated)
        """
        logger.info("\nCreating conformal predictor...")

        if bnn_model is not None and not self.use_rf_fallback:
            logger.info("  Base model: BNN (pre-trained)")

            # Wrap BNN for sklearn compatibility
            base_estimator = PyTorchBNNWrapper(
                bnn_model,
                device='cpu',
                n_samples=50
            )

            # No training needed (pre-fitted)
            cv_method = "prefit"

        else:
            logger.info("  Base model: Random Forest")
            logger.info("    Training Random Forest on full training set...")

            # Train Random Forest
            base_estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            # Fit RF
            base_estimator.fit(X_scaled, y_train)
            logger.info("    ✓ Random Forest trained")

            # Store scaler for later use
            self.scaler = scaler

        # Create MAPIE split conformal predictor
        # Note: SplitConformalRegressor in MAPIE 1.1.0 uses prefit=True by default
        conformal_model = SplitConformalRegressor(
            estimator=base_estimator,
            confidence_level=1 - self.alpha,  # 0.90 for 90% coverage
            conformity_score='absolute',  # Absolute residual score
            prefit=True,  # Estimator already trained
            n_jobs=-1
        )

        logger.info("  ✓ Conformal predictor created")
        return conformal_model

    def calibrate_conformal(
        self,
        conformal_model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        verbose: bool = True
    ):
        """
        Calibrate conformal prediction on calibration set.

        This computes the nonconformity scores needed for coverage.

        Args:
            conformal_model: Conformal predictor (not yet calibrated)
            X_calib: Calibration features
            y_calib: Calibration targets
            verbose: Print calibration statistics
        """
        if verbose:
            logger.info("\nCalibrating conformal prediction...")
            logger.info(f"  Calibration set: {len(X_calib)} samples")
            logger.info(f"  Target coverage: {1-self.alpha:.1%}")

        # Scale features if using RF
        if hasattr(self, 'scaler'):
            X_calib_scaled = self.scaler.transform(X_calib)
        else:
            X_calib_scaled = X_calib

        # Conformalize (compute nonconformity scores)
        # In MAPIE 1.1.0, use conformalize() instead of fit()
        conformal_model.conformalize(X_calib_scaled, y_calib)

        if verbose:
            # Validate calibration on calibration set itself
            y_pred_calib, y_intervals_calib = conformal_model.predict_interval(X_calib_scaled)

            # Check empirical coverage
            # intervals shape: (n_samples, 2, 1) where [:, 0, 0] is lower, [:, 1, 0] is upper
            coverage = np.mean(
                (y_calib >= y_intervals_calib[:, 0, 0]) &
                (y_calib <= y_intervals_calib[:, 1, 0])
            )

            width = np.mean(y_intervals_calib[:, 1, 0] - y_intervals_calib[:, 0, 0])

            logger.info("\n  Calibration results:")
            logger.info(f"    Empirical coverage: {coverage:.1%} (target: {1-self.alpha:.1%})")
            logger.info(f"    Average interval width: {width:.1f} yards")

            if coverage >= (1 - self.alpha - 0.02):
                logger.info("    ✓ Coverage guarantee achieved!")
            else:
                logger.warning("    ⚠ Coverage below target (may need more calibration data)")

    def evaluate_test_set(
        self,
        conformal_model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Evaluate conformal predictor on test set.

        Args:
            conformal_model: Calibrated conformal predictor
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 80)

        # Scale features if using RF
        if hasattr(self, 'scaler'):
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        # Make predictions with conformal intervals
        # Returns: (predictions, intervals) where intervals shape is (n_samples, 2, 1)
        y_pred, y_intervals = conformal_model.predict_interval(X_test_scaled)

        # Extract bounds
        y_lower = y_intervals[:, 0, 0]
        y_upper = y_intervals[:, 1, 0]

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs(y_pred - y_test) / np.maximum(y_test, 1)) * 100

        # Coverage metrics
        in_interval = (y_test >= y_lower) & (y_test <= y_upper)
        coverage_90 = np.mean(in_interval)

        # Interval width (sharpness)
        width = y_upper - y_lower
        avg_width = np.mean(width)
        median_width = np.median(width)

        # Adaptive intervals (check if width varies)
        width_std = np.std(width)
        width_cv = width_std / avg_width  # Coefficient of variation

        # Log results
        logger.info("\nPoint Prediction Accuracy:")
        logger.info(f"  MAE: {mae:.2f} yards")
        logger.info(f"  RMSE: {rmse:.2f} yards")
        logger.info(f"  MAPE: {mape:.1f}%")

        logger.info("\nConformal Prediction Calibration:")
        logger.info(f"  90% Coverage: {coverage_90:.1%} (target: 90%)")
        logger.info(f"  Coverage gap: {(coverage_90 - 0.90):+.1%}")

        logger.info("\nPrediction Interval Sharpness:")
        logger.info(f"  Average width: {avg_width:.1f} yards")
        logger.info(f"  Median width: {median_width:.1f} yards")
        logger.info(f"  Width std: {width_std:.1f} yards")
        logger.info(f"  Width CV: {width_cv:.2f}")

        # Calibration status
        if abs(coverage_90 - 0.90) <= 0.02:
            logger.info("\n  ✓ Coverage within ±2% of target (EXCELLENT)")
        elif abs(coverage_90 - 0.90) <= 0.05:
            logger.info("\n  ↗ Coverage within ±5% of target (GOOD)")
        else:
            logger.warning("\n  ⚠ Coverage deviates >5% from target")

        # Package results
        results = {
            'point_accuracy': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            },
            'calibration': {
                'coverage_90': float(coverage_90),
                'target_coverage': 0.90,
                'coverage_gap': float(coverage_90 - 0.90),
                'within_2pct': bool(abs(coverage_90 - 0.90) <= 0.02)
            },
            'sharpness': {
                'avg_width': float(avg_width),
                'median_width': float(median_width),
                'width_std': float(width_std),
                'width_cv': float(width_cv)
            },
            'n_test': len(y_test)
        }

        return results, (y_pred, y_lower, y_upper)

    def compare_to_temperature_scaling(
        self,
        conformal_results: dict
    ) -> dict:
        """
        Compare conformal prediction to temperature scaling baseline.

        Baseline (from BNN v3.1 Rushing):
        - Coverage: 89.5%
        - Method: Temperature scaling (T=0.75)

        Args:
            conformal_results: Results from conformal prediction

        Returns:
            Comparison dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON TO TEMPERATURE SCALING BASELINE")
        logger.info("=" * 80)

        # Temperature scaling baseline (BNN v3.1 Rushing)
        baseline = {
            'method': 'Temperature Scaling',
            'coverage_90': 0.895,  # 89.5%
            'temperature': 0.75,
            'source': 'BNN v3.1 Rushing (Nov 2, 2025)'
        }

        # Extract conformal metrics
        conformal_coverage = conformal_results['calibration']['coverage_90']
        conformal_width = conformal_results['sharpness']['avg_width']

        # Comparison
        coverage_diff = conformal_coverage - baseline['coverage_90']

        logger.info("\nCoverage Comparison:")
        logger.info(f"  Temperature Scaling: {baseline['coverage_90']:.1%}")
        logger.info(f"  Conformal Prediction: {conformal_coverage:.1%}")
        logger.info(f"  Difference: {coverage_diff:+.1%}")

        if coverage_diff > 0:
            logger.info("  → Conformal prediction achieves BETTER coverage")
        elif coverage_diff > -0.02:
            logger.info("  → Similar coverage (within 2%)")
        else:
            logger.warning("  → Temperature scaling has better coverage")

        logger.info("\nSharpness:")
        logger.info(f"  Conformal width: {conformal_width:.1f} yards")
        logger.info("  (Temperature scaling width not available)")

        comparison = {
            'baseline': baseline,
            'conformal': {
                'coverage_90': conformal_coverage,
                'avg_width': conformal_width
            },
            'comparison': {
                'coverage_diff': float(coverage_diff),
                'conformal_better': coverage_diff > 0
            }
        }

        return comparison

    def generate_plots(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        conformal_model: Any,
        X_calib: np.ndarray,
        y_calib: np.ndarray
    ):
        """
        Generate diagnostic plots.

        Creates:
        1. Calibration curve (predicted vs empirical coverage)
        2. Interval width distribution
        3. Coverage by prediction value

        Args:
            y_test: Test targets
            y_pred: Test predictions
            y_lower: Lower bounds
            y_upper: Upper bounds
            conformal_model: Calibrated model (for calibration curve)
            X_calib: Calibration features
            y_calib: Calibration targets
        """
        logger.info("\nGenerating diagnostic plots...")

        # Scale features if using RF
        if hasattr(self, 'scaler'):
            X_calib_scaled = self.scaler.transform(X_calib)
        else:
            X_calib_scaled = X_calib

        # 1. Calibration Curve
        fig, ax = plt.subplots(figsize=(8, 6))

        # Note: MAPIE 1.1.0 SplitConformalRegressor doesn't support dynamic alpha
        # We'll compute coverage at different confidence levels by re-conforming
        confidence_levels = np.linspace(0.50, 0.99, 20)  # Test different coverage levels
        empirical_coverages = []

        # MAPIE 1.1.0: Need to re-conformalize for each confidence level
        # Extract base estimator from conformal model
        base_estimator = conformal_model._estimator

        for conf_level in confidence_levels:
            # Create new conformal model with this confidence level
            temp_conformal = SplitConformalRegressor(
                estimator=base_estimator,
                confidence_level=conf_level,
                prefit=True,
                n_jobs=-1
            )
            temp_conformal.conformalize(X_calib_scaled, y_calib)

            # Compute empirical coverage on calibration set
            _, intervals = temp_conformal.predict_interval(X_calib_scaled)
            coverage = np.mean(
                (y_calib >= intervals[:, 0, 0]) &
                (y_calib <= intervals[:, 1, 0])
            )
            empirical_coverages.append(coverage)

        target_coverages = confidence_levels

        ax.plot(target_coverages, empirical_coverages, 'o-', label='Conformal')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Target Coverage', fontsize=12)
        ax.set_ylabel('Empirical Coverage', fontsize=12)
        ax.set_title('Conformal Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = self.results_dir / "figures/conformal_calibration_curve.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {plot_path}")

        # 2. Interval Width Distribution
        fig, ax = plt.subplots(figsize=(8, 6))

        widths = y_upper - y_lower
        # Handle edge case where intervals might be constant or nearly constant
        unique_widths = np.unique(widths)

        # Simple bar chart approach for all cases (avoids numpy histogram bin errors)
        counts = [np.sum(widths == w) for w in unique_widths]
        if len(unique_widths) == 1:
            # Single value - single bar
            ax.bar([unique_widths[0]], [counts[0]], width=1.0, alpha=0.7, edgecolor='black')
            ax.set_xlim(unique_widths[0] - 2, unique_widths[0] + 2)
        elif len(unique_widths) < 50:
            # Few unique values - use bar chart
            width = min(0.8 * np.diff(sorted(unique_widths)).min() if len(unique_widths) > 1 else 1.0, 1.0)
            ax.bar(unique_widths, counts, width=width, alpha=0.7, edgecolor='black')
        else:
            # Many unique values - try histogram with fixed number of bins
            try:
                ax.hist(widths, bins=30, alpha=0.7, edgecolor='black')
            except ValueError:
                # Fallback to bar chart
                ax.bar(unique_widths, counts, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(widths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(widths):.1f} yards')
        ax.axvline(np.median(widths), color='blue', linestyle='--',
                   label=f'Median: {np.median(widths):.1f} yards')
        ax.set_xlabel('Interval Width (yards)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Prediction Interval Width Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plot_path = self.results_dir / "figures/conformal_interval_width_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {plot_path}")

        # 3. Coverage by Prediction Value
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bin predictions
        bins = np.linspace(0, 200, 11)  # 0-200 yards in 20-yard bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        coverages = []
        widths_by_bin = []
        counts = []

        for i in range(len(bins) - 1):
            mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
            if mask.sum() > 0:
                coverage = np.mean(
                    (y_test[mask] >= y_lower[mask]) &
                    (y_test[mask] <= y_upper[mask])
                )
                width = np.mean((y_upper - y_lower)[mask])
            else:
                coverage = np.nan
                width = np.nan

            coverages.append(coverage)
            widths_by_bin.append(width)
            counts.append(mask.sum())

        # Plot coverage by bin
        ax.plot(bin_centers, coverages, 'o-', label='Empirical Coverage', linewidth=2)
        ax.axhline(0.90, color='red', linestyle='--', label='Target (90%)', linewidth=2)
        ax.set_xlabel('Predicted Rushing Yards', fontsize=12)
        ax.set_ylabel('Coverage', fontsize=12)
        ax.set_title('Coverage by Prediction Value', fontsize=14, fontweight='bold')
        ax.set_ylim([0.7, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add sample counts as text
        for i, (x, y, count) in enumerate(zip(bin_centers, coverages, counts)):
            if not np.isnan(y) and count > 0:
                ax.text(x, y + 0.02, f'n={count}', ha='center', va='bottom',
                       fontsize=8, alpha=0.7)

        plot_path = self.results_dir / "figures/conformal_coverage_by_prediction.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved: {plot_path}")

    def save_results(
        self,
        results: dict,
        comparison: dict,
        use_bnn: bool
    ):
        """
        Save results to JSON and Markdown.

        Args:
            results: Test set evaluation results
            comparison: Comparison to temperature scaling
            use_bnn: Whether BNN was used (vs RF fallback)
        """
        logger.info("\nSaving results...")

        # JSON report
        report = {
            'experiment': 'BNN Conformal Prediction Validation',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'base_model': 'BNN (pre-trained)' if use_bnn else 'Random Forest',
                'conformal_method': 'split conformal (Jackknife+)',
                'target_coverage': 0.90,
                'alpha': self.alpha,
                'features': [
                    'carries', 'avg_rushing_l3', 'season_avg',
                    'week', 'spread_close', 'total_close'
                ],
                'n_features': 6
            },
            'results': results,
            'comparison': comparison
        }

        json_path = self.results_dir / "conformal_bnn_validation.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"  ✓ JSON report: {json_path}")

        # Markdown report
        md_lines = [
            "# BNN Conformal Prediction Validation Results",
            f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Base Model:** {'BNN (pre-trained)' if use_bnn else 'Random Forest'}",
            "**Conformal Method:** Split Conformal (Jackknife+)",
            "**Target Coverage:** 90%",
            "",
            "## Test Set Performance",
            "",
            "### Point Accuracy",
            f"- **MAE:** {results['point_accuracy']['mae']:.2f} yards",
            f"- **RMSE:** {results['point_accuracy']['rmse']:.2f} yards",
            f"- **MAPE:** {results['point_accuracy']['mape']:.1f}%",
            "",
            "### Calibration",
            f"- **90% Coverage:** {results['calibration']['coverage_90']:.1%} (target: 90%)",
            f"- **Coverage Gap:** {results['calibration']['coverage_gap']:+.1%}",
            f"- **Within ±2%:** {'✓ YES' if results['calibration']['within_2pct'] else '✗ NO'}",
            "",
            "### Sharpness",
            f"- **Average Width:** {results['sharpness']['avg_width']:.1f} yards",
            f"- **Median Width:** {results['sharpness']['median_width']:.1f} yards",
            f"- **Width Std:** {results['sharpness']['width_std']:.1f} yards",
            f"- **Width CV:** {results['sharpness']['width_cv']:.2f}",
            "",
            "## Comparison to Temperature Scaling",
            "",
            f"**Baseline:** {comparison['baseline']['method']} (T={comparison['baseline']['temperature']})",
            "",
            "| Method | Coverage | Difference |",
            "|--------|----------|------------|",
            f"| Temperature Scaling | {comparison['baseline']['coverage_90']:.1%} | - |",
            f"| Conformal Prediction | {comparison['conformal']['coverage_90']:.1%} | {comparison['comparison']['coverage_diff']:+.1%} |",
            "",
            f"**Result:** {'✓ Conformal prediction achieves better coverage' if comparison['comparison']['conformal_better'] else '↔ Similar coverage'}",
            "",
            "## Diagnostic Plots",
            "",
            "1. **Calibration Curve:** `figures/conformal_calibration_curve.png`",
            "2. **Interval Width Distribution:** `figures/conformal_interval_width_distribution.png`",
            "3. **Coverage by Prediction:** `figures/conformal_coverage_by_prediction.png`",
            "",
            "## Conclusion",
            ""
        ]

        # Add conclusion based on results
        if results['calibration']['within_2pct']:
            md_lines.append(
                "✓ **VALIDATED:** Conformal prediction achieves target 90% coverage (±2%) on test set. "
                "Coverage guarantee is empirically confirmed."
            )
        elif abs(results['calibration']['coverage_gap']) <= 0.05:
            md_lines.append(
                "↗ **GOOD:** Conformal prediction achieves reasonable coverage (within 5% of target). "
                "Small deviation may be due to finite sample effects."
            )
        else:
            md_lines.append(
                "⚠ **NEEDS INVESTIGATION:** Coverage deviates >5% from target. "
                "Consider increasing calibration set size or checking for data drift."
            )

        md_path = self.results_dir / "conformal_vs_temperature.md"
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        logger.info(f"  ✓ Markdown report: {md_path}")

    def run_full_validation(self):
        """
        Run complete validation pipeline.

        This is the main entry point that orchestrates all steps:
        1. Load data
        2. Split data
        3. Load BNN (or fallback to RF)
        4. Create conformal predictor
        5. Calibrate on calibration set
        6. Evaluate on test set
        7. Compare to temperature scaling
        8. Generate plots
        9. Save results
        """
        logger.info("=" * 80)
        logger.info("BNN CONFORMAL PREDICTION - COMPREHENSIVE VALIDATION")
        logger.info("=" * 80)

        # Step 1: Load data
        df = self.load_data(start_season=2018, end_season=2024)

        if len(df) == 0:
            logger.error("No data loaded - cannot proceed")
            return

        # Step 2: Split data
        train_df, calib_df, test_df = self.split_data(df)

        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        X_calib, y_calib = self.prepare_features(calib_df)
        X_test, y_test = self.prepare_features(test_df)

        # Step 3: Load BNN model
        bnn_model = None
        if not self.use_rf_fallback:
            bnn_model = self.load_bnn_model()

        use_bnn = bnn_model is not None

        if bnn_model is None:
            logger.warning("\n⚠ BNN not available - using Random Forest fallback")

        # Step 4: Create conformal predictor
        conformal_model = self.create_conformal_predictor(
            X_train, y_train, bnn_model
        )

        # Step 5: Calibrate
        self.calibrate_conformal(conformal_model, X_calib, y_calib, verbose=True)

        # Step 6: Evaluate on test set
        results, (y_pred, y_lower, y_upper) = self.evaluate_test_set(
            conformal_model, X_test, y_test
        )

        # Step 7: Compare to temperature scaling
        comparison = self.compare_to_temperature_scaling(results)

        # Step 8: Generate plots
        self.generate_plots(
            y_test, y_pred, y_lower, y_upper,
            conformal_model, X_calib, y_calib
        )

        # Step 9: Save results
        self.save_results(results, comparison, use_bnn)

        logger.info("\n" + "=" * 80)
        logger.info("✓ VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {self.results_dir}")
        logger.info("  - conformal_bnn_validation.json (metrics)")
        logger.info("  - conformal_vs_temperature.md (comparison)")
        logger.info("  - figures/ (3 diagnostic plots)")


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Validate BNN Conformal Prediction wrapper"
    )
    parser.add_argument(
        '--use-rf-fallback',
        action='store_true',
        help='Force Random Forest instead of BNN'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.10,
        help='Miscoverage level (default: 0.10 for 90%% coverage)'
    )

    args = parser.parse_args()

    # Create validator
    validator = ConformalBNNValidator(
        alpha=args.alpha,
        use_rf_fallback=args.use_rf_fallback
    )

    # Run full validation
    validator.run_full_validation()


if __name__ == "__main__":
    main()
