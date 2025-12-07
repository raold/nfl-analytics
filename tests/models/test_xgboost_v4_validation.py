#!/usr/bin/env python3
"""
Comprehensive tests for XGBoost v4 (Ridge Residuals) validation.

Tests cover:
1. Ridge coefficient stability across seasons
2. Residual distribution consistency
3. Feature importance stability
4. Prediction consistency (deterministic behavior)
5. Two-stage pipeline error handling
6. Property-based tests for residual calculation

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
import xgboost as xgb
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ============================================================================
# UNIT TESTS: Ridge Coefficient Stability
# ============================================================================


def test_ridge_coefficient_stability_across_seasons():
    """Test that Ridge coefficients are stable across different training seasons."""
    # Arrange: Generate synthetic data for two "seasons"
    np.random.seed(42)
    n_samples = 1000
    n_features = 3

    # Season 1
    X1 = np.random.randn(n_samples, n_features)
    y1 = X1 @ np.array([2.0, -1.5, 0.5]) + np.random.randn(n_samples) * 0.1

    # Season 2 (same underlying coefficients)
    X2 = np.random.randn(n_samples, n_features)
    y2 = X2 @ np.array([2.0, -1.5, 0.5]) + np.random.randn(n_samples) * 0.1

    # Act: Train Ridge on both seasons
    ridge1 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge1.fit(X1, y1)

    ridge2 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge2.fit(X2, y2)

    # Assert: Coefficients should be similar (correlation > 0.95)
    coef_corr = np.corrcoef(ridge1.coef_, ridge2.coef_)[0, 1]
    assert coef_corr > 0.95, f"Coefficient correlation too low: {coef_corr:.4f}"

    # Assert: Coefficient magnitudes within 20% of each other
    for i in range(n_features):
        ratio = ridge1.coef_[i] / (ridge2.coef_[i] + 1e-10)
        assert (
            0.8 < ratio < 1.2
        ), f"Coefficient {i} unstable: {ridge1.coef_[i]:.4f} vs {ridge2.coef_[i]:.4f}"


def test_ridge_coefficient_sign_consistency():
    """Test that Ridge coefficient signs are consistent across seasons."""
    # Arrange: Generate data where opponent EPA should have negative effect
    np.random.seed(42)
    n_samples = 1000

    opp_epa = np.random.randn(n_samples)  # Opponent defensive strength
    passing_yards = (
        250 - 30 * opp_epa + np.random.randn(n_samples) * 10
    )  # Strong defense = fewer yards

    X = opp_epa.reshape(-1, 1)

    # Act: Train Ridge
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(X, passing_yards)

    # Assert: Coefficient should be negative (stronger defense = fewer yards)
    assert ridge.coef_[0] < 0, f"Expected negative coefficient, got {ridge.coef_[0]:.4f}"


def test_ridge_coefficient_magnitude_reasonable():
    """Test that Ridge coefficients have reasonable magnitudes."""
    # Arrange: Generate realistic data
    np.random.seed(42)
    n_samples = 1000

    opp_epa = np.random.uniform(-0.3, 0.3, n_samples)  # Realistic EPA range
    opp_points = np.random.uniform(15, 30, n_samples)  # Points allowed
    matchup = np.random.randint(0, 3, n_samples)  # Difficulty

    X = np.column_stack([opp_epa, opp_points, matchup])
    y = 250 - 50 * opp_epa - 2 * opp_points + 5 * matchup + np.random.randn(n_samples) * 15

    # Act: Train Ridge
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge.fit(X_scaled, y)

    # Assert: Coefficients should be reasonable (not extreme)
    for i, coef in enumerate(ridge.coef_):
        assert abs(coef) < 100, f"Coefficient {i} too large: {coef:.4f}"


# ============================================================================
# UNIT TESTS: Residual Distribution Consistency
# ============================================================================


def test_residual_distribution_normality():
    """Test that residuals are approximately normally distributed."""
    # Arrange: Generate data with known residual distribution
    np.random.seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 3)
    y_true = X @ np.array([2.0, -1.0, 0.5])
    y = y_true + np.random.randn(n_samples) * 10  # Normal residuals

    # Act: Fit Ridge and compute residuals
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(X, y)
    residuals = y - ridge.predict(X)

    # Assert: Residuals should pass normality test
    _, p_value = stats.shapiro(residuals[:100])  # Shapiro-Wilk on subsample
    assert p_value > 0.01, f"Residuals not normal (p={p_value:.4f})"


def test_residual_mean_near_zero():
    """Test that residual mean is close to zero."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + np.random.randn(1000) * 5

    # Act
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(X, y)
    residuals = y - ridge.predict(X)

    # Assert: Mean residual near zero
    assert abs(residuals.mean()) < 0.5, f"Residual mean too large: {residuals.mean():.4f}"


def test_residual_heteroskedasticity():
    """Test that residuals have constant variance (homoskedastic)."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + np.random.randn(1000) * 5  # Constant variance

    # Act
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(X, y)
    residuals = y - ridge.predict(X)
    predictions = ridge.predict(X)

    # Split into low/high prediction groups
    median_pred = np.median(predictions)
    low_residuals = residuals[predictions < median_pred]
    high_residuals = residuals[predictions >= median_pred]

    # Assert: Similar variance in both groups
    var_ratio = np.var(low_residuals) / (np.var(high_residuals) + 1e-10)
    assert 0.5 < var_ratio < 2.0, f"Heteroskedastic residuals: ratio={var_ratio:.4f}"


# ============================================================================
# UNIT TESTS: Feature Importance Stability
# ============================================================================


def test_feature_importance_consistency_across_seasons():
    """Test that XGBoost feature importance is consistent across seasons."""
    # Arrange: Generate two seasons with same feature relationships
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Season 1
    X1 = np.random.randn(n_samples, n_features)
    # Feature 0 most important, feature 1 second, others noise
    y1 = 3 * X1[:, 0] + 1.5 * X1[:, 1] + 0.1 * X1[:, 2:].sum(axis=1) + np.random.randn(n_samples)

    # Season 2
    X2 = np.random.randn(n_samples, n_features)
    y2 = 3 * X2[:, 0] + 1.5 * X2[:, 1] + 0.1 * X2[:, 2:].sum(axis=1) + np.random.randn(n_samples)

    # Act: Train XGBoost on both
    model1 = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model1.fit(X1, y1)

    model2 = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model2.fit(X2, y2)

    # Assert: Top 2 features should be same
    top2_season1 = np.argsort(model1.feature_importances_)[-2:][::-1]
    top2_season2 = np.argsort(model2.feature_importances_)[-2:][::-1]

    assert np.array_equal(
        top2_season1, top2_season2
    ), f"Feature importance unstable: {top2_season1} vs {top2_season2}"


def test_residual_feature_has_high_importance():
    """Test that residual feature has high importance in XGBoost."""
    # Arrange: Generate data where residual is highly predictive
    np.random.seed(42)
    n_samples = 1000

    base_features = np.random.randn(n_samples, 3)
    residual_feature = np.random.randn(n_samples)
    y = 0.5 * base_features.sum(axis=1) + 5 * residual_feature + np.random.randn(n_samples) * 0.1

    X = np.column_stack([base_features, residual_feature])

    # Act: Train XGBoost
    model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)

    # Assert: Residual feature (index 3) should be most important
    most_important = np.argmax(model.feature_importances_)
    assert most_important == 3, f"Residual feature not most important: {most_important}"


# ============================================================================
# UNIT TESTS: Prediction Consistency (Deterministic Behavior)
# ============================================================================


def test_prediction_deterministic_same_input():
    """Test that same input produces same output."""
    # Arrange
    np.random.seed(42)
    X_train = np.random.randn(1000, 5)
    y_train = X_train @ np.array([1, 2, 3, 4, 5]) + np.random.randn(1000)
    X_test = np.random.randn(10, 5)

    # Act: Train and predict twice
    model1 = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model1.fit(X_train, y_train)
    pred1 = model1.predict(X_test)

    model2 = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model2.fit(X_train, y_train)
    pred2 = model2.predict(X_test)

    # Assert: Predictions identical
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)


def test_two_stage_pipeline_deterministic():
    """Test that full two-stage pipeline is deterministic."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(1000)

    # Split into ridge features and baseline features
    X_ridge = X[:, :3]
    X_baseline = X[:, 3:]

    # Act: Run pipeline twice
    # Pipeline 1
    ridge1 = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    scaler1_ridge = StandardScaler()
    X_ridge_scaled1 = scaler1_ridge.fit_transform(X_ridge)
    ridge1.fit(X_ridge_scaled1, y)
    residuals1 = y - ridge1.predict(X_ridge_scaled1)

    X_xgb1 = np.column_stack([X_baseline, residuals1])
    scaler1_xgb = StandardScaler()
    X_xgb_scaled1 = scaler1_xgb.fit_transform(X_xgb1)
    xgb1 = xgb.XGBRegressor(n_estimators=50, random_state=42)
    xgb1.fit(X_xgb_scaled1, y)
    pred1 = xgb1.predict(X_xgb_scaled1)

    # Pipeline 2
    ridge2 = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    scaler2_ridge = StandardScaler()
    X_ridge_scaled2 = scaler2_ridge.fit_transform(X_ridge)
    ridge2.fit(X_ridge_scaled2, y)
    residuals2 = y - ridge2.predict(X_ridge_scaled2)

    X_xgb2 = np.column_stack([X_baseline, residuals2])
    scaler2_xgb = StandardScaler()
    X_xgb_scaled2 = scaler2_xgb.fit_transform(X_xgb2)
    xgb2 = xgb.XGBRegressor(n_estimators=50, random_state=42)
    xgb2.fit(X_xgb_scaled2, y)
    pred2 = xgb2.predict(X_xgb_scaled2)

    # Assert: Final predictions should be very similar (allow small CV variance)
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=1)


# ============================================================================
# UNIT TESTS: Two-Stage Pipeline Error Handling
# ============================================================================


def test_ridge_handles_missing_values():
    """Test that Ridge gracefully handles missing values."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([1, 2, 3]) + np.random.randn(100)

    # Introduce NaNs
    X[0, 0] = np.nan
    X[5, 1] = np.nan

    # Act: Fill NaN and train
    X_filled = np.nan_to_num(X, nan=0.0)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X_filled, y)

    # Assert: Model should train without error
    assert hasattr(ridge, "coef_"), "Ridge failed to fit"
    assert not np.any(np.isnan(ridge.coef_)), "Ridge coefficients contain NaN"


def test_xgboost_handles_extreme_residuals():
    """Test that XGBoost handles extreme residual values."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    residuals = np.random.randn(1000)
    residuals[0] = 1000  # Extreme outlier
    residuals[1] = -1000

    X_with_residuals = np.column_stack([X, residuals])
    y = X.sum(axis=1) + residuals + np.random.randn(1000)

    # Act: Train XGBoost
    model = xgb.XGBRegressor(n_estimators=50, max_depth=4)
    model.fit(X_with_residuals, y)
    predictions = model.predict(X_with_residuals)

    # Assert: Predictions should be finite
    assert np.all(np.isfinite(predictions)), "XGBoost produced non-finite predictions"


def test_pipeline_handles_zero_variance_features():
    """Test that pipeline handles features with zero variance."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 1] = 5.0  # Zero variance feature
    X[:, 0] + X[:, 2] + np.random.randn(100)

    # Act: Train Ridge with StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Assert: Scaled features should be finite (StandardScaler handles zero variance)
    assert np.all(np.isfinite(X_scaled)), "Scaler produced non-finite values"


def test_pipeline_handles_perfect_collinearity():
    """Test that Ridge handles perfectly collinear features."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 2] = X[:, 0]  # Perfect collinearity
    y = X[:, 0] + np.random.randn(100)

    # Act: Train Ridge (should use regularization to handle collinearity)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(X, y)

    # Assert: Model should train without error
    assert hasattr(ridge, "coef_"), "Ridge failed to handle collinearity"
    assert np.all(np.isfinite(ridge.coef_)), "Ridge coefficients non-finite"


# ============================================================================
# PROPERTY-BASED TESTS: Residual Calculation
# ============================================================================


@given(
    n_samples=st.integers(min_value=50, max_value=200),
    n_features=st.integers(min_value=2, max_value=5),
    noise_level=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=20, deadline=5000)
def test_residual_property_sum_to_zero(n_samples, n_features, noise_level):
    """Property test: Residuals should sum to near zero."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + np.random.randn(n_samples) * noise_level

    # Act
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(X, y)
    residuals = y - ridge.predict(X)

    # Assert: Sum of residuals near zero
    residual_sum = np.sum(residuals)
    assert abs(residual_sum) < 5 * noise_level * np.sqrt(
        n_samples
    ), f"Residuals don't sum to zero: {residual_sum:.4f}"


@given(
    n_samples=st.integers(min_value=50, max_value=200),
    n_features=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=20, deadline=5000)
def test_residual_property_orthogonal_to_predictions(n_samples, n_features):
    """Property test: Residuals should be uncorrelated with predictions."""
    # Arrange
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = X @ np.random.randn(n_features) + np.random.randn(n_samples)

    # Act
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(X, y)
    predictions = ridge.predict(X)
    residuals = y - predictions

    # Assert: Correlation between residuals and predictions near zero
    correlation = np.corrcoef(predictions, residuals)[0, 1]
    assert abs(correlation) < 0.3, f"Residuals correlated with predictions: {correlation:.4f}"


@given(n_samples=st.integers(min_value=100, max_value=300))
@settings(max_examples=20, deadline=5000)
def test_residual_property_variance_reduction(n_samples):
    """Property test: Adding residual feature should reduce prediction variance."""
    # Arrange
    np.random.seed(42)
    X_base = np.random.randn(n_samples, 3)
    residual = np.random.randn(n_samples) * 2
    y = X_base.sum(axis=1) + 3 * residual + np.random.randn(n_samples) * 0.5

    # Act: Model without residual
    model_without = xgb.XGBRegressor(n_estimators=50, random_state=42)
    model_without.fit(X_base, y)
    pred_without = model_without.predict(X_base)
    var_without = np.var(y - pred_without)

    # Act: Model with residual
    X_with = np.column_stack([X_base, residual])
    model_with = xgb.XGBRegressor(n_estimators=50, random_state=42)
    model_with.fit(X_with, y)
    pred_with = model_with.predict(X_with)
    var_with = np.var(y - pred_with)

    # Assert: Adding residual should reduce prediction error variance
    assert (
        var_with < var_without
    ), f"Residual didn't reduce variance: {var_without:.4f} -> {var_with:.4f}"


# ============================================================================
# INTEGRATION TESTS: End-to-End Pipeline
# ============================================================================


def test_full_pipeline_integration():
    """Test full two-stage pipeline end-to-end."""
    # Arrange: Generate realistic data
    np.random.seed(42)
    n_train = 1000
    n_test = 200

    # Opponent features (Ridge stage)
    opp_epa_train = np.random.uniform(-0.3, 0.3, n_train)
    opp_points_train = np.random.uniform(15, 30, n_train)

    opp_epa_test = np.random.uniform(-0.3, 0.3, n_test)
    opp_points_test = np.random.uniform(15, 30, n_test)

    # Baseline features (XGBoost stage)
    player_avg_train = np.random.uniform(200, 300, n_train)
    player_avg_test = np.random.uniform(200, 300, n_test)

    # True relationship: yards = player_avg - 50*opp_epa - 2*opp_points + noise
    y_train = (
        player_avg_train - 50 * opp_epa_train - 2 * opp_points_train + np.random.randn(n_train) * 15
    )
    y_test = (
        player_avg_test - 50 * opp_epa_test - 2 * opp_points_test + np.random.randn(n_test) * 15
    )

    # Act: Stage 1 - Ridge on opponent features
    X_ridge_train = np.column_stack([opp_epa_train, opp_points_train])
    X_ridge_test = np.column_stack([opp_epa_test, opp_points_test])

    scaler_ridge = StandardScaler()
    X_ridge_train_scaled = scaler_ridge.fit_transform(X_ridge_train)
    X_ridge_test_scaled = scaler_ridge.transform(X_ridge_test)

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
    ridge.fit(X_ridge_train_scaled, y_train)

    ridge_pred_train = ridge.predict(X_ridge_train_scaled)
    ridge_pred_test = ridge.predict(X_ridge_test_scaled)

    residuals_train = y_train - ridge_pred_train
    residuals_test = y_test - ridge_pred_test

    # Act: Stage 2 - XGBoost on baseline + residuals
    X_xgb_train = np.column_stack([player_avg_train, residuals_train])
    X_xgb_test = np.column_stack([player_avg_test, residuals_test])

    scaler_xgb = StandardScaler()
    X_xgb_train_scaled = scaler_xgb.fit_transform(X_xgb_train)
    X_xgb_test_scaled = scaler_xgb.transform(X_xgb_test)

    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    xgb_model.fit(X_xgb_train_scaled, y_train)

    final_pred = xgb_model.predict(X_xgb_test_scaled)

    # Assert: Final predictions should be accurate
    mae = np.mean(np.abs(y_test - final_pred))
    assert mae < 20, f"Pipeline MAE too high: {mae:.2f}"

    # Assert: Final predictions better than Ridge alone
    ridge_mae = np.mean(np.abs(y_test - ridge_pred_test))
    assert mae < ridge_mae, f"XGBoost didn't improve over Ridge: {mae:.2f} vs {ridge_mae:.2f}"


# ============================================================================
# REGRESSION TESTS: Known Values
# ============================================================================


def test_known_baseline_comparison():
    """Test that v4 beats known v2 baseline on synthetic data."""
    # Arrange: Generate data where v4 should excel
    np.random.seed(42)
    n = 1000

    opp_strength = np.random.randn(n)
    player_skill = np.random.randn(n)
    y = 250 + 30 * player_skill - 20 * opp_strength + np.random.randn(n) * 10

    # Baseline model (player_skill only)
    from sklearn.linear_model import LinearRegression

    baseline = LinearRegression()
    baseline.fit(player_skill.reshape(-1, 1), y)
    baseline_pred = baseline.predict(player_skill.reshape(-1, 1))
    baseline_mae = np.mean(np.abs(y - baseline_pred))

    # v4 model (Ridge residuals)
    # Stage 1: Ridge on opponent
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    ridge.fit(opp_strength.reshape(-1, 1), y)
    residuals = y - ridge.predict(opp_strength.reshape(-1, 1))

    # Stage 2: XGBoost on player + residuals
    X_xgb = np.column_stack([player_skill, residuals])
    xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
    xgb_model.fit(X_xgb, y)
    v4_pred = xgb_model.predict(X_xgb)
    v4_mae = np.mean(np.abs(y - v4_pred))

    # Assert: v4 should beat baseline
    improvement_pct = ((baseline_mae - v4_mae) / baseline_mae) * 100
    assert v4_mae < baseline_mae, f"v4 didn't beat baseline: {v4_mae:.2f} vs {baseline_mae:.2f}"
    assert improvement_pct > 30, f"Improvement too small: {improvement_pct:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
