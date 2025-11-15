# XGBoost Opponent Features: Deep Diagnostic Analysis Plan

**Date:** November 14, 2025
**Author:** Claude Code (Sonnet 4.5)
**Status:** ANALYSIS PLAN (Awaiting Experiment 1 Execution)
**Context:** 3 consecutive failures with opponent features (-6.6%, -42.71%, interaction experiment pending)

---

## Executive Summary

This document provides a **diagnostic analysis framework** for investigating why opponent defensive features consistently degrade XGBoost model performance despite being theoretically sound and showing non-zero feature importance.

**Current State:**
- **v2 Baseline (no opponent):** 40.20 MAE, 11 features ✓ PRODUCTION
- **v3.0 (raw opponent):** 42.86 MAE, 14 features ✗ -6.6% regression
- **v3.1 (aligned window):** 57.37 MAE, 14 features ✗ -42.71% regression
- **Experiment 1 (interactions):** **PENDING** (blocked by Docker)

**Critical Question:** Is there ANY way opponent features can help XGBoost, or is this a fundamental incompatibility?

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Diagnostic Framework](#diagnostic-framework)
3. [Analysis 1: Feature Correlation & Multicollinearity](#analysis-1-feature-correlation--multicollinearity)
4. [Analysis 2: SHAP Value Investigation](#analysis-2-shap-value-investigation)
5. [Analysis 3: Data Quality Audit](#analysis-3-data-quality-audit)
6. [Analysis 4: Prediction Error Decomposition](#analysis-4-prediction-error-decomposition)
7. [Analysis 5: Tree Depth & Overfitting Analysis](#analysis-5-tree-depth--overfitting-analysis)
8. [Root Cause Hypothesis](#root-cause-hypothesis)
9. [Actionable Insights](#actionable-insights)
10. [Go/No-Go Decision Framework](#gono-go-decision-framework)

---

## Problem Statement

### Failure Pattern

| Version | Opponent Features | MAE | vs Baseline | Top Opp Feature Importance |
|---------|------------------|-----|-------------|---------------------------|
| v2 | None | 40.20 | 0% (baseline) | N/A |
| v3.0 | Raw (3 features) | 42.86 | -6.6% | 3.3% (`opp_def_epa_exp`) |
| v3.1 | Raw (aligned) | 57.37 | -42.71% | Unknown (model corrupted) |
| Exp 1 | Interactions (8 features) | **PENDING** | **?** | **?** |

### Key Observations

1. **Opponent features ARE being used** (3.3%+ importance) but hurt performance
2. **Alignment fix improved coverage** (85.6%) but made performance WORSE
3. **Interaction features** (if implemented) show 9%+ importance but **still regress** (per user's statement)

### Research Question

**Is opponent strength modeling fundamentally incompatible with XGBoost for NFL QB prediction?**

Possible answers:
- **A.** Feature engineering issue (fixable with interactions, normalization, regularization)
- **B.** Data quality issue (opponent metrics contain more noise than signal)
- **C.** Architectural incompatibility (tree-based models can't properly contextualize opponent strength)
- **D.** Multicollinearity issue (opponent info redundant with rolling player averages)

---

## Diagnostic Framework

### Required Inputs

1. **Trained Models:**
   - `models/xgboost/passing_xgboost_v2_canonical.pkl` (baseline)
   - `models/xgboost/passing_xgboost_v3_with_opponent_strength.pkl` (v3.0)
   - `models/xgboost/passing_xgboost_v3_1_aligned.pkl` (v3.1)
   - Experiment 1 interaction model (once trained)

2. **Raw Data:**
   - Training data (2018-2024) with opponent features
   - Feature matrix (`X_train`, `X_test`)
   - Target variable (`y_train`, `y_test`)
   - Opponent metrics: `opp_def_epa_exp`, `opp_pass_def_rank`, `matchup_difficulty`

3. **Metadata:**
   - Feature importance rankings
   - Training/validation MAE curves
   - Hyperparameters used

### Tools Required

```python
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import pickle
```

---

## Analysis 1: Feature Correlation & Multicollinearity

### Hypothesis

Opponent features are highly correlated with player rolling averages (e.g., `passing_yards_L3` already captures opponent strength implicitly).

### Method

```python
def analyze_feature_correlation(df, baseline_features, opponent_features):
    """
    Calculate correlation matrix and identify multicollinearity.

    Returns:
        - Correlation heatmap
        - VIF (Variance Inflation Factor) scores
        - Pairwise correlations > 0.8
    """

    # 1. Correlation matrix
    all_features = baseline_features + opponent_features
    corr_matrix = df[all_features].corr()

    # 2. Identify high correlations with opponent features
    high_corr_pairs = []
    for opp_feat in opponent_features:
        for base_feat in baseline_features:
            corr = corr_matrix.loc[opp_feat, base_feat]
            if abs(corr) > 0.7:  # Strong correlation
                high_corr_pairs.append({
                    'opponent_feature': opp_feat,
                    'baseline_feature': base_feat,
                    'correlation': corr,
                    'abs_corr': abs(corr)
                })

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('abs_corr', ascending=False)

    # 3. Variance Inflation Factor (VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["feature"] = all_features
    vif_data["VIF"] = [
        variance_inflation_factor(df[all_features].values, i)
        for i in range(len(all_features))
    ]
    vif_data = vif_data.sort_values('VIF', ascending=False)

    # 4. Visualize
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('docs/research/xgb_correlation_heatmap.png', dpi=300)

    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_df,
        'vif_scores': vif_data
    }
```

### Expected Findings

**If multicollinearity is the root cause:**
- `opp_def_epa_exp` correlates > 0.7 with `passing_yards_L3`
- VIF scores > 10 for opponent features
- XGBoost picks noisy opponent feature over informative player feature in splits

**Action if confirmed:**
- Use L1 regularization (Experiment 5) to force feature selection
- Create orthogonal opponent features (ridge residuals, Approach 2)
- Remove redundant opponent features entirely

### Success Criteria

- Correlation < 0.7 between opponent and baseline features → Proceed with interactions
- Correlation > 0.8 → High multicollinearity, switch to ridge residuals approach

---

## Analysis 2: SHAP Value Investigation

### Hypothesis

Opponent features have high importance but **negative** SHAP values (i.e., they push predictions in the wrong direction).

### Method

```python
def analyze_shap_values(baseline_model, interaction_model, X_test):
    """
    Compare SHAP values to understand feature contribution direction.

    Key insights:
    1. Are opponent features helpful or harmful?
    2. Do interactions "steal" importance from baseline features?
    3. Are SHAP values directionally correct?
    """

    # SHAP explainer for both models
    explainer_baseline = shap.TreeExplainer(baseline_model)
    explainer_interaction = shap.TreeExplainer(interaction_model)

    shap_values_baseline = explainer_baseline.shap_values(X_test)
    shap_values_interaction = explainer_interaction.shap_values(X_test)

    # 1. Feature importance comparison
    shap_importance_baseline = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': np.abs(shap_values_baseline).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    shap_importance_interaction = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': np.abs(shap_values_interaction).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    # 2. Check if baseline features LOSE importance with opponent features
    baseline_core_features = ['passing_yards_L3', 'attempts_L3', 'completion_pct_L3']

    importance_change = []
    for feat in baseline_core_features:
        base_imp = shap_importance_baseline[
            shap_importance_baseline['feature'] == feat
        ]['mean_abs_shap'].values[0] if feat in shap_importance_baseline['feature'].values else 0

        inter_imp = shap_importance_interaction[
            shap_importance_interaction['feature'] == feat
        ]['mean_abs_shap'].values[0] if feat in shap_importance_interaction['feature'].values else 0

        importance_change.append({
            'feature': feat,
            'baseline_importance': base_imp,
            'interaction_importance': inter_imp,
            'change': inter_imp - base_imp,
            'pct_change': ((inter_imp - base_imp) / base_imp * 100) if base_imp > 0 else 0
        })

    importance_change_df = pd.DataFrame(importance_change)

    # 3. SHAP summary plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    shap.summary_plot(shap_values_baseline, X_test, show=False, ax=axes[0])
    axes[0].set_title('Baseline Model SHAP Values')

    shap.summary_plot(shap_values_interaction, X_test, show=False, ax=axes[1])
    axes[1].set_title('Interaction Model SHAP Values')

    plt.tight_layout()
    plt.savefig('docs/research/xgb_shap_comparison.png', dpi=300)

    # 4. Check opponent feature SHAP directionality
    opponent_features = [f for f in X_test.columns if 'opp_' in f or 'elite_def' in f or 'weak_def' in f]

    opponent_shap_stats = []
    for feat in opponent_features:
        if feat in X_test.columns:
            feat_idx = list(X_test.columns).index(feat)
            shap_vals = shap_values_interaction[:, feat_idx]

            opponent_shap_stats.append({
                'feature': feat,
                'mean_shap': shap_vals.mean(),
                'median_shap': np.median(shap_vals),
                'std_shap': shap_vals.std(),
                'positive_pct': (shap_vals > 0).mean() * 100,
                'negative_pct': (shap_vals < 0).mean() * 100
            })

    opponent_shap_df = pd.DataFrame(opponent_shap_stats)

    return {
        'shap_importance_baseline': shap_importance_baseline,
        'shap_importance_interaction': shap_importance_interaction,
        'baseline_feature_importance_change': importance_change_df,
        'opponent_shap_stats': opponent_shap_df
    }
```

### Expected Findings

**If SHAP reveals the problem:**
- Opponent features have **inconsistent** SHAP directionality (50% positive, 50% negative)
- Baseline features (`passing_yards_L3`) LOSE importance when opponent features added
- Interaction features "steal" importance but don't improve predictions

**Action if confirmed:**
- Opponent features add noise, not signal → ABANDON
- Interaction features poorly constructed → Redesign using domain knowledge
- Need stronger regularization to suppress weak features

### Success Criteria

- Opponent features show **consistent positive SHAP** (> 60% positive) → Proceed
- Opponent features show **random SHAP** (40-60% positive) → Likely noise, abandon
- Baseline features **maintain importance** with opponent features → Healthy interaction

---

## Analysis 3: Data Quality Audit

### Hypothesis

Opponent defensive metrics (`opp_def_epa_exp`) contain systematic biases or measurement errors that introduce noise.

### Method

```python
def audit_opponent_feature_quality(df):
    """
    Investigate opponent feature data quality issues.

    Checks:
    1. Distribution sanity (outliers, skewness)
    2. Temporal stability (does opponent EPA change drastically week-to-week?)
    3. Signal-to-noise ratio (correlation with actual outcomes)
    4. Coverage gaps (missing values, zero variance)
    """

    # 1. Distribution analysis
    opponent_features = ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']

    dist_stats = []
    for feat in opponent_features:
        dist_stats.append({
            'feature': feat,
            'mean': df[feat].mean(),
            'std': df[feat].std(),
            'skewness': df[feat].skew(),
            'kurtosis': df[feat].kurtosis(),
            'min': df[feat].min(),
            'max': df[feat].max(),
            'missing_pct': (df[feat].isnull().sum() / len(df)) * 100,
            'zero_variance_pct': (df[feat].nunique() / len(df)) * 100
        })

    dist_df = pd.DataFrame(dist_stats)

    # 2. Temporal stability (week-to-week changes)
    df_sorted = df.sort_values(['player_id', 'kickoff'])

    temporal_stats = []
    for feat in opponent_features:
        df_sorted[f'{feat}_change'] = df_sorted.groupby('player_id')[feat].diff()

        temporal_stats.append({
            'feature': feat,
            'mean_weekly_change': df_sorted[f'{feat}_change'].abs().mean(),
            'std_weekly_change': df_sorted[f'{feat}_change'].std(),
            'max_weekly_change': df_sorted[f'{feat}_change'].abs().max()
        })

    temporal_df = pd.DataFrame(temporal_stats)

    # 3. Signal-to-noise ratio (correlation with outcome)
    signal_stats = []
    for feat in opponent_features:
        corr_pearson, p_pearson = pearsonr(df[feat].dropna(), df['passing_yards'].loc[df[feat].dropna().index])
        corr_spearman, p_spearman = spearmanr(df[feat].dropna(), df['passing_yards'].loc[df[feat].dropna().index])

        signal_stats.append({
            'feature': feat,
            'pearson_corr': corr_pearson,
            'pearson_p_value': p_pearson,
            'spearman_corr': corr_spearman,
            'spearman_p_value': p_spearman,
            'signal_significant': p_pearson < 0.05
        })

    signal_df = pd.DataFrame(signal_stats)

    # 4. Outlier detection
    outlier_stats = []
    for feat in opponent_features:
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[feat] < lower_bound) | (df[feat] > upper_bound)]

        outlier_stats.append({
            'feature': feat,
            'outlier_count': len(outliers),
            'outlier_pct': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })

    outlier_df = pd.DataFrame(outlier_stats)

    # 5. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distribution plots
    for i, feat in enumerate(opponent_features):
        ax = axes[i // 2, i % 2]
        df[feat].hist(bins=50, ax=ax, edgecolor='black')
        ax.set_title(f'{feat} Distribution')
        ax.set_xlabel(feat)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('docs/research/opponent_feature_distributions.png', dpi=300)

    return {
        'distribution_stats': dist_df,
        'temporal_stability': temporal_df,
        'signal_to_noise': signal_df,
        'outlier_analysis': outlier_df
    }
```

### Expected Findings

**If data quality is the issue:**
- Opponent features show **weak correlation** with passing yards (r < 0.1)
- High temporal volatility (week-to-week changes > 0.5 standard deviations)
- Outliers or measurement errors (> 10% outliers)
- Missing values or zero variance (coverage gaps)

**Action if confirmed:**
- Clean opponent data (remove outliers, impute missing values)
- Use smoothed opponent metrics (exponentially weighted moving averages)
- Switch to more stable opponent proxies (season-long averages, Elo ratings)

### Success Criteria

- Pearson correlation > 0.15 with outcome → Signal exists
- Temporal stability (mean weekly change < 0.3 std) → Reliable metric
- Outliers < 5% → Clean data

---

## Analysis 4: Prediction Error Decomposition

### Hypothesis

Opponent features systematically over-predict or under-predict in specific contexts (elite defenses, weak defenses, high-scoring games).

### Method

```python
def decompose_prediction_errors(baseline_model, interaction_model, X_test, y_test, df_test):
    """
    Analyze WHERE opponent features hurt performance.

    Segmentation:
    1. By opponent strength (elite, average, weak defenses)
    2. By QB performance level (high, medium, low passing yards)
    3. By game context (high total, low total, home/away)
    4. By season (2022, 2023, 2024)
    """

    # Generate predictions
    pred_baseline = baseline_model.predict(X_test)
    pred_interaction = interaction_model.predict(X_test)

    # Residuals
    residuals_baseline = y_test - pred_baseline
    residuals_interaction = y_test - pred_interaction

    df_test = df_test.copy()
    df_test['actual'] = y_test
    df_test['pred_baseline'] = pred_baseline
    df_test['pred_interaction'] = pred_interaction
    df_test['residual_baseline'] = residuals_baseline
    df_test['residual_interaction'] = residuals_interaction
    df_test['error_baseline'] = np.abs(residuals_baseline)
    df_test['error_interaction'] = np.abs(residuals_interaction)

    # Segmentation 1: By opponent strength
    df_test['opp_tier'] = pd.cut(
        df_test['opp_pass_def_rank'],
        bins=[0, 10, 22, 32],
        labels=['elite_def', 'avg_def', 'weak_def']
    )

    opp_strength_analysis = df_test.groupby('opp_tier').agg({
        'actual': 'mean',
        'pred_baseline': 'mean',
        'pred_interaction': 'mean',
        'error_baseline': 'mean',
        'error_interaction': 'mean',
        'residual_baseline': 'mean',
        'residual_interaction': 'mean'
    }).reset_index()

    opp_strength_analysis['interaction_improvement'] = (
        (opp_strength_analysis['error_baseline'] - opp_strength_analysis['error_interaction']) /
        opp_strength_analysis['error_baseline'] * 100
    )

    # Segmentation 2: By QB performance level
    df_test['yards_tier'] = pd.cut(
        df_test['actual'],
        bins=[0, 200, 280, 1000],
        labels=['low_yards', 'avg_yards', 'high_yards']
    )

    yards_analysis = df_test.groupby('yards_tier').agg({
        'actual': 'mean',
        'pred_baseline': 'mean',
        'pred_interaction': 'mean',
        'error_baseline': 'mean',
        'error_interaction': 'mean'
    }).reset_index()

    yards_analysis['interaction_improvement'] = (
        (yards_analysis['error_baseline'] - yards_analysis['error_interaction']) /
        yards_analysis['error_baseline'] * 100
    )

    # Segmentation 3: By game context
    df_test['game_context'] = 'neutral'
    df_test.loc[df_test['total_close'] > 47, 'game_context'] = 'high_total'
    df_test.loc[df_test['total_close'] < 44, 'game_context'] = 'low_total'

    context_analysis = df_test.groupby('game_context').agg({
        'actual': 'mean',
        'pred_baseline': 'mean',
        'pred_interaction': 'mean',
        'error_baseline': 'mean',
        'error_interaction': 'mean'
    }).reset_index()

    context_analysis['interaction_improvement'] = (
        (context_analysis['error_baseline'] - context_analysis['error_interaction']) /
        context_analysis['error_baseline'] * 100
    )

    # Segmentation 4: By season
    season_analysis = df_test.groupby('season').agg({
        'actual': 'mean',
        'pred_baseline': 'mean',
        'pred_interaction': 'mean',
        'error_baseline': 'mean',
        'error_interaction': 'mean'
    }).reset_index()

    season_analysis['interaction_improvement'] = (
        (season_analysis['error_baseline'] - season_analysis['error_interaction']) /
        season_analysis['error_baseline'] * 100
    )

    # Visualization: Residual plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # By opponent strength
    axes[0, 0].bar(opp_strength_analysis['opp_tier'], opp_strength_analysis['interaction_improvement'])
    axes[0, 0].set_title('Interaction Improvement by Opponent Strength')
    axes[0, 0].set_ylabel('Improvement (%)')
    axes[0, 0].axhline(0, color='red', linestyle='--')

    # By QB performance
    axes[0, 1].bar(yards_analysis['yards_tier'], yards_analysis['interaction_improvement'])
    axes[0, 1].set_title('Interaction Improvement by QB Performance')
    axes[0, 1].set_ylabel('Improvement (%)')
    axes[0, 1].axhline(0, color='red', linestyle='--')

    # By game context
    axes[1, 0].bar(context_analysis['game_context'], context_analysis['interaction_improvement'])
    axes[1, 0].set_title('Interaction Improvement by Game Context')
    axes[1, 0].set_ylabel('Improvement (%)')
    axes[1, 0].axhline(0, color='red', linestyle='--')

    # By season
    axes[1, 1].bar(season_analysis['season'].astype(str), season_analysis['interaction_improvement'])
    axes[1, 1].set_title('Interaction Improvement by Season')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].axhline(0, color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig('docs/research/error_decomposition_analysis.png', dpi=300)

    return {
        'by_opponent_strength': opp_strength_analysis,
        'by_qb_performance': yards_analysis,
        'by_game_context': context_analysis,
        'by_season': season_analysis
    }
```

### Expected Findings

**If systematic bias exists:**
- Interaction features **hurt performance against elite defenses** (over-adjust)
- Interaction features **hurt performance in low-scoring games** (context mismatch)
- Interaction features **work in 2022 but fail in 2024** (non-stationary)

**Action if confirmed:**
- Segment model by opponent strength (separate models for elite/weak defenses)
- Add game context features (interaction with total_line, spread_close)
- Use time-varying opponent adjustments (recency-weighted EPA)

### Success Criteria

- Interaction improvement **consistent across all segments** (±5%) → Robust
- Interaction improvement **negative in all segments** → Abandon
- Interaction improvement **positive for elite defenses only** → Niche use case

---

## Analysis 5: Tree Depth & Overfitting Analysis

### Hypothesis

Opponent features cause XGBoost to build deeper, more complex trees that overfit the training data.

### Method

```python
def analyze_tree_complexity(baseline_model, interaction_model, X_train, y_train, X_test, y_test):
    """
    Compare tree structure and overfitting between baseline and interaction models.

    Metrics:
    1. Average tree depth
    2. Number of leaves per tree
    3. Train vs test MAE gap (overfitting indicator)
    4. Feature splits distribution
    """

    # 1. Get tree structure info
    baseline_booster = baseline_model.get_booster()
    interaction_booster = interaction_model.get_booster()

    # Parse tree structure (XGBoost internal format)
    baseline_trees = baseline_booster.get_dump(dump_format='json')
    interaction_trees = interaction_booster.get_dump(dump_format='json')

    import json

    def calculate_tree_depth(tree_json):
        tree = json.loads(tree_json)

        def get_depth(node):
            if 'leaf' in node:
                return 1
            left_depth = get_depth(node['children'][0])
            right_depth = get_depth(node['children'][1])
            return 1 + max(left_depth, right_depth)

        return get_depth(tree)

    def count_leaves(tree_json):
        tree = json.loads(tree_json)

        def count(node):
            if 'leaf' in node:
                return 1
            return count(node['children'][0]) + count(node['children'][1])

        return count(tree)

    baseline_depths = [calculate_tree_depth(t) for t in baseline_trees]
    interaction_depths = [calculate_tree_depth(t) for t in interaction_trees]

    baseline_leaves = [count_leaves(t) for t in baseline_trees]
    interaction_leaves = [count_leaves(t) for t in interaction_trees]

    tree_stats = {
        'baseline': {
            'mean_depth': np.mean(baseline_depths),
            'std_depth': np.std(baseline_depths),
            'max_depth': np.max(baseline_depths),
            'mean_leaves': np.mean(baseline_leaves),
            'total_leaves': np.sum(baseline_leaves)
        },
        'interaction': {
            'mean_depth': np.mean(interaction_depths),
            'std_depth': np.std(interaction_depths),
            'max_depth': np.max(interaction_depths),
            'mean_leaves': np.mean(interaction_leaves),
            'total_leaves': np.sum(interaction_leaves)
        }
    }

    # 2. Train vs test MAE gap (overfitting check)
    train_pred_baseline = baseline_model.predict(X_train)
    test_pred_baseline = baseline_model.predict(X_test)

    train_pred_interaction = interaction_model.predict(X_train)
    test_pred_interaction = interaction_model.predict(X_test)

    overfitting_stats = {
        'baseline': {
            'train_mae': mean_absolute_error(y_train, train_pred_baseline),
            'test_mae': mean_absolute_error(y_test, test_pred_baseline),
            'gap': mean_absolute_error(y_test, test_pred_baseline) - mean_absolute_error(y_train, train_pred_baseline)
        },
        'interaction': {
            'train_mae': mean_absolute_error(y_train, train_pred_interaction),
            'test_mae': mean_absolute_error(y_test, test_pred_interaction),
            'gap': mean_absolute_error(y_test, test_pred_interaction) - mean_absolute_error(y_train, train_pred_interaction)
        }
    }

    # 3. Feature split frequency
    def extract_split_features(tree_json):
        tree = json.loads(tree_json)
        features = []

        def traverse(node):
            if 'split' in node:
                features.append(node['split'])
                traverse(node['children'][0])
                traverse(node['children'][1])

        traverse(tree)
        return features

    baseline_splits = [feat for tree in baseline_trees for feat in extract_split_features(tree)]
    interaction_splits = [feat for tree in interaction_trees for feat in extract_split_features(tree)]

    from collections import Counter

    baseline_split_counts = Counter(baseline_splits)
    interaction_split_counts = Counter(interaction_splits)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Tree depth distribution
    axes[0, 0].hist([baseline_depths, interaction_depths], label=['Baseline', 'Interaction'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Tree Depth Distribution')
    axes[0, 0].set_xlabel('Depth')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

    # Leaf count distribution
    axes[0, 1].hist([baseline_leaves, interaction_leaves], label=['Baseline', 'Interaction'], bins=20, alpha=0.7)
    axes[0, 1].set_title('Leaf Count Distribution')
    axes[0, 1].set_xlabel('Leaves')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # Train vs test MAE
    models = ['Baseline', 'Interaction']
    train_maes = [overfitting_stats['baseline']['train_mae'], overfitting_stats['interaction']['train_mae']]
    test_maes = [overfitting_stats['baseline']['test_mae'], overfitting_stats['interaction']['test_mae']]

    x = np.arange(len(models))
    width = 0.35

    axes[1, 0].bar(x - width/2, train_maes, width, label='Train MAE', alpha=0.8)
    axes[1, 0].bar(x + width/2, test_maes, width, label='Test MAE', alpha=0.8)
    axes[1, 0].set_title('Train vs Test MAE (Overfitting Check)')
    axes[1, 0].set_ylabel('MAE (yards)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].legend()

    # Feature split frequency (top 15)
    top_baseline_splits = dict(baseline_split_counts.most_common(15))
    top_interaction_splits = dict(interaction_split_counts.most_common(15))

    axes[1, 1].barh(list(top_interaction_splits.keys())[::-1], list(top_interaction_splits.values())[::-1])
    axes[1, 1].set_title('Top 15 Features by Split Frequency (Interaction Model)')
    axes[1, 1].set_xlabel('Number of Splits')

    plt.tight_layout()
    plt.savefig('docs/research/tree_complexity_analysis.png', dpi=300)

    return {
        'tree_stats': tree_stats,
        'overfitting_stats': overfitting_stats,
        'baseline_split_counts': baseline_split_counts,
        'interaction_split_counts': interaction_split_counts
    }
```

### Expected Findings

**If overfitting is the issue:**
- Interaction model has **deeper trees** (mean depth > 6 vs baseline 5)
- Interaction model has **larger train-test gap** (gap > 5 MAE vs baseline 2 MAE)
- Opponent features used in many splits but add complexity without improving test performance

**Action if confirmed:**
- Increase regularization (`min_child_weight`, `gamma`)
- Reduce `max_depth` from 6 to 4
- Add L1/L2 regularization (`reg_alpha`, `reg_lambda`)

### Success Criteria

- Train-test gap < 3 MAE for both models → No overfitting
- Interaction model depth similar to baseline (±0.5) → Not adding unnecessary complexity
- Opponent features used in < 20% of splits → Not dominant, likely noise

---

## Root Cause Hypothesis

### Scenario A: Feature Redundancy (Most Likely)

**Evidence Required:**
- High correlation (> 0.7) between `opp_def_epa_exp` and `passing_yards_L3`
- Opponent features have low VIF but low SHAP importance
- Interaction features don't beat baseline

**Root Cause:**
Player rolling averages (`passing_yards_L3`) already **implicitly** capture opponent strength because QBs who face tougher defenses naturally have lower L3 averages.

**Example:**
- QB faces 3 elite defenses → `passing_yards_L3` = 220 yards (naturally lower)
- Adding `opp_def_epa_exp` = -0.15 (elite defense) is **redundant**
- XGBoost picks noisy `opp_def_epa_exp` over informative `passing_yards_L3`, hurting predictions

**Solution:**
- Use ridge residuals (Approach 2) to isolate QB performance **beyond** opponent expectation
- Or abandon opponent features entirely (keep v2 baseline)

### Scenario B: Data Quality Issues (Medium Likelihood)

**Evidence Required:**
- Opponent features show weak correlation (< 0.1) with passing yards
- High temporal volatility (week-to-week EPA changes > 0.5 std)
- Outliers or missing values > 10%

**Root Cause:**
Opponent defensive EPA is **noisy** due to:
- Garbage time effects (prevent defense in blowouts)
- Small sample size (defensive EPA based on 10-15 pass attempts per game)
- Non-stationary (defensive performance changes drastically within season)

**Solution:**
- Use season-long opponent averages (more stable)
- Switch to Elo ratings or Sagarin ratings (already implemented for ensemble v6)
- Clean outliers and smooth temporal changes

### Scenario C: Interaction Design Flaws (Low Likelihood)

**Evidence Required:**
- Interaction features show high importance (> 10%) but still regress
- SHAP values inconsistent (50% positive, 50% negative)
- Error decomposition shows interaction features hurt in all segments

**Root Cause:**
Interaction features are **poorly designed**:
- `yards_L3 × opp_difficulty` should be `yards_L3 / opp_def_epa_exp` (division, not multiplication)
- Categorical bins (elite_def, weak_def) too coarse
- Missing game context interactions (opponent × total_line)

**Solution:**
- Redesign interactions with domain expertise
- Test ratio-based interactions (QB efficiency relative to opponent)
- Add three-way interactions (QB × opponent × game context)

### Scenario D: Architectural Incompatibility (Lowest Likelihood)

**Evidence Required:**
- ALL approaches fail (interactions, ridge residuals, L1 regularization, categorical)
- Linear models (ridge, lasso) also fail with opponent features
- Only ensemble with separate opponent-specific model works

**Root Cause:**
Tree-based models fundamentally **cannot properly contextualize** opponent strength because:
- Trees split on single features, not feature combinations
- Opponent adjustments require **multiplicative** interactions, not additive
- Non-linear opponent effects (diminishing returns) not captured by trees

**Solution:**
- Use opponent features ONLY in linear models (ridge regression)
- Ensemble XGBoost (no opponent) with ridge regression (with opponent)
- Switch to neural networks (can model complex interactions)

---

## Actionable Insights

### If Root Cause is A (Feature Redundancy)

**Recommendation:** Try Approach 2 (ridge residuals) ONCE, then abandon opponent features.

**Rationale:**
- Ridge residuals are theoretically sound and address redundancy directly
- If ridge residuals fail, opponent features are fundamentally incompatible with XGBoost
- Don't waste more time on variations (interactions, regularization, categorical)

**Timeline:** 2 hours (1 hour implementation, 1 hour validation)

### If Root Cause is B (Data Quality)

**Recommendation:** Clean opponent data and test with smoothed metrics.

**Actions:**
1. Remove outliers (> 3 standard deviations)
2. Use exponentially weighted moving average (EWMA) for opponent EPA
3. Impute missing values with league median
4. Test on single season (2024) first

**Timeline:** 3-4 hours (data cleaning + retraining)

### If Root Cause is C (Interaction Design)

**Recommendation:** Redesign interactions with domain expertise and test narrow use case.

**Actions:**
1. Switch from multiplication to division: `yards_L3 / opp_def_epa_exp`
2. Add three-way interactions: `yards_L3 × opp_difficulty × high_total`
3. Use finer categorical bins (deciles, not tertiles)
4. Test on subset (QBs vs elite defenses only)

**Timeline:** 4-5 hours (redesign + testing)

### If Root Cause is D (Architectural)

**Recommendation:** Abandon XGBoost opponent features, use ensemble with ridge regression.

**Actions:**
1. Train separate ridge regression model with opponent features
2. Ensemble: 80% XGBoost (no opponent) + 20% ridge (with opponent)
3. Document in POSTMORTEMS.md
4. Update CLAUDE.md: "Opponent features incompatible with XGBoost for NFL QB prediction"

**Timeline:** 2 hours (ensemble implementation)

---

## Go/No-Go Decision Framework

### After Experiment 1 (Interactions) Completes

| Result | Decision | Next Action | Timeline |
|--------|----------|-------------|----------|
| **MAE < 40.20** | ✓ BREAKTHROUGH | Deploy immediately, run multi-season validation | 1 day |
| **40.20 ≤ MAE < 42.00** | ⚠ MARGINAL | Run Analysis 1-5, try Approach 2 (ridge residuals) | 1-2 days |
| **42.00 ≤ MAE < 45.00** | ⚠ INVESTIGATE | Run Analysis 1-3, try Approach 5 (L1 regularization) | 2-3 days |
| **MAE ≥ 45.00** | ✗ ABANDON | Document failure, return to v2 baseline | 0.5 day |

### After BFS Experiments 1-7 Complete

| Scenario | Decision | Rationale |
|----------|----------|-----------|
| **Any experiment beats 40.20 MAE** | Proceed to DFS | At least one approach works, explore deeper |
| **All experiments fail (MAE > 42.00)** | ABANDON opponent features | 7 approaches failed, likely fundamental incompatibility |
| **Experiments show marginal improvement (40-42 MAE)** | Cost-benefit analysis | Complexity vs. 0-5% improvement |

### Final Go/No-Go Decision

**DEPLOY if:**
- ✓ MAE < 40.20 (beats baseline)
- ✓ Multi-season validation consistent (±2 MAE across 2022-2024)
- ✓ Statistical significance (Diebold-Mariano p < 0.05)
- ✓ No overfitting (train-test gap < 3 MAE)
- ✓ Feature importance sensible (opponent interactions in top 10)

**ABANDON if:**
- ✗ All BFS experiments fail (MAE > 42.00)
- ✗ Ridge residuals (Approach 2) fail
- ✗ L1 regularization (Approach 5) fails
- ✗ Data quality audit reveals systematic issues
- ✗ Error decomposition shows opponent features hurt in all segments

**INVESTIGATE FURTHER if:**
- ⚠ Marginal improvement (40-42 MAE) but not statistically significant
- ⚠ Works for specific contexts only (elite defenses, high-scoring games)
- ⚠ Ensemble with ridge regression shows promise

---

## Execution Checklist

### Phase 1: Immediate (Once Docker Available)

- [ ] **Run Experiment 1** (interactions) - 30-45 minutes
- [ ] **Check MAE result** - 2 minutes
- [ ] **IF MAE < 42.00:** Run Analysis 1 (correlation) - 20 minutes
- [ ] **IF MAE ≥ 42.00:** Skip to Phase 3 (abandon)

### Phase 2: Deep Dive (If Experiment 1 Shows Promise)

- [ ] **Analysis 1:** Feature correlation & VIF - 30 minutes
- [ ] **Analysis 2:** SHAP value investigation - 45 minutes
- [ ] **Analysis 3:** Data quality audit - 30 minutes
- [ ] **Analysis 4:** Error decomposition - 45 minutes
- [ ] **Analysis 5:** Tree depth analysis - 30 minutes
- [ ] **Synthesize findings** - 30 minutes
- [ ] **Write root cause summary** - 15 minutes

### Phase 3: Decision & Documentation

- [ ] **Go/No-Go decision** - 15 minutes
- [ ] **Update POSTMORTEMS.md** (if abandon) - 30 minutes
- [ ] **Update CLAUDE.md** - 15 minutes
- [ ] **Create final analysis document** - 30 minutes
- [ ] **Git commit with findings** - 10 minutes

**Total Time Estimate:**
- Phase 1: 32-47 minutes
- Phase 2 (if needed): 3.5 hours
- Phase 3: 1.5 hours
- **Grand Total: 5-6 hours** (if full investigation required)

---

## Expected Outputs

### Deliverable 1: Root Cause Summary (2-3 sentences)

**Template:**
> XGBoost opponent features failed because [ROOT CAUSE: redundancy/data quality/interaction design/architectural]. Analysis revealed [KEY EVIDENCE: correlation/SHAP/data quality metrics]. Recommendation: [ABANDON/PROCEED WITH APPROACH X/ENSEMBLE STRATEGY].

**Example (Feature Redundancy):**
> XGBoost opponent features failed due to feature redundancy - player rolling averages (`passing_yards_L3`) already implicitly capture opponent strength through historical performance. Correlation analysis revealed 0.82 correlation between `opp_def_epa_exp` and `passing_yards_L3`, and SHAP values showed opponent features had inconsistent directionality (48% positive, 52% negative). Recommendation: ABANDON opponent features for XGBoost, use ridge regression opponent-adjusted residuals in ensemble instead.

### Deliverable 2: Statistical Evidence (Visualizations + Tables)

**Required Plots:**
1. `xgb_correlation_heatmap.png` - Feature correlation matrix
2. `xgb_shap_comparison.png` - SHAP value comparison (baseline vs interaction)
3. `opponent_feature_distributions.png` - Data quality distributions
4. `error_decomposition_analysis.png` - Error analysis by segment
5. `tree_complexity_analysis.png` - Tree depth and overfitting metrics

**Required Tables:**
1. High correlation pairs (> 0.7)
2. VIF scores
3. SHAP importance rankings
4. Opponent feature signal-to-noise ratios
5. Error decomposition by opponent strength/QB performance/game context/season
6. Overfitting statistics (train vs test MAE)

### Deliverable 3: Actionable Insights (Decision Tree)

```
IF Experiment 1 MAE < 40.20:
    → DEPLOY immediately
    → Multi-season validation
    → Production integration

ELSE IF 40.20 ≤ MAE < 42.00:
    IF Analysis 1 shows high correlation (> 0.7):
        → Try Approach 2 (ridge residuals)
        IF ridge residuals beat baseline:
            → DEPLOY ridge residuals
        ELSE:
            → ABANDON opponent features
    ELSE IF Analysis 3 shows data quality issues:
        → Clean data, retry Experiment 1
    ELSE:
        → Try Approach 5 (L1 regularization)

ELSE IF 42.00 ≤ MAE < 45.00:
    → Run Analysis 2 (SHAP)
    → IF SHAP shows systematic bias:
        → Redesign interactions (Approach C)
    → ELSE:
        → Try Approach 5 (L1 regularization)

ELSE (MAE ≥ 45.00):
    → ABANDON immediately
    → Document in POSTMORTEMS.md
    → Return to v2 baseline
```

### Deliverable 4: Recommendation (Production Decision)

**Template:**

```markdown
## XGBoost Opponent Features: Final Recommendation

**Date:** [YYYY-MM-DD]
**Analysis Duration:** [X hours]
**Root Cause:** [Feature Redundancy / Data Quality / Interaction Design / Architectural]

### Decision: [DEPLOY / ABANDON / CONTINUE INVESTIGATION]

**Rationale:**
[2-3 sentences explaining why]

**Evidence:**
1. [Key finding 1]
2. [Key finding 2]
3. [Key finding 3]

**Action Plan:**
1. [Immediate action]
2. [Follow-up action]
3. [Documentation]

**Risk Assessment:**
- Probability of success: [X%]
- Expected MAE improvement: [+X% to +Y%]
- Time to production: [X days]

**Alternatives Considered:**
1. [Alternative 1]: [Reason rejected]
2. [Alternative 2]: [Reason rejected]

**Sign-off:**
- [ ] Multi-season validation complete
- [ ] Statistical significance confirmed
- [ ] Production integration tested
- [ ] Documentation updated
- [ ] POSTMORTEMS.md updated (if abandoned)
```

---

## Summary

This analysis framework provides a **systematic, evidence-based approach** to diagnosing why XGBoost opponent features have failed 3 times and determining whether to continue investigation or abandon the approach entirely.

**Key Principles:**
1. **Data-driven:** Every decision backed by statistical evidence
2. **Time-boxed:** Maximum 6 hours for full investigation
3. **Go/No-Go gates:** Clear decision criteria at each phase
4. **Comprehensive:** 5 complementary analyses cover all failure modes
5. **Actionable:** Specific recommendations, not vague suggestions

**Critical Question Answered:**
> **Is there ANY way opponent features can help XGBoost?**

**Answer will be one of:**
- **A.** YES - Ridge residuals (Approach 2) or improved interactions work → DEPLOY
- **B.** MARGINALLY - Works for niche contexts only → Cost-benefit decision
- **C.** NO - All approaches fail, fundamental incompatibility → ABANDON

**Next Step:** Execute Experiment 1 (interactions) once Docker is running, then run this analysis framework.

---

**Created:** November 14, 2025
**Status:** READY FOR EXECUTION
**Prerequisites:** Docker running, database accessible, Experiment 1 trained
**Estimated Time:** 5-6 hours for full analysis
**Deliverables:** Root cause summary, statistical evidence, actionable insights, production decision
