#!/usr/bin/env python3
"""
Propensity Score Matching (PSM) Pipeline for Travel & Weather Treatment Effects

Implements PSM to estimate causal effects of:
1. Long-distance travel on QB passing performance
2. Extreme weather on QB passing performance

Uses propensity score matching with logistic regression and nearest-neighbor matching.

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass

# Statistical libraries
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


@dataclass
class PSMResults:
    """Results from PSM analysis."""
    treatment_effect: float
    std_error: float
    p_value: float
    conf_int_lower: float
    conf_int_upper: float
    n_treated: int
    n_control: int
    n_matched_pairs: int
    covariate_balance: Dict[str, float]
    treated_mean: float
    control_mean: float


def load_combined_data() -> pd.DataFrame:
    """
    Load and combine travel, weather, and QB stats data.

    Returns:
        Combined DataFrame
    """
    data_dir = project_root / 'data'

    # Load travel features
    df_travel = pd.read_csv(data_dir / 'travel_features_2018_2024.csv')

    # Load weather features
    df_weather = pd.read_csv(data_dir / 'historical_weather_2018_2024.csv')

    # Merge on game_id
    df = df_travel.merge(
        df_weather[['game_id', 'temp_numeric', 'wind_numeric', 'is_bad_weather',
                    'is_extreme_cold', 'is_high_wind', 'is_precipitation',
                    'weather_classification', 'roof']],
        on='game_id',
        how='left'
    )

    return df


def prepare_psm_data(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for PSM analysis.

    Args:
        df: Full DataFrame
        treatment_col: Name of treatment indicator column
        outcome_col: Name of outcome variable column
        covariate_cols: List of covariate column names

    Returns:
        Tuple of (treated DataFrame, control DataFrame)
    """
    # Filter to non-null outcome
    df_clean = df[df[outcome_col].notna()].copy()

    # Separate treated and control
    df_treated = df_clean[df_clean[treatment_col] == True].copy()
    df_control = df_clean[df_clean[treatment_col] == False].copy()

    # Remove rows with missing covariates
    all_cols = [outcome_col] + covariate_cols
    df_treated = df_treated.dropna(subset=all_cols)
    df_control = df_control.dropna(subset=all_cols)

    return df_treated, df_control


def estimate_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: List[str]
) -> np.ndarray:
    """
    Estimate propensity scores using logistic regression.

    Args:
        df: DataFrame with treatment and covariates
        treatment_col: Name of treatment column
        covariate_cols: List of covariate column names

    Returns:
        Array of propensity scores
    """
    # Prepare features
    X = df[covariate_cols].values
    y = df[treatment_col].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    # Predict propensity scores
    propensity_scores = model.predict_proba(X_scaled)[:, 1]

    return propensity_scores


def nearest_neighbor_matching(
    treated_ps: np.ndarray,
    control_ps: np.ndarray,
    caliper: float = 0.2
) -> Dict[int, List[int]]:
    """
    Perform nearest-neighbor matching with replacement.

    Args:
        treated_ps: Propensity scores for treated units
        control_ps: Propensity scores for control units
        caliper: Maximum allowable propensity score difference (in std dev)

    Returns:
        Dictionary mapping treated index to list of matched control indices
    """
    # Calculate caliper in propensity score units
    ps_std = np.std(np.concatenate([treated_ps, control_ps]))
    caliper_value = caliper * ps_std

    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
    nn.fit(control_ps.reshape(-1, 1))

    # Find matches for each treated unit
    matches = {}
    for i, ps in enumerate(treated_ps):
        distances, indices = nn.kneighbors([[ps]])

        # Filter by caliper
        valid_matches = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist <= caliper_value:
                valid_matches.append(idx)

        if valid_matches:
            matches[i] = valid_matches

    return matches


def compute_treatment_effect(
    df_treated: pd.DataFrame,
    df_control: pd.DataFrame,
    matches: Dict[int, List[int]],
    outcome_col: str
) -> PSMResults:
    """
    Compute average treatment effect on the treated (ATT).

    Args:
        df_treated: Treated units DataFrame
        df_control: Control units DataFrame
        matches: Matching dictionary
        outcome_col: Name of outcome variable

    Returns:
        PSMResults object
    """
    # Get matched pairs
    treated_outcomes = []
    control_outcomes = []

    treated_indices = list(df_treated.index)
    control_indices = list(df_control.index)

    for treated_idx_pos, control_idx_list in matches.items():
        treated_idx = treated_indices[treated_idx_pos]
        treated_outcome = df_treated.loc[treated_idx, outcome_col]

        # Average over matched controls
        control_outcome_list = [
            df_control.loc[control_indices[ctrl_idx], outcome_col]
            for ctrl_idx in control_idx_list
        ]
        control_outcome = np.mean(control_outcome_list)

        treated_outcomes.append(treated_outcome)
        control_outcomes.append(control_outcome)

    # Convert to arrays
    treated_outcomes = np.array(treated_outcomes)
    control_outcomes = np.array(control_outcomes)

    # Compute treatment effect
    individual_effects = treated_outcomes - control_outcomes
    att = np.mean(individual_effects)
    std_error = stats.sem(individual_effects)

    # Confidence interval
    conf_int = stats.t.interval(
        0.95,
        len(individual_effects) - 1,
        loc=att,
        scale=std_error
    )

    # P-value from t-test
    t_stat, p_value = stats.ttest_1samp(individual_effects, 0)

    # Create results object
    results = PSMResults(
        treatment_effect=att,
        std_error=std_error,
        p_value=p_value,
        conf_int_lower=conf_int[0],
        conf_int_upper=conf_int[1],
        n_treated=len(df_treated),
        n_control=len(df_control),
        n_matched_pairs=len(matches),
        covariate_balance={},  # Will be computed separately
        treated_mean=np.mean(treated_outcomes),
        control_mean=np.mean(control_outcomes)
    )

    return results


def compute_covariate_balance(
    df_treated: pd.DataFrame,
    df_control: pd.DataFrame,
    matches: Dict[int, List[int]],
    covariate_cols: List[str]
) -> Dict[str, float]:
    """
    Compute standardized mean differences for covariate balance.

    Args:
        df_treated: Treated units DataFrame
        df_control: Control units DataFrame
        matches: Matching dictionary
        covariate_cols: List of covariate column names

    Returns:
        Dictionary of standardized mean differences
    """
    balance = {}

    treated_indices = list(df_treated.index)
    control_indices = list(df_control.index)

    for covar in covariate_cols:
        # Get matched treated and control covariate values
        treated_vals = []
        control_vals = []

        for treated_idx_pos, control_idx_list in matches.items():
            treated_idx = treated_indices[treated_idx_pos]
            treated_val = df_treated.loc[treated_idx, covar]

            control_val_list = [
                df_control.loc[control_indices[ctrl_idx], covar]
                for ctrl_idx in control_idx_list
            ]
            control_val = np.mean(control_val_list)

            treated_vals.append(treated_val)
            control_vals.append(control_val)

        # Compute standardized mean difference
        mean_diff = np.mean(treated_vals) - np.mean(control_vals)
        pooled_std = np.sqrt(
            (np.var(treated_vals) + np.var(control_vals)) / 2
        )

        smd = mean_diff / pooled_std if pooled_std > 0 else 0.0
        balance[covar] = abs(smd)

    return balance


def run_psm_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: List[str],
    caliper: float = 0.2
) -> PSMResults:
    """
    Run complete PSM analysis pipeline.

    Args:
        df: Full DataFrame
        treatment_col: Treatment indicator column
        outcome_col: Outcome variable column
        covariate_cols: List of covariates for matching
        caliper: Caliper for propensity score matching

    Returns:
        PSMResults object
    """
    print(f"\nRunning PSM analysis: {treatment_col} → {outcome_col}")
    print("=" * 70)

    # Prepare data
    df_treated, df_control = prepare_psm_data(df, treatment_col, outcome_col, covariate_cols)
    print(f"Treated units: {len(df_treated)}")
    print(f"Control units: {len(df_control)}")

    # Estimate propensity scores
    print("\nEstimating propensity scores...")
    df_combined = pd.concat([df_treated, df_control])
    ps = estimate_propensity_scores(df_combined, treatment_col, covariate_cols)

    treated_ps = ps[:len(df_treated)]
    control_ps = ps[len(df_treated):]

    # Perform matching
    print(f"Performing nearest-neighbor matching (caliper={caliper})...")
    matches = nearest_neighbor_matching(treated_ps, control_ps, caliper=caliper)
    print(f"Matched pairs: {len(matches)}")

    # Compute treatment effect
    print("\nComputing treatment effect...")
    results = compute_treatment_effect(df_treated, df_control, matches, outcome_col)

    # Compute covariate balance
    print("Checking covariate balance...")
    balance = compute_covariate_balance(df_treated, df_control, matches, covariate_cols)
    results.covariate_balance = balance

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Treatment Effect (ATT): {results.treatment_effect:.2f} yards")
    print(f"Standard Error:         {results.std_error:.2f}")
    print(f"95% Confidence Int:     [{results.conf_int_lower:.2f}, {results.conf_int_upper:.2f}]")
    print(f"P-value:                {results.p_value:.4f}")
    print(f"Treated mean:           {results.treated_mean:.2f} yards")
    print(f"Control mean:           {results.control_mean:.2f} yards")
    print()

    print("Covariate Balance (Standardized Mean Difference):")
    for covar, smd in balance.items():
        status = "✓" if smd < 0.1 else "✗"
        print(f"  {status} {covar:30s}: {smd:.4f}")

    return results


def sensitivity_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: List[str],
    calipers: List[float] = [0.1, 0.2, 0.3]
) -> Dict[float, PSMResults]:
    """
    Run sensitivity analysis with different caliper values.

    Args:
        df: Full DataFrame
        treatment_col: Treatment column
        outcome_col: Outcome column
        covariate_cols: Covariates
        calipers: List of caliper values to test

    Returns:
        Dictionary mapping caliper to results
    """
    results = {}

    for caliper in calipers:
        print(f"\n{'=' * 70}")
        print(f"SENSITIVITY ANALYSIS: Caliper = {caliper}")
        print(f"{'=' * 70}")

        result = run_psm_analysis(df, treatment_col, outcome_col, covariate_cols, caliper=caliper)
        results[caliper] = result

    return results


def export_matched_pairs(
    df_treated: pd.DataFrame,
    df_control: pd.DataFrame,
    matches: Dict[int, List[int]],
    output_path: Path
):
    """
    Export matched pairs to CSV.

    Args:
        df_treated: Treated DataFrame
        df_control: Control DataFrame
        matches: Matching dictionary
        output_path: Output file path
    """
    matched_data = []

    treated_indices = list(df_treated.index)
    control_indices = list(df_control.index)

    for treated_idx_pos, control_idx_list in matches.items():
        treated_idx = treated_indices[treated_idx_pos]

        for ctrl_idx in control_idx_list:
            control_idx = control_indices[ctrl_idx]

            matched_data.append({
                'treated_game_id': df_treated.loc[treated_idx, 'game_id'],
                'control_game_id': df_control.loc[control_idx, 'game_id'],
                'treated_season': df_treated.loc[treated_idx, 'season'],
                'treated_week': df_treated.loc[treated_idx, 'week'],
                'control_season': df_control.loc[control_idx, 'season'],
                'control_week': df_control.loc[control_idx, 'week'],
            })

    df_matched = pd.DataFrame(matched_data)
    df_matched.to_csv(output_path, index=False)
    print(f"✓ Saved matched pairs: {output_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("PROPENSITY SCORE MATCHING (PSM) PIPELINE")
    print("Travel & Weather Treatment Effects on QB Passing Performance")
    print("=" * 80)

    # Load data
    print("\nLoading combined data...")
    df = load_combined_data()
    print(f"✓ Loaded {len(df)} games")

    data_dir = project_root / 'data'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # ========================================================================
    # ANALYSIS 1: Long-Distance Travel Treatment
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: LONG-DISTANCE TRAVEL EFFECT")
    print("=" * 80)

    travel_covariates = [
        'spread_close',        # Opponent quality
        'week',                # Game importance
        'away_days_rest',      # Rest days
    ]

    # Filter to games with valid data
    df_travel = df[
        df['away_passing_yards'].notna() &
        df['spread_close'].notna()
    ].copy()

    print(f"\nAnalyzing {len(df_travel)} games with complete travel data...")

    travel_results = run_psm_analysis(
        df_travel,
        treatment_col='is_long_distance_travel',
        outcome_col='away_passing_yards',
        covariate_cols=travel_covariates,
        caliper=0.2
    )

    # ========================================================================
    # ANALYSIS 2: Extreme Weather Treatment
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: EXTREME WEATHER EFFECT")
    print("=" * 80)

    weather_covariates = [
        'spread_close',        # Opponent quality
        'week',                # Game importance
    ]

    # Filter to outdoor games only (exclude domes)
    df_weather = df[
        df['away_passing_yards'].notna() &
        df['spread_close'].notna() &
        (df['roof'].isin(['outdoors', 'open']))
    ].copy()

    print(f"\nAnalyzing {len(df_weather)} outdoor games with weather data...")

    weather_results = run_psm_analysis(
        df_weather,
        treatment_col='is_bad_weather',
        outcome_col='away_passing_yards',
        covariate_cols=weather_covariates,
        caliper=0.2
    )

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    import json

    all_results = {
        'travel_treatment': {
            'treatment_effect': travel_results.treatment_effect,
            'std_error': travel_results.std_error,
            'p_value': travel_results.p_value,
            'conf_int_lower': travel_results.conf_int_lower,
            'conf_int_upper': travel_results.conf_int_upper,
            'n_treated': travel_results.n_treated,
            'n_control': travel_results.n_control,
            'n_matched_pairs': travel_results.n_matched_pairs,
            'covariate_balance': travel_results.covariate_balance,
            'treated_mean': travel_results.treated_mean,
            'control_mean': travel_results.control_mean
        },
        'weather_treatment': {
            'treatment_effect': weather_results.treatment_effect,
            'std_error': weather_results.std_error,
            'p_value': weather_results.p_value,
            'conf_int_lower': weather_results.conf_int_lower,
            'conf_int_upper': weather_results.conf_int_upper,
            'n_treated': weather_results.n_treated,
            'n_control': weather_results.n_control,
            'n_matched_pairs': weather_results.n_matched_pairs,
            'covariate_balance': weather_results.covariate_balance,
            'treated_mean': weather_results.treated_mean,
            'control_mean': weather_results.control_mean
        }
    }

    results_path = results_dir / 'psm_treatment_effects.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved: {results_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print("1. LONG-DISTANCE TRAVEL EFFECT:")
    print(f"   Treatment: ≥2000 miles coast-to-coast travel")
    print(f"   Effect:    {travel_results.treatment_effect:.1f} yards "
          f"[{travel_results.conf_int_lower:.1f}, {travel_results.conf_int_upper:.1f}]")
    print(f"   P-value:   {travel_results.p_value:.4f}")
    print(f"   Matched:   {travel_results.n_matched_pairs} pairs")
    print()
    print("2. EXTREME WEATHER EFFECT:")
    print(f"   Treatment: Temp <32°F OR wind >20mph OR precipitation")
    print(f"   Effect:    {weather_results.treatment_effect:.1f} yards "
          f"[{weather_results.conf_int_lower:.1f}, {weather_results.conf_int_upper:.1f}]")
    print(f"   P-value:   {weather_results.p_value:.4f}")
    print(f"   Matched:   {weather_results.n_matched_pairs} pairs")
    print()
    print("✓ PSM analysis complete!")


if __name__ == '__main__':
    main()
