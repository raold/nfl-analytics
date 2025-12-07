"""
Difference-in-Differences Analysis: 2018 Roughing-the-Passer Rule Change
==========================================================================

This script implements a complete DiD pipeline for the 2018 NFL roughing-the-passer
rule change, which serves as a natural experiment for causal inference.

Background:
-----------
In 2018, the NFL significantly tightened enforcement of roughing-the-passer penalties.
This created a natural experiment where defenses with historically high roughing rates
in 2017 (pre-treatment) were "treated" more heavily post-2018.

Expected Effect:
----------------
QB passing yards should increase more when facing high-roughing defenses post-2018,
as these defenses must adjust their pass rush techniques.

Treatment Assignment:
---------------------
- Treatment Group: Defenses in top 2 quintiles of 2017 roughing rate
- Control Group: Defenses in bottom 3 quintiles of 2017 roughing rate
- Treatment Period: 2018-2024
- Pre-Treatment Period: 2015-2017 (for parallel trends validation)

Outputs:
--------
1. data/did_roughing_defense_classifications_2017.csv - Defense baseline rates
2. data/did_roughing_treatment_2017_2024.csv - Full treatment dataset
3. results/did_roughing_validation.json - Validation results

Author: NFL Analytics Research Team
Date: November 16, 2025
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

warnings.filterwarnings('ignore')

# Database connection
DB_URL = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"  # pragma: allowlist secret

# Ensure output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Create SQLAlchemy engine
engine = create_engine(DB_URL)


def get_db_connection():
    """Get SQLAlchemy engine for pandas read_sql operations."""
    return engine


def extract_2017_roughing_baseline() -> pd.DataFrame:
    """
    Extract 2017 roughing-the-passer penalties per defense.

    Returns defense-level statistics:
    - Total roughing penalties
    - Total pass attempts faced
    - Roughing rate (penalties per 100 pass attempts)
    - Quintile classification
    """
    print("\n" + "="*80)
    print("STEP 1: Extract 2017 Defense Roughing Baseline")
    print("="*80)

    query = """
    WITH defense_pass_attempts AS (
        -- Count pass attempts faced by each defense in 2017
        SELECT
            p.defteam AS defense_team,
            COUNT(*) AS pass_attempts_faced
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.season = 2017
            AND g.week >= 1  -- Exclude preseason/aggregates
            AND p.pass_attempt = TRUE
            AND p.aborted_play = FALSE  -- Exclude aborted plays
        GROUP BY p.defteam
    ),
    defense_roughing AS (
        -- Count roughing-the-passer penalties by defense in 2017
        SELECT
            p.penalty_team AS defense_team,
            COUNT(*) AS roughing_penalties
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.season = 2017
            AND g.week >= 1
            AND p.penalty = 1
            AND p.penalty_type = 'Roughing the Passer'
        GROUP BY p.penalty_team
    )
    SELECT
        dpa.defense_team,
        COALESCE(dr.roughing_penalties, 0) AS roughing_penalties,
        dpa.pass_attempts_faced,
        (COALESCE(dr.roughing_penalties, 0)::NUMERIC / dpa.pass_attempts_faced) * 100
            AS roughing_rate_per_100_attempts
    FROM defense_pass_attempts dpa
    LEFT JOIN defense_roughing dr ON dpa.defense_team = dr.defense_team
    ORDER BY roughing_rate_per_100_attempts DESC;
    """

    df = pd.read_sql_query(query, engine)

    # Ensure numeric types
    df['roughing_penalties'] = pd.to_numeric(df['roughing_penalties'], errors='coerce')
    df['pass_attempts_faced'] = pd.to_numeric(df['pass_attempts_faced'], errors='coerce')
    df['roughing_rate_per_100_attempts'] = pd.to_numeric(df['roughing_rate_per_100_attempts'], errors='coerce')

    # Classify defenses into quintiles using rank percentiles (handles ties better)
    df['roughing_percentile'] = df['roughing_rate_per_100_attempts'].rank(pct=True)

    # Create quintile labels based on percentile thresholds
    df['quintile'] = pd.cut(
        df['roughing_percentile'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Conservative', 'Conservative', 'Moderate', 'Aggressive', 'Very Aggressive'],
        include_lowest=True
    )

    # Binary treatment indicator (top 2 quintiles = treated)
    df['is_high_roughing'] = df['quintile'].isin(['Aggressive', 'Very Aggressive'])

    print(f"\nExtracted data for {len(df)} defenses")
    print("\nQuintile Distribution:")
    print(df['quintile'].value_counts().sort_index())
    print("\nTop 5 Most Aggressive Defenses (2017):")
    print(df.head()[['defense_team', 'roughing_penalties', 'pass_attempts_faced',
                      'roughing_rate_per_100_attempts', 'quintile']])
    print("\nBottom 5 Most Conservative Defenses (2017):")
    print(df.tail()[['defense_team', 'roughing_penalties', 'pass_attempts_faced',
                      'roughing_rate_per_100_attempts', 'quintile']])

    # Save classifications
    output_path = "data/did_roughing_defense_classifications_2017.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved defense classifications to {output_path}")

    return df


def extract_qb_performance_data(defense_classifications: pd.DataFrame) -> pd.DataFrame:
    """
    Extract QB game-level performance data (2015-2024) with opponent defense treatment status.

    Includes:
    - QB passing yards (outcome variable)
    - Opponent defense's 2017 roughing classification
    - Control variables: home/away, spread, weather, dome
    - Pre-treatment (2015-2017) and post-treatment (2018-2024) periods
    """
    print("\n" + "="*80)
    print("STEP 2: Extract QB Performance Data (2015-2024)")
    print("="*80)

    # Create defense classification mapping
    defense_map = defense_classifications.set_index('defense_team')[
        ['roughing_rate_per_100_attempts', 'quintile', 'is_high_roughing']
    ].to_dict('index')

    query = """
    WITH qb_games AS (
        -- Aggregate QB passing stats per game from plays table
        SELECT
            p.game_id,
            g.season,
            g.week,
            p.passer_player_id AS player_id,
            p.passer_player_name AS player_name,
            p.posteam AS qb_team,
            p.defteam AS defense_team,
            SUM(p.passing_yards) AS passing_yards,
            COUNT(CASE WHEN p.pass_attempt = TRUE THEN 1 END) AS pass_attempts,
            COUNT(CASE WHEN p.complete_pass = 1 THEN 1 END) AS completions,
            COUNT(CASE WHEN p.touchdown = 1 AND p.td_team = p.posteam THEN 1 END) AS passing_tds,
            COUNT(CASE WHEN p.interception = 1 THEN 1 END) AS interceptions
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.passer_player_id IS NOT NULL
            AND g.season BETWEEN 2015 AND 2024
            AND g.week >= 1  -- Regular season only
            AND p.aborted_play = FALSE
        GROUP BY p.game_id, g.season, g.week, p.passer_player_id, p.passer_player_name,
                 p.posteam, p.defteam
        HAVING COUNT(CASE WHEN p.pass_attempt = TRUE THEN 1 END) > 0
    )
    SELECT
        qb.player_id,
        qb.player_name,
        qb.season,
        qb.week,
        qb.qb_team,
        qb.defense_team,
        qb.passing_yards,
        qb.pass_attempts,
        qb.completions,
        qb.passing_tds,
        qb.interceptions,
        -- Game context
        CASE
            WHEN qb.qb_team = g.home_team THEN 'home'
            WHEN qb.qb_team = g.away_team THEN 'away'
            ELSE NULL
        END AS location,
        g.spread_close,
        g.total_close,
        g.roof,
        g.surface,
        -- Game outcome
        CASE
            WHEN qb.qb_team = g.home_team THEN g.home_score
            WHEN qb.qb_team = g.away_team THEN g.away_score
            ELSE NULL
        END AS team_score,
        CASE
            WHEN qb.qb_team = g.home_team THEN g.away_score
            WHEN qb.qb_team = g.away_team THEN g.home_score
            ELSE NULL
        END AS opponent_score
    FROM qb_games qb
    JOIN games g ON qb.game_id = g.game_id
    ORDER BY qb.season, qb.week, qb.player_id;
    """

    print("Executing query (this may take 30-60 seconds)...")
    df = pd.read_sql_query(query, engine)

    print(f"Extracted {len(df):,} QB game observations")

    # Join with defense classifications
    df['defense_2017_roughing_rate'] = df['defense_team'].map(
        lambda x: defense_map.get(x, {}).get('roughing_rate_per_100_attempts', np.nan)
    )
    df['defense_quintile'] = df['defense_team'].map(
        lambda x: defense_map.get(x, {}).get('quintile', 'Unknown')
    )
    df['is_treated'] = df['defense_team'].map(
        lambda x: defense_map.get(x, {}).get('is_high_roughing', False)
    )

    # Create time period indicators
    df['post_treatment'] = (df['season'] >= 2018).astype(int)
    df['pre_treatment'] = (df['season'] <= 2017).astype(int)

    # DiD interaction term
    df['treated_x_post'] = df['is_treated'] * df['post_treatment']

    # Handle missing values
    missing_defenses = df[df['defense_2017_roughing_rate'].isna()]['defense_team'].unique()
    if len(missing_defenses) > 0:
        print(f"\n⚠️  Warning: {len(missing_defenses)} defenses missing 2017 baseline data:")
        print(f"   {', '.join(missing_defenses)}")
        print("   These observations will be excluded from DiD analysis")

    # Summary statistics
    print("\nData Summary:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Unique QBs: {df['player_id'].nunique()}")
    print(f"  Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"  Pre-treatment (2015-2017): {len(df[df['pre_treatment']==1]):,} games")
    print(f"  Post-treatment (2018-2024): {len(df[df['post_treatment']==1]):,} games")
    print("\nTreatment Assignment:")
    print(f"  High-roughing defenses (treated): {df['is_treated'].sum():,} games")
    print(f"  Low-roughing defenses (control): {(~df['is_treated']).sum():,} games")

    # Save full dataset
    output_path = "data/did_roughing_treatment_2015_2024.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved treatment dataset to {output_path}")

    return df


def calculate_did_treatment_effect(df: pd.DataFrame) -> dict:
    """
    Calculate Difference-in-Differences treatment effect.

    DiD Estimator:
    --------------
    β = [E(Y|Treated=1, Post=1) - E(Y|Treated=1, Post=0)] -
        [E(Y|Treated=0, Post=1) - E(Y|Treated=0, Post=0)]

    Where Y = QB passing yards

    Returns:
    --------
    Dictionary with treatment effect, standard error, p-value, and confidence interval
    """
    print("\n" + "="*80)
    print("STEP 3: Calculate DiD Treatment Effect")
    print("="*80)

    # Exclude observations with missing defense classifications
    df_clean = df.dropna(subset=['defense_2017_roughing_rate'])

    # Calculate group means
    treated_post = df_clean[(df_clean['is_treated']) & (df_clean['post_treatment']==1)]['passing_yards'].mean()
    treated_pre = df_clean[(df_clean['is_treated']) & (df_clean['pre_treatment']==1)]['passing_yards'].mean()
    control_post = df_clean[(~df_clean['is_treated']) & (df_clean['post_treatment']==1)]['passing_yards'].mean()
    control_pre = df_clean[(~df_clean['is_treated']) & (df_clean['pre_treatment']==1)]['passing_yards'].mean()

    # DiD estimator
    did_effect = (treated_post - treated_pre) - (control_post - control_pre)

    print("\nGroup Means (Average Passing Yards):")
    print(f"  Treated (High Roughing) - Pre-2018:  {treated_pre:.2f} yards")
    print(f"  Treated (High Roughing) - Post-2018: {treated_post:.2f} yards")
    print(f"  Control (Low Roughing) - Pre-2018:   {control_pre:.2f} yards")
    print(f"  Control (Low Roughing) - Post-2018:  {control_post:.2f} yards")

    print("\nFirst Differences:")
    print(f"  Treated group change: {treated_post - treated_pre:+.2f} yards")
    print(f"  Control group change: {control_post - control_pre:+.2f} yards")

    print(f"\n{'='*50}")
    print(f"DiD Treatment Effect: {did_effect:+.2f} yards")
    print(f"{'='*50}")

    # Statistical inference using two-sample t-test on first differences
    treated_post_data = df_clean[(df_clean['is_treated']) & (df_clean['post_treatment']==1)]['passing_yards']
    treated_pre_data = df_clean[(df_clean['is_treated']) & (df_clean['pre_treatment']==1)]['passing_yards']
    control_post_data = df_clean[(~df_clean['is_treated']) & (df_clean['post_treatment']==1)]['passing_yards']
    control_pre_data = df_clean[(~df_clean['is_treated']) & (df_clean['pre_treatment']==1)]['passing_yards']

    # Calculate first differences
    treated_diff = treated_post_data.mean() - treated_pre_data.mean()
    control_diff = control_post_data.mean() - control_pre_data.mean()

    # Standard errors (pooled variance)
    se_treated = np.sqrt(treated_post_data.var()/len(treated_post_data) + treated_pre_data.var()/len(treated_pre_data))
    se_control = np.sqrt(control_post_data.var()/len(control_post_data) + control_pre_data.var()/len(control_pre_data))
    se_did = np.sqrt(se_treated**2 + se_control**2)

    # t-statistic and p-value
    t_stat = did_effect / se_did
    df_degrees = len(treated_post_data) + len(treated_pre_data) + len(control_post_data) + len(control_pre_data) - 4
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_degrees))

    # 95% confidence interval
    ci_lower = did_effect - 1.96 * se_did
    ci_upper = did_effect + 1.96 * se_did

    print("\nStatistical Inference:")
    print(f"  Standard Error: {se_did:.2f} yards")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] yards")

    if p_value < 0.05:
        print("\n✅ Treatment effect is statistically significant at α=0.05")
    else:
        print("\n⚠️  Treatment effect is NOT statistically significant at α=0.05")

    # Bootstrap confidence intervals for robustness
    print("\nCalculating bootstrap confidence intervals (1000 iterations)...")
    bootstrap_effects = []
    n_bootstrap = 1000

    for _ in range(n_bootstrap):
        # Resample with replacement
        treated_post_boot = np.random.choice(treated_post_data, size=len(treated_post_data), replace=True)
        treated_pre_boot = np.random.choice(treated_pre_data, size=len(treated_pre_data), replace=True)
        control_post_boot = np.random.choice(control_post_data, size=len(control_post_data), replace=True)
        control_pre_boot = np.random.choice(control_pre_data, size=len(control_pre_data), replace=True)

        # Calculate DiD effect
        boot_effect = (treated_post_boot.mean() - treated_pre_boot.mean()) - \
                      (control_post_boot.mean() - control_pre_boot.mean())
        bootstrap_effects.append(boot_effect)

    bootstrap_ci_lower = np.percentile(bootstrap_effects, 2.5)
    bootstrap_ci_upper = np.percentile(bootstrap_effects, 97.5)

    print(f"Bootstrap 95% CI: [{bootstrap_ci_lower:.2f}, {bootstrap_ci_upper:.2f}] yards")

    results = {
        'did_effect': float(did_effect),
        'standard_error': float(se_did),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'bootstrap_ci_lower': float(bootstrap_ci_lower),
        'bootstrap_ci_upper': float(bootstrap_ci_upper),
        'group_means': {
            'treated_pre': float(treated_pre),
            'treated_post': float(treated_post),
            'control_pre': float(control_pre),
            'control_post': float(control_post)
        },
        'sample_sizes': {
            'treated_pre': int(len(treated_pre_data)),
            'treated_post': int(len(treated_post_data)),
            'control_pre': int(len(control_pre_data)),
            'control_post': int(len(control_post_data))
        }
    }

    return results


def validate_parallel_trends(df: pd.DataFrame) -> dict:
    """
    Validate parallel trends assumption using pre-treatment period (2015-2017).

    Tests whether treated and control groups had parallel trends before 2018.
    Uses linear regression of passing yards on year for each group.
    """
    print("\n" + "="*80)
    print("STEP 4: Validate Parallel Trends Assumption (2015-2017)")
    print("="*80)

    # Pre-treatment data only
    df_pre = df[(df['season'] <= 2017) & (df['defense_2017_roughing_rate'].notna())].copy()

    # Calculate annual means for each group
    treated_trends = df_pre[df_pre['is_treated']].groupby('season')['passing_yards'].mean()
    control_trends = df_pre[~df_pre['is_treated']].groupby('season')['passing_yards'].mean()

    print("\nAnnual Mean Passing Yards (Pre-Treatment):")
    print(f"{'Year':<6} {'Treated (High Roughing)':<25} {'Control (Low Roughing)':<25} {'Difference':<15}")
    print("-" * 75)
    for year in sorted(treated_trends.index):
        treated_val = treated_trends.get(year, np.nan)
        control_val = control_trends.get(year, np.nan)
        diff = treated_val - control_val
        print(f"{year:<6} {treated_val:>20.2f}     {control_val:>20.2f}     {diff:>10.2f}")

    # Linear trend test
    from scipy.stats import linregress

    years = sorted(treated_trends.index)
    treated_slope, treated_intercept, treated_r, treated_p, treated_se = linregress(
        years, [treated_trends[y] for y in years]
    )
    control_slope, control_intercept, control_r, control_p, control_se = linregress(
        years, [control_trends[y] for y in years]
    )

    slope_diff = treated_slope - control_slope
    slope_diff_se = np.sqrt(treated_se**2 + control_se**2)
    slope_diff_t = slope_diff / slope_diff_se
    slope_diff_p = 2 * (1 - stats.t.cdf(abs(slope_diff_t), len(years) - 2))

    print("\nLinear Trend Analysis:")
    print(f"  Treated group slope: {treated_slope:+.2f} yards/year (p={treated_p:.4f})")
    print(f"  Control group slope: {control_slope:+.2f} yards/year (p={control_p:.4f})")
    print(f"  Difference in slopes: {slope_diff:+.2f} yards/year")
    print(f"  t-statistic: {slope_diff_t:.3f}")
    print(f"  p-value: {slope_diff_p:.4f}")

    if slope_diff_p > 0.05:
        print(f"\n✅ Parallel trends assumption SATISFIED (p={slope_diff_p:.4f} > 0.05)")
        print("   → Treated and control groups had statistically similar trends pre-2018")
    else:
        print(f"\n⚠️  Parallel trends assumption VIOLATED (p={slope_diff_p:.4f} < 0.05)")
        print("   → DiD estimates may be biased")

    results = {
        'treated_slope': float(treated_slope),
        'control_slope': float(control_slope),
        'slope_difference': float(slope_diff),
        'slope_difference_se': float(slope_diff_se),
        'slope_difference_t': float(slope_diff_t),
        'slope_difference_p': float(slope_diff_p),
        'parallel_trends_satisfied': bool(slope_diff_p > 0.05),
        'annual_means': {
            'treated': {int(year): float(treated_trends[year]) for year in treated_trends.index},
            'control': {int(year): float(control_trends[year]) for year in control_trends.index}
        }
    }

    return results


def placebo_test(df: pd.DataFrame) -> dict:
    """
    Placebo test: Run DiD on 2016-2017 period (should show no effect).

    If we find a "treatment effect" before the actual treatment, it suggests
    spurious correlation or violation of identifying assumptions.
    """
    print("\n" + "="*80)
    print("STEP 5: Placebo Test (2016-2017 Fake Treatment)")
    print("="*80)

    # Use only 2016-2017 data
    df_placebo = df[(df['season'].isin([2016, 2017])) & (df['defense_2017_roughing_rate'].notna())].copy()

    # Fake treatment: pretend 2017 is "post-treatment"
    df_placebo['fake_post'] = (df_placebo['season'] == 2017).astype(int)
    df_placebo['fake_pre'] = (df_placebo['season'] == 2016).astype(int)

    # Calculate placebo DiD
    treated_post = df_placebo[(df_placebo['is_treated']) & (df_placebo['fake_post']==1)]['passing_yards'].mean()
    treated_pre = df_placebo[(df_placebo['is_treated']) & (df_placebo['fake_pre']==1)]['passing_yards'].mean()
    control_post = df_placebo[(~df_placebo['is_treated']) & (df_placebo['fake_post']==1)]['passing_yards'].mean()
    control_pre = df_placebo[(~df_placebo['is_treated']) & (df_placebo['fake_pre']==1)]['passing_yards'].mean()

    placebo_effect = (treated_post - treated_pre) - (control_post - control_pre)

    print("\nPlacebo DiD (2016 → 2017):")
    print(f"  Treated (High Roughing) - 2016: {treated_pre:.2f} yards")
    print(f"  Treated (High Roughing) - 2017: {treated_post:.2f} yards")
    print(f"  Control (Low Roughing) - 2016:  {control_pre:.2f} yards")
    print(f"  Control (Low Roughing) - 2017:  {control_post:.2f} yards")
    print(f"\n  Placebo Effect: {placebo_effect:+.2f} yards")

    # Statistical test
    treated_post_data = df_placebo[(df_placebo['is_treated']) & (df_placebo['fake_post']==1)]['passing_yards']
    treated_pre_data = df_placebo[(df_placebo['is_treated']) & (df_placebo['fake_pre']==1)]['passing_yards']
    control_post_data = df_placebo[(~df_placebo['is_treated']) & (df_placebo['fake_post']==1)]['passing_yards']
    control_pre_data = df_placebo[(~df_placebo['is_treated']) & (df_placebo['fake_pre']==1)]['passing_yards']

    se_treated = np.sqrt(treated_post_data.var()/len(treated_post_data) + treated_pre_data.var()/len(treated_pre_data))
    se_control = np.sqrt(control_post_data.var()/len(control_post_data) + control_pre_data.var()/len(control_pre_data))
    se_placebo = np.sqrt(se_treated**2 + se_control**2)

    t_stat = placebo_effect / se_placebo
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(treated_post_data) + len(treated_pre_data) +
                                                  len(control_post_data) + len(control_pre_data) - 4))

    print(f"\n  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value > 0.05:
        print(f"\n✅ Placebo test PASSED (p={p_value:.4f} > 0.05)")
        print("   → No spurious effect detected in pre-treatment period")
    else:
        print(f"\n⚠️  Placebo test FAILED (p={p_value:.4f} < 0.05)")
        print("   → Detected spurious effect - DiD assumptions may be violated")

    results = {
        'placebo_effect': float(placebo_effect),
        'standard_error': float(se_placebo),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'placebo_test_passed': bool(p_value > 0.05)
    }

    return results


def robustness_checks(df: pd.DataFrame) -> dict:
    """
    Robustness checks: Try different treatment definitions.

    1. Terciles (top third vs bottom third)
    2. Quartiles (top quartile vs bottom quartile)
    3. Median split (top half vs bottom half)
    """
    print("\n" + "="*80)
    print("STEP 6: Robustness Checks (Alternative Treatment Definitions)")
    print("="*80)

    df_clean = df[(df['defense_2017_roughing_rate'].notna())].copy()

    results = {}

    # 1. Tercile split
    print("\n1. TERCILE SPLIT (Top third vs Bottom third)")
    print("-" * 60)
    df_clean['tercile'] = pd.qcut(df_clean['defense_2017_roughing_rate'], q=3, labels=['Low', 'Mid', 'High'])
    df_tercile = df_clean[df_clean['tercile'].isin(['Low', 'High'])].copy()
    df_tercile['is_treated_tercile'] = (df_tercile['tercile'] == 'High')

    treated_post = df_tercile[(df_tercile['is_treated_tercile']) & (df_tercile['post_treatment']==1)]['passing_yards'].mean()
    treated_pre = df_tercile[(df_tercile['is_treated_tercile']) & (df_tercile['pre_treatment']==1)]['passing_yards'].mean()
    control_post = df_tercile[(~df_tercile['is_treated_tercile']) & (df_tercile['post_treatment']==1)]['passing_yards'].mean()
    control_pre = df_tercile[(~df_tercile['is_treated_tercile']) & (df_tercile['pre_treatment']==1)]['passing_yards'].mean()

    tercile_effect = (treated_post - treated_pre) - (control_post - control_pre)
    print(f"   DiD Effect: {tercile_effect:+.2f} yards")
    results['tercile_effect'] = float(tercile_effect)

    # 2. Quartile split
    print("\n2. QUARTILE SPLIT (Top quartile vs Bottom quartile)")
    print("-" * 60)
    df_clean['quartile'] = pd.qcut(df_clean['defense_2017_roughing_rate'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df_quartile = df_clean[df_clean['quartile'].isin(['Q1', 'Q4'])].copy()
    df_quartile['is_treated_quartile'] = (df_quartile['quartile'] == 'Q4')

    treated_post = df_quartile[(df_quartile['is_treated_quartile']) & (df_quartile['post_treatment']==1)]['passing_yards'].mean()
    treated_pre = df_quartile[(df_quartile['is_treated_quartile']) & (df_quartile['pre_treatment']==1)]['passing_yards'].mean()
    control_post = df_quartile[(~df_quartile['is_treated_quartile']) & (df_quartile['post_treatment']==1)]['passing_yards'].mean()
    control_pre = df_quartile[(~df_quartile['is_treated_quartile']) & (df_quartile['pre_treatment']==1)]['passing_yards'].mean()

    quartile_effect = (treated_post - treated_pre) - (control_post - control_pre)
    print(f"   DiD Effect: {quartile_effect:+.2f} yards")
    results['quartile_effect'] = float(quartile_effect)

    # 3. Median split
    print("\n3. MEDIAN SPLIT (Top half vs Bottom half)")
    print("-" * 60)
    median_roughing = df_clean['defense_2017_roughing_rate'].median()
    df_clean['is_treated_median'] = (df_clean['defense_2017_roughing_rate'] > median_roughing)

    treated_post = df_clean[(df_clean['is_treated_median']) & (df_clean['post_treatment']==1)]['passing_yards'].mean()
    treated_pre = df_clean[(df_clean['is_treated_median']) & (df_clean['pre_treatment']==1)]['passing_yards'].mean()
    control_post = df_clean[(~df_clean['is_treated_median']) & (df_clean['post_treatment']==1)]['passing_yards'].mean()
    control_pre = df_clean[(~df_clean['is_treated_median']) & (df_clean['pre_treatment']==1)]['passing_yards'].mean()

    median_effect = (treated_post - treated_pre) - (control_post - control_pre)
    print(f"   DiD Effect: {median_effect:+.2f} yards")
    results['median_effect'] = float(median_effect)

    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY:")
    print(f"  Quintile split (baseline): {results.get('quintile_effect', 'N/A')} yards")
    print(f"  Tercile split:             {tercile_effect:+.2f} yards")
    print(f"  Quartile split:            {quartile_effect:+.2f} yards")
    print(f"  Median split:              {median_effect:+.2f} yards")

    return results


def main():
    """Main execution pipeline."""
    print("\n" + "="*80)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS: 2018 ROUGHING-THE-PASSER RULE CHANGE")
    print("="*80)
    print("\nThis pipeline implements a complete DiD analysis for the 2018 NFL")
    print("roughing-the-passer rule change as a natural experiment.")
    print("\nExpected runtime: 2-3 minutes")

    # Step 1: Extract 2017 defense baseline
    defense_classifications = extract_2017_roughing_baseline()

    # Step 2: Extract QB performance data
    qb_data = extract_qb_performance_data(defense_classifications)

    # Step 3: Calculate DiD treatment effect
    did_results = calculate_did_treatment_effect(qb_data)

    # Step 4: Validate parallel trends
    parallel_trends_results = validate_parallel_trends(qb_data)

    # Step 5: Placebo test
    placebo_results = placebo_test(qb_data)

    # Step 6: Robustness checks
    did_results['quintile_effect'] = did_results['did_effect']  # Add to robustness comparison
    robustness_results = robustness_checks(qb_data)

    # Compile final results
    final_results = {
        'treatment_effect': did_results,
        'parallel_trends': parallel_trends_results,
        'placebo_test': placebo_results,
        'robustness': robustness_results,
        'metadata': {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'num_defenses': len(defense_classifications),
            'num_qb_games': len(qb_data),
            'seasons_analyzed': f"{qb_data['season'].min()}-{qb_data['season'].max()}",
            'treatment_year': 2018
        }
    }

    # Save results
    output_path = "results/did_roughing_validation.json"
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✅ Results saved to {output_path}")
    print("\nKey Findings:")
    print(f"  • DiD Treatment Effect: {did_results['did_effect']:+.2f} yards (p={did_results['p_value']:.4f})")
    print(f"  • Parallel Trends: {'SATISFIED' if parallel_trends_results['parallel_trends_satisfied'] else 'VIOLATED'}")
    print(f"  • Placebo Test: {'PASSED' if placebo_results['placebo_test_passed'] else 'FAILED'}")
    print(f"  • Robustness: Effects range from {min(robustness_results.values()):+.2f} to {max(robustness_results.values()):+.2f} yards")

    return final_results


if __name__ == "__main__":
    results = main()
