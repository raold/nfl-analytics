"""
Unit Tests for Difference-in-Differences Injury Analysis Module

Tests for data preparation, estimation, and visualization components.

Run with:
    pytest py/causal/test_did_injury_analysis.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from did_injury_analysis import (
    InjuryEvent,
    DIDResults,
    InjuryShockDatabase,
    DifferenceInDifferencesEstimator,
    InjuryShockModeler
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_injury_events():
    """Create sample injury events for testing"""
    events = [
        InjuryEvent(
            player_id='TEST_RB1',
            player_name='Star RB1',
            position='RB',
            team='KC',
            season=2020,
            week=4,
            injury_type='Knee',
            injury_status='Out',
            is_star=True
        ),
        InjuryEvent(
            player_id='TEST_RB2',
            player_name='Star RB2',
            position='RB',
            team='BUF',
            season=2020,
            week=8,
            injury_type='Ankle',
            injury_status='Out',
            is_star=True
        ),
        InjuryEvent(
            player_id='TEST_QB1',
            player_name='Star QB1',
            position='QB',
            team='SF',
            season=2021,
            week=4,
            injury_type='Shoulder',
            injury_status='Out',
            is_star=True
        )
    ]
    return events


@pytest.fixture
def synthetic_panel_data():
    """Create synthetic panel data for testing"""
    np.random.seed(42)

    # Treatment cohorts
    cohorts = ['2020W4', '2020W8']
    n_treated_per_cohort = 5
    n_control = 10
    pre_periods = 4
    post_periods = 4

    panel_rows = []

    # Treated units with parallel pre-trends and treatment effect
    for cohort in cohorts:
        season, week = cohort.split('W')
        season = int(season)
        treatment_week = int(week)

        for unit_id in range(n_treated_per_cohort):
            player_id = f"TREATED_{cohort}_{unit_id}"
            baseline = 50 + np.random.randn() * 5

            # Pre-treatment (parallel trends)
            for rel_week in range(-pre_periods, 0):
                week = treatment_week + rel_week
                panel_rows.append({
                    'player_id': player_id,
                    'player_name': f"Backup RB {unit_id}",
                    'team': f'TEAM_{cohort}',
                    'season': season,
                    'week': week,
                    'outcome': baseline + np.random.randn() * 8,
                    'treated': 0,
                    'time_to_treatment': rel_week,
                    'cohort': cohort
                })

            # Post-treatment (ATT = +12 yards)
            treatment_effect = 12
            for rel_week in range(0, post_periods + 1):
                week = treatment_week + rel_week
                panel_rows.append({
                    'player_id': player_id,
                    'player_name': f"Backup RB {unit_id}",
                    'team': f'TEAM_{cohort}',
                    'season': season,
                    'week': week,
                    'outcome': baseline + treatment_effect + np.random.randn() * 8,
                    'treated': 1,
                    'time_to_treatment': rel_week,
                    'cohort': cohort
                })

    # Never-treated control units
    for unit_id in range(n_control):
        player_id = f"CONTROL_{unit_id}"
        baseline = 50 + np.random.randn() * 5

        for week in range(1, 17):
            panel_rows.append({
                'player_id': player_id,
                'player_name': f"Control RB {unit_id}",
                'team': f'CONTROL_TEAM_{unit_id}',
                'season': 2020,
                'week': week,
                'outcome': baseline + np.random.randn() * 8,
                'treated': 0,
                'time_to_treatment': np.nan,
                'cohort': 'never_treated'
            })

    df = pd.DataFrame(panel_rows)
    return df


# =============================================================================
# TEST DATA CLASSES
# =============================================================================

def test_injury_event_creation():
    """Test InjuryEvent dataclass creation"""
    event = InjuryEvent(
        player_id='TEST123',
        player_name='Test Player',
        position='RB',
        team='KC',
        season=2024,
        week=5,
        injury_type='Hamstring',
        injury_status='Out',
        is_star=True
    )

    assert event.player_id == 'TEST123'
    assert event.cohort == '2024W5'
    assert event.is_star is True


def test_did_results_creation():
    """Test DIDResults dataclass"""
    results = DIDResults(
        att_overall=10.5,
        se_overall=2.3,
        pvalue_overall=0.001,
        ci_lower=6.0,
        ci_upper=15.0,
        group_time_atts=pd.DataFrame(),
        att_by_period=pd.DataFrame(),
        parallel_trends_test={},
        n_treated=100,
        n_control=200,
        n_cohorts=3
    )

    assert results.att_overall == 10.5
    assert results.se_overall == 2.3
    assert results.n_treated == 100


# =============================================================================
# TEST INJURY SHOCK DATABASE
# =============================================================================

def test_injury_database_initialization():
    """Test database initialization"""
    db = InjuryShockDatabase()
    assert db.db_params['dbname'] == 'devdb01'
    assert db.db_params['port'] == 5544


def test_create_synthetic_injuries():
    """Test synthetic injury creation"""
    db = InjuryShockDatabase()
    events = db._create_synthetic_injuries(2020, 2022)

    assert len(events) > 0
    assert all(isinstance(e, InjuryEvent) for e in events)
    assert all(e.is_star for e in events)
    assert all(2020 <= e.season <= 2022 for e in events)


def test_create_synthetic_panel():
    """Test synthetic panel creation"""
    db = InjuryShockDatabase()
    panel = db._create_synthetic_panel(pre_periods=4, post_periods=4)

    assert isinstance(panel, pd.DataFrame)
    assert len(panel) > 0

    # Check required columns
    required_cols = ['player_id', 'outcome', 'treated', 'time_to_treatment', 'cohort']
    for col in required_cols:
        assert col in panel.columns

    # Check treatment indicator
    assert panel['treated'].isin([0, 1]).all()

    # Check never-treated group exists
    assert 'never_treated' in panel['cohort'].values


def test_identify_star_injuries(sample_injury_events):
    """Test star player identification"""
    db = InjuryShockDatabase()

    # With database connection failure, should mark all as star
    star_events = db.identify_star_injuries(sample_injury_events)

    assert len(star_events) > 0
    assert all(e.is_star for e in star_events)


# =============================================================================
# TEST DIFFERENCE-IN-DIFFERENCES ESTIMATOR
# =============================================================================

def test_estimator_initialization():
    """Test DiD estimator initialization"""
    estimator = DifferenceInDifferencesEstimator()
    assert estimator.results is None
    assert estimator.panel_data is None


def test_estimate_group_time_att(synthetic_panel_data):
    """Test group-time ATT estimation"""
    estimator = DifferenceInDifferencesEstimator()

    group_time_atts = estimator.estimate_group_time_att(
        synthetic_panel_data,
        bootstrap_reps=10  # Small for speed
    )

    assert isinstance(group_time_atts, pd.DataFrame)
    assert len(group_time_atts) > 0

    # Check required columns
    required_cols = ['cohort', 'rel_time', 'att', 'se', 'pvalue']
    for col in required_cols:
        assert col in group_time_atts.columns

    # Check ATT values are numeric
    assert group_time_atts['att'].dtype in [np.float64, np.float32, float]
    assert group_time_atts['se'].dtype in [np.float64, np.float32, float]

    # Check treatment effect is positive (true effect = +12)
    post_treatment_att = group_time_atts[group_time_atts['rel_time'] >= 0]['att'].mean()
    assert post_treatment_att > 0, f"Expected positive ATT, got {post_treatment_att}"


def test_aggregate_effects_event_study(synthetic_panel_data):
    """Test event study aggregation"""
    estimator = DifferenceInDifferencesEstimator()

    group_time_atts = estimator.estimate_group_time_att(
        synthetic_panel_data,
        bootstrap_reps=10
    )

    agg = estimator.aggregate_effects(group_time_atts, aggregation='event_study')

    assert isinstance(agg, pd.DataFrame)
    assert 'rel_time' in agg.columns
    assert 'att' in agg.columns
    assert 'se' in agg.columns


def test_aggregate_effects_overall(synthetic_panel_data):
    """Test overall ATT aggregation"""
    estimator = DifferenceInDifferencesEstimator()

    group_time_atts = estimator.estimate_group_time_att(
        synthetic_panel_data,
        bootstrap_reps=10
    )

    agg = estimator.aggregate_effects(group_time_atts, aggregation='overall')

    assert isinstance(agg, pd.DataFrame)
    assert len(agg) == 1
    assert 'att' in agg.columns
    assert 'se' in agg.columns
    assert 'pvalue' in agg.columns


def test_parallel_trends_test(synthetic_panel_data):
    """Test parallel trends assumption"""
    estimator = DifferenceInDifferencesEstimator()

    pt_results = estimator.test_parallel_trends(synthetic_panel_data)

    assert isinstance(pt_results, dict)
    assert 'parallel_trends' in pt_results
    assert 'pvalue' in pt_results

    # With synthetic data (designed with parallel trends), should pass
    assert pt_results['parallel_trends'] is True


def test_placebo_test(synthetic_panel_data):
    """Test placebo test with random treatment"""
    estimator = DifferenceInDifferencesEstimator()

    placebo_results = estimator.placebo_test(
        synthetic_panel_data,
        n_placebo=10  # Small for speed
    )

    assert isinstance(placebo_results, dict)
    assert 'actual_att' in placebo_results
    assert 'placebo_mean' in placebo_results
    assert 'pvalue' in placebo_results


# =============================================================================
# TEST INJURY SHOCK MODELER
# =============================================================================

def test_modeler_initialization():
    """Test InjuryShockModeler initialization"""
    modeler = InjuryShockModeler()
    assert isinstance(modeler.db, InjuryShockDatabase)
    assert isinstance(modeler.estimator, DifferenceInDifferencesEstimator)


def test_analyze_injury_shock_synthetic():
    """Test complete analysis pipeline with synthetic data"""
    modeler = InjuryShockModeler()

    # This will use synthetic data (database connection will fail in test env)
    results = modeler.analyze_injury_shock(
        start_season=2020,
        end_season=2021,
        outcome_type='backup_rushing_yards',
        positions=['RB'],
        pre_periods=4,
        post_periods=4,
        run_placebo=False
    )

    assert isinstance(results, DIDResults)
    assert results.att_overall is not None
    assert results.se_overall > 0
    assert results.n_treated > 0
    assert results.n_control > 0


def test_visualize_results():
    """Test visualization generation"""
    modeler = InjuryShockModeler()

    # Run analysis
    results = modeler.analyze_injury_shock(
        start_season=2020,
        end_season=2021,
        positions=['RB'],
        run_placebo=False
    )

    # Create visualization (don't display)
    fig = modeler.visualize_results(results)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # Should have 4 subplots

    # Clean up
    plt.close(fig)


def test_generate_report():
    """Test markdown report generation"""
    modeler = InjuryShockModeler()

    # Run analysis
    results = modeler.analyze_injury_shock(
        start_season=2020,
        end_season=2021,
        positions=['RB'],
        run_placebo=False
    )

    # Generate report
    report = modeler.generate_report(results)

    assert isinstance(report, str)
    assert len(report) > 0

    # Check key sections
    assert 'Difference-in-Differences Analysis' in report
    assert 'Overall Average Treatment Effect' in report
    assert 'Parallel Trends Assumption' in report
    assert 'References' in report
    assert 'Callaway' in report


# =============================================================================
# TEST ERROR HANDLING
# =============================================================================

def test_empty_injury_events():
    """Test handling of empty injury events list"""
    db = InjuryShockDatabase()
    panel = db.create_panel_data([], pre_periods=4, post_periods=4)

    # Should create synthetic panel
    assert isinstance(panel, pd.DataFrame)
    assert len(panel) > 0


def test_no_results_visualization_error():
    """Test that visualization fails gracefully without results"""
    modeler = InjuryShockModeler()

    with pytest.raises(ValueError, match="No results to visualize"):
        modeler.visualize_results()


def test_no_results_report_error():
    """Test that report generation fails gracefully without results"""
    modeler = InjuryShockModeler()

    with pytest.raises(ValueError, match="No results to report"):
        modeler.generate_report()


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_full_pipeline_integration():
    """Integration test: Full analysis pipeline"""
    # Initialize modeler
    modeler = InjuryShockModeler()

    # Run analysis (will use synthetic data)
    results = modeler.analyze_injury_shock(
        start_season=2020,
        end_season=2021,
        outcome_type='backup_rushing_yards',
        positions=['RB'],
        pre_periods=4,
        post_periods=4,
        run_placebo=False
    )

    # Validate results structure
    assert isinstance(results, DIDResults)
    assert results.att_overall is not None
    assert results.se_overall > 0
    assert results.pvalue_overall >= 0 and results.pvalue_overall <= 1
    assert results.ci_lower < results.ci_upper
    assert results.n_treated > 0
    assert results.n_control > 0
    assert results.n_cohorts > 0
    assert isinstance(results.group_time_atts, pd.DataFrame)
    assert isinstance(results.att_by_period, pd.DataFrame)
    assert isinstance(results.parallel_trends_test, dict)

    # Validate parallel trends test
    pt = results.parallel_trends_test
    assert 'parallel_trends' in pt
    assert 'pvalue' in pt

    # Generate visualizations
    fig = modeler.visualize_results(results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Generate report
    report = modeler.generate_report(results)
    assert isinstance(report, str)
    assert len(report) > 100

    print("\n" + "=" * 70)
    print("INTEGRATION TEST PASSED")
    print("=" * 70)
    print(f"Overall ATT: {results.att_overall:.2f} ± {results.se_overall:.2f}")
    print(f"P-value: {results.pvalue_overall:.4f}")
    print(f"Parallel Trends: {pt['parallel_trends']}")
    print(f"Sample Size: {results.n_treated} treated, {results.n_control} control")


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

def test_synthetic_data_recovers_true_effect():
    """Test that estimator recovers known treatment effect from synthetic data"""
    np.random.seed(123)

    # Create data with KNOWN treatment effect = +15 yards
    true_effect = 15.0
    n_treated = 20
    n_control = 30
    n_periods = 8

    panel_rows = []

    # Treated units
    for i in range(n_treated):
        player_id = f"TREATED_{i}"
        baseline = 50 + np.random.randn() * 3

        for t in range(n_periods):
            if t < 4:  # Pre-treatment
                outcome = baseline + np.random.randn() * 5
                treated = 0
                rel_time = t - 4
            else:  # Post-treatment
                outcome = baseline + true_effect + np.random.randn() * 5
                treated = 1
                rel_time = t - 4

            panel_rows.append({
                'player_id': player_id,
                'outcome': outcome,
                'treated': treated,
                'time_to_treatment': rel_time,
                'cohort': '2020W4'
            })

    # Control units
    for i in range(n_control):
        player_id = f"CONTROL_{i}"
        baseline = 50 + np.random.randn() * 3

        for t in range(n_periods):
            panel_rows.append({
                'player_id': player_id,
                'outcome': baseline + np.random.randn() * 5,
                'treated': 0,
                'time_to_treatment': np.nan,
                'cohort': 'never_treated'
            })

    df = pd.DataFrame(panel_rows)

    # Estimate
    estimator = DifferenceInDifferencesEstimator()
    group_time_atts = estimator.estimate_group_time_att(df, bootstrap_reps=50)
    overall_agg = estimator.aggregate_effects(group_time_atts, 'overall')

    estimated_effect = overall_agg.iloc[0]['att']
    se = overall_agg.iloc[0]['se']

    print(f"\nTrue effect: {true_effect:.2f}")
    print(f"Estimated effect: {estimated_effect:.2f} ± {se:.2f}")

    # Check that estimate is within 2 SEs of true effect
    assert abs(estimated_effect - true_effect) < 2 * se, \
        f"Estimate {estimated_effect:.2f} is too far from true effect {true_effect:.2f}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
