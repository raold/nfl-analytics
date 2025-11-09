"""
Example Scripts for Difference-in-Differences Injury Analysis

This module demonstrates various use cases for the DiD injury analysis framework.

Examples:
1. Basic backup RB analysis (star RB injury → backup performance)
2. QB injury analysis (QB1 injury → passing game)
3. WR injury analysis (WR1 injury → WR2 targets)
4. Team-level analysis (star player injury → team points)
5. Custom analysis with control variables

Run examples:
    python py/causal/examples_did_injury.py --example 1
    python py/causal/examples_did_injury.py --example all
"""

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from did_injury_analysis import InjuryShockModeler, InjuryShockDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# EXAMPLE 1: BACKUP RB PERFORMANCE AFTER STAR RB INJURY
# =============================================================================

def example_1_backup_rb_analysis():
    """
    Research Question:
        When a star RB is injured, how does the backup RB's rushing production change?

    Hypothesis:
        Backup RBs receive more carries and opportunities, but may be less efficient.

    Expected Result:
        Positive ATT (more carries/yards) but potentially lower efficiency.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 1: Backup RB Performance After Star RB Injury")
    logger.info("=" * 70)

    # Initialize modeler
    modeler = InjuryShockModeler()

    # Run analysis
    results = modeler.analyze_injury_shock(
        start_season=2018,
        end_season=2024,
        outcome_type='backup_rushing_yards',
        positions=['RB'],  # Only RB injuries
        pre_periods=4,
        post_periods=6,
        run_placebo=False
    )

    # Print results
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"Overall ATT: {results.att_overall:.2f} yards (SE: {results.se_overall:.2f})")
    print(f"95% CI: [{results.ci_lower:.2f}, {results.ci_upper:.2f}]")
    print(f"P-value: {results.pvalue_overall:.4f}")
    print(f"Significant: {'YES ✓' if results.pvalue_overall < 0.05 else 'NO ✗'}")
    print(f"\nParallel Trends: {'SATISFIED ✓' if results.parallel_trends_test['parallel_trends'] else 'VIOLATED ✗'}")
    print(f"Sample Size: {results.n_treated} treated, {results.n_control} control")
    print(f"Treatment Cohorts: {results.n_cohorts}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    if results.att_overall > 0 and results.pvalue_overall < 0.05:
        print(f"When a star RB is injured, backup RBs gain +{results.att_overall:.1f} rushing yards")
        print("per game on average. This suggests they receive significantly more usage.")
        print("\nBetting Implication: Consider OVER on backup RB rushing props when RB1 is out.")
    elif results.att_overall < 0 and results.pvalue_overall < 0.05:
        print(f"When a star RB is injured, backup RBs LOSE {abs(results.att_overall):.1f} rushing yards")
        print("per game. The offense may struggle, or defenses focus more on stopping the run.")
        print("\nBetting Implication: Consider UNDER on backup RB props despite increased usage.")
    else:
        print("No statistically significant effect detected.")
        print("Backup RB performance may be too variable to predict consistently.")

    # Visualize
    output_dir = Path('output/did_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = modeler.visualize_results(
        results,
        save_path=str(output_dir / 'example1_backup_rb.pdf')
    )
    plt.show()

    # Generate report
    report = modeler.generate_report(
        results,
        output_path=str(output_dir / 'example1_backup_rb_report.md')
    )

    print(f"\n✓ Visualizations saved to {output_dir / 'example1_backup_rb.pdf'}")
    print(f"✓ Report saved to {output_dir / 'example1_backup_rb_report.md'}")

    return results


# =============================================================================
# EXAMPLE 2: QB INJURY ANALYSIS (PASSING GAME)
# =============================================================================

def example_2_qb_injury_analysis():
    """
    Research Question:
        When a starting QB is injured, how does the team's passing production change?

    Hypothesis:
        Backup QBs throw fewer yards/TDs, and the offense becomes more conservative.

    Expected Result:
        Negative ATT (reduced passing production).
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Team Passing Production After QB Injury")
    logger.info("=" * 70)

    modeler = InjuryShockModeler()

    # For this example, we'd need passing data
    # Here we demonstrate the structure with rushing data as placeholder
    results = modeler.analyze_injury_shock(
        start_season=2018,
        end_season=2024,
        outcome_type='backup_rushing_yards',  # Would be 'team_passing_yards' in real analysis
        positions=['QB'],
        pre_periods=3,
        post_periods=5,
        run_placebo=False
    )

    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"Overall ATT: {results.att_overall:.2f} yards")
    print(f"P-value: {results.pvalue_overall:.4f}")

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print("NOTE: This example uses rushing data as placeholder.")
    print("Real analysis would use team passing yards as outcome.")
    print("\nExpected finding: Negative effect on passing production when QB1 is injured.")
    print("Betting implication: Fade team passing props when backup QB starts.")

    return results


# =============================================================================
# EXAMPLE 3: WR TARGET REDISTRIBUTION
# =============================================================================

def example_3_wr_target_redistribution():
    """
    Research Question:
        When WR1 is injured, does WR2 receive significantly more targets?

    Hypothesis:
        WR2 target share increases, but efficiency (yards/target) may decrease
        due to increased defensive attention.

    Expected Result:
        Positive ATT for targets, unclear for yards.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: WR2 Target Share After WR1 Injury")
    logger.info("=" * 70)

    modeler = InjuryShockModeler()

    results = modeler.analyze_injury_shock(
        start_season=2018,
        end_season=2024,
        outcome_type='backup_rushing_yards',  # Would be 'wr2_targets' in real analysis
        positions=['WR'],
        pre_periods=4,
        post_periods=4,
        run_placebo=False
    )

    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"Overall ATT: {results.att_overall:.2f}")
    print(f"P-value: {results.pvalue_overall:.4f}")

    print("\n" + "-" * 70)
    print("PRACTICAL APPLICATION")
    print("-" * 70)
    print("When betting on WR2 props after WR1 injury:")
    print("1. Target share likely increases (more opportunities)")
    print("2. Efficiency (yards/target) may decrease (tougher coverage)")
    print("3. Check QB quality - backup QBs may struggle to find WR2")
    print("4. Consider defensive matchup - does defense focus on stopping #1 WR?")

    return results


# =============================================================================
# EXAMPLE 4: TEAM-LEVEL ANALYSIS (OVERALL OFFENSIVE PRODUCTION)
# =============================================================================

def example_4_team_offensive_production():
    """
    Research Question:
        How much does overall team offensive production decline when a star
        player (QB/RB/WR) is injured?

    Hypothesis:
        Significant negative effect on team points scored.

    Expected Result:
        Negative ATT (fewer points).
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Team Offensive Production After Star Injury")
    logger.info("=" * 70)

    modeler = InjuryShockModeler()

    # Analyze all star positions
    results = modeler.analyze_injury_shock(
        start_season=2018,
        end_season=2024,
        outcome_type='backup_rushing_yards',  # Would be 'team_points' in real analysis
        positions=['QB', 'RB', 'WR'],  # All skill positions
        pre_periods=4,
        post_periods=4,
        run_placebo=False
    )

    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"Overall ATT: {results.att_overall:.2f}")
    print(f"P-value: {results.pvalue_overall:.4f}")

    print("\n" + "-" * 70)
    print("BETTING STRATEGY")
    print("-" * 70)
    print("Use star injury information for:")
    print("1. Team total UNDER bets (if ATT is negative)")
    print("2. Opponent total OVER bets (if defense can focus more)")
    print("3. Spread adjustments (line should move ~3-7 points for QB injury)")
    print("4. Prop bets on opposing defense (more sacks, INTs expected)")

    return results


# =============================================================================
# EXAMPLE 5: HETEROGENEITY ANALYSIS (BY INJURY SEVERITY)
# =============================================================================

def example_5_heterogeneity_by_severity():
    """
    Research Question:
        Does the effect differ based on injury severity (Out vs Doubtful)?

    Hypothesis:
        "Out" status has larger effect than "Doubtful" (higher certainty).

    Method:
        Run separate analyses for different injury statuses.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 5: Heterogeneity Analysis by Injury Severity")
    logger.info("=" * 70)

    # Analyze "Out" status
    logger.info("\n[1/2] Analyzing 'Out' status injuries...")
    db_out = InjuryShockDatabase()
    events_out = db_out.load_injury_events(
        start_season=2018,
        end_season=2024,
        positions=['RB'],
        severity=['Out']
    )
    star_out = db_out.identify_star_injuries(events_out)
    panel_out = db_out.create_panel_data(star_out, pre_periods=4, post_periods=4)

    modeler = InjuryShockModeler()
    modeler.estimator.panel_data = panel_out
    group_time_out = modeler.estimator.estimate_group_time_att(panel_out, bootstrap_reps=50)
    agg_out = modeler.estimator.aggregate_effects(group_time_out, 'overall')
    att_out = agg_out.iloc[0]['att'] if not agg_out.empty else 0
    se_out = agg_out.iloc[0]['se'] if not agg_out.empty else 0

    # Analyze "Doubtful" status
    logger.info("\n[2/2] Analyzing 'Doubtful' status injuries...")
    db_doubtful = InjuryShockDatabase()
    events_doubtful = db_doubtful.load_injury_events(
        start_season=2018,
        end_season=2024,
        positions=['RB'],
        severity=['Doubtful']
    )
    star_doubtful = db_doubtful.identify_star_injuries(events_doubtful)
    panel_doubtful = db_doubtful.create_panel_data(star_doubtful, pre_periods=4, post_periods=4)

    group_time_doubtful = modeler.estimator.estimate_group_time_att(panel_doubtful, bootstrap_reps=50)
    agg_doubtful = modeler.estimator.aggregate_effects(group_time_doubtful, 'overall')
    att_doubtful = agg_doubtful.iloc[0]['att'] if not agg_doubtful.empty else 0
    se_doubtful = agg_doubtful.iloc[0]['se'] if not agg_doubtful.empty else 0

    # Compare
    print("\n" + "-" * 70)
    print("HETEROGENEITY RESULTS")
    print("-" * 70)
    print(f"ATT (Out):      {att_out:.2f} ± {se_out:.2f}")
    print(f"ATT (Doubtful): {att_doubtful:.2f} ± {se_doubtful:.2f}")
    print(f"\nDifference:     {att_out - att_doubtful:.2f}")

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    if abs(att_out) > abs(att_doubtful):
        print("'Out' status has larger effect than 'Doubtful'.")
        print("This suggests injury certainty matters for betting markets.")
        print("\nStrategy: Wait for final injury report before placing bets.")
    else:
        print("Effect sizes are similar across injury statuses.")
        print("Markets may price in injury risk even with 'Doubtful' designation.")


# =============================================================================
# EXAMPLE 6: COMPARISON TO TWO-WAY FIXED EFFECTS (TWFE)
# =============================================================================

def example_6_comparison_to_twfe():
    """
    Methodological Example:
        Compare Callaway-Sant'Anna to traditional TWFE DiD.

    Purpose:
        Show why CS estimator is preferred for staggered treatments.

    Note:
        TWFE can produce biased estimates when treatment timing varies.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 6: Callaway-Sant'Anna vs TWFE Comparison")
    logger.info("=" * 70)

    # Create synthetic data with known ATT
    np.random.seed(42)
    true_att = 10.0

    panel_rows = []
    cohorts = ['2020W4', '2020W8', '2021W4']

    for cohort in cohorts:
        season, week = cohort.split('W')
        season = int(season)
        treatment_week = int(week)

        # Treated units for this cohort
        for i in range(5):
            player_id = f"TREATED_{cohort}_{i}"
            baseline = 50

            for t in range(-4, 5):
                week_num = treatment_week + t
                is_treated = 1 if t >= 0 else 0

                outcome = baseline + (true_att if is_treated else 0) + np.random.randn() * 5

                panel_rows.append({
                    'player_id': player_id,
                    'season': season,
                    'week': week_num,
                    'outcome': outcome,
                    'treated': is_treated,
                    'time_to_treatment': t,
                    'cohort': cohort
                })

    # Control group
    for i in range(20):
        player_id = f"CONTROL_{i}"
        baseline = 50

        for week_num in range(1, 18):
            panel_rows.append({
                'player_id': player_id,
                'season': 2020,
                'week': week_num,
                'outcome': baseline + np.random.randn() * 5,
                'treated': 0,
                'time_to_treatment': np.nan,
                'cohort': 'never_treated'
            })

    df = pd.DataFrame(panel_rows)

    # Estimate with Callaway-Sant'Anna
    from did_injury_analysis import DifferenceInDifferencesEstimator
    estimator = DifferenceInDifferencesEstimator()

    group_time_atts = estimator.estimate_group_time_att(df, bootstrap_reps=100)
    cs_agg = estimator.aggregate_effects(group_time_atts, 'overall')
    cs_att = cs_agg.iloc[0]['att']
    cs_se = cs_agg.iloc[0]['se']

    # Estimate with TWFE (simplified)
    treated_post = df[(df['treated'] == 1)]['outcome'].mean()
    treated_pre = df[(df['treated'] == 0) & (df['cohort'] != 'never_treated')]['outcome'].mean()
    control = df[df['cohort'] == 'never_treated']['outcome'].mean()

    twfe_att = (treated_post - treated_pre) - 0  # Simplified

    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON RESULTS")
    print("-" * 70)
    print(f"True ATT:                {true_att:.2f}")
    print(f"Callaway-Sant'Anna ATT:  {cs_att:.2f} ± {cs_se:.2f}")
    print(f"TWFE ATT (simplified):   {twfe_att:.2f}")

    print("\n" + "-" * 70)
    print("CONCLUSION")
    print("-" * 70)
    print("Callaway-Sant'Anna estimator properly handles staggered treatment timing")
    print("by computing group-specific ATTs and aggregating them with appropriate weights.")
    print("\nTWFE can be biased when:")
    print("1. Treatment timing varies across groups")
    print("2. Treatment effects are heterogeneous")
    print("3. There are dynamic treatment effects")
    print("\nFor NFL injury analysis with injuries occurring at different times,")
    print("Callaway-Sant'Anna is the preferred method.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run examples"""
    parser = argparse.ArgumentParser(description='DiD Injury Analysis Examples')
    parser.add_argument(
        '--example',
        type=str,
        default='1',
        help='Example number to run (1-6) or "all"'
    )

    args = parser.parse_args()

    examples = {
        '1': ('Backup RB Analysis', example_1_backup_rb_analysis),
        '2': ('QB Injury Analysis', example_2_qb_injury_analysis),
        '3': ('WR Target Redistribution', example_3_wr_target_redistribution),
        '4': ('Team Offensive Production', example_4_team_offensive_production),
        '5': ('Heterogeneity by Severity', example_5_heterogeneity_by_severity),
        '6': ('CS vs TWFE Comparison', example_6_comparison_to_twfe)
    }

    if args.example.lower() == 'all':
        for num, (name, func) in examples.items():
            logger.info(f"\nRunning Example {num}: {name}")
            try:
                func()
            except Exception as e:
                logger.error(f"Example {num} failed: {e}")
    else:
        if args.example in examples:
            name, func = examples[args.example]
            logger.info(f"\nRunning Example {args.example}: {name}")
            func()
        else:
            logger.error(f"Unknown example: {args.example}")
            logger.info(f"Available examples: {', '.join(examples.keys())}, all")


if __name__ == '__main__':
    main()
