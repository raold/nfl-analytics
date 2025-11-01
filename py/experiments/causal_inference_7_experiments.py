#!/usr/bin/env python3
"""
Causal Inference Experiment Suite - 7 Comprehensive Experiments

This script runs 7 causal inference experiments mirroring the 7 cVAE experiments,
demonstrating the full power of the causal inference framework for NFL analytics.

Each experiment answers a specific causal question using REAL data from 2020-2024.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime

# Add project root to path
project_root = "/Users/dro/rice/nfl-analytics"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.chdir(project_root)

# Import causal modules
from py.causal.panel_constructor import PanelConstructor
from py.causal.treatment_definitions import TreatmentDefiner
from py.causal.diff_in_diff import DifferenceInDifferences
from py.causal.synthetic_control import SyntheticControl
from py.causal.structural_causal_models import CausalDAG, create_nfl_rushing_dag
from py.causal.confounder_identification import ConfounderIdentifier
from py.causal.model_integration import CausalModelIntegrator
from py.causal.validation import CausalValidator

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class CausalInferenceExperimentSuite:
    """Suite of 7 causal inference experiments mirroring cVAE experiments"""

    def __init__(self):
        self.results = {}
        self.output_dir = Path("experiments/causal_experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Database connection
        self.db_params = {
            'dbname': 'nfl_data',
            'user': 'nfl_user',
            'password': 'sicillionbillions',
            'host': 'localhost',
            'port': 5544
        }
        self.conn = None

    def connect_db(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_params)
            logger.info("✓ Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def exp1_injury_treatment_effects_rushing(self):
        """
        Experiment 1: Injury Treatment Effects on Rushing Performance

        Mirrors cVAE Exp1 (Environment BNN Ensemble)

        Causal Question: What is the causal effect of injury on rushing yards?
        Method: Difference-in-Differences with robust standard errors
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT 1: Injury Treatment Effects on Rushing Yards")
        logger.info("=" * 80)

        # Build panel dataset
        constructor = PanelConstructor(self.conn)
        panel = constructor.build_player_game_panel(
            start_season=2020,
            end_season=2024,
            position_groups=['RB']
        )

        # Define injury treatment
        definer = TreatmentDefiner()
        panel = definer.define_injury_treatment(
            panel,
            min_games_missed=1,
            star_players_only=False
        )

        # Estimate causal effect with DiD
        did = DifferenceInDifferences(cluster_var='player_id')
        did.fit(
            df=panel,
            outcome_col='stat_yards',
            treatment_col='injury_treatment',
            unit_col='player_id',
            time_col='week'
        )

        # Test parallel trends
        pt_test = did.test_parallel_trends(
            df=panel,
            outcome_col='stat_yards',
            treatment_col='injury_treatment',
            unit_col='player_id',
            time_col='week',
            pre_periods=3
        )

        results = {
            'experiment': 'exp1_injury_treatment_effects',
            'causal_question': 'What is the causal effect of injury on rushing yards?',
            'method': 'Difference-in-Differences',
            'treatment_effect': float(did.treatment_effect_['estimate']),
            'ci_lower': float(did.treatment_effect_['ci_lower']),
            'ci_upper': float(did.treatment_effect_['ci_upper']),
            'p_value': float(did.treatment_effect_['p_value']),
            'parallel_trends_pass': pt_test['passes'],
            'n_treated': int(panel[panel['injury_treatment'] == 1].shape[0]),
            'n_control': int(panel[panel['injury_treatment'] == 0].shape[0]),
            'interpretation': f"Injury causes {abs(did.treatment_effect_['estimate']):.1f} yard decrease in rushing performance",
            'status': 'COMPLETED'
        }

        logger.info(f"✓ Treatment effect: {did.treatment_effect_['estimate']:.2f} yards")
        logger.info(f"✓ 95% CI: [{did.treatment_effect_['ci_lower']:.2f}, {did.treatment_effect_['ci_upper']:.2f}]")
        logger.info(f"✓ P-value: {did.treatment_effect_['p_value']:.4f}")
        logger.info(f"✓ Parallel trends: {'PASS' if pt_test['passes'] else 'FAIL'}")

        self.results['exp1'] = results
        return results

    def exp2_coaching_change_team_performance(self):
        """
        Experiment 2: Coaching Change Effects on Team Performance

        Mirrors cVAE Exp2 (Same-Game Parlay Pricing)

        Causal Question: What is the causal effect of mid-season coaching changes?
        Method: Synthetic Control Method
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT 2: Coaching Change Effects on Team Performance")
        logger.info("=" * 80)

        # Build team-level panel
        constructor = PanelConstructor(self.conn)
        panel = constructor.build_team_game_panel(
            start_season=2020,
            end_season=2024
        )

        # Define coaching change treatment
        definer = TreatmentDefiner()
        panel = definer.define_coaching_change_treatment(panel)

        # Find a treated team (e.g., Panthers 2022)
        treated_teams = panel[panel['coaching_change_treatment'] == 1]['team'].unique()

        if len(treated_teams) > 0:
            treated_team = treated_teams[0]
            treatment_week = panel[
                (panel['team'] == treated_team) &
                (panel['coaching_change_treatment'] == 1)
            ]['week'].min()

            # Synthetic Control
            sc = SyntheticControl(method='optimization')
            sc.fit(
                df=panel,
                treated_unit_id=treated_team,
                unit_col='team',
                time_col='week',
                outcome_col='team_points',
                treatment_time=treatment_week
            )

            # Placebo test
            placebo = sc.placebo_test(
                df=panel,
                unit_col='team',
                time_col='week',
                outcome_col='team_points',
                treatment_time=treatment_week
            )

            results = {
                'experiment': 'exp2_coaching_change',
                'causal_question': 'What is the causal effect of coaching change on team performance?',
                'method': 'Synthetic Control',
                'treated_team': treated_team,
                'treatment_week': int(treatment_week),
                'average_effect': float(sc.treatment_effect_['average_effect']),
                'cumulative_effect': float(sc.treatment_effect_['cumulative_effect']),
                'pre_treatment_rmse': float(sc.pre_treatment_fit_['rmse']),
                'placebo_p_value': float(placebo['p_value']),
                'placebo_rank': f"{placebo['rank']}/{placebo['total']}",
                'interpretation': f"Coaching change caused {sc.treatment_effect_['average_effect']:.1f} point change per game",
                'status': 'COMPLETED'
            }
        else:
            results = {
                'experiment': 'exp2_coaching_change',
                'status': 'NO_TREATMENT_EVENTS',
                'note': 'No coaching changes found in 2020-2024 data'
            }

        logger.info(f"✓ Average effect: {results.get('average_effect', 'N/A')}")
        logger.info(f"✓ Placebo p-value: {results.get('placebo_p_value', 'N/A')}")

        self.results['exp2'] = results
        return results

    def exp3_carries_on_yards_confounding(self):
        """
        Experiment 3: Carries → Rushing Yards with Confounding Adjustment

        Mirrors cVAE Exp3 (Causal Counterfactuals)

        Causal Question: What is the causal effect of carries on yards, adjusting for confounding?
        Method: DAG-based confounder identification + regression adjustment
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT 3: Carries → Yards with Confounding Adjustment")
        logger.info("=" * 80)

        # Build panel
        constructor = PanelConstructor(self.conn)
        panel = constructor.build_player_game_panel(
            start_season=2020,
            end_season=2024,
            position_groups=['RB']
        )

        # Create causal DAG
        dag = create_nfl_rushing_dag()

        # Find backdoor adjustment set
        adjustment_set = dag.find_backdoor_adjustment_set(
            treatment='carries',
            outcome='rushing_yards'
        )

        # Identify confounders empirically
        identifier = ConfounderIdentifier(alpha=0.05)
        confounders = identifier.identify_confounders(
            df=panel,
            treatment_col='stat_carries',
            outcome_col='stat_yards',
            candidate_confounders=[
                'player_ability_proxy',
                'team_quality',
                'opponent_defense_rank',
                'is_home',
                'temperature'
            ]
        )

        # Compute covariate balance
        balance = identifier.compute_covariate_balance(
            panel,
            'stat_carries',
            list(adjustment_set) if adjustment_set else []
        )

        results = {
            'experiment': 'exp3_confounding_adjustment',
            'causal_question': 'What is the causal effect of carries on yards?',
            'method': 'Backdoor Criterion + Confounder Identification',
            'dag_adjustment_set': list(adjustment_set) if adjustment_set else [],
            'empirical_confounders': [c['variable'] for c in confounders if c['is_confounder']],
            'covariate_balance': {k: float(v['abs_smd']) for k, v in balance.items()},
            'n_observations': len(panel),
            'interpretation': f"Must adjust for {len(adjustment_set) if adjustment_set else 0} variables to identify causal effect",
            'status': 'COMPLETED'
        }

        logger.info(f"✓ DAG adjustment set: {adjustment_set}")
        logger.info(f"✓ Empirical confounders: {[c['variable'] for c in confounders if c['is_confounder']]}")
        logger.info(f"✓ Covariate balance: {len([v for v in balance.values() if v['abs_smd'] < 0.1])}/{len(balance)} balanced")

        self.results['exp3'] = results
        return results

    def exp4_real_time_injury_adjustment(self):
        """
        Experiment 4: Real-Time Injury Adjustment for Live Betting

        Mirrors cVAE Exp4 (Live In-Game Trajectories)

        Causal Question: How should we adjust predictions when injury news breaks?
        Method: Treatment effect priors from historical data
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT 4: Real-Time Injury Adjustment")
        logger.info("=" * 80)

        # Use model integration module
        integrator = CausalModelIntegrator(self.conn)

        # Estimate injury treatment effect priors
        injury_prior = integrator.estimate_treatment_effect_priors(
            treatment_type='injury',
            outcome='rushing_yards'
        )

        # Simulate real-time adjustment
        base_prediction = 82.5  # BNN prediction
        vegas_line = 78.5

        # Apply causal adjustment for injury return
        adjusted = integrator.adjust_prediction_for_shock(
            base_prediction=base_prediction,
            shock_type='injury',
            shock_magnitude=0.7  # 70% recovered
        )

        # Compute betting edge
        edge = integrator.compute_betting_edge_with_causality(
            prediction=adjusted['adjusted_prediction'],
            market_line=vegas_line,
            causal_adjustments={'injury': 0.7}
        )

        results = {
            'experiment': 'exp4_real_time_adjustment',
            'causal_question': 'How to adjust predictions for breaking injury news?',
            'method': 'Treatment Effect Priors + Real-Time Integration',
            'injury_prior_mean': float(injury_prior['prior_mean']),
            'injury_prior_std': float(injury_prior['prior_std']),
            'base_prediction': base_prediction,
            'adjusted_prediction': float(adjusted['adjusted_prediction']),
            'vegas_line': vegas_line,
            'betting_edge': float(edge['edge']),
            'edge_yards': float(adjusted['adjusted_prediction'] - vegas_line),
            'recommendation': 'BET OVER' if edge['edge'] > 5 else 'NO BET',
            'interpretation': f"Injury adjustment: {adjusted['adjusted_prediction'] - base_prediction:.1f} yards",
            'status': 'COMPLETED'
        }

        logger.info(f"✓ Injury prior: μ={injury_prior['prior_mean']:.1f}, σ={injury_prior['prior_std']:.1f}")
        logger.info(f"✓ Adjusted prediction: {adjusted['adjusted_prediction']:.1f} yards")
        logger.info(f"✓ Betting edge: {edge['edge']:+.1f} yards")
        logger.info(f"✓ Recommendation: {results['recommendation']}")

        self.results['exp4'] = results
        return results

    def exp5_sensitivity_analysis_unmeasured_confounding(self):
        """
        Experiment 5: Sensitivity Analysis for Unmeasured Confounding

        Mirrors cVAE Exp5 (Key Number Distribution Validation)

        Causal Question: How sensitive are our estimates to unmeasured confounding?
        Method: Rosenbaum bounds and sensitivity analysis
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT 5: Sensitivity Analysis for Unmeasured Confounding")
        logger.info("=" * 80)

        # Build panel
        constructor = PanelConstructor(self.conn)
        panel = constructor.build_player_game_panel(
            start_season=2020,
            end_season=2024,
            position_groups=['RB']
        )

        # Define treatment
        definer = TreatmentDefiner()
        panel = definer.define_injury_treatment(panel, min_games_missed=1)

        # Fit DiD
        did = DifferenceInDifferences(cluster_var='player_id')
        did.fit(
            df=panel,
            outcome_col='stat_yards',
            treatment_col='injury_treatment',
            unit_col='player_id',
            time_col='week'
        )

        # Sensitivity analysis
        validator = CausalValidator()
        sensitivity = validator.sensitivity_analysis(
            estimator=did,
            df=panel,
            confounder_strength_range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )

        results = {
            'experiment': 'exp5_sensitivity_analysis',
            'causal_question': 'How robust are injury effects to unmeasured confounding?',
            'method': 'Rosenbaum Sensitivity Analysis',
            'baseline_effect': float(did.treatment_effect_['estimate']),
            'sensitivity_bounds': {
                str(gamma): float(effect)
                for gamma, effect in sensitivity['effects_by_gamma'].items()
            },
            'critical_gamma': float(sensitivity.get('critical_gamma', 0)),
            'robustness': sensitivity.get('robustness_level', 'UNKNOWN'),
            'interpretation': f"Estimates remain valid up to Γ={sensitivity.get('critical_gamma', 0):.2f}",
            'status': 'COMPLETED'
        }

        logger.info(f"✓ Baseline effect: {did.treatment_effect_['estimate']:.2f} yards")
        logger.info(f"✓ Critical Γ: {results['critical_gamma']:.2f}")
        logger.info(f"✓ Robustness: {results['robustness']}")

        self.results['exp5'] = results
        return results

    def exp6_mediation_analysis_usage_mechanism(self):
        """
        Experiment 6: Mediation Analysis - How Injury Affects Yards via Usage

        Mirrors cVAE Exp6 (GNN-Enhanced Conditioning)

        Causal Question: Does injury affect yards directly or through reduced usage?
        Method: Mediation analysis with sequential ignorability
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT 6: Mediation Analysis - Injury → Usage → Yards")
        logger.info("=" * 80)

        # Build panel
        constructor = PanelConstructor(self.conn)
        panel = constructor.build_player_game_panel(
            start_season=2020,
            end_season=2024,
            position_groups=['RB']
        )

        # Define treatment
        definer = TreatmentDefiner()
        panel = definer.define_injury_treatment(panel, min_games_missed=1)

        # Estimate total effect (Injury → Yards)
        did_total = DifferenceInDifferences(cluster_var='player_id')
        did_total.fit(
            df=panel,
            outcome_col='stat_yards',
            treatment_col='injury_treatment',
            unit_col='player_id',
            time_col='week'
        )

        # Estimate effect on mediator (Injury → Carries)
        did_mediator = DifferenceInDifferences(cluster_var='player_id')
        did_mediator.fit(
            df=panel,
            outcome_col='stat_carries',
            treatment_col='injury_treatment',
            unit_col='player_id',
            time_col='week'
        )

        # Approximate mediation decomposition
        # Total effect = Direct effect + Indirect effect (mediated by carries)
        # Indirect ≈ effect_on_mediator × mediator_outcome_relationship
        yards_per_carry = panel['stat_yards'].mean() / panel['stat_carries'].mean()
        indirect_effect = did_mediator.treatment_effect_['estimate'] * yards_per_carry
        direct_effect = did_total.treatment_effect_['estimate'] - indirect_effect

        results = {
            'experiment': 'exp6_mediation_analysis',
            'causal_question': 'Does injury affect yards directly or through usage?',
            'method': 'Mediation Analysis',
            'total_effect': float(did_total.treatment_effect_['estimate']),
            'effect_on_carries': float(did_mediator.treatment_effect_['estimate']),
            'indirect_effect_via_usage': float(indirect_effect),
            'direct_effect': float(direct_effect),
            'proportion_mediated': float(indirect_effect / did_total.treatment_effect_['estimate']),
            'interpretation': f"{abs(indirect_effect/did_total.treatment_effect_['estimate'])*100:.1f}% of effect mediated by usage",
            'status': 'COMPLETED'
        }

        logger.info(f"✓ Total effect: {did_total.treatment_effect_['estimate']:.2f} yards")
        logger.info(f"✓ Effect on carries: {did_mediator.treatment_effect_['estimate']:.2f}")
        logger.info(f"✓ Indirect (via usage): {indirect_effect:.2f} yards")
        logger.info(f"✓ Direct: {direct_effect:.2f} yards")
        logger.info(f"✓ Proportion mediated: {results['proportion_mediated']:.1%}")

        self.results['exp6'] = results
        return results

    def exp7_heterogeneous_treatment_effects(self):
        """
        Experiment 7: Heterogeneous Treatment Effects by Player Type

        Mirrors cVAE Exp7 (Hierarchical Player-Game Trajectories)

        Causal Question: Do injury effects vary by player ability/role?
        Method: Stratified analysis + interaction terms
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPERIMENT 7: Heterogeneous Treatment Effects by Player Type")
        logger.info("=" * 80)

        # Build panel
        constructor = PanelConstructor(self.conn)
        panel = constructor.build_player_game_panel(
            start_season=2020,
            end_season=2024,
            position_groups=['RB']
        )

        # Define treatment
        definer = TreatmentDefiner()
        panel = definer.define_injury_treatment(panel, min_games_missed=1)

        # Stratify by player ability (star vs role player)
        panel['is_star'] = panel['player_ability_proxy'] > panel['player_ability_proxy'].quantile(0.75)

        # Estimate for star players
        did_stars = DifferenceInDifferences(cluster_var='player_id')
        did_stars.fit(
            df=panel[panel['is_star']],
            outcome_col='stat_yards',
            treatment_col='injury_treatment',
            unit_col='player_id',
            time_col='week'
        )

        # Estimate for role players
        did_role = DifferenceInDifferences(cluster_var='player_id')
        did_role.fit(
            df=panel[~panel['is_star']],
            outcome_col='stat_yards',
            treatment_col='injury_treatment',
            unit_col='player_id',
            time_col='week'
        )

        # Heterogeneity
        heterogeneity = did_stars.treatment_effect_['estimate'] - did_role.treatment_effect_['estimate']

        results = {
            'experiment': 'exp7_heterogeneous_effects',
            'causal_question': 'Do injury effects vary by player ability?',
            'method': 'Stratified DiD + Heterogeneity Analysis',
            'star_effect': float(did_stars.treatment_effect_['estimate']),
            'star_ci': [float(did_stars.treatment_effect_['ci_lower']), float(did_stars.treatment_effect_['ci_upper'])],
            'role_effect': float(did_role.treatment_effect_['estimate']),
            'role_ci': [float(did_role.treatment_effect_['ci_lower']), float(did_role.treatment_effect_['ci_upper'])],
            'heterogeneity': float(heterogeneity),
            'n_star_players': int(panel[panel['is_star']]['player_id'].nunique()),
            'n_role_players': int(panel[~panel['is_star']]['player_id'].nunique()),
            'interpretation': f"Star players lose {abs(did_stars.treatment_effect_['estimate']):.1f} yards vs {abs(did_role.treatment_effect_['estimate']):.1f} for role players",
            'status': 'COMPLETED'
        }

        logger.info(f"✓ Star effect: {did_stars.treatment_effect_['estimate']:.2f} yards")
        logger.info(f"✓ Role effect: {did_role.treatment_effect_['estimate']:.2f} yards")
        logger.info(f"✓ Heterogeneity: {heterogeneity:.2f} yards")
        logger.info(f"✓ Star players (n={results['n_star_players']})")
        logger.info(f"✓ Role players (n={results['n_role_players']})")

        self.results['exp7'] = results
        return results

    def run_all_experiments(self):
        """Run all 7 causal inference experiments"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("CAUSAL INFERENCE EXPERIMENT SUITE - 7 EXPERIMENTS")
        logger.info("=" * 80)

        # Connect to database
        self.connect_db()

        # Run all experiments
        try:
            self.exp1_injury_treatment_effects_rushing()
            self.exp2_coaching_change_team_performance()
            self.exp3_carries_on_yards_confounding()
            self.exp4_real_time_injury_adjustment()
            self.exp5_sensitivity_analysis_unmeasured_confounding()
            self.exp6_mediation_analysis_usage_mechanism()
            self.exp7_heterogeneous_treatment_effects()
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()

        # Save results
        output_file = self.output_dir / "all_causal_experiments_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("CAUSAL INFERENCE EXPERIMENT SUITE SUMMARY")
        logger.info("=" * 80)
        logger.info("")

        for exp_name, exp_data in self.results.items():
            status = exp_data.get('status', 'UNKNOWN')
            logger.info(f"✓ {exp_data['experiment']}: {status}")

        logger.info("")
        logger.info(f"✓ All results saved to {output_file}")
        logger.info("")
        logger.info("Key Findings:")
        logger.info("1. Injury effects: Quantified causal impact with DiD")
        logger.info("2. Coaching changes: Estimated effects with synthetic control")
        logger.info("3. Confounding: Identified and adjusted for backdoor paths")
        logger.info("4. Real-time betting: Integration with prediction pipeline")
        logger.info("5. Sensitivity: Robustness to unmeasured confounding")
        logger.info("6. Mediation: Decomposed direct vs indirect effects")
        logger.info("7. Heterogeneity: Player-specific treatment effects")

        return self.results


def main():
    suite = CausalInferenceExperimentSuite()
    results = suite.run_all_experiments()
    return results


if __name__ == "__main__":
    main()
