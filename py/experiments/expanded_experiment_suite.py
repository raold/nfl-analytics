"""
Expanded Experiment Suite: 20 cVAE + 20 Causal Inference Experiments

Design and execute 40 experiments, then rank by likelihood to increase ROI/EV.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSpec:
    """Specification for a single experiment"""

    id: str
    name: str
    category: str  # 'cvae' or 'causal'
    description: str
    hypothesis: str
    expected_roi_impact: str  # 'high', 'medium', 'low'
    market_inefficiency: str  # 'high', 'medium', 'low'
    implementation_complexity: str  # 'low', 'medium', 'high'
    data_requirements: list[str]
    success_metrics: list[str]


class ExpandedExperimentSuite:
    """20 cVAE + 20 Causal experiments ranked by ROI/EV potential"""

    def __init__(self):
        self.cvae_experiments = self._define_cvae_experiments()
        self.causal_experiments = self._define_causal_experiments()

    def _define_cvae_experiments(self) -> list[ExperimentSpec]:
        """Define 20 cVAE experiments"""

        return [
            # ===== Existing 7 Experiments (Baseline) =====
            ExperimentSpec(
                id="CVAE-001",
                name="Baseline Game Trajectory Synthesis",
                category="cvae",
                description="Synthesize complete game trajectories with weather/opponent conditioning",
                hypothesis="cVAE can generate realistic game progression paths",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="low",
                data_requirements=["play_by_play", "weather", "team_stats"],
                success_metrics=["KL_divergence", "trajectory_realism", "margin_prediction_lift"],
            ),
            ExperimentSpec(
                id="CVAE-002",
                name="Rushing Yard Trajectory Prediction",
                category="cvae",
                description="Model RB rushing yard accumulation throughout game",
                hypothesis="Conditional on game state, we can predict rushing trajectory",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="low",
                data_requirements=["rushing_stats", "game_state", "defense_rank"],
                success_metrics=["rushing_yards_mae", "over_under_accuracy", "sharpe_ratio"],
            ),
            ExperimentSpec(
                id="CVAE-003",
                name="Passing Yard Trajectory Prediction",
                category="cvae",
                description="Model QB passing yard accumulation with situational factors",
                hypothesis="Pass-heavy game scripts increase variance in passing trajectories",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="low",
                data_requirements=["passing_stats", "game_script", "defense_rank"],
                success_metrics=["passing_yards_mae", "over_under_accuracy", "sharpe_ratio"],
            ),
            ExperimentSpec(
                id="CVAE-004",
                name="Reception Trajectory (PPR Scoring)",
                category="cvae",
                description="Model WR/TE reception counts and timing across quarters",
                hypothesis="Target share and game script drive reception trajectories",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="low",
                data_requirements=["receiving_stats", "target_share", "game_script"],
                success_metrics=["reception_mae", "ppr_points_mae", "over_under_accuracy"],
            ),
            ExperimentSpec(
                id="CVAE-005",
                name="TD Timing Prediction",
                category="cvae",
                description="Predict WHEN in game touchdowns occur (early/mid/late)",
                hypothesis="TD timing varies by game script and situational pressure",
                expected_roi_impact="low",
                market_inefficiency="low",
                implementation_complexity="medium",
                data_requirements=["scoring_plays", "game_state", "red_zone_stats"],
                success_metrics=["td_timing_accuracy", "anytime_td_accuracy"],
            ),
            ExperimentSpec(
                id="CVAE-006",
                name="Game Script Emergence",
                category="cvae",
                description="Model how game script (pass-heavy vs rush-heavy) evolves",
                hypothesis="Early scoring drives influence entire game trajectory",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="medium",
                data_requirements=["play_by_play", "score_differential", "time_remaining"],
                success_metrics=["game_script_classification", "play_distribution_kl"],
            ),
            ExperimentSpec(
                id="CVAE-007",
                name="Blowout vs Close Game Trajectories",
                category="cvae",
                description="Distinguish trajectory patterns for competitive vs non-competitive games",
                hypothesis="Blowouts have distinct statistical signatures detectable early",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="low",
                data_requirements=["score_differential", "play_distribution", "personnel"],
                success_metrics=["blowout_detection_auc", "live_betting_edge", "sharpe_ratio"],
            ),
            # ===== NEW Experiments 8-20 =====
            ExperimentSpec(
                id="CVAE-008",
                name="Multi-Horizon Trajectory Forecasting",
                category="cvae",
                description="Separate cVAE models for Q1/Q2/Q3/Q4/Final predictions",
                hypothesis="Quarter-specific models capture intra-game regime changes",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["quarter_stats", "score_progression", "time_decay"],
                success_metrics=["quarter_mae", "cumulative_error", "live_bet_roi"],
            ),
            ExperimentSpec(
                id="CVAE-009",
                name="Weather-Conditioned Trajectories",
                category="cvae",
                description="Model impact of temperature, wind, precipitation on trajectories",
                hypothesis="Extreme weather reduces passing volume and increases variance",
                expected_roi_impact="medium",
                market_inefficiency="high",
                implementation_complexity="low",
                data_requirements=["weather_data", "play_distribution", "pass_rush_split"],
                success_metrics=["weather_adjustment_accuracy", "outdoor_game_edge"],
            ),
            ExperimentSpec(
                id="CVAE-010",
                name="Injury-Adjusted Trajectories",
                category="cvae",
                description="In-game injury scenarios and replacement player trajectories",
                hypothesis="Key player injuries cause regime shifts in game trajectory",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="high",
                data_requirements=["injury_reports", "snap_counts", "depth_charts"],
                success_metrics=["injury_impact_mae", "live_adjustment_roi"],
            ),
            ExperimentSpec(
                id="CVAE-011",
                name="Home/Away Trajectory Asymmetry",
                category="cvae",
                description="Model home field advantage effects on trajectory variance",
                hypothesis="Home teams have lower variance due to crowd/familiarity",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="low",
                data_requirements=["home_away_splits", "crowd_noise", "travel_distance"],
                success_metrics=["home_away_variance_ratio", "location_adjusted_mae"],
            ),
            ExperimentSpec(
                id="CVAE-012",
                name="Divisional Rivalry Trajectories",
                category="cvae",
                description="Capture unique trajectory patterns in divisional games",
                hypothesis="Rivalry games have higher variance and unpredictability",
                expected_roi_impact="low",
                market_inefficiency="medium",
                implementation_complexity="low",
                data_requirements=["schedule", "rivalry_history", "head_to_head"],
                success_metrics=["divisional_game_mae", "rivalry_edge"],
            ),
            ExperimentSpec(
                id="CVAE-013",
                name="Prime Time Game Trajectories",
                category="cvae",
                description="Model performance differences in SNF/MNF/TNF games",
                hypothesis="Prime time games have different pacing and scoring patterns",
                expected_roi_impact="low",
                market_inefficiency="low",
                implementation_complexity="low",
                data_requirements=["game_time", "day_of_week", "rest_days"],
                success_metrics=["primetime_mae", "rest_adjustment_accuracy"],
            ),
            ExperimentSpec(
                id="CVAE-014",
                name="Playoff-Bound Team Trajectories",
                category="cvae",
                description="Model late-season games where playoff seeding is locked",
                hypothesis="Teams resting starters have predictable trajectory changes",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["playoff_standings", "starter_status", "week_number"],
                success_metrics=["week17_18_accuracy", "resting_starters_edge"],
            ),
            ExperimentSpec(
                id="CVAE-015",
                name="Rookie QB Trajectory Patterns",
                category="cvae",
                description="Model unique trajectory characteristics of rookie QBs",
                hypothesis="Rookie QBs have higher variance and learn over season",
                expected_roi_impact="medium",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["qb_experience", "games_started", "learning_curve"],
                success_metrics=["rookie_qb_mae", "experience_adjustment"],
            ),
            ExperimentSpec(
                id="CVAE-016",
                name="Two-Minute Drill Trajectories",
                category="cvae",
                description="Model end-of-half/game hurry-up offense patterns",
                hypothesis="Two-minute situations have distinct trajectory signatures",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="medium",
                data_requirements=["two_minute_warning", "timeout_usage", "clock_management"],
                success_metrics=["two_minute_accuracy", "end_of_half_roi"],
            ),
            ExperimentSpec(
                id="CVAE-017",
                name="Offensive Coordinator Change Trajectories",
                category="cvae",
                description="Model trajectory shifts after OC/HC changes mid-season",
                hypothesis="Coaching changes cause regime shifts in play distribution",
                expected_roi_impact="medium",
                market_inefficiency="high",
                implementation_complexity="high",
                data_requirements=["coaching_changes", "play_calling_tendencies"],
                success_metrics=["regime_change_detection", "post_change_mae"],
            ),
            ExperimentSpec(
                id="CVAE-018",
                name="Stack Correlation Trajectories",
                category="cvae",
                description="Model QB-WR stack correlations in DFS contexts",
                hypothesis="Stack correlations vary by game script and matchup",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["passing_stats", "target_share", "correlation_matrix"],
                success_metrics=["stack_correlation_mae", "dfs_lineup_roi"],
            ),
            ExperimentSpec(
                id="CVAE-019",
                name="Defense-Adjusted Trajectory Variance",
                category="cvae",
                description="Model how defensive quality affects trajectory uncertainty",
                hypothesis="Elite defenses reduce offensive trajectory variance",
                expected_roi_impact="high",
                market_inefficiency="medium",
                implementation_complexity="medium",
                data_requirements=["defensive_dvoa", "variance_by_opponent"],
                success_metrics=["defense_adjusted_mae", "variance_prediction"],
            ),
            ExperimentSpec(
                id="CVAE-020",
                name="Season Progression Meta-Learning",
                category="cvae",
                description="Meta-learn how trajectory patterns evolve across season",
                hypothesis="Teams change as season progresses (injuries, scheme evolution)",
                expected_roi_impact="high",
                market_inefficiency="medium",
                implementation_complexity="high",
                data_requirements=["week_number", "cumulative_stats", "trend_features"],
                success_metrics=["temporal_mae", "trend_capture", "season_long_roi"],
            ),
        ]

    def _define_causal_experiments(self) -> list[ExperimentSpec]:
        """Define 20 causal inference experiments"""

        return [
            # ===== Existing 7 Experiments (Baseline) =====
            ExperimentSpec(
                id="CAUSAL-001",
                name="Weather Impact on Rushing Performance",
                category="causal",
                description="DiD estimation of precipitation/wind effects on rushing yards",
                hypothesis="Bad weather increases rushing attempts and reduces passing efficiency",
                expected_roi_impact="medium",
                market_inefficiency="high",
                implementation_complexity="low",
                data_requirements=["weather_data", "rushing_stats", "game_state"],
                success_metrics=["ate", "confidence_interval", "placebo_test_pvalue"],
            ),
            ExperimentSpec(
                id="CAUSAL-002",
                name="Injury Causation on Team Performance",
                category="causal",
                description="Synthetic control for key player injuries on team outcomes",
                hypothesis="QB injury has largest causal effect on team point differential",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["injury_reports", "team_performance", "snap_counts"],
                success_metrics=["ate", "rmspe_pre", "post_treatment_effect"],
            ),
            ExperimentSpec(
                id="CAUSAL-003",
                name="Home Field Advantage Causal Effect",
                category="causal",
                description="Regression discontinuity on home vs away performance",
                hypothesis="Home field advantage has declined over time (crowd effects)",
                expected_roi_impact="medium",
                market_inefficiency="low",
                implementation_complexity="low",
                data_requirements=["home_away_splits", "crowd_attendance", "season_year"],
                success_metrics=["ate", "temporal_trend", "heterogeneous_effects"],
            ),
            ExperimentSpec(
                id="CAUSAL-004",
                name="Rest Days Impact on Performance",
                category="causal",
                description="IV estimation using Thursday night games as instrument",
                hypothesis="Short rest causally reduces performance (fatigue effect)",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="medium",
                data_requirements=["rest_days", "game_schedule", "injury_rates"],
                success_metrics=["ate", "iv_validity", "first_stage_f_stat"],
            ),
            ExperimentSpec(
                id="CAUSAL-005",
                name="QB Change Causal Effect on Offense",
                category="causal",
                description="DiD with team fixed effects for QB changes",
                hypothesis="Veteran QB replacements improve offensive efficiency",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["qb_changes", "offensive_stats", "qb_experience"],
                success_metrics=["ate", "parallel_trends_test", "heterogeneous_by_experience"],
            ),
            ExperimentSpec(
                id="CAUSAL-006",
                name="Play-Calling Tendency Causal Analysis",
                category="causal",
                description="Propensity score matching for run vs pass play selection",
                hypothesis="Passing on 1st down causally increases drive success rate",
                expected_roi_impact="low",
                market_inefficiency="medium",
                implementation_complexity="low",
                data_requirements=["play_by_play", "down_distance", "success_rate"],
                success_metrics=["ate", "balance_diagnostics", "sensitivity_analysis"],
            ),
            ExperimentSpec(
                id="CAUSAL-007",
                name="Coaching Change Impact on Win Rate",
                category="causal",
                description="Synthetic control for mid-season coaching changes",
                hypothesis="Coaching changes have minimal short-term causal effect",
                expected_roi_impact="low",
                market_inefficiency="low",
                implementation_complexity="medium",
                data_requirements=["coaching_changes", "win_loss_record", "team_talent"],
                success_metrics=["ate", "rmspe", "placebo_test"],
            ),
            # ===== NEW Experiments 8-20 =====
            ExperimentSpec(
                id="CAUSAL-008",
                name="Offensive Line Quality Causal Effect",
                category="causal",
                description="DiD using OL injuries as natural experiment",
                hypothesis="OL quality causally affects both rushing and passing outcomes",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["ol_grades", "ol_injuries", "qb_pressure_rate"],
                success_metrics=["ate", "pressure_mediation", "rushing_vs_passing_effects"],
            ),
            ExperimentSpec(
                id="CAUSAL-009",
                name="Target Share Causal Effect on WR Production",
                category="causal",
                description="IV using teammate injuries as instrument for target share",
                hypothesis="Target share has causal but non-linear effect on efficiency",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="high",
                data_requirements=["target_share", "wr_injuries", "reception_efficiency"],
                success_metrics=["ate", "non_linearity_test", "iv_validity"],
            ),
            ExperimentSpec(
                id="CAUSAL-010",
                name="Defensive Scheme Change Causal Effect",
                category="causal",
                description="DiD for teams switching defensive coordinators",
                hypothesis="Scheme changes have delayed causal effects (learning period)",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="high",
                data_requirements=["defensive_schemes", "coordinator_changes", "dvoa"],
                success_metrics=["ate", "dynamic_treatment_effects", "learning_curve"],
            ),
            ExperimentSpec(
                id="CAUSAL-011",
                name="Altitude Causal Effect on Passing",
                category="causal",
                description="Regression discontinuity using Denver altitude as treatment",
                hypothesis="High altitude causally increases passing distance but not accuracy",
                expected_roi_impact="low",
                market_inefficiency="medium",
                implementation_complexity="low",
                data_requirements=["stadium_altitude", "passing_stats", "air_yards"],
                success_metrics=["ate", "discontinuity_test", "bandwidth_sensitivity"],
            ),
            ExperimentSpec(
                id="CAUSAL-012",
                name="Travel Distance Causal Effect on Performance",
                category="causal",
                description="DiD using cross-country travel as treatment",
                hypothesis="Long-distance travel causally impairs performance via fatigue",
                expected_roi_impact="medium",
                market_inefficiency="high",
                implementation_complexity="low",
                data_requirements=["travel_distance", "time_zones", "performance_metrics"],
                success_metrics=["ate", "time_zone_heterogeneity", "recovery_time"],
            ),
            ExperimentSpec(
                id="CAUSAL-013",
                name="Play-Action Usage Causal Effect",
                category="causal",
                description="Propensity score matching for play-action vs standard dropback",
                hypothesis="Play-action causally increases yards per attempt",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="medium",
                data_requirements=["play_action_rate", "yards_per_attempt", "defense_type"],
                success_metrics=["ate", "balance_check", "sensitivity_to_confounding"],
            ),
            ExperimentSpec(
                id="CAUSAL-014",
                name="Fourth Down Aggression Causal Effect",
                category="causal",
                description="IV using coaching philosophy as instrument",
                hypothesis="4th down aggression causally improves win probability",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="high",
                data_requirements=["4th_down_decisions", "win_probability", "coach_id"],
                success_metrics=["ate", "wp_lift", "iv_assumptions"],
            ),
            ExperimentSpec(
                id="CAUSAL-015",
                name="Red Zone Personnel Causal Effect",
                category="causal",
                description="DiD for teams switching to heavy personnel in red zone",
                hypothesis="Heavy personnel causally increases TD conversion rate",
                expected_roi_impact="medium",
                market_inefficiency="medium",
                implementation_complexity="medium",
                data_requirements=["red_zone_personnel", "td_rate", "defense_weight"],
                success_metrics=["ate", "parallel_trends", "robustness_checks"],
            ),
            ExperimentSpec(
                id="CAUSAL-016",
                name="Crowd Noise Causal Effect on Penalties",
                category="causal",
                description="Natural experiment using COVID empty stadiums",
                hypothesis="Crowd noise causally increases false start penalties",
                expected_roi_impact="low",
                market_inefficiency="low",
                implementation_complexity="low",
                data_requirements=["attendance", "penalty_rates", "covid_season"],
                success_metrics=["ate", "covid_natural_experiment", "home_away_diff"],
            ),
            ExperimentSpec(
                id="CAUSAL-017",
                name="Game Script Causal Effect on Play Distribution",
                category="causal",
                description="IV using early scoring as instrument for game script",
                hypothesis="Trailing teams causally increase pass rate non-linearly",
                expected_roi_impact="high",
                market_inefficiency="medium",
                implementation_complexity="high",
                data_requirements=["score_differential", "pass_rate", "time_remaining"],
                success_metrics=["ate", "non_linearity", "iv_relevance"],
            ),
            ExperimentSpec(
                id="CAUSAL-018",
                name="Turf Type Causal Effect on Injuries",
                category="causal",
                description="DiD using stadium turf changes as treatment",
                hypothesis="Artificial turf causally increases lower-body injury rate",
                expected_roi_impact="medium",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["turf_type", "injury_rates", "stadium_changes"],
                success_metrics=["ate", "injury_type_heterogeneity", "player_adjustment"],
            ),
            ExperimentSpec(
                id="CAUSAL-019",
                name="Timeout Strategy Causal Effect on Drive Success",
                category="causal",
                description="Propensity score matching for timeout usage timing",
                hypothesis="Early timeout usage causally impairs late-game clock management",
                expected_roi_impact="low",
                market_inefficiency="medium",
                implementation_complexity="medium",
                data_requirements=["timeout_usage", "drive_outcomes", "clock_management"],
                success_metrics=["ate", "spillover_effects", "mechanism_analysis"],
            ),
            ExperimentSpec(
                id="CAUSAL-020",
                name="Draft Capital Causal Effect on Player Usage",
                category="causal",
                description="Regression discontinuity using draft position cutoffs",
                hypothesis="High draft picks get more opportunities despite performance",
                expected_roi_impact="high",
                market_inefficiency="high",
                implementation_complexity="medium",
                data_requirements=["draft_position", "snap_counts", "performance_metrics"],
                success_metrics=["ate", "sunk_cost_bias", "discontinuity_validity"],
            ),
        ]

    def get_all_experiments(self) -> list[ExperimentSpec]:
        """Return all 40 experiments"""
        return self.cvae_experiments + self.causal_experiments

    def export_to_json(self, output_path: str):
        """Export experiment specifications to JSON"""

        all_experiments = self.get_all_experiments()

        export_data = {
            "metadata": {
                "total_experiments": len(all_experiments),
                "cvae_experiments": len(self.cvae_experiments),
                "causal_experiments": len(self.causal_experiments),
                "created_date": pd.Timestamp.now().isoformat(),
            },
            "experiments": [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "category": exp.category,
                    "description": exp.description,
                    "hypothesis": exp.hypothesis,
                    "expected_roi_impact": exp.expected_roi_impact,
                    "market_inefficiency": exp.market_inefficiency,
                    "implementation_complexity": exp.implementation_complexity,
                    "data_requirements": exp.data_requirements,
                    "success_metrics": exp.success_metrics,
                }
                for exp in all_experiments
            ],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(all_experiments)} experiments to {output_path}")

        return export_data


def main():
    """Generate and export experiment suite"""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    suite = ExpandedExperimentSuite()

    # Export to JSON
    output_path = "experiments/expanded_experiment_suite.json"
    export_data = suite.export_to_json(output_path)

    # Summary statistics
    print("\n" + "=" * 80)
    print("EXPANDED EXPERIMENT SUITE - 40 Total Experiments")
    print("=" * 80)

    print(f"\n📊 Total Experiments: {export_data['metadata']['total_experiments']}")
    print(f"   - cVAE Experiments: {export_data['metadata']['cvae_experiments']}")
    print(f"   - Causal Experiments: {export_data['metadata']['causal_experiments']}")

    # ROI Impact Distribution
    all_exps = suite.get_all_experiments()

    roi_high = sum(1 for e in all_exps if e.expected_roi_impact == "high")
    roi_medium = sum(1 for e in all_exps if e.expected_roi_impact == "medium")
    roi_low = sum(1 for e in all_exps if e.expected_roi_impact == "low")

    print("\n💰 Expected ROI Impact:")
    print(f"   - High:   {roi_high} experiments ({roi_high/40*100:.0f}%)")
    print(f"   - Medium: {roi_medium} experiments ({roi_medium/40*100:.0f}%)")
    print(f"   - Low:    {roi_low} experiments ({roi_low/40*100:.0f}%)")

    # Market Inefficiency Distribution
    ineff_high = sum(1 for e in all_exps if e.market_inefficiency == "high")
    ineff_medium = sum(1 for e in all_exps if e.market_inefficiency == "medium")
    ineff_low = sum(1 for e in all_exps if e.market_inefficiency == "low")

    print("\n🎯 Market Inefficiency:")
    print(f"   - High:   {ineff_high} experiments ({ineff_high/40*100:.0f}%)")
    print(f"   - Medium: {ineff_medium} experiments ({ineff_medium/40*100:.0f}%)")
    print(f"   - Low:    {ineff_low} experiments ({ineff_low/40*100:.0f}%)")

    # Implementation Complexity
    complex_low = sum(1 for e in all_exps if e.implementation_complexity == "low")
    complex_medium = sum(1 for e in all_exps if e.implementation_complexity == "medium")
    complex_high = sum(1 for e in all_exps if e.implementation_complexity == "high")

    print("\n⚙️  Implementation Complexity:")
    print(f"   - Low:    {complex_low} experiments ({complex_low/40*100:.0f}%)")
    print(f"   - Medium: {complex_medium} experiments ({complex_medium/40*100:.0f}%)")
    print(f"   - High:   {complex_high} experiments ({complex_high/40*100:.0f}%)")

    # Top candidates (high ROI + high inefficiency + low complexity)
    top_candidates = [
        e
        for e in all_exps
        if e.expected_roi_impact == "high"
        and e.market_inefficiency == "high"
        and e.implementation_complexity in ["low", "medium"]
    ]

    print("\n🏆 Prime Candidates (High ROI + High Inefficiency + Feasible):")
    print(f"   {len(top_candidates)} experiments")
    for exp in top_candidates[:5]:
        print(f"   - {exp.id}: {exp.name}")

    print(f"\n✅ Exported to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
