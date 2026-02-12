"""
ROI/EV Scoring Framework for Experiment Prioritization

Ranks 40 experiments by expected impact on betting ROI/EV using composite metric.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ROIScore:
    """ROI/EV Score for an experiment"""

    experiment_id: str
    composite_score: float  # 0-100 scale
    predictive_lift: float  # Expected improvement in prediction accuracy (%)
    market_inefficiency: float  # Market mispricing magnitude (0-1)
    implementation_feasibility: float  # Ease of implementation (0-1)
    sharpe_potential: float  # Expected Sharpe ratio improvement
    confidence: float  # Score confidence (0-1)
    rank: int  # Overall rank (1-40)
    components: dict[str, float]  # Detailed breakdown


class ExperimentROIEvaluator:
    """
    Evaluate and rank experiments by ROI/EV potential.

    Composite Score Formula:
    -----------------------
    Score = w1*PredictiveLift + w2*MarketInefficiency + w3*Feasibility + w4*SharpePotential

    Where:
    - PredictiveLift: Expected % improvement in prediction accuracy
    - MarketInefficiency: Degree of market mispricing (0-1)
    - Feasibility: Implementation ease and data availability (0-1)
    - SharpePotential: Expected Sharpe ratio improvement

    Weights (sum to 1.0):
    - w1 = 0.35 (Predictive Lift - most important)
    - w2 = 0.30 (Market Inefficiency - exploitation potential)
    - w3 = 0.20 (Feasibility - practical considerations)
    - w4 = 0.15 (Sharpe Potential - risk-adjusted returns)
    """

    def __init__(self):
        self.weights = {
            "predictive_lift": 0.35,
            "market_inefficiency": 0.30,
            "feasibility": 0.20,
            "sharpe_potential": 0.15,
        }

        # Domain knowledge priors for experiment scoring
        self.cvae_priors = self._initialize_cvae_priors()
        self.causal_priors = self._initialize_causal_priors()

    def _initialize_cvae_priors(self) -> dict:
        """
        Initialize priors for cVAE experiments based on literature and domain knowledge.

        Key insights:
        - Multi-horizon models typically improve accuracy by 5-15%
        - Weather conditioning shows 8-12% lift in outdoor games
        - Injury adjustments can improve by 10-20% when key players affected
        """

        return {
            "CVAE-001": {"lift": 5, "inefficiency": 0.5, "feasibility": 0.9, "sharpe": 0.3},
            "CVAE-002": {"lift": 15, "inefficiency": 0.8, "feasibility": 0.9, "sharpe": 0.6},
            "CVAE-003": {"lift": 15, "inefficiency": 0.8, "feasibility": 0.9, "sharpe": 0.6},
            "CVAE-004": {"lift": 10, "inefficiency": 0.6, "feasibility": 0.9, "sharpe": 0.4},
            "CVAE-005": {"lift": 5, "inefficiency": 0.4, "feasibility": 0.7, "sharpe": 0.2},
            "CVAE-006": {"lift": 12, "inefficiency": 0.7, "feasibility": 0.8, "sharpe": 0.5},
            "CVAE-007": {"lift": 18, "inefficiency": 0.85, "feasibility": 0.9, "sharpe": 0.7},
            "CVAE-008": {"lift": 20, "inefficiency": 0.9, "feasibility": 0.7, "sharpe": 0.8},
            "CVAE-009": {"lift": 12, "inefficiency": 0.85, "feasibility": 0.9, "sharpe": 0.5},
            "CVAE-010": {"lift": 25, "inefficiency": 0.9, "feasibility": 0.5, "sharpe": 0.9},
            "CVAE-011": {"lift": 8, "inefficiency": 0.6, "feasibility": 0.9, "sharpe": 0.4},
            "CVAE-012": {"lift": 6, "inefficiency": 0.5, "feasibility": 0.9, "sharpe": 0.3},
            "CVAE-013": {"lift": 5, "inefficiency": 0.4, "feasibility": 0.9, "sharpe": 0.2},
            "CVAE-014": {"lift": 22, "inefficiency": 0.95, "feasibility": 0.8, "sharpe": 0.85},
            "CVAE-015": {"lift": 14, "inefficiency": 0.75, "feasibility": 0.7, "sharpe": 0.6},
            "CVAE-016": {"lift": 10, "inefficiency": 0.65, "feasibility": 0.7, "sharpe": 0.45},
            "CVAE-017": {"lift": 16, "inefficiency": 0.8, "feasibility": 0.5, "sharpe": 0.65},
            "CVAE-018": {"lift": 18, "inefficiency": 0.85, "feasibility": 0.7, "sharpe": 0.75},
            "CVAE-019": {"lift": 17, "inefficiency": 0.75, "feasibility": 0.7, "sharpe": 0.7},
            "CVAE-020": {"lift": 19, "inefficiency": 0.8, "feasibility": 0.5, "sharpe": 0.75},
        }

    def _initialize_causal_priors(self) -> dict:
        """
        Initialize priors for causal experiments.

        Key insights:
        - Causal effects provide interpretable, stable edges
        - IV methods typically have 10-15% lift when instruments are strong
        - DiD with good parallel trends can be 12-18% lift
        """

        return {
            "CAUSAL-001": {"lift": 10, "inefficiency": 0.7, "feasibility": 0.9, "sharpe": 0.5},
            "CAUSAL-002": {"lift": 18, "inefficiency": 0.85, "feasibility": 0.7, "sharpe": 0.7},
            "CAUSAL-003": {"lift": 8, "inefficiency": 0.5, "feasibility": 0.9, "sharpe": 0.4},
            "CAUSAL-004": {"lift": 12, "inefficiency": 0.65, "feasibility": 0.7, "sharpe": 0.55},
            "CAUSAL-005": {"lift": 20, "inefficiency": 0.9, "feasibility": 0.7, "sharpe": 0.8},
            "CAUSAL-006": {"lift": 7, "inefficiency": 0.5, "feasibility": 0.9, "sharpe": 0.35},
            "CAUSAL-007": {"lift": 6, "inefficiency": 0.4, "feasibility": 0.7, "sharpe": 0.3},
            "CAUSAL-008": {"lift": 22, "inefficiency": 0.9, "feasibility": 0.7, "sharpe": 0.85},
            "CAUSAL-009": {"lift": 24, "inefficiency": 0.95, "feasibility": 0.5, "sharpe": 0.9},
            "CAUSAL-010": {"lift": 14, "inefficiency": 0.7, "feasibility": 0.5, "sharpe": 0.6},
            "CAUSAL-011": {"lift": 5, "inefficiency": 0.5, "feasibility": 0.9, "sharpe": 0.25},
            "CAUSAL-012": {"lift": 11, "inefficiency": 0.75, "feasibility": 0.9, "sharpe": 0.5},
            "CAUSAL-013": {"lift": 13, "inefficiency": 0.7, "feasibility": 0.7, "sharpe": 0.6},
            "CAUSAL-014": {"lift": 25, "inefficiency": 0.95, "feasibility": 0.5, "sharpe": 0.95},
            "CAUSAL-015": {"lift": 12, "inefficiency": 0.65, "feasibility": 0.7, "sharpe": 0.55},
            "CAUSAL-016": {"lift": 4, "inefficiency": 0.35, "feasibility": 0.9, "sharpe": 0.2},
            "CAUSAL-017": {"lift": 16, "inefficiency": 0.75, "feasibility": 0.5, "sharpe": 0.7},
            "CAUSAL-018": {"lift": 15, "inefficiency": 0.8, "feasibility": 0.7, "sharpe": 0.65},
            "CAUSAL-019": {"lift": 8, "inefficiency": 0.55, "feasibility": 0.7, "sharpe": 0.4},
            "CAUSAL-020": {"lift": 21, "inefficiency": 0.9, "feasibility": 0.7, "sharpe": 0.8},
        }

    def score_experiment(self, experiment_id: str) -> ROIScore:
        """
        Score a single experiment.

        Args:
            experiment_id: Experiment ID (e.g., 'CVAE-001' or 'CAUSAL-001')

        Returns:
            ROIScore object with composite score and breakdown
        """

        # Get priors
        if experiment_id.startswith("CVAE"):
            priors = self.cvae_priors[experiment_id]
        else:
            priors = self.causal_priors[experiment_id]

        # Extract components
        predictive_lift = priors["lift"]  # 0-30% scale
        market_inefficiency = priors["inefficiency"]  # 0-1 scale
        feasibility = priors["feasibility"]  # 0-1 scale
        sharpe_potential = priors["sharpe"]  # 0-1 scale

        # Normalize predictive lift to 0-100 scale
        predictive_lift_normalized = (predictive_lift / 30.0) * 100

        # Market inefficiency already 0-1, scale to 0-100
        market_inefficiency_normalized = market_inefficiency * 100

        # Feasibility already 0-1, scale to 0-100
        feasibility_normalized = feasibility * 100

        # Sharpe potential already 0-1, scale to 0-100
        sharpe_potential_normalized = sharpe_potential * 100

        # Composite score (weighted average)
        composite_score = (
            self.weights["predictive_lift"] * predictive_lift_normalized
            + self.weights["market_inefficiency"] * market_inefficiency_normalized
            + self.weights["feasibility"] * feasibility_normalized
            + self.weights["sharpe_potential"] * sharpe_potential_normalized
        )

        # Confidence (based on feasibility and data availability)
        confidence = 0.7 + 0.3 * feasibility  # Higher feasibility → higher confidence

        components = {
            "predictive_lift_raw": predictive_lift,
            "predictive_lift_norm": predictive_lift_normalized,
            "market_inefficiency_raw": market_inefficiency,
            "market_inefficiency_norm": market_inefficiency_normalized,
            "feasibility_raw": feasibility,
            "feasibility_norm": feasibility_normalized,
            "sharpe_potential_raw": sharpe_potential,
            "sharpe_potential_norm": sharpe_potential_normalized,
        }

        return ROIScore(
            experiment_id=experiment_id,
            composite_score=composite_score,
            predictive_lift=predictive_lift,
            market_inefficiency=market_inefficiency,
            implementation_feasibility=feasibility,
            sharpe_potential=sharpe_potential,
            confidence=confidence,
            rank=-1,  # Will be assigned after ranking
            components=components,
        )

    def rank_all_experiments(self) -> tuple[list[ROIScore], list[ROIScore], list[ROIScore]]:
        """
        Score and rank all 40 experiments.

        Returns:
            Tuple of (all_experiments, top_10_cvae, top_10_causal)
        """

        # Score all experiments
        all_scores = []

        for exp_id in list(self.cvae_priors.keys()) + list(self.causal_priors.keys()):
            score = self.score_experiment(exp_id)
            all_scores.append(score)

        # Rank by composite score
        all_scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Assign ranks
        for i, score in enumerate(all_scores):
            score.rank = i + 1

        # Split by category
        cvae_scores = [s for s in all_scores if s.experiment_id.startswith("CVAE")]
        causal_scores = [s for s in all_scores if s.experiment_id.startswith("CAUSAL")]

        # Top 10 from each
        top_10_cvae = cvae_scores[:10]
        top_10_causal = causal_scores[:10]

        logger.info("Ranked 40 experiments: Top 10 cVAE + Top 10 Causal selected")

        return all_scores, top_10_cvae, top_10_causal

    def export_rankings(
        self,
        all_scores: list[ROIScore],
        top_10_cvae: list[ROIScore],
        top_10_causal: list[ROIScore],
        output_dir: str = "experiments",
    ):
        """Export rankings to JSON and CSV"""

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Convert to DataFrames
        all_df = self._scores_to_dataframe(all_scores)
        cvae_df = self._scores_to_dataframe(top_10_cvae)
        causal_df = self._scores_to_dataframe(top_10_causal)

        # Export CSVs
        all_df.to_csv(f"{output_dir}/all_experiments_ranked.csv", index=False)
        cvae_df.to_csv(f"{output_dir}/top_10_cvae.csv", index=False)
        causal_df.to_csv(f"{output_dir}/top_10_causal.csv", index=False)

        # Export JSON with full details
        export_data = {
            "metadata": {
                "total_experiments": len(all_scores),
                "top_cvae_count": len(top_10_cvae),
                "top_causal_count": len(top_10_causal),
                "scoring_weights": self.weights,
                "created_date": pd.Timestamp.now().isoformat(),
            },
            "all_experiments": [self._score_to_dict(s) for s in all_scores],
            "top_10_cvae": [self._score_to_dict(s) for s in top_10_cvae],
            "top_10_causal": [self._score_to_dict(s) for s in top_10_causal],
        }

        with open(f"{output_dir}/experiment_rankings.json", "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported rankings to {output_dir}/")

        return all_df, cvae_df, causal_df

    def _scores_to_dataframe(self, scores: list[ROIScore]) -> pd.DataFrame:
        """Convert scores to DataFrame"""

        data = []
        for s in scores:
            data.append(
                {
                    "rank": s.rank,
                    "experiment_id": s.experiment_id,
                    "category": "cVAE" if s.experiment_id.startswith("CVAE") else "Causal",
                    "composite_score": round(s.composite_score, 2),
                    "predictive_lift_pct": round(s.predictive_lift, 1),
                    "market_inefficiency": round(s.market_inefficiency, 2),
                    "feasibility": round(s.implementation_feasibility, 2),
                    "sharpe_potential": round(s.sharpe_potential, 2),
                    "confidence": round(s.confidence, 2),
                }
            )

        return pd.DataFrame(data)

    def _score_to_dict(self, score: ROIScore) -> dict:
        """Convert ROIScore to dict for JSON export"""

        return {
            "experiment_id": score.experiment_id,
            "rank": score.rank,
            "composite_score": round(score.composite_score, 2),
            "predictive_lift": round(score.predictive_lift, 1),
            "market_inefficiency": round(score.market_inefficiency, 2),
            "implementation_feasibility": round(score.implementation_feasibility, 2),
            "sharpe_potential": round(score.sharpe_potential, 2),
            "confidence": round(score.confidence, 2),
            "components": {k: round(v, 2) for k, v in score.components.items()},
        }

    def print_summary(
        self, all_scores: list[ROIScore], top_10_cvae: list[ROIScore], top_10_causal: list[ROIScore]
    ):
        """Print summary of rankings"""

        print("\n" + "=" * 100)
        print("EXPERIMENT RANKINGS - TOP 10 cVAE + TOP 10 CAUSAL")
        print("=" * 100)

        print("\n📊 Composite Scoring Weights:")
        for component, weight in self.weights.items():
            print(f"   - {component}: {weight:.0%}")

        # Overall top 10
        print("\n🏆 TOP 10 EXPERIMENTS (Overall):")
        print(
            f"{'Rank':<6} {'ID':<15} {'Category':<10} {'Score':<8} {'Lift%':<8} {'Ineff':<8} {'Feas':<8} {'Sharpe':<8}"
        )
        print("-" * 100)

        for score in all_scores[:10]:
            category = "cVAE" if score.experiment_id.startswith("CVAE") else "Causal"
            print(
                f"{score.rank:<6} {score.experiment_id:<15} {category:<10} "
                f"{score.composite_score:>6.1f}   {score.predictive_lift:>6.1f}%  "
                f"{score.market_inefficiency:>6.2f}   {score.implementation_feasibility:>6.2f}   "
                f"{score.sharpe_potential:>6.2f}"
            )

        # Top 10 cVAE
        print("\n\n🔬 TOP 10 cVAE EXPERIMENTS:")
        print(
            f"{'Rank':<6} {'ID':<15} {'Score':<8} {'Lift%':<8} {'Ineff':<8} {'Feas':<8} {'Sharpe':<8}"
        )
        print("-" * 100)

        for score in top_10_cvae:
            print(
                f"{score.rank:<6} {score.experiment_id:<15} "
                f"{score.composite_score:>6.1f}   {score.predictive_lift:>6.1f}%  "
                f"{score.market_inefficiency:>6.2f}   {score.implementation_feasibility:>6.2f}   "
                f"{score.sharpe_potential:>6.2f}"
            )

        # Top 10 Causal
        print("\n\n🧪 TOP 10 CAUSAL EXPERIMENTS:")
        print(
            f"{'Rank':<6} {'ID':<15} {'Score':<8} {'Lift%':<8} {'Ineff':<8} {'Feas':<8} {'Sharpe':<8}"
        )
        print("-" * 100)

        for score in top_10_causal:
            print(
                f"{score.rank:<6} {score.experiment_id:<15} "
                f"{score.composite_score:>6.1f}   {score.predictive_lift:>6.1f}%  "
                f"{score.market_inefficiency:>6.2f}   {score.implementation_feasibility:>6.2f}   "
                f"{score.sharpe_potential:>6.2f}"
            )

        # Statistics
        all_df = self._scores_to_dataframe(all_scores)

        print("\n\n📈 SUMMARY STATISTICS:")
        print(
            f"   Composite Score Range: {all_df['composite_score'].min():.1f} - {all_df['composite_score'].max():.1f}"
        )
        print(f"   Mean Score: {all_df['composite_score'].mean():.1f}")
        print(f"   Median Score: {all_df['composite_score'].median():.1f}")
        print(f"   Std Dev: {all_df['composite_score'].std():.1f}")

        cvae_df = self._scores_to_dataframe(top_10_cvae)
        causal_df = self._scores_to_dataframe(top_10_causal)

        print(f"\n   Top 10 cVAE Mean Score: {cvae_df['composite_score'].mean():.1f}")
        print(f"   Top 10 Causal Mean Score: {causal_df['composite_score'].mean():.1f}")

        print("\n" + "=" * 100 + "\n")


def main():
    """Run experiment ranking"""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    evaluator = ExperimentROIEvaluator()

    # Rank all experiments
    all_scores, top_10_cvae, top_10_causal = evaluator.rank_all_experiments()

    # Print summary
    evaluator.print_summary(all_scores, top_10_cvae, top_10_causal)

    # Export results
    evaluator.export_rankings(all_scores, top_10_cvae, top_10_causal)

    print("✅ Rankings exported to experiments/ directory")


if __name__ == "__main__":
    main()
